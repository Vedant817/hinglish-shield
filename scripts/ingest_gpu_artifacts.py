"""Inspect and import a Kaggle parity ZIP. Never promotes fixtures to ASR."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN .*PRIVATE KEY")
MAX_FILE_BYTES = 64 * 1024**2
MAX_ARCHIVE_BYTES = 256 * 1024**2
PINNED_PROMPTS_SHA256 = "3ab3ad15fba852e34e08b8c485e2e3c41adc8702aa80a0bbf62215d5001fabab"
ALLOWED = {"env_report.json", "versions.json", "parity_original_sequential.json",
           "parity_long_prompt.json", "parity_variant_bf16.json", "parity_pytest.json",
           "parity_pytest.log", "fp16_guard.json", "predictions.jsonl", "metrics_stub.json",
           "baseline_manifest.json", "dependency_check.json", "dependency_check_global.json",
           "commands.json", "failure.log", "install.log", "imports.log", "manifest.json",
           "pins.json", "preflight_source.py", "source_notebook.ipynb"}
DEPENDENCY_FILES = {"commands.json", "dependency_check.json", "dependency_check_global.json",
                    "failure.log", "install.log", "imports.log", "manifest.json", "pins.json",
                    "preflight_source.py", "source_notebook.ipynb", "versions.json"}


def strict_json(data: bytes):
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant")
    return json.loads(data, parse_constant=invalid_constant)


def parity_summary(files: dict[str, bytes]) -> dict:
    """Recompute the original gate; never trust its recorded passed boolean."""
    report = strict_json(files.get("parity_original_sequential.json", b"{}"))
    notebook = json.loads((ROOT / "notebooks/gpu_parity_and_baseline.ipynb").read_text(encoding="utf-8"))
    expected = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARITY_PROMPTS" for t in node.targets):
                prompts = ast.literal_eval(node.value)
                digest = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
                if digest != PINNED_PROMPTS_SHA256:
                    raise ValueError("Notebook prompts disagree with the pinned upstream copy")
                expected = set(prompts)
    if expected is None:
        raise ValueError("Cannot resolve pinned short parity prompts")
    short = report.get("kl_short") or {}
    long = report.get("kl_long") or {}
    long_record = strict_json(files.get("parity_long_prompt.json", b"{}"))
    def finite_kls(values):
        return all(type(v) in {int, float} and math.isfinite(v) and v >= 0 for v in values)
    short_present = isinstance(short, dict) and set(short) == expected
    long_present = (report.get("long_ran") is True and isinstance(long, dict)
                    and set(long) == {"sliding_window_4196"}
                    and long_record.get("ran") is True and long_record.get("seq_len") == 4196
                    and long_record.get("window") == 4096
                    and long_record.get("kl") == long.get("sliding_window_4196"))
    provenance_ok = (report.get("dtype") == "float32" and report.get("hardware") == "cpu"
                     and report.get("error") is None and report.get("ran") is True
                     and report.get("ran_short") is True
                     and long_record.get("dtype") == "float32" and long_record.get("hardware") == "cpu")
    status = "incomplete"
    if short_present and long_present:
        passed = (provenance_ok and finite_kls(short.values()) and finite_kls(long.values())
                  and all(v < 1e-4 for v in (*short.values(), *long.values())))
        status = "pass" if passed else "fail"
    return {"status": status, "short_kl": short, "long_kl": long,
            "recorded_status": report.get("status"), "recorded_passed": report.get("passed"),
            "recorded_claim_consistent": (None if report.get("passed") is None
                                          else report.get("passed") is (status == "pass")),
            "gate": "original_fp32_short_and_long", "kl_tolerance": 1e-4}


def inspect_archive(archive: Path) -> tuple[dict[str, bytes], dict]:
    files = {}
    normalized = set()
    total = 0
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = info.filename
            if TOKEN.search(name.encode("utf-8")):
                raise ValueError("Secret-like archive entry name")
            path = PurePosixPath(name)
            if ("\\" in name or ":" in name or path.is_absolute() or ".." in path.parts
                    or any(part.endswith((".", " ")) for part in path.parts)
                    or len(path.parts) > 2):
                raise ValueError("Unsafe archive path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Archive symlinks are forbidden")
            key = name.rstrip("/").casefold()
            if key in normalized:
                raise ValueError("Duplicate archive entry")
            normalized.add(key)
            if info.is_dir():
                continue
            if path.name not in ALLOWED or path.name in files:
                raise ValueError("Unexpected or duplicate artifact filename")
            if info.file_size > MAX_FILE_BYTES:
                raise ValueError("Artifact exceeds import size limit")
            total += info.file_size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("Archive exceeds import size limit")
            # Reading the full entry verifies its ZIP CRC. No extractall or shell paths.
            with zipped.open(info) as handle:
                data = handle.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES or len(data) != info.file_size:
                raise ValueError("Artifact length mismatch")
            if TOKEN.search(data):
                raise ValueError("Secret-like content in artifact; import refused")
            files[path.name] = data
    if not files:
        raise ValueError("Archive contains no permitted artifacts")
    for name, data in files.items():
        if name.endswith(".json"):
            strict_json(data)
        if name in {"dependency_check.json", "dependency_check_global.json"}:
            obj = strict_json(data)
            if (not isinstance(obj, dict) or obj.get("artifact_kind") != "setup-smoke"
                    or obj.get("split") != "example" or type(obj.get("passed")) is not bool):
                raise ValueError("Dependency checks must remain setup-smoke diagnostics")
        if name == "metrics_stub.json":
            obj = strict_json(data)
            if (obj.get("artifact_kind") != "fixture" or obj.get("split") != "example"
                    or not isinstance(obj.get("metrics"), dict)
                    or any(value is not None for value in obj["metrics"].values())):
                raise ValueError("GPU notebook metrics must remain null fixture metrics")
        if name == "baseline_manifest.json":
            obj = strict_json(data)
            if obj.get("artifact_kind") != "fixture" or obj.get("split") != "example":
                raise ValueError("GPU generation manifest must remain a fixture")
        if name == "predictions.jsonl":
            for line in data.splitlines():
                if line.strip():
                    row = strict_json(line)
                    if row.get("artifact_kind") != "fixture" or row.get("split") != "example":
                        raise ValueError("GPU notebook predictions must remain fixtures")
    standalone_dependency = {"preflight_source.py", "source_notebook.ipynb", "commands.json",
                             "install.log", "imports.log", "failure.log", "pins.json"}
    if set(files) & standalone_dependency and "manifest.json" not in files:
        raise ValueError("Dependency preflight manifest missing")
    if "manifest.json" in files:
        manifest = strict_json(files["manifest.json"])
        if (not isinstance(manifest, dict) or set(files) - DEPENDENCY_FILES or manifest.get("artifact_kind") != "setup-smoke"
                or manifest.get("split") != "example" or manifest.get("status") not in {"complete", "failed"}
                or manifest.get("model_weights_loaded") is not False or manifest.get("model_inference") is not False):
            raise ValueError("Dependency preflight cannot claim model execution or mix parity artifacts")
        hashes = manifest.get("files_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(files) - {"manifest.json"}:
            raise ValueError("Dependency preflight file coverage mismatch")
        if any(hashes[name] != hashlib.sha256(files[name]).hexdigest() for name in hashes):
            raise ValueError("Dependency preflight file hash mismatch")
        if (manifest.get("source_sha256") != hashes.get("preflight_source.py")
                or manifest.get("notebook_sha256") != hashes.get("source_notebook.ipynb")
                or not hashes.get("preflight_source.py") or not hashes.get("source_notebook.ipynb")):
            raise ValueError("Dependency preflight source identity mismatch")
    summary = {"artifact_kind": "setup-smoke", "status": "inspected", "metrics_eligible": False,
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
               "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
               "original_parity": parity_summary(files),
               "limitations": ["ZIP inspection does not authenticate the producing machine",
                               "bf16 variant is separate from original fp32 parity",
                               "Fixture generation is not ASR or a scored baseline"]}
    return files, summary


def ingest(archive: Path, destination: Path) -> dict:
    files, summary = inspect_archive(archive)
    # Only validated bytes are copied. A durable completion marker is written last.
    destination.mkdir(parents=True, exist_ok=False)
    # Byte hashes must survive later Windows checkouts, including log newlines.
    (destination / ".gitattributes").write_bytes(b"* -text\n")
    for name, data in files.items():
        (destination / name).write_bytes(data)
    marker = destination / "ingestion.json.tmp"
    marker.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    marker.replace(destination / "ingestion.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", args.run_id):
        parser.error("run-id must be a simple directory name")
    result = ingest(args.archive, ROOT / "runs" / args.run_id)
    print(json.dumps(result, indent=2, allow_nan=False))
