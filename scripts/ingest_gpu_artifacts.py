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
           "pins.json", "preflight_source.py", "source_notebook.ipynb",
           "identity.json", "tokenizer_config.json", "identity_status.json", "fp32_not_run.json",
           "d8_descriptive_status.json", "d8_descriptive.json", "d8_status.json",
           "d8_generations.jsonl", "d8_judge_scores.jsonl",
           "locate_direction.json", "locate_status.json"}
DEPENDENCY_FILES = {"commands.json", "dependency_check.json", "dependency_check_global.json",
                    "failure.log", "install.log", "imports.log", "manifest.json", "pins.json",
                    "preflight_source.py", "source_notebook.ipynb", "versions.json"}
PHASE_STATUSES = {"ok", "failed", "blocked", "skipped", "incomplete"}
D8_SPLITS = {"calibration", "dev"}
D8_SLICES = {"benign_calibration", "benign_dev", "harm_calibration", "harm_dev"}
D8_METRIC_LABEL = "automated-judge only, human calibration pending"
PINNED_MODEL_ID = "CohereLabs/tiny-aya-global"
PINNED_MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"



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


def _canonical_content_sha256(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    encoded = json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _validate_phase_status(obj: object, phase: str) -> dict:
    if (not isinstance(obj, dict) or obj.get("phase") != phase
            or obj.get("status") not in PHASE_STATUSES
            or obj.get("metrics_eligible") is not False
            or obj.get("primary_asr") is not None):
        raise ValueError(f"{phase} phase status is not a fail-closed status artifact")
    return obj


def _validate_identity_status(obj: object, identity: object) -> None:
    payload = _validate_phase_status(obj, "identity")
    ok = isinstance(identity, dict) and identity.get("status") == "metadata_retrieved"
    if payload["status"] == "ok" and not ok:
        raise ValueError("identity status claims success without a retrieved identity")
    if payload["status"] in {"failed", "blocked", "skipped"} and ok:
        raise ValueError("identity phase status contradicts a retrieved identity")


def _validate_d8_rows(data: bytes, label: str) -> None:
    for line in data.splitlines():
        if not line.strip():
            continue
        row = strict_json(line)
        if not isinstance(row, dict) or row.get("split") not in D8_SPLITS:
            raise ValueError(f"{label} rows must stay on calibration/dev only")
        if row.get("slice") not in D8_SLICES:
            raise ValueError(f"{label} rows must use a known calibration/dev slice")
        if row.get("metrics_eligible") is not False or row.get("primary_asr") is not None:
            raise ValueError(f"{label} rows cannot claim a metric")


def _validate_d8_summary(obj: object) -> None:
    if (not isinstance(obj, dict) or obj.get("artifact_kind") != "descriptive"
            or obj.get("scoring_path") != "d8_descriptive"
            or obj.get("metric_label") != D8_METRIC_LABEL
            or obj.get("scored_baseline") is not False
            or obj.get("metrics_eligible") is not False
            or obj.get("primary_asr") is not None
            or obj.get("human_calibration") != "pending"):
        raise ValueError("D8 summary cannot claim a scored or human-calibrated baseline")
    if obj.get("split") not in {"calibration", "dev", "calibration+dev"}:
        raise ValueError("D8 summary split must be calibration/dev only")
    metrics = obj.get("metrics")
    if (not isinstance(metrics, dict) or not metrics
            or any(value is not None for value in metrics.values())):
        raise ValueError("D8 summary metrics must stay null")
    by_slice = obj.get("by_slice")
    if not isinstance(by_slice, dict) or not set(by_slice) <= D8_SLICES:
        raise ValueError("D8 summary contains an unknown or non-calibration/dev slice")
    data_pin = obj.get("data_pin")
    if not isinstance(data_pin, dict) or not data_pin:
        raise ValueError("D8 summary is missing its data pin")
    for name, record in data_pin.items():
        if "test" in str(name) or not isinstance(record, dict):
            raise ValueError("D8 summary data pin must not include the locked test split")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError("D8 summary data pin is missing a SHA-256")


def _validate_d8_status(obj: object, has_summary: bool) -> None:
    payload = _validate_phase_status(obj, "d8")
    if payload["status"] in {"ok", "incomplete"} and not has_summary:
        raise ValueError("D8 phase status claims a run without a summary artifact")
    if payload["status"] in {"failed", "blocked", "skipped"} and has_summary:
        raise ValueError("D8 phase status contradicts a written summary")


def _validate_locate_direction(obj: object) -> None:
    if (not isinstance(obj, dict) or obj.get("schema_version") != "provisional_locate_v1"
            or obj.get("provisional") is not True
            or obj.get("original_fp32_credit") is not False
            or obj.get("parity_credit") is not None
            or obj.get("metrics_eligible") is not False
            or obj.get("primary_asr") is not None
            or obj.get("dtype") != "bfloat16"
            or obj.get("model_id") != PINNED_MODEL_ID
            or obj.get("model_revision") != PINNED_MODEL_REVISION
            or obj.get("node") != "MLP 0"
            or type(obj.get("declared_score")) not in {int, float}
            or obj.get("declared_score") != 1.0
            or obj.get("content_sha256") != _canonical_content_sha256(obj)):
        raise ValueError("provisional locate direction is missing its no-credit bindings")
    if "test" in str(obj.get("split", "")):
        raise ValueError("provisional locate cannot use the locked test split")
    vector = obj.get("vector")
    if (not isinstance(vector, list) or len(vector) < 2
            or not all(_finite_number(value) for value in vector)):
        raise ValueError("provisional locate vector is missing or nonfinite")
    norm = math.sqrt(sum(float(value) ** 2 for value in vector))
    if norm == 0.0 or not _finite_number(obj.get("l2_norm")) or abs(obj["l2_norm"] - norm) > 1e-6:
        raise ValueError("provisional locate vector norm does not match")
    controls = obj.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("provisional locate controls are missing")
    base = controls.get("base_no_intervention")
    zero = controls.get("alpha_zero_with_wrapper")
    random_arm = controls.get("random_norm_matched_seed_7")
    candidate = controls.get("candidate_alpha_1")
    if (not isinstance(base, dict) or base.get("ran") is not True
            or not isinstance(zero, dict) or zero.get("ran") is not True
            or zero.get("matches_base") is not True or zero.get("max_abs_logit_diff") != 0.0
            or not isinstance(random_arm, dict) or random_arm.get("ran") is not True
            or not _finite_number(random_arm.get("max_abs_logit_diff"))
            or random_arm["max_abs_logit_diff"] <= 0
            or not _finite_number(random_arm.get("norm_gap"))
            or random_arm["norm_gap"] > 1e-3
            or not isinstance(candidate, dict) or candidate.get("ran") is not True
            or not _finite_number(candidate.get("max_abs_logit_diff"))
            or candidate["max_abs_logit_diff"] <= 0):
        raise ValueError("provisional locate controls did not run or failed")
    checks = obj.get("checks")
    if not isinstance(checks, dict) or checks.get("alpha_zero_matches_base") is not True:
        raise ValueError("provisional locate alpha-zero check is missing")


def _validate_locate_status(obj: object, has_direction: bool) -> None:
    payload = _validate_phase_status(obj, "locate")
    if payload.get("provisional") is not True or payload.get("original_fp32_credit") is not False:
        raise ValueError("locate phase status cannot claim original fp32 credit")
    if payload["status"] in {"ok", "incomplete"} and not has_direction:
        raise ValueError("locate phase status claims a run without a direction artifact")
    if payload["status"] in {"failed", "blocked", "skipped"} and has_direction:
        raise ValueError("locate phase status contradicts a written direction")


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
            # Identity is the casefolded basename. A directory and a file that
            # differ only by a trailing slash, or by parent directory, still collide.
            key = path.name.casefold()
            if not key or key in {".", ".."}:
                raise ValueError("Unsafe archive path")
            if info.is_dir() or name.endswith("/"):
                if key in normalized:
                    raise ValueError("Duplicate archive entry")
                normalized.add(key)
                continue
            if path.name not in ALLOWED or key in normalized or path.name in files:
                raise ValueError("Unexpected or duplicate artifact filename")
            normalized.add(key)
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
        if name == "fp32_not_run.json":
            obj = strict_json(data)
            if (not isinstance(obj, dict) or obj.get("artifact_kind") != "setup-smoke"
                    or obj.get("split") != "example" or obj.get("status") != "not_run"
                    or obj.get("passed") is not False or obj.get("run") is not False
                    or obj.get("not_run") not in {"insufficient_host_ram", "disabled_by_config", "host_ram_unavailable"}):
                raise ValueError("FP32 skip cannot grant original parity credit")
            original = strict_json(files.get("parity_original_sequential.json", b"{}"))
            if original.get("ran") is True or original.get("ran_short") is True or original.get("long_ran") is True or original.get("passed") is True:
                raise ValueError("FP32 skip contradicts recorded execution")
        if name == "d8_descriptive_status.json":
            obj = strict_json(data)
            if obj.get("artifact_kind") == "blocker":
                if (obj.get("scored_baseline") is not False or obj.get("metrics_eligible") is not False
                        or obj.get("primary_asr") is not None or not isinstance(obj.get("metrics"), dict)
                        or any(value is not None for value in obj["metrics"].values())):
                    raise ValueError("D8 notebook status cannot claim a scored baseline")
            elif obj.get("artifact_kind") == "descriptive":
                _validate_d8_summary(obj)
            else:
                raise ValueError("D8 notebook status has an unknown artifact kind")
        if name == "d8_descriptive.json":
            _validate_d8_summary(strict_json(data))
        if name in {"d8_generations.jsonl", "d8_judge_scores.jsonl"}:
            _validate_d8_rows(data, name)
        if name == "locate_direction.json":
            _validate_locate_direction(strict_json(data))
    identity = None
    if "identity.json" in files:
        identity = strict_json(files["identity.json"])
        if (not isinstance(identity, dict) or identity.get("artifact_kind") != "setup-smoke"
                or identity.get("split") != "example" or identity.get("weights_loaded") is not False
                or identity.get("model_imported", False) is not False
                or identity.get("generation_performed") is not False or identity.get("metrics_eligible") is not False
                or identity.get("primary_asr") is not None):
            raise ValueError("Tokenizer identity must remain metadata-only evidence")
        if identity.get("model_id") != PINNED_MODEL_ID:
            raise ValueError("Tokenizer identity model is not the pinned target")
        if identity.get("status") == "metadata_retrieved":
            asset = files.get("tokenizer_config.json")
            config = strict_json(asset) if asset else None
            stored_template = identity.get("chat_template")
            config_template = config.get("chat_template") if isinstance(config, dict) else None
            tokenizer_files = identity.get("tokenizer_files")

            def _strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from _strings(item)
                elif isinstance(value, list):
                    for item in value:
                        yield from _strings(item)

            config_templates = [item for item in _strings(config_template) if item]
            template_ok = (
                isinstance(stored_template, str) and bool(stored_template)
                and identity.get("chat_template_sha256")
                == hashlib.sha256(stored_template.encode("utf-8")).hexdigest()
                and (not config_templates or stored_template in config_templates)
            )
            if (asset is None or not isinstance(tokenizer_files, dict) or not tokenizer_files
                    or tokenizer_files.get("tokenizer_config.json", {}).get("sha256")
                    != hashlib.sha256(asset).hexdigest()
                    or hashlib.sha256(asset).hexdigest() != identity.get("tokenizer_config_sha256")
                    or not template_ok
                    or identity.get("model_revision") != PINNED_MODEL_REVISION
                    or identity.get("requested_revision") != PINNED_MODEL_REVISION
                    or identity.get("revision_matches_pin") is not True
                    or identity.get("hf_token_present") is not True):
                raise ValueError("Tokenizer identity asset or hash binding mismatch")
        elif "tokenizer_config.json" in files:
            raise ValueError("Tokenizer asset without successful identity")
    elif "tokenizer_config.json" in files:
        raise ValueError("Tokenizer asset requires identity")
    if "identity_status.json" in files:
        _validate_identity_status(strict_json(files["identity_status.json"]), identity)
    if "d8_status.json" in files:
        _validate_d8_status(
            strict_json(files["d8_status.json"]),
            "d8_descriptive.json" in files,
        )
    if "locate_status.json" in files:
        _validate_locate_status(strict_json(files["locate_status.json"]), "locate_direction.json" in files)
    if "locate_direction.json" in files and "locate_status.json" not in files:
        raise ValueError("Locate direction requires its phase status")
    if "d8_descriptive.json" in files and "d8_status.json" not in files:
        raise ValueError("D8 summary requires its phase status")
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
               "target_identity": identity,
               "d8_descriptive": (strict_json(files["d8_descriptive.json"])
                                  if "d8_descriptive.json" in files else None),
               "provisional_locate": (strict_json(files["locate_direction.json"])
                                      if "locate_direction.json" in files else None),
               "phase_status": {phase: strict_json(files[name])
                                for phase, name in (("identity", "identity_status.json"),
                                                    ("d8", "d8_status.json"),
                                                    ("locate", "locate_status.json"))
                                if name in files},
               "fp32_execution": strict_json(files["fp32_not_run.json"]) if "fp32_not_run.json" in files else None,
               "limitations": ["ZIP inspection does not authenticate the producing machine",
                               "bf16 variant is separate from original fp32 parity",
                               "provisional bf16 locate is not original fp32 parity credit",
                               "D8 automated-judge rates are not human-calibrated ASR",
                               "Fixture generation is not ASR or a scored baseline"]}
    return files, summary


def metrics_document(summary: dict) -> dict:
    """One-command metrics view for a validated GPU artifact archive.

    Descriptive judge counts only. Primary metrics stay null because human
    calibration is pending; provisional locate is not fp32 parity credit.
    """
    return {
        "schema_version": "gpu_metrics_v1",
        "artifact_kind": "setup-smoke",
        "status": "inspected",
        "metrics_eligible": False,
        "primary_asr": None,
        "metric_label": D8_METRIC_LABEL,
        "human_calibration": "pending",
        "scores": "not_a_scored_test",
        "split": "calibration+dev" if (summary.get("d8_descriptive") or summary.get("provisional_locate")) else "example",
        "archive_sha256": summary.get("archive_sha256"),
        "original_parity": summary.get("original_parity"),
        "target_identity": summary.get("target_identity"),
        "d8_descriptive": summary.get("d8_descriptive"),
        "provisional_locate": summary.get("provisional_locate"),
        "phase_status": summary.get("phase_status", {}),
        "fp32_execution": summary.get("fp32_execution"),
        "limitations": summary.get("limitations", []),
    }


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
    metrics = destination / "metrics.json.tmp"
    metrics.write_text(json.dumps(metrics_document(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metrics.replace(destination / "metrics.json")
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
