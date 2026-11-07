#!/usr/bin/env python3
"""Run the parity notebook's install/import steps, without weights or secrets.

Upload this script as a private Kaggle kernel on an existing free account.
The source notebook is immutable and hash checked before extracting literals.
No Hugging Face authentication, model loading or inference is performed.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

NOTEBOOK_REVISION = "fceabfa61bea73bf0b81dadc316766c02b9b82c2"
# SHA-256 of committed notebook bytes, not a Windows checkout's line endings.
NOTEBOOK_SHA256 = "864bc460772afc11fcda476400d3601487ed90c5be8978344614fb058017aa19"
NOTEBOOK_URL = (
    "https://raw.githubusercontent.com/Vedant817/hinglish-shield/"
    + NOTEBOOK_REVISION + "/notebooks/gpu_parity_and_baseline.ipynb"
)
PIN_NAMES = frozenset({"TRANSFORMERS_PIN", "HF_HUB_PIN", "CIRCUITKIT_GIT", "CIRCUITKIT_SHA",
                       "SAFETUNE_GIT", "SAFETUNE_SHA", "DEPENDENCY_ROOTS"})


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def notebook_pins(data: bytes) -> dict:
    if sha256(data) != NOTEBOOK_SHA256:
        raise ValueError("pinned notebook hash mismatch")
    notebook = json.loads(data)
    pins = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in PIN_NAMES:
                    if target.id in pins:
                        raise ValueError("duplicate notebook pin")
                    value = ast.literal_eval(node.value)
                    valid = (isinstance(value, list) and value and all(isinstance(v, str) and v for v in value)
                             if target.id == "DEPENDENCY_ROOTS" else isinstance(value, str) and value)
                    if not valid:
                        raise ValueError("invalid notebook pin")
                    pins[target.id] = value
    if set(pins) != PIN_NAMES:
        raise ValueError("missing notebook install pins")
    return pins


def notebook_checker(data: bytes):
    """Compile only the checker definition from the hash-verified notebook."""
    notebook_pins(data)
    definitions = []
    for cell in json.loads(data)["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        definitions.extend(node for node in ast.parse(source).body
                           if isinstance(node, ast.FunctionDef) and node.name == "check_dependency_closure")
    if len(definitions) != 1:
        raise ValueError("missing or duplicate dependency checker")
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "pinned_dependency_checker", "exec"), namespace)
    return namespace["check_dependency_closure"]


def install_arguments(pins: dict) -> list[str]:
    return [
        f"transformers=={pins['TRANSFORMERS_PIN']}", f"huggingface-hub=={pins['HF_HUB_PIN']}",
        "accelerate", "safetensors", "sentencepiece", "protobuf", "psutil", "pytest",
        f"git+{pins['CIRCUITKIT_GIT']}@{pins['CIRCUITKIT_SHA']}",
        f"git+{pins['SAFETUNE_GIT']}@{pins['SAFETUNE_SHA']}",
    ]


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(output_parent: Path = Path("/kaggle/working")) -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = output_parent / ("dependency-preflight-" + stamp)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "artifact_kind": "setup-smoke", "split": "example", "status": "started",
        "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "notebook_revision": NOTEBOOK_REVISION, "notebook_sha256": NOTEBOOK_SHA256,
        "model_weights_loaded": False, "model_inference": False,
        "limitations": ["Dependency and import verification only; no parity, dtype guard, judge accuracy or ASR",
                        "Does not verify gated access or platform secret attachment"],
    }
    source = Path(__file__).read_bytes()
    (out / "preflight_source.py").write_bytes(source)
    manifest["source_sha256"] = sha256(source)
    write_json(out / "manifest.json", manifest)
    success = False
    try:
        if sys.version_info < (3, 12):
            raise RuntimeError("SafeTune requires Python 3.12 or newer")
        manifest["phase"] = "fetch_pinned_notebook"
        with urllib.request.urlopen(NOTEBOOK_URL, timeout=60) as response:
            notebook_bytes = response.read()
        pins = notebook_pins(notebook_bytes)
        (out / "source_notebook.ipynb").write_bytes(notebook_bytes)
        write_json(out / "pins.json", pins)
        install = [sys.executable, "-m", "pip", "install", "-q", *install_arguments(pins)]
        write_json(out / "commands.json", {"install": install, "check": [sys.executable, "-m", "pip", "check"]})
        manifest["phase"] = "install"
        write_json(out / "manifest.json", manifest)
        with (out / "install.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(install, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        manifest["install_exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError("pinned dependency install failed; inspect install.log")
        manifest["phase"] = "dependency_check"
        packages = ("torch", "transformers", "huggingface-hub", "transformer-lens", "accelerate", "circuitkit", "safetune")
        versions = {name: importlib.metadata.version(name) for name in packages}
        write_json(out / "versions.json", versions)
        if versions["transformers"] != pins["TRANSFORMERS_PIN"] or versions["huggingface-hub"] != pins["HF_HUB_PIN"]:
            raise RuntimeError("resolved HF stack differs from pinned notebook")
        check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=120)
        write_json(out / "dependency_check_global.json", {"artifact_kind": "setup-smoke", "split": "example",
                   "scope": "entire_preinstalled_environment", "passed": check.returncode == 0,
                   "stdout": check.stdout, "stderr": check.stderr})
        checker = notebook_checker(notebook_bytes)
        scoped = checker(pins["DEPENDENCY_ROOTS"])
        write_json(out / "dependency_check.json", scoped)
        if not scoped["passed"]:
            raise RuntimeError("project dependency closure failed; imports skipped as in the notebook")
        direct_sources = {}
        for package, pin in (("circuitkit", "CIRCUITKIT_SHA"), ("safetune", "SAFETUNE_SHA")):
            dist = importlib.metadata.distribution(package)
            direct = json.loads(dist.read_text("direct_url.json") or "{}")
            observed = direct.get("vcs_info", {}).get("commit_id")
            direct_sources[package] = observed
            if observed != pins[pin]:
                raise RuntimeError("installed Lexsi source revision differs from the notebook pin")
        versions["direct_sources"] = direct_sources
        write_json(out / "versions.json", versions)
        manifest["phase"] = "imports"
        # CircuitKIT must install its compatibility hooks before TransformerLens.
        with (out / "imports.log").open("w", encoding="utf-8") as log:
            imports = subprocess.run([sys.executable, "-c", (
                "import circuitkit.backends._tl_compat; "
                "import transformer_lens; import safetune; "
                "import torch; import json; "
                "print(json.dumps({'cuda_available':torch.cuda.is_available(),"
                "'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
            )], stdout=log, stderr=subprocess.STDOUT, timeout=180)
        manifest["imports_exit_code"] = imports.returncode
        if imports.returncode:
            raise RuntimeError("dependency import smoke failed; inspect imports.log")
        success = True
        manifest["phase"] = "complete"
    except Exception as exc:
        manifest["error_type"] = type(exc).__name__
        # This program never receives a secret. Preserve the exact install/import
        # failure locally; only the small manifest summary is printed to stdout.
        (out / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        manifest["status"] = "complete" if success else "failed"
        manifest["files_sha256"] = {p.name: sha256(p.read_bytes()) for p in out.iterdir()
                                    if p.is_file() and p.name != "manifest.json"}
        write_json(out / "manifest.json", manifest)
        archive = output_parent / (out.name + ".zip")
        with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(out.iterdir()):
                bundle.write(path, path.name)
        print(json.dumps({"artifact_kind": "setup-smoke", "status": manifest["status"],
                          "phase": manifest.get("phase"), "archive": str(archive)}))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
