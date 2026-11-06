"""Synthetic ZIP fixtures, not GPU results."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.ingest_gpu_artifacts import ROOT, ingest, inspect_archive


def payloads():
    notebook = json.loads((ROOT / "notebooks/gpu_parity_and_baseline.ipynb").read_text(encoding="utf-8"))
    prompts = None
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARITY_PROMPTS" for t in node.targets):
                    prompts = ast.literal_eval(node.value)
    report = {"artifact_kind": "setup-smoke", "dtype": "float32", "hardware": "cpu",
              "ran": True, "ran_short": True, "long_ran": True, "error": None,
              "kl_short": {prompt: 0.0 for prompt in prompts},
              "kl_long": {"sliding_window_4196": 0.0}, "passed": True, "status": "pass"}
    long = {"ran": True, "seq_len": 4196, "window": 4096, "kl": 0.0,
            "dtype": "float32", "hardware": "cpu"}
    return {"parity_original_sequential.json": json.dumps(report).encode(),
            "parity_long_prompt.json": json.dumps(long).encode()}


def archive(tmp_path, files):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr("hs_gpu_artifacts/" + name, data)
    return path


def test_import_has_hashes_is_exclusive_and_never_metrics_eligible(tmp_path):
    files = payloads()
    zip_path = archive(tmp_path, files)
    destination = tmp_path / "imported"
    summary = ingest(zip_path, destination)
    assert summary["original_parity"]["status"] == "pass"
    assert summary["metrics_eligible"] is False
    assert summary["archive_sha256"] == hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert not (destination / "metrics.json").exists()
    with pytest.raises(FileExistsError):
        ingest(zip_path, destination)


def test_claimed_pass_without_long_artifact_is_incomplete(tmp_path):
    files = payloads()
    del files["parity_long_prompt.json"]
    _, summary = inspect_archive(archive(tmp_path, files))
    assert summary["original_parity"]["status"] == "incomplete"
    assert summary["original_parity"]["recorded_claim_consistent"] is False


def test_import_recomputes_threshold_instead_of_trusting_passed(tmp_path):
    files = payloads()
    report = json.loads(files["parity_original_sequential.json"])
    report["kl_short"][next(iter(report["kl_short"]))] = 0.5
    files["parity_original_sequential.json"] = json.dumps(report).encode()
    assert inspect_archive(archive(tmp_path, files))[1]["original_parity"]["status"] == "fail"


@pytest.mark.parametrize("name,data", [
    ("../../escaped.json", b"{}"), ("weights.safetensors", b"weights"),
    ("env_report.json", ("hf_" + "x" * 30).encode()),
    ("metrics_stub.json", b'{"artifact_kind":"run","split":"dev","metrics":{"asr":0}}'),
    ("metrics_stub.json", b'{"artifact_kind":"fixture","split":"example","metrics":null}'),
    ("baseline_manifest.json", b'{"artifact_kind":"run","split":"dev"}'),
    ("predictions.jsonl", b'{"artifact_kind":"run","split":"dev"}\n'),
])
def test_invalid_archives_fail_before_destination_creation(tmp_path, name, data):
    files = payloads()
    files[name] = data
    destination = tmp_path / "imported"
    with pytest.raises(ValueError):
        ingest(archive(tmp_path, files), destination)
    assert not destination.exists()


def test_nonfinite_json_refused(tmp_path):
    files = payloads()
    files["env_report.json"] = b'{"value":NaN}'
    with pytest.raises(ValueError, match="Nonfinite"):
        inspect_archive(archive(tmp_path, files))
