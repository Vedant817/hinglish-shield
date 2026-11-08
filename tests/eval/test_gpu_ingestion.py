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


def test_optional_fp32_skip_and_d8_status_import_without_score(tmp_path):
    skip = {"artifact_kind": "setup-smoke", "split": "example", "status": "not_run",
        "passed": False, "run": False, "not_run": "insufficient_host_ram",
        "ram_available_gib": 29.05, "ram_total_gib": 31.35}
    d8 = {"artifact_kind": "blocker", "scored_baseline": False, "metrics_eligible": False,
        "primary_asr": None, "metrics": {"asr": None}}
    files = {"fp32_not_run.json": json.dumps(skip).encode(),
        "d8_descriptive_status.json": json.dumps(d8).encode()}
    result = ingest(archive(tmp_path, files), tmp_path / "imported")
    assert result["original_parity"]["status"] == "incomplete"
    assert result["fp32_execution"]["ram_total_gib"] == 31.35
    assert result["metrics_eligible"] is False
    files.update(payloads())
    with pytest.raises(ValueError, match="contradicts"):
        inspect_archive(archive(tmp_path, files))


def _v2_identity(asset, template):
    return {
        "schema_version": "target_identity_v2",
        "artifact_kind": "setup-smoke", "split": "example",
        "weights_loaded": False, "model_imported": False, "generation_performed": False,
        "metrics_eligible": False, "primary_asr": None,
        "status": "metadata_retrieved",
        "model_id": "CohereLabs/tiny-aya-global",
        "requested_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
        "model_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
        "revision_matches_pin": True,
        "tokenizer_files": {
            "tokenizer_config.json": {"sha256": hashlib.sha256(asset).hexdigest(), "size": len(asset)},
            "special_tokens_map.json": {"sha256": "a" * 64, "size": 2},
            "tokenizer.json": {"sha256": "b" * 64, "size": 2},
        },
        "tokenizer_config_sha256": hashlib.sha256(asset).hexdigest(),
        "chat_template": template,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "hf_token_present": True,
    }


def test_identity_hash_mismatch_or_orphan_asset_refuses(tmp_path):
    files = {"tokenizer_config.json": b'{"chat_template":"fixture"}'}
    with pytest.raises(ValueError, match="requires identity"):
        inspect_archive(archive(tmp_path, files))
    identity = _v2_identity(files["tokenizer_config.json"], "fixture")
    identity["tokenizer_config_sha256"] = "a" * 64
    files["identity.json"] = json.dumps(identity).encode()
    with pytest.raises(ValueError, match="hash binding"):
        inspect_archive(archive(tmp_path, files))


def test_identity_template_hash_binds_the_complete_config_template(tmp_path):
    template = "synthetic template\n"
    asset = json.dumps({"chat_template": template}).encode()
    identity = _v2_identity(asset, template)
    files = {
        "identity.json": json.dumps(identity).encode(),
        "tokenizer_config.json": asset,
    }
    _, summary = inspect_archive(archive(tmp_path, files))
    assert summary["target_identity"]["status"] == "metadata_retrieved"
    identity["chat_template_sha256"] = hashlib.sha256(template.rstrip("\n").encode("utf-8")).hexdigest()
    files["identity.json"] = json.dumps(identity).encode()
    with pytest.raises(ValueError, match="hash binding"):
        inspect_archive(archive(tmp_path, files))


def _phase_status(phase, status, **extra):
    payload = {"phase": phase, "status": status,
               "metrics_eligible": False, "primary_asr": None}
    payload.update(extra)
    return json.dumps(payload).encode()


def _d8_summary():
    data_pin = {
        "benign_calibration": {"sha256": "a" * 64, "rows": 1},
        "benign_dev": {"sha256": "b" * 64, "rows": 1},
        "harm_calibration": {"sha256": "c" * 64, "rows": 1},
        "harm_dev": {"sha256": "d" * 64, "rows": 1},
    }
    return {
        "artifact_kind": "descriptive", "scoring_path": "d8_descriptive",
        "metric_label": "automated-judge only, human calibration pending",
        "scored_baseline": False, "metrics_eligible": False, "primary_asr": None,
        "human_calibration": "pending", "split": "calibration+dev",
        "metrics": {"asr": None, "over_refusal": None, "language_gap": None},
        "by_slice": {name: {"n": 1} for name in data_pin},
        "data_pin": data_pin,
    }


def _locate_direction():
    vector = [1.0, 2.0, 2.0]
    body = {
        "schema_version": "provisional_locate_v1", "provisional": True,
        "original_fp32_credit": False, "parity_credit": None,
        "metrics_eligible": False, "primary_asr": None,
        "model_id": "CohereLabs/tiny-aya-global",
        "model_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
        "dtype": "bfloat16", "node": "MLP 0", "declared_score": 1.0,
        "split": "calibration+dev", "vector": vector, "l2_norm": 3.0,
        "controls": {
            "base_no_intervention": {"ran": True},
            "alpha_zero_with_wrapper": {"ran": True, "matches_base": True, "max_abs_logit_diff": 0.0},
            "random_norm_matched_seed_7": {"ran": True, "max_abs_logit_diff": 0.4, "norm_gap": 0.0},
            "candidate_alpha_1": {"ran": True, "max_abs_logit_diff": 0.9},
        },
        "checks": {"alpha_zero_matches_base": True},
    }
    encoded = json.dumps(body, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    body["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


def test_phase_chain_artifacts_import_without_metrics(tmp_path):
    template = "synthetic template\n"
    asset = json.dumps({"chat_template": template}).encode()
    gen = {"item_id": "x1", "split": "dev", "slice": "harm_dev",
           "metrics_eligible": False, "primary_asr": None}
    files = {
        "identity.json": json.dumps(_v2_identity(asset, template)).encode(),
        "tokenizer_config.json": asset,
        "identity_status.json": _phase_status("identity", "ok"),
        "d8_descriptive.json": json.dumps(_d8_summary()).encode(),
        "d8_status.json": _phase_status("d8", "ok"),
        "d8_generations.jsonl": (json.dumps(gen) + "\n").encode(),
        "d8_judge_scores.jsonl": (json.dumps(gen) + "\n").encode(),
        "locate_direction.json": json.dumps(_locate_direction()).encode(),
        "locate_status.json": _phase_status("locate", "ok", provisional=True, original_fp32_credit=False),
    }
    _, summary = inspect_archive(archive(tmp_path, files))
    assert summary["phase_status"]["identity"]["status"] == "ok"
    assert summary["d8_descriptive"]["human_calibration"] == "pending"
    assert summary["provisional_locate"]["provisional"] is True
    assert summary["metrics_eligible"] is False


def test_failed_phases_stay_failures_and_refuse_score_claims(tmp_path):
    files = {
        "d8_status.json": _phase_status("d8", "failed", error="boom"),
        "locate_status.json": _phase_status(
            "locate", "blocked", provisional=True, original_fp32_credit=False,
            error="identity unavailable"),
    }
    _, summary = inspect_archive(archive(tmp_path, files))
    assert summary["phase_status"]["d8"]["status"] == "failed"
    assert summary["phase_status"]["locate"]["status"] == "blocked"


@pytest.mark.parametrize("files", [
    {"d8_descriptive.json": b'{"artifact_kind":"descriptive","scoring_path":"d8_descriptive","metric_label":"automated-judge only, human calibration pending","scored_baseline":true,"metrics_eligible":false,"primary_asr":null,"human_calibration":"pending","split":"dev","metrics":{"asr":null},"by_slice":{},"data_pin":{"benign_dev":{"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}}',
     "d8_status.json": b'{"phase":"d8","status":"ok","metrics_eligible":false,"primary_asr":null}'},
    {"locate_status.json": b'{"phase":"locate","status":"ok","metrics_eligible":false,"primary_asr":null,"provisional":true,"original_fp32_credit":false}'},
    {"d8_generations.jsonl": b'{"split":"test","slice":"harm_dev","metrics_eligible":false,"primary_asr":null}\n'},
])
def test_phase_chain_refuses_credit_or_test_split(tmp_path, files):
    with pytest.raises(ValueError):
        inspect_archive(archive(tmp_path, files))


def test_import_recomputes_threshold_instead_of_trusting_passed(tmp_path):
    files = payloads()
    report = json.loads(files["parity_original_sequential.json"])
    report["kl_short"][next(iter(report["kl_short"]))] = 0.5
    files["parity_original_sequential.json"] = json.dumps(report).encode()
    assert inspect_archive(archive(tmp_path, files))[1]["original_parity"]["status"] == "fail"


@pytest.mark.parametrize("value", [-1.0, -1e-8, 1e-4])
@pytest.mark.parametrize("case", ["short", "long"])
def test_invalid_kl_cannot_pass(tmp_path, value, case):
    files = payloads()
    report = json.loads(files["parity_original_sequential.json"])
    if case == "short":
        report["kl_short"][next(iter(report["kl_short"]))] = value
    else:
        report["kl_long"]["sliding_window_4196"] = value
        long = json.loads(files["parity_long_prompt.json"])
        long["kl"] = value
        files["parity_long_prompt.json"] = json.dumps(long).encode()
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


def test_wrapper_directory_entry_is_not_an_artifact(tmp_path):
    path = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        zipped.writestr("hs_gpu_artifacts/", b"")
        for name, data in payloads().items():
            zipped.writestr("hs_gpu_artifacts/" + name, data)
    assert inspect_archive(path)[1]["original_parity"]["status"] == "pass"


def test_trailing_slash_twin_of_an_artifact_is_refused(tmp_path):
    path = tmp_path / "twin.zip"
    with zipfile.ZipFile(path, "w") as zipped:
        for name, data in payloads().items():
            zipped.writestr("hs_gpu_artifacts/" + name, data)
        zipped.writestr("hs_gpu_artifacts/parity_long_prompt.json/", b"")
    with pytest.raises(ValueError, match="Duplicate archive entry"):
        inspect_archive(path)


def test_nonfinite_json_refused(tmp_path):
    files = payloads()
    files["env_report.json"] = b'{"value":NaN}'
    with pytest.raises(ValueError, match="Nonfinite"):
        inspect_archive(archive(tmp_path, files))
