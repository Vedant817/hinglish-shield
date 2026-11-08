"""Adversarial admission checks on synthetic rows, never dataset prompts."""
import json
import sys
from types import SimpleNamespace

import pytest

from src.eval import aggregate_dev as agg
from src.eval.calibration import ingest_human_labels, prepare_blinded_sheet
from src.eval.dev_runner import sha256_bytes
from tests.eval.test_aggregate_dev import (
    JUDGE_REV, REV, _aggregate, _jsonl, _spec, _write, build_ledger,
    arithmetic_uses_explicit_synthetic_authority,  # noqa: F401; explicit test authority
)


@pytest.mark.parametrize("mutation", ["empty", "inflate", "duplicate"])
def test_calibration_claims_cannot_replace_unique_paired_labels(tmp_path, mutation):
    built = build_ledger(tmp_path, [_spec("one"), _spec("two", category="refusal")])
    report = json.loads(built["calibration"].read_text())
    if mutation == "empty":
        report.update(pairs=[], n_paired=0, cohens_kappa=None, agreement=None, n_missing=999)
    elif mutation == "inflate":
        report["by_language"]["en"].update(n_labeled=100, n_expected=100)
        report["human_sample_target_met"] = True
    else:
        report["pairs"] = [report["pairs"][0], report["pairs"][0]]
    _write(built["calibration"], report)
    result = _aggregate(tmp_path, built)
    assert result["metrics_written"] is False
    assert not (tmp_path / "out" / "metrics.json").exists()
    assert any("calibration" in reason or "coverage" in reason for reason in result["reasons"])


@pytest.mark.parametrize(("field", "value", "needle"), [
    ("run_id", "another-run", "run id"),
    ("sample_id", "another-sample", "sample id"),
    ("prompt_hash", "0" * 64, "prompt hash"),
])
def test_calibration_report_cannot_replay_another_ledger_identity(tmp_path, field, value, needle):
    built = build_ledger(tmp_path, [_spec("one", category="refusal")])
    report = json.loads(built["calibration"].read_text())
    if field == "run_id":
        report["run_id"] = value
    else:
        report["pairs"][0][field] = value
    _write(built["calibration"], report)
    result = _aggregate(tmp_path, built)
    assert result["metrics_written"] is False
    assert not (tmp_path / "out" / "metrics.json").exists()
    assert any(needle in reason for reason in result["reasons"])


def test_partial_machine_category_coverage_cannot_validate_a_language(tmp_path):
    built = build_ledger(tmp_path, [_spec("scored"), _spec("pending", category="needs_category_review")],
                         calibrate=False)
    key = tmp_path / "key.json"
    prepare_blinded_sheet(built["ledger"], tmp_path / "sheet.jsonl", key)
    rows = json.loads(key.read_text())["rows"]
    labels = tmp_path / "labels.jsonl"
    _jsonl(labels, [{"output_hash": row["output_hash"], "label": "refusal", "rater_id": "synthetic-rater"}
                    for row in rows])
    report = ingest_human_labels(built["ledger"], key, labels)
    assert report["n_paired"] == 1
    assert report["n_skipped_unresolved_judge"] == 1
    assert report["kappa_coverage"] == "partial_not_validated"
    assert report["by_language"]["en"]["validated"] is False
    assert report["labels_admissible_for_run"] is False


def _arms(tmp_path):
    reports = []
    for name, category, method in (("base", "harmful_compliant", "base"), ("candidate", "refusal", "steering")):
        root = tmp_path / name
        built = build_ledger(root, [
            _spec("en-harm", family_id="en-family", category=category),
            _spec("hi-harm", family_id="hi-family", lang="hi", category=category),
            _spec("en-benign", family_id="benign-family", hs_label="benign", category="truncated", truncated=True),
        ], method=method)
        reports.append(_aggregate(root, built))
    assert all(report["metrics_eligible"] for report in reports), [r["reasons"] for r in reports]
    return reports


def test_disjoint_language_families_pair_but_missing_hi_benign_is_unresolved(tmp_path):
    base, candidate = _arms(tmp_path)
    result = agg.release_decision(base, candidate, None, rollback_path=None)
    assert result["decision"] == "revise"
    assert set(result["harm_by_language"]) == {"en", "hi"}
    assert all(block["bootstrap"]["n_families"] == 1 for block in result["harm_by_language"].values())
    assert result["over_refusal_gate"] == "unresolved"
    assert result["over_refusal_by_language"]["hi"]["bootstrap"] is None
    # A family that exists in this language but fails in both arms must remain.
    assert agg._paired_method_gap({"failed": None}, {"failed": None})["incomplete_families"] == ["failed"]


@pytest.mark.parametrize("key,value", [("model_revision", "f" * 40), ("model_id", "other-model"), ("method", "base")])
def test_acceptance_binds_arm_identity_and_non_base_method(tmp_path, key, value):
    base, candidate = _arms(tmp_path)
    candidate[key] = value
    result = agg.release_decision(base, candidate, None, rollback_path=None)
    assert result["decision"] == "unavailable"
    assert result["arms"]["candidate"][key] == value


def test_renderer_requires_the_saved_admission_sidecar_and_metrics_hash(tmp_path):
    built = build_ledger(tmp_path, [_spec("one")])
    _aggregate(tmp_path, built)
    path = tmp_path / "out" / "metrics.json"
    assert "1.000000" in agg.render_publication([path], heading="Fixture contract")
    sidecar = path.parent / "aggregation.json"
    sidecar.unlink()
    assert "1.000000" not in agg.render_publication([path], heading="Fixture contract")


def test_runtime_revision_and_artifact_hash_are_checked(tmp_path):
    built = build_ledger(tmp_path, [_spec("one")])
    evidence = json.loads(built["runtime"].read_text())
    manifest_path = tmp_path / "sources" / "judge_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["config"]["model_revision"] = "f" * 40
    _write(manifest_path, payload)
    result = _aggregate(tmp_path, built)
    assert any("artifact hash" in reason for reason in result["reasons"])
    assert result["metrics_written"] is False
    # Updating a reference hash cannot authorize a different judge revision.
    evidence["judge_manifest_sha256"] = sha256_bytes(manifest_path.read_bytes())
    _write(built["runtime"], evidence)
    reasons = agg._assess_runtime(json.loads((built["ledger"] / "manifest.json").read_text())["provenance"], built["runtime"])
    assert any("pins" in reason for reason in reasons)


def test_missing_provenance_returns_withheld_report(tmp_path):
    built = build_ledger(tmp_path, [_spec("one")])
    path = built["ledger"] / "manifest.json"
    payload = json.loads(path.read_text())
    del payload["provenance"]
    _write(path, payload)
    result = _aggregate(tmp_path, built)
    assert result["metrics_written"] is False
    assert any("provenance" in reason for reason in result["reasons"])


def test_capability_rubric_is_bound_to_benign_families(tmp_path):
    rows = [{"family_id": "harm-family", "hs_label": "harm"},
            {"family_id": "benign-family", "hs_label": "benign"}]
    path = tmp_path / "rubric.json"
    _write(path, {"rubric_id": "synthetic-capability", "rubric_frozen": True, "scores": {"benign-family": 0.8}})
    value, reason, _ = agg._assess_capability_arm(rows, path)
    assert value == 0.8 and reason is None
    _write(path, {"rubric_id": "synthetic-capability", "rubric_frozen": True, "scores": {"harm-family": 0.8}})
    value, reason, _ = agg._assess_capability_arm(rows, path)
    assert value is None and "benign" in reason


def test_wildguard_producer_refuses_missing_commit_before_torch_import(tmp_path, monkeypatch):
    import builtins
    from src.eval import smoke_wildguard as producer
    def unavailable(*args, **kwargs):
        raise OSError("fixture git unavailable")
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("weights path reached before source provenance")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(producer.subprocess, "check_output", unavailable)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    monkeypatch.setenv("HF_TOKEN", "fixture-placeholder-only")
    out = tmp_path / "no-git"
    assert producer.main(["--revision", JUDGE_REV, "--out", str(out)]) == 2
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and manifest["full_commit"] is None
    assert manifest["attempts"] == []


def test_actual_wildguard_fixture_producer_format_is_consumable_with_fake_backend(tmp_path, monkeypatch):
    """Execute the real artifact writer using fake inference, not model weights."""
    from auditkit.model import AutoModel
    from src.eval import smoke_wildguard as producer
    from src.eval.judges import JudgeVerdict
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fixture-gpu",
                           reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                           empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, __version__="fixture", float16="fp16", float32="fp32",
                                                             manual_seed=lambda _: None))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(
        from_pretrained=lambda *a, **k: lambda *a, **k: {"input_ids": [1, 2, 3]})))
    monkeypatch.setattr(AutoModel, "resolve", lambda *a, **k: SimpleNamespace(unload=lambda: None))
    monkeypatch.setattr(producer, "version", lambda name: "1.1.3" if name == "auditkit" else "fixture")
    monkeypatch.setenv("HF_TOKEN", "fixture-placeholder-only")
    class Judge:
        def __init__(self, **kwargs):
            self._scorer = SimpleNamespace(identity=lambda: {"profile": "hs_wildguard_v1"},
                                          _profile={"build_prompt": lambda *a: "fixture input"})
        def judge_one(self, prompt, response, **kwargs):
            return JudgeVerdict(kwargs["prompt_id"], 0.0, "safe", {
                "harmful_request": "no", "response_refusal": "yes", "harmful_response": "no",
                "raw": "fixture completion", "auditkit_version": "1.1.3"})
    monkeypatch.setattr(producer, "WildGuardPrimaryJudge", Judge)
    out = tmp_path / "producer"
    assert producer.main(["--revision", JUDGE_REV, "--out", str(out)]) == 0
    evidence = {"observed_dtype": "bfloat16", "model_id": "CohereLabs/tiny-aya-global", "model_revision": REV,
                "judge_model_id": "allenai/wildguard", "judge_revision": JUDGE_REV}
    for stem, filename in (("judge_manifest", "manifest.json"), ("judge_results", "fixture_results.json")):
        path = out / filename
        evidence[stem + "_uri"] = str(path)
        evidence[stem + "_sha256"] = sha256_bytes(path.read_bytes())
    evidence_path = tmp_path / "evidence.json"
    _write(evidence_path, evidence)
    assert agg._assess_runtime({**evidence, "dtype": "bfloat16"}, evidence_path) == []
