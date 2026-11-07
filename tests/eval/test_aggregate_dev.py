"""Synthetic ledger contracts for dev aggregation. No model weights are loaded."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.eval.aggregate_dev import aggregate_ledger, release_decision, render_publication
from src.eval.calibration import ingest_human_labels, prepare_blinded_sheet
from src.eval.dev_runner import canonical_dataset_bytes, sha256_bytes, sha256_text
from src.eval.metrics_io import is_fixture, load_and_validate

PY = Path(sys.executable)
REV = "a" * 40
JUDGE_REV = "b" * 40
COMMIT = "c" * 40
CANONICAL = "334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531"
SYNTHETIC_AUTHORITY = {}


@pytest.fixture(autouse=True)
def arithmetic_uses_explicit_synthetic_authority(monkeypatch):
    """Arithmetic fixtures get a test authority; production has no override."""
    import src.eval.aggregate_dev as module
    SYNTHETIC_AUTHORITY.clear()
    monkeypatch.setattr(module, "load_canonical_authority", lambda *args, **kwargs: {
        "by_source": SYNTHETIC_AUTHORITY,
    })


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8")


def _spec(sample_id: str, **overrides) -> dict:
    row = {
        "sample_id": sample_id,
        "family_id": "f1",
        "lang": "en",
        "hs_label": "harm",
        "status": "ok",
        "category": "harmful_compliant",
        "response": f"response-{sample_id}",
        "truncated": False,
        "human_category": None,
    }
    row.update(overrides)
    return row


def build_ledger(root: Path, specs: list[dict], *, kind: str = "run", method: str = "base", lie_counts: bool = False, calibrate: bool = True, runtime: bool = True, dtype: str = "bfloat16", dtype_guard=None):
    ledger = root / "ledger"
    ledger.mkdir(parents=True)
    index_rows = []
    prompt_rows = []
    members: dict[str, list[dict]] = {}
    labels: dict[str, set[str]] = {}
    generations = []
    samples = []
    adjudication = []
    for spec in specs:
        sample_id = spec["sample_id"]
        item_id = f"item-{sample_id}"
        prompt = f"synthetic prompt {sample_id}"
        prompt_hash = sha256_text(prompt)
        index_rows.append({
            "sample_id": sample_id,
            "item_id": item_id,
            "family_id": spec["family_id"],
            "source_dataset": "synthetic",
            "source_id": sample_id,
            "lang": spec["lang"],
            "hs_label": spec["hs_label"],
            "split": "dev",
            "prompt_hash": prompt_hash,
            "review_status": "reviewed",
        })
        prompt_rows.append({"sample_id": sample_id, "item_id": item_id, "prompt": prompt})
        members.setdefault(spec["family_id"], []).append({
            "item_id": item_id,
            "source_dataset": "synthetic",
            "source_id": sample_id,
            "lang": spec["lang"],
            "hs_label": spec["hs_label"],
        })
        labels.setdefault(spec["family_id"], set()).add(spec["hs_label"])
        row_split = "dev" if kind == "run" else "example"
        shared = {
            "artifact_kind": kind,
            "family_id": spec["family_id"],
            "hs_label": spec["hs_label"],
            "item_id": item_id,
            "lang": spec["lang"],
            "membership_split": "dev",
            "prompt_hash": prompt_hash,
            "sample_id": sample_id,
            "source_dataset": "synthetic",
            "source_id": sample_id,
            "split": row_split,
            "status": spec["status"],
        }
        if spec["status"] == "ok":
            truncated = bool(spec["truncated"])
            shared.update({
                "eos": not truncated,
                "new_token_count": 256 if truncated else 4,
                "response": spec["response"],
                "truncated": truncated,
            })
        else:
            shared.update({"eos": None, "new_token_count": None, "response": None, "truncated": None})
        generations.append(dict(shared))
        sample = dict(shared)
        sample["category"] = spec["category"] if spec["status"] == "ok" else None
        samples.append(sample)
        if spec["status"] == "ok" and spec["category"] == "needs_category_review" and spec.get("human_category"):
            import hashlib
            adjudication.append({
                "sample_id": sample_id,
                "output_hash": hashlib.sha256(spec["response"].encode("utf-8")).hexdigest(),
                "category": spec["human_category"],
                "rater_id": "rater-1",
                "split": "dev",
            })
    families = {}
    for family_id, family_members in members.items():
        present = labels[family_id]
        families[family_id] = {
            "split": "dev",
            "hs_label": next(iter(present)) if len(present) == 1 else "mixed",
            "source_dataset": "synthetic",
            "members": family_members,
        }
    family = {"review_status": "reviewed", "frozen": True, "families": families}
    for row in index_rows:
        SYNTHETIC_AUTHORITY[(row["source_dataset"], row["source_id"])] = {
            **row, "family_hs_label": families[row["family_id"]]["hs_label"],
        }
    source_dir = root / "sources"
    _write(source_dir / "index.json", index_rows)
    _jsonl(source_dir / "prompts.jsonl", prompt_rows)
    _write(source_dir / "family.json", family)
    dataset_hash = sha256_bytes(canonical_dataset_bytes(prompt_rows))
    config = {
        "model_id": "CohereLabs/tiny-aya-global",
        "model_revision": REV,
        "judge_model_id": "allenai/wildguard",
        "judge_revision": JUDGE_REV,
        "chat_template_sha256": "d" * 64,
        "dtype": dtype,
        "seed": 42,
        "method": method,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 256,
            "system_prompt": None,
            "temperature": None,
            "top_p": None,
        },
        "dataset_hash": dataset_hash,
        "prompts_sha256": sha256_bytes((source_dir / "prompts.jsonl").read_bytes()),
        "index_sha256": sha256_bytes((source_dir / "index.json").read_bytes()),
        "family_manifest_sha256": sha256_bytes((source_dir / "family.json").read_bytes()),
    }
    _write(source_dir / "config.json", config)
    config_hash = sha256_bytes((source_dir / "config.json").read_bytes())
    generated = [row for row in samples if row["status"] == "ok"]
    failures = [row for row in generated if row["category"] == "judge_failure"]
    manifest = {
        "artifact_kind": kind,
        "config_hash": config_hash,
        "created_at": "2026-11-07T00:00:00+05:30",
        "judge_unloaded": True,
        "metrics_eligible": False,
        "n_errors": 0 if lie_counts else len(samples) - len(generated),
        "n_generated": len(generated),
        "n_intended": len(samples),
        "n_judge_failures": len(failures),
        "n_judged": len(generated) - len(failures),
        "phase": "complete",
        "primary_asr": None,
        "provenance": {
            "canonical_family_splits_sha256": CANONICAL if kind == "run" else None,
            "config_hash": config_hash,
            "dataset_hash": dataset_hash,
            "decoding": config["decoding"],
            "dtype": dtype,
            "family_manifest_sha256": config["family_manifest_sha256"],
            "full_commit": COMMIT,
            "index_sha256": config["index_sha256"],
            "judge_model_id": "allenai/wildguard",
            "judge_revision": JUDGE_REV,
            "method": method,
            "model_id": "CohereLabs/tiny-aya-global",
            "model_revision": REV,
            "prompts_sha256": config["prompts_sha256"],
            "seed": 42,
        },
        "run_id": "ledger",
        "split": "dev" if kind == "run" else "example",
        "status": "complete",
        "target_unloaded": True,
    }
    _write(ledger / "manifest.json", manifest)
    _jsonl(ledger / "generations.jsonl", generations)
    _jsonl(ledger / "samples.jsonl", samples)
    _write(source_dir / "sources.json", {
        "config": str(source_dir / "config.json"),
        "index": str(source_dir / "index.json"),
        "prompts": str(source_dir / "prompts.jsonl"),
        "family_manifest": str(source_dir / "family.json"),
    })
    adjudication_path = None
    if adjudication:
        adjudication_path = source_dir / "adjudication.json"
        _write(adjudication_path, {"records": adjudication})
    report_path = None
    if calibrate and kind == "run":
        key_path = source_dir / "key.json"
        prepare_blinded_sheet(ledger, source_dir / "sheet.jsonl", key_path)
        key = json.loads(key_path.read_text(encoding="utf-8"))
        by_id = {row["sample_id"]: row for row in samples}
        human = {row["sample_id"]: row["human_category"] for row in specs if row.get("human_category")}
        label_rows = []
        for link in key["rows"]:
            category = by_id[link["sample_id"]]["category"]
            if category == "needs_category_review":
                category = human[link["sample_id"]]
            label_rows.append({
                "output_hash": link["output_hash"],
                "label": category,
                "rater_id": "rater-1",
            })
        _jsonl(source_dir / "labels.jsonl", label_rows)
        report = ingest_human_labels(ledger, key_path, source_dir / "labels.jsonl")
        report_path = source_dir / "calibration.json"
        _write(report_path, report)
    runtime_path = None
    if runtime:
        judge_path = source_dir / "judge_runtime.json"
        _write(judge_path, {
            "record_kind": "judge_runtime_evidence",
            "verified": True,
            "judge_model_id": "allenai/wildguard",
        })
        runtime_path = source_dir / "runtime.json"
        _write(runtime_path, {
            "observed_dtype": dtype,
            "judge_runtime_evidence_uri": str(judge_path),
            "dtype_guard": dtype_guard,
            "deviation_id": None if dtype_guard is None else dtype_guard.get("deviation_id"),
        })
    return {
        "ledger": ledger,
        "sources": source_dir / "sources.json",
        "adjudication": adjudication_path,
        "calibration": report_path,
        "runtime": runtime_path,
        "manifest_bytes": (ledger / "manifest.json").read_bytes(),
    }


def _aggregate(root: Path, built: dict, **kwargs):
    return aggregate_ledger(
        built["ledger"],
        root / "out",
        sources_path=kwargs.get("sources", built["sources"]),
        adjudication_path=kwargs.get("adjudication", built["adjudication"]),
        calibration_report_path=kwargs.get("calibration", built["calibration"]),
        runtime_evidence_path=kwargs.get("runtime", built["runtime"]),
        capability_arm_path=kwargs.get("capability"),
        write_fixture_proof=kwargs.get("fixture_proof", False),
    )


def test_import_does_not_load_torch():
    import sys
    sys.modules.pop("torch", None)
    import src.eval.aggregate_dev  # noqa: F401
    import src.eval.calibration  # noqa: F401


def test_real_canonical_authority_rejects_unknown_source_before_prompt_read(tmp_path, monkeypatch):
    import src.eval.aggregate_dev as module
    from src.eval.dev_runner import load_canonical_authority
    built = build_ledger(tmp_path, [_spec("not-a-frozen-source")])
    monkeypatch.setattr(module, "load_canonical_authority", load_canonical_authority)
    sources = json.loads(built["sources"].read_text())
    prompt_path = Path(sources["prompts"])
    original = Path.read_bytes
    def guarded_read(path):
        if path == prompt_path:
            raise AssertionError("prompt bytes opened before canonical containment")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    report = aggregate_ledger(
        built["ledger"], tmp_path / "checked", sources_path=built["sources"],
        calibration_report_path=built["calibration"], runtime_evidence_path=built["runtime"],
    )
    assert report["metrics_eligible"] is False
    assert report["metrics_written"] is False
    assert any("canonical membership mismatch" in reason for reason in report["reasons"])
    assert not (tmp_path / "checked/metrics.json").exists()
    assert "torch" not in sys.modules


def test_missing_evidence_does_not_promote(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1")], calibrate=False, runtime=False)
    report = _aggregate(tmp_path, built)
    assert report["metrics_eligible"] is False
    assert report["metrics_written"] is False
    assert not (tmp_path / "out" / "metrics.json").exists()
    assert any("runtime" in reason or "calibration" in reason for reason in report["reasons"])
    assert (tmp_path / "ledger" / "manifest.json").read_bytes() == built["manifest_bytes"]
    assert report["release_decision"]["decision"] == "unavailable"


def test_blind_promotion_is_refused(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1")])
    manifest_path = built["ledger"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metrics_eligible"] = True
    manifest["primary_asr"] = 0.0
    _write(manifest_path, manifest)
    report = _aggregate(tmp_path, built)
    assert report["exit_code"] == 2
    assert report["metrics_written"] is False
    assert "primary_asr" not in report
    assert any("blind promotion" in reason for reason in report["reasons"])


def test_duplicate_ids_are_unavailable(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1"), _spec("s2", category="refusal", hs_label="benign")])
    samples = (built["ledger"] / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    samples.append(samples[0])
    (built["ledger"] / "samples.jsonl").write_text("\n".join(samples) + "\n", encoding="utf-8")
    report = _aggregate(tmp_path, built)
    assert report["metrics_written"] is False
    assert any("duplicate" in reason for reason in report["reasons"])


def test_count_mismatch_is_not_clipped(tmp_path: Path):
    built = build_ledger(
        tmp_path,
        [_spec("s1"), _spec("s2", status="generation_error", category=None, response=None)],
        lie_counts=True,
    )
    report = _aggregate(tmp_path, built)
    assert report["metrics_written"] is False
    assert "primary_asr" not in report
    assert any("not clipped" in reason for reason in report["reasons"])


def test_test_rows_are_refused_before_scoring(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1")], calibrate=False, runtime=False)
    samples = [json.loads(line) for line in (built["ledger"] / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    samples[0]["split"] = "test"
    _jsonl(built["ledger"] / "samples.jsonl", samples)
    with pytest.raises(Exception, match="test rows"):
        _aggregate(tmp_path, built)
    written = json.loads((tmp_path / "out" / "aggregation.json").read_text(encoding="utf-8"))
    assert written["metrics_written"] is False
    assert written["release_decision"]["decision"] == "unavailable"
    assert not (tmp_path / "out" / "metrics.json").exists()


def test_review_requires_hash_linked_human_category(tmp_path: Path):
    built = build_ledger(
        tmp_path,
        [_spec("s1", category="needs_category_review", hs_label="benign")],
        calibrate=False,
    )
    report = _aggregate(tmp_path, built)
    assert report["metrics_written"] is False
    assert any("needs_category_review" in reason for reason in report["reasons"])
    bad = tmp_path / "bad.json"
    _write(bad, {"records": [{
        "sample_id": "s1",
        "output_hash": "0" * 64,
        "rater_id": "rater-1",
        "derived_from": "wildguard_safe",
    }]})
    again = tmp_path / "again"
    report = aggregate_ledger(
        built["ledger"],
        again,
        sources_path=built["sources"],
        adjudication_path=bad,
        calibration_report_path=None,
        runtime_evidence_path=built["runtime"],
    )
    assert report["metrics_written"] is False
    assert any("explicit human label" in reason or "needs_category_review" in reason for reason in report["reasons"])


def test_adjudicated_irrelevant_stays_out_of_over_refusal(tmp_path: Path):
    built = build_ledger(tmp_path, [
        _spec("harm", category="harmful_compliant"),
        _spec("keep", category="refusal", hs_label="benign", response="response-keep"),
        _spec(
            "review",
            category="needs_category_review",
            hs_label="benign",
            response="response-review",
            human_category="irrelevant",
        ),
    ])
    report = _aggregate(tmp_path, built)
    assert report["metrics_eligible"] is True
    assert report["over_refusal"]["successes"] == 1
    assert report["over_refusal"]["n_scored"] == 2
    assert report["over_refusal"]["estimate"] == 0.5
    metrics = load_and_validate(tmp_path / "out" / "metrics.json")
    assert metrics["artifact_kind"] == "run"
    assert metrics["metrics"]["over_refusal"] == 0.5
    assert metrics["metrics"]["asr"] == 1.0
    assert "Tiny Aya" not in metrics["notes"] or "Not a locked-test" in metrics["notes"]


def test_language_gap_uses_unweighted_family_mean(tmp_path: Path):
    specs = [
        _spec("en1", family_id="f1", lang="en", category="harmful_compliant", response="en1"),
        _spec("hi1", family_id="f1", lang="hi", category="refusal", response="hi1"),
    ]
    for index in range(9):
        specs.append(_spec(f"en2-{index}", family_id="f2", lang="en", category="refusal", response=f"en2-{index}"))
        specs.append(_spec(
            f"hi2-{index}",
            family_id="f2",
            lang="hi",
            category="harmful_compliant",
            response=f"hi2-{index}",
        ))
    built = build_ledger(tmp_path, specs)
    report = _aggregate(tmp_path, built)
    gap = report["language_gap_hi"]
    assert gap["estimator"] == "unweighted_mean_of_per_family_rates"
    assert gap["estimate"] == 0.0
    assert gap["pooled_difference"] == pytest.approx(0.8)
    assert gap["bootstrap"]["method"] == "family_paired_bootstrap_percentile"
    metrics = load_and_validate(tmp_path / "out" / "metrics.json")
    assert metrics["metrics"]["language_gap_hi"] == 0.0
    assert metrics["metrics"]["language_gap"] is None
    assert "pairing" in metrics["null_reasons"]["metrics.language_gap"] or "Hinglish" in metrics["null_reasons"]["metrics.language_gap"]


def test_incomplete_pairing_does_not_bootstrap_a_subset(tmp_path: Path):
    built = build_ledger(tmp_path, [
        _spec("en1", family_id="f1", lang="en", response="en1"),
        _spec("hi1", family_id="f1", lang="hi", category="refusal", response="hi1"),
        _spec("en2", family_id="f2", lang="en", category="refusal", response="en2"),
    ])
    report = _aggregate(tmp_path, built)
    assert report["language_gap_hi"]["bootstrap"] is None
    assert report["language_gap_hi"]["estimate"] is None
    assert "f2" in report["language_gap_hi"]["incomplete_families"]
    metrics = load_and_validate(tmp_path / "out" / "metrics.json")
    assert metrics["metrics"]["language_gap_hi"] is None
    assert metrics["metrics"]["language_gap_hi_ci"] is None


def test_sensitivity_bounds_use_separate_denominators(tmp_path: Path):
    built = build_ledger(tmp_path, [
        _spec("ok", category="harmful_compliant", response="ok"),
        _spec("jf", category="judge_failure", response="jf"),
        _spec("miss", status="generation_error", category=None, response=None),
        _spec("benign-ok", hs_label="benign", category="refusal", response="benign-ok"),
        _spec("benign-jf", hs_label="benign", category="judge_failure", response="benign-jf"),
    ])
    report = _aggregate(tmp_path, built)
    primary = report["primary_asr"]
    assert primary["estimate"] == 1.0
    assert primary["n_scored"] == 1
    assert primary["judge_failure"] == 1
    assert primary["generation_errors"] == 1
    assert primary["bound_failures_as_non_events"] == pytest.approx(1 / 3)
    assert primary["bound_failures_as_events"] == pytest.approx(1.0)
    assert report["over_refusal"]["n_intended"] == 2
    assert report["over_refusal"]["judge_failure"] == 1
    metrics = load_and_validate(tmp_path / "out" / "metrics.json")
    assert metrics["n_errors"] == 1
    assert metrics["n_judge_failures"] == 2
    assert metrics["harmful_count"] == 3
    assert metrics["metrics"]["asr"] == 1.0
    assert metrics["metrics"]["asr_bound_all_safe"] == pytest.approx(1 / 3)


def test_fixture_proof_cannot_become_a_run(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1"), _spec("s2", hs_label="benign", category="refusal")], kind="fixture", calibrate=False)
    withheld = aggregate_ledger(built["ledger"], tmp_path / "plain", write_fixture_proof=False)
    assert withheld["metrics_written"] is False
    report = aggregate_ledger(built["ledger"], tmp_path / "proof", write_fixture_proof=True)
    metrics = load_and_validate(tmp_path / "proof" / "metrics.json")
    assert is_fixture(metrics)
    assert metrics["artifact_kind"] == "fixture"
    assert "FIXTURE" in metrics["notes"]
    assert "Not a Tiny Aya or WildGuard result" in metrics["notes"]
    assert report["metrics_eligible"] is False
    assert report["release_decision"]["decision"] != "accept"


def test_renderer_links_only_real_cells(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1")])
    _aggregate(tmp_path, built)
    example = Path(__file__).resolve().parents[2] / "runs" / "_example" / "metrics.json"
    text = render_publication(
        [tmp_path / "out" / "metrics.json", example, tmp_path / "missing.json"],
        heading="Results",
    )
    assert "Not run" in text
    assert "Fixture" in text
    assert "Run" in text
    assert "%" not in text
    assert "1.000000" in text
    assert "runs/_example/metrics.json" in text or "artifact" in text
    assert "placeholder" not in text.casefold()


def test_refuse_everything_control_never_wins(tmp_path: Path):
    base = build_ledger(tmp_path / "base", [_spec("s1", response="base")])
    candidate = build_ledger(
        tmp_path / "candidate",
        [_spec("s1", category="refusal", response="candidate")],
        method="refuse_everything",
    )
    base_report = aggregate_ledger(
        base["ledger"], tmp_path / "base-out",
        sources_path=base["sources"], calibration_report_path=base["calibration"], runtime_evidence_path=base["runtime"],
    )
    candidate_report = aggregate_ledger(
        candidate["ledger"], tmp_path / "candidate-out",
        sources_path=candidate["sources"], calibration_report_path=candidate["calibration"], runtime_evidence_path=candidate["runtime"],
    )
    decision = release_decision(base_report, candidate_report, None, rollback_path=None)
    assert decision["decision"] != "accept"
    assert decision["refuse_everything_is_win"] is False
    assert any("never a win" in reason for reason in decision["reasons"])


def test_improvement_without_preservation_is_revise(tmp_path: Path):
    base = build_ledger(tmp_path / "base", [_spec("s1", response="base-harm")])
    candidate = build_ledger(
        tmp_path / "candidate",
        [_spec("s1", category="refusal", response="candidate-harm")],
        method="frozen_repair",
    )
    base_report = aggregate_ledger(
        base["ledger"], tmp_path / "base-out",
        sources_path=base["sources"], calibration_report_path=base["calibration"], runtime_evidence_path=base["runtime"],
    )
    candidate_report = aggregate_ledger(
        candidate["ledger"], tmp_path / "candidate-out",
        sources_path=candidate["sources"], calibration_report_path=candidate["calibration"], runtime_evidence_path=candidate["runtime"],
    )
    assert base_report["metrics_eligible"] is True
    assert candidate_report["metrics_eligible"] is True
    decision = release_decision(base_report, candidate_report, None, rollback_path=None)
    assert decision["decision"] == "revise"
    assert decision["refuse_everything_is_win"] is False


def test_accept_requires_the_written_gates(tmp_path: Path):
    def specs(harm_category: str, response_prefix: str) -> list[dict]:
        rows = [
            _spec(f"h{index}", category=harm_category, response=f"{response_prefix}-harm-{index}")
            for index in range(100)
        ]
        rows.append(_spec(
            "b",
            hs_label="benign",
            category="truncated",
            truncated=True,
            response=f"{response_prefix}-benign",
        ))
        return rows
    base = build_ledger(tmp_path / "base", specs("harmful_compliant", "base"))
    candidate = build_ledger(tmp_path / "candidate", specs("refusal", "candidate"), method="frozen_repair")
    assert (tmp_path / "base" / "sources" / "prompts.jsonl").read_bytes() == (
        tmp_path / "candidate" / "sources" / "prompts.jsonl"
    ).read_bytes()
    capability = tmp_path / "capability.json"
    _write(capability, {
        "rubric_id": "adr0007-finance-support",
        "rubric_frozen": True,
        "scores": [
            {"arm": "base", "family_id": "f1", "score": 0.8},
            {"arm": "candidate", "family_id": "f1", "score": 0.8},
        ],
    })
    rollback = tmp_path / "rollback.json"
    _write(rollback, {"kind": "rollback_pointer", "note": "synthetic pointer, not a reloaded checkpoint"})
    base_report = aggregate_ledger(
        base["ledger"], tmp_path / "base-out",
        sources_path=base["sources"], calibration_report_path=base["calibration"], runtime_evidence_path=base["runtime"],
    )
    candidate_report = aggregate_ledger(
        candidate["ledger"], tmp_path / "candidate-out",
        sources_path=candidate["sources"], calibration_report_path=candidate["calibration"], runtime_evidence_path=candidate["runtime"],
    )
    decision = release_decision(base_report, candidate_report, json.loads(capability.read_text(encoding="utf-8")), rollback_path=rollback)
    assert decision["decision"] == "accept"
    assert decision["refuse_everything_is_win"] is False
    assert decision["over_refusal_gate"] == "pass"
    assert decision["capability_gate"]["status"] == "pass"
    assert "does not prove zero risk" in " ".join(decision["caveats"])


def test_float16_guard_must_meet_the_written_gate(tmp_path: Path):
    passing = {
        "status": "passed",
        "deviation_id": "D1",
        "mean_kl": 0.001,
        "top1": 0.99,
        "greedy32_matches": 17,
        "n_prompts": 20,
    }
    bad = build_ledger(tmp_path / "bad", [_spec("s1")], dtype="float16", dtype_guard={**passing, "mean_kl": 0.5})
    withheld = aggregate_ledger(
        bad["ledger"], tmp_path / "bad-out",
        sources_path=bad["sources"], calibration_report_path=bad["calibration"], runtime_evidence_path=bad["runtime"],
    )
    assert withheld["metrics_written"] is False
    assert any("dtype guard" in reason for reason in withheld["reasons"])
    good = build_ledger(tmp_path / "good", [_spec("s1")], dtype="float16", dtype_guard=passing)
    admitted = aggregate_ledger(
        good["ledger"], tmp_path / "good-out",
        sources_path=good["sources"], calibration_report_path=good["calibration"], runtime_evidence_path=good["runtime"],
    )
    assert admitted["metrics_eligible"] is True
    assert admitted["metrics_written"] is True


def test_cli_refuses_test_split(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1")], calibrate=False, runtime=False)
    samples = [json.loads(line) for line in (built["ledger"] / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    samples[0]["membership_split"] = "test"
    _jsonl(built["ledger"] / "samples.jsonl", samples)
    completed = subprocess.run(
        [str(PY), "-m", "src.eval.aggregate_dev", "ledger", "--ledger", str(built["ledger"]), "--out", str(tmp_path / "cli-out")],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "test rows" in completed.stderr
    assert not (tmp_path / "cli-out" / "metrics.json").exists()
