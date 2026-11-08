"""Dev calibration sheet and human-label contracts. No test prompts are opened."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.calibration import CalibrationError, ingest_human_labels, output_hash, prepare_blinded_sheet
from src.eval.dev_runner import sha256_text
from src.eval.judges import cohens_kappa
from tests.eval.test_aggregate_dev import build_ledger, _jsonl, _spec


def _sheet(tmp_path: Path, specs: list[dict]):
    built = build_ledger(tmp_path, specs, calibrate=False, runtime=False)
    sheet = tmp_path / "sheet.jsonl"
    key = tmp_path / "key.json"
    prepare_blinded_sheet(built["ledger"], sheet, key)
    return built, sheet, key


def test_sheet_is_blinded_and_dev_only(tmp_path: Path):
    _built, sheet, key_path = _sheet(tmp_path, [
        _spec("s1", response="visible output"),
        _spec("s2", status="generation_error", category=None, response=None),
    ])
    rows = [json.loads(line) for line in sheet.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) == {"sheet_row_id", "output_hash", "lang", "response"}
    assert rows[0]["response"] == "visible output"
    assert rows[0]["output_hash"] == output_hash("visible output")
    blob = sheet.read_text(encoding="utf-8")
    assert "harmful_compliant" not in blob
    assert "sample_id" not in blob
    assert "wildguard" not in blob.casefold()
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key["schema_version"] == "calibration_sheet_key_v2"
    assert key["sheet_kind"] == "dev"
    assert key["ledger_run_id"] == "ledger"
    assert key["omitted_generation_errors"] == 1
    assert key["rows"][0]["run_id"] == "ledger"
    assert key["rows"][0]["sample_id"] == "s1"
    assert key["rows"][0]["prompt_hash"] == sha256_text("synthetic prompt s1")


def test_ingest_binds_pairs_to_run_sample_and_prompt_hash(tmp_path: Path):
    built, _sheet_path, key_path = _sheet(tmp_path, [_spec("s1", category="refusal", response="only")])
    key = json.loads(key_path.read_text(encoding="utf-8"))
    _jsonl(tmp_path / "labels.jsonl", [{
        "output_hash": key["rows"][0]["output_hash"],
        "label": "refusal",
        "rater_id": "rater-1",
    }])
    report = ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")
    assert report["run_id"] == "ledger"
    pair = report["pairs"][0]
    assert pair["run_id"] == "ledger"
    assert pair["sample_id"] == "s1"
    assert pair["prompt_hash"] == sha256_text("synthetic prompt s1")


@pytest.mark.parametrize(("mutation", "needle"), [
    ("top_run", "run id"),
    ("row_run", "run id"),
    ("sample", "sample id"),
    ("prompt", "prompt hash"),
])
def test_key_must_match_the_ledger_identity(tmp_path: Path, mutation: str, needle: str):
    built, _sheet_path, key_path = _sheet(tmp_path, [_spec("s1", category="refusal", response="only")])
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if mutation == "top_run":
        key["ledger_run_id"] = "another-run"
    elif mutation == "row_run":
        key["rows"][0]["run_id"] = "another-run"
    elif mutation == "sample":
        key["rows"][0]["sample_id"] = "another-sample"
    else:
        key["rows"][0]["prompt_hash"] = "0" * 64
    key_path.write_text(json.dumps(key), encoding="utf-8")
    _jsonl(tmp_path / "labels.jsonl", [{
        "output_hash": key["rows"][0]["output_hash"],
        "label": "refusal",
        "rater_id": "rater-1",
    }])
    with pytest.raises(CalibrationError, match=needle):
        ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")


def test_test_split_does_not_write_a_sheet(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1")], calibrate=False, runtime=False)
    samples = [json.loads(line) for line in (built["ledger"] / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    samples[0]["split"] = "test"
    _jsonl(built["ledger"] / "samples.jsonl", samples)
    sheet = tmp_path / "sheet.jsonl"
    with pytest.raises(CalibrationError, match="test"):
        prepare_blinded_sheet(built["ledger"], sheet, tmp_path / "key.json")
    assert not sheet.exists()


def test_ingest_matches_kappa_and_keeps_hinglish_unvalidated(tmp_path: Path):
    built, sheet, key_path = _sheet(tmp_path, [
        _spec("en", lang="en", category="refusal", response="en-out"),
        _spec("hi", lang="hi", category="harmful_compliant", response="hi-out"),
    ])
    key = json.loads(key_path.read_text(encoding="utf-8"))
    labels = [
        {"output_hash": key["rows"][0]["output_hash"], "label": "refusal", "rater_id": "rater-1"},
        {"output_hash": key["rows"][1]["output_hash"], "label": "refusal", "rater_id": "rater-1"},
    ]
    _jsonl(tmp_path / "labels.jsonl", labels)
    report = ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")
    assert report["labels_admissible"] is True
    assert report["n_duplicate"] == 0
    assert report["n_unmatched"] == 0
    assert report["n_missing"] == 0
    assert report["agreement"] == 0.5
    assert report["cohens_kappa"] == cohens_kappa(
        [pair["human_label"] for pair in report["pairs"]],
        [pair["judge_category"] for pair in report["pairs"]],
    )
    assert report["by_language"]["en"]["validated"] is True
    assert report["by_language"]["hi"]["validated"] is True
    assert report["by_language"]["hinglish"]["validated"] is False
    assert "unvalidated until human labels" in report["by_language"]["hinglish"]["reason"]
    assert "category" not in sheet.read_text(encoding="utf-8")


def test_duplicate_and_unmatched_labels_are_errors(tmp_path: Path):
    built, _sheet_path, key_path = _sheet(tmp_path, [_spec("s1", category="refusal", response="only")])
    key = json.loads(key_path.read_text(encoding="utf-8"))
    digest = key["rows"][0]["output_hash"]
    _jsonl(tmp_path / "labels.jsonl", [
        {"output_hash": digest, "label": "refusal", "rater_id": "rater-1"},
        {"output_hash": digest, "label": "irrelevant", "rater_id": "rater-2"},
        {"output_hash": "f" * 64, "label": "refusal", "rater_id": "rater-1"},
    ])
    report = ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")
    assert report["labels_admissible"] is False
    assert report["n_duplicate"] == 1
    assert report["n_unmatched"] == 1
    assert report["n_missing"] == 1
    assert report["by_language"]["en"]["validated"] is False
    assert report["kappa_coverage"] == "partial_not_validated"


def test_missing_hi_labels_stay_unvalidated(tmp_path: Path):
    built, _sheet_path, key_path = _sheet(tmp_path, [
        _spec("en", lang="en", category="refusal", response="en-out"),
        _spec("hi", lang="hi", category="refusal", response="hi-out"),
    ])
    key = json.loads(key_path.read_text(encoding="utf-8"))
    en_hash = next(row["output_hash"] for row in key["rows"] if row["lang"] == "en")
    _jsonl(tmp_path / "labels.jsonl", [{"output_hash": en_hash, "label": "refusal", "rater_id": "rater-1"}])
    report = ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")
    assert report["n_missing"] == 1
    assert report["by_language"]["hi"]["validated"] is False
    assert report["by_language"]["hi"]["n_labeled"] == 0
    assert report["labels_admissible"] is False


def test_degenerate_kappa_is_null_not_zero(tmp_path: Path):
    built, _sheet_path, key_path = _sheet(tmp_path, [
        _spec("a", category="refusal", response="a-out"),
        _spec("b", category="refusal", response="b-out"),
    ])
    key = json.loads(key_path.read_text(encoding="utf-8"))
    _jsonl(tmp_path / "labels.jsonl", [
        {"output_hash": row["output_hash"], "label": "refusal", "rater_id": "rater-1"}
        for row in key["rows"]
    ])
    report = ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")
    assert report["agreement"] == 1.0
    assert report["cohens_kappa"] is None
    assert report["cohens_kappa"] != 0


def test_fixture_sheet_is_not_validation(tmp_path: Path):
    built = build_ledger(tmp_path, [_spec("s1", lang="hinglish", response="mix")], kind="fixture", calibrate=False, runtime=False)
    key_path = tmp_path / "key.json"
    prepare_blinded_sheet(built["ledger"], tmp_path / "sheet.jsonl", key_path)
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key["sheet_kind"] == "fixture"
    _jsonl(tmp_path / "labels.jsonl", [{
        "output_hash": key["rows"][0]["output_hash"],
        "label": "harmful_compliant",
        "rater_id": "rater-1",
    }])
    report = ingest_human_labels(built["ledger"], key_path, tmp_path / "labels.jsonl")
    assert report["by_language"]["hinglish"]["validated"] is False
    assert report["labels_admissible_for_run"] is False
