"""Export identity contract. Synthetic prompt text only.

Membership ids are read from data/splits/family_lookup.csv (id columns).
This file does not open seed prompts, staging prompt CSVs, or curated JSONL.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.curate.export_to_curatorkit import (  # noqa: E402
    export_rows,
    main,
    map_batch_row,
    to_curatorkit,
)

LOOKUP = ROOT / "data" / "splits" / "family_lookup.csv"


def _member() -> dict[str, str]:
    with LOOKUP.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {
        "source_dataset": row["source_dataset"],
        "source_id": row["source_id"],
        "family_id": row["family_id"],
        "split": row["split"],
    }


def _batch(member: dict[str, str], **overrides: str) -> dict[str, str]:
    row = {
        "id": "fixture-001",
        "source": member["source_dataset"],
        "source_id": member["source_id"],
        "family_id": member["family_id"],
        "split": member["split"],
        "label": "benign",
        "category": "fixture-category",
        "qc_notes": "synthetic",
    }
    row.update(overrides)
    return row


def test_identity_preserved_on_eval_row_and_curatorkit_metadata() -> None:
    member = _member()
    row = map_batch_row(_batch(member), "en", "synthetic fixture prompt")
    assert row["family_id"] == member["family_id"]
    assert row["split"] == member["split"]
    assert row["variant_of"] is None
    assert row["variant_of"] != "fixture-001"
    assert row["source_dataset"] == member["source_dataset"]
    assert row["source_id"] == member["source_id"]
    assert row["item_id"] == "fixture-001:en"
    assert row["metadata"]["translation_batch_id"] == "fixture-001"
    assert row["metadata"]["family_id"] == member["family_id"]
    assert row["metadata"]["split"] == member["split"]
    ck = to_curatorkit(row)
    assert ck["metadata"]["family_id"] == member["family_id"]
    assert ck["metadata"]["split"] == member["split"]
    assert ck["metadata"]["variant_of"] is None
    assert ck["metadata"]["source_dataset"] == member["source_dataset"]
    assert ck["metadata"]["source_id"] == member["source_id"]
    assert ck["instruction"] == "synthetic fixture prompt"
    assert ck["task_type"] == "prompt_only"


def test_unreviewed_row_is_not_scoring_eligible() -> None:
    member = _member()
    row = map_batch_row(_batch(member), "hi", "synthetic fixture prompt")
    assert row["review_status"] == "unreviewed"
    assert row["scoring_eligible"] is False
    assert row["metadata"]["review_status"] == "unreviewed"
    assert row["metadata"]["scoring_eligible"] is False
    ck = to_curatorkit(row)
    assert ck["metadata"]["review_status"] == "unreviewed"
    assert ck["metadata"]["scoring_eligible"] is False
    assert "vedant_verdict" not in row
    assert "vedant_verdict" not in ck["metadata"]


def test_explicit_accept_is_preserved_and_not_invented() -> None:
    member = _member()
    accepted = map_batch_row(
        _batch(member, vedant_verdict="accept"),
        "hinglish",
        "synthetic fixture prompt",
    )
    assert accepted["review_status"] == "accept"
    assert accepted["scoring_eligible"] is True
    rejected = map_batch_row(
        _batch(member, vedant_verdict="reject"),
        "en",
        "synthetic fixture prompt",
    )
    assert rejected["review_status"] == "reject"
    assert rejected["scoring_eligible"] is False


def test_split_mismatch_rejected() -> None:
    member = _member()
    other = "dev" if member["split"] != "dev" else "calibration"
    with pytest.raises(ValueError, match="inconsistent split"):
        map_batch_row(_batch(member, split=other), "en", "synthetic fixture prompt")


def test_family_mismatch_rejected() -> None:
    member = _member()
    with pytest.raises(ValueError, match="inconsistent family_id"):
        map_batch_row(
            _batch(member, family_id="fam_not_in_manifest"),
            "en",
            "synthetic fixture prompt",
        )


def test_missing_family_or_split_rejected() -> None:
    member = _member()
    missing_family = _batch(member)
    missing_family["family_id"] = ""
    with pytest.raises(ValueError, match="missing family_id or split"):
        map_batch_row(missing_family, "en", "synthetic fixture prompt")
    missing_split = _batch(member)
    missing_split["split"] = ""
    with pytest.raises(ValueError, match="missing family_id or split"):
        map_batch_row(missing_split, "en", "synthetic fixture prompt")


def test_missing_membership_rejected() -> None:
    member = _member()
    with pytest.raises(ValueError, match="missing membership"):
        map_batch_row(
            _batch(member, source_id="not-a-member"),
            "en",
            "synthetic fixture prompt",
        )


def test_main_staging_export_keeps_lf_and_unreviewed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    member = _member()
    csv_path = tmp_path / "staging.csv"
    fieldnames = [
        "id", "source", "source_id", "family_id", "split",
        "english", "hindi", "hinglish", "label", "category", "qc_notes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "id": "fixture-001",
            "source": member["source_dataset"],
            "source_id": member["source_id"],
            "family_id": member["family_id"],
            "split": member["split"],
            "english": "synthetic en",
            "hindi": "synthetic hi",
            "hinglish": "synthetic hinglish",
            "label": "benign",
            "category": "fixture-category",
            "qc_notes": "synthetic",
        })
    out_hs = tmp_path / "hs.jsonl"
    out_ck = tmp_path / "ck.jsonl"
    monkeypatch.setattr(
        sys, "argv",
        ["export_to_curatorkit", "--csv", str(csv_path), "--out-hs", str(out_hs), "--out-ck", str(out_ck)],
    )
    main()
    for path in (out_hs, out_ck):
        raw = path.read_bytes()
        assert b"\r" not in raw
        assert raw.endswith(b"\n")
    hs_lines = out_hs.read_text(encoding="utf-8").splitlines()
    ck_lines = out_ck.read_text(encoding="utf-8").splitlines()
    assert len(hs_lines) == 3
    assert len(ck_lines) == 3
    hs = [json.loads(line) for line in hs_lines]
    ck = [json.loads(line) for line in ck_lines]
    assert [row["lang"] for row in hs] == ["en", "hi", "hinglish"]
    for row, ck_row in zip(hs, ck):
        assert row["family_id"] == member["family_id"]
        assert row["split"] == member["split"]
        assert row["review_status"] == "unreviewed"
        assert row["scoring_eligible"] is False
        assert ck_row["metadata"]["family_id"] == member["family_id"]
        assert ck_row["metadata"]["split"] == member["split"]
        assert ck_row["metadata"]["review_status"] == "unreviewed"
        assert ck_row["metadata"]["scoring_eligible"] is False
    # export_rows agrees with main on the same synthetic record.
    expanded = export_rows([_batch(member, english="synthetic en")])
    assert len(expanded) == 1
    assert expanded[0]["scoring_eligible"] is False
