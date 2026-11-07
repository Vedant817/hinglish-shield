"""Identity-only holdout candidate checks. No prompt files or model calls."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from src.curate.holdout_candidate import HISTORICAL_FIELDS, clarify_candidate_summaries

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "data/splits/family_splits_v2.json"
CANDIDATE = ROOT / "data/splits/proposals/family_splits_v3_holdout_candidate.json"


def test_historical_v2_summaries_cannot_be_mistaken_for_candidate_population():
    original = json.loads(ORIGINAL.read_bytes())
    candidate = json.loads(CANDIDATE.read_bytes())
    review = candidate["candidate_review"]
    assert review["historical_summary_fields"] == list(HISTORICAL_FIELDS)
    assert "not candidate denominators" in review["historical_summary_scope"]
    for field in HISTORICAL_FIELDS:
        assert candidate[field] == original[field]
    families, members = Counter(), Counter()
    for family in candidate["families"].values():
        families[family["split"]] += 1
        members[family["split"]] += len(family["members"])
    assert review["prospective_population"] == {
        split: {"families": families[split], "members": members[split]}
        for split in families
    }
    assert review["prospective_population"]["test"] == {"families": 2396, "members": 2548}
    assert review["prospective_population"]["quarantine"] == {"families": 1, "members": 2}


def test_family_membership_and_active_inputs_are_preserved():
    original = json.loads(ORIGINAL.read_bytes())
    candidate = json.loads(CANDIDATE.read_bytes())
    expected = deepcopy(original["families"])
    expected["fam_983e49e2566f"]["split"] = "quarantine"
    assert candidate["families"] == expected
    assert candidate["frozen"] is False
    assert candidate["candidate_review"]["execution_authorized"] is False
    assert candidate["candidate_review"]["replacement_count"] == 0
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == "334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531"


def test_receipts_and_correction_are_consistent_and_idempotent():
    raw = CANDIDATE.read_bytes()
    candidate = json.loads(raw)
    original = json.loads(ORIGINAL.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    assert Path(str(CANDIDATE) + ".sha256").read_text().split()[0] == digest
    report = json.loads((CANDIDATE.parent / "holdout_v3_candidate_review.json").read_bytes())
    assert report["candidate_sha256"] == digest
    for key in ("historical_summary_fields", "historical_summary_scope", "prospective_population"):
        assert report[key] == candidate["candidate_review"][key]
    assert candidate == clarify_candidate_summaries(candidate, original)


def test_correction_refuses_activation_or_rewriting_v2_history():
    original = {"counts": {"test": 2}, "realized_ratios": {}, "harm_share": {}}
    candidate = {**deepcopy(original), "frozen": False,
                 "candidate_review": {"execution_authorized": False}, "families": {}}
    candidate["frozen"] = True
    with pytest.raises(ValueError, match="inactive"):
        clarify_candidate_summaries(candidate, original)
    candidate["frozen"] = False
    candidate["counts"]["test"] = 1
    with pytest.raises(ValueError, match="historical"):
        clarify_candidate_summaries(candidate, original)
