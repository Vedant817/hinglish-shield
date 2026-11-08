"""D8 descriptive status. No weights and no scored baseline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.d8_descriptive import d8_blocker_record
from src.eval.dev_runner import D8_METRIC_LABEL, default_d8_judge_factory


def test_factory_names_both_judges_without_torch():
    sys.modules.pop("torch", None)
    assert "default_judge_factory" in default_d8_judge_factory.__code__.co_names
    assert "LLMSecondaryJudge" in default_d8_judge_factory.__code__.co_names
    assert "torch" not in sys.modules


def test_v11_identity_blocker_is_not_a_scored_baseline():
    identity = ROOT / "runs" / "tokenizer-metadata-v11-2026-11-08" / "identity.json"
    record = d8_blocker_record(identity)
    assert record["artifact_kind"] == "blocker"
    assert record["scored_baseline"] is False
    assert record["metrics_eligible"] is False
    assert record["primary_asr"] is None
    assert record["metrics"]["asr"] is None
    assert record["metric_label"] == D8_METRIC_LABEL
    assert record["blocker"] == "target chat-template pin is unavailable"
    assert record["identity"]["failure_phase"] == "secret_missing"
    assert record["identity"]["public_http_status"] == 401
    assert record["human_labels"] == "not invented"
    blob = json.dumps(record)
    assert "HF_TOKEN" not in blob
    assert "hf_" not in blob.casefold()


def test_missing_identity_stays_unscored(tmp_path: Path):
    record = d8_blocker_record(tmp_path / "missing.json")
    assert record["artifact_kind"] == "blocker"
    assert record["metrics"]["asr"] is None
    assert record["blocker"] == "identity artifact missing"
