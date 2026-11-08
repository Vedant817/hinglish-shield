"""Locked-test preparation contracts. Never executes or reads the test split."""
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.eval import run_baseline as runner

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "LOCKED_TEST_PREP.md"
CONFIG = ROOT / "configs" / "eval" / "test_locked.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_embargo_is_2026_11_11_ist():
    expected = datetime(2026, 11, 11, tzinfo=runner.IST)
    assert runner.TEST_EMBARGO_END == expected


def test_frozen_config_is_still_not_admissible():
    cfg = _config()
    assert cfg["status"] == "locked_test_pending"
    assert cfg["frozen_repair"]["manifest"] is None
    assert cfg["frozen_repair"]["sha256"] is None
    assert cfg["datasets"]["dataset_hash"] is None


def test_guard_refuses_before_embargo_without_executing():
    cfg = {"splits": ["test"], "role": "h1_primary_single_shot",
           "method_arms": ["base", "refuse_everything", "frozen_repair"]}
    original_now = runner._utc_now
    runner._utc_now = lambda: datetime(2026, 11, 10, 18, 29, tzinfo=timezone.utc)
    try:
        problems = runner._validate_test_single_shot(
            cfg, split="test", i_understand=True,
            config_path=Path("configs/eval/test_locked.yaml"))
    finally:
        runner._utc_now = original_now
    assert any("test embargo" in problem for problem in problems)


def test_prep_doc_states_command_and_not_run():
    text = DOC.read_text(encoding="utf-8")
    assert "not run" in text
    assert ("python -m src.eval.run_baseline --config configs/eval/test_locked.yaml "
            "--split test --i-understand-single-shot") in text
    assert "--dry-check" in text
    assert "2026-11-11" in text
    assert not (ROOT / "runs" / runner.ATTEMPT_FILENAME).exists()
    assert "\u2014" not in text
