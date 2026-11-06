"""Guards for the locked test path (no scoring; synthetic config checks)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import run_baseline  # noqa: E402


class TestLockedGuards(unittest.TestCase):
    def test_baseline_yaml_still_dev_only(self) -> None:
        cfg = run_baseline._load_yaml(ROOT / "configs" / "eval" / "baseline.yaml")
        self.assertEqual(cfg.get("splits"), ["dev"])
        self.assertNotIn("test", cfg.get("splits") or [])

    def test_test_locked_lists_test_and_arms(self) -> None:
        cfg = run_baseline._load_yaml(ROOT / "configs" / "eval" / "test_locked.yaml")
        self.assertEqual(cfg.get("splits"), ["test"])
        arms = cfg.get("method_arms") or []
        self.assertIn("base", arms)
        self.assertIn("refuse_everything", arms)
        self.assertIn("frozen_repair", arms)
        unc = cfg.get("uncertainty") or {}
        self.assertEqual(unc.get("bootstrap_reps"), 2000)
        self.assertEqual(unc.get("bootstrap_seed"), 20261106)
        asr = cfg.get("asr") or {}
        self.assertEqual(asr.get("denominator_primary"), "n_scored")

    def test_test_without_flag_refuses(self) -> None:
        code = run_baseline.main(
            [
                "--config",
                str(ROOT / "configs" / "eval" / "test_locked.yaml"),
                "--split",
                "test",
                "--dry-check",
            ]
        )
        self.assertEqual(code, 2)

    def test_baseline_config_rejects_split_test(self) -> None:
        code = run_baseline.main(
            [
                "--config",
                str(ROOT / "configs" / "eval" / "baseline.yaml"),
                "--split",
                "test",
                "--i-understand-single-shot",
                "--dry-check",
            ]
        )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
