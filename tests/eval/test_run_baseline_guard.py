"""Guards for the locked test path (no scoring; synthetic config checks)."""
from __future__ import annotations

import sys
import json
import hashlib
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


def test_exposure_ledger_blocks_old_holdout_without_reading_prompts(tmp_path, monkeypatch):
    monkeypatch.setattr(run_baseline, "ROOT", tmp_path)
    split_dir = tmp_path / "data/splits"
    split_dir.mkdir(parents=True)
    manifest = split_dir / "membership.json"
    raw = json.dumps({"frozen": True, "families": {"exposed": {"split": "test"}}}).encode()
    manifest.write_bytes(raw)
    manifest.with_suffix(".json.sha256").write_text(hashlib.sha256(raw).hexdigest())
    (split_dir / "development_exposures.json").write_text(json.dumps({"entries": [
        {"family_id": "exposed", "source_dataset": "fixture", "source_id": "one"}]}))
    config = {"model_revision": "a" * 40, "splits": ["test"],
              "datasets": {"dataset_hash": "b" * 64, "split_manifest": "data/splits/membership.json"}}
    problems = run_baseline._validate_test_single_shot(config, split="test", i_understand=True,
                                                       config_path=tmp_path / "test_locked.yaml")
    assert any("development-exposed" in p for p in problems)
    (split_dir / "development_exposures.json").write_text("{")
    problems = run_baseline._validate_test_single_shot(config, split="test", i_understand=True,
                                                       config_path=tmp_path / "test_locked.yaml")
    assert any("unreadable or invalid" in p for p in problems)


def test_exposure_guard_requires_ledger_and_tracks_source_across_family_renames(tmp_path, monkeypatch):
    monkeypatch.setattr(run_baseline, "ROOT", tmp_path)
    split_dir = tmp_path / "data/splits"
    split_dir.mkdir(parents=True)
    manifest = split_dir / "membership.json"
    raw = json.dumps({"frozen": True, "families": {"renamed": {"split": "test", "members": [
        {"source_dataset": "fixture", "source_id": "one"}]}}}).encode()
    manifest.write_bytes(raw)
    Path(str(manifest) + ".sha256").write_text(hashlib.sha256(raw).hexdigest())
    config = {"model_revision": "a" * 40, "splits": ["test"],
              "datasets": {"dataset_hash": "b" * 64, "split_manifest": str(manifest)}}
    def problems():
        return run_baseline._validate_test_single_shot(config, split="test", i_understand=True,
                                                     config_path=tmp_path / "test_locked.yaml")
    assert any("exposure ledger" in p for p in problems())
    (split_dir / "development_exposures.json").write_text(json.dumps({"entries": [
        {"family_id": "original", "source_dataset": "fixture", "source_id": "one"}]}))
    assert any("development-exposed" in p for p in problems())
