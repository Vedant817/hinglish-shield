"""Synthetic contracts for the Hinglish slice builder. No model, no network."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def fixture_files(tmp_path):
    manifest = {
        "frozen": True,
        "families": {
            "fam_cal": {
                "split": "calibration",
                "members": [
                    {"source_dataset": "aya_redteaming", "source_id": "hin:0", "item_id": "aya_hin:0",
                     "lang": "hi", "hs_label": "harm"},
                ],
            },
            "fam_dev": {
                "split": "dev",
                "members": [
                    {"source_dataset": "aya_redteaming", "source_id": "hin:1", "item_id": "aya_hin:1",
                     "lang": "hi", "hs_label": "harm"},
                ],
            },
            "fam_test": {
                "split": "test",
                "members": [
                    {"source_dataset": "aya_redteaming", "source_id": "hin:2", "item_id": "aya_hin:2",
                     "lang": "hi", "hs_label": "harm"},
                ],
            },
        },
    }
    seeds = "\n".join([
        json.dumps({"prompt": "एक परीक्षण", "literal_translation": "test"}),
        json.dumps({"prompt": "दूसरा परीक्षण", "literal_translation": "second"}),
        json.dumps({"prompt": "तीसरा परीक्षण", "literal_translation": "third"}),
    ]) + "\n"
    manifest_path = tmp_path / "manifest.json"
    seeds_path = tmp_path / "seeds.jsonl"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    seeds_path.write_text(seeds, encoding="utf-8")
    return manifest_path, seeds_path


def run_builder(tmp_path, manifest_path, seeds_path):
    out = tmp_path / "out"
    review = tmp_path / "review.csv"
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/curate/build_hinglish_slices.py"),
         "--manifest", str(manifest_path), "--seeds", str(seeds_path),
         "--out", str(out), "--review-out", str(review), "--stub-generate"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return out, review


def test_builder_keeps_test_split_out_and_labels_rows(tmp_path):
    manifest_path, seeds_path = fixture_files(tmp_path)
    out, review = run_builder(tmp_path, manifest_path, seeds_path)
    cal = (out / "calibration" / "hs_eval.jsonl").read_text(encoding="utf-8")
    dev = (out / "dev" / "hs_eval.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in cal.splitlines() if line.strip()]
    rows += [json.loads(line) for line in dev.splitlines() if line.strip()]
    assert len(rows) == 2
    assert all(row["translation_review"] == "unreviewed machine translation" for row in rows)
    assert all(row["lang"] == "hinglish" for row in rows)
    assert all(row["metrics_eligible"] is False for row in rows)
    assert all(row["primary_asr"] is None for row in rows)
    lineage = json.loads((out / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["status"] if "status" in lineage else True
    assert lineage["generator"]["stub"] is True
    assert lineage["translation_review"] == "unreviewed machine translation"
    assert lineage["per_slice_counts"] == {"harm_calibration": 1, "harm_dev": 1}
    review_lines = review.read_text(encoding="utf-8").splitlines()
    assert review_lines and "hinglish" in review_lines[1]
    assert all(row["split"] != "test" for row in rows)
    assert all("aya_hin:2" not in row["item_id"] for row in rows)


def test_builder_refuses_unknown_member_shape(tmp_path):
    manifest_path, seeds_path = fixture_files(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = manifest["families"]["fam_cal"]["members"][0]
    member["source_id"] = "oops"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/curate/build_hinglish_slices.py"),
         "--manifest", str(manifest_path), "--seeds", str(seeds_path),
         "--out", str(tmp_path / "out"), "--stub-generate"],
        capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode != 0
