"""CLI tests for the family-split checker.

The checker is python -m src.curate.check_family_splits. These tests use a
synthetic manifest so they do not read staging prompts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "splits" / "family_splits_v2.json"
BATCH = ROOT / "data" / "staging" / "translations" / "batch01.csv"
AYA = ROOT / "data" / "staging" / "translations" / "aya_hinglish" / "aya_hinglish_sample50.csv"


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _manifest(tmp_path: Path) -> Path:
    payload = {
        "frozen": True,
        "test_access": "locked",
        "families": {
            "fam_dev": {
                "split": "dev",
                "members": [{"source_dataset": "xstest", "source_id": "1"}],
            },
            "fam_test": {
                "split": "test",
                "members": [{"source_dataset": "xstest", "source_id": "2"}],
            },
        },
    }
    path = tmp_path / "families.json"
    raw = (json.dumps(payload) + "\n").encode("utf-8")
    path.write_bytes(raw)
    (tmp_path / "families.json.sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n", encoding="utf-8")
    return path


def _run(manifest: Path, rows: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.curate.check_family_splits",
            "--manifest",
            str(manifest),
            str(rows),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cli_accepts_source_membership(tmp_path: Path):
    manifest = _manifest(tmp_path)
    rows = tmp_path / "ok.jsonl"
    _write_jsonl(rows, [{
        "source_dataset": "xstest",
        "source_id": "1",
        "family_id": "fam_dev",
        "split": "dev",
    }])
    result = _run(manifest, rows)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout


def test_cli_rejects_wrong_family(tmp_path: Path):
    manifest = _manifest(tmp_path)
    rows = tmp_path / "wrong.jsonl"
    _write_jsonl(rows, [{
        "source_dataset": "xstest",
        "source_id": "1",
        "family_id": "fam_test",
        "split": "test",
    }])
    result = _run(manifest, rows)
    assert result.returncode == 1
    assert "wrong family" in result.stdout


def test_cli_rejects_cross_split(tmp_path: Path):
    manifest = _manifest(tmp_path)
    rows = tmp_path / "cross.jsonl"
    _write_jsonl(rows, [
        {"source_dataset": "xstest", "source_id": "1", "family_id": "fam_dev", "split": "dev"},
        {"source_dataset": "xstest", "source_id": "1", "family_id": "fam_dev", "split": "test"},
    ])
    result = _run(manifest, rows)
    assert result.returncode == 1
    assert "crosses splits" in result.stdout


def test_one_split_per_family_within_staging_csvs():
    """Local invariant on filled rows: each family_id has exactly one split value."""
    paths = [p for p in (BATCH, AYA) if p.exists()]
    if not paths:
        pytest.skip("no staging translation CSVs")
    fam_splits: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        for r in _read(path):
            fam = (r.get("family_id") or "").strip()
            split = (r.get("split") or "").strip()
            if not fam:
                continue
            assert split, f"{path} row {r.get('id')} has family_id but empty split"
            fam_splits[fam].add(split)
    # If nothing filled yet (manifest absent), pass vacuously.
    for fam, splits in fam_splits.items():
        assert len(splits) == 1, f"family {fam} maps to multiple splits: {sorted(splits)}"
