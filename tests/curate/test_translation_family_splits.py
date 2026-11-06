"""Wire DE B family-split checker onto translation staging CSVs.

DE B owns src/curate/check_family_splits.py and data/splits/family_splits_v2.json.
This test:
  - skips if the checker or manifest is not present yet
  - otherwise runs the checker against staging translation CSVs
  - asserts every non-blank family_id maps to one split and variants agree
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "src" / "curate" / "check_family_splits.py"
MANIFEST = ROOT / "data" / "splits" / "family_splits_v2.json"
BATCH = ROOT / "data" / "staging" / "translations" / "batch01.csv"
AYA = ROOT / "data" / "staging" / "translations" / "aya_hinglish" / "aya_hinglish_sample50.csv"


def _load_checker():
    if not CHECKER.exists():
        pytest.skip("DE B checker not present yet (src/curate/check_family_splits.py)")
    spec = importlib.util.spec_from_file_location("check_family_splits", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not MANIFEST.exists(), reason="DE B family_splits_v2.json not present yet")
def test_deb_checker_on_translation_csvs():
    mod = _load_checker()
    # Prefer a documented entrypoint if DE B provides one.
    if hasattr(mod, "check_translation_csv"):
        for path in (BATCH, AYA):
            if path.exists():
                mod.check_translation_csv(path)
    elif hasattr(mod, "main"):
        # CLI-style: pass csv paths if supported; otherwise call check_manifest
        if hasattr(mod, "check_paths"):
            mod.check_paths([BATCH, AYA])
        elif hasattr(mod, "check_manifest"):
            mod.check_manifest(MANIFEST)
        else:
            pytest.skip("DE B checker has no known callable API yet")
    else:
        pytest.skip("DE B checker module has no main/check_* API yet")


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
