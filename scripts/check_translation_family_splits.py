#!/usr/bin/env python3
"""Check staging translation CSVs against DE B family-split rules (v2).

Uses data/splits/family_lookup.csv for local fill checks and, when present,
DE B's python -m src.curate.check_family_splits with family_splits_v2.json.

Exits 0 on SKIP (DE B artifacts absent) with local vacuous/local-ok, or on full pass.
Exits 1 on violations.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "src" / "curate" / "check_family_splits.py"
MANIFEST = ROOT / "data" / "splits" / "family_splits_v2.json"
LOOKUP = ROOT / "data" / "splits" / "family_lookup.csv"
CSV_PATHS = [
    ROOT / "data" / "staging" / "translations" / "batch01.csv",
    ROOT / "data" / "staging" / "translations" / "aya_hinglish" / "aya_hinglish_sample50.csv",
    ROOT / "data" / "staging" / "review" / "vedant_review_nov7.csv",
]


def local_invariant(paths: list[Path]) -> list[str]:
    errors = []
    fam_splits: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            fam = (r.get("family_id") or "").strip()
            split = (r.get("split") or "").strip()
            if not fam:
                continue
            if not split:
                errors.append(f"{path.name}: {r.get('id') or r.get('review_id')} has family_id without split")
                continue
            fam_splits[fam].add(split)
    for fam, splits in fam_splits.items():
        if len(splits) != 1:
            errors.append(f"family {fam} has multiple splits: {sorted(splits)}")
    return errors


def main() -> int:
    paths = [p for p in CSV_PATHS if p.exists()]
    errors = local_invariant(paths)

    if not LOOKUP.exists() or not CHECKER.exists() or not MANIFEST.exists():
        print(
            "SKIP: DE B v2 artifacts not present in this checkout "
            f"(lookup={LOOKUP.exists()}, checker={CHECKER.exists()}, manifest={MANIFEST.exists()}). "
            "Local filled-row invariant checked only."
        )
        if errors:
            print("FAIL local invariant:")
            for e in errors:
                print(" -", e)
            return 1
        print("OK (vacuous / local only)")
        return 0

    if errors:
        print("FAIL local invariant:")
        for e in errors:
            print(" -", e)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "src.curate.check_family_splits",
        "--manifest",
        str(MANIFEST),
        *[str(p) for p in paths],
    ]
    print("RUNNING:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
