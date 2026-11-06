#!/usr/bin/env python3
"""Check staging translation CSVs against DE B family-split rules.

Exits 0 if:
  - DE B checker/manifest not present yet (prints SKIP), or
  - DE B checker passes and local one-split-per-family invariant holds for filled rows.
Exits 1 on violations.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "src" / "curate" / "check_family_splits.py"
MANIFEST = ROOT / "data" / "splits" / "family_splits_v1.json"
CSV_PATHS = [
    ROOT / "data" / "staging" / "translations" / "batch01.csv",
    ROOT / "data" / "staging" / "translations" / "aya_hinglish" / "aya_hinglish_sample50.csv",
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
                errors.append(f"{path.name}: {r.get('id')} has family_id without split")
                continue
            fam_splits[fam].add(split)
    for fam, splits in fam_splits.items():
        if len(splits) != 1:
            errors.append(f"family {fam} has multiple splits: {sorted(splits)}")
    return errors


def main() -> int:
    errors = local_invariant(CSV_PATHS)
    if not MANIFEST.exists() or not CHECKER.exists():
        print(
            "SKIP: DE B manifest/checker not present yet "
            f"(manifest={MANIFEST.exists()}, checker={CHECKER.exists()}). "
            "Local filled-row invariant checked."
        )
        if errors:
            print("FAIL local invariant:")
            for e in errors:
                print(" -", e)
            return 1
        print("OK (vacuous / local only)")
        return 0

    spec = importlib.util.spec_from_file_location("check_family_splits", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    if hasattr(mod, "check_translation_csv"):
        for path in CSV_PATHS:
            if path.exists():
                mod.check_translation_csv(path)
    elif hasattr(mod, "check_paths"):
        mod.check_paths([p for p in CSV_PATHS if p.exists()])
    elif hasattr(mod, "check_manifest"):
        mod.check_manifest(MANIFEST)
    elif hasattr(mod, "main"):
        # best-effort: run main with no args if it self-discovers
        rc = mod.main([]) if "argv" in (mod.main.__code__.co_varnames) else mod.main()
        if isinstance(rc, int) and rc != 0:
            return rc
    else:
        print("WARN: DE B checker present but no known API; local invariant only")

    if errors:
        print("FAIL local invariant:")
        for e in errors:
            print(" -", e)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
