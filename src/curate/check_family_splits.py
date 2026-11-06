#!/usr/bin/env python3
"""Check that no semantic family crosses splits (prereg research/baseline-prereg@8b6af17).

Usage:
  python -m src.curate.check_family_splits [--manifest data/splits/family_splits_v2.json] FILE [FILE ...]

FILE is JSONL or CSV. A row is resolved by (family_id) if present, else by (source, source_id)
or (source_dataset, source_id) via the manifest lookup. If the row has a `split` column it must
match the manifest. Exit 1 if: manifest SHA-256 mismatches its .sha256 sidecar; the manifest
puts a family in >1 split; any row lacks a family; any row's split disagrees with the manifest;
any family appears under >1 split across the given files. Prints per-file coverage by split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.curate.family_splits import canon  # noqa: E402

LEGACY = {"calibration": "calibration", "dev": "dev", "test": "test"}


def rows_of(path: Path):
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            yield from csv.DictReader(f)
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data/splits/family_splits_v2.json"))
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    mpath = Path(a.manifest)
    errors = []
    side = Path(str(mpath) + ".sha256")
    digest = hashlib.sha256(mpath.read_bytes()).hexdigest()
    if not side.exists() or side.read_text().split()[0] != digest:
        errors.append(f"manifest sha256 {digest} does not match sidecar {side}")
    man = json.loads(mpath.read_text(encoding="utf-8"))
    fam_split, key_fam = {}, {}
    for fid, f in man["families"].items():
        fam_split[fid] = f["split"]
        for m in f["members"]:
            k = (m["source_dataset"], m["source_id"])
            if k in key_fam and key_fam[k] != fid:
                errors.append(f"member {k} in two families")
            key_fam[k] = fid
    print(f"manifest {mpath.name} sha256={digest} families={len(fam_split)} members={len(key_fam)} "
          f"frozen={man.get('frozen')} test_access={man.get('test_access')}")
    seen = defaultdict(set)
    for fp in a.files:
        p = Path(fp)
        cov, bad = Counter(), 0
        for i, r in enumerate(rows_of(p)):
            fid = r.get("family_id") or None
            if not fid:
                src = r.get("source") or r.get("source_dataset")
                sid = r.get("source_id")
                if src is not None and sid is not None:
                    fid = key_fam.get(canon(src, sid))
            if not fid or fid not in fam_split:
                cov["UNMAPPED"] += 1
                bad += 1
                if bad <= 5:
                    errors.append(f"{p}:{i}: no family for {r.get('source') or r.get('source_dataset')}:{r.get('source_id')}")
                continue
            s = fam_split[fid]
            cov[s] += 1
            seen[fid].add(s)
            rs = r.get("split")
            if rs and rs != s:
                errors.append(f"{p}:{i}: split {rs} != manifest {s} for {fid}")
        if bad > 5:
            errors.append(f"{p}: {bad} unmapped rows total")
        print(f"{p}: " + " ".join(f"{k}={cov[k]}" for k in ("calibration", "dev", "test", "UNMAPPED")))
    for fid, ss in seen.items():
        if len(ss) > 1:
            errors.append(f"family {fid} crosses splits {sorted(ss)}")
    for e in errors[:50]:
        print("FAIL:", e)
    print("RESULT:", "FAIL" if errors else "PASS", f"({len(errors)} errors)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
