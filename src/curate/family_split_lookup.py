"""Lookup family_id and split from DE B artifacts (no local family scheme).

Primary: data/splits/family_lookup.csv
  columns: source_dataset, source_id, item_id, family_id, split

Manifest (DE B owned, not vendored here): data/splits/family_splits_v2.json
Checker (DE B owned): python -m src.curate.check_family_splits [--manifest data/splits/family_splits_v2.json] FILE...

Source names accepted (aliases normalized to DE B source_dataset):
  xstest, harmbench, or_bench_toxic, or_bench_hard, or_bench_80k,
  aya / aya_redteaming / aya_redteaming_hi

source_id rules:
  OR-Bench: accept "row:<i>" or "<i>" (looked up as bare index string)
  Aya: "hin:<line>" / "eng:<line>" (0-based JSONL line). Staging alias
  aya_redteaming_hi + bare line becomes aya_redteaming + hin:<line>.

Splits: calibration / dev / test. Test is frozen; default sampling excludes it
unless allow_test is explicitly set (needed so all language variants exist for test).
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOOKUP_CSV = ROOT / "data" / "splits" / "family_lookup.csv"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "family_splits_v2.json"
VALID_SPLITS = frozenset({"calibration", "dev", "test"})
TRANSLATABLE_SPLITS = frozenset({"calibration", "dev"})

SOURCE_ALIASES = {
    "aya": "aya_redteaming",
    "aya_redteaming": "aya_redteaming",
    "aya_redteaming_hi": "aya_redteaming",
    "aya_redteaming_hin": "aya_redteaming",
    "aya_redteaming_eng": "aya_redteaming",
    "aya_redteaming_en": "aya_redteaming",
    "xstest": "xstest",
    "harmbench": "harmbench",
    "or_bench_toxic": "or_bench_toxic",
    "or_bench_hard": "or_bench_hard",
    "or_bench_80k": "or_bench_80k",
}


class ManifestMissingError(FileNotFoundError):
    pass


class FamilyAssignmentError(ValueError):
    pass


def normalize_source(source: str) -> str:
    return SOURCE_ALIASES.get((source or "").strip(), (source or "").strip())


def normalize_source_id(source: str, source_id: str) -> tuple[str, str]:
    """Return (source_dataset, source_id) in DE B lookup form."""
    raw_src = (source or "").strip()
    src = normalize_source(raw_src)
    sid = str(source_id).strip()

    if raw_src in {"aya_redteaming_hi", "aya_redteaming_hin"} and not sid.startswith(("hin:", "eng:")):
        sid = f"hin:{sid}"
    if raw_src in {"aya_redteaming_eng", "aya_redteaming_en"} and not sid.startswith(("hin:", "eng:")):
        sid = f"eng:{sid}"

    if src.startswith("or_bench") and sid.startswith("row:"):
        sid = sid.split(":", 1)[1]

    return src, sid


def member_key(source: str, source_id: str) -> str:
    src, sid = normalize_source_id(source, source_id)
    return f"{src}:{sid}"


@lru_cache(maxsize=4)
def _load_lookup(path_str: str) -> dict[tuple[str, str], tuple[str, str]]:
    p = Path(path_str)
    if not p.exists():
        return {}
    out: dict[tuple[str, str], tuple[str, str]] = {}
    with p.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[(r["source_dataset"], r["source_id"])] = (r["family_id"], r["split"])
    return out


def lookup_family(
    source: str,
    source_id: str,
    lookup_csv: Path | None = None,
) -> tuple[str, str]:
    """Return (family_id, split). Empty strings if lookup file missing or member unknown."""
    p = lookup_csv or LOOKUP_CSV
    idx = _load_lookup(str(p.resolve()) if p.exists() else str(p))
    if not idx:
        return "", ""
    src, sid = normalize_source_id(source, source_id)
    hit = idx.get((src, sid))
    return (hit[0], hit[1]) if hit else ("", "")


def require_assignment(
    source: str,
    source_id: str,
    lookup_csv: Path | None = None,
    *,
    allow_test: bool = False,
) -> tuple[str, str]:
    p = lookup_csv or LOOKUP_CSV
    if not p.exists():
        raise ManifestMissingError(
            f"DE B family_lookup.csv not found at {p}. "
            "Expected data/splits/family_lookup.csv from branch data/family-splits (v2)."
        )
    fam, split = lookup_family(source, source_id, p)
    if not fam or not split:
        src, sid = normalize_source_id(source, source_id)
        raise FamilyAssignmentError(f"No family assignment for {src}:{sid} in {p}")
    if split not in VALID_SPLITS:
        raise FamilyAssignmentError(f"Unknown split {split!r} for {fam}")
    if split == "test" and not allow_test:
        raise FamilyAssignmentError(
            f"Family {fam} is in frozen test split; refuse for default translation sampling "
            "(pass --include-test-families only when building full-family test language variants)."
        )
    return fam, split


def annotate_row(
    row: dict,
    *,
    source_field: str = "source",
    id_field: str = "source_id",
    normalize_inplace: bool = False,
) -> dict:
    out = dict(row)
    src_raw = out.get(source_field, "")
    sid_raw = out.get(id_field, "")
    src, sid = normalize_source_id(src_raw, sid_raw)
    if normalize_inplace:
        out[source_field] = src
        out[id_field] = sid
    fam, split = lookup_family(src_raw, sid_raw)
    out["family_id"] = fam
    out["split"] = split
    return out
