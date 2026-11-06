"""Lookup family_id and split from DE B's manifest.

DE B owns: data/splits/family_splits_v1.json and src/curate/check_family_splits.py
(branch data/family-splits). This module only reads that manifest.

Expected manifest shape (tolerant):
  {
    "seed": 42,
    "ratios": {"calibration": 0.2, "dev": 0.3, "test": 0.5},
    "families": {
      "<family_id>": {
        "split": "calibration"|"dev"|"test",
        "members": ["<source_dataset>:<source_id>", ...]
      }
    }
  }
or equivalently a top-level "member_index" map member -> {family_id, split}.

Splits are named calibration / dev / test. Test is frozen; do not sample it for new translations.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "family_splits_v1.json"
VALID_SPLITS = frozenset({"calibration", "dev", "test"})
TRANSLATABLE_SPLITS = frozenset({"calibration", "dev"})  # test frozen until Nov 11 pass


class ManifestMissingError(FileNotFoundError):
    pass


class FamilyAssignmentError(ValueError):
    pass


def _normalize_source(source: str) -> str:
    s = (source or "").strip()
    # Allow DE A staging aliases to match DE B source_dataset names when identical.
    aliases = {
        "aya_redteaming_hi": "aya_redteaming",
        "or_bench_toxic": "or_bench_toxic",
        "or_bench_hard": "or_bench_hard",
        "or_bench_80k": "or_bench_80k",
        "xstest": "xstest",
        "harmbench": "harmbench",
    }
    return aliases.get(s, s)


def member_key(source: str, source_id: str) -> str:
    return f"{_normalize_source(source)}:{str(source_id).strip()}"


def load_manifest(path: Path | None = None) -> dict[str, Any] | None:
    p = path or DEFAULT_MANIFEST
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_member_index(manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return member_key -> {family_id, split}."""
    if "member_index" in manifest and isinstance(manifest["member_index"], dict):
        out: dict[str, dict[str, str]] = {}
        for k, v in manifest["member_index"].items():
            if isinstance(v, dict):
                out[str(k)] = {
                    "family_id": str(v["family_id"]),
                    "split": str(v["split"]),
                }
            else:
                # rare: value is family_id only
                fam = str(v)
                split = str(manifest.get("families", {}).get(fam, {}).get("split", ""))
                out[str(k)] = {"family_id": fam, "split": split}
        return out

    out = {}
    families = manifest.get("families") or {}
    for fam_id, meta in families.items():
        split = str((meta or {}).get("split") or "")
        for m in (meta or {}).get("members") or []:
            out[str(m)] = {"family_id": str(fam_id), "split": split}
    return out


@lru_cache(maxsize=4)
def _cached_index(manifest_path: str) -> dict[str, dict[str, str]] | None:
    p = Path(manifest_path)
    man = load_manifest(p)
    if man is None:
        return None
    return build_member_index(man)


def lookup_family(
    source: str,
    source_id: str,
    manifest_path: Path | None = None,
) -> tuple[str, str]:
    """Return (family_id, split). Empty strings if manifest missing or member unknown."""
    p = manifest_path or DEFAULT_MANIFEST
    idx = _cached_index(str(p.resolve()) if p.exists() else str(p))
    if idx is None:
        return "", ""
    hit = idx.get(member_key(source, source_id))
    if not hit:
        # try raw source without alias normalization already applied in member_key
        hit = idx.get(f"{source}:{source_id}")
    if not hit:
        return "", ""
    return hit.get("family_id", ""), hit.get("split", "")


def require_assignment(
    source: str,
    source_id: str,
    manifest_path: Path | None = None,
    *,
    allow_test: bool = False,
) -> tuple[str, str]:
    """Raise if manifest missing or row unassigned. Blocks test split unless allow_test."""
    p = manifest_path or DEFAULT_MANIFEST
    if not p.exists():
        raise ManifestMissingError(
            f"DE B split manifest not found at {p}. "
            "Expected data/splits/family_splits_v1.json from branch data/family-splits."
        )
    fam, split = lookup_family(source, source_id, p)
    if not fam or not split:
        raise FamilyAssignmentError(
            f"No family assignment for {member_key(source, source_id)} in {p}"
        )
    if split not in VALID_SPLITS:
        raise FamilyAssignmentError(f"Unknown split {split!r} for {fam}")
    if split == "test" and not allow_test:
        raise FamilyAssignmentError(
            f"Family {fam} is in frozen test split; refuse for translation sampling "
            "(test read only by Nov 11 pass)."
        )
    return fam, split


def annotate_row(row: dict, *, source_field: str = "source", id_field: str = "source_id") -> dict:
    """Fill family_id and split columns via lookup; leave blank if manifest absent."""
    out = dict(row)
    fam, split = lookup_family(row.get(source_field, ""), row.get(id_field, ""))
    out["family_id"] = fam
    out["split"] = split
    return out
