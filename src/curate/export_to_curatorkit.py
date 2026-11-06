#!/usr/bin/env python3
"""Map staging translation CSVs to HS eval rows + CuratorKIT prompt_only JSONL (no upload).

Canonical family_id and split come from the checksum-verified frozen manifest.
The batch id is a staging key only.
It is not the semantic family.

Rows without an explicit accept verdict stay unreviewed and are not
scoring-eligible. This module does not invent review labels. main() is
staging preparation: it writes the rows it was given, including unreviewed
ones, and it does not mark them accepted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import uuid
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curate.family_splits import canon

LABEL = {"harmful": "harm", "benign": "benign"}
_ACCEPT = {"accept", "accepted"}
_REVISE = {"revise", "revised"}
_REJECT = {"reject", "rejected"}


@lru_cache(maxsize=1)
def _manifest_members(raw: bytes) -> dict:
    manifest = json.loads(raw)
    if manifest.get("frozen") is not True:
        raise ValueError("Family manifest is not frozen")
    members = {}
    for family_id, family in manifest["families"].items():
        for member in family["members"]:
            key = canon(member["source_dataset"], member["source_id"])
            if key in members:
                raise ValueError("Duplicate canonical member in manifest")
            members[key] = (family_id, family["split"])
    return members


def _lookup_verified(source: str, source_id: str) -> tuple[str, str] | None:
    path = ROOT / "data/splits/family_splits_v2.json"
    raw = path.read_bytes()
    expected = Path(str(path) + ".sha256").read_text().split()[0]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Frozen family manifest checksum mismatch")
    return _manifest_members(raw).get(canon(source, source_id))


def _review_status(row: dict) -> str:
    """Map an explicit human label. Blank stays unreviewed. Never default to accept."""
    raw = row.get("vedant_verdict")
    if raw is None or str(raw).strip() == "":
        raw = row.get("review_status")
    text = "" if raw is None else str(raw).strip().lower()
    if text == "" or text == "unreviewed":
        return "unreviewed"
    if text in _ACCEPT:
        return "accept"
    if text in _REVISE:
        return "revise"
    if text in _REJECT:
        return "reject"
    raise ValueError(
        f"unknown review label {text!r}; refusing to treat it as accepted"
    )


def resolve_membership(row: dict) -> tuple[str, str, str, str]:
    """Return (source_dataset, source_id, family_id, split) from the frozen manifest.

    The row must already carry family_id and split, and both must match the
    lookup. Missing membership, a missing id, or a mismatch raises ValueError.
    """
    source = str(row.get("source") or row.get("source_dataset") or "").strip()
    source_id = str(row.get("source_id") if row.get("source_id") is not None else "").strip()
    if not source or source_id == "":
        raise ValueError("missing source or source_id for family membership")
    hit = _lookup_verified(source, source_id)
    src, sid = canon(source, source_id)
    if hit is None:
        raise ValueError(f"missing membership for {src}:{sid}")
    family_id, split = hit
    given_family = str(row.get("family_id") or "").strip()
    given_split = str(row.get("split") or "").strip()
    if not given_family or not given_split:
        raise ValueError("missing family_id or split")
    if given_family != family_id:
        raise ValueError(
            f"inconsistent family_id {given_family!r} for {src}:{sid}; manifest has {family_id}"
        )
    if given_split != split:
        raise ValueError(
            f"inconsistent split {given_split!r} for family {family_id}; manifest has {split}"
        )
    return src, sid, family_id, split


def map_batch_row(r: dict, lang: str, text: str) -> dict:
    source_dataset, source_id, family_id, split = resolve_membership(r)
    review_status = _review_status(r)
    scoring_eligible = review_status == "accept"
    batch_id = str(r.get("id") or "").strip()
    if lang not in {"en", "hi", "hinglish"}:
        raise ValueError("Unsupported language variant")
    item_id = f"{batch_id}:{lang}" if batch_id else f"member:{source_dataset}:{source_id}:{lang}"
    return {
        "item_id": item_id,
        "prompt": text,
        "hs_label": LABEL.get(r.get("label", ""), r.get("label", "")),
        "source_dataset": source_dataset,
        "source_id": source_id,
        "category": r.get("category", ""),
        "lang": lang,
        "split": split,
        "family_id": family_id,
        # variant_of names a canonical EN item, not the semantic family. The
        # canonical EN item relationship is not resolved here, so do not trust a
        # supplied link. Family membership remains in its own verified field.
        "variant_of": None,
        "curatorkit_id": None,
        "review_status": review_status,
        "scoring_eligible": scoring_eligible,
        "metadata": {
            "qc_notes": r.get("qc_notes", ""),
            "translation_batch_id": batch_id,
            "label_raw": r.get("label", ""),
            "family_id": family_id,
            "split": split,
            "review_status": review_status,
            "scoring_eligible": scoring_eligible,
        },
    }


def to_curatorkit(hs: dict) -> dict:
    meta = dict(hs.get("metadata") or {})
    meta.update({
        "item_id": hs["item_id"],
        "hs_label": hs["hs_label"],
        "source_dataset": hs["source_dataset"],
        "source_id": hs["source_id"],
        "category": hs["category"],
        "lang": hs["lang"],
        "split": hs["split"],
        "family_id": hs["family_id"],
        "variant_of": hs["variant_of"],
        "review_status": hs["review_status"],
        "scoring_eligible": hs["scoring_eligible"],
    })
    return {
        "id": str(uuid.uuid4()),
        "source_uri": f"hs://{hs['source_dataset']}#{hs['source_id']}",
        "instruction": hs["prompt"],
        "input": "",
        "output": "",
        "task_type": "prompt_only",
        "metadata": meta,
        "provenance_chain": [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps LF on Windows. Do not reopen this in text mode
    # that translates newlines.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_rows(rows: list[dict]) -> list[dict]:
    """Expand staging rows to one eval row per non-empty language cell."""
    hs_rows: list[dict] = []
    seen = set()
    for row in rows:
        present = [
            (lang, row.get(column))
            for lang, column in (("en", "english"), ("hi", "hindi"), ("hinglish", "hinglish"))
            if row.get(column)
        ]
        if not present:
            raise ValueError("Staging row has no language text; refusing silent row loss")
        for lang, text in present:
            mapped = map_batch_row(row, lang, text)
            if mapped["item_id"] in seen:
                raise ValueError("Duplicate exported item_id")
            seen.add(mapped["item_id"])
            hs_rows.append(mapped)
    return hs_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-hs", required=True)
    ap.add_argument("--out-ck", required=True)
    args = ap.parse_args()
    with Path(args.csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hs_rows = export_rows(rows)
    _write_jsonl(Path(args.out_hs), hs_rows)
    _write_jsonl(Path(args.out_ck), [to_curatorkit(h) for h in hs_rows])
    print({"hs_rows": len(hs_rows), "out_hs": args.out_hs, "out_ck": args.out_ck})


if __name__ == "__main__":
    main()
