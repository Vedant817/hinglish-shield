#!/usr/bin/env python3
"""Map staging translation CSVs to HS eval rows + CuratorKIT prompt_only JSONL (no upload)."""
from __future__ import annotations
import argparse, csv, json, uuid
from pathlib import Path

LABEL = {"harmful": "harm", "benign": "benign"}

def map_batch_row(r: dict, lang: str, text: str) -> dict:
    family = r["id"]
    return {
        "item_id": f"{family}:{lang}",
        "prompt": text,
        "hs_label": LABEL.get(r.get("label", ""), r.get("label", "")),
        "source_dataset": r.get("source", ""),
        "source_id": r.get("source_id", ""),
        "category": r.get("category", ""),
        "lang": lang,
        "split": "staging_review",
        "variant_of": family,
        "curatorkit_id": None,
        "metadata": {
            "qc_notes": r.get("qc_notes", ""),
            "translation_batch_id": r.get("id", ""),
            "label_raw": r.get("label", ""),
        },
    }

def to_curatorkit(hs: dict) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "source_uri": f"hs://{hs['source_dataset']}#{hs['source_id']}",
        "instruction": hs["prompt"],
        "input": "",
        "output": "",
        "task_type": "prompt_only",
        "metadata": {
            "item_id": hs["item_id"],
            "hs_label": hs["hs_label"],
            "source_dataset": hs["source_dataset"],
            "source_id": hs["source_id"],
            "category": hs["category"],
            "lang": hs["lang"],
            "split": hs["split"],
            "variant_of": hs["variant_of"],
            **hs.get("metadata", {}),
        },
        "provenance_chain": [],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-hs", required=True)
    ap.add_argument("--out-ck", required=True)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    hs_rows = []
    for r in rows:
        if r.get("english"):
            hs_rows.append(map_batch_row(r, "en", r["english"]))
        if r.get("hindi"):
            hs_rows.append(map_batch_row(r, "hi", r["hindi"]))
        if r.get("hinglish"):
            hs_rows.append(map_batch_row(r, "hinglish", r["hinglish"]))
    Path(args.out_hs).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_hs, "w", encoding="utf-8") as f:
        for h in hs_rows:
            f.write(json.dumps(h, ensure_ascii=False) + "
")
    with open(args.out_ck, "w", encoding="utf-8") as f:
        for h in hs_rows:
            f.write(json.dumps(to_curatorkit(h), ensure_ascii=False) + "
")
    print({"hs_rows": len(hs_rows), "out_hs": args.out_hs, "out_ck": args.out_ck})

if __name__ == "__main__":
    main()
