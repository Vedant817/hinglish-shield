"""Prepare an unlabelled calibration/dev-only human review queue, without scoring."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.curate.family_splits import canon


def prepare(source: Path, manifest: Path, output: Path) -> dict:
    raw = manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = Path(str(manifest) + ".sha256").read_text().split()[0]
    if digest != expected:
        raise ValueError("Frozen manifest hash mismatch")
    manifest_data = json.loads(raw)
    if manifest_data.get("frozen") is not True:
        raise ValueError("Family manifest is not frozen")
    families = manifest_data["families"]
    members = {}
    for family_id, family in families.items():
        for member in family["members"]:
            key = canon(member["source_dataset"], member["source_id"])
            if key in members:
                raise ValueError("Duplicate canonical member")
            members[key] = (family_id, family["split"])
    selected = []
    seen = set()
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        for row in reader:
            key = canon(row.get("source") or row.get("source_dataset") or "", row.get("source_id") or "")
            if members.get(key) != (row.get("family_id"), row.get("split")):
                raise ValueError("Source membership disagrees with review family/split")
            family = families.get(row.get("family_id"))
            if family is None or row.get("split") != family["split"]:
                raise ValueError("Review membership missing or inconsistent")
            # Drop test before retaining any prompt or reviewer fields.
            if family["split"] == "test":
                continue
            if family["split"] not in {"calibration", "dev"}:
                raise ValueError("Unsupported review partition")
            rid = row["review_id"]
            if rid in seen:
                raise ValueError("Duplicate review ID")
            seen.add(rid)
            selected.append(row)
    if not selected:
        raise ValueError("No calibration/dev review rows")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing human review sheet")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    return {"manifest_sha256": digest, "sheet_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "rows": len(selected), "test_rows": 0, "status": "awaiting_human_review"}


if __name__ == "__main__":
    result = prepare(
        ROOT / "data/staging/review/vedant_review_nov7.csv",
        ROOT / "data/splits/family_splits_v2.json",
        ROOT / "data/staging/review/vedant_review_nov7_calibration_dev.csv",
    )
    print(json.dumps(result, indent=2))
