"""Export explicitly accepted review rows as lineage-backed language variants.

The default sheet is the calibration/dev queue. Blank verdicts produce a
coverage report and no eligible prompt files. This script does not score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curate.reviewed_variants import (  # noqa: E402
    ACCEPTANCE_POLICY,
    ReviewedVariantError,
    export_reviewed_variants,
)

DEFAULT_SHEET = ROOT / "data" / "staging" / "review" / "vedant_review_nov7_calibration_dev.csv"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "family_splits_v2.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Freeze reviewed translation variants for calibration and dev. "
        "Requires an explicit acceptance policy. Refuses test rows."
    ))
    parser.add_argument("--review-sheet", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--acceptance-policy", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.acceptance_policy != ACCEPTANCE_POLICY:
        print("REFUSING REVIEWED VARIANTS: acceptance policy is required", file=sys.stderr)
        return 2
    try:
        report = export_reviewed_variants(
            review_sheet=args.review_sheet,
            manifest=args.manifest,
            out_dir=args.out,
            acceptance_policy=args.acceptance_policy,
        )
    except ReviewedVariantError as exc:
        print(f"REFUSING REVIEWED VARIANTS: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "corpus_complete": report["corpus_complete"],
        "eligible_behavior_count": report["eligible_behavior_count"],
        "eligible_variant_count": report["eligible_variant_count"],
        "prompt_files_written": report["prompt_files_written"],
        "sheet_coverage_complete": report["sheet_coverage_complete"],
        "status": report["status"],
    }, sort_keys=True))
    return 0 if report["prompt_files_written"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
