#!/usr/bin/env python3
"""CLI stub for per-language baseline eval (AuditKIT + judges).

Does NOT invent metric numbers. Does NOT call models or judges until datasets
and RUNTIME-VERIFIED configs exist.

Required frozen inputs (see configs/eval/baseline.yaml):
  - model_id + model_revision
  - dataset paths / dataset_hash for HS-Harm and HS-Benign per language
  - judges config (WildGuard primary, LLM secondary)
  - seed, query_budget, attack suite id
  - output runs/<run_id>/metrics.json conforming to schemas/metrics.schema.json

StrongREJECT: use maintained ``dsbowen/strong_reject`` (not deprecated
``alexandrasouly/strongreject``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _dataset_missing(cfg_path: Path) -> list[str]:
    """Best-effort check for curated dataset paths mentioned in YAML text."""
    missing: list[str] = []
    if not cfg_path.exists():
        return [str(cfg_path)]
    candidates = [
        ROOT / "data" / "curated" / "hs_harm",
        ROOT / "data" / "curated" / "hs_benign",
    ]
    for path in candidates:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hinglish Shield baseline eval stub. Refuses to invent scores. "
            "Exits non-zero when frozen datasets/config are missing."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval" / "baseline.yaml",
        help="Frozen baseline YAML (default: configs/eval/baseline.yaml)",
    )
    parser.add_argument(
        "--judges-config",
        type=Path,
        default=ROOT / "configs" / "eval" / "judges.yaml",
        help="Judges YAML (default: configs/eval/judges.yaml)",
    )
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="Only verify config/dataset presence; never score.",
    )
    args = parser.parse_args(argv)

    problems: list[str] = []
    if not args.config.exists():
        problems.append(f"missing config: {args.config}")
    if not args.judges_config.exists():
        problems.append(f"missing judges config: {args.judges_config}")
    problems.extend(
        f"missing dataset path: {p}" for p in _dataset_missing(args.config)
    )

    if problems:
        print("REFUSING TO RUN BASELINE (no invented numbers).", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "Next: freeze HS-Harm / HS-Benign per language under data/curated/, "
            "pin model_revision + dataset_hash in configs/eval/baseline.yaml, "
            "RUNTIME-VERIFY WildGuard via AuditKIT GuardJudge, then re-run.",
            file=sys.stderr,
        )
        return 2

    if args.dry_check:
        print("OK: config and dataset paths present (dry-check only; no scores).")
        return 0

    print(
        "Baseline harness is still a stub. Datasets/config look present, but "
        "scoring is gated on RUNTIME-VERIFIED judges. Not writing metrics.json "
        "with placeholder scores.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
