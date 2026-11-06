#!/usr/bin/env python3
"""CLI stub for per-language baseline eval (AuditKIT + judges).

Does NOT invent metric numbers. Does NOT call models or judges until datasets
and RUNTIME-VERIFIED configs exist. Does NOT run the locked test split unless
partitions are frozen and the single-shot flags are set.

Configs:
  - configs/eval/baseline.yaml      -> Nov 8 preliminary H1 on split=dev only
  - configs/eval/test_locked.yaml   -> Nov 11 H1 primary on split=test, once

StrongREJECT: use maintained ``dsbowen/strong_reject`` (not deprecated
``alexandrasouly/strongreject``).

ASR / CI: see src/eval/stats.py and docs/EVAL_PLAN.md (match prereg).
"""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

try:
    import yaml
except ImportError:  # pragma: no cover - smoke installs PyYAML
    yaml = None  # type: ignore


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required; run make install or use the project venv")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _dataset_missing(_cfg: dict[str, Any]) -> list[str]:
    """Best-effort check for curated dataset paths."""
    missing: list[str] = []
    candidates = [
        ROOT / "data" / "curated" / "hs_harm",
        ROOT / "data" / "curated" / "hs_benign",
    ]
    for path in candidates:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
    return missing


def _git_full_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _prior_test_run_ids(runs_dir: Path) -> list[str]:
    """Find existing non-fixture metrics.json rows with split == test."""
    found: list[str] = []
    if not runs_dir.is_dir():
        return found
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        if metrics_path.parent.name.startswith("_"):
            continue
        try:
            data = json.loads(metrics_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("artifact_kind") == "fixture":
            continue
        if data.get("split") == "test":
            found.append(str(metrics_path.relative_to(ROOT)))
    return found


def _validate_test_single_shot(
    cfg: dict[str, Any],
    *,
    split: str,
    i_understand: bool,
    config_path: Path,
) -> list[str]:
    """Guards for the locked Nov 11 test pass. Returns problem strings."""
    problems: list[str] = []
    cfg_splits = list(cfg.get("splits") or [])
    is_test_config = (
        "test" in cfg_splits
        or cfg.get("role") == "h1_primary_single_shot"
        or config_path.name == "test_locked.yaml"
    )

    if split == "test" or is_test_config:
        if split != "test":
            problems.append(
                "test_locked path requires --split test "
                f"(got --split {split!r})"
            )
        if not i_understand:
            problems.append(
                "test split requires --i-understand-single-shot "
                "(Build Plan: evaluate frozen test once; never for selection)"
            )
        datasets = cfg.get("datasets") or {}
        if not re.fullmatch(r"[0-9a-f]{64}", str(datasets.get("dataset_hash") or "")):
            problems.append(
                "test split requires a full SHA-256 datasets.dataset_hash "
                "(partitions not frozen)"
            )
        manifest = datasets.get("split_manifest")
        if not manifest:
            problems.append(
                "test split requires datasets.split_manifest "
                "(frozen family-split membership path)"
            )
        else:
            manifest_path = ROOT / str(manifest)
            if not manifest_path.exists():
                problems.append(
                    f"test split manifest missing: {manifest}"
                )
            else:
                try:
                    raw = manifest_path.read_bytes()
                    membership = json.loads(raw)
                    sidecar = Path(str(manifest_path) + ".sha256")
                    if (membership.get("frozen") is not True or not membership.get("families")
                            or sidecar.read_text().split()[0] != hashlib.sha256(raw).hexdigest()):
                        problems.append("test membership must be frozen and checksum-verified")
                    exposure_path = ROOT / "data/splits/development_exposures.json"
                    if not exposure_path.is_file():
                        problems.append("required test exposure ledger is missing")
                    else:
                        exposures = json.loads(exposure_path.read_text(encoding="utf-8"))
                        if not isinstance(exposures["entries"], list):
                            raise ValueError("Invalid exposure entries")
                        for entry in exposures["entries"]:
                            source = entry["source_dataset"]
                            source_id = entry["source_id"]
                            if not all(isinstance(value, str) and value for value in (source, source_id)):
                                raise ValueError("Invalid exposure identity")
                            family = membership["families"].get(entry["family_id"])
                            member_in_test = any(
                                candidate.get("split") == "test" and any(
                                    member.get("source_dataset") == source and str(member.get("source_id")) == source_id
                                    for member in candidate.get("members", [])
                                ) for candidate in membership["families"].values()
                            )
                            if (family and family.get("split") == "test") or member_in_test:
                                problems.append("test holdout contains a development-exposed family: " + entry["family_id"])
                except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                    problems.append("test membership or exposure ledger is unreadable or invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("model_revision") or "")):
            problems.append("test split requires a pinned full model_revision")
        prior = _prior_test_run_ids(ROOT / "runs")
        if prior:
            problems.append(
                "prior test run record exists; refusing second test pass: "
                + ", ".join(prior)
            )
    elif split not in ("dev", "calibration"):
        problems.append(
            f"unsupported --split {split!r}; use dev (Nov 8 preliminary) "
            "or test with single-shot flags"
        )

    # baseline.yaml must stay off test
    if config_path.name == "baseline.yaml" and split == "test":
        problems.append(
            "configs/eval/baseline.yaml is dev-only (Nov 8 preliminary H1). "
            "Use configs/eval/test_locked.yaml for the Nov 11 locked test pass."
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hinglish Shield baseline eval stub. Refuses to invent scores. "
            "Exits non-zero when frozen datasets/config are missing. "
            "Test split is single-shot only via configs/eval/test_locked.yaml."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "eval" / "baseline.yaml",
        help="Eval YAML (default: configs/eval/baseline.yaml for Nov 8 dev)",
    )
    parser.add_argument(
        "--judges-config",
        type=Path,
        default=ROOT / "configs" / "eval" / "judges.yaml",
        help="Judges YAML (default: configs/eval/judges.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Partition to evaluate: dev (default, Nov 8) or test (locked)",
    )
    parser.add_argument(
        "--i-understand-single-shot",
        action="store_true",
        help=(
            "Required with --split test. Confirms partitions are frozen and "
            "this is the one allowed test evaluation (not for selection)."
        ),
    )
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="Only verify config/dataset/guards; never score.",
    )
    args = parser.parse_args(argv)

    problems: list[str] = []
    if not args.config.exists():
        problems.append(f"missing config: {args.config}")
    if not args.judges_config.exists():
        problems.append(f"missing judges config: {args.judges_config}")

    cfg: dict[str, Any] = {}
    if args.config.exists() and yaml is not None:
        try:
            cfg = _load_yaml(args.config)
        except (OSError, ValueError, yaml.YAMLError) as exc:  # type: ignore[union-attr]
            problems.append(f"unreadable config: {exc}")
    elif args.config.exists() and yaml is None:
        problems.append("PyYAML not installed; cannot load config")

    if cfg:
        problems.extend(
            _validate_test_single_shot(
                cfg,
                split=args.split,
                i_understand=args.i_understand_single_shot,
                config_path=args.config,
            )
        )
        problems.extend(
            f"missing dataset path: {p}" for p in _dataset_missing(cfg)
        )

    if problems:
        print("REFUSING TO RUN BASELINE (no invented numbers).", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print(
            "Next: freeze HS-Harm / HS-Benign under data/curated/, pin "
            "model_revision + dataset_hash, RUNTIME-VERIFY WildGuard, and for "
            "test use configs/eval/test_locked.yaml with "
            "--split test --i-understand-single-shot after partitions freeze.",
            file=sys.stderr,
        )
        return 2

    full_commit = _git_full_commit()
    seed = cfg.get("seed")
    print(
        "OK dry-check inputs present."
        if args.dry_check
        else "Inputs present; scoring still gated."
    )
    print(f"  config={args.config}")
    print(f"  split={args.split}")
    print(f"  seed={seed}")
    print(f"  full_commit={full_commit}")
    print(f"  dataset_hash={((cfg.get('datasets') or {}).get('dataset_hash'))}")
    if args.split == "test":
        print(
            "  NOTE: test is single-shot / never for selection. "
            "Log run_id, full_commit, seed into runs/<id>/metrics.json and LOG.md."
        )

    if args.dry_check:
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
