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
import os
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    # Support the documented direct-script CLI as well as python -m.
    sys.path.insert(0, str(ROOT))

from src.curate.family_splits import canon

IST = timezone(timedelta(hours=5, minutes=30))
TEST_EMBARGO_END = datetime(2026, 11, 11, tzinfo=IST)
ATTEMPT_FILENAME = "_locked_test_attempt.json"
FROZEN_MANIFEST = Path("data/splits/family_splits_v2.json")
FROZEN_MANIFEST_SHA256 = "334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _frozen_repair(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Verify a pinned real repair artifact using CPU evidence only."""
    from src.repair.freeze import verify_measured
    from src.repair.errors import RepairError

    pin = cfg.get("frozen_repair")
    if not isinstance(pin, dict) or not isinstance(pin.get("manifest"), str) or not pin["manifest"]:
        raise ValueError("test requires frozen_repair.manifest")
    digest = pin.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("test requires frozen_repair.sha256")
    try:
        raw = (ROOT / pin["manifest"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("frozen repair byte hash mismatch")
        arm = json.loads(raw)
        if not isinstance(arm, dict):
            raise ValueError("frozen repair must be an object")
        producer = arm.get("producer")
        if (cfg.get("model_id") != "CohereLabs/tiny-aya-global"
                or arm.get("artifact_kind") != "run" or arm.get("not_tiny_aya") is not False
                or arm.get("model_id") != cfg.get("model_id")
                or not isinstance(producer, dict)
                or producer.get("model_revision") != cfg.get("model_revision")):
            raise ValueError("frozen repair is not a matching real-model arm")
        verify_measured(arm)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, RepairError) as exc:
        raise ValueError("frozen repair verification failed: " + str(exc)) from exc
    return arm, digest

try:
    import yaml
except ImportError:  # pragma: no cover - smoke installs PyYAML
    yaml = None  # type: ignore


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required; run make install or use the project venv")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    """Find test rows or unreadable history; either blocks a fresh attempt."""
    found: list[str] = []
    if not runs_dir.is_dir():
        return found
    for metrics_path in sorted(runs_dir.rglob("metrics.json")):
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("metrics must be an object")
        except (OSError, UnicodeError, ValueError):
            found.append(str(metrics_path.relative_to(ROOT)) + " (unreadable history)")
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
        if _utc_now() < TEST_EMBARGO_END:
            problems.append("test embargo: no attempt before 2026-11-11 00:00:00 IST")
        if os.path.lexists(ROOT / "runs" / ATTEMPT_FILENAME):
            problems.append("locked test attempt already reserved; refusing another attempt")
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
            if manifest_path.resolve() != (ROOT / FROZEN_MANIFEST).resolve():
                problems.append("test requires the canonical frozen family manifest")
            if not manifest_path.exists():
                problems.append(
                    f"test split manifest missing: {manifest}"
                )
            else:
                try:
                    raw = manifest_path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != FROZEN_MANIFEST_SHA256:
                        problems.append("test frozen family manifest differs from the committed pin")
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
                            identity = canon(source.casefold(), source_id)
                            family = membership["families"].get(entry["family_id"])
                            member_in_test = any(
                                candidate.get("split") == "test" and any(
                                    canon(member["source_dataset"].casefold(), str(member["source_id"])) == identity
                                    for member in candidate.get("members", [])
                                ) for candidate in membership["families"].values()
                            )
                            if (family and family.get("split") == "test") or member_in_test:
                                problems.append("test holdout contains a development-exposed family: " + entry["family_id"])
                except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
                    problems.append("test membership or exposure ledger is unreadable or invalid")
        if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("model_revision") or "")):
            problems.append("test split requires a pinned full model_revision")
        try:
            _frozen_repair(cfg)
        except ValueError as exc:
            problems.append(str(exc))
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


def _execute_locked_test(
    config_path: Path,
    *,
    run_id: str,
    i_understand: bool,
    generate: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any],
) -> Any:
    """Only execution boundary for a future locked scorer, not the stub CLI.

    No prompt reads/model calls may precede this boundary. The callback gets
    the admitted config and frozen arm snapshots. A crash, callback failure,
    or incomplete reservation write consumes the attempt; never delete it to
    retry. One fixed path in this checkout protects against concurrent starts.
    """
    config_bytes = config_path.read_bytes()
    cfg = yaml.safe_load(config_bytes.decode("utf-8")) if yaml is not None else None
    if not isinstance(cfg, dict):
        raise ValueError("locked config must be a mapping")
    if cfg.get("splits") != ["test"] or cfg.get("method_arms") != ["base", "refuse_everything", "frozen_repair"]:
        raise ValueError("locked execution requires test split and all three frozen arms")
    problems = _validate_test_single_shot(
        cfg, split="test", i_understand=i_understand, config_path=config_path,
    )
    if problems:
        raise ValueError("; ".join(problems))
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) is None:
        raise ValueError("locked attempt requires a valid run_id")
    full_commit = _git_full_commit()
    if not isinstance(full_commit, str) or re.fullmatch(r"[0-9a-f]{40}", full_commit) is None:
        raise ValueError("locked attempt requires a full source commit")
    arm, arm_hash = _frozen_repair(cfg)
    manifest_bytes = (ROOT / cfg["datasets"]["split_manifest"]).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen family manifest changed during admission")
    record = {
        "status": "reserved", "run_id": run_id, "split": "test",
        "reserved_at": _utc_now().isoformat(), "full_commit": full_commit,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_hash": cfg["datasets"]["dataset_hash"],
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "frozen_repair_sha256": arm_hash, "model_revision": cfg["model_revision"],
        "method_arms": cfg["method_arms"], "seed": cfg.get("seed"),
    }
    path = ROOT / "runs" / ATTEMPT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation is the arbitration, not a read-then-write check.
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError as exc:
        raise ValueError("locked attempt reservation failed; generation refused") from exc
    return generate(cfg, arm, record)


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
