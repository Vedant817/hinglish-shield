"""Command line for locate, steer, sweep, and freeze."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .errors import RepairError
from .freeze import (
    build_provisional_bf16_freeze,
    verify_any,
    verify_provisional_bf16_freeze,
    write_freeze_manifest,
)
from .jsonutil import content_sha256, write_json
from .parity import locate_credit
from .protocol import load_provenance
from .real_runtime import write_tensor_fixture_evidence
from .workflow import locate, sweep

ROOT = Path(__file__).resolve().parents[2]


def _path(value: str) -> Path:
    return Path(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.repair", description="Calibration and dev repair workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    credit = sub.add_parser("credit", help="Recompute Locate credit from a parity run directory")
    credit.add_argument("--parity-run", type=_path, required=True)
    credit.add_argument("--provenance", type=_path, required=True)

    locate_parser = sub.add_parser("locate", help="Extract a calibration direction and freeze it")
    _add_common(locate_parser)
    locate_parser.add_argument("--index", type=_path, required=True)
    locate_parser.add_argument("--prompts", type=_path)
    locate_parser.add_argument("--activations", type=_path)
    locate_parser.add_argument("--out", type=_path, required=True)
    locate_parser.add_argument("--backend", default="fixture", choices=("fixture", "circuitkit"))
    locate_parser.add_argument("--parity-run", type=_path)
    locate_parser.add_argument("--runtime-config", type=_path, default=ROOT / "configs" / "repair" / "real_runtime.yaml")
    locate_parser.add_argument("--circuit-scores", type=_path)
    locate_parser.add_argument("--circuitkit-source", type=_path)
    locate_parser.add_argument("--reviewed-lineage", type=_path)
    locate_parser.add_argument("--review-sheet", type=_path)
    _add_lineage_pins(locate_parser)

    steer_parser = sub.add_parser("steer", help="Apply the frozen direction on dev with runtime hooks")
    _add_sweep_io(steer_parser)

    sweep_parser = sub.add_parser("sweep", help="Record every fixed arm, then choose from supplied evidence")
    _add_sweep_io(sweep_parser)

    freeze_parser = sub.add_parser("freeze", help="Write the rollback manifest when selection allows it")
    freeze_parser.add_argument("--sweep", type=_path, required=True)
    freeze_parser.add_argument("--direction", type=_path, required=True)
    freeze_parser.add_argument("--out", type=_path, required=True)

    provisional = sub.add_parser(
        "freeze-provisional",
        help="Freeze a provisional bf16 locate direction (calibration/dev only)",
    )
    provisional.add_argument("--direction", type=_path, required=True)
    provisional.add_argument("--out", type=_path, required=True)

    verify_parser = sub.add_parser("verify-freeze", help="Re-run the frozen hook in a fresh process")
    verify_parser.add_argument("--manifest", type=_path, required=True)

    child = sub.add_parser("verify-freeze-child")
    child.add_argument("manifest", type=_path)

    fixture = sub.add_parser("tensor-fixture", help="Write a deterministic tensor fixture ledger")
    fixture.add_argument("--out-dir", type=_path, required=True)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", type=_path, required=True)
    parser.add_argument("--provenance", type=_path, required=True)
    parser.add_argument("--manifest", type=_path, required=True)
    parser.add_argument("--artifact-kind", required=True, choices=("fixture", "run"))
    parser.add_argument("--allow-unpinned-manifest", action="store_true")


def _add_sweep_io(parser: argparse.ArgumentParser) -> None:
    _add_common(parser)
    parser.add_argument("--sweep-config", type=_path, required=True)
    parser.add_argument("--direction", type=_path, required=True)
    parser.add_argument("--calibration-index", type=_path, required=True)
    parser.add_argument("--dev-index", type=_path, required=True)
    parser.add_argument("--dev-prompts", type=_path, required=True)
    parser.add_argument("--judgments", type=_path)
    parser.add_argument("--out", type=_path, required=True)
    parser.add_argument("--parity-run", type=_path)
    parser.add_argument("--runtime-config", type=_path, default=ROOT / "configs" / "repair" / "real_runtime.yaml")
    parser.add_argument("--circuit-scores", type=_path)
    parser.add_argument("--circuitkit-source", type=_path)
    parser.add_argument("--reviewed-lineage", type=_path)
    parser.add_argument("--review-sheet", type=_path)
    _add_lineage_pins(parser)


def _add_lineage_pins(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reviewed-lineage-sha256")
    parser.add_argument("--reviewed-evidence-sha256")
    parser.add_argument("--review-sheet-sha256")
    parser.add_argument("--canonical-source-ids-sha256")
    parser.add_argument("--prompts-sha256")


def _pins(args: argparse.Namespace) -> dict[str, str] | None:
    names = (
        "reviewed_lineage_sha256",
        "reviewed_evidence_sha256",
        "review_sheet_sha256",
        "canonical_source_ids_sha256",
        "prompts_sha256",
    )
    values = {name: getattr(args, name) for name in names}
    if all(value is None for value in values.values()):
        return None
    if any(not isinstance(value, str) or len(value) != 64 for value in values.values()):
        raise RepairError("reviewed lineage contract is required", stage="membership")
    return values


def _scores(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("circuit scores are unreadable", stage="circuitkit") from exc
    if not isinstance(payload, dict):
        raise RepairError("circuit scores are unreadable", stage="circuitkit")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "credit":
            provenance = load_provenance(args.provenance)
            report = locate_credit(
                args.parity_run,
                accept_variant=provenance["accept_preregistered_bf16_variant"],
            )
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "locate":
            locate(
                protocol_path=args.protocol,
                provenance_path=args.provenance,
                index_path=args.index,
                manifest_path=args.manifest,
                out_path=args.out,
                artifact_kind=args.artifact_kind,
                prompts_path=args.prompts,
                activations_path=args.activations,
                parity_run=args.parity_run,
                allow_unpinned_manifest=args.allow_unpinned_manifest,
                backend=args.backend,
                runtime_config_path=args.runtime_config,
                circuit_scores=_scores(args.circuit_scores),
                circuitkit_source=args.circuitkit_source,
                reviewed_lineage_path=args.reviewed_lineage,
                review_sheet_path=args.review_sheet,
                reviewed_pins=_pins(args),
            )
            return 0
        if args.command in {"steer", "sweep"}:
            sweep(
                protocol_path=args.protocol,
                provenance_path=args.provenance,
                sweep_path=args.sweep_config,
                direction_path=args.direction,
                calibration_index_path=args.calibration_index,
                dev_index_path=args.dev_index,
                dev_prompts_path=args.dev_prompts,
                manifest_path=args.manifest,
                out_path=args.out,
                artifact_kind=args.artifact_kind,
                judgments_path=args.judgments,
                allow_unpinned_manifest=args.allow_unpinned_manifest,
                parity_run=args.parity_run,
                runtime_config_path=args.runtime_config,
                circuit_scores=_scores(args.circuit_scores),
                circuitkit_source=args.circuitkit_source,
                reviewed_lineage_path=args.reviewed_lineage,
                review_sheet_path=args.review_sheet,
                reviewed_pins=_pins(args),
            )
            return 0
        if args.command == "freeze":
            write_freeze_manifest(args.sweep, args.direction, args.out)
            return 0
        if args.command == "freeze-provisional":
            manifest = build_provisional_bf16_freeze(args.direction)
            report = verify_provisional_bf16_freeze(manifest)
            manifest["verification"] = report
            manifest["content_sha256"] = content_sha256(manifest)
            write_json(args.out, manifest)
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "verify-freeze":
            completed = subprocess.run(
                [sys.executable, "-m", "src.repair", "verify-freeze-child", str(args.manifest)],
                cwd=str(ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
            if completed.returncode != 0:
                if completed.stderr:
                    print(completed.stderr, file=sys.stderr)
                return completed.returncode
            return 0
        if args.command == "verify-freeze-child":
            try:
                payload = json.loads(args.manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RepairError("freeze manifest is unreadable", stage="freeze") from exc
            if not isinstance(payload, dict):
                raise RepairError("freeze manifest is unreadable", stage="freeze")
            report = verify_any(payload)
            print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "tensor-fixture":
            write_tensor_fixture_evidence(args.out_dir)
            return 0
        raise RepairError(f"unknown command {args.command}", stage="cli")
    except RepairError as exc:
        print(f"repair refused ({exc.stage}): {exc}", file=sys.stderr)
        return 2
