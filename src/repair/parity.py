"""Recompute CircuitKIT Locate credit from artifact files.

The recorded passed boolean is not authority. ingestion.json is not authority.
Missing, failed, and incomplete results refuse credit.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .errors import RepairError

ROOT = Path(__file__).resolve().parents[2]
PINNED_PROMPTS_SHA256 = "3ab3ad15fba852e34e08b8c485e2e3c41adc8702aa80a0bbf62215d5001fabab"
VARIANT_TOP1_WITH_FLOOR = 0.99
VARIANT_TOP1_PROVISIONAL = 0.98
VARIANT_MEAN_KL_PROVISIONAL = 1e-2
VARIANT_LONG_KL_PROVISIONAL = 1e-2
LONG_KEY = "sliding_window_4196"


def _ingest_module():
    path = ROOT / "scripts" / "ingest_gpu_artifacts.py"
    spec = importlib.util.spec_from_file_location("hs_ingest_gpu_artifacts", path)
    if spec is None or spec.loader is None:
        raise RepairError("parity ingest helper is missing", stage="parity")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pinned_short_prompts() -> list[str]:
    notebook = json.loads((ROOT / "notebooks" / "gpu_parity_and_baseline.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for node in ast.parse("".join(cell.get("source", []))).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "PARITY_PROMPTS" for target in node.targets
            ):
                prompts = ast.literal_eval(node.value)
                digest = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
                if digest != PINNED_PROMPTS_SHA256:
                    raise RepairError("notebook prompts disagree with the pinned upstream copy", stage="parity")
                if not isinstance(prompts, list) or not prompts:
                    raise RepairError("pinned short prompts are missing", stage="parity")
                return list(prompts)
    raise RepairError("pinned short prompts are missing", stage="parity")


def _finite_nonnegative(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and float(value) >= 0.0


def _finite_unit(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0


def _finite_map(values: Any) -> bool:
    return isinstance(values, dict) and bool(values) and all(_finite_nonnegative(item) for item in values.values())


def recompute_original(run_dir: Path) -> dict[str, Any]:
    """Same original-gate recomputation as scripts/ingest_gpu_artifacts.parity_summary."""
    files: dict[str, bytes] = {}
    for name in ("parity_original_sequential.json", "parity_long_prompt.json"):
        path = run_dir / name
        if path.is_file():
            files[name] = path.read_bytes()
    try:
        summary = _ingest_module().parity_summary(files)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RepairError("original parity artifacts are unreadable", stage="parity") from exc
    status = summary.get("status")
    if status not in {"pass", "fail", "incomplete"}:
        status = "incomplete"
    long_kl = summary.get("long_kl")
    return {
        "status": status,
        "gate": summary.get("gate"),
        "kl_tolerance": summary.get("kl_tolerance"),
        "recorded_passed": summary.get("recorded_passed"),
        "recorded_status": summary.get("recorded_status"),
        "recorded_claim_consistent": summary.get("recorded_claim_consistent"),
        "short_present": isinstance(summary.get("short_kl"), dict) and bool(summary.get("short_kl")),
        "long_present": isinstance(long_kl, dict) and LONG_KEY in long_kl,
    }


def recompute_variant(payload: Mapping[str, Any]) -> dict[str, Any]:
    """D4 bf16 rules from the notebook decision functions. Ignores payload passed."""
    recorded = payload.get("passed")
    if payload.get("dtype") != "bfloat16" or payload.get("hardware") != "cpu":
        return _variant_result("fail", "dtype or hardware is not CPU bfloat16", recorded)
    if payload.get("used_fp32_tol") is True:
        return _variant_result("fail", "fp32 tolerance was applied to the variant", recorded)
    accelerator = payload.get("accelerator")
    if not isinstance(accelerator, str) or "T4" not in accelerator:
        return _variant_result("fail", "CPU bf16 variant is permitted only when the accelerator name contains T4", recorded)
    if payload.get("error") not in (None,):
        if payload.get("long_ran") is True:
            return _variant_result("fail", "variant recorded an error after the long case", recorded)
        return _variant_result("incomplete", "variant recorded an error before a finished long case", recorded)
    prompts = set(pinned_short_prompts())
    short = payload.get("kl_hf_tl_bf16")
    if not _finite_map(short) or set(short) != prompts:
        return _variant_result("incomplete", "short bf16 KL is missing or does not match the pinned prompts", recorded)
    long = payload.get("kl_long_hf_tl_bf16")
    if payload.get("long_ran") is not True or not _finite_map(long) or set(long) != {LONG_KEY}:
        return _variant_result("incomplete", "long 4196 bf16 KL is missing", recorded)
    floor = payload.get("floor_kl_fp32_bf16")
    floor_ok = _finite_map(floor) and set(floor) == prompts
    if floor_ok:
        short_ok = all(float(short[prompt]) <= 2.0 * float(floor[prompt]) for prompt in prompts)
        provisional_short = False
        top1_need = VARIANT_TOP1_WITH_FLOOR
    else:
        mean_kl = sum(float(value) for value in short.values()) / len(short)
        short_ok = mean_kl < VARIANT_MEAN_KL_PROVISIONAL
        provisional_short = True
        top1_need = VARIANT_TOP1_PROVISIONAL
    top1 = payload.get("top1_agreement")
    top1_ok = _finite_unit(top1) and float(top1) >= top1_need
    long_floor = payload.get("long_floor")
    long_floor_ok = (
        isinstance(long_floor, dict)
        and set(long_floor) == {LONG_KEY}
        and _finite_nonnegative(long_floor.get(LONG_KEY))
    )
    if long_floor_ok:
        long_ok = float(long[LONG_KEY]) <= 2.0 * float(long_floor[LONG_KEY])
        provisional_long = False
    else:
        long_ok = float(long[LONG_KEY]) < VARIANT_LONG_KL_PROVISIONAL
        provisional_long = True
    passed = bool(short_ok and top1_ok and long_ok)
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "provisional": bool(provisional_short or provisional_long),
        "recorded_passed": recorded,
        "reason": None if passed else "variant threshold",
        "top1_need": top1_need,
        "top1_ok": top1_ok,
        "used_fp32_tol": False,
    }


def _variant_result(status: str, reason: str, recorded: Any) -> dict[str, Any]:
    return {
        "status": status,
        "passed": False,
        "provisional": None,
        "recorded_passed": recorded,
        "reason": reason,
        "used_fp32_tol": False,
    }


def locate_credit(run_dir: Path | None, *, accept_variant: bool) -> dict[str, Any]:
    """Grant CircuitKIT credit only from a recomputed pass."""
    if type(accept_variant) is not bool:
        raise RepairError("variant acceptance flag must be a boolean", stage="parity")
    if run_dir is None or not run_dir.is_dir():
        return {
            "credit": False,
            "status": "missing",
            "reason": "no parity run directory",
            "accept_preregistered_bf16_variant": accept_variant,
            "original": {"status": "missing"},
            "variant": None,
            "ignored_ingestion_json": True,
        }
    original = recompute_original(run_dir)
    variant = None
    variant_path = run_dir / "parity_variant_bf16.json"
    if variant_path.is_file():
        try:
            payload = json.loads(variant_path.read_text(encoding="utf-8"), parse_constant=lambda _value: None)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RepairError("variant parity artifact is unreadable", stage="parity") from exc
        if not isinstance(payload, dict):
            raise RepairError("variant parity artifact is unreadable", stage="parity")
        variant = recompute_variant(payload)
    original_ok = original["status"] == "pass"
    variant_ok = bool(accept_variant and variant is not None and variant["status"] == "pass")
    if original_ok:
        status = "pass"
        reason = "original_fp32_short_and_4196_long"
        credit = True
    elif variant_ok:
        status = "pass"
        reason = "explicitly_accepted_d4_bf16_variant"
        credit = True
    else:
        credit = False
        if variant is not None and accept_variant and original["status"] != "pass":
            status = variant["status"]
            reason = variant.get("reason") or variant["status"]
        else:
            status = original["status"]
            reason = original["status"]
            if original["status"] != "pass" and variant is not None and variant["status"] == "pass" and not accept_variant:
                reason = "variant pass is not accepted"
                status = original["status"]
    return {
        "credit": credit,
        "status": status,
        "reason": reason,
        "accept_preregistered_bf16_variant": accept_variant,
        "original": original,
        "variant": variant,
        "ignored_ingestion_json": True,
    }
