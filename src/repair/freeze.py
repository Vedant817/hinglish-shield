"""Freeze one selected arm and prove the intervention is a runtime hook."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import RepairError
from .fixture_model import MODEL_ID, TINY_AYA_ID, RandomWeightFixture
from .hooks import run_base, run_steered
from .jsonutil import content_sha256, write_json
from .vectors import l2_norm, max_abs_diff


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"missing {label}", stage="freeze") from exc
    if not isinstance(payload, dict):
        raise RepairError(f"missing {label}", stage="freeze")
    if payload.get("content_sha256") != content_sha256(payload):
        raise RepairError(f"{label} hash does not match its contents", stage="freeze")
    return payload


def build_freeze_manifest(sweep_path: Path, direction_path: Path) -> dict[str, Any]:
    sweep = _load_object(sweep_path, "sweep report")
    direction = _load_object(direction_path, "frozen direction")
    decision = sweep.get("decision")
    if not isinstance(decision, dict) or decision.get("freeze") is not True or sweep.get("freeze_allowed") is not True:
        raise RepairError("freeze refused: selection is unavailable or rejected", stage="freeze")
    if decision.get("decision") not in {"accept", "revise"}:
        raise RepairError("freeze refused: selection is unavailable or rejected", stage="freeze")
    if sweep.get("direction_sha256") != direction.get("content_sha256"):
        raise RepairError("sweep direction hash does not match the frozen direction", stage="freeze")
    if sweep.get("artifact_kind") != "fixture" or sweep.get("model_id") != MODEL_ID:
        raise RepairError("freeze refused: this manifest only covers the random-weight fixture", stage="freeze")
    if sweep.get("not_tiny_aya") is not True or direction.get("not_tiny_aya") is not True:
        raise RepairError("freeze refused: Tiny Aya label is missing", stage="freeze")
    vector = direction.get("vector")
    if not isinstance(vector, list) or l2_norm(vector) == 0.0:
        raise RepairError("freeze refused: direction norm is zero", stage="freeze")
    example = sweep.get("check_example")
    if not isinstance(example, dict) or not isinstance(example.get("token_ids"), list):
        raise RepairError("sweep report is missing the fresh-process check", stage="freeze")
    model = RandomWeightFixture(
        seed=sweep["weight_seed"],
        n_layers=sweep["n_layers"],
        d_model=sweep["d_model"],
        vocab=sweep["vocab"],
    )
    fingerprint = model.weight_fingerprint()
    layer = sweep["layer_index"]
    tokens = example["token_ids"]
    base = run_base(model, tokens, layer)
    steered, hook_count = run_steered(model, tokens, vector, 1.0, layer)
    zero, zero_hooks = run_steered(model, tokens, vector, 0.0, layer)
    if hook_count < 1 or zero_hooks < 1:
        raise RepairError("effective hook count is 0", stage="runtime")
    if model.hook_count() != 0:
        raise RepairError("hooks survived cleanup", stage="runtime")
    if model.weight_fingerprint() != fingerprint:
        raise RepairError("weights changed while building the freeze manifest", stage="runtime")
    if max_abs_diff(zero, base) != 0.0:
        raise RepairError("alpha zero is not equivalent to the base forward", stage="runtime")
    if max_abs_diff(steered, base) == 0.0:
        raise RepairError("runtime intervention did not change the forward", stage="runtime")
    manifest: dict[str, Any] = {
        "intervention_type": "runtime_hook",
        "static_checkpoint": False,
        "safetune_superior": False,
        "circuit_superiority": False,
        "deployable_claim": decision.get("deployable_claim") is True,
        "decision": decision.get("decision"),
        "decision_reason": decision.get("reason"),
        "artifact_kind": "fixture",
        "model_id": MODEL_ID,
        "model_revision": "fixture-not-a-checkpoint",
        "dtype": "float32",
        "not_tiny_aya": True,
        "not_wildguard": True,
        "not_a_scored_test": True,
        "proven_refusal_mechanism": False,
        "direction_sha256": direction["content_sha256"],
        "sweep_sha256": sweep["content_sha256"],
        "vector": vector,
        "l2_norm": l2_norm(vector),
        "alpha": 1.0,
        "layer_index": layer,
        "position": direction.get("position"),
        "convention": direction.get("convention"),
        "weight_seed": sweep["weight_seed"],
        "n_layers": sweep["n_layers"],
        "d_model": sweep["d_model"],
        "vocab": sweep["vocab"],
        "weight_fingerprint": fingerprint,
        "check_token_ids": tokens,
        "check_base_residual": base,
        "check_steered_residual": steered,
        "rollback": {
            "method": "remove_runtime_hooks",
            "weights_untouched": True,
            "alpha_zero_restores_base": True,
            "weight_fingerprint": fingerprint,
        },
        "disclaimer": direction.get("disclaimer"),
    }
    if manifest["model_id"] == TINY_AYA_ID:
        raise RepairError("freeze manifest must not claim Tiny Aya", stage="freeze")
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def write_freeze_manifest(sweep_path: Path, direction_path: Path, out_path: Path) -> dict[str, Any]:
    manifest = build_freeze_manifest(sweep_path, direction_path)
    write_json(out_path, manifest)
    return manifest


def verify_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Reload the frozen arm and show the change is a hook, not a weight edit."""
    if payload.get("content_sha256") != content_sha256(payload):
        raise RepairError("freeze manifest hash does not match its contents", stage="freeze")
    if payload.get("intervention_type") != "runtime_hook":
        raise RepairError("freeze manifest is not a runtime hook", stage="freeze")
    if payload.get("static_checkpoint") is not False:
        raise RepairError("freeze manifest claims a static checkpoint", stage="freeze")
    if payload.get("safetune_superior") is not False or payload.get("circuit_superiority") is not False:
        raise RepairError("freeze manifest claims an unmeasured method win", stage="freeze")
    if payload.get("model_id") != MODEL_ID or payload.get("not_tiny_aya") is not True:
        raise RepairError("freeze manifest model is not the random-weight fixture", stage="freeze")
    if payload.get("artifact_kind") != "fixture":
        raise RepairError("freeze verification refuses a non-fixture artifact", stage="freeze")
    model = RandomWeightFixture(
        seed=payload["weight_seed"],
        n_layers=payload["n_layers"],
        d_model=payload["d_model"],
        vocab=payload["vocab"],
    )
    fingerprint = model.weight_fingerprint()
    if fingerprint != payload.get("weight_fingerprint"):
        raise RepairError("fresh process weights do not match the freeze pin", stage="freeze")
    if fingerprint != payload.get("rollback", {}).get("weight_fingerprint"):
        raise RepairError("rollback fingerprint does not match the model", stage="freeze")
    tokens = payload["check_token_ids"]
    layer = payload["layer_index"]
    vector = payload["vector"]
    before = model.weight_fingerprint()
    base = run_base(model, tokens, layer)
    steered, hook_count = run_steered(model, tokens, vector, payload["alpha"], layer)
    zero, zero_hooks = run_steered(model, tokens, vector, 0.0, layer)
    after = model.weight_fingerprint()
    if before != after:
        raise RepairError("weights changed during the runtime intervention", stage="freeze")
    if hook_count < 1 or zero_hooks < 1 or model.hook_count() != 0:
        raise RepairError("hook lifecycle failed in the fresh process", stage="freeze")
    if max_abs_diff(zero, base) != 0.0:
        raise RepairError("alpha zero is not equivalent to the base forward", stage="freeze")
    if payload["alpha"] != 0.0 and max_abs_diff(steered, base) == 0.0:
        raise RepairError("fresh process found no runtime intervention", stage="freeze")
    if max_abs_diff(base, payload["check_base_residual"]) != 0.0:
        raise RepairError("fresh process base residual does not match the manifest", stage="freeze")
    if max_abs_diff(steered, payload["check_steered_residual"]) != 0.0:
        raise RepairError("fresh process steered residual does not match the manifest", stage="freeze")
    return {
        "verified": True,
        "runtime_hook": True,
        "weights_unchanged": True,
        "alpha_zero_matches_base": True,
        "effective_hook_count": hook_count,
        "hooks_after_cleanup": model.hook_count(),
        "artifact_kind": "fixture",
        "model_id": MODEL_ID,
        "not_tiny_aya": True,
        "not_wildguard": True,
        "not_a_scored_test": True,
        "safetune_superior": False,
        "static_checkpoint": False,
        "circuit_superiority": False,
        "weight_fingerprint": fingerprint,
    }
