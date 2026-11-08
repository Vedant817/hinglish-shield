"""Freeze one selected arm and prove the intervention is a runtime hook."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

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


PROVISIONAL_MODEL_ID = "CohereLabs/tiny-aya-global"
PROVISIONAL_MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
PROVISIONAL_SCHEMA = "provisional_locate_v1"


def _finite_vector(values: Any) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return False
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def _require_provisional_locate(direction: Mapping[str, Any]) -> None:
    if (direction.get("schema_version") != PROVISIONAL_SCHEMA
            or direction.get("provisional") is not True
            or direction.get("original_fp32_credit") is not False
            or direction.get("parity_credit") is not None):
        raise RepairError(
            "freeze refused: locate artifact is not provisional bf16 evidence", stage="freeze"
        )
    if (direction.get("model_id") != PROVISIONAL_MODEL_ID
            or direction.get("model_revision") != PROVISIONAL_MODEL_REVISION
            or direction.get("dtype") != "bfloat16"
            or direction.get("node") != "MLP 0"
            or type(direction.get("declared_score")) not in {int, float}
            or isinstance(direction.get("declared_score"), bool)
            or direction.get("declared_score") != 1.0):
        raise RepairError("freeze refused: locate artifact pins do not match", stage="freeze")
    if direction.get("metrics_eligible") is not False or direction.get("primary_asr") is not None:
        raise RepairError("freeze refused: locate artifact invented metrics", stage="freeze")
    if "test" in str(direction.get("split", "")):
        raise RepairError("freeze refused: locate cannot use the locked test split", stage="freeze")
    vector = direction.get("vector")
    if not _finite_vector(vector):
        raise RepairError("freeze refused: locate direction is missing or nonfinite", stage="freeze")
    norm = l2_norm(vector)
    declared_norm = direction.get("l2_norm")
    if (isinstance(declared_norm, bool) or not isinstance(declared_norm, (int, float))
            or not math.isfinite(float(declared_norm)) or norm == 0.0
            or abs(float(declared_norm) - norm) > 1e-6):
        raise RepairError("freeze refused: locate direction norm does not match", stage="freeze")
    controls = direction.get("controls")
    if not isinstance(controls, dict):
        raise RepairError("freeze refused: locate controls are missing", stage="freeze")
    zero = controls.get("alpha_zero_with_wrapper")
    random_arm = controls.get("random_norm_matched_seed_7")
    candidate = controls.get("candidate_alpha_1")
    def _num(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    if (not isinstance(zero, dict) or zero.get("ran") is not True
            or zero.get("matches_base") is not True or zero.get("max_abs_logit_diff") != 0.0
            or not isinstance(candidate, dict) or candidate.get("ran") is not True
            or not _num(candidate.get("max_abs_logit_diff"))
            or candidate["max_abs_logit_diff"] <= 0
            or not isinstance(random_arm, dict) or random_arm.get("ran") is not True
            or not _num(random_arm.get("max_abs_logit_diff"))
            or random_arm["max_abs_logit_diff"] <= 0
            or not _num(random_arm.get("norm_gap"))
            or random_arm["norm_gap"] > 1e-3):
        raise RepairError("freeze refused: locate controls did not pass", stage="freeze")
    checks = direction.get("checks")
    if not isinstance(checks, dict) or checks.get("alpha_zero_matches_base") is not True:
        raise RepairError("freeze refused: locate alpha-zero check is missing", stage="freeze")


def build_provisional_bf16_freeze(direction_path: Path) -> dict[str, Any]:
    """Freeze a provisional bf16 locate arm without granting fp32 parity credit."""
    direction = _load_object(direction_path, "provisional locate direction")
    _require_provisional_locate(direction)
    manifest: dict[str, Any] = {
        "schema_version": "provisional_bf16_freeze_v1",
        "verification_pathway": "provisional_bf16_locate",
        "intervention_type": "ActivationSteering.steer",
        "static_checkpoint": False,
        "safetune_superior": False,
        "circuit_superiority": False,
        "deployable_claim": False,
        "provisional": True,
        "original_fp32_credit": False,
        "parity_credit": None,
        "metrics_eligible": False,
        "primary_asr": None,
        "human_calibration": "pending",
        "not_a_scored_test": True,
        "proven_refusal_mechanism": False,
        "model_id": direction["model_id"],
        "model_revision": direction["model_revision"],
        "split": direction.get("split"),
        "dtype": "bfloat16",
        "library": direction.get("library"),
        "node": direction["node"],
        "hook_point": direction.get("hook_point"),
        "protocol_layer": direction.get("protocol_layer"),
        "node_mapping": direction.get("node_mapping"),
        "declared_score": direction["declared_score"],
        "score_kind": direction.get("score_kind"),
        "declared_arm": "candidate_alpha_1",
        "alpha": 1.0,
        "vector": direction["vector"],
        "l2_norm": direction["l2_norm"],
        "direction_sha256": direction["content_sha256"],
        "controls": direction["controls"],
        "checks": direction["checks"],
        "limitations": direction.get("limitations", []),
    }
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def verify_provisional_bf16_freeze(payload: dict[str, Any]) -> dict[str, Any]:
    """Fresh-process check of a provisional locate freeze. No weights are reloaded."""
    if payload.get("content_sha256") != content_sha256(payload):
        raise RepairError("provisional freeze hash does not match its contents", stage="freeze")
    if payload.get("verification_pathway") != "provisional_bf16_locate":
        raise RepairError("freeze manifest is not a provisional locate freeze", stage="freeze")
    if (payload.get("provisional") is not True
            or payload.get("original_fp32_credit") is not False
            or payload.get("parity_credit") is not None):
        raise RepairError("provisional freeze cannot claim original fp32 credit", stage="freeze")
    if (payload.get("primary_asr") is not None or payload.get("metrics_eligible") is not False
            or payload.get("deployable_claim") is not False):
        raise RepairError("provisional freeze invented metrics or a deployable claim", stage="freeze")
    if payload.get("intervention_type") != "ActivationSteering.steer":
        raise RepairError("provisional freeze is not an ActivationSteering arm", stage="freeze")
    if payload.get("declared_arm") != "candidate_alpha_1" or payload.get("alpha") != 1.0:
        raise RepairError("provisional freeze arm is not the declared candidate", stage="freeze")
    _require_provisional_locate(
        {
            "schema_version": PROVISIONAL_SCHEMA,
            "provisional": payload.get("provisional"),
            "original_fp32_credit": payload.get("original_fp32_credit"),
            "parity_credit": payload.get("parity_credit"),
            "model_id": payload.get("model_id"),
            "model_revision": payload.get("model_revision"),
            "dtype": payload.get("dtype"),
            "node": payload.get("node"),
            "declared_score": payload.get("declared_score"),
            "metrics_eligible": payload.get("metrics_eligible"),
            "primary_asr": payload.get("primary_asr"),
            "split": payload.get("split", "calibration"),
            "vector": payload.get("vector"),
            "l2_norm": payload.get("l2_norm"),
            "controls": payload.get("controls"),
            "checks": payload.get("checks"),
        }
    )
    return {
        "verified": True,
        "verification_pathway": "provisional_bf16_locate",
        "provisional": True,
        "original_fp32_credit": False,
        "parity_credit": None,
        "weights_reloaded": False,
        "runtime_verified_on_pretrained_weights": True,
        "metrics_eligible": False,
        "primary_asr": None,
        "deployable_claim": False,
        "not_a_scored_test": True,
        "safetune_superior": False,
        "static_checkpoint": False,
        "circuit_superiority": False,
    }


def _peek(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("missing sweep report", stage="freeze") from exc
    if not isinstance(payload, dict):
        raise RepairError("missing sweep report", stage="freeze")
    return payload


def write_freeze_manifest(sweep_path: Path, direction_path: Path, out_path: Path) -> dict[str, Any]:
    peeked = _peek(sweep_path)
    if peeked.get("generation") in {"per_step_hook_forward", "ActivationSteering.steer"}:
        manifest = build_measured_freeze(sweep_path, direction_path)
    else:
        manifest = build_freeze_manifest(sweep_path, direction_path)
    write_json(out_path, manifest)
    return manifest


def _bound_producer(sweep_path: Path, sweep: Mapping[str, Any], judgments: Any) -> dict[str, Any] | None:
    """Recompute the producer from the sibling generation ledger."""
    if not isinstance(judgments, dict):
        return None
    from .selection import primary_rubric, producer_from_ledger

    disk = producer_from_ledger(
        sweep_path.parent / "generations.jsonl",
        sweep_path.parent / "samples.jsonl",
        identity={
            "model_id": sweep.get("model_id"),
            "model_revision": sweep.get("model_revision"),
            "judge_model_id": sweep.get("judge_model_id"),
            "judge_revision": sweep.get("judge_revision"),
            "rubric": primary_rubric(),
        },
    )
    if sweep.get("producer") != disk:
        raise RepairError(
            "freeze refused: producer evidence does not match the generation ledger",
            stage="freeze",
        )
    return disk


def _measured_intervention(sweep: Mapping[str, Any]) -> str:
    generation = sweep.get("generation")
    if generation == "per_step_hook_forward":
        return "runtime_hook"
    if generation == "ActivationSteering.steer":
        if sweep.get("library_steer_per_new_token") is not True or sweep.get("not_a_single_steer_call") is not False:
            raise RepairError(
                "freeze refused: library generation is not per-token ActivationSteering.steer",
                stage="freeze",
            )
        if (
            sweep.get("artifact_kind") != "run"
            or sweep.get("factory_injected") is not False
            or sweep.get("not_tiny_aya") is not False
            or sweep.get("weights_executed_by_this_command") is not True
        ):
            raise RepairError("freeze refused: library generation is not a real-weight run", stage="freeze")
        return "ActivationSteering.steer"
    raise RepairError("freeze refused: sweep is not a per-step hook ledger", stage="freeze")


def build_measured_freeze(sweep_path: Path, direction_path: Path) -> dict[str, Any]:
    """Bind the measured arm to recomputed evidence. Does not load pretrained weights."""
    from .protocol import load_sweep_config
    from .real_runtime import assert_accept_bindings
    from .selection import decide_arm

    sweep = _load_object(sweep_path, "sweep report")
    direction = _load_object(direction_path, "frozen direction")
    intervention = _measured_intervention(sweep)
    if sweep.get("direction_sha256") != direction.get("content_sha256"):
        raise RepairError("sweep direction hash does not match the frozen direction", stage="freeze")
    if sweep.get("primary_asr") is not None or sweep.get("metrics_eligible") is not False:
        raise RepairError("freeze refused: metrics were invented", stage="freeze")
    if sweep.get("proven_refusal_mechanism") is not False:
        raise RepairError("freeze refused: direction claims a proven refusal mechanism", stage="freeze")
    judgments = sweep.get("judgments")
    members = sweep.get("dev_members")
    if not isinstance(members, list) or not members:
        raise RepairError("freeze refused: dev members are missing", stage="freeze")
    sweep_config = load_sweep_config(Path(__file__).resolve().parents[2] / "configs" / "repair" / "sweep.yaml")
    if sweep.get("sweep_sha256") != sweep_config["sha256"]:
        raise RepairError("freeze refused: sweep config hash does not match the committed sweep", stage="freeze")
    producer = _bound_producer(sweep_path, sweep, judgments)
    recomputed = decide_arm(
        members,
        judgments if isinstance(judgments, dict) else None,
        over_refusal_max_rise=sweep_config["over_refusal_max_rise"],
        capability_max_drop_points=sweep_config["capability_max_drop_points"],
        bootstrap_reps=sweep_config["bootstrap_reps"],
        bootstrap_seed=sweep_config["bootstrap_seed"],
        min_families_for_accept=sweep_config["min_families_for_accept"],
        producer=producer,
    )
    stored = sweep.get("decision")
    if not isinstance(stored, dict) or stored.get("decision") != recomputed.get("decision"):
        raise RepairError("freeze refused: stored decision does not match the evidence", stage="freeze")
    if recomputed.get("freeze") is not True:
        raise RepairError("freeze refused: selection is unavailable or rejected", stage="freeze")
    assert_accept_bindings(recomputed, min_families=sweep_config["min_families_for_accept"])
    vector = direction.get("vector")
    if not isinstance(vector, list) or l2_norm(vector) == 0.0:
        raise RepairError("freeze refused: direction norm is zero", stage="freeze")
    manifest: dict[str, Any] = {
        "verification_pathway": "measured_evidence",
        "intervention_type": intervention,
        "static_checkpoint": False,
        "safetune_superior": False,
        "circuit_superiority": False,
        "weights_reloaded": False,
        "runtime_verified_on_pretrained_weights": sweep.get("runtime_verified_on_pretrained_weights") is True,
        "deployable_claim": recomputed.get("deployable_claim") is True,
        "decision": recomputed.get("decision"),
        "decision_reason": recomputed.get("reason"),
        "n_families": recomputed.get("n_families"),
        "min_families_for_accept": sweep_config["min_families_for_accept"],
        "over_refusal_gate": recomputed.get("over_refusal_gate"),
        "capability_gate": recomputed.get("capability_gate"),
        "harm_low": (recomputed.get("harm_improvement") or {}).get("low"),
        "declared_arm": "candidate_alpha_1",
        "alpha": 1.0,
        "artifact_kind": sweep.get("artifact_kind"),
        "split": sweep.get("split"),
        "model_id": sweep.get("model_id"),
        "not_tiny_aya": sweep.get("not_tiny_aya"),
        "not_wildguard": True,
        "not_a_scored_test": True,
        "proven_refusal_mechanism": False,
        "primary_asr": None,
        "metrics_eligible": False,
        "direction_sha256": direction["content_sha256"],
        "sweep_sha256": sweep["content_sha256"],
        "vector": vector,
        "l2_norm": l2_norm(vector),
        "layer_index": direction.get("layer_index"),
        "judgments": judgments,
        "producer": producer,
        "dev_members": members,
        "disclaimer": direction.get("disclaimer"),
    }
    manifest["content_sha256"] = content_sha256(manifest)
    return manifest


def verify_measured(payload: dict[str, Any]) -> dict[str, Any]:
    """Fresh-process check of evidence bindings. Does not reload pretrained weights."""
    from .protocol import load_sweep_config
    from .real_runtime import assert_accept_bindings
    from .selection import decide_arm

    if payload.get("content_sha256") != content_sha256(payload):
        raise RepairError("freeze manifest hash does not match its contents", stage="freeze")
    if payload.get("verification_pathway") != "measured_evidence":
        raise RepairError("freeze manifest is not measured evidence", stage="freeze")
    intervention = payload.get("intervention_type")
    if intervention not in {"runtime_hook", "ActivationSteering.steer"}:
        raise RepairError("freeze manifest is not a measured intervention", stage="freeze")
    if payload.get("static_checkpoint") is not False or payload.get("safetune_superior") is not False:
        raise RepairError("freeze manifest claims an unmeasured method win", stage="freeze")
    if payload.get("circuit_superiority") is not False or payload.get("proven_refusal_mechanism") is not False:
        raise RepairError("freeze manifest claims an unmeasured method win", stage="freeze")
    if payload.get("primary_asr") is not None or payload.get("metrics_eligible") is not False:
        raise RepairError("freeze manifest invented metrics", stage="freeze")
    if payload.get("weights_reloaded") is not False:
        raise RepairError("freeze manifest claims a weight reload that this pathway does not do", stage="freeze")
    sweep_config = load_sweep_config(Path(__file__).resolve().parents[2] / "configs" / "repair" / "sweep.yaml")
    judgments = payload.get("judgments")
    producer = payload.get("producer") if isinstance(judgments, dict) else None
    if isinstance(judgments, dict) and not isinstance(producer, dict):
        raise RepairError("freeze manifest is missing producer evidence", stage="freeze")
    recomputed = decide_arm(
        payload["dev_members"],
        judgments if isinstance(judgments, dict) else None,
        over_refusal_max_rise=sweep_config["over_refusal_max_rise"],
        capability_max_drop_points=sweep_config["capability_max_drop_points"],
        bootstrap_reps=sweep_config["bootstrap_reps"],
        bootstrap_seed=sweep_config["bootstrap_seed"],
        min_families_for_accept=sweep_config["min_families_for_accept"],
        producer=producer if isinstance(producer, dict) else None,
    )
    if payload.get("decision") != recomputed.get("decision"):
        raise RepairError("fresh process decision does not match the evidence", stage="freeze")
    assert_accept_bindings(recomputed, min_families=sweep_config["min_families_for_accept"])
    if payload.get("declared_arm") != "candidate_alpha_1" or payload.get("alpha") != 1.0:
        raise RepairError("freeze manifest arm is not the declared candidate", stage="freeze")
    if payload.get("vector") is None or payload.get("direction_sha256") is None:
        raise RepairError("freeze manifest is missing the measured vector", stage="freeze")
    return {
        "verified": True,
        "verification_pathway": "measured_evidence",
        "weights_reloaded": False,
        "runtime_verified_on_pretrained_weights": False,
        "runtime_hook": intervention == "runtime_hook",
        "activation_steering": intervention == "ActivationSteering.steer",
        "decision": recomputed.get("decision"),
        "n_families": recomputed.get("n_families"),
        "primary_asr": None,
        "metrics_eligible": False,
        "safetune_superior": False,
        "static_checkpoint": False,
        "circuit_superiority": False,
        "not_a_scored_test": True,
    }


def verify_any(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("tensor_kind") == "deterministic_last_token_function":
        from .real_runtime import verify_tensor_fixture_payload

        return verify_tensor_fixture_payload(payload)
    if payload.get("verification_pathway") == "provisional_bf16_locate":
        return verify_provisional_bf16_freeze(payload)
    if payload.get("verification_pathway") == "measured_evidence":
        return verify_measured(payload)
    return verify_manifest(payload)


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
