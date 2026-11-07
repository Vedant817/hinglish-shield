"""Locate, steer, and sweep on calibration and dev only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .direction import assert_frozen_direction, extract_direction
from .errors import RepairError
from .fixture_model import MODEL_ID, TINY_AYA_ID, RandomWeightFixture
from .hooks import run_base, run_steered
from .jsonutil import content_sha256, strict_json_bytes, write_json
from .membership import (
    assert_activations_are_not_prompts,
    assert_repair_membership,
    load_authority,
    load_index,
    load_prompts,
    require_both_classes,
)
from .parity import locate_credit
from .protocol import load_provenance, load_protocol, load_sweep_config
from .selection import decide_arm
from .vectors import l2_norm, max_abs_diff, random_norm_matched

SHA40 = __import__("re").compile(r"[0-9a-f]{40}")
SHA256 = __import__("re").compile(r"[0-9a-f]{64}")
ALLOWED_DTYPES = frozenset({"bfloat16", "float16", "float32"})


def _members(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "sample_id": row["sample_id"],
            "item_id": row["item_id"],
            "family_id": row["family_id"],
            "source_dataset": row["source_dataset"],
            "source_id": row["source_id"],
            "lang": row["lang"],
            "hs_label": row["hs_label"],
            "split": row["split"],
        }
        for row in rows
    ]


def _assert_real_pins(provenance: Mapping[str, Any]) -> dict[str, Any]:
    real = provenance["real_model"]
    model_id = real.get("model_id")
    if model_id != TINY_AYA_ID:
        raise RepairError("real repair model id is not the pinned Tiny Aya id", stage="provenance")
    revision = real.get("model_revision")
    template = real.get("chat_template_sha256")
    dtype = real.get("dtype")
    if not isinstance(revision, str) or SHA40.fullmatch(revision) is None:
        raise RepairError("unpinned model revision", stage="provenance")
    if not isinstance(template, str) or SHA256.fullmatch(template) is None:
        raise RepairError("unpinned chat template", stage="provenance")
    if dtype not in ALLOWED_DTYPES:
        raise RepairError("unpinned dtype", stage="provenance")
    allowed = real.get("allowed_dtypes")
    if not isinstance(allowed, list) or dtype not in allowed:
        raise RepairError("dtype is outside the pinned set", stage="provenance")
    return {"model_id": model_id, "model_revision": revision, "dtype": dtype, "chat_template_sha256": template}


def _assert_fixture_pins(provenance: Mapping[str, Any]) -> dict[str, Any]:
    fixture = provenance["fixture_model"]
    if fixture.get("model_id") != MODEL_ID:
        raise RepairError("fixture model id is not the random-weight fixture", stage="provenance")
    if fixture.get("model_id") == TINY_AYA_ID:
        raise RepairError("fixture model id must not be Tiny Aya", stage="provenance")
    if fixture.get("not_tiny_aya") is not True or fixture.get("not_wildguard") is not True:
        raise RepairError("fixture provenance must say it is not Tiny Aya and not WildGuard", stage="provenance")
    if fixture.get("dtype") != "float32" or fixture.get("artifact_kind") != "fixture":
        raise RepairError("fixture dtype pin is wrong", stage="provenance")
    for name in ("weight_seed", "n_layers", "d_model", "vocab"):
        if type(fixture.get(name)) is not int:
            raise RepairError(f"fixture {name} is unpinned", stage="provenance")
    return dict(fixture)


def _authority(manifest: Path, *, artifact_kind: str, allow_unpinned_manifest: bool):
    if allow_unpinned_manifest and artifact_kind != "fixture":
        raise RepairError("unpinned manifests are fixture-only", stage="membership")
    return load_authority(manifest, require_pinned_hash=not allow_unpinned_manifest)


def _checked_index(index: Path, manifest: Path, *, artifact_kind: str, allow_unpinned: bool, allowed: frozenset[str]):
    authority = _authority(manifest, artifact_kind=artifact_kind, allow_unpinned_manifest=allow_unpinned)
    rows = load_index(index)
    assert_repair_membership(rows, authority, allowed_splits=allowed)
    return authority, rows


def _tokens(model: RandomWeightFixture, row: Mapping[str, Any], *, artifact_kind: str) -> list[int]:
    supplied = row.get("fixture_token_ids")
    if supplied is not None:
        if artifact_kind != "fixture":
            raise RepairError("fixture token ids are not valid on a run", stage="runtime")
        if any(type(token) is not int or token < 0 or token >= model.vocab for token in supplied):
            raise RepairError("fixture token is outside the vocab", stage="runtime")
        return list(supplied)
    return model.tokens_for_prompt(row["prompt"])


def _stamp_common(payload: dict[str, Any], *, artifact_kind: str, model_pins: Mapping[str, Any]) -> None:
    payload["artifact_kind"] = artifact_kind
    payload["not_wildguard"] = True
    payload["not_a_scored_test"] = True
    payload["proven_refusal_mechanism"] = False
    payload["weights_executed_by_this_command"] = artifact_kind == "fixture"
    if artifact_kind == "fixture":
        payload["not_tiny_aya"] = True
        payload["model_id"] = MODEL_ID
        payload["model_revision"] = "fixture-not-a-checkpoint"
        payload["dtype"] = "float32"
    else:
        payload["not_tiny_aya"] = False
        payload["activation_source"] = "caller_supplied"
        payload["model_id"] = model_pins["model_id"]
        payload["model_revision"] = model_pins["model_revision"]
        payload["dtype"] = model_pins["dtype"]
        payload["chat_template_sha256"] = model_pins["chat_template_sha256"]


def locate(
    *,
    protocol_path: Path,
    provenance_path: Path,
    index_path: Path,
    manifest_path: Path,
    out_path: Path,
    artifact_kind: str,
    prompts_path: Path | None = None,
    activations_path: Path | None = None,
    parity_run: Path | None = None,
    allow_unpinned_manifest: bool = False,
    backend: str = "fixture",
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    provenance = load_provenance(provenance_path)
    if artifact_kind == "run":
        pins = _assert_real_pins(provenance)
    elif artifact_kind == "fixture":
        pins = _assert_fixture_pins(provenance)
    else:
        raise RepairError("artifact kind must be fixture or run", stage="provenance")
    if backend == "circuitkit":
        credit = locate_credit(parity_run, accept_variant=provenance["accept_preregistered_bf16_variant"])
        if not credit["credit"]:
            raise RepairError(
                "CircuitKIT locate refused: parity credit is missing, failed, or incomplete",
                stage="parity",
            )
        raise RepairError(
            "CircuitKIT locate needs an in-process model. This command does not download weights.",
            stage="circuitkit",
        )
    if backend != "fixture":
        raise RepairError("unknown locate backend", stage="protocol")
    authority, rows = _checked_index(
        index_path,
        manifest_path,
        artifact_kind=artifact_kind,
        allow_unpinned=allow_unpinned_manifest,
        allowed=frozenset({"calibration"}),
    )
    require_both_classes(rows, stage="direction")
    records = _calibration_records(
        rows,
        protocol=protocol,
        artifact_kind=artifact_kind,
        pins=pins,
        prompts_path=prompts_path,
        activations_path=activations_path,
    )
    direction = extract_direction(
        records,
        layer_index=protocol["layer_index"],
        layer_name=str(protocol["layer_name"]),
        position=protocol["position"],
        target_label=str(protocol["target_hs_label"]),
        source_label=str(protocol["source_hs_label"]),
        protocol_sha256=protocol["sha256"],
        protocol_id=str(protocol["protocol_id"]),
    )
    direction["calibration_members"] = _members(rows)
    direction["canonical_manifest_sha256"] = authority["sha256"]
    _stamp_common(direction, artifact_kind=artifact_kind, model_pins=pins)
    direction["weight_seed"] = pins.get("weight_seed")
    direction["n_layers"] = pins.get("n_layers")
    direction["d_model"] = pins.get("d_model")
    direction["vocab"] = pins.get("vocab")
    direction["content_sha256"] = content_sha256(direction)
    write_json(out_path, direction)
    return direction


def _calibration_records(
    rows: Sequence[Mapping[str, str]],
    *,
    protocol: Mapping[str, Any],
    artifact_kind: str,
    pins: Mapping[str, Any],
    prompts_path: Path | None,
    activations_path: Path | None,
) -> list[dict[str, Any]]:
    layer = protocol["layer_index"]
    position = protocol["position"]
    if activations_path is not None:
        payload = _read_activations(activations_path)
        by_id = {row["sample_id"]: row for row in rows}
        records = []
        seen: set[str] = set()
        for item in payload:
            sample_id = item.get("sample_id")
            if sample_id not in by_id:
                raise RepairError("activation row is outside the checked calibration index", stage="activations")
            if sample_id in seen:
                raise RepairError("duplicate activation row", stage="activations")
            seen.add(sample_id)
            index_row = by_id[sample_id]
            records.append(
                {
                    "sample_id": sample_id,
                    "split": index_row["split"],
                    "hs_label": index_row["hs_label"],
                    "layer_index": layer,
                    "position": position,
                    "vector": item.get("vector"),
                }
            )
        if seen != set(by_id):
            raise RepairError("activation file does not cover the calibration index", stage="activations")
        return records
    if prompts_path is None:
        raise RepairError("missing prompts", stage="prompts")
    if artifact_kind != "fixture":
        raise RepairError(
            "a real run must pass calibration activations. This command does not download weights.",
            stage="provenance",
        )
    prompts = load_prompts(prompts_path, rows)
    model = RandomWeightFixture(
        seed=pins["weight_seed"],
        n_layers=pins["n_layers"],
        d_model=pins["d_model"],
        vocab=pins["vocab"],
    )
    records = []
    for row in prompts:
        tokens = _tokens(model, row, artifact_kind=artifact_kind)
        vector = model.forward_last(tokens, layer=layer, apply_hooks=False)
        records.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "hs_label": row["hs_label"],
                "layer_index": layer,
                "position": position,
                "vector": vector,
            }
        )
    return records


def _read_activations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RepairError("missing activations", stage="activations")
    try:
        payload = strict_json_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepairError("activation file is unreadable", stage="activations") from exc
    assert_activations_are_not_prompts(payload)
    if isinstance(payload, dict):
        payload = payload.get("rows")
    if not isinstance(payload, list) or not payload:
        raise RepairError("activation file is empty", stage="activations")
    return payload


def _assert_calibration_members(direction: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    members = direction.get("calibration_members")
    if not isinstance(members, list) or not members:
        raise RepairError("direction is missing calibration ids", stage="direction")
    assert_repair_membership(members, authority, allowed_splits=frozenset({"calibration"}))
    ids = direction.get("calibration_ids")
    if sorted(member["sample_id"] for member in members) != sorted(ids):
        raise RepairError("direction calibration ids do not match the frozen members", stage="direction")


def steer_rows(
    model: RandomWeightFixture,
    rows: Sequence[Mapping[str, Any]],
    vector: Sequence[float],
    alpha: float | None,
    layer: int,
    *,
    artifact_kind: str,
    hooks: bool,
) -> list[dict[str, Any]]:
    outputs = []
    for row in rows:
        tokens = _tokens(model, row, artifact_kind=artifact_kind)
        if not hooks:
            if model.hook_count() != 0:
                raise RepairError("base forward found a leftover hook", stage="runtime")
            residual = run_base(model, tokens, layer)
            hook_count = 0
        else:
            if alpha is None:
                raise RepairError("hooked arm is missing alpha", stage="runtime")
            residual, hook_count = run_steered(model, tokens, vector, alpha, layer)
            if hook_count < 1:
                raise RepairError("effective hook count is 0", stage="runtime")
        if model.hook_count() != 0:
            raise RepairError("hooks survived cleanup", stage="runtime")
        outputs.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "hs_label": row["hs_label"],
                "greedy_token": model.greedy_token(residual),
                "hook_count": hook_count,
                "residual": residual,
            }
        )
    return outputs


def sweep(
    *,
    protocol_path: Path,
    provenance_path: Path,
    sweep_path: Path,
    direction_path: Path,
    calibration_index_path: Path,
    dev_index_path: Path,
    dev_prompts_path: Path,
    manifest_path: Path,
    out_path: Path,
    artifact_kind: str,
    judgments_path: Path | None = None,
    allow_unpinned_manifest: bool = False,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    provenance = load_provenance(provenance_path)
    sweep_config = load_sweep_config(sweep_path)
    if artifact_kind == "fixture":
        pins = _assert_fixture_pins(provenance)
    elif artifact_kind == "run":
        _assert_real_pins(provenance)
        raise RepairError(
            "runtime hooks on the pinned model are not loaded by this command",
            stage="provenance",
        )
    else:
        raise RepairError("artifact kind must be fixture or run", stage="provenance")
    try:
        raw_direction = json.loads(direction_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("missing frozen direction", stage="direction") from exc
    if not isinstance(raw_direction, dict):
        raise RepairError("missing frozen direction", stage="direction")
    direction = assert_frozen_direction(raw_direction, protocol_sha256=protocol["sha256"])
    if direction.get("artifact_kind") != "fixture" or direction.get("model_id") != MODEL_ID:
        raise RepairError("dev sweep only accepts a frozen random-weight or matching fixture direction", stage="direction")
    if direction.get("not_tiny_aya") is not True:
        raise RepairError("dev sweep refuses a direction that is not marked as a fixture", stage="direction")
    for key in ("weight_seed", "n_layers", "d_model", "vocab"):
        if direction.get(key) != pins[key]:
            raise RepairError("direction fixture pins do not match provenance", stage="direction")
    authority = _authority(manifest_path, artifact_kind=artifact_kind, allow_unpinned_manifest=allow_unpinned_manifest)
    _assert_calibration_members(direction, authority)
    _authority_dev, dev_rows = _checked_index(
        dev_index_path,
        manifest_path,
        artifact_kind=artifact_kind,
        allow_unpinned=allow_unpinned_manifest,
        allowed=frozenset({"dev"}),
    )
    prompts = load_prompts(dev_prompts_path, dev_rows)
    model = RandomWeightFixture(
        seed=pins["weight_seed"],
        n_layers=pins["n_layers"],
        d_model=pins["d_model"],
        vocab=pins["vocab"],
    )
    layer = direction["layer_index"]
    if layer != protocol["layer_index"]:
        raise RepairError("direction layer does not match the proposed protocol", stage="direction")
    vector = direction["vector"]
    if l2_norm(vector) == 0.0:
        random_vector = None
    else:
        random_vector = random_norm_matched(vector, seed=7)
        if abs(l2_norm(random_vector) - l2_norm(vector)) > 1e-9:
            raise RepairError("random control norm does not match", stage="runtime")
    base = steer_rows(model, prompts, vector, None, layer, artifact_kind=artifact_kind, hooks=False)
    alpha_zero = steer_rows(model, prompts, vector, 0.0, layer, artifact_kind=artifact_kind, hooks=True)
    if any(row["hook_count"] < 1 for row in alpha_zero):
        raise RepairError("effective hook count is 0", stage="runtime")
    for left, right in zip(base, alpha_zero):
        if max_abs_diff(left["residual"], right["residual"]) != 0.0:
            raise RepairError("alpha zero is not equivalent to the base forward", stage="runtime")
    if random_vector is None:
        raise RepairError("random control refuses a zero direction", stage="runtime")
    random_arm = steer_rows(model, prompts, random_vector, 1.0, layer, artifact_kind=artifact_kind, hooks=True)
    candidate = steer_rows(model, prompts, vector, 1.0, layer, artifact_kind=artifact_kind, hooks=True)
    candidate_shift = max(max_abs_diff(left["residual"], right["residual"]) for left, right in zip(base, candidate))
    if candidate_shift == 0.0:
        raise RepairError("steering hook did not change the residual", stage="runtime")
    judgments = _load_judgments(judgments_path) if judgments_path is not None else None
    decision = decide_arm(
        dev_rows,
        judgments,
        over_refusal_max_rise=sweep_config["over_refusal_max_rise"],
        capability_max_drop_points=sweep_config["capability_max_drop_points"],
        bootstrap_reps=sweep_config["bootstrap_reps"],
        bootstrap_seed=sweep_config["bootstrap_seed"],
        min_families_for_accept=sweep_config["min_families_for_accept"],
    )
    if l2_norm(vector) == 0.0:
        decision = {
            **decision,
            "decision": "unavailable",
            "freeze": False,
            "reason": "zero direction",
        }
    report = {
        "sweep_id": sweep_config["sweep_id"],
        "sweep_sha256": sweep_config["sha256"],
        "protocol_sha256": protocol["sha256"],
        "direction_sha256": direction["content_sha256"],
        "canonical_manifest_sha256": authority["sha256"],
        "arms": {
            "base_no_intervention": _public_rows(base),
            "alpha_zero_with_wrapper": _public_rows(alpha_zero),
            "random_norm_matched_seed_7": _public_rows(random_arm),
            "candidate_alpha_1": _public_rows(candidate),
        },
        "alpha_zero_matches_base": True,
        "random_control": {
            "seed": 7,
            "match": "l2_norm",
            "extracted_l2": l2_norm(vector),
            "random_l2": l2_norm(random_vector),
            "norm_gap": abs(l2_norm(random_vector) - l2_norm(vector)),
        },
        "candidate_residual_shift": candidate_shift,
        "check_example": {
            "sample_id": prompts[0]["sample_id"],
            "token_ids": _tokens(model, prompts[0], artifact_kind=artifact_kind),
            "base_residual": base[0]["residual"],
            "candidate_residual": candidate[0]["residual"],
        },
        "decision": decision,
        "freeze_allowed": bool(decision.get("freeze")),
        "weight_seed": pins["weight_seed"],
        "n_layers": pins["n_layers"],
        "d_model": pins["d_model"],
        "vocab": pins["vocab"],
        "layer_index": layer,
        "disclaimer": protocol["disclaimer"],
    }
    _stamp_common(report, artifact_kind=artifact_kind, model_pins=pins)
    report["content_sha256"] = content_sha256(report)
    write_json(out_path, report)
    return report


def _public_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "hs_label": row["hs_label"],
            "greedy_token": row["greedy_token"],
            "hook_count": row["hook_count"],
        }
        for row in rows
    ]


def _load_judgments(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RepairError("missing judgments", stage="selection")
    try:
        payload = strict_json_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepairError("judgment file is unreadable", stage="selection") from exc
    if not isinstance(payload, dict):
        raise RepairError("judgment file is unreadable", stage="selection")
    assert_activations_are_not_prompts(payload)
    return payload
