"""Calibration-only direction record. Frozen before any dev read."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .errors import RepairError
from .jsonutil import content_sha256
from .vectors import as_vector, difference_of_means, l2_norm


def extract_direction(
    records: Sequence[Mapping[str, Any]],
    *,
    layer_index: int,
    layer_name: str,
    position: str,
    target_label: str,
    source_label: str,
    protocol_sha256: str,
    protocol_id: str,
) -> dict[str, Any]:
    if not records:
        raise RepairError("missing calibration activations", stage="direction")
    targets: list[list[float]] = []
    sources: list[list[float]] = []
    sample_ids: list[str] = []
    for record in records:
        if record.get("split") != "calibration":
            raise RepairError("direction extraction refuses a non-calibration row", stage="direction")
        if record.get("layer_index") != layer_index or record.get("position") != position:
            raise RepairError("direction site does not match the proposed protocol", stage="direction")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id == "":
            raise RepairError("missing calibration id", stage="direction")
        vector = as_vector(record.get("vector"), label="activation")
        label = record.get("hs_label")
        if label == target_label:
            targets.append(vector)
        elif label == source_label:
            sources.append(vector)
        else:
            raise RepairError("missing hs_label", stage="direction")
        sample_ids.append(sample_id)
    if not targets or not sources:
        raise RepairError("missing calibration class", stage="direction")
    direction = difference_of_means(targets, sources)
    payload: dict[str, Any] = {
        "frozen": True,
        "proven_refusal_mechanism": False,
        "disclaimer": (
            "This vector is a proposed difference of means. It is not a preregistered "
            "refusal mechanism and it is not evidence that the direction mediates refusal."
        ),
        "convention": "mean_target_minus_mean_source",
        "target_hs_label": target_label,
        "source_hs_label": source_label,
        "layer_index": layer_index,
        "layer_name": layer_name,
        "position": position,
        "vector_norm_kind": "raw_l2",
        "l2_norm": l2_norm(direction),
        "vector": direction,
        "calibration_ids": sorted(sample_ids),
        "n_target": len(targets),
        "n_source": len(sources),
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "unit_normalized": False,
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def assert_frozen_direction(payload: Mapping[str, Any], *, protocol_sha256: str) -> dict[str, Any]:
    if payload.get("frozen") is not True:
        raise RepairError("direction is not frozen", stage="direction")
    if payload.get("proven_refusal_mechanism") is not False:
        raise RepairError("direction claims a proven refusal mechanism", stage="direction")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise RepairError("direction protocol hash does not match the current protocol", stage="direction")
    if payload.get("convention") != "mean_target_minus_mean_source":
        raise RepairError("direction convention does not match the proposed protocol", stage="direction")
    if payload.get("position") != "last_nonpad_token":
        raise RepairError("direction token position does not match the proposed protocol", stage="direction")
    vector = as_vector(payload.get("vector"), label="direction")
    expected = content_sha256(dict(payload))
    if payload.get("content_sha256") != expected:
        raise RepairError("direction artifact hash does not match its contents", stage="direction")
    ids = payload.get("calibration_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or item == "" for item in ids):
        raise RepairError("direction is missing calibration ids", stage="direction")
    checked = dict(payload)
    checked["vector"] = vector
    return checked
