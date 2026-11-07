"""Calls that exist on CircuitKIT ActivationSteering at SHA e71d5e17.

Source checked in circuitkit/applications/steering/steering.py:
  ActivationSteering.__init__(model, circuit_scores, score_threshold=0.0)
  compute_steering_vector(source_examples, target_examples, batch_size=32, return_all_positions=False)
  steer(inputs, steering_vectors=None, coefficient=1.0, layer_weights=None)
  get_random_baseline_vectors() takes no seed and is not the prereg seed 7 control.

Nothing in this module is a Tiny Aya result. Import happens only after credit.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .errors import RepairError

SOURCE_SHA = "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78"


def _require_credit(credit: Mapping[str, Any]) -> None:
    if credit.get("credit") is not True:
        raise RepairError(
            "CircuitKIT locate refused: parity credit is missing, failed, or incomplete",
            stage="parity",
        )


def _load_class():
    from circuitkit.applications.steering.steering import ActivationSteering

    return ActivationSteering


def _reset(model: Any) -> None:
    reset = getattr(model, "reset_hooks", None)
    if callable(reset):
        reset()


def extract_last_token(
    model: Any,
    circuit_scores: Mapping[str, float],
    source_examples: Sequence[Mapping[str, Any]],
    target_examples: Sequence[Mapping[str, Any]],
    *,
    credit: Mapping[str, Any],
) -> dict[str, Any]:
    """Real compute_steering_vector call. Refuses an already-averaged vector."""
    _require_credit(credit)
    if not circuit_scores:
        raise RepairError("CircuitKIT steering needs caller-supplied node scores", stage="circuitkit")
    if not source_examples or not target_examples:
        raise RepairError("CircuitKIT steering needs source and target examples", stage="circuitkit")
    activation_steering = _load_class()
    steering = activation_steering(model, dict(circuit_scores), score_threshold=0.0)
    try:
        vectors = steering.compute_steering_vector(
            list(source_examples),
            list(target_examples),
            batch_size=1,
            return_all_positions=True,
        )
    finally:
        _reset(model)
    if not vectors:
        raise RepairError("CircuitKIT returned no steering vectors", stage="circuitkit")
    reduced: dict[str, Any] = {}
    for name, tensor in vectors.items():
        if getattr(tensor, "dim", lambda: 0)() != 2:
            raise RepairError(
                "CircuitKIT vector is not position-resolved; refusing an averaged vector",
                stage="circuitkit",
            )
        reduced[name] = tensor[-1]
    return {
        "source_sha": SOURCE_SHA,
        "call": "ActivationSteering.compute_steering_vector",
        "return_all_positions": True,
        "position_reduction": "last_index",
        "nodes": sorted(reduced),
        "vectors": reduced,
        "proven_refusal_mechanism": False,
        "not_tiny_aya": True,
    }


def steer_coefficient(
    model: Any,
    circuit_scores: Mapping[str, float],
    steering_vectors: Mapping[str, Any],
    inputs: Any,
    coefficient: float,
    *,
    credit: Mapping[str, Any],
) -> dict[str, Any]:
    """Real steer call. The wrapper removes hooks again if the library leaks one."""
    _require_credit(credit)
    if not steering_vectors:
        raise RepairError("CircuitKIT steer needs vectors", stage="circuitkit")
    activation_steering = _load_class()
    steering = activation_steering(model, dict(circuit_scores), score_threshold=0.0)
    try:
        result = steering.steer(inputs, dict(steering_vectors), coefficient=coefficient)
    finally:
        _reset(model)
    nodes = result.get("steered_nodes") if isinstance(result, dict) else None
    if not isinstance(nodes, list) or len(nodes) < 1:
        raise RepairError("effective hook count is 0", stage="runtime")
    if getattr(model, "hook_count", None) is not None and model.hook_count() != 0:
        raise RepairError("hooks survived cleanup", stage="runtime")
    hooks = getattr(model, "hooks", None)
    if isinstance(hooks, list) and len(hooks) != 0:
        raise RepairError("hooks survived cleanup", stage="runtime")
    return {
        "source_sha": SOURCE_SHA,
        "call": "ActivationSteering.steer",
        "coefficient": coefficient,
        "steered_nodes": list(nodes),
        "effective_hook_count": len(nodes),
        "output": result.get("output") if isinstance(result, dict) else None,
        "proven_refusal_mechanism": False,
        "not_tiny_aya": True,
    }
