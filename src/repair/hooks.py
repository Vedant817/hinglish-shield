"""Runtime hook lifecycle. Cleanup runs when the forward fails."""
from __future__ import annotations

from typing import Sequence

from .errors import RepairError
from .fixture_model import RandomWeightFixture
from .vectors import add_scaled


def run_steered(
    model: RandomWeightFixture,
    token_ids: Sequence[int],
    vector: Sequence[float],
    alpha: float,
    layer: int,
) -> tuple[list[float], int]:
    """Install one wrapper hook, run one forward, then remove it."""
    applied = {"n": 0}

    def hook(hook_layer: int, activation: list[float]) -> list[float]:
        if hook_layer != layer:
            raise RepairError("hook layer does not match the frozen direction", stage="runtime")
        applied["n"] += 1
        return add_scaled(activation, vector, alpha)

    try:
        model.add_hook(hook)
        if model.hook_count() < 1:
            raise RepairError("effective hook count is 0", stage="runtime")
        residual = model.forward_last(token_ids, layer=layer, apply_hooks=True)
        if applied["n"] < 1:
            raise RepairError("effective hook count is 0", stage="runtime")
        return residual, applied["n"]
    finally:
        model.reset_hooks()
        if model.hook_count() != 0:
            raise RepairError("hooks survived cleanup", stage="runtime")


def run_base(model: RandomWeightFixture, token_ids: Sequence[int], layer: int) -> list[float]:
    if model.hook_count() != 0:
        raise RepairError("base forward found a leftover hook", stage="runtime")
    return model.forward_last(token_ids, layer=layer, apply_hooks=False)
