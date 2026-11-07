"""Deterministic tensor stand-in for hook tests.

This module does not load pretrained weights. Its activations are a pure
function of the last token id. It is not Tiny Aya and it is not a parity pass.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .errors import RepairError

Hook = Callable[[Sequence[float], Any], Sequence[float] | None]


class DeterministicTensorFixture:
    """List-valued residual with TransformerLens-style hook names."""

    uses_tensors = False
    d_model = 4
    vocab = 8

    def __init__(self) -> None:
        self.hooks: list[tuple[str, Hook, str]] = []

    def add_hook(self, name: str, fn: Hook, dir: str = "fwd") -> None:
        if dir != "fwd":
            raise RepairError("hook direction is not fwd", stage="runtime")
        if not isinstance(name, str) or name == "":
            raise RepairError("hook layer is missing", stage="runtime")
        self.hooks.append((name, fn, dir))

    def reset_hooks(self) -> None:
        self.hooks.clear()

    def hook_count(self) -> int:
        return len(self.hooks)

    def step(self, token_ids: Sequence[int]) -> list[float]:
        if not token_ids:
            raise RepairError("missing token ids", stage="runtime")
        last = int(token_ids[-1])
        activation = [float((last + index) % 5) for index in range(self.d_model)]
        for _name, fn, _direction in list(self.hooks):
            updated = fn(activation, None)
            if updated is not None:
                activation = [float(value) for value in updated]
        logits = [0.0] * self.vocab
        logits[1] = 10.0
        logits[3] = float(activation[0])
        return logits
