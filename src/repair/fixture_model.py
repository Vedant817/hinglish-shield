"""Random-weight residual fixture.

This model is not Tiny Aya. Its forwards are not WildGuard results and they
are not scored safety metrics.
"""
from __future__ import annotations

import hashlib
import random
from typing import Callable, Sequence

from .errors import RepairError
from .jsonutil import canonical_dumps, sha256_text

MODEL_ID = "fixture/random-weight"
TINY_AYA_ID = "CohereLabs/tiny-aya-global"
Hook = Callable[[int, list[float]], list[float]]


class RandomWeightFixture:
    """One residual stream, fixed seed, explicit runtime hooks."""

    def __init__(
        self,
        *,
        seed: int = 42,
        n_layers: int = 2,
        d_model: int = 8,
        vocab: int = 32,
    ) -> None:
        if type(seed) is not int or seed < 0:
            raise RepairError("fixture weight seed is unpinned")
        if type(n_layers) is not int or n_layers < 1:
            raise RepairError("fixture depth is unpinned")
        if type(d_model) is not int or d_model < 1:
            raise RepairError("fixture width is unpinned")
        if type(vocab) is not int or vocab < 2:
            raise RepairError("fixture vocab is unpinned")
        self.model_id = MODEL_ID
        self.artifact_kind = "fixture"
        self.dtype = "float32"
        self.seed = seed
        self.n_layers = n_layers
        self.d_model = d_model
        self.vocab = vocab
        rng = random.Random(seed)
        self.embed = [[rng.random() * 2.0 - 1.0 for _ in range(d_model)] for _ in range(vocab)]
        self.layers = [
            [[rng.random() * 2.0 - 1.0 for _ in range(d_model)] for _ in range(d_model)]
            for _ in range(n_layers)
        ]
        self.unembed = [[rng.random() * 2.0 - 1.0 for _ in range(vocab)] for _ in range(d_model)]
        self._hooks: list[Hook] = []

    def hook_count(self) -> int:
        return len(self._hooks)

    def add_hook(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def reset_hooks(self) -> None:
        self._hooks.clear()

    def weight_fingerprint(self) -> str:
        payload = {"embed": self.embed, "layers": self.layers, "unembed": self.unembed}
        return sha256_text(canonical_dumps(payload))

    def tokens_for_prompt(self, prompt: str) -> list[int]:
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        return [digest[index] % self.vocab for index in range(4)]

    def forward_last(
        self,
        token_ids: Sequence[int],
        *,
        layer: int,
        apply_hooks: bool,
    ) -> list[float]:
        if not token_ids:
            raise RepairError("missing tokens", stage="runtime")
        if type(layer) is not int or layer < 0 or layer >= self.n_layers:
            raise RepairError("steering layer is outside the fixture", stage="runtime")
        for token_id in token_ids:
            if type(token_id) is not int or token_id < 0 or token_id >= self.vocab:
                raise RepairError("fixture token is outside the vocab", stage="runtime")
        hidden = list(self.embed[token_ids[-1]])
        for layer_index, matrix in enumerate(self.layers):
            mixed = _matvec(matrix, hidden)
            hidden = [left + right for left, right in zip(hidden, mixed)]
            if layer_index == layer:
                if apply_hooks:
                    for hook in list(self._hooks):
                        updated = hook(layer_index, hidden)
                        if not isinstance(updated, list) or len(updated) != self.d_model:
                            raise RepairError("hook returned a bad activation", stage="runtime")
                        hidden = updated
                break
        return hidden

    def greedy_token(self, residual: Sequence[float]) -> int:
        if len(residual) != self.d_model:
            raise RepairError("activation width mismatch", stage="runtime")
        logits = [0.0] * self.vocab
        for dim, value in enumerate(residual):
            row = self.unembed[dim]
            for token_id, weight in enumerate(row):
                logits[token_id] += value * weight
        return max(range(self.vocab), key=lambda token_id: (logits[token_id], -token_id))


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]
