"""Small-vector math for the proposed difference of means. Pure Python."""
from __future__ import annotations

import math
import random
from typing import Sequence

from .errors import RepairError


def as_vector(values: Sequence[float], *, label: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise RepairError(f"missing {label}")
    vector: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RepairError(f"nonfinite {label}")
        number = float(value)
        if not math.isfinite(number):
            raise RepairError(f"nonfinite {label}")
        vector.append(number)
    return vector


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def mean_vector(rows: Sequence[Sequence[float]]) -> list[float]:
    if not rows:
        raise RepairError("missing calibration class")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise RepairError("activation width mismatch")
    totals = [0.0] * width
    for row in rows:
        for index, value in enumerate(row):
            totals[index] += value
    scale = 1.0 / len(rows)
    return [value * scale for value in totals]


def difference_of_means(target_rows: Sequence[Sequence[float]], source_rows: Sequence[Sequence[float]]) -> list[float]:
    target = mean_vector(target_rows)
    source = mean_vector(source_rows)
    return [left - right for left, right in zip(target, source)]


def random_norm_matched(vector: Sequence[float], *, seed: int) -> list[float]:
    if type(seed) is not int:
        raise RepairError("random control seed must be an integer")
    target = l2_norm(vector)
    if target == 0.0:
        raise RepairError("random control refuses a zero direction")
    rng = random.Random(seed)
    draw = [rng.gauss(0.0, 1.0) for _ in vector]
    scale = l2_norm(draw)
    if scale == 0.0:
        raise RepairError("random control draw has zero norm")
    return [value * target / scale for value in draw]


def max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise RepairError("activation width mismatch")
    return max(abs(a - b) for a, b in zip(left, right))


def add_scaled(activation: Sequence[float], vector: Sequence[float], alpha: float) -> list[float]:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)):
        raise RepairError("steering alpha is not finite")
    if len(activation) != len(vector):
        raise RepairError("activation width mismatch")
    coefficient = float(alpha)
    return [left + coefficient * right for left, right in zip(activation, vector)]
