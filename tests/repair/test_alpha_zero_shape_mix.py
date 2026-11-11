"""CPU regression test for the notebook base-vs-control indexing mix.

Base logits were taken with ``[-1]`` (last batch row: seq x vocab) while
controls used ``[0, -1]`` (last-position row). The max-abs difference of
those two shapes is on the scale of a full steering effect (20-30 in the
v15 run), so a real 0-coefficient pass still produced a nonzero
max_abs_logit_diff and the gate halted Locate. The pinned hook arithmetic
itself (``target.add_(vector * coeff)``) is bit-identical at 0. This test
pins both facts on a tiny random model with no weights.
"""
import math
import random
import struct


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _log_softmax(vec):
    peak = max(vec)
    total = sum(math.exp(v - peak) for v in vec)
    return [v - peak - math.log(total) for v in vec]


class TinyRandomModel:
    def __init__(self, seed=11, d=4, vocab=6):
        rng = random.Random(seed)
        self.W = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(d)]
        self.U = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(vocab)]
        self.bias = [0.05 * i for i in range(vocab)]
        self.hooks = []

    def install(self, fn):
        self.hooks.append(fn)

    def detach(self):
        self.hooks = []

    def forward(self, x, with_hook=True):
        state = [math.tanh(sum(w * xi for w, xi in zip(row, x))) for row in self.W]
        if with_hook:
            for hook in self.hooks:
                state = hook(state)
        logits = [sum(u * s for u, s in zip(row, state)) + b for row, b in zip(self.U, self.bias)]
        return [_f32(v) for v in logits]


def _batched_logits(model, x):
    """HookedTransformer-style logits: [1, 1, vocab] on a single-row batch."""
    row = model.forward(x)
    return [row]


def test_mixed_fetch_sees_steering_on_the_wrong_axis():
    """Notebook base used [-1] (whole batch row) while controls used [0, -1]."""
    # On a [1, seq, vocab] tensor, [-1] selects the whole last row
    # (seq x vocab), while [0, -1] selects the last-position vector.
    seq = 5
    logits = [[[float((i * 7 + j) % 13) for j in range(3)] for i in range(seq)]]
    base_mixed = logits[-1]
    control_row = logits[0][-1]
    assert base_mixed is not control_row
    assert len(base_mixed) == seq
    assert len(control_row) == 3

    model = TinyRandomModel()
    x = [0.3, -0.2, 0.5, 0.1]
    base = model.forward(x)
    other = TinyRandomModel()
    other.install(lambda state: [a + 0.0 * v for a, v in zip(state, [3.0, 4.0, 0.0, 0.0])])
    assert other.forward(x) == base


def test_notebook_base_fetch_uses_control_row_shape():
    import json
    from pathlib import Path

    notebook = json.loads((Path("notebooks/gpu_parity_and_baseline.ipynb")).read_text(encoding="utf-8"))
    cell = None
    for candidate in notebook["cells"]:
        text = "".join(candidate["source"]) if isinstance(candidate["source"], list) else candidate["source"]
        if "def write_locate_diagnostics" in text:
            cell = text
    assert cell is not None
    assert "cpu()[0, -1])" in cell
    assert "cpu()[-1])" not in cell
    assert cell.index("cpu()[0, -1])") < cell.index("_zero_diff = ")


def test_alpha_zero_stays_bit_identical_and_candidate_moves():
    model = TinyRandomModel()
    x = [0.3, -0.2, 0.5, 0.1]
    vector = [3.0, 4.0, 0.0, 0.0]
    base = model.forward(x)
    model.install(lambda state: [a + 0.0 * v for a, v in zip(state, vector)])
    zero_logits = model.forward(x)
    assert zero_logits == base

    other = TinyRandomModel()
    other.install(lambda state: [a + 1.0 * v for a, v in zip(state, vector)])
    moved = other.forward(x)
    assert moved != base
    assert max(abs(a - b) for a, b in zip(moved, base)) > 0


def test_log_softmax_diff_tracks_logits_diff():
    model = TinyRandomModel()
    x = [0.3, -0.2, 0.5, 0.1]
    base = model.forward(x)
    other = TinyRandomModel()
    other.install(lambda state: [a + 1.0 * v for a, v in zip(state, [3.0, 4.0, 0.0, 0.0])])
    moved = other.forward(x)
    logp_diff = max(abs(a - b) for a, b in zip(_log_softmax(moved), _log_softmax(base)))
    assert logp_diff > 0
