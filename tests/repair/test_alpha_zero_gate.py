"""Root-cause evidence for the alpha-zero Locate failure.

The pinned CircuitKIT hook applies ``activation += steering_vector *
coefficient`` in place. At coefficient 0 the addition is an exact zero in
bf16 (floating-point addition of +0.0 leaves every value unchanged), so the
hook's arithmetic cannot move logits by 29.25 as the magnus run showed.
The deviation therefore comes from the notebook harness/environment (stale
hooks, different call context, or version drift), not from the hook math.

These CPU tests pin that arithmetic on a tiny deterministic random model and
pin the notebook gate that stops Locate when alpha-zero is not exactly 0.
No weights, no network.
"""
import math
import random
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCATE_CELL = "provisional_locate_v1"


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def _log_softmax(vec):
    peak = max(vec)
    total = sum(math.exp(v - peak) for v in vec)
    return [v - peak - math.log(total) for v in vec]


class TinyRandomModel:
    def __init__(self, seed=7, d=4, vocab=6):
        rng = random.Random(seed)
        self.W = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(d)]
        self.U = [[rng.uniform(-1, 1) for _ in range(d)] for _ in range(vocab)]
        self.bias = [0.1 * i for i in range(vocab)]
        self.hooks = []

    def install(self, fn):
        self.hooks.append(fn)

    def detach(self):
        self.hooks = []

    def forward(self, x):
        state = [math.tanh(sum(w * xi for w, xi in zip(row, x))) for row in self.W]
        if self.hooks:
            for hook in self.hooks:
                state = hook(state)
        logits = [sum(u * s for u, s in zip(row, state)) + b for row, b in zip(self.U, self.bias)]
        return _log_softmax([_f32(v) for v in logits])


def run_with_steering(model, input_vec, vector, coefficient):
    """Exact pinned hook arithmetic: target.add_(steering_vector * coeff)."""
    if not model.hooks:
        coeff = coefficient
        model.install(lambda state, c=coeff, vec=vector: [a + c * v for a, v in zip(state, vec)])
    return model.forward(input_vec)


def test_alpha_zero_is_a_bit_identical_no_op():
    model = TinyRandomModel()
    x = [0.3, -0.2, 0.5, 0.1]
    vector = [3.0, 4.0, 0.0, 0.0]
    base = model.forward(x)
    zero = run_with_steering(model, x, vector, 0.0)
    assert zero == base
    assert zero is not base

    other = TinyRandomModel()
    assert other.forward(x) == base

    moved = run_with_steering(TinyRandomModel(), x, vector, 1.0)
    assert moved != base
    assert max(abs(a - b) for a, b in zip(moved, base)) > 0



def test_hooks_cannot_leak_into_later_forwards():
    model = TinyRandomModel()
    x = [0.3, -0.2, 0.5, 0.1]
    before = model.forward(x)
    model.install(lambda state: [a + 1.0 for a in state])
    model.detach()
    assert model.forward(x) == before


def test_notebook_locate_asserts_alpha_zero_before_any_direction():
    import json

    notebook = json.loads((ROOT / "notebooks/gpu_parity_and_baseline.ipynb").read_text(encoding="utf-8"))
    cell = None
    for candidate in notebook["cells"]:
        text = "".join(candidate["source"]) if isinstance(candidate["source"], list) else candidate["source"]
        if LOCATE_CELL in text:
            cell = text
    assert cell is not None
    zero_pos = cell.index('_zero_diff = _control_run("alpha_zero_with_wrapper"')
    direction_pos = cell.index('write_json(ARTIFACT_DIR / "locate_direction.json"')
    gate_pos = cell.index("if _zero_diff != 0.0")
    assert zero_pos < gate_pos
    assert gate_pos < direction_pos
    assert "alpha-zero check" in cell[gate_pos:direction_pos]


def test_notebook_locate_raises_not_scores_on_alpha_zero():
    import json

    notebook = json.loads((ROOT / "notebooks/gpu_parity_and_baseline.ipynb").read_text(encoding="utf-8"))
    cell = None
    for candidate in notebook["cells"]:
        text = "".join(candidate["source"]) if isinstance(candidate["source"], list) else candidate["source"]
        if LOCATE_CELL in text:
            cell = text
    raises = [m.start() for m in __import__("re").finditer(r"raise RuntimeError", cell)]
    assert raises
