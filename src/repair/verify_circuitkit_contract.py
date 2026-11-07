"""Runtime-check ActivationSteering on a tiny random module, not Tiny Aya.

Run with an external interpreter that has torch and the pinned CircuitKIT
source on PYTHONPATH. A missing library is an unverified gate, not a pass.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

SOURCE_SHA = "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78"
DEFAULT_SOURCE = Path(r"C:\Users\vedan\Documents\Resources\Code\hinglish-shield-scratch\CircuitKIT-source")


def _source_sha(path: Path) -> str | None:
    if not path.is_dir():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _fail(report: dict) -> int:
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args(argv)
    head = _source_sha(args.source)
    report: dict = {
        "runtime_verified": False,
        "source_path": str(args.source),
        "source_head": head,
        "expected_sha": SOURCE_SHA,
        "model": "in-process random tensors",
        "not_tiny_aya": True,
        "not_wildguard": True,
        "weights_downloaded": False,
    }
    if head != SOURCE_SHA:
        report["reason"] = "CircuitKIT source SHA does not match the pin"
        return _fail(report)
    src = args.source / "src"
    sys.path.insert(0, str(src))
    try:
        import torch
    except ImportError as exc:
        report["reason"] = f"torch is not installed: {exc}"
        return _fail(report)
    try:
        module = importlib.import_module("circuitkit.applications.steering.steering")
    except Exception as exc:
        report["reason"] = f"ActivationSteering import failed: {type(exc).__name__}: {exc}"
        return _fail(report)
    steering_cls = module.ActivationSteering
    import inspect

    extract_params = inspect.signature(steering_cls.compute_steering_vector).parameters
    steer_params = inspect.signature(steering_cls.steer).parameters
    random_params = inspect.signature(steering_cls.get_random_baseline_vectors).parameters
    if "source_examples" not in extract_params or "target_examples" not in extract_params:
        report["reason"] = "compute_steering_vector signature does not match the pinned source"
        return _fail(report)
    if "return_all_positions" not in extract_params or "coefficient" not in steer_params:
        report["reason"] = "steering signature does not match the pinned source"
        return _fail(report)
    if "seed" in random_params:
        report["reason"] = "get_random_baseline_vectors grew a seed parameter; re-check the source"
        return _fail(report)

    class Config:
        device = "cpu"
        use_attn_result = False

    class RandomModule:
        """Enough of the HookedTransformer surface for ActivationSteering."""

        def __init__(self) -> None:
            self.cfg = Config()
            self.hooks: list = []
            self.calls = 0

        def set_use_attn_result(self, enabled: bool) -> None:
            self.cfg.use_attn_result = enabled

        def add_hook(self, name: str, fn, dir: str = "fwd") -> None:
            self.hooks.append((name, fn, dir))

        def reset_hooks(self) -> None:
            self.hooks.clear()

        def __call__(self, input_ids):
            self.calls += 1
            batch, seq = input_ids.shape
            scale = input_ids.to(dtype=torch.float32).mean()
            activation = torch.ones(batch, seq, 2, 4) * scale
            for _name, fn, _direction in list(self.hooks):
                # TransformerLens treats a None hook return as "leave the
                # activation in place". CircuitKIT's collect hook does that.
                updated = fn(activation, None)
                if updated is not None:
                    activation = updated
            self.last = activation.detach().clone()
            return torch.zeros(batch, seq, 3)

    import logging

    logging.disable(logging.INFO)
    model = RandomModule()
    scores = {"A0.0": 1.0}
    source = [{"input_ids": [1, 1, 1, 1]}]
    target = [{"input_ids": [3, 3, 3, 3]}]
    from src.repair.circuitkit_backend import extract_last_token, steer_coefficient

    credit = {"credit": True, "reason": "contract-script-only"}
    try:
        extracted = extract_last_token(model, scores, source, target, credit=credit)
    except Exception as exc:
        report["reason"] = f"extract_last_token failed: {type(exc).__name__}: {exc}"
        return _fail(report)
    vector = extracted["vectors"]["A0.0"]
    if model.hooks:
        report["reason"] = "extract left hooks installed"
        return _fail(report)
    if float(torch.linalg.vector_norm(vector)) == 0.0:
        report["reason"] = "random contract vector norm is zero"
        return _fail(report)
    full = {name: tensor.unsqueeze(0).repeat(4, 1) for name, tensor in extracted["vectors"].items()}
    # Rank-1 last-token slice is what we keep. steer expects the tensor the
    # library would add. Give it the extracted last-token vector directly.
    probe = torch.tensor([[3, 3, 3, 3]])
    try:
        bare = model(probe)
        bare_last = model.last.detach().clone()
        if model.hooks:
            report["reason"] = "unhooked forward left hooks installed"
            return _fail(report)
        steered = steer_coefficient(
            model,
            scores,
            {"A0.0": vector},
            probe,
            1.0,
            credit=credit,
        )
        alpha_one_last = model.last.detach().clone()
        zero = steer_coefficient(
            model,
            scores,
            {"A0.0": vector},
            probe,
            0.0,
            credit=credit,
        )
        alpha_zero_last = model.last.detach().clone()
    except Exception as exc:
        report["reason"] = f"steer_coefficient failed: {type(exc).__name__}: {exc}"
        return _fail(report)
    del bare
    if model.hooks:
        report["reason"] = "steer left hooks installed"
        return _fail(report)
    if steered["effective_hook_count"] < 1 or zero["effective_hook_count"] < 1:
        report["reason"] = "effective hook count is 0"
        return _fail(report)
    alpha_zero_diff = float((alpha_zero_last - bare_last).abs().max())
    alpha_one_diff = float((alpha_one_last - bare_last).abs().max())
    if alpha_zero_diff != 0.0:
        report["reason"] = "alpha 0 residual does not match the unhooked forward"
        report["alpha_zero_max_abs_diff"] = alpha_zero_diff
        return _fail(report)
    if alpha_one_diff == 0.0:
        report["reason"] = "alpha 1 did not change the residual"
        return _fail(report)
    report.update(
        {
            "runtime_verified": True,
            "reason": None,
            "credit_scope": "contract-script-only synthetic credit, not Locate credit",
            "alpha_zero_max_abs_diff": alpha_zero_diff,
            "alpha_one_max_abs_diff": alpha_one_diff,
            "calls": [
                "ActivationSteering.compute_steering_vector",
                "ActivationSteering.steer",
            ],
            "nodes": extracted["nodes"],
            "vector_norm": float(torch.linalg.vector_norm(vector)),
            "alpha_zero_hook_count": zero["effective_hook_count"],
            "alpha_one_hook_count": steered["effective_hook_count"],
            "random_baseline_accepts_seed": False,
            "proven_refusal_mechanism": False,
            "unused_rank2_placeholder": {key: list(value.shape) for key, value in full.items()},
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
