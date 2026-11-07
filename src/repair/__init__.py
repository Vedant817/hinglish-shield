"""Calibration and dev Locate plus steering repair.

Fixture runs use a random-weight model. They are not Tiny Aya results and
they are not WildGuard results. A difference of means recorded here is a
proposed protocol, not a proven refusal mechanism.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path


def _allow_eval_submodules_without_metrics_io() -> None:
    """The torch contract environment has no jsonschema.

    src.eval imports metrics_io at package import. That import is not needed
    to load a repair submodule. If it fails only because jsonschema is absent,
    register the package path so dev_runner, judges, and stats can still load.
    """
    if "src.eval" in sys.modules:
        return
    try:
        import src.eval  # noqa: F401
    except ImportError as exc:
        if "jsonschema" not in str(exc):
            raise
        sys.modules.pop("src.eval", None)
        package = types.ModuleType("src.eval")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "eval")]
        package.__package__ = "src.eval"
        sys.modules["src.eval"] = package


_allow_eval_submodules_without_metrics_io()
