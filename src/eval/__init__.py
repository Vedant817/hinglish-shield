"""Eval: baseline / after-eval metrics, judges, and metrics.json I/O.

Submodules are import-safe (no network). Scoring paths raise UnavailableError
until RUNTIME-VERIFIED.
"""

from .judges import (
    UnavailableError,
    JudgeVerdict,
    Judge,
    WildGuardPrimaryJudge,
    LLMSecondaryJudge,
    cohens_kappa,
    kappa_report,
)
from .metrics_io import (
    load_schema,
    load_metrics,
    validate_metrics,
    load_and_validate,
    is_fixture,
)

__all__ = [
    "UnavailableError",
    "JudgeVerdict",
    "Judge",
    "WildGuardPrimaryJudge",
    "LLMSecondaryJudge",
    "cohens_kappa",
    "kappa_report",
    "load_schema",
    "load_metrics",
    "validate_metrics",
    "load_and_validate",
    "is_fixture",
]
