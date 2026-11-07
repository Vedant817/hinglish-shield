"""Eval: baseline / after-eval metrics, judges, stats, and metrics.json I/O.

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
    shared_binary_label,
    secondary_binary_agreement,
)
from .metrics_io import (
    load_schema,
    load_metrics,
    validate_metrics,
    load_and_validate,
    is_fixture,
)
from .stats import (
    WILSON_Z_95,
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    RESPONSE_CATEGORIES,
    WilsonInterval,
    BootstrapDelta,
    AsrResult,
    wilson_ci,
    asr_from_counts,
    family_bootstrap_delta,
    mean_gap,
)

__all__ = [
    "UnavailableError",
    "JudgeVerdict",
    "Judge",
    "WildGuardPrimaryJudge",
    "LLMSecondaryJudge",
    "cohens_kappa",
    "kappa_report",
    "shared_binary_label",
    "secondary_binary_agreement",
    "load_schema",
    "load_metrics",
    "validate_metrics",
    "load_and_validate",
    "is_fixture",
    "WILSON_Z_95",
    "DEFAULT_BOOTSTRAP_REPS",
    "DEFAULT_BOOTSTRAP_SEED",
    "RESPONSE_CATEGORIES",
    "WilsonInterval",
    "BootstrapDelta",
    "AsrResult",
    "wilson_ci",
    "asr_from_counts",
    "family_bootstrap_delta",
    "mean_gap",
]
