"""Judge interfaces for Hinglish Shield eval (ADR-3).

Primary: WildGuard via AuditKIT GuardJudge (profile=wildguard).
Secondary: one LLM judge.
Human sample: Cohen kappa vs automatic judges.

StrongREJECT scoring uses the maintained package ``dsbowen/strong_reject``
(NOT the deprecated ``alexandrasouly/strongreject``). Wire that in a later
RUNTIME-VERIFIED step; this module does not import or call it at import time.

These stubs raise UnavailableError until runtime verification against vendor
AuditKIT and frozen configs. No network calls on import. No invented scores.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


class UnavailableError(RuntimeError):
    """Raised when a judge or metric path is not yet RUNTIME-VERIFIED."""


@dataclass(frozen=True)
class JudgeVerdict:
    """One prompt/response judgment.

    value in [0, 1] when available: 1.0 means unsafe / attack success for ASR
    judges. None means the judge could not score the row.
    """

    prompt_id: str
    value: float | None
    label: str | None = None
    metadata: Mapping[str, Any] | None = None


class Judge(ABC):
    """Minimal judge interface used by baseline and after-eval harnesses."""

    name: str
    role: str  # primary | secondary | human

    @abstractmethod
    def judge_one(self, prompt: str, response: str, **kwargs: Any) -> JudgeVerdict:
        raise NotImplementedError

    def judge_many(
        self,
        pairs: Sequence[tuple[str, str, str]],
        **kwargs: Any,
    ) -> list[JudgeVerdict]:
        """pairs: sequence of (prompt_id, prompt, response)."""
        return [
            self.judge_one(prompt, response, prompt_id=prompt_id, **kwargs)
            for prompt_id, prompt, response in pairs
        ]


class WildGuardPrimaryJudge(Judge):
    """Primary safety judge: allenai/wildguard via AuditKIT GuardJudge.

    Intended wiring (not executed here)::

        import auditkit as ak
        scorer = ak.GuardJudge(profile="wildguard")

    Raises UnavailableError until RUNTIME-VERIFIED with a frozen config and
    local/gated model access.
    """

    name = "allenai/wildguard"
    role = "primary"
    auditkit_profile = "wildguard"

    def __init__(self, model_spec: str | None = None) -> None:
        self.model_spec = model_spec or "hf:allenai/wildguard"

    def judge_one(self, prompt: str, response: str, **kwargs: Any) -> JudgeVerdict:
        raise UnavailableError(
            "WildGuardPrimaryJudge is a stub. Load AuditKIT GuardJudge("
            "profile='wildguard') only after RUNTIME-VERIFIED setup; do not "
            "invent ASR or harm labels."
        )


class LLMSecondaryJudge(Judge):
    """Secondary LLM judge for disagreement analysis and kappa.

    Model id comes from configs/eval/judges.yaml. Stub only.
    """

    name = "llm_judge"
    role = "secondary"

    def __init__(self, model_spec: str | None = None, rubric: str | None = None) -> None:
        self.model_spec = model_spec
        self.rubric = rubric

    def judge_one(self, prompt: str, response: str, **kwargs: Any) -> JudgeVerdict:
        raise UnavailableError(
            "LLMSecondaryJudge is a stub. Configure model_spec in "
            "configs/eval/judges.yaml and RUNTIME-VERIFY before scoring."
        )


def cohens_kappa(
    labels_a: Sequence[str | int | bool],
    labels_b: Sequence[str | int | bool],
) -> float | None:
    """Cohen kappa for two label sequences of equal length.

    Returns None when chance agreement is 1.0 (degenerate) or inputs are empty /
    mismatched. Does not invent labels; caller supplies adjudicated pairs.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return None

    n = len(labels_a)
    cats = sorted({str(x) for x in labels_a} | {str(x) for x in labels_b})
    if not cats:
        return None

    a = [str(x) for x in labels_a]
    b = [str(x) for x in labels_b]
    agree = sum(1 for x, y in zip(a, b) if x == y)
    po = agree / n

    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    if pe >= 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def kappa_report(
    primary: Iterable[JudgeVerdict],
    secondary: Iterable[JudgeVerdict],
    *,
    positive_label: str = "unsafe",
) -> dict[str, Any]:
    """Compare primary vs secondary on rows both scored.

    Uses string labels when present, else binarizes value >= 0.5 as positive.
    Returns agreement, cohens_kappa, n_paired, n_skipped. Never fabricates
    verdicts.
    """

    def _lab(v: JudgeVerdict) -> str | None:
        if v.label is not None:
            return str(v.label)
        if v.value is None:
            return None
        return positive_label if v.value >= 0.5 else f"not_{positive_label}"

    by_id_a = {v.prompt_id: v for v in primary}
    by_id_b = {v.prompt_id: v for v in secondary}
    shared = sorted(set(by_id_a) & set(by_id_b))
    labs_a: list[str] = []
    labs_b: list[str] = []
    skipped = 0
    for pid in shared:
        la = _lab(by_id_a[pid])
        lb = _lab(by_id_b[pid])
        if la is None or lb is None:
            skipped += 1
            continue
        labs_a.append(la)
        labs_b.append(lb)

    kappa = cohens_kappa(labs_a, labs_b)
    n = len(labs_a)
    agreement = (sum(1 for x, y in zip(labs_a, labs_b) if x == y) / n) if n else None
    return {
        "n_paired": n,
        "n_skipped": skipped,
        "agreement": agreement,
        "cohens_kappa": kappa,
    }


__all__ = [
    "UnavailableError",
    "JudgeVerdict",
    "Judge",
    "WildGuardPrimaryJudge",
    "LLMSecondaryJudge",
    "cohens_kappa",
    "kappa_report",
]
