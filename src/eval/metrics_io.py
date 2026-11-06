"""Load and validate runs/*/metrics.json against schemas/metrics.schema.json."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError as exc:  # pragma: no cover - smoke installs jsonschema
    raise ImportError(
        "jsonschema is required for metrics_io; run: pip install -r requirements.txt"
    ) from exc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = ROOT / "schemas" / "metrics.schema.json"


def load_schema(schema_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(schema_path) if schema_path else DEFAULT_SCHEMA
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"metrics file must be a JSON object: {p}")
    return data


def validate_metrics(
    instance: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    schema_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate and return the instance. Raises jsonschema.ValidationError."""
    sch = schema if schema is not None else load_schema(schema_path)
    jsonschema.validate(instance=instance, schema=sch)
    if instance.get("artifact_kind") == "run":
        def finite_json(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite run artifact value")
            if isinstance(value, dict):
                for child in value.values():
                    finite_json(child)
            elif isinstance(value, list):
                for child in value:
                    finite_json(child)
        finite_json(instance)
        if instance["n_generated"] + instance["n_errors"] != instance["n_intended"]:
            raise ValueError("Run counts must conserve intended = generated + generation errors")
        if instance["n_judged"] + instance["n_judge_failures"] != instance["n_generated"]:
            raise ValueError("Run counts must conserve generated = judged + judge failures")
        if instance["harmful_count"] > instance["n_intended"]:
            raise ValueError("harmful_count cannot exceed n_intended")
        if instance["metrics"].get("asr") is not None and (
            instance["n_judged"] == 0 or instance["harmful_count"] == 0
        ):
            raise ValueError("Numeric ASR requires judged harmful examples")
        reasons = instance.get("null_reasons", {})
        for key, value in instance["metrics"].items():
            if value is None and not (reasons.get(f"metrics.{key}") or reasons.get("metrics.*")):
                raise ValueError(f"Missing null reason for metrics.{key}")
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite metric: {key}")
        low = instance["metrics"].get("asr_ci_low")
        high = instance["metrics"].get("asr_ci_high")
        if low is not None and high is not None and low > high:
            raise ValueError("Reversed ASR confidence interval")
    return instance


def load_and_validate(
    path: Path | str,
    schema_path: Path | str | None = None,
) -> dict[str, Any]:
    return validate_metrics(load_metrics(path), schema_path=schema_path)


def is_fixture(metrics: dict[str, Any]) -> bool:
    """Exclude all non-run artifacts from measured-result consumers.

    This compatibility predicate does not certify a run's eligibility. Consumers
    must also validate provenance and saved per-sample evidence.
    """
    kind = metrics.get("artifact_kind")
    if kind != "run":
        return True
    return metrics.get("split") in {"example", "fixture"}


__all__ = [
    "DEFAULT_SCHEMA",
    "load_schema",
    "load_metrics",
    "validate_metrics",
    "load_and_validate",
    "is_fixture",
]
