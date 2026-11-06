"""Load and validate runs/*/metrics.json against schemas/metrics.schema.json."""
from __future__ import annotations

import json
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
    return instance


def load_and_validate(
    path: Path | str,
    schema_path: Path | str | None = None,
) -> dict[str, Any]:
    return validate_metrics(load_metrics(path), schema_path=schema_path)


def is_fixture(metrics: dict[str, Any]) -> bool:
    """True when artifact is explicitly a fixture or example split."""
    kind = metrics.get("artifact_kind")
    if kind == "fixture":
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
