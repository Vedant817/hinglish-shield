"""D8 descriptive status. This module does not score and does not load weights.

A blocked tokenizer identity is not a baseline. The returned record keeps
every metric null and repeats the pending-calibration label.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dev_runner import (
    D8_METRIC_LABEL,
    D8_SCORING_PATH,
    SECONDARY_MODEL_ID,
    SECONDARY_REVISION,
)

_SAFE_IDENTITY_FIELDS = (
    "artifact_kind",
    "chat_template_sha256",
    "error_type",
    "failure_phase",
    "metrics_eligible",
    "primary_asr",
    "public_fetch",
    "public_http_status",
    "secret_status",
    "split",
    "status",
    "tokenizer_config_sha256",
)


def d8_blocker_record(identity_path: Path) -> dict[str, Any]:
    """Read one identity artifact and refuse to turn it into a scored baseline.

    Only the fields listed above are copied. Tokens, notes, and URLs are not.
    """
    record: dict[str, Any] = {
        "schema_version": "d8_descriptive_status_v1",
        "scoring_path": D8_SCORING_PATH,
        "metric_label": D8_METRIC_LABEL,
        "artifact_kind": "blocker",
        "scored_baseline": False,
        "metrics_eligible": False,
        "primary_asr": None,
        "metrics": {
            "asr": None,
            "over_refusal": None,
            "language_gap": None,
            "capability": None,
        },
        "human_labels": "not invented",
        "judges": [
            {
                "name": "allenai/wildguard",
                "role": "primary",
                "version": "cbba4823f3e8020e5a74a5e29bf85072def6f2ff",
            },
            {
                "name": SECONDARY_MODEL_ID,
                "role": "secondary",
                "version": SECONDARY_REVISION,
            },
        ],
        "identity_path": identity_path.name,
        "blocker": None,
        "identity": None,
    }
    if not identity_path.is_file():
        record["blocker"] = "identity artifact missing"
        return record
    try:
        payload = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        record["blocker"] = "identity artifact unreadable"
        return record
    if not isinstance(payload, dict):
        record["blocker"] = "identity artifact unreadable"
        return record
    record["identity"] = {key: payload.get(key) for key in _SAFE_IDENTITY_FIELDS}
    blocked = (
        payload.get("secret_status") == "missing"
        or payload.get("failure_phase") == "secret_missing"
        or payload.get("public_http_status") == 401
        or payload.get("chat_template_sha256") in (None, "")
    )
    if blocked:
        record["blocker"] = "target chat-template pin is unavailable"
        return record
    record["blocker"] = "identity is not a scored D8 baseline"
    return record
