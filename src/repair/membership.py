"""Canonical calibration and dev membership. Prompt files stay closed until this passes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.dev_runner import (
    CANONICAL_FAMILY_MANIFEST,
    CANONICAL_FAMILY_MANIFEST_SHA256,
    is_lower_hex,
    load_canonical_authority,
)

from .errors import RepairError
from .jsonutil import sha256_bytes, sha256_text, strict_json_bytes

REPAIR_SPLITS = frozenset({"calibration", "dev"})
INDEX_FIELDS = (
    "sample_id",
    "item_id",
    "family_id",
    "source_dataset",
    "source_id",
    "lang",
    "hs_label",
    "split",
    "prompt_hash",
)
ITEM_LANGS = frozenset({"en", "hi", "hinglish"})
ITEM_HS_LABELS = frozenset({"harm", "benign"})


def canonical_manifest_path() -> Path:
    return CANONICAL_FAMILY_MANIFEST


def canonical_manifest_sha256() -> str:
    return CANONICAL_FAMILY_MANIFEST_SHA256


def assert_prompt_path_allowed(path: Path) -> None:
    parts = {part.casefold() for part in path.parts}
    if "test" in parts:
        raise RepairError("refusing a test prompt path", stage="membership")


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "" or value.strip() != value:
        raise RepairError(f"missing {label}", stage="membership")
    if any(char in value for char in "\r\n"):
        raise RepairError(f"missing {label}", stage="membership")
    return value


def load_authority(path: Path, *, require_pinned_hash: bool) -> dict[str, Any]:
    try:
        authority = load_canonical_authority(path, require_pinned_hash=require_pinned_hash)
    except Exception as exc:
        message = str(exc)
        if "prompt" in message.casefold():
            message = "canonical family manifest was refused"
        raise RepairError(message, stage="membership") from None
    return authority


def load_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RepairError("missing repair index", stage="membership")
    try:
        payload = strict_json_bytes(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepairError("repair index is unreadable", stage="membership") from exc
    if not isinstance(payload, list) or not payload:
        raise RepairError("repair index is empty", stage="membership")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise RepairError("repair index row is unreadable", stage="membership")
        if any(key in raw for key in ("prompt", "response", "completion", "text")):
            raise RepairError("repair index must stay metadata-only", stage="membership")
        extra = set(raw) - set(INDEX_FIELDS)
        if extra or any(field not in raw for field in INDEX_FIELDS):
            raise RepairError("repair index field set is wrong", stage="membership")
        row = {field: _token(raw[field], field) for field in INDEX_FIELDS}
        if row["split"] not in REPAIR_SPLITS:
            raise RepairError(f"repair index refuses split {row['split']}", stage="membership")
        if row["lang"] not in ITEM_LANGS:
            raise RepairError("unsupported language", stage="membership")
        if row["hs_label"] not in ITEM_HS_LABELS:
            raise RepairError("missing hs_label", stage="membership")
        if not is_lower_hex(row["prompt_hash"], 64):
            raise RepairError("missing prompt_hash", stage="membership")
        if row["sample_id"] in seen:
            raise RepairError("duplicate repair index id", stage="membership")
        seen.add(row["sample_id"])
        rows.append(row)
    return rows


def assert_repair_membership(
    rows: Sequence[Mapping[str, str]],
    authority: Mapping[str, Any],
    *,
    allowed_splits: frozenset[str],
) -> None:
    """Refuse test members and split laundering. Does not open a prompt file."""
    if not rows:
        raise RepairError("repair index is empty", stage="membership")
    if not allowed_splits or not allowed_splits <= REPAIR_SPLITS:
        raise RepairError("repair refuses a test split", stage="membership")
    by_source = authority.get("by_source")
    if not isinstance(by_source, dict):
        raise RepairError("canonical family manifest is missing families", stage="membership")
    for row in rows:
        if row["split"] not in allowed_splits:
            raise RepairError(f"repair refuses split {row['split']}", stage="membership")
        found = by_source.get((row["source_dataset"], row["source_id"]))
        if not isinstance(found, dict):
            raise RepairError("canonical membership mismatch", stage="membership")
        if found.get("split") == "test":
            raise RepairError("canonical split refuses test membership", stage="membership")
        if found.get("split") not in REPAIR_SPLITS:
            raise RepairError("canonical split is not calibration or dev", stage="membership")
        if found.get("split") != row["split"]:
            raise RepairError("canonical split refuses relabeling", stage="membership")
        if found.get("family_id") != row["family_id"] or found.get("item_id") != row["item_id"]:
            raise RepairError("canonical membership mismatch", stage="membership")
        if found.get("lang") != row["lang"] or found.get("hs_label") != row["hs_label"]:
            raise RepairError("canonical label mismatch", stage="membership")
        family_label = found.get("family_hs_label")
        if family_label in ITEM_HS_LABELS and family_label != row["hs_label"]:
            raise RepairError("canonical label mismatch", stage="membership")


def require_both_classes(rows: Sequence[Mapping[str, str]], *, stage: str) -> None:
    labels = {row["hs_label"] for row in rows}
    if "harm" not in labels or "benign" not in labels:
        raise RepairError("missing calibration class", stage=stage)


def load_prompts(path: Path, rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    """Open prompts only after membership has already passed."""
    assert_prompt_path_allowed(path)
    if not path.is_file():
        raise RepairError("missing prompts", stage="prompts")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError("prompts file is unreadable", stage="prompts") from exc
    found: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RepairError(f"prompts file is unreadable at line {line_number}", stage="prompts") from exc
        if not isinstance(payload, dict):
            raise RepairError("prompts file is unreadable", stage="prompts")
        sample_id = payload.get("sample_id")
        prompt = payload.get("prompt")
        if not isinstance(sample_id, str) or sample_id not in {row["sample_id"] for row in rows}:
            raise RepairError("prompts file contains a row outside the checked index", stage="prompts")
        if not isinstance(prompt, str) or prompt == "":
            raise RepairError("missing prompt text", stage="prompts")
        if sha256_text(prompt) != next(row["prompt_hash"] for row in rows if row["sample_id"] == sample_id):
            raise RepairError("prompt hash mismatch", stage="prompts")
        if sample_id in found:
            raise RepairError("duplicate prompt row", stage="prompts")
        token_ids = payload.get("fixture_token_ids")
        if token_ids is not None:
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or any(type(item) is not int or item < 0 for item in token_ids)
            ):
                raise RepairError("fixture token ids are unreadable", stage="prompts")
        found[sample_id] = {"sample_id": sample_id, "prompt": prompt, "fixture_token_ids": token_ids}
    if set(found) != {row["sample_id"] for row in rows}:
        raise RepairError("prompts file does not match the checked index", stage="prompts")
    joined: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["prompt"] = found[row["sample_id"]]["prompt"]
        item["fixture_token_ids"] = found[row["sample_id"]]["fixture_token_ids"]
        joined.append(item)
    return joined


def assert_activations_are_not_prompts(payload: Any) -> None:
    if isinstance(payload, dict):
        if any(key in payload for key in ("prompt", "response", "completion", "text")):
            raise RepairError("activation file must not carry prompt text", stage="activations")
        for value in payload.values():
            assert_activations_are_not_prompts(value)
    elif isinstance(payload, list):
        for value in payload:
            assert_activations_are_not_prompts(value)


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())
