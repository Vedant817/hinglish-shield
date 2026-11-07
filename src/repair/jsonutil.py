"""Canonical JSON helpers. No NaN, no prompt logging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def lf_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def sha256_file(path: Path) -> str:
    return sha256_bytes(lf_bytes(path))


def strict_json_bytes(raw: bytes) -> Any:
    def invalid(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    return json.loads(raw.decode("utf-8"), parse_constant=invalid)


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(payload: dict[str, Any], *, drop: str = "content_sha256") -> str:
    body = {key: value for key, value in payload.items() if key != drop}
    return sha256_text(canonical_dumps(body))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
