"""Validate preserved hardware evidence bytes, not model or safety results."""
import hashlib
import json
from pathlib import Path


def test_committed_hardware_preflight_hashes_match_checkout_bytes():
    root = Path(__file__).resolve().parents[2]
    directory = root / "runs/kaggle-preflight-2026-11-07"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
    source = (root / "scripts/kaggle_runtime_preflight.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == manifest["source_sha256"]
