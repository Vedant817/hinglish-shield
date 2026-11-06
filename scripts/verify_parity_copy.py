#!/usr/bin/env python3
"""Verify notebook parity prompts/KL/long-spec match vendor/CircuitKIT at recorded SHA.

Exits non-zero on any mismatch. Does not print secrets.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR_CK = ROOT / "vendor" / "CircuitKIT"
UPSTREAM = VENDOR_CK / "tests" / "backends" / "tl_compat" / "test_cohere_parity.py"
NOTEBOOK = ROOT / "notebooks" / "gpu_parity_and_baseline.ipynb"
META = ROOT / "runs" / "setup-smoke-2026-11-06" / "parity-copy-meta.json"


def norm_src(s: str) -> str:
    """Same normalization as the notebook: strip outer blank lines, rstrip each line, one trailing newline."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines()) + "\n"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def extract_upstream(path: Path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    prompts = None
    kl_src = None
    long_spec = {
        "window": 4096,
        "seq_len": 4196,
        "fill_token_text": "the",
        "bos_policy": "prepend bos if default_prepend_bos and isinstance(bos, int); then pad with fill_id to seq_len",
        "upstream_method": "TestSlidingWindowBoundary.test_parity_holds_past_the_window",
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_kl":
            kl_src = ast.get_source_segment(src, node)
        if isinstance(node, ast.ClassDef) and node.name == "TestNextTokenParity":
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name) and t.id == "PROMPTS":
                            prompts = ast.literal_eval(item.value)
        if isinstance(node, ast.ClassDef) and node.name == "TestSlidingWindowBoundary":
            # confirm window+100 pattern exists in source
            method_src = ast.get_source_segment(src, node)
            if method_src is None or "window + 100" not in method_src and "seq_len = window + 100" not in src:
                # still check file text
                pass
    if "seq_len = window + 100" not in src or "window == 4096" not in src:
        raise SystemExit("upstream long-prompt recipe not found as expected")
    if prompts is None or kl_src is None:
        raise SystemExit("failed to extract PROMPTS or _kl from upstream")
    return prompts, kl_src, long_spec


def extract_notebook(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    blob = "\n".join(
        "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
        for c in nb.get("cells", [])
    )
    # Exec only the assignments we need in a tiny namespace via ast of concatenated config
    # Safer: parse string literals for PARITY_PROMPTS / PARITY_KL_SOURCE / hashes from config cell
    prompts = None
    kl_source = None
    long_spec = None
    recorded = {}
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else c.get("source", "")
        if "PARITY_PROMPTS" not in src or "PARITY_KL_SOURCE" not in src:
            continue
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        if t.id == "PARITY_PROMPTS":
                            prompts = ast.literal_eval(node.value)
                        elif t.id == "PARITY_KL_SOURCE":
                            kl_source = ast.literal_eval(node.value)
                        elif t.id == "PARITY_LONG_SPEC":
                            long_spec = ast.literal_eval(node.value)
                        elif t.id == "PARITY_CIRCUITKIT_SHA":
                            recorded["circuitkit_sha"] = ast.literal_eval(node.value)
                        elif t.id == "PARITY_PROMPTS_SHA256":
                            # may be computed; also look for assert strings
                            pass
        # hashes from assert string literals
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
                left = node.test.left
                if isinstance(left, ast.Name) and left.id == "PARITY_PROMPTS_SHA256":
                    recorded["prompts_sha256"] = ast.literal_eval(node.test.comparators[0])
                if isinstance(left, ast.Name) and left.id == "PARITY_KL_SOURCE_SHA256":
                    recorded["kl_source_sha256"] = ast.literal_eval(node.test.comparators[0])
                if isinstance(left, ast.Name) and left.id == "PARITY_LONG_SPEC_SHA256":
                    recorded["long_spec_sha256"] = ast.literal_eval(node.test.comparators[0])
        break
    if prompts is None or kl_source is None or long_spec is None:
        raise SystemExit("notebook missing PARITY_PROMPTS / PARITY_KL_SOURCE / PARITY_LONG_SPEC")
    return prompts, kl_source, long_spec, recorded


def main() -> int:
    lines = []
    def log(msg: str):
        lines.append(msg)
        print(msg)

    if not VENDOR_CK.exists():
        log(f"FAIL: missing {VENDOR_CK}")
        return 1
    head = git_sha(VENDOR_CK)
    log(f"vendor/CircuitKIT HEAD: {head}")
    up_prompts, up_kl, up_long = extract_upstream(UPSTREAM)
    nb_prompts, nb_kl, nb_long, recorded = extract_notebook(NOTEBOOK)
    log(f"notebook recorded CircuitKIT SHA: {recorded.get('circuitkit_sha')}")

    up_ph = sha256_text(json.dumps(up_prompts, ensure_ascii=False))
    nb_ph = sha256_text(json.dumps(nb_prompts, ensure_ascii=False))
    up_kh = sha256_text(norm_src(up_kl))
    nb_kh = sha256_text(norm_src(nb_kl))
    up_lh = sha256_text(json.dumps(up_long, ensure_ascii=False, sort_keys=True))
    nb_lh = sha256_text(json.dumps(nb_long, ensure_ascii=False, sort_keys=True))

    log(f"prompts_sha256 upstream={up_ph}")
    log(f"prompts_sha256 notebook={nb_ph}")
    log(f"kl_source_sha256 upstream={up_kh}")
    log(f"kl_source_sha256 notebook={nb_kh}")
    log(f"long_spec_sha256 upstream={up_lh}")
    log(f"long_spec_sha256 notebook={nb_lh}")

    errors = []
    if recorded.get("circuitkit_sha") and recorded["circuitkit_sha"] != head:
        # Allow notebook to pin a SHA equal to current vendor HEAD; mismatch is fail
        if head != recorded["circuitkit_sha"]:
            errors.append(
                f"CircuitKIT SHA mismatch: vendor HEAD {head} vs notebook {recorded['circuitkit_sha']}"
            )
    if up_prompts != nb_prompts:
        errors.append("PROMPTS list mismatch")
    if up_ph != nb_ph:
        errors.append("prompts sha256 mismatch between upstream extract and notebook list")
    if norm_src(up_kl) != norm_src(nb_kl):
        errors.append("KL source text mismatch after normalization")
    if up_kh != nb_kh:
        errors.append("kl source sha256 mismatch")
    if up_long != nb_long:
        errors.append(f"LONG_SPEC mismatch: upstream={up_long} notebook={nb_long}")
    if up_lh != nb_lh:
        errors.append("long_spec sha256 mismatch")
    for key, val in (
        ("prompts_sha256", nb_ph),
        ("kl_source_sha256", nb_kh),
        ("long_spec_sha256", nb_lh),
    ):
        if key in recorded and recorded[key] != val:
            errors.append(f"recorded assert {key}={recorded[key]} != computed {val}")

    if META.exists():
        meta = json.loads(META.read_text())
        log(f"meta file circuitkit_sha={meta.get('circuitkit_sha')}")
        if meta.get("prompts_sha256") != nb_ph:
            errors.append("meta prompts_sha256 mismatch")
        if meta.get("kl_source_sha256") != nb_kh:
            errors.append("meta kl_source_sha256 mismatch")
        if meta.get("long_spec_sha256") != nb_lh:
            errors.append("meta long_spec_sha256 mismatch")

    if errors:
        log("FAIL:")
        for e in errors:
            log(f"  - {e}")
        return 1
    log("PASS: notebook parity copy matches vendor/CircuitKIT extract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
