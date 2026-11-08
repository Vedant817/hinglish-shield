"""Synthetic provisional bf16 locate freeze. No weights, no GPU, no metrics."""
import json
import math
from pathlib import Path

import pytest

from src.repair.errors import RepairError
from src.repair.freeze import (
    build_provisional_bf16_freeze,
    verify_any,
    verify_provisional_bf16_freeze,
)
from src.repair.jsonutil import content_sha256

REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def synthetic_direction(vector=(3.0, 4.0, 0.0, 0.0), **overrides):
    direction = {
        "schema_version": "provisional_locate_v1",
        "provisional": True,
        "original_fp32_credit": False,
        "parity_credit": None,
        "metrics_eligible": False,
        "primary_asr": None,
        "model_id": "CohereLabs/tiny-aya-global",
        "model_revision": REVISION,
        "dtype": "bfloat16",
        "node": "MLP 0",
        "declared_score": 1.0,
        "split": "calibration+dev",
        "vector": list(vector),
        "l2_norm": math.sqrt(sum(v * v for v in vector)),
        "controls": {
            "base_no_intervention": {"ran": True},
            "alpha_zero_with_wrapper": {"ran": True, "matches_base": True, "max_abs_logit_diff": 0.0},
            "random_norm_matched_seed_7": {"ran": True, "max_abs_logit_diff": 0.5, "norm_gap": 0.0},
            "candidate_alpha_1": {"ran": True, "max_abs_logit_diff": 0.7},
        },
        "checks": {"alpha_zero_matches_base": True},
        "limitations": ["synthetic fixture"],
    }
    direction.update(overrides)
    direction["content_sha256"] = content_sha256(direction)
    return direction


def write_direction(tmp_path: Path, direction) -> Path:
    path = tmp_path / "locate_direction.json"
    path.write_text(json.dumps(direction, indent=2) + "\n", encoding="utf-8")
    return path


def test_build_and_verify_provisional_freeze(tmp_path):
    path = write_direction(tmp_path, synthetic_direction())
    manifest = build_provisional_bf16_freeze(path)
    assert manifest["verification_pathway"] == "provisional_bf16_locate"
    assert manifest["provisional"] is True
    assert manifest["original_fp32_credit"] is False
    assert manifest["parity_credit"] is None
    assert manifest["metrics_eligible"] is False
    assert manifest["primary_asr"] is None
    assert manifest["split"] == "calibration+dev"
    assert manifest["content_sha256"] == content_sha256(manifest)
    checked = verify_provisional_bf16_freeze(manifest)
    assert checked["verified"] is True
    assert checked["weights_reloaded"] is False
    via_any = verify_any(manifest)
    assert via_any["verification_pathway"] == "provisional_bf16_locate"


@pytest.mark.parametrize("patch", [
    {"provisional": False},
    {"original_fp32_credit": True},
    {"parity_credit": "fp32-original"},
    {"dtype": "float32"},
    {"node": "blocks.0.hook_resid_post"},
    {"model_revision": "0" * 40},
    {"split": "test"},
    {"declared_score": 2.0},
    {"vector": [0.0, 0.0]},
    {"l2_norm": 99.0},
])
def test_freeze_refuses_unpinned_or_credit_claims(tmp_path, patch):
    path = write_direction(tmp_path, synthetic_direction(**patch))
    with pytest.raises(RepairError):
        build_provisional_bf16_freeze(path)


def test_freeze_refuses_failed_controls(tmp_path):
    broken = synthetic_direction()
    broken["controls"]["alpha_zero_with_wrapper"]["matches_base"] = False
    broken["content_sha256"] = content_sha256(broken)
    with pytest.raises(RepairError):
        build_provisional_bf16_freeze(write_direction(tmp_path, broken))


def test_verify_refuses_tampered_manifest(tmp_path):
    manifest = build_provisional_bf16_freeze(write_direction(tmp_path, synthetic_direction()))
    manifest["vector"] = [9.0, 9.0, 9.0, 9.0]
    with pytest.raises(RepairError):
        verify_provisional_bf16_freeze(manifest)
