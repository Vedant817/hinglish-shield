"""Synthetic CLI contracts for the provisional bf16 freeze step. No weights."""
import json
import math

from src.repair.cli import main
from src.repair.jsonutil import content_sha256
from src.repair.freeze import verify_provisional_bf16_freeze

REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def synthetic_direction(**overrides):
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
        "vector": [3.0, 4.0],
        "l2_norm": 5.0,
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


def test_freeze_provisional_cli_writes_and_verifies(tmp_path, capsys):
    direction = tmp_path / "locate_direction.json"
    direction.write_text(json.dumps(synthetic_direction()) + "\n", encoding="utf-8")
    out = tmp_path / "freeze.json"
    assert main(["freeze-provisional", "--direction", str(direction), "--out", str(out)]) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["verification_pathway"] == "provisional_bf16_locate"
    assert manifest["verification"]["verified"] is True
    assert manifest["content_sha256"] == content_sha256(manifest)
    checked = verify_provisional_bf16_freeze(manifest)
    assert checked["verified"] is True
    assert "verified" in capsys.readouterr().out


def test_freeze_provisional_cli_refuses_test_split_and_bad_controls(tmp_path):
    bad_split = tmp_path / "bad_split.json"
    bad_split.write_text(json.dumps(synthetic_direction(split="test")) + "\n", encoding="utf-8")
    out = tmp_path / "freeze.json"
    assert main(["freeze-provisional", "--direction", str(bad_split), "--out", str(out)]) == 2
    assert not out.exists()

    broken = synthetic_direction()
    broken["controls"]["alpha_zero_with_wrapper"] = {
        "ran": True, "matches_base": False, "max_abs_logit_diff": 0.25}
    broken["content_sha256"] = content_sha256(broken)
    bad_controls = tmp_path / "bad_controls.json"
    bad_controls.write_text(json.dumps(broken) + "\n", encoding="utf-8")
    assert main(["freeze-provisional", "--direction", str(bad_controls), "--out", str(out)]) == 2
    assert not out.exists()


def test_freeze_provisional_cli_refuses_fp32_claim(tmp_path):
    direction = tmp_path / "locate_direction.json"
    direction.write_text(json.dumps(synthetic_direction(original_fp32_credit=True)) + "\n",
                         encoding="utf-8")
    out = tmp_path / "freeze.json"
    assert main(["freeze-provisional", "--direction", str(direction), "--out", str(out)]) == 2
    assert not out.exists()
