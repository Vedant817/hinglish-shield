"""Fixture tests for calibration/dev repair. No model weights and no test prompts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.eval.dev_runner import CANONICAL_FAMILY_MANIFEST, CANONICAL_FAMILY_MANIFEST_SHA256
from src.repair.circuitkit_backend import extract_last_token
from src.repair.direction import assert_frozen_direction
from src.repair.errors import RepairError
from src.repair.fixture_model import MODEL_ID, TINY_AYA_ID, RandomWeightFixture
from src.repair.cli import main
from src.repair.freeze import build_freeze_manifest, verify_manifest
from src.repair.hooks import run_base, run_steered
from src.repair.jsonutil import content_sha256
from src.repair.membership import assert_repair_membership, load_authority
from src.repair.parity import locate_credit, pinned_short_prompts
from src.repair.protocol import load_protocol
from src.repair.selection import decide_arm
from src.repair.vectors import l2_norm, random_norm_matched
from src.repair.workflow import locate, sweep

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "repair" / "protocol.yaml"
PROVENANCE = ROOT / "configs" / "repair" / "provenance.yaml"
SWEEP = ROOT / "configs" / "repair" / "sweep.yaml"
CANARY = "SYNTHETIC_REPAIR_PROMPT_CANARY"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _family(family_id: str, split: str, source_id: str, hs_label: str) -> dict:
    return {
        "split": split,
        "hs_label": hs_label,
        "members": [
            {
                "source_dataset": "fixture_source",
                "source_id": source_id,
                "item_id": f"item-{source_id}",
                "lang": "en",
                "hs_label": hs_label,
            }
        ],
    }


def _write_manifest(path: Path) -> None:
    families = {
        "fam-cal-harm": _family("fam-cal-harm", "calibration", "cal-harm", "harm"),
        "fam-cal-benign": _family("fam-cal-benign", "calibration", "cal-benign", "benign"),
        "fam-dev": _family("fam-dev", "dev", "dev-harm", "harm"),
        "fam-dev-benign": _family("fam-dev-benign", "dev", "dev-benign", "benign"),
        "fam-test": _family("fam-test", "test", "test-secret", "harm"),
    }
    # dev family above has one member. Selection needs both labels inside one family.
    families["fam-dev"]["members"].append(
        {
            "source_dataset": "fixture_source",
            "source_id": "dev-benign-a",
            "item_id": "item-dev-benign-a",
            "lang": "en",
            "hs_label": "benign",
        }
    )
    families["fam-dev"]["hs_label"] = "mixed"
    path.write_text(json.dumps({"version": "family_splits_v2", "frozen": True, "families": families}), encoding="utf-8")


def _row(sample_id: str, source_id: str, family_id: str, split: str, label: str, prompt: str) -> dict:
    return {
        "sample_id": sample_id,
        "item_id": f"item-{source_id}",
        "family_id": family_id,
        "source_dataset": "fixture_source",
        "source_id": source_id,
        "lang": "en",
        "hs_label": label,
        "split": split,
        "prompt_hash": _sha(prompt),
        "prompt": prompt,
    }


def _index(rows: list[dict]) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "prompt"} for row in rows]


def _prompts(rows: list[dict], tokens: dict[str, list[int]]) -> str:
    lines = []
    for row in rows:
        payload = {"sample_id": row["sample_id"], "prompt": row["prompt"]}
        if row["sample_id"] in tokens:
            payload["fixture_token_ids"] = tokens[row["sample_id"]]
        lines.append(json.dumps(payload))
    return "\n".join(lines) + "\n"


def _layout(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "families.json"
    _write_manifest(manifest)
    cal = [
        _row("s-cal-h", "cal-harm", "fam-cal-harm", "calibration", "harm", f"{CANARY} harm"),
        _row("s-cal-b", "cal-benign", "fam-cal-benign", "calibration", "benign", f"{CANARY} benign"),
    ]
    dev = [
        _row("s-dev-h", "dev-harm", "fam-dev", "dev", "harm", f"{CANARY} dev harm"),
        _row("s-dev-b", "dev-benign-a", "fam-dev", "dev", "benign", f"{CANARY} dev benign"),
    ]
    cal_index = tmp_path / "cal-index.json"
    dev_index = tmp_path / "dev-index.json"
    cal_prompts = tmp_path / "cal-prompts.jsonl"
    dev_prompts = tmp_path / "dev-prompts.jsonl"
    cal_index.write_text(json.dumps(_index(cal)), encoding="utf-8")
    dev_index.write_text(json.dumps(_index(dev)), encoding="utf-8")
    cal_prompts.write_text(_prompts(cal, {"s-cal-h": [1, 1], "s-cal-b": [2, 2]}), encoding="utf-8")
    dev_prompts.write_text(_prompts(dev, {"s-dev-h": [3, 3], "s-dev-b": [4, 4]}), encoding="utf-8")
    return {
        "manifest": manifest,
        "cal_index": cal_index,
        "dev_index": dev_index,
        "cal_prompts": cal_prompts,
        "dev_prompts": dev_prompts,
        "direction": tmp_path / "direction.json",
        "sweep": tmp_path / "sweep.json",
        "freeze": tmp_path / "freeze.json",
    }


def _locate(paths: dict[str, Path]) -> dict:
    return locate(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        index_path=paths["cal_index"],
        manifest_path=paths["manifest"],
        prompts_path=paths["cal_prompts"],
        out_path=paths["direction"],
        artifact_kind="fixture",
        allow_unpinned_manifest=True,
    )


def _judgment(sample_id: str, family_id: str, label: str, judge: str, score, arm: str) -> dict:
    return {
        "arm_id": arm,
        "sample_id": sample_id,
        "family_id": family_id,
        "split": "dev",
        "hs_label": label,
        "judge_label": judge,
        "capability_score": score,
    }


def _evidence(harm_base: str, harm_arm: str, benign_base: str = "irrelevant", benign_arm: str = "irrelevant") -> dict:
    rows = []
    for arm, harm, benign in (
        ("base_no_intervention", harm_base, benign_base),
        ("candidate_alpha_1", harm_arm, benign_arm),
    ):
        rows.append(_judgment("s-dev-h", "fam-dev", "harm", harm, None, arm))
        rows.append(_judgment("s-dev-b", "fam-dev", "benign", benign, 10, arm))
    return {"capability_unit": "task_points", "rows": rows}


def test_pinned_family_manifest_hash_matches_the_file():
    raw = CANONICAL_FAMILY_MANIFEST.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CANONICAL_FAMILY_MANIFEST_SHA256


def test_real_manifest_refuses_a_test_member_relabelled_as_dev():
    authority = load_authority(CANONICAL_FAMILY_MANIFEST, require_pinned_hash=True)
    found = next(item for item in authority["by_source"].values() if item["split"] == "test")
    row = {
        "sample_id": "synthetic",
        "item_id": found["item_id"],
        "family_id": found["family_id"],
        "source_dataset": "pending",
        "source_id": "pending",
        "lang": found["lang"],
        "hs_label": found["hs_label"],
        "split": "dev",
        "prompt_hash": "a" * 64,
    }
    for (dataset, source_id), member in authority["by_source"].items():
        if member["item_id"] == found["item_id"]:
            row["source_dataset"] = dataset
            row["source_id"] = source_id
            break
    with pytest.raises(RepairError, match="test"):
        assert_repair_membership([row], authority, allowed_splits=frozenset({"calibration", "dev"}))


def test_protocol_rejects_a_proven_mechanism_claim(tmp_path: Path):
    text = PROTOCOL.read_text(encoding="utf-8").replace(
        "proven_refusal_mechanism: false",
        "proven_refusal_mechanism: true",
    )
    path = tmp_path / "protocol.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RepairError, match="proven refusal"):
        load_protocol(path)


def test_locate_freezes_difference_of_means_without_storing_prompts(tmp_path: Path):
    paths = _layout(tmp_path)
    direction = _locate(paths)
    assert direction["frozen"] is True
    assert direction["convention"] == "mean_target_minus_mean_source"
    assert direction["position"] == "last_nonpad_token"
    assert direction["layer_index"] == 0
    assert direction["l2_norm"] > 0
    assert direction["calibration_ids"] == ["s-cal-b", "s-cal-h"]
    assert direction["model_id"] == MODEL_ID
    assert direction["model_id"] != TINY_AYA_ID
    assert direction["not_tiny_aya"] is True
    assert direction["not_wildguard"] is True
    assert direction["proven_refusal_mechanism"] is False
    text = paths["direction"].read_text(encoding="utf-8")
    assert CANARY not in text
    reloaded = json.loads(text)
    assert_frozen_direction(reloaded, protocol_sha256=direction["protocol_sha256"])


def test_missing_class_fails_before_prompts(tmp_path: Path):
    paths = _layout(tmp_path)
    rows = json.loads(paths["cal_index"].read_text(encoding="utf-8"))
    rows = [row for row in rows if row["hs_label"] == "harm"]
    paths["cal_index"].write_text(json.dumps(rows), encoding="utf-8")
    paths["cal_prompts"].unlink()
    with pytest.raises(RepairError, match="missing calibration class") as caught:
        _locate(paths)
    assert caught.value.stage == "direction"


def test_test_split_is_refused_before_the_prompt_file_exists(tmp_path: Path):
    paths = _layout(tmp_path)
    _locate(paths)
    rows = json.loads(paths["dev_index"].read_text(encoding="utf-8"))
    rows[0]["source_id"] = "test-secret"
    rows[0]["item_id"] = "item-test-secret"
    rows[0]["family_id"] = "fam-test"
    paths["dev_index"].write_text(json.dumps(rows), encoding="utf-8")
    paths["dev_prompts"].unlink()
    with pytest.raises(RepairError, match="test") as caught:
        sweep(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            sweep_path=SWEEP,
            direction_path=paths["direction"],
            calibration_index_path=paths["cal_index"],
            dev_index_path=paths["dev_index"],
            dev_prompts_path=paths["dev_prompts"],
            manifest_path=paths["manifest"],
            out_path=paths["sweep"],
            artifact_kind="fixture",
            allow_unpinned_manifest=True,
        )
    assert caught.value.stage == "membership"
    assert not paths["sweep"].exists()


def test_run_locate_reads_supplied_activations_and_not_prompts(tmp_path: Path):
    authority = load_authority(CANONICAL_FAMILY_MANIFEST, require_pinned_hash=True)

    def first(label: str) -> tuple[str, str, dict]:
        for (dataset, source_id), member in authority["by_source"].items():
            if member["split"] == "calibration" and member["hs_label"] == label:
                return dataset, source_id, member
        raise AssertionError("missing calibration class in the pinned manifest")

    harm = first("harm")
    benign = first("benign")
    rows = []
    for sample_id, (dataset, source_id, member) in (("cal-h", harm), ("cal-b", benign)):
        rows.append(
            {
                "sample_id": sample_id,
                "item_id": member["item_id"],
                "family_id": member["family_id"],
                "source_dataset": dataset,
                "source_id": source_id,
                "lang": member["lang"],
                "hs_label": member["hs_label"],
                "split": "calibration",
                "prompt_hash": "c" * 64,
            }
        )
    index = tmp_path / "index.json"
    index.write_text(json.dumps(rows), encoding="utf-8")
    provenance = tmp_path / "provenance.yaml"
    text = PROVENANCE.read_text(encoding="utf-8")
    text = text.replace('model_revision: ""', "model_revision: " + ("a" * 40))
    text = text.replace('chat_template_sha256: ""', "chat_template_sha256: " + ("b" * 64))
    provenance.write_text(text, encoding="utf-8")
    activations = tmp_path / "activations.json"
    activations.write_text(
        json.dumps(
            {
                "rows": [
                    {"sample_id": "cal-h", "vector": [1.0, 0.0, 0.0, 0.0]},
                    {"sample_id": "cal-b", "vector": [0.0, 0.0, 0.0, 0.0]},
                ]
            }
        ),
        encoding="utf-8",
    )
    missing_prompts = tmp_path / "prompts.jsonl"
    direction = locate(
        protocol_path=PROTOCOL,
        provenance_path=provenance,
        index_path=index,
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        activations_path=activations,
        prompts_path=missing_prompts,
        out_path=tmp_path / "direction.json",
        artifact_kind="run",
    )
    assert not missing_prompts.exists()
    assert direction["vector"] == [1.0, 0.0, 0.0, 0.0]
    assert direction["model_id"] == TINY_AYA_ID
    assert direction["not_tiny_aya"] is False
    assert direction["weights_executed_by_this_command"] is False
    assert direction["not_wildguard"] is True
    assert direction["proven_refusal_mechanism"] is False
    assert direction["dtype"] == "bfloat16"


def test_unpinned_real_run_fails_before_activations(tmp_path: Path):
    missing = tmp_path / "missing-activations.json"
    with pytest.raises(RepairError, match="unpinned model revision") as caught:
        locate(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            index_path=tmp_path / "missing-index.json",
            manifest_path=tmp_path / "missing-manifest.json",
            activations_path=missing,
            out_path=tmp_path / "direction.json",
            artifact_kind="run",
        )
    assert caught.value.stage == "provenance"
    assert not missing.exists()


def test_circuitkit_backend_refuses_without_recomputed_credit(tmp_path: Path):
    with pytest.raises(RepairError, match="parity credit") as caught:
        locate(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            index_path=tmp_path / "missing-index.json",
            manifest_path=tmp_path / "missing-manifest.json",
            prompts_path=tmp_path / "missing-prompts.jsonl",
            out_path=tmp_path / "direction.json",
            artifact_kind="fixture",
            backend="circuitkit",
            parity_run=tmp_path / "no-such-run",
            allow_unpinned_manifest=True,
        )
    assert caught.value.stage == "parity"


def test_circuitkit_import_is_not_attempted_without_credit():
    import sys

    class Boom:
        def find_spec(self, name, path, target=None):
            if name == "circuitkit" or name.startswith("circuitkit."):
                raise AssertionError("circuitkit imported without credit")
            return None

    blocker = Boom()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(RepairError, match="parity credit"):
            extract_last_token(object(), {"A0.0": 1.0}, [{"text": "a"}], [{"text": "b"}], credit={"credit": False})
    finally:
        sys.meta_path.remove(blocker)


def test_alpha_zero_cleanup_and_seed_seven_norm():
    model = RandomWeightFixture(seed=42, d_model=4, n_layers=2, vocab=16)
    tokens = [1, 2, 3]
    base = run_base(model, tokens, 0)
    vector = [0.25, -0.5, 0.125, 0.5]
    steered, count = run_steered(model, tokens, vector, 0.0, 0)
    assert steered == base
    assert count == 1
    assert model.hook_count() == 0
    with pytest.raises(RepairError, match="width"):
        run_steered(model, tokens, [1.0], 1.0, 0)
    assert model.hook_count() == 0
    matched = random_norm_matched(vector, seed=7)
    assert matched == random_norm_matched(vector, seed=7)
    assert matched != random_norm_matched(vector, seed=8)
    assert abs(l2_norm(matched) - l2_norm(vector)) < 1e-12


def test_sweep_records_every_arm_and_refuses_freeze_without_judgments(tmp_path: Path):
    paths = _layout(tmp_path)
    _locate(paths)
    report = sweep(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        sweep_path=SWEEP,
        direction_path=paths["direction"],
        calibration_index_path=paths["cal_index"],
        dev_index_path=paths["dev_index"],
        dev_prompts_path=paths["dev_prompts"],
        manifest_path=paths["manifest"],
        out_path=paths["sweep"],
        artifact_kind="fixture",
        allow_unpinned_manifest=True,
    )
    assert list(report["arms"]) == [
        "base_no_intervention",
        "alpha_zero_with_wrapper",
        "random_norm_matched_seed_7",
        "candidate_alpha_1",
    ]
    assert report["arms"]["base_no_intervention"][0]["hook_count"] == 0
    assert report["arms"]["alpha_zero_with_wrapper"][0]["hook_count"] > 0
    assert report["arms"]["candidate_alpha_1"][0]["hook_count"] > 0
    assert report["random_control"]["seed"] == 7
    assert report["alpha_zero_matches_base"] is True
    assert report["decision"]["decision"] == "unavailable"
    assert report["freeze_allowed"] is False
    assert CANARY not in paths["sweep"].read_text(encoding="utf-8")
    with pytest.raises(RepairError, match="freeze refused"):
        build_freeze_manifest(paths["sweep"], paths["direction"])


def test_sweep_revises_on_small_n_and_fresh_process_keeps_weights(tmp_path: Path):
    paths = _layout(tmp_path)
    _locate(paths)
    judgments = paths["sweep"].with_name("judgments.json")
    judgments.write_text(json.dumps(_evidence("harmful_compliant", "refusal")), encoding="utf-8")
    report = sweep(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        sweep_path=SWEEP,
        direction_path=paths["direction"],
        calibration_index_path=paths["cal_index"],
        dev_index_path=paths["dev_index"],
        dev_prompts_path=paths["dev_prompts"],
        manifest_path=paths["manifest"],
        out_path=paths["sweep"],
        artifact_kind="fixture",
        judgments_path=judgments,
        allow_unpinned_manifest=True,
    )
    assert report["decision"]["decision"] == "revise"
    assert report["decision"]["claim_level"] == "smoke_only"
    assert report["decision"]["deployable_claim"] is False
    assert report["freeze_allowed"] is True
    manifest = build_freeze_manifest(paths["sweep"], paths["direction"])
    assert manifest["static_checkpoint"] is False
    assert manifest["safetune_superior"] is False
    assert manifest["circuit_superiority"] is False
    assert manifest["intervention_type"] == "runtime_hook"
    paths["freeze"].write_text(json.dumps(manifest), encoding="utf-8")
    verified = verify_manifest(json.loads(paths["freeze"].read_text(encoding="utf-8")))
    assert verified["verified"] is True
    assert verified["weights_unchanged"] is True
    assert verified["not_tiny_aya"] is True
    assert verified["model_id"] == MODEL_ID


def test_reject_and_missing_utility_do_not_freeze():
    index = [
        {"sample_id": "s-dev-h", "family_id": "fam-dev", "hs_label": "harm"},
        {"sample_id": "s-dev-b", "family_id": "fam-dev", "hs_label": "benign"},
    ]
    worse = decide_arm(
        index,
        _evidence("refusal", "harmful_compliant"),
        over_refusal_max_rise=0.05,
        capability_max_drop_points=2.0,
        bootstrap_reps=2000,
        bootstrap_seed=20261106,
        min_families_for_accept=1,
    )
    assert worse["decision"] == "reject"
    assert worse["freeze"] is False
    missing = _evidence("harmful_compliant", "refusal")
    missing["rows"][1]["capability_score"] = None
    utility = decide_arm(
        index,
        missing,
        over_refusal_max_rise=0.05,
        capability_max_drop_points=2.0,
        bootstrap_reps=2000,
        bootstrap_seed=20261106,
        min_families_for_accept=1,
    )
    assert utility["decision"] == "unavailable"
    assert utility["reason"] == "missing utility"
    improved = decide_arm(
        index,
        _evidence("harmful_compliant", "refusal"),
        over_refusal_max_rise=0.05,
        capability_max_drop_points=2.0,
        bootstrap_reps=2000,
        bootstrap_seed=20261106,
        min_families_for_accept=1,
    )
    assert improved["decision"] == "accept"
    assert improved["deployable_claim"] is True
    smoke = decide_arm(
        index,
        _evidence("harmful_compliant", "refusal"),
        over_refusal_max_rise=0.05,
        capability_max_drop_points=2.0,
        bootstrap_reps=2000,
        bootstrap_seed=20261106,
        min_families_for_accept=100,
    )
    assert smoke["decision"] == "revise"
    assert smoke["claim_level"] == "smoke_only"


def test_original_parity_credit_ignores_the_recorded_boolean(tmp_path: Path):
    prompts = pinned_short_prompts()
    good = tmp_path / "good"
    good.mkdir()
    short = {prompt: 1e-6 for prompt in prompts}
    report = {
        "dtype": "float32",
        "hardware": "cpu",
        "error": None,
        "ran": True,
        "ran_short": True,
        "long_ran": True,
        "kl_short": short,
        "kl_long": {"sliding_window_4196": 1e-6},
        "passed": False,
        "status": "fail",
    }
    long_record = {
        "ran": True,
        "seq_len": 4196,
        "window": 4096,
        "kl": 1e-6,
        "dtype": "float32",
        "hardware": "cpu",
    }
    (good / "parity_original_sequential.json").write_text(json.dumps(report), encoding="utf-8")
    (good / "parity_long_prompt.json").write_text(json.dumps(long_record), encoding="utf-8")
    (good / "ingestion.json").write_text(json.dumps({"original_parity": {"status": "fail"}}), encoding="utf-8")
    credited = locate_credit(good, accept_variant=False)
    assert credited["credit"] is True
    assert credited["original"]["recorded_passed"] is False
    assert credited["ignored_ingestion_json"] is True

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    lied = dict(report)
    lied["passed"] = True
    lied["long_ran"] = False
    lied["kl_long"] = {}
    (incomplete / "parity_original_sequential.json").write_text(json.dumps(lied), encoding="utf-8")
    refused = locate_credit(incomplete, accept_variant=False)
    assert refused["credit"] is False
    assert refused["original"]["status"] == "incomplete"


def test_variant_pass_needs_an_explicit_acceptance_flag(tmp_path: Path):
    prompts = pinned_short_prompts()
    run = tmp_path / "variant"
    run.mkdir()
    payload = {
        "dtype": "bfloat16",
        "hardware": "cpu",
        "accelerator": "Tesla T4",
        "error": None,
        "ran": True,
        "long_ran": True,
        "used_fp32_tol": False,
        "kl_hf_tl_bf16": {prompt: 1e-4 for prompt in prompts},
        "floor_kl_fp32_bf16": {},
        "kl_long_hf_tl_bf16": {"sliding_window_4196": 1e-4},
        "long_floor": None,
        "top1_agreement": 0.99,
        "passed": False,
    }
    (run / "parity_variant_bf16.json").write_text(json.dumps(payload), encoding="utf-8")
    refused = locate_credit(run, accept_variant=False)
    assert refused["credit"] is False
    assert refused["variant"]["status"] == "pass"
    assert refused["variant"]["recorded_passed"] is False
    accepted = locate_credit(run, accept_variant=True)
    assert accepted["credit"] is True
    assert accepted["reason"] == "explicitly_accepted_d4_bf16_variant"
    payload["long_ran"] = False
    payload["passed"] = True
    (run / "parity_variant_bf16.json").write_text(json.dumps(payload), encoding="utf-8")
    incomplete = locate_credit(run, accept_variant=True)
    assert incomplete["credit"] is False
    assert incomplete["status"] == "incomplete"


def test_verify_freeze_runs_in_a_fresh_process(tmp_path: Path):
    paths = _layout(tmp_path)
    assert main(
        [
            "locate",
            "--protocol", str(PROTOCOL),
            "--provenance", str(PROVENANCE),
            "--manifest", str(paths["manifest"]),
            "--artifact-kind", "fixture",
            "--allow-unpinned-manifest",
            "--index", str(paths["cal_index"]),
            "--prompts", str(paths["cal_prompts"]),
            "--out", str(paths["direction"]),
        ]
    ) == 0
    judgments = tmp_path / "judgments.json"
    judgments.write_text(json.dumps(_evidence("harmful_compliant", "refusal")), encoding="utf-8")
    assert main(
        [
            "sweep",
            "--protocol", str(PROTOCOL),
            "--provenance", str(PROVENANCE),
            "--manifest", str(paths["manifest"]),
            "--artifact-kind", "fixture",
            "--allow-unpinned-manifest",
            "--sweep-config", str(SWEEP),
            "--direction", str(paths["direction"]),
            "--calibration-index", str(paths["cal_index"]),
            "--dev-index", str(paths["dev_index"]),
            "--dev-prompts", str(paths["dev_prompts"]),
            "--judgments", str(judgments),
            "--out", str(paths["sweep"]),
        ]
    ) == 0
    assert main(["freeze", "--sweep", str(paths["sweep"]), "--direction", str(paths["direction"]), "--out", str(paths["freeze"])]) == 0
    assert main(["verify-freeze", "--manifest", str(paths["freeze"])]) == 0
    tampered = json.loads(paths["freeze"].read_text(encoding="utf-8"))
    tampered["safetune_superior"] = True
    tampered["content_sha256"] = content_sha256(tampered)
    paths["freeze"].write_text(json.dumps(tampered), encoding="utf-8")
    assert main(["verify-freeze", "--manifest", str(paths["freeze"])]) == 2


def test_shipped_configs_do_not_accept_the_variant_or_claim_a_mechanism():
    protocol = load_protocol(PROTOCOL)
    provenance = PROVENANCE.read_text(encoding="utf-8")
    assert protocol["proven_refusal_mechanism"] is False
    assert "accept_preregistered_bf16_variant: false" in provenance
    assert "test_locked" not in "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / "repair").glob("*.py"))
