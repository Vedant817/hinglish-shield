"""Contract tests for the in-process repair runtime. No weight download and no test prompts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.dev_runner import CANONICAL_FAMILY_MANIFEST
from src.repair.cli import main
from src.repair.errors import RepairError
from src.repair.freeze import build_measured_freeze
from src.repair.jsonutil import content_sha256
from src.repair.membership import load_authority
from src.repair.parity import pinned_short_prompts
from src.repair.real_runtime import (
    LAYER_NAME,
    PINNED_REVISION,
    SOURCE_SHA,
    assert_accept_bindings,
    assert_import_order_state,
    generate_greedy,
    loader_import_order,
    locate_with_model,
    run_real_sweep,
)
from src.repair.tensor_fixture import DeterministicTensorFixture
from src.repair.workflow import locate

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "repair" / "protocol.yaml"
PROVENANCE = ROOT / "configs" / "repair" / "provenance.yaml"
SWEEP = ROOT / "configs" / "repair" / "sweep.yaml"
RUNTIME = ROOT / "configs" / "repair" / "real_runtime.yaml"
TEMPLATE = "ab" * 32
AUTHORITY = load_authority(CANONICAL_FAMILY_MANIFEST, require_pinned_hash=True)


def _pick(split: str, label: str) -> tuple[str, str, dict]:
    for (dataset, source_id), found in AUTHORITY["by_source"].items():
        family_label = found.get("family_hs_label")
        if found["split"] == split and found["hs_label"] == label and family_label in {label, "mixed"}:
            return dataset, source_id, found
    raise AssertionError(f"canonical manifest has no {split} {label} member")


def _row(split: str, label: str, *, override_split: str | None = None) -> dict:
    dataset, source_id, found = _pick(split, label)
    return {
        "sample_id": f"repair-{split}-{label}-{source_id}",
        "item_id": found["item_id"],
        "family_id": found["family_id"],
        "source_dataset": dataset,
        "source_id": source_id,
        "lang": found["lang"],
        "hs_label": found["hs_label"],
        "split": override_split or found["split"],
        "prompt_hash": "a" * 64,
    }


def _write_index(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _good_parity(path: Path) -> None:
    path.mkdir()
    prompts = pinned_short_prompts()
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
    (path / "parity_original_sequential.json").write_text(json.dumps(report), encoding="utf-8")
    (path / "parity_long_prompt.json").write_text(json.dumps(long_record), encoding="utf-8")
    (path / "ingestion.json").write_text(json.dumps({"original_parity": {"status": "fail"}}), encoding="utf-8")


def _variant_parity(path: Path) -> None:
    path.mkdir()
    prompts = pinned_short_prompts()
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
    (path / "parity_variant_bf16.json").write_text(json.dumps(payload), encoding="utf-8")


def _runtime(path: Path, *, template: str = TEMPLATE) -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    text = text.replace('chat_template_sha256: ""', f'chat_template_sha256: "{template}"', 1)
    path.write_text(text, encoding="utf-8")


def _accepted_provenance(path: Path) -> None:
    text = PROVENANCE.read_text(encoding="utf-8")
    text = text.replace(
        "accept_preregistered_bf16_variant: false",
        "accept_preregistered_bf16_variant: true",
        1,
    )
    path.write_text(text, encoding="utf-8")


def _refuse_factory(*_args, **_kwargs):
    raise AssertionError("model factory ran")


def _refuse_prompts(*_args, **_kwargs):
    raise AssertionError("prompts were read")


def _clean(_path: Path) -> dict:
    return {"git_dirty": False, "source_matches_commit": True, "git_head": "injected-check"}


def _source_ok(_path: Path) -> dict:
    return {
        "ok": True,
        "head": SOURCE_SHA,
        "dirty": False,
        "steering_matches_head": True,
        "reason": None,
    }


class LifecycleBundle(DeterministicTensorFixture):
    """Injected stand-in. real_weights stays false, so this is not Tiny Aya."""

    real_weights = False
    unloaded = False

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events
        events.append("target")

    def protocol_activation(self, token_ids: list[int], layer_name: str) -> list[float]:
        if layer_name != LAYER_NAME:
            raise RepairError("unexpected layer", stage="runtime")
        if int(token_ids[-1]) == 5:
            return [20.0, 0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0]

    def eos_ids(self) -> set[int]:
        return set()

    def unload(self) -> None:
        self.reset_hooks()
        self.unloaded = True
        self.events.append("target_unloaded")


def _prompts(rows: list[dict]) -> list[dict]:
    loaded = []
    for row in rows:
        item = dict(row)
        item["prompt"] = "SYNTHETIC_REPAIR_PROMPT_CANARY"
        item["fixture_token_ids"] = [5, 5] if row["hs_label"] == "harm" else [1, 1]
        loaded.append(item)
    return loaded


def test_loader_imports_circuitkit_before_transformer_lens():
    order = loader_import_order()
    assert order.index("circuitkit") < order.index("transformer_lens")
    assert order.index("circuitkit.backends._tl_compat") < order.index("transformer_lens")
    assert "CohereLabs/tiny-aya-global" in RUNTIME.read_text(encoding="utf-8")
    text = RUNTIME.read_text(encoding="utf-8")
    assert PINNED_REVISION in text
    assert 'chat_template_sha256: ""' in text
    assert "load_dtype: float32" in text
    assert "score_dtype_evidenced: false" in text
    assert "max_new_tokens: 256" in text
    assert "seed: 42" in text
    assert "method: base" in text
    assert "declared_arm: candidate_alpha_1" in text
    assert "do_sample: false" in text
    assert "circuit_scores: null" in text
    workflow = (ROOT / "src" / "repair" / "workflow.py").read_text(encoding="utf-8")
    assert "needs an in-process model" not in workflow


def test_import_order_state_refuses_transformer_lens_first():
    import sys
    import types

    saved_lens = sys.modules.get("transformer_lens")
    saved_circuit = sys.modules.get("circuitkit")
    sys.modules["transformer_lens"] = types.ModuleType("transformer_lens")
    sys.modules.pop("circuitkit", None)
    try:
        with pytest.raises(RepairError, match="before") as caught:
            assert_import_order_state()
        assert caught.value.stage == "circuitkit"
    finally:
        if saved_lens is None:
            sys.modules.pop("transformer_lens", None)
        else:
            sys.modules["transformer_lens"] = saved_lens
        if saved_circuit is None:
            sys.modules.pop("circuitkit", None)
        else:
            sys.modules["circuitkit"] = saved_circuit


def test_passing_credit_still_refuses_an_unpinned_template_before_the_factory(tmp_path: Path):
    parity = tmp_path / "parity"
    _good_parity(parity)
    with pytest.raises(RepairError, match="unpinned chat template") as caught:
        locate(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            index_path=tmp_path / "index.json",
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            artifact_kind="run",
            backend="circuitkit",
            parity_run=parity,
            runtime_config_path=RUNTIME,
            model_factory=_refuse_factory,
            prompt_loader=_refuse_prompts,
        )
    assert caught.value.stage == "provenance"
    assert "in-process model" not in str(caught.value)


def test_test_member_and_relabel_fail_before_factory(tmp_path: Path):
    parity = tmp_path / "parity"
    _good_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    test_row = _row("test", "harm", override_split="calibration")
    benign = _row("calibration", "benign")
    index = tmp_path / "index.json"
    _write_index(index, [test_row, benign])
    with pytest.raises(RepairError, match="test") as caught:
        locate_with_model(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            runtime_config_path=runtime,
            index_path=index,
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            parity_run=parity,
            circuit_scores={"MLP 0": 1.0},
            model_factory=_refuse_factory,
            prompt_loader=_refuse_prompts,
            code_check=_refuse_factory,
            source_check=_refuse_factory,
        )
    assert caught.value.stage == "membership"

    relabel = _row("dev", "harm", override_split="calibration")
    _write_index(index, [relabel, benign])
    with pytest.raises(RepairError, match="relabel") as relabelled:
        locate_with_model(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            runtime_config_path=runtime,
            index_path=index,
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            parity_run=parity,
            circuit_scores={"MLP 0": 1.0},
            model_factory=_refuse_factory,
            prompt_loader=_refuse_prompts,
            code_check=_refuse_factory,
            source_check=_refuse_factory,
        )
    assert relabelled.value.stage == "membership"


def test_derived_lineage_is_required_and_canonical_rows_cannot_borrow_it(tmp_path: Path):
    parity = tmp_path / "parity"
    _good_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    derived = _row("dev", "harm", override_split="calibration")
    derived["source_dataset"] = "reviewed_variant"
    derived["review_status"] = "reviewed"
    benign = _row("calibration", "benign")
    index = tmp_path / "index.json"
    _write_index(index, [derived, benign])
    with pytest.raises(RepairError, match="reviewed lineage") as missing:
        locate_with_model(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            runtime_config_path=runtime,
            index_path=index,
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            parity_run=parity,
            circuit_scores={"MLP 0": 1.0},
            model_factory=_refuse_factory,
            prompt_loader=_refuse_prompts,
        )
    assert missing.value.stage == "membership"

    canonical = [_row("calibration", "harm"), _row("calibration", "benign")]
    _write_index(index, canonical)
    lineage = tmp_path / "lineage.json"
    lineage.write_text("{}", encoding="utf-8")
    with pytest.raises(RepairError, match="relabel") as borrowed:
        locate_with_model(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            runtime_config_path=runtime,
            index_path=index,
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            parity_run=parity,
            circuit_scores={"MLP 0": 1.0},
            reviewed_lineage_path=lineage,
            model_factory=_refuse_factory,
            prompt_loader=_refuse_prompts,
        )
    assert borrowed.value.stage == "membership"


def test_variant_credit_does_not_unlock_the_weight_path(tmp_path: Path):
    parity = tmp_path / "variant"
    _variant_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    provenance = tmp_path / "provenance.yaml"
    _accepted_provenance(provenance)
    with pytest.raises(RepairError, match="original fp32") as caught:
        locate_with_model(
            protocol_path=PROTOCOL,
            provenance_path=provenance,
            runtime_config_path=runtime,
            index_path=tmp_path / "index.json",
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            parity_run=parity,
            circuit_scores={"MLP 0": 1.0},
            model_factory=_refuse_factory,
            prompt_loader=_refuse_prompts,
        )
    assert caught.value.stage == "parity"
    assert "in-process model" not in str(caught.value)


def test_fixture_artifact_cannot_authorize_a_load(tmp_path: Path):
    parity = tmp_path / "parity"
    _good_parity(parity)
    with pytest.raises(RepairError, match="fixture artifact cannot authorize") as caught:
        locate(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            index_path=tmp_path / "index.json",
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            prompts_path=tmp_path / "prompts.jsonl",
            out_path=tmp_path / "direction.json",
            artifact_kind="fixture",
            backend="circuitkit",
            parity_run=parity,
            runtime_config_path=RUNTIME,
            model_factory=_refuse_factory,
        )
    assert caught.value.stage == "provenance"


def test_alpha_zero_matches_base_and_hooks_clear_on_error_and_cancel():
    vector = [20.0, 0.0, 0.0, 0.0]
    model = DeterministicTensorFixture()
    base = generate_greedy(
        model, [1, 4], vector=None, alpha=None, layer_name=LAYER_NAME, hooks=False, max_new_tokens=4, eos_ids=set()
    )
    zero = generate_greedy(
        model, [1, 4], vector=vector, alpha=0.0, layer_name=LAYER_NAME, hooks=True, max_new_tokens=4, eos_ids=set()
    )
    candidate = generate_greedy(
        model, [1, 4], vector=vector, alpha=1.0, layer_name=LAYER_NAME, hooks=True, max_new_tokens=4, eos_ids=set()
    )
    assert zero["new_token_ids"] == base["new_token_ids"]
    assert candidate["new_token_ids"] != base["new_token_ids"]
    assert zero["hook_count"] >= 1
    assert model.hook_count() == 0

    class Boom(DeterministicTensorFixture):
        def step(self, token_ids: list[int]) -> list[float]:
            raise RuntimeError("boom")

    broken = Boom()
    with pytest.raises(RuntimeError, match="boom"):
        generate_greedy(
            broken, [1], vector=vector, alpha=1.0, layer_name=LAYER_NAME, hooks=True, max_new_tokens=2, eos_ids=set()
        )
    assert broken.hook_count() == 0

    class Cancel(DeterministicTensorFixture):
        def step(self, token_ids: list[int]) -> list[float]:
            raise KeyboardInterrupt

    cancelled = Cancel()
    with pytest.raises(KeyboardInterrupt):
        generate_greedy(
            cancelled,
            [1],
            vector=vector,
            alpha=1.0,
            layer_name=LAYER_NAME,
            hooks=True,
            max_new_tokens=2,
            eos_ids=set(),
        )
    assert cancelled.hook_count() == 0


def test_injected_ledger_unloads_the_target_before_the_judge(tmp_path: Path):
    parity = tmp_path / "parity"
    _good_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    calibration = [_row("calibration", "harm"), _row("calibration", "benign")]
    dev = [_row("dev", "harm"), _row("dev", "benign")]
    cal_index = tmp_path / "cal.json"
    dev_index = tmp_path / "dev.json"
    _write_index(cal_index, calibration)
    _write_index(dev_index, dev)
    events: list[str] = []

    def factory(_runtime, _source):
        return LifecycleBundle(events)

    def prompts(_path, rows):
        events.append("prompts")
        return _prompts(rows)

    direction = locate_with_model(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        index_path=cal_index,
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        prompts_path=tmp_path / "cal-prompts.jsonl",
        out_path=tmp_path / "direction.json",
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=factory,
        prompt_loader=prompts,
        code_check=_clean,
        source_check=_source_ok,
    )
    assert events[0] == "target"
    assert "prompts" in events
    assert events.index("target") < events.index("prompts")
    assert events[-1] == "target_unloaded"
    assert direction["artifact_kind"] == "fixture"
    assert direction["split"] == "example"
    assert direction["model_id"] == "fixture/injected"
    assert direction["not_tiny_aya"] is True
    assert direction["weights_executed_by_this_command"] is False
    assert direction["primary_asr"] is None
    assert direction["metrics_eligible"] is False
    assert direction["proven_refusal_mechanism"] is False
    assert direction["activation_steering"]["ran"] is False
    assert direction["activation_steering"]["not_the_protocol_resid_vector"] is True

    judged: list[str] = []

    class Judge:
        def judge_one(self, *_args, **_kwargs):
            raise AssertionError("missing response text was judged")

        def unload(self) -> None:
            judged.append("judge_unloaded")

    def judge_factory(_runtime):
        assert events[-1] == "target_unloaded"
        judged.append("judge")
        return Judge()

    events.clear()
    report = run_real_sweep(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        sweep_path=SWEEP,
        direction_path=tmp_path / "direction.json",
        calibration_index_path=cal_index,
        dev_index_path=dev_index,
        dev_prompts_path=tmp_path / "dev-prompts.jsonl",
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        out_path=tmp_path / "sweep.json",
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=factory,
        judge_factory=judge_factory,
        prompt_loader=prompts,
        code_check=_clean,
        source_check=_source_ok,
    )
    assert events == ["target", "prompts", "target_unloaded"]
    assert judged == ["judge", "judge_unloaded"]
    assert report["artifact_kind"] == "fixture"
    assert report["split"] == "example"
    assert report["model_id"] == "fixture/injected"
    assert report["not_tiny_aya"] is True
    assert report["primary_asr"] is None
    assert report["metrics_eligible"] is False
    assert report["generation"] == "per_step_hook_forward"
    assert report["not_a_single_steer_call"] is True
    assert report["library_steer_per_new_token"] is False
    assert report["library_call"] is None
    assert report["alpha_zero_matches_base"] is True
    assert report["judge"]["primary_asr"] is None
    assert report["judge"]["judge_constructed"] is True
    assert report["judge"]["judge_unloaded"] is True
    assert report["freeze_allowed"] is False


class _FakeTorch:
    long = "long"
    float32 = "float32"

    def tensor(self, data, dtype=None):
        return {"data": data, "dtype": dtype}


class RealWeightStandIn(LifecycleBundle):
    """real_weights is true so the sweep must call ActivationSteering, not the residual hook."""

    real_weights = True

    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.model = object()
        self.torch = _FakeTorch()
        self.hook_installs = 0

    def add_hook(self, name: str, fn, dir: str = "fwd") -> None:
        self.hook_installs += 1
        super().add_hook(name, fn, dir)

    def encode_user(self, prompt: str) -> list[int]:
        if prompt == "harm":
            return [5, 5]
        if prompt == "benign":
            return [1, 1]
        raise RepairError("unexpected prompt", stage="runtime")


def _text_prompts(_path, rows: list[dict]) -> list[dict]:
    loaded = []
    for row in rows:
        item = dict(row)
        item["prompt"] = row["hs_label"]
        loaded.append(item)
    return loaded


def _judge_factory(_runtime):
    class Judge:
        def judge_one(self, *_args, **_kwargs):
            raise AssertionError("missing response text was judged")

        def unload(self) -> None:
            return None

    return Judge()


def test_unknown_command_returns_two(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    class Args:
        command = "not-a-command"

    class Parser:
        def parse_args(self, _argv):
            return Args()

    monkeypatch.setattr("src.repair.cli.build_parser", Parser)
    assert main([]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_unreadable_freeze_child_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    missing = tmp_path / "missing.json"
    assert main(["verify-freeze-child", str(missing)]) == 2
    assert "unreadable" in capsys.readouterr().err
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert main(["verify-freeze-child", str(bad)]) == 2


def test_real_weights_refuse_the_residual_hook():
    model = DeterministicTensorFixture()
    model.real_weights = True
    with pytest.raises(RepairError, match="residual hook") as caught:
        generate_greedy(
            model,
            [1],
            vector=[1.0, 0.0, 0.0, 0.0],
            alpha=1.0,
            layer_name=LAYER_NAME,
            hooks=True,
            max_new_tokens=1,
            eos_ids=set(),
        )
    assert caught.value.stage == "runtime"
    assert model.hook_count() == 0


def test_real_weights_refuse_a_sweep_without_library_vectors(tmp_path: Path):
    parity = tmp_path / "parity"
    _good_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    calibration = [_row("calibration", "harm"), _row("calibration", "benign")]
    dev = [_row("dev", "harm"), _row("dev", "benign")]
    cal_index = tmp_path / "cal.json"
    dev_index = tmp_path / "dev.json"
    _write_index(cal_index, calibration)
    _write_index(dev_index, dev)
    events: list[str] = []
    direction = locate_with_model(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        index_path=cal_index,
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        prompts_path=tmp_path / "cal-prompts.jsonl",
        out_path=tmp_path / "direction.json",
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=lambda _runtime, _source: LifecycleBundle(events),
        prompt_loader=lambda _path, rows: _prompts(rows),
        code_check=_clean,
        source_check=_source_ok,
    )
    assert direction["activation_steering"]["ran"] is False
    with pytest.raises(RepairError, match="residual hook") as caught:
        run_real_sweep(
            protocol_path=PROTOCOL,
            provenance_path=PROVENANCE,
            runtime_config_path=runtime,
            sweep_path=SWEEP,
            direction_path=tmp_path / "direction.json",
            calibration_index_path=cal_index,
            dev_index_path=dev_index,
            dev_prompts_path=tmp_path / "dev-prompts.jsonl",
            manifest_path=CANONICAL_FAMILY_MANIFEST,
            out_path=tmp_path / "sweep.json",
            parity_run=parity,
            circuit_scores={"MLP 0": 1.0},
            model_factory=lambda _runtime, _source: RealWeightStandIn(events),
            judge_factory=_judge_factory,
            prompt_loader=_text_prompts,
            code_check=_clean,
            source_check=_source_ok,
        )
    assert caught.value.stage == "circuitkit"


def test_real_weights_steer_with_activation_steering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict] = []

    def fake_extract(_model, scores, source_examples, target_examples, *, credit):
        assert credit["credit"] is True
        assert scores == {"MLP 0": 1.0}
        assert source_examples and target_examples
        return {"nodes": ["MLP 0"], "vectors": {"MLP 0": [0.0, 1.0, 0.0, 0.0]}}

    def fake_steer(_model, scores, vectors, inputs, coefficient, *, credit):
        assert credit["credit"] is True
        assert scores == {"MLP 0": 1.0}
        assert isinstance(inputs, dict) and inputs["dtype"] == "long"
        assert isinstance(inputs["data"][0], list)
        vector = vectors["MLP 0"]
        assert vector["dtype"] == "float32"
        calls.append({"coefficient": float(coefficient), "vector": tuple(vector["data"])})
        if float(coefficient) == 0.0:
            output = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            output = [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        return {"effective_hook_count": 2, "steered_nodes": ["MLP 0"], "output": output}

    monkeypatch.setattr("src.repair.circuitkit_backend.extract_last_token", fake_extract)
    monkeypatch.setattr("src.repair.circuitkit_backend.steer_coefficient", fake_steer)
    parity = tmp_path / "parity"
    _good_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    calibration = [_row("calibration", "harm"), _row("calibration", "benign")]
    dev = [_row("dev", "harm"), _row("dev", "benign")]
    cal_index = tmp_path / "cal.json"
    dev_index = tmp_path / "dev.json"
    _write_index(cal_index, calibration)
    _write_index(dev_index, dev)
    events: list[str] = []

    def factory(_runtime, _source):
        return RealWeightStandIn(events)

    direction = locate_with_model(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        index_path=cal_index,
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        prompts_path=tmp_path / "cal-prompts.jsonl",
        out_path=tmp_path / "direction.json",
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=factory,
        prompt_loader=_text_prompts,
        code_check=_clean,
        source_check=_source_ok,
    )
    assert direction["activation_steering"]["ran"] is True
    assert direction["activation_steering"]["call"] == "ActivationSteering.compute_steering_vector"
    assert direction["activation_steering"]["vectors_as_lists"] == {"MLP 0": [0.0, 1.0, 0.0, 0.0]}
    assert direction["activation_steering"]["protocol_resid_not_used_for_steering"] is True
    assert direction["not_tiny_aya"] is True
    assert direction["artifact_kind"] == "fixture"
    assert direction["primary_asr"] is None
    events.clear()
    handles: list[RealWeightStandIn] = []

    def sweep_factory(_runtime, _source):
        handle = RealWeightStandIn(events)
        handles.append(handle)
        return handle

    report = run_real_sweep(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        sweep_path=SWEEP,
        direction_path=tmp_path / "direction.json",
        calibration_index_path=cal_index,
        dev_index_path=dev_index,
        dev_prompts_path=tmp_path / "dev-prompts.jsonl",
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        out_path=tmp_path / "sweep.json",
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=sweep_factory,
        judge_factory=_judge_factory,
        prompt_loader=_text_prompts,
        code_check=_clean,
        source_check=_source_ok,
    )
    assert calls
    assert {item["coefficient"] for item in calls} == {0.0, 1.0}
    library_vector = (0.0, 1.0, 0.0, 0.0)
    steered = {item["vector"] for item in calls if item["coefficient"] == 1.0}
    assert library_vector in steered
    assert any(vector != library_vector for vector in steered)
    assert events[-1] == "target_unloaded"
    assert handles[-1].hook_installs == 0
    assert report["generation"] == "ActivationSteering.steer"
    assert report["library_steer_per_new_token"] is True
    assert report["not_a_single_steer_call"] is False
    assert report["library_call"] == "ActivationSteering.steer"
    assert report["artifact_kind"] == "fixture"
    assert report["model_id"] == "fixture/injected"
    assert report["primary_asr"] is None
    assert report["metrics_eligible"] is False
    assert report["not_tiny_aya"] is True
    assert report["alpha_zero_matches_base"] is True
    assert report["freeze_allowed"] is False


def test_library_freeze_refuses_a_fixture_label(tmp_path: Path):
    payload = {
        "generation": "ActivationSteering.steer",
        "library_steer_per_new_token": True,
        "not_a_single_steer_call": False,
        "artifact_kind": "fixture",
        "factory_injected": True,
        "not_tiny_aya": True,
        "weights_executed_by_this_command": False,
    }
    payload["content_sha256"] = content_sha256(payload)
    direction = {"vector": [1.0, 0.0]}
    direction["content_sha256"] = content_sha256(direction)
    sweep_path = tmp_path / "sweep.json"
    direction_path = tmp_path / "direction.json"
    sweep_path.write_text(json.dumps(payload), encoding="utf-8")
    direction_path.write_text(json.dumps(direction), encoding="utf-8")
    with pytest.raises(RepairError, match="real-weight") as caught:
        build_measured_freeze(sweep_path, direction_path)
    assert caught.value.stage == "freeze"


def _measured_shell(tmp_path: Path, judgments: dict, producer: dict | None) -> tuple[Path, Path]:
    from src.repair.protocol import load_sweep_config

    sweep_config = load_sweep_config(ROOT / "configs" / "repair" / "sweep.yaml")
    direction = {"vector": [1.0, 0.0], "layer_index": 0, "not_tiny_aya": True}
    direction["content_sha256"] = content_sha256(direction)
    payload = {
        "generation": "per_step_hook_forward",
        "direction_sha256": direction["content_sha256"],
        "primary_asr": None,
        "metrics_eligible": False,
        "proven_refusal_mechanism": False,
        "dev_members": [
            {"sample_id": "s1", "family_id": "f", "hs_label": "harm", "split": "dev"},
            {"sample_id": "b1", "family_id": "f", "hs_label": "benign", "split": "dev"},
        ],
        "judgments": judgments,
        "producer": producer,
        "sweep_sha256": sweep_config["sha256"],
        "decision": {"decision": "accept", "freeze": True},
        "artifact_kind": "fixture",
        "model_id": "fixture/injected",
        "model_revision": "not-loaded",
        "judge_model_id": "allenai/wildguard",
        "judge_revision": "cbba4823f3e8020e5a74a5e29bf85072def6f2ff",
    }
    payload["content_sha256"] = content_sha256(payload)
    sweep_path = tmp_path / "sweep.json"
    direction_path = tmp_path / "direction.json"
    sweep_path.write_text(json.dumps(payload), encoding="utf-8")
    direction_path.write_text(json.dumps(direction), encoding="utf-8")
    return sweep_path, direction_path


def test_measured_freeze_rejects_judgments_that_are_not_on_the_ledger(tmp_path: Path):
    from src.repair.selection import primary_rubric, producer_from_ledger

    sweep_path, direction_path = _measured_shell(tmp_path, {"capability_unit": "task_points", "rows": []}, None)
    with pytest.raises(RepairError, match="generation ledger is missing"):
        build_measured_freeze(sweep_path, direction_path)

    records = []
    for arm in ("base_no_intervention", "candidate_alpha_1"):
        for sample_id, hs_label, text in (("s1", "harm", "harm text"), ("b1", "benign", "benign text")):
            records.append({
                "arm_id": arm,
                "sample_id": sample_id,
                "family_id": "f",
                "hs_label": hs_label,
                "new_token_ids": [1],
                "response": text,
                "status": "ok",
            })
    generations = tmp_path / "generations.jsonl"
    samples = tmp_path / "samples.jsonl"
    generations.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    labeled = [{**row, "judge_label": "refusal"} for row in records]
    samples.write_text("".join(json.dumps(row) + "\n" for row in labeled), encoding="utf-8")
    identity = {
        "model_id": "fixture/injected",
        "model_revision": "not-loaded",
        "judge_model_id": "allenai/wildguard",
        "judge_revision": "cbba4823f3e8020e5a74a5e29bf85072def6f2ff",
        "rubric": primary_rubric(),
    }
    producer = producer_from_ledger(generations, samples, identity=identity)
    rows = []
    for item in producer["responses"]:
        rows.append({
            "arm_id": item["arm_id"],
            "sample_id": item["sample_id"],
            "family_id": "f",
            "split": "dev",
            "hs_label": "harm" if item["sample_id"] == "s1" else "benign",
            "judge_label": item["judge_label"],
            "capability_score": None if item["sample_id"] == "s1" else 10,
            "response_sha256": "0" * 64 if item["sample_id"] == "s1" else item["response_sha256"],
            "run_id": producer["run_id"],
            "model_id": producer["model_id"],
            "model_revision": producer["model_revision"],
            "judge_model_id": producer["judge_model_id"],
            "judge_revision": producer["judge_revision"],
            "rubric": producer["rubric"],
        })
    sweep_path, direction_path = _measured_shell(
        tmp_path,
        {"capability_unit": "task_points", "rows": rows},
        producer,
    )
    with pytest.raises(RepairError, match="response hash"):
        build_measured_freeze(sweep_path, direction_path)


def test_accept_requires_the_family_floor_and_recomputed_evidence(tmp_path: Path):
    forged = {
        "decision": "accept",
        "harm_improvement": {"low": 0.2},
        "over_refusal_gate": "pass",
        "capability_gate": "pass",
        "n_families": 1,
        "min_families_for_accept": 100,
        "claim_level": "eligible",
        "deployable_claim": True,
        "freeze": True,
    }
    with pytest.raises(RepairError, match="family count") as caught:
        assert_accept_bindings(forged, min_families=100)
    assert caught.value.stage == "freeze"

    parity = tmp_path / "parity"
    _good_parity(parity)
    runtime = tmp_path / "runtime.yaml"
    _runtime(runtime)
    calibration = [_row("calibration", "harm"), _row("calibration", "benign")]
    dev = [_row("dev", "harm"), _row("dev", "benign")]
    cal_index = tmp_path / "cal.json"
    dev_index = tmp_path / "dev.json"
    _write_index(cal_index, calibration)
    _write_index(dev_index, dev)
    events: list[str] = []

    def factory(_runtime, _source):
        return LifecycleBundle(events)

    direction_path = tmp_path / "direction.json"
    locate_with_model(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        index_path=cal_index,
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        prompts_path=tmp_path / "cal-prompts.jsonl",
        out_path=direction_path,
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=factory,
        prompt_loader=lambda _path, rows: _prompts(rows),
        code_check=_clean,
        source_check=_source_ok,
    )
    sweep_path = tmp_path / "sweep.json"
    run_real_sweep(
        protocol_path=PROTOCOL,
        provenance_path=PROVENANCE,
        runtime_config_path=runtime,
        sweep_path=SWEEP,
        direction_path=direction_path,
        calibration_index_path=cal_index,
        dev_index_path=dev_index,
        dev_prompts_path=tmp_path / "dev-prompts.jsonl",
        manifest_path=CANONICAL_FAMILY_MANIFEST,
        out_path=sweep_path,
        parity_run=parity,
        circuit_scores={"MLP 0": 1.0},
        model_factory=factory,
        prompt_loader=lambda _path, rows: _prompts(rows),
        code_check=_clean,
        source_check=_source_ok,
    )
    with pytest.raises(RepairError, match="unavailable") as unavailable:
        build_measured_freeze(sweep_path, direction_path)
    assert unavailable.value.stage == "freeze"
    payload = json.loads(sweep_path.read_text(encoding="utf-8"))
    payload["decision"] = {"decision": "accept", "freeze": True}
    payload["content_sha256"] = content_sha256(payload)
    sweep_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RepairError, match="stored decision") as mismatch:
        build_measured_freeze(sweep_path, direction_path)
    assert mismatch.value.stage == "freeze"


def test_tensor_fixture_reloads_in_a_fresh_process(tmp_path: Path):
    out = tmp_path / "fixture"
    assert main(["tensor-fixture", "--out-dir", str(out)]) == 0
    payload = json.loads((out / "fixture_results.json").read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "fixture"
    assert payload["split"] == "example"
    assert payload["primary_asr"] is None
    assert payload["not_pretrained_weights"] is True
    assert payload["tensor_kind"] == "deterministic_last_token_function"
    assert main(["verify-freeze", "--manifest", str(out / "fixture_results.json")]) == 0
