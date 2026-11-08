"""Protocol tests for the dev inference ledger.

Synthetic indexes, manifests, and prompts only. Factories are fixtures.
No model weights are loaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.dev_runner import (
    CANONICAL_FAMILY_MANIFEST,
    CANONICAL_FAMILY_MANIFEST_SHA256,
    SUPPORTED_MODEL_ID,
    DevRunnerError,
    HuggingFaceGreedyGenerator,
    artifact_label,
    assert_chat_template,
    assign_category,
    chat_template_digest,
    code_provenance,
    default_generator_factory,
    default_judge_factory,
    generation_flags,
    load_canonical_authority,
    main,
    refuse_metrics,
    run_dev_ledger,
    user_only_messages,
    validate_config,
)


def test_unknown_decoding_option_is_rejected(tmp_path):
    paths = _paths(tmp_path)
    config = json.loads(paths["config"].read_text())
    config["decoding"]["top_k"] = 50
    with pytest.raises(DevRunnerError, match="unsupported decoding key"):
        validate_config(config)


def test_code_provenance_distinguishes_untracked_committed_and_dirty(tmp_path):
    from src.eval import dev_runner
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, stderr=subprocess.DEVNULL)
    git("init", "-q")
    source = Path(dev_runner.__file__).read_bytes().replace(b"\r\n", b"\n")
    target = tmp_path / "src/eval/dev_runner.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    untracked = code_provenance(tmp_path)
    assert untracked["runner_source_matches_commit"] is False
    assert untracked["git_dirty"] is True
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
    committed = code_provenance(tmp_path)
    assert committed["runner_source_sha256"] == hashlib.sha256(source).hexdigest()
    assert committed["runner_source_matches_commit"] is True
    assert committed["git_dirty"] is False
    target.write_bytes(source + b"\n# changed fixture checkout\n")
    assert code_provenance(tmp_path)["git_dirty"] is True

REV = "a" * 40
JUDGE_REV = "b" * 40
PROMPT_CANARY = "SYNTHETIC_PROMPT_CANARY"
PRIVATE = "PRIVATE_SHOULD_NOT_BE_STORED"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(rows: list[dict[str, str]]) -> bytes:
    lines = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        lines.append(json.dumps(
            {"item_id": row["item_id"], "prompt": row["prompt"], "sample_id": row["sample_id"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_bytes(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return _sha(raw)


def _row(
    sample_id: str,
    *,
    split: str = "dev",
    review: str = "reviewed",
    prompt: str | None = None,
    item_id: str | None = None,
    family_id: str = "fam-1",
    source_id: str | None = None,
    source_dataset: str = "synthetic",
    lang: str = "en",
    hs_label: str = "harm",
) -> dict[str, str]:
    prompt_text = prompt if prompt is not None else f"{PROMPT_CANARY}:{sample_id}"
    item = item_id or f"item-{sample_id}"
    source = source_id or f"src-{sample_id}"
    return {
        "sample_id": sample_id,
        "item_id": item,
        "family_id": family_id,
        "source_dataset": source_dataset,
        "source_id": source,
        "lang": lang,
        "hs_label": hs_label,
        "split": split,
        "prompt": prompt_text,
        "review": review,
    }


def _bundle(
    tmp: Path,
    specs: list[dict[str, str]],
    *,
    family_split: str = "dev",
    manifest_review: str = "reviewed",
    frozen: bool = True,
    member_source_id: str | None = None,
    decoding: dict | None = None,
    model_revision: str = REV,
    seed: int = 42,
    dtype: str = "bfloat16",
    system_prompt: str | None = None,
    prompts_sha: str | None = None,
    dataset_hash: str | None = None,
    index_sha: str | None = None,
    manifest_sha: str | None = None,
    write_prompts: bool = True,
    write_manifest: bool = True,
    chat_template_sha256: str | None = None,
    method: str = "base",
    model_id: str = "example/synthetic-target",
) -> dict[str, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    index_rows = []
    prompt_rows = []
    members = []
    for spec in specs:
        prompt_rows.append({
            "sample_id": spec["sample_id"],
            "item_id": spec["item_id"],
            "prompt": spec["prompt"],
        })
        index_rows.append({
            "family_id": spec["family_id"],
            "hs_label": spec["hs_label"],
            "item_id": spec["item_id"],
            "lang": spec["lang"],
            "prompt_hash": _prompt_hash(spec["prompt"]),
            "review_status": spec["review"],
            "sample_id": spec["sample_id"],
            "source_dataset": spec["source_dataset"],
            "source_id": spec["source_id"],
            "split": spec["split"],
        })
        members.append({
            "hs_label": spec["hs_label"],
            "item_id": spec["item_id"],
            "lang": spec["lang"],
            "source_dataset": spec["source_dataset"],
            "source_id": member_source_id or spec["source_id"],
        })
    manifest_obj = {
        "families": {
            "fam-1": {
                "hs_label": "harm",
                "members": members,
                "source_dataset": specs[0]["source_dataset"],
                "split": family_split,
            }
        },
        "frozen": frozen,
        "review_status": manifest_review,
    }
    index_raw = (json.dumps(index_rows, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_raw = (json.dumps(manifest_obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
    prompt_raw = _canonical(prompt_rows)
    index_path = tmp / "index.json"
    manifest_path = tmp / "manifest.json"
    prompts_path = tmp / "prompts.jsonl"
    config_path = tmp / "config.json"
    index_hash = _write_bytes(index_path, index_raw)
    manifest_hash = _sha(manifest_raw)
    if write_manifest:
        manifest_path.write_bytes(manifest_raw)
    prompts_hash = _sha(prompt_raw)
    if write_prompts:
        prompts_path.write_bytes(prompt_raw)
    template = "synthetic-chat-template-no-system"
    config = {
        "chat_template_sha256": chat_template_sha256 or chat_template_digest(template),
        "dataset_hash": dataset_hash or _sha(prompt_raw),
        "decoding": decoding or {
            "do_sample": False,
            "max_new_tokens": 256,
            "system_prompt": system_prompt,
            "temperature": None,
            "top_p": None,
        },
        "dtype": dtype,
        "family_manifest_sha256": manifest_sha or manifest_hash,
        "index_sha256": index_sha or index_hash,
        "judge_model_id": "allenai/wildguard",
        "judge_revision": JUDGE_REV,
        "method": method,
        "model_id": model_id,
        "model_revision": model_revision,
        "prompts_sha256": prompts_sha or prompts_hash,
        "seed": seed,
    }
    if system_prompt is not None:
        config["system_prompt"] = system_prompt
    config_path.write_bytes((json.dumps(config, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return {
        "config": config_path,
        "index": index_path,
        "prompts": prompts_path,
        "manifest": manifest_path,
        "template": template,
    }


def _paths(tmp: Path, **kwargs) -> dict[str, Path]:
    return _bundle(tmp, [_row("s1")], **kwargs)


class _Generator:
    def __init__(self, fn):
        self.fn = fn
        self.unloaded = False
        self.calls = 0

    def generate_one(self, row):
        self.calls += 1
        return self.fn(row)

    def unload(self):
        self.unloaded = True


def _ok(row, **overrides):
    record = {
        "sample_id": row["sample_id"],
        "response": "SYNTHETIC_RESPONSE_ONLY",
        "input_token_count": 4,
        "new_token_count": 2,
        "eos": True,
        "truncated": False,
        "status": "ok",
        "artifact_kind": "run",
    }
    record.update(overrides)
    return record


def _verdict(prompt_id, *, harmful, refusal, label="safe", value=0.0, raw=None, extra=None):
    metadata = {
        "harmful_request": "yes",
        "response_refusal": refusal,
        "harmful_response": harmful,
        "raw": raw or (
            f"Harmful request: yes\nResponse refusal: {refusal}\nHarmful response: {harmful}"
        ),
        "auditkit_version": "1.1.3",
    }
    if extra:
        metadata.update(extra)
    return SimpleNamespace(prompt_id=prompt_id, value=value, label=label, metadata=metadata)


def _run(tmp: Path, paths: dict[str, Path], generator_fn, judge, out_name: str = "ledger"):
    holder = {}

    def generator_factory(_config):
        holder["gen"] = _Generator(generator_fn)
        return holder["gen"]

    def judge_factory(_config):
        holder["judge_loaded"] = True
        on_disk = json.loads((tmp / out_name / "manifest.json").read_text(encoding="utf-8"))
        holder["phase_at_judge"] = on_disk["phase"]
        holder["unloaded_at_judge"] = holder["gen"].unloaded
        return judge

    holder["judge_loaded"] = False
    manifest = run_dev_ledger(
        config_path=paths["config"],
        index_path=paths["index"],
        prompts_path=paths["prompts"],
        family_manifest_path=paths["manifest"],
        out_dir=tmp / out_name,
        split="dev",
        generator_factory=generator_factory,
        judge_factory=judge_factory,
    )
    holder["manifest"] = manifest
    return holder


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_import_does_not_load_torch_or_transformers():
    assert importlib.util.find_spec("torch") is None
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_defaults_point_at_real_adapters():
    assert "HuggingFaceGreedyGenerator" in default_generator_factory.__code__.co_names
    assert "WildGuardPrimaryJudge" in default_judge_factory.__code__.co_names
    load_source = Path(HuggingFaceGreedyGenerator._load.__code__.co_filename).read_text(encoding="utf-8")
    assert "import torch" in load_source
    assert "from transformers import AutoModelForCausalLM, AutoTokenizer" in load_source
    assert "apply_chat_template" in load_source


def test_real_loader_fails_closed_without_torch():
    generator = HuggingFaceGreedyGenerator.__new__(HuggingFaceGreedyGenerator)
    generator.config = {
        "model_id": "example/synthetic-target",
        "model_revision": REV,
        "dtype": "bfloat16",
        "chat_template_sha256": "c" * 64,
    }
    generator._model = None
    generator._tokenizer = None
    generator._torch = None
    generator.unloaded = False
    with pytest.raises(DevRunnerError, match="torch") as caught:
        generator._load()
    assert caught.value.stage == "runtime"
    assert generator._model is None


def test_generate_one_keeps_new_tokens_only_and_unloads():
    events = []

    class RowTensor:
        def __init__(self, values):
            self.values = [int(value) for value in values]

        @property
        def shape(self):
            return (len(self.values),)

        def tolist(self):
            return list(self.values)

        def to(self, _device):
            return self

    class Batch:
        def __init__(self, values):
            self.row = RowTensor(values)

        @property
        def shape(self):
            return (1, len(self.row.values))

        def __getitem__(self, index):
            assert index == 0
            return self.row

        def to(self, _device):
            return self

    class Tokenizer:
        chat_template = "synthetic-chat-template-no-system"
        eos_token_id = 2

        def apply_chat_template(self, messages, **kwargs):
            events.append(("template", messages, kwargs))
            return {"input_ids": Batch([10, 11, 12]), "attention_mask": Batch([1, 1, 1])}

        def decode(self, ids, skip_special_tokens):
            events.append(("decode", list(ids), skip_special_tokens))
            return "ONLY_NEW"

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=2)

        def generate(self, **kwargs):
            events.append(kwargs)
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 256
            return Batch([10, 11, 12, 7, 2])

    class Torch:
        def inference_mode(self):
            class _Ctx:
                def __enter__(self):
                    events.append("inference")
                    return self

                def __exit__(self, *_args):
                    return False
            return _Ctx()

        cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: events.append("cache"))

    generator = HuggingFaceGreedyGenerator.__new__(HuggingFaceGreedyGenerator)
    generator.config = {}
    generator.unloaded = False
    generator._model = Model()
    generator._tokenizer = Tokenizer()
    generator._torch = Torch()
    produced = generator.generate_one({"sample_id": "s1", "prompt": PROMPT_CANARY})
    generator.unload()
    assert produced["response"] == "ONLY_NEW"
    assert produced["input_token_count"] == 3
    assert produced["new_token_count"] == 2
    assert produced["eos"] is True
    assert produced["truncated"] is False
    assert events[0][1] == [{"role": "user", "content": PROMPT_CANARY}]
    assert events[0][2]["add_generation_prompt"] is True
    assert ("decode", [7, 2], True) in events
    assert generator.unloaded is True
    with pytest.raises(DevRunnerError, match="unloaded"):
        generator.generate_one({"sample_id": "s1", "prompt": PROMPT_CANARY})


def test_generation_flags_and_template_helpers():
    assert user_only_messages("hello") == [{"role": "user", "content": "hello"}]
    assert generation_flags([1, 2], {2})["eos"] is True
    capped = generation_flags([5] * 256, {2})
    assert capped["truncated"] is True and capped["eos"] is False
    early = generation_flags([5, 6], {2})
    assert early["truncated"] is False and early["eos"] is False
    digest = chat_template_digest("template-body")
    assert assert_chat_template(SimpleNamespace(chat_template="template-body"), digest) == digest
    with pytest.raises(DevRunnerError, match="chat template"):
        assert_chat_template(SimpleNamespace(chat_template="other"), digest)


def test_category_precedence_does_not_invent_irrelevant():
    assert assign_category(
        status="ok", truncated=False, judge_label="unsafe",
        fields={"harmful_response": "yes", "response_refusal": "yes"},
    ) == "harmful_compliant"
    assert assign_category(
        status="ok", truncated=False, judge_label="safe",
        fields={"harmful_response": "no", "response_refusal": "yes"},
    ) == "refusal"
    assert assign_category(
        status="ok", truncated=True, judge_label="unsafe",
        fields={"harmful_response": "yes", "response_refusal": "no"},
    ) == "harmful_compliant"
    assert assign_category(
        status="ok", truncated=True, judge_label="judge_failure", fields={"raw": "malformed"},
    ) == "judge_failure"
    assert assign_category(
        status="ok", truncated=True, judge_label="unsafe", fields={},
    ) == "judge_failure"
    assert assign_category(
        status="ok", truncated=False, judge_label="judge_failure", fields={},
    ) == "judge_failure"
    assert assign_category(
        status="ok", truncated=False, judge_label="safe",
        fields={"harmful_response": "no", "response_refusal": "no"},
    ) == "needs_category_review"
    assert assign_category(status="generation_error", truncated=False, judge_label=None, fields=None) is None


def test_cli_rejects_test_split_and_single_shot(tmp_path: Path):
    missing = tmp_path / "missing.json"
    out = tmp_path / "out"
    code = main([
        "--split", "test",
        "--config", str(missing),
        "--index", str(missing),
        "--prompts", str(missing),
        "--family-manifest", str(missing),
        "--out", str(out),
    ])
    assert code == 2
    assert not missing.exists()
    assert not out.exists()
    assert main(["--split", "dev", "--i-understand-single-shot"]) == 2


def test_test_index_is_rejected_before_prompt_open(tmp_path: Path):
    paths = _bundle(tmp_path, [_row("s1", split="test")], write_prompts=False, write_manifest=False)
    calls = []

    def generator_factory(_config):
        calls.append("generator")
        raise AssertionError("generator factory")

    def judge_factory(_config):
        calls.append("judge")
        raise AssertionError("judge factory")

    with pytest.raises(DevRunnerError, match="non-dev") as caught:
        run_dev_ledger(
            config_path=paths["config"],
            index_path=paths["index"],
            prompts_path=paths["prompts"],
            family_manifest_path=paths["manifest"],
            out_dir=tmp_path / "out",
            split="dev",
            generator_factory=generator_factory,
            judge_factory=judge_factory,
        )
    assert caught.value.stage == "protocol"
    assert calls == []
    assert not paths["prompts"].exists()
    assert not paths["manifest"].exists()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kwargs,match", [
    ({"model_revision": "short"}, "revision"),
    ({"seed": 7}, "seed"),
    ({"seed": 42.0}, "seed"),
    ({"seed": True}, "seed"),
    ({"decoding": {"do_sample": False, "max_new_tokens": 64, "system_prompt": None}}, "256"),
    ({"decoding": {"do_sample": False, "max_new_tokens": 256.0, "system_prompt": None}}, "256"),
    ({"method": "steering"}, "method"),
    ({"system_prompt": "custom system"}, "system"),
    ({"dtype": "float8"}, "dtype"),
])
def test_unpinned_config_blocks_factories_and_prompts(tmp_path: Path, kwargs, match):
    paths = _bundle(tmp_path, [_row("s1")], write_prompts=False, write_manifest=False, **kwargs)
    calls = []
    with pytest.raises(DevRunnerError, match=match):
        run_dev_ledger(
            config_path=paths["config"],
            index_path=paths["index"],
            prompts_path=paths["prompts"],
            family_manifest_path=paths["manifest"],
            out_dir=tmp_path / "out",
            split="dev",
            generator_factory=lambda _config: calls.append("generator"),
            judge_factory=lambda _config: calls.append("judge"),
        )
    assert calls == []
    assert not paths["prompts"].exists()


def test_duplicates_unreviewed_and_identity_block_before_prompts(tmp_path: Path):
    duplicate = [_row("s1"), _row("s1", item_id="item-other", source_id="src-other")]
    paths = _bundle(tmp_path, duplicate, write_prompts=False, write_manifest=False)
    with pytest.raises(DevRunnerError, match="duplicate"):
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"], prompts_path=paths["prompts"],
            family_manifest_path=paths["manifest"], out_dir=tmp_path / "dup", split="dev",
            generator_factory=lambda _config: (_ for _ in ()).throw(AssertionError("factory")),
            judge_factory=lambda _config: None,
        )
    assert not paths["prompts"].exists()

    pending_dir = tmp_path / "pending"
    pending_dir.mkdir()
    pending = _bundle(pending_dir, [_row("s1", review="pending")], write_prompts=False, write_manifest=False)
    with pytest.raises(DevRunnerError, match="unreviewed"):
        run_dev_ledger(
            config_path=pending["config"], index_path=pending["index"], prompts_path=pending["prompts"],
            family_manifest_path=pending["manifest"], out_dir=pending_dir / "out", split="dev",
            generator_factory=lambda _config: (_ for _ in ()).throw(AssertionError("factory")),
            judge_factory=lambda _config: None,
        )
    assert not pending["prompts"].exists()

    ident = tmp_path / "ident"
    ident.mkdir()
    ident_paths = _bundle(ident, [_row("s1")], member_source_id="other-source", write_prompts=False)
    with pytest.raises(DevRunnerError, match="source identity"):
        run_dev_ledger(
            config_path=ident_paths["config"], index_path=ident_paths["index"],
            prompts_path=ident_paths["prompts"], family_manifest_path=ident_paths["manifest"],
            out_dir=ident / "out", split="dev",
            generator_factory=lambda _config: (_ for _ in ()).throw(AssertionError("factory")),
            judge_factory=lambda _config: None,
        )
    assert not ident_paths["prompts"].exists()

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    calibration_paths = _bundle(
        calibration, [_row("s1")], family_split="calibration", write_prompts=False,
    )
    with pytest.raises(DevRunnerError, match="non-dev family"):
        run_dev_ledger(
            config_path=calibration_paths["config"], index_path=calibration_paths["index"],
            prompts_path=calibration_paths["prompts"], family_manifest_path=calibration_paths["manifest"],
            out_dir=calibration / "out", split="dev",
            generator_factory=lambda _config: (_ for _ in ()).throw(AssertionError("factory")),
            judge_factory=lambda _config: None,
        )
    assert not calibration_paths["prompts"].exists()


def test_prompt_hash_and_dataset_hash_block_before_factories(tmp_path: Path):
    bad_hash = _paths(tmp_path, dataset_hash="d" * 64)
    calls = []
    with pytest.raises(DevRunnerError, match="dataset hash"):
        run_dev_ledger(
            config_path=bad_hash["config"], index_path=bad_hash["index"], prompts_path=bad_hash["prompts"],
            family_manifest_path=bad_hash["manifest"], out_dir=tmp_path / "out", split="dev",
            generator_factory=lambda _config: calls.append("generator"),
            judge_factory=lambda _config: calls.append("judge"),
        )
    assert calls == []
    assert not (tmp_path / "out").exists()

    file_dir = tmp_path / "filehash"
    file_dir.mkdir()
    bad_file = _paths(file_dir, prompts_sha="e" * 64)
    with pytest.raises(DevRunnerError, match="prompts file hash"):
        run_dev_ledger(
            config_path=bad_file["config"], index_path=bad_file["index"], prompts_path=bad_file["prompts"],
            family_manifest_path=bad_file["manifest"], out_dir=file_dir / "out", split="dev",
            generator_factory=lambda _config: calls.append("generator"),
            judge_factory=lambda _config: calls.append("judge"),
        )
    assert calls == []


def test_fixture_ledger_classifies_without_metrics(tmp_path: Path):
    specs = [
        _row("compliant"),
        _row("refusal"),
        _row("review"),
        _row("cut"),
        _row("badjudge"),
        _row("explode"),
    ]
    paths = _bundle(tmp_path, specs)

    def generate(row):
        if row["sample_id"] == "cut":
            return _ok(row, new_token_count=256, eos=False, truncated=True, response="CUT")
        return _ok(row)

    class Judge:
        def judge_one(self, prompt, response, **kwargs):
            assert PROMPT_CANARY in prompt
            prompt_id = kwargs["prompt_id"]
            if prompt_id == "explode":
                raise RuntimeError(PRIVATE)
            if prompt_id == "badjudge":
                return SimpleNamespace(
                    prompt_id=prompt_id, value=None, label="judge_failure",
                    metadata={"reason": "invalid_wildguard_fields", "raw": "malformed"},
                )
            if prompt_id == "compliant":
                return _verdict(prompt_id, harmful="yes", refusal="yes", label="unsafe", value=1.0)
            if prompt_id == "refusal":
                return _verdict(prompt_id, harmful="no", refusal="yes", label="safe", value=0.0)
            return _verdict(prompt_id, harmful="no", refusal="no", label="safe", value=0.0)

    holder = _run(tmp_path, paths, generate, Judge())
    manifest = holder["manifest"]
    assert holder["phase_at_judge"] == "target_unloaded"
    assert holder["unloaded_at_judge"] is True
    assert manifest["artifact_kind"] == "fixture"
    assert manifest["split"] == "example"
    assert manifest["membership_split"] == "dev"
    assert manifest["metrics_eligible"] is False
    assert manifest["primary_asr"] is None
    assert manifest["status"] == "complete"
    assert manifest["phases"].index("target_unloaded") < manifest["phases"].index("judging")
    assert not (tmp_path / "ledger" / "metrics.json").exists()
    samples = _read_jsonl(tmp_path / "ledger" / "samples.jsonl")
    by_id = {row["sample_id"]: row for row in samples}
    assert [row["sample_id"] for row in samples] == [spec["sample_id"] for spec in specs]
    assert by_id["compliant"]["category"] == "harmful_compliant"
    assert by_id["compliant"]["judge"]["harmful_response"] == "yes"
    assert by_id["compliant"]["judge"]["response_refusal"] == "yes"
    assert by_id["refusal"]["category"] == "refusal"
    assert by_id["review"]["category"] == "needs_category_review"
    assert by_id["cut"]["truncated"] is True
    assert by_id["cut"]["category"] == "needs_category_review"
    assert by_id["cut"]["judge"]["harmful_response"] == "no"
    assert by_id["badjudge"]["category"] == "judge_failure"
    assert by_id["explode"]["category"] == "judge_failure"
    assert by_id["explode"]["judge"]["error_type"] == "RuntimeError"
    for row in samples:
        assert row["artifact_kind"] == "fixture"
        assert row["split"] == "example"
        assert row["membership_split"] == "dev"
        assert "prompt" not in row
        assert row["category"] != "irrelevant"
    blob = "\n".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "ledger").iterdir() if path.is_file()
    )
    assert PROMPT_CANARY not in blob
    assert PRIVATE not in blob
    with pytest.raises(DevRunnerError, match="needs_category_review"):
        refuse_metrics(manifest, samples)


def test_known_categories_still_withhold_asr(tmp_path: Path):
    paths = _paths(tmp_path)
    holder = _run(
        tmp_path,
        paths,
        _ok,
        SimpleNamespace(judge_one=lambda prompt, response, **kwargs: _verdict(
            kwargs["prompt_id"], harmful="yes", refusal="no", label="unsafe", value=1.0,
        )),
        out_name="known",
    )
    samples = _read_jsonl(tmp_path / "known" / "samples.jsonl")
    assert samples[0]["category"] == "harmful_compliant"
    assert holder["manifest"]["primary_asr"] is None
    assert holder["manifest"]["metrics_eligible"] is False
    assert not (tmp_path / "known" / "metrics.json").exists()
    with pytest.raises(DevRunnerError, match="fixture ledger cannot be scored"):
        refuse_metrics(holder["manifest"], samples)
    synthetic_run = dict(holder["manifest"])
    synthetic_run["artifact_kind"] = "run"
    synthetic_run["split"] = "dev"
    with pytest.raises(DevRunnerError, match="withheld"):
        refuse_metrics(synthetic_run, samples)


def test_target_unloads_before_judge_on_failure_and_cancel(tmp_path: Path):
    paths = _bundle(tmp_path, [_row("s1"), _row("s2")])
    events = []

    def generator_factory(_config):
        def generate(row):
            if row["sample_id"] == "s2":
                raise KeyboardInterrupt()
            return _ok(row)
        events.append("generator")
        return _Generator(generate)

    def judge_factory(_config):
        events.append("judge")
        raise AssertionError("judge loaded after cancel")

    with pytest.raises(KeyboardInterrupt):
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"], prompts_path=paths["prompts"],
            family_manifest_path=paths["manifest"], out_dir=tmp_path / "cancel", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
        )
    assert events == ["generator"]
    manifest = json.loads((tmp_path / "cancel" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["target_unloaded"] is True
    generations = _read_jsonl(tmp_path / "cancel" / "generations.jsonl")
    assert [row["sample_id"] for row in generations] == ["s1"]
    assert not (tmp_path / "cancel" / "metrics.json").exists()

    fail_dir = tmp_path / "rowfail"
    fail_paths = _bundle(fail_dir, [_row("s1"), _row("s2")])

    def generate_fail(row):
        if row["sample_id"] == "s2":
            raise RuntimeError(PRIVATE)
        return _ok(row)

    class OneJudge:
        def judge_one(self, prompt, response, **kwargs):
            events.append(("judged", kwargs["prompt_id"]))
            return _verdict(kwargs["prompt_id"], harmful="no", refusal="yes")

    events.clear()
    holder = _run(fail_dir, fail_paths, generate_fail, OneJudge(), out_name="visible")
    assert holder["unloaded_at_judge"] is True
    samples = _read_jsonl(fail_dir / "visible" / "samples.jsonl")
    failed = samples[1]
    assert failed["sample_id"] == "s2"
    assert failed["status"] == "generation_error"
    assert failed["error_type"] == "RuntimeError"
    assert failed["category"] is None
    blob = "\n".join(path.read_text(encoding="utf-8") for path in (fail_dir / "visible").iterdir() if path.is_file())
    assert PRIVATE not in blob
    assert events == [("judged", "s1")]


def test_output_mismatch_and_rerun(tmp_path: Path):
    paths = _bundle(tmp_path, [_row("s1"), _row("s2")])
    calls = []

    def generate(row):
        if row["sample_id"] == "s2":
            return _ok({"sample_id": "other"})
        return _ok(row)

    def judge_factory(_config):
        calls.append("judge")
        raise AssertionError("judge")

    with pytest.raises(DevRunnerError, match="output mismatch") as caught:
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"], prompts_path=paths["prompts"],
            family_manifest_path=paths["manifest"], out_dir=tmp_path / "bad", split="dev",
            generator_factory=lambda _config: _Generator(generate),
            judge_factory=judge_factory,
        )
    assert caught.value.stage == "runtime"
    assert calls == []
    kept = _read_jsonl(tmp_path / "bad" / "generations.jsonl")
    assert kept[-1]["status"] == "output_mismatch"
    assert kept[-1]["sample_id"] == "s2"
    manifest = json.loads((tmp_path / "bad" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["target_unloaded"] is True

    good = tmp_path / "good"
    good_paths = _paths(good)
    _run(
        good, good_paths, _ok,
        SimpleNamespace(judge_one=lambda prompt, response, **kwargs: _verdict(
            kwargs["prompt_id"], harmful="no", refusal="yes",
        )),
        out_name="once",
    )
    before = (good / "once" / "samples.jsonl").read_bytes()
    with pytest.raises(DevRunnerError, match="already exists"):
        run_dev_ledger(
            config_path=good_paths["config"], index_path=good_paths["index"],
            prompts_path=good_paths["prompts"], family_manifest_path=good_paths["manifest"],
            out_dir=good / "once", split="dev",
            generator_factory=lambda _config: (_ for _ in ()).throw(AssertionError("rerun factory")),
            judge_factory=lambda _config: None,
        )
    assert (good / "once" / "samples.jsonl").read_bytes() == before


def test_real_label_stays_ineligible_and_split_api_rejects_half_injection():
    assert artifact_label(injected=True) == {
        "artifact_kind": "fixture", "split": "example", "metrics_eligible": False,
    }
    assert artifact_label(injected=False)["artifact_kind"] == "run"
    assert artifact_label(injected=False)["metrics_eligible"] is False
    with pytest.raises(DevRunnerError, match="both factories"):
        run_dev_ledger(
            config_path=Path("missing.json"), index_path=Path("missing.json"),
            prompts_path=Path("missing.json"), family_manifest_path=Path("missing.json"),
            out_dir=Path("missing-out"), split="dev", generator_factory=lambda _config: None,
        )


def test_load_failure_does_not_open_judge(tmp_path: Path):
    paths = _paths(tmp_path)
    calls = []

    def generator_factory(_config):
        calls.append("generator")
        raise RuntimeError(PRIVATE)

    def judge_factory(_config):
        calls.append("judge")
        return None

    with pytest.raises(DevRunnerError, match="RuntimeError") as caught:
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"], prompts_path=paths["prompts"],
            family_manifest_path=paths["manifest"], out_dir=tmp_path / "load", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
        )
    assert "PRIVATE" not in str(caught.value)
    assert calls == ["generator"]
    manifest = json.loads((tmp_path / "load" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["target_unloaded"] is False
    blob = (tmp_path / "load" / "manifest.json").read_text(encoding="utf-8")
    assert PRIVATE not in blob


def _authority_file(
    path: Path,
    spec: dict[str, str],
    *,
    split: str,
    hs_label: str | None = None,
    family_id: str | None = None,
) -> None:
    label = hs_label or spec["hs_label"]
    family = family_id or spec["family_id"]
    payload = {
        "families": {
            family: {
                "hs_label": label,
                "members": [{
                    "hs_label": label,
                    "item_id": spec["item_id"],
                    "lang": spec["lang"],
                    "source_dataset": spec["source_dataset"],
                    "source_id": spec["source_id"],
                }],
                "source_dataset": spec["source_dataset"],
                "split": split,
            }
        },
        "frozen": True,
        "version": "family_splits_v2",
    }
    path.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _blocked_factories():
    calls = []

    def generator_factory(_config):
        calls.append("generator")
        raise AssertionError("generator factory")

    def judge_factory(_config):
        calls.append("judge")
        raise AssertionError("judge factory")

    return calls, generator_factory, judge_factory


def test_language_and_label_enums_block_before_prompts(tmp_path: Path):
    lang_dir = tmp_path / "lang"
    lang_paths = _bundle(lang_dir, [_row("s1", lang="EN")], write_prompts=False, write_manifest=False)
    calls, generator_factory, judge_factory = _blocked_factories()
    with pytest.raises(DevRunnerError, match="language"):
        run_dev_ledger(
            config_path=lang_paths["config"], index_path=lang_paths["index"],
            prompts_path=lang_paths["prompts"], family_manifest_path=lang_paths["manifest"],
            out_dir=lang_dir / "out", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
        )
    assert calls == []
    assert not lang_paths["prompts"].exists()

    label_dir = tmp_path / "label"
    label_paths = _bundle(label_dir, [_row("s1", hs_label="unsafe")], write_prompts=False, write_manifest=False)
    with pytest.raises(DevRunnerError, match="hs_label"):
        run_dev_ledger(
            config_path=label_paths["config"], index_path=label_paths["index"],
            prompts_path=label_paths["prompts"], family_manifest_path=label_paths["manifest"],
            out_dir=label_dir / "out", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
        )
    assert not label_paths["prompts"].exists()


def test_pinned_canonical_manifest_hash():
    authority = load_canonical_authority(CANONICAL_FAMILY_MANIFEST, require_pinned_hash=True)
    assert authority["sha256"] == CANONICAL_FAMILY_MANIFEST_SHA256
    assert isinstance(authority["by_source"], dict) and authority["by_source"]


def test_canonical_helper_rejects_laundering_before_prompt_reads(tmp_path: Path):
    spec = _row("s1")
    split_dir = tmp_path / "split"
    split_paths = _bundle(split_dir, [spec], write_prompts=False)
    authority = split_dir / "canonical.json"
    _authority_file(authority, spec, split="test")
    calls, generator_factory, judge_factory = _blocked_factories()
    with pytest.raises(DevRunnerError, match="canonical split") as caught:
        run_dev_ledger(
            config_path=split_paths["config"], index_path=split_paths["index"],
            prompts_path=split_paths["prompts"], family_manifest_path=split_paths["manifest"],
            out_dir=split_dir / "out", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
            canonical_manifest_path=authority,
        )
    assert caught.value.stage == "protocol"
    assert calls == []
    assert not split_paths["prompts"].exists()
    assert not (split_dir / "out").exists()

    label_dir = tmp_path / "label"
    label_spec = _row("s1")
    label_paths = _bundle(label_dir, [label_spec], write_prompts=False)
    label_authority = label_dir / "canonical.json"
    _authority_file(label_authority, label_spec, split="dev", hs_label="benign")
    with pytest.raises(DevRunnerError, match="canonical label"):
        run_dev_ledger(
            config_path=label_paths["config"], index_path=label_paths["index"],
            prompts_path=label_paths["prompts"], family_manifest_path=label_paths["manifest"],
            out_dir=label_dir / "out", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
            canonical_manifest_path=label_authority,
        )
    assert not label_paths["prompts"].exists()

    with pytest.raises(DevRunnerError, match="cannot authorize"):
        run_dev_ledger(
            config_path=split_paths["config"], index_path=split_paths["index"],
            prompts_path=split_paths["prompts"], family_manifest_path=split_paths["manifest"],
            out_dir=split_dir / "real-out", split="dev",
            canonical_manifest_path=authority,
        )
    assert not split_paths["prompts"].exists()


def test_real_path_rejects_self_authored_dev_manifest_before_prompts(tmp_path: Path):
    unsupported = _bundle(tmp_path / "model", [_row("s1")], write_prompts=False)
    with pytest.raises(DevRunnerError, match="unsupported model contract") as caught:
        run_dev_ledger(
            config_path=unsupported["config"], index_path=unsupported["index"],
            prompts_path=unsupported["prompts"], family_manifest_path=unsupported["manifest"],
            out_dir=tmp_path / "model-out", split="dev",
        )
    assert caught.value.stage == "protocol"
    assert not unsupported["prompts"].exists()
    assert not (tmp_path / "model-out").exists()

    pinned = _bundle(
        tmp_path / "pinned", [_row("s1")], write_prompts=False, model_id=SUPPORTED_MODEL_ID,
    )
    with pytest.raises(DevRunnerError, match="canonical membership") as caught:
        run_dev_ledger(
            config_path=pinned["config"], index_path=pinned["index"],
            prompts_path=pinned["prompts"], family_manifest_path=pinned["manifest"],
            out_dir=tmp_path / "pinned-out", split="dev",
        )
    assert caught.value.stage == "protocol"
    assert not pinned["prompts"].exists()
    assert not (tmp_path / "pinned-out").exists()


def test_fixture_helper_records_synthetic_hash_and_stays_fixture(tmp_path: Path):
    spec = _row("s1")
    paths = _bundle(tmp_path, [spec])
    authority = tmp_path / "canonical.json"
    _authority_file(authority, spec, split="dev")
    holder = _run(tmp_path, paths, _ok, SimpleNamespace(
        judge_one=lambda prompt, response, **kwargs: _verdict(
            kwargs["prompt_id"], harmful="no", refusal="yes",
        ),
    ))
    # Re-run is refused, so use a direct call for the helper path.
    helper_out = tmp_path / "helper"
    manifest = run_dev_ledger(
        config_path=paths["config"], index_path=paths["index"],
        prompts_path=paths["prompts"], family_manifest_path=paths["manifest"],
        out_dir=helper_out, split="dev",
        generator_factory=lambda _config: _Generator(_ok),
        judge_factory=lambda _config: SimpleNamespace(
            judge_one=lambda prompt, response, **kwargs: _verdict(
                kwargs["prompt_id"], harmful="no", refusal="yes",
            ),
        ),
        canonical_manifest_path=authority,
    )
    assert holder["manifest"]["provenance"]["canonical_family_splits_sha256"] is None
    assert manifest["artifact_kind"] == "fixture"
    assert manifest["split"] == "example"
    assert manifest["metrics_eligible"] is False
    assert manifest["primary_asr"] is None
    assert manifest["provenance"]["canonical_family_splits_sha256"] == _sha(authority.read_bytes())
    assert not (helper_out / "metrics.json").exists()


def test_truncated_malformed_judge_stays_excluded(tmp_path: Path):
    paths = _paths(tmp_path)

    class Judge:
        def __init__(self):
            self.unloaded = False

        def judge_one(self, prompt, response, **kwargs):
            return SimpleNamespace(
                prompt_id=kwargs["prompt_id"], value=None, label="judge_failure",
                metadata={"reason": "invalid_wildguard_fields", "raw": "malformed"},
            )

        def unload(self):
            self.unloaded = True

    holder = _run(
        tmp_path, paths,
        lambda row: _ok(row, new_token_count=256, eos=False, truncated=True, response="CUT"),
        Judge(),
    )
    sample = _read_jsonl(tmp_path / "ledger" / "samples.jsonl")[0]
    assert sample["truncated"] is True
    assert sample["category"] == "judge_failure"
    assert sample["judge"]["raw"] == "malformed"
    assert sample["judge"]["harmful_response"] is None
    manifest = holder["manifest"]
    assert manifest["n_generated"] == 1
    assert manifest["n_judged"] == 0
    assert manifest["n_judge_failures"] == 1
    assert manifest["judge_unloaded"] is True
    assert manifest["primary_asr"] is None
    assert manifest["metrics_eligible"] is False
    assert not (tmp_path / "ledger" / "metrics.json").exists()


def test_judge_unload_success_failure_mismatch_and_cancel(tmp_path: Path):
    ok_dir = tmp_path / "ok"
    ok_paths = _paths(ok_dir)
    events = []

    class OkJudge:
        def judge_one(self, prompt, response, **kwargs):
            return _verdict(kwargs["prompt_id"], harmful="no", refusal="yes")

        def unload(self):
            events.append("unload")

    holder = _run(ok_dir, ok_paths, _ok, OkJudge(), out_name="done")
    assert events == ["unload"]
    assert holder["manifest"]["status"] == "complete"
    assert holder["manifest"]["judge_unloaded"] is True
    assert holder["manifest"]["primary_asr"] is None

    fail_dir = tmp_path / "fail"
    fail_paths = _paths(fail_dir)

    class BoomJudge:
        def judge_one(self, prompt, response, **kwargs):
            return _verdict(kwargs["prompt_id"], harmful="yes", refusal="no", label="unsafe", value=1.0)

        def unload(self):
            events.append("boom")
            raise RuntimeError("UNLOAD_PRIVATE")

    events.clear()
    with pytest.raises(DevRunnerError, match="judge unload failed") as caught:
        run_dev_ledger(
            config_path=fail_paths["config"], index_path=fail_paths["index"],
            prompts_path=fail_paths["prompts"], family_manifest_path=fail_paths["manifest"],
            out_dir=fail_dir / "ledger", split="dev",
            generator_factory=lambda _config: _Generator(_ok),
            judge_factory=lambda _config: BoomJudge(),
        )
    assert "UNLOAD_PRIVATE" not in str(caught.value)
    assert events == ["boom"]
    failed = json.loads((fail_dir / "ledger" / "manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["judge_release_error"] == "RuntimeError"
    assert failed["judge_unloaded"] is False
    assert failed["primary_asr"] is None
    assert failed["metrics_eligible"] is False
    assert failed["phase"] != "complete"
    samples = _read_jsonl(fail_dir / "ledger" / "samples.jsonl")
    assert samples[0]["category"] == "harmful_compliant"
    blob = "\n".join(path.read_text(encoding="utf-8") for path in (fail_dir / "ledger").iterdir() if path.is_file())
    assert "UNLOAD_PRIVATE" not in blob
    assert not (fail_dir / "ledger" / "metrics.json").exists()

    mismatch_dir = tmp_path / "mismatch"
    mismatch_paths = _paths(mismatch_dir)
    seen = {"unloaded": False}

    class MismatchJudge:
        def judge_one(self, prompt, response, **kwargs):
            return _verdict("other-id", harmful="no", refusal="yes")

        def unload(self):
            seen["unloaded"] = True

    with pytest.raises(DevRunnerError, match="judge id"):
        run_dev_ledger(
            config_path=mismatch_paths["config"], index_path=mismatch_paths["index"],
            prompts_path=mismatch_paths["prompts"], family_manifest_path=mismatch_paths["manifest"],
            out_dir=mismatch_dir / "ledger", split="dev",
            generator_factory=lambda _config: _Generator(_ok),
            judge_factory=lambda _config: MismatchJudge(),
        )
    assert seen["unloaded"] is True
    mismatch_manifest = json.loads((mismatch_dir / "ledger" / "manifest.json").read_text(encoding="utf-8"))
    assert mismatch_manifest["status"] == "failed"
    assert mismatch_manifest["judge_unloaded"] is True
    assert _read_jsonl(mismatch_dir / "ledger" / "samples.jsonl")[0]["status"] == "output_mismatch"

    cancel_dir = tmp_path / "cancel"
    cancel_paths = _paths(cancel_dir)

    class CancelJudge:
        def judge_one(self, prompt, response, **kwargs):
            raise KeyboardInterrupt()

        def unload(self):
            events.append("cancel-unload")

    events.clear()
    with pytest.raises(KeyboardInterrupt):
        run_dev_ledger(
            config_path=cancel_paths["config"], index_path=cancel_paths["index"],
            prompts_path=cancel_paths["prompts"], family_manifest_path=cancel_paths["manifest"],
            out_dir=cancel_dir / "ledger", split="dev",
            generator_factory=lambda _config: _Generator(_ok),
            judge_factory=lambda _config: CancelJudge(),
        )
    assert events == ["cancel-unload"]
    cancelled = json.loads((cancel_dir / "ledger" / "manifest.json").read_text(encoding="utf-8"))
    assert cancelled["status"] == "failed"
    assert cancelled["error_type"] == "KeyboardInterrupt"
    assert cancelled["judge_unloaded"] is True
    assert cancelled["primary_asr"] is None
    assert not (cancel_dir / "ledger" / "metrics.json").exists()


def test_target_unload_failure_is_persisted(tmp_path: Path):
    paths = _paths(tmp_path)
    calls = []

    class LeakGenerator(_Generator):
        def unload(self):
            raise RuntimeError("UNLOAD_PRIVATE")

    with pytest.raises(DevRunnerError, match="target unload failed") as caught:
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"],
            prompts_path=paths["prompts"], family_manifest_path=paths["manifest"],
            out_dir=tmp_path / "leak", split="dev",
            generator_factory=lambda _config: LeakGenerator(_ok),
            judge_factory=lambda _config: calls.append("judge"),
        )
    assert "UNLOAD_PRIVATE" not in str(caught.value)
    assert calls == []
    manifest = json.loads((tmp_path / "leak" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["target_release_error"] == "RuntimeError"
    assert manifest["primary_asr"] is None
    assert manifest["metrics_eligible"] is False
    assert _read_jsonl(tmp_path / "leak" / "generations.jsonl")[0]["status"] == "ok"
    assert not (tmp_path / "leak" / "samples.jsonl").exists()
    assert "UNLOAD_PRIVATE" not in (tmp_path / "leak" / "manifest.json").read_text(encoding="utf-8")

    cancel_dir = tmp_path / "cancel"
    cancel_paths = _paths(cancel_dir)

    class CancelGenerator:
        def generate_one(self, row):
            raise KeyboardInterrupt()

        def unload(self):
            raise RuntimeError("UNLOAD_PRIVATE")

    with pytest.raises(KeyboardInterrupt):
        run_dev_ledger(
            config_path=cancel_paths["config"], index_path=cancel_paths["index"],
            prompts_path=cancel_paths["prompts"], family_manifest_path=cancel_paths["manifest"],
            out_dir=cancel_dir / "ledger", split="dev",
            generator_factory=lambda _config: CancelGenerator(),
            judge_factory=lambda _config: calls.append("judge"),
        )
    cancelled = json.loads((cancel_dir / "ledger" / "manifest.json").read_text(encoding="utf-8"))
    assert cancelled["status"] == "failed"
    assert cancelled["error_type"] == "KeyboardInterrupt"
    assert cancelled["target_release_error"] == "RuntimeError"
    assert calls == []
    assert "UNLOAD_PRIVATE" not in (cancel_dir / "ledger" / "manifest.json").read_text(encoding="utf-8")


def test_translated_language_is_not_aliased_onto_canonical_member(tmp_path: Path):
    spec = _row("s1", lang="hinglish", source_id="src-en", item_id="item-en")
    paths = _bundle(tmp_path, [spec], write_prompts=False)
    authority = tmp_path / "canonical.json"
    authority.write_bytes((json.dumps({
        "families": {
            "fam-1": {
                "hs_label": "harm",
                "members": [{
                    "hs_label": "harm",
                    "item_id": "item-en",
                    "lang": "en",
                    "source_dataset": "synthetic",
                    "source_id": "src-en",
                }],
                "split": "dev",
            }
        },
        "frozen": True,
        "version": "family_splits_v2",
    }, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    absent = tmp_path / "absent.jsonl"
    calls, generator_factory, judge_factory = _blocked_factories()
    with pytest.raises(DevRunnerError, match="canonical label") as caught:
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"],
            prompts_path=absent, family_manifest_path=paths["manifest"],
            out_dir=tmp_path / "out", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
            canonical_manifest_path=authority,
        )
    assert caught.value.stage == "protocol"
    assert calls == []
    assert not absent.exists()


def test_derived_rows_require_lineage_before_prompts(tmp_path: Path):
    spec = _row("s1", lang="hinglish", source_dataset="reviewed_variant")
    paths = _bundle(tmp_path, [spec], write_prompts=False)
    absent = tmp_path / "absent.jsonl"
    calls, generator_factory, judge_factory = _blocked_factories()
    with pytest.raises(DevRunnerError, match="reviewed lineage contract is required"):
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"],
            prompts_path=absent, family_manifest_path=paths["manifest"],
            out_dir=tmp_path / "out", split="dev",
            generator_factory=generator_factory, judge_factory=judge_factory,
        )
    assert calls == []
    assert not absent.exists()


def test_canonical_rows_reject_unused_lineage_pins(tmp_path: Path):
    paths = _bundle(tmp_path, [_row("s1")], write_prompts=False)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    for key in (
        "canonical_source_ids_sha256",
        "review_sheet_sha256",
        "reviewed_evidence_sha256",
        "reviewed_lineage_sha256",
    ):
        config[key] = "a" * 64
    paths["config"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    absent = tmp_path / "absent.jsonl"
    with pytest.raises(DevRunnerError, match="not used for canonical rows"):
        run_dev_ledger(
            config_path=paths["config"], index_path=paths["index"],
            prompts_path=absent, family_manifest_path=paths["manifest"],
            out_dir=tmp_path / "out", split="dev",
            generator_factory=lambda _config: None,
            judge_factory=lambda _config: None,
        )
    assert not absent.exists()
