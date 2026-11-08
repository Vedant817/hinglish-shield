"""Synthetic v2 identity-helper contracts. No network, no model, no secrets."""
import hashlib
import json
import sys
import types

import pytest

from scripts.kaggle_tokenizer_metadata import (
    MODEL_ID,
    PINNED_REVISION,
    collect_identity,
    select_chat_template,
    tokenizer_file_names,
)


def test_select_chat_template_string_mapping_and_list_sources():
    name, source, template = select_chat_template({"chat_template": "tpl-a"}, None)
    assert (name, source, template) == (None, "tokenizer_config.json", "tpl-a")
    name, source, template = select_chat_template(
        {"chat_template": {"other": "tpl-b", "default": "tpl-default"}}, None)
    assert (name, source, template) == ("default", "tokenizer_config.json", "tpl-default")
    name, source, template = select_chat_template(
        {}, {"chat_template": [{"name": "x", "template": "tpl-c"}]})
    assert (name, source, template) == ("x", "tokenizer.json", "tpl-c")
    assert select_chat_template({}, None) == (None, None, None)
    assert select_chat_template({"chat_template": "  "}, {"chat_template": ""}) == (None, None, None)


def test_tokenizer_file_names_keeps_root_level_only():
    names = tokenizer_file_names([
        "tokenizer_config.json", "sub/tokenizer.json", "../tokenizer.json",
        "merges.txt", "model.safetensors", "config.json", "vocab.json",
    ])
    assert names == ["merges.txt", "tokenizer_config.json", "vocab.json"]


def test_missing_secret_writes_unavailable_without_hub_calls(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = tmp_path / "identity"
    result = collect_identity(out, source_commit="a" * 40, source_sha256="b" * 64,
                              notebook_run_id="run-1")
    assert result["status"] == "unavailable"
    assert result["failure_phase"] == "secret_missing"
    assert result["hf_token_present"] is False
    assert result["secret_status"] == "missing"
    assert result["weights_loaded"] is False
    assert result["metrics_eligible"] is False
    assert result["primary_asr"] is None
    assert result["model_id"] == MODEL_ID
    assert result["requested_revision"] == PINNED_REVISION
    assert result["schema_version"] == "target_identity_v2"
    assert result["source_commit_role"] == "repository_data_pin"
    on_disk = json.loads((out / "identity.json").read_text(encoding="utf-8"))
    assert on_disk == result
    assert not (out / "tokenizer_config.json").exists()


def _stub_hub(monkeypatch, tmp_path, *, files, sha=None, fail_download=False):
    class Sibling:
        def __init__(self, rfilename):
            self.rfilename = rfilename

    class Info:
        def __init__(self):
            self.sha = sha or PINNED_REVISION
            self.siblings = [Sibling(name) for name in files]

    class HfApi:
        def __init__(self, **kwargs):
            pass

        def model_info(self, repo_id, revision=None, token=None):
            assert repo_id == MODEL_ID
            assert revision == PINNED_REVISION
            return Info()

    def hf_hub_download(*, repo_id, filename, revision, token):
        if fail_download:
            raise RuntimeError("download exploded with hf_ fixture-secret-value-0123456789")
        path = tmp_path / ("cache-" + filename)
        path.write_bytes(files[filename])
        return str(path)

    module = types.ModuleType("huggingface_hub")
    module.HfApi = HfApi
    module.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_success_hub_path_hashes_files_and_pins_template(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fixture-token-value-0123456789")
    config = json.dumps({"chat_template": "tpl exact\n"}).encode()
    tokenizer = json.dumps({"chat_template": "unused"}).encode()
    maps = b"{}"
    _stub_hub(monkeypatch, tmp_path, files={
        "tokenizer_config.json": config, "tokenizer.json": tokenizer,
        "special_tokens_map.json": maps, "sub/tokenizer.json": b"{}",
    })
    out = tmp_path / "ok"
    result = collect_identity(out, source_commit="a" * 40, source_sha256="b" * 64)
    assert result["status"] == "metadata_retrieved"
    assert result["model_revision"] == PINNED_REVISION
    assert result["revision_matches_pin"] is True
    assert result["hf_token_present"] is True
    assert result["chat_template"] == "tpl exact\n"
    assert result["chat_template_sha256"] == hashlib.sha256(b"tpl exact\n").hexdigest()
    assert result["chat_template_source"] == "tokenizer_config.json"
    assert result["tokenizer_files"]["tokenizer_config.json"]["sha256"] == hashlib.sha256(config).hexdigest()
    assert (out / "tokenizer_config.json").read_bytes() == config
    stored = (out / "identity.json").read_text(encoding="utf-8")
    assert "hf_fixture-token-value" not in stored


def test_hub_failure_redacts_secret_and_drops_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fixture-token-value-0123456789")
    _stub_hub(monkeypatch, tmp_path,
              files={"tokenizer_config.json": b"{}", "tokenizer.json": b"{}",
                     "special_tokens_map.json": b"{}"},
              fail_download=True)
    out = tmp_path / "fail"
    (out).mkdir()
    (out / "tokenizer_config.json").write_bytes(b"stale")
    result = collect_identity(out, source_commit="a" * 40, source_sha256="b" * 64)
    assert result["status"] == "unavailable"
    assert result["failure_phase"] == "download_failed"
    assert result["error_type"] == "RuntimeError"
    assert "hf_fixture-token-value" not in json.dumps(result)
    assert not (out / "tokenizer_config.json").exists()
    assert result["tokenizer_files"] == {}


def test_revision_mismatch_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fixture-token-value-0123456789")
    _stub_hub(monkeypatch, tmp_path,
              files={"tokenizer_config.json": b"{}", "tokenizer.json": b"{}",
                     "special_tokens_map.json": b"{}"},
              sha="c" * 40)
    result = collect_identity(tmp_path / "rev", source_commit="a" * 40, source_sha256="b" * 64)
    assert result["status"] == "unavailable"
    assert result["failure_phase"] == "revision_unavailable"
    assert result["revision_matches_pin"] is False
