"""Model-free checks that the repair loader pins the complete chat template."""
from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.repair.errors import RepairError
from src.repair.real_runtime import PINNED_REVISION, load_tiny_aya

TINY_AYA_ID = "CohereLabs/tiny-aya-global"


def _append_space(value: str) -> str:
    return value + " "


def _drop_final_byte(value: str) -> str:
    return value[:-1]


def _crlf_first_line(value: str) -> str:
    return value.replace("\n", "\r\n", 1)


def _prefix_only(value: str) -> str:
    return value[: len(value) // 2]


MUTATIONS = (
    pytest.param(_append_space, id="trailing-space"),
    pytest.param(_drop_final_byte, id="drop-final-byte"),
    pytest.param(_crlf_first_line, id="crlf-first-line"),
    pytest.param(_prefix_only, id="prefix-only"),
)


def _install_fake_runtime(monkeypatch, calls: list, loaded_template: str) -> None:
    def tokenizer_from_pretrained(*_args, **_kwargs):
        calls.append("tokenizer")
        return SimpleNamespace(chat_template=loaded_template)

    def model_from_pretrained(*_args, **_kwargs):
        calls.append("hf_model")
        model = SimpleNamespace()
        model.eval = lambda: model
        return model

    def tl_from_pretrained(model_name, *, hf_model=None, device=None, dtype=None, tokenizer=None):
        calls.append("tl_model")
        model = SimpleNamespace()
        model.eval = lambda: model
        return model

    fake_torch = SimpleNamespace(float32="float32", manual_seed=lambda seed: calls.append("seed"))
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_from_pretrained),
    )
    fake_lens = SimpleNamespace(
        HookedTransformer=SimpleNamespace(from_pretrained=tl_from_pretrained),
    )
    for name, module in (
        ("circuitkit", _module("circuitkit")),
        ("circuitkit.backends", _module("circuitkit.backends")),
        ("circuitkit.backends._tl_compat", _module("circuitkit.backends._tl_compat")),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformer_lens", fake_lens)


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "circuitkit-source"
    (source / "src").mkdir(parents=True)
    return source


def _pins(template: str) -> dict:
    return {
        "model_id": TINY_AYA_ID,
        "model_revision": PINNED_REVISION,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def test_repair_loader_accepts_only_the_complete_pinned_template(tmp_path, monkeypatch):
    template = "user: synthetic prompt\nassistant:\n"
    calls: list = []
    _install_fake_runtime(monkeypatch, calls, template)
    pins = _pins(template)
    handle = load_tiny_aya(pins, _source(tmp_path))
    assert handle.tokenizer.chat_template == template
    assert handle.pins["chat_template_sha256"] == hashlib.sha256(template.encode("utf-8")).hexdigest()
    assert calls == ["tokenizer", "hf_model", "seed", "tl_model"]


@pytest.mark.parametrize("mutate", MUTATIONS)
def test_repair_loader_refuses_every_byte_difference_before_model_load(tmp_path, monkeypatch, mutate):
    template = "user: synthetic prompt\nassistant:\n"
    changed = mutate(template)
    assert changed != template
    calls: list = []
    _install_fake_runtime(monkeypatch, calls, changed)
    with pytest.raises(RepairError, match="chat template hash mismatch") as caught:
        load_tiny_aya(_pins(template), _source(tmp_path))
    assert caught.value.stage == "provenance"
    assert calls == ["tokenizer"]


def test_repair_loader_refuses_a_changed_pin_before_model_load(tmp_path, monkeypatch):
    template = "user: synthetic prompt\nassistant:\n"
    calls: list = []
    _install_fake_runtime(monkeypatch, calls, template)
    pins = _pins(template)
    pins["chat_template_sha256"] = hashlib.sha256(_append_space(template).encode("utf-8")).hexdigest()
    with pytest.raises(RepairError, match="chat template hash mismatch") as caught:
        load_tiny_aya(pins, _source(tmp_path))
    assert caught.value.stage == "provenance"
    assert calls == ["tokenizer"]
