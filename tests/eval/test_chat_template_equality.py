"""Model-free checks that chat-template pins cover the complete template bytes."""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval.dev_runner import DevRunnerError, HuggingFaceGreedyGenerator
from src.eval.dev_runner import assert_chat_template as assert_dev_template
from src.eval.dev_runner import chat_template_digest
from src.eval.judges import UnavailableError
from src.eval.secondary_adapter import Qwen3GuardSecondaryAdapter
from src.eval.secondary_adapter import assert_chat_template as assert_secondary_template
from src.eval.secondary_adapter import template_sha256

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/eval/prospective_secondary.json"
FIXTURE = ROOT / "tests/fixtures/qwen3guard-template/chat_template.jinja"
UPSTREAM_TEMPLATE_SHA256 = "b4230b556815baa1982c0b1eacb181ce965483ee20b1f6e4d9f375294900e02b"


def pins() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def full_template() -> str:
    return FIXTURE.read_bytes().decode("utf-8")


def _append_space(value: str) -> str:
    return value + " "


def _append_newline(value: str) -> str:
    return value + "\n"


def _drop_final_byte(value: str) -> str:
    return value[:-1]


def _prefix_only(value: str) -> str:
    return value[: len(value) // 2]


def _crlf_first_line(value: str) -> str:
    return value.replace("\n", "\r\n", 1)


def _change_final_byte(value: str) -> str:
    return value[:-1] + ("X" if value[-1] != "X" else "Y")


MUTATIONS = (
    pytest.param(_append_space, id="trailing-space"),
    pytest.param(_append_newline, id="trailing-newline"),
    pytest.param(_drop_final_byte, id="drop-final-byte"),
    pytest.param(_prefix_only, id="prefix-only"),
    pytest.param(_crlf_first_line, id="crlf-first-line"),
    pytest.param(_change_final_byte, id="change-final-byte"),
)


def test_qwen3guard_fixture_is_the_complete_pinned_template():
    raw = FIXTURE.read_bytes()
    template = raw.decode("utf-8")
    pin = pins()["chat_template_sha256"]
    assert raw
    assert hashlib.sha256(raw).hexdigest() == UPSTREAM_TEMPLATE_SHA256
    assert template_sha256(template) == UPSTREAM_TEMPLATE_SHA256
    assert pin == UPSTREAM_TEMPLATE_SHA256
    assert assert_secondary_template(template, pin) == pin


@pytest.mark.parametrize("mutate", MUTATIONS)
def test_secondary_template_pin_rejects_any_byte_difference(mutate):
    template = full_template()
    pin = pins()["chat_template_sha256"]
    changed = mutate(template)
    assert changed != template
    assert template_sha256(changed) != pin
    with pytest.raises(UnavailableError, match="chat template"):
        assert_secondary_template(changed, pin)


def test_secondary_adapter_loads_only_the_complete_pinned_template():
    template = full_template()
    pin = pins()["chat_template_sha256"]

    class Backend:
        chat_template = template
        unloaded = False

        def unload(self):
            self.unloaded = True

    backend = Backend()
    adapter = Qwen3GuardSecondaryAdapter(pins(), backend=backend)
    adapter.load()
    assert adapter.template_sha256 == pin
    assert adapter.unload() == "success"
    assert backend.unloaded is True

    class ChangedBackend(Backend):
        chat_template = _append_space(template)

    refused = Qwen3GuardSecondaryAdapter(pins(), backend=ChangedBackend())
    with pytest.raises(UnavailableError, match="chat template"):
        refused.load()
    assert refused.template_sha256 is None


def test_dev_runner_pin_rejects_any_byte_difference():
    template = "user: {{ messages }}\nassistant:\n"
    digest = chat_template_digest(template)
    assert assert_dev_template(SimpleNamespace(chat_template=template), digest) == digest
    for changed in (template + " ", template + "\n", template[:-1], " " + template, template[:10]):
        with pytest.raises(DevRunnerError, match="chat template"):
            assert_dev_template(SimpleNamespace(chat_template=changed), digest)


def _install_dev_runtime(monkeypatch, calls: list, loaded_template: str) -> None:
    def tokenizer_from_pretrained(*_args, **_kwargs):
        calls.append("tokenizer")
        return SimpleNamespace(chat_template=loaded_template)

    def model_from_pretrained(*_args, **_kwargs):
        calls.append("model")
        model = SimpleNamespace()
        model.eval = lambda: model
        return model

    fake_torch = SimpleNamespace(
        __version__="fixture",
        float16="float16",
        cuda=SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda seed: calls.append("seed"),
    )
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_from_pretrained),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_dev_runner_loader_loads_only_the_complete_pinned_template(monkeypatch):
    template = "user: synthetic prompt\nassistant:\n"
    calls: list = []
    _install_dev_runtime(monkeypatch, calls, template)
    config = {
        "dtype": "float16",
        "model_id": "example/synthetic-target",
        "model_revision": "a" * 40,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }
    generator = HuggingFaceGreedyGenerator(config)
    assert generator._tokenizer.chat_template == template
    assert generator.hardware["device"] == "cpu"
    assert calls == ["tokenizer", "model", "seed"]
    generator.unload()


def test_dev_runner_loader_refuses_a_changed_template_before_model_load(monkeypatch):
    template = "user: synthetic prompt\nassistant:\n"
    calls: list = []
    _install_dev_runtime(monkeypatch, calls, _append_space(template))
    config = {
        "dtype": "float16",
        "model_id": "example/synthetic-target",
        "model_revision": "a" * 40,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }
    with pytest.raises(DevRunnerError, match="chat template hash mismatch"):
        HuggingFaceGreedyGenerator(config)
    assert calls == ["tokenizer"]


def test_template_pins_never_normalize_empty_or_non_string_values():
    with pytest.raises(DevRunnerError, match="unpinned chat template"):
        chat_template_digest("")
    with pytest.raises(DevRunnerError, match="unpinned chat template"):
        chat_template_digest(None)  # type: ignore[arg-type]
    with pytest.raises(UnavailableError, match="chat template missing"):
        template_sha256("")
    with pytest.raises(UnavailableError, match="chat template missing"):
        template_sha256(None)
