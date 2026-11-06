"""Orchestration fixtures only. No model weights or GPU calls."""
import json
import sys
from types import SimpleNamespace

import pytest

from src.eval.smoke_wildguard import main


@pytest.mark.parametrize("first_oom", [False, True])
def test_smoke_keeps_fixture_status_and_unloads_before_retry(tmp_path, monkeypatch, first_oom):
    pytest.importorskip("auditkit")
    from auditkit.model import AutoModel
    backends = []
    class Backend:
        name = "fixture-transport"
        def __init__(self, device):
            self.device = device
            self.unloaded = False
        def generate(self, requests):
            if first_oom and self.device == "cuda":
                raise MemoryError("fixture allocation failure")
            text = "Harmful request: no\nResponse refusal: no\nHarmful response: no"
            return [SimpleNamespace(completions=[SimpleNamespace(text=text)])]
        def unload(self):
            self.unloaded = True
    def resolve(spec, **kwargs):
        assert all(b.unloaded for b in backends)
        backend = Backend(kwargs["device"])
        backends.append(backend)
        return backend
    monkeypatch.setattr(AutoModel, "resolve", resolve)
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "fixture T4",
                           reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0,
                           empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=cuda, manual_seed=lambda _: None, float16="float16", float32="float32",
        __version__="fixture", OutOfMemoryError=MemoryError,
    ))
    class Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return lambda *a, **k: {"input_ids": [1, 2, 3]}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=32 * 1024**3)))
    monkeypatch.setenv("HF_TOKEN", "fixture-nonsecret")
    monkeypatch.setattr("src.eval.smoke_wildguard.version", lambda _: "fixture")
    output = tmp_path / "fixture-smoke"
    assert main(["--revision", "a" * 40, "--out", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    assert manifest["status"] == "complete" and manifest["artifact_kind"] == "fixture"
    assert metrics["metrics"]["asr"] is None
    assert all(b.unloaded for b in backends)
    assert len(backends) == (2 if first_oom else 1)
    assert "fixture-nonsecret" not in "".join(p.read_text() for p in output.iterdir())
