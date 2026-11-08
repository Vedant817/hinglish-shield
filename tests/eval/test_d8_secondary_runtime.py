"""CPU runtime selection for the D8 secondary judge. No weights are loaded."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.d8_secondary_runtime import (
    PinnedCpuSecondaryProcess,
    _PROBE,
    select_secondary_runtime,
    serve,
)
from src.eval.dev_runner import DevRunnerError, default_d8_judge_factory
from src.eval.judges import JudgeVerdict
from src.eval.secondary_adapter import REQUIRED_VERSIONS, UnavailableError


def test_import_does_not_load_torch():
    assert "torch" not in sys.modules
    assert "import torch" not in _PROBE


def test_select_returns_none_when_current_interpreter_matches(monkeypatch):
    monkeypatch.setattr(
        "src.eval.d8_secondary_runtime.assert_library_versions",
        lambda required: dict(required),
    )
    monkeypatch.setenv("HS_SECONDARY_PYTHON", "not-a-python")

    def boom(*_args, **_kwargs):
        raise AssertionError("probe")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.subprocess.run", boom)
    assert select_secondary_runtime() is None


def test_select_refuses_when_the_cpu_pin_is_missing(monkeypatch):
    def mismatch(_required):
        raise UnavailableError("secondary runtime library pins differ")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.assert_library_versions", mismatch)
    monkeypatch.delenv("HS_SECONDARY_PYTHON", raising=False)
    monkeypatch.setattr(
        "src.eval.d8_secondary_runtime.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("probe")),
    )
    with pytest.raises(DevRunnerError, match="HS_SECONDARY_PYTHON"):
        select_secondary_runtime()


def test_select_accepts_a_matching_probe(monkeypatch):
    def mismatch(_required):
        raise UnavailableError("secondary runtime library pins differ")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.assert_library_versions", mismatch)
    executable = r"C:\cpu\python.exe"
    monkeypatch.setenv("HS_SECONDARY_PYTHON", executable)

    def fake_run(argv, **kwargs):
        assert argv == [executable, "-c", _PROBE]
        assert kwargs.get("shell") is not True
        assert "import torch" not in argv[-1]
        body = json.dumps(dict(REQUIRED_VERSIONS))
        return SimpleNamespace(returncode=0, stdout=body + "\n", stderr="")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.subprocess.run", fake_run)
    assert select_secondary_runtime() == executable


def test_select_refuses_a_cuda_probe_without_echoing_it(monkeypatch):
    def mismatch(_required):
        raise UnavailableError("secondary runtime library pins differ")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.assert_library_versions", mismatch)
    monkeypatch.setenv("HS_SECONDARY_PYTHON", "cpu-python")

    def fake_run(_argv, **_kwargs):
        found = dict(REQUIRED_VERSIONS)
        found["torch"] = "2.15.0+cu124"
        return SimpleNamespace(returncode=0, stdout=json.dumps(found), stderr="hf_secret")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.subprocess.run", fake_run)
    with pytest.raises(DevRunnerError, match="does not match") as caught:
        select_secondary_runtime()
    text = str(caught.value)
    assert "cu124" not in text
    assert "hf_secret" not in text
    assert "HF_TOKEN" not in text


def test_factory_refuses_before_the_primary_judge(monkeypatch):
    order = []

    def refuse():
        order.append("select")
        raise DevRunnerError("D8 secondary judge requires the pinned CPU runtime", stage="protocol")

    monkeypatch.setattr("src.eval.d8_secondary_runtime.select_secondary_runtime", refuse)
    monkeypatch.setattr(
        "src.eval.dev_runner.default_judge_factory",
        lambda _config: order.append("primary"),
    )
    with pytest.raises(DevRunnerError, match="pinned CPU runtime"):
        default_d8_judge_factory({})
    assert order == ["select"]
    assert "LLMSecondaryJudge" in default_d8_judge_factory.__code__.co_names
    assert "torch" not in default_d8_judge_factory.__code__.co_names


def test_factory_uses_the_process_only_when_a_runtime_is_selected(monkeypatch):
    monkeypatch.setattr("src.eval.d8_secondary_runtime.select_secondary_runtime", lambda: None)

    class _InProcess:
        @staticmethod
        def from_protocol(_path):
            return "in-process"

    monkeypatch.setattr("src.eval.dev_runner.LLMSecondaryJudge", _InProcess)
    monkeypatch.setattr("src.eval.dev_runner.default_judge_factory", lambda _config: "primary")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("popen")),
    )
    pair = default_d8_judge_factory({})
    assert pair.primary == "primary"
    assert pair.secondary == "in-process"

    created = []

    class _Process:
        def __init__(self, executable):
            created.append(executable)

    monkeypatch.setattr("src.eval.d8_secondary_runtime.select_secondary_runtime", lambda: r"C:\cpu\python.exe")
    monkeypatch.setattr("src.eval.d8_secondary_runtime.PinnedCpuSecondaryProcess", _Process)
    pair = default_d8_judge_factory({})
    assert created == [r"C:\cpu\python.exe"]
    assert isinstance(pair.secondary, _Process)


class _FakeProc:
    def __init__(self, replies: list[str]):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("".join(replies))
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = 1


def test_process_returns_parser_failure_and_hides_transport_errors():
    prompt = "do not store this prompt"
    seen = {}

    def popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        replies = [
            json.dumps({
                "label": "judge_failure",
                "metadata": {"raw": "bad judge text", "reason": "invalid_secondary_output"},
                "ok": True,
                "prompt_id": "s1",
                "value": None,
            }) + "\n",
            json.dumps({"ok": True, "unloaded": True}) + "\n",
        ]
        return _FakeProc(replies)

    judge = PinnedCpuSecondaryProcess(r"C:\cpu\python.exe", popen=popen)
    verdict = judge.judge_one(prompt, "response text", prompt_id="s1")
    assert verdict.label == "judge_failure"
    assert verdict.value is None
    assert verdict.prompt_id == "s1"
    judge.unload()
    assert seen["argv"] == [r"C:\cpu\python.exe", "-m", "src.eval.d8_secondary_runtime"]
    assert seen["kwargs"]["cwd"] == str(ROOT)
    assert seen["kwargs"]["stderr"] is subprocess.DEVNULL
    assert seen["kwargs"]["text"] is True
    assert seen["kwargs"]["env"]["PYTHONPATH"].startswith(str(ROOT))
    assert "shell" not in seen["kwargs"]

    def fail_popen(_argv, **_kwargs):
        return _FakeProc([
            json.dumps({"detail": prompt, "ok": False, "reason": "runtime"}) + "\n",
        ])

    failing = PinnedCpuSecondaryProcess(r"C:\cpu\python.exe", popen=fail_popen)
    with pytest.raises(DevRunnerError, match="D8 secondary process failed") as caught:
        failing.judge_one(prompt, "response text", prompt_id="s1")
    assert prompt not in str(caught.value)
    assert "runtime" not in str(caught.value)


def test_serve_does_not_echo_the_prompt_or_exception_text():
    prompt = "private prompt text"

    class _Adapter:
        def judge_one(self, got_prompt, _response, **kwargs):
            assert got_prompt == prompt
            return JudgeVerdict(
                kwargs["prompt_id"],
                None,
                "judge_failure",
                {"raw": "unparsed", "reason": "invalid_secondary_output"},
            )

        def unload(self):
            return "success"

    stdin = io.StringIO(json.dumps({
        "op": "judge",
        "prompt": prompt,
        "response": "model response",
        "sample_id": "s1",
    }) + "\n" + json.dumps({"op": "unload"}) + "\n")
    stdout = io.StringIO()
    serve(_Adapter(), stdin, stdout)
    text = stdout.getvalue()
    assert prompt not in text
    assert "model response" not in text
    payload = json.loads(text.splitlines()[0])
    assert payload["ok"] is True
    assert payload["label"] == "judge_failure"
    assert json.loads(text.splitlines()[1])["unloaded"] is True

    class _Broken:
        def judge_one(self, _prompt, _response, **_kwargs):
            raise UnavailableError("cache path contains hf_secret")

        def unload(self):
            return "success"

    broken_out = io.StringIO()
    serve(_Broken(), io.StringIO(json.dumps({
        "op": "judge",
        "prompt": prompt,
        "response": "model response",
        "sample_id": "s1",
    }) + "\n"), broken_out)
    failure = broken_out.getvalue()
    assert json.loads(failure) == {"ok": False, "reason": "runtime"}
    assert "hf_secret" not in failure
    assert prompt not in failure
