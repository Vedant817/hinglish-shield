"""Pinned CPU runtime for the D8 secondary judge.

Importing this module does not load weights. ``select_secondary_runtime``
returns None when this interpreter already matches the secondary library pin.
Otherwise the parent talks to ``python -m src.eval.d8_secondary_runtime`` and
does not import torch for that judge.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO

from .judges import JudgeVerdict
from .secondary_adapter import REQUIRED_VERSIONS, UnavailableError, assert_library_versions

ROOT = Path(__file__).resolve().parents[2]
SECONDARY_PROTOCOL = ROOT / "configs" / "eval" / "prospective_secondary.json"
SECONDARY_PYTHON_ENV = "HS_SECONDARY_PYTHON"
_PROCESS_ERROR = "D8 secondary process failed"
_INTERPRETER_ERROR = "D8 secondary interpreter does not match the pinned CPU runtime"
_MISSING_RUNTIME = (
    "D8 secondary judge requires the pinned CPU runtime. "
    "HS_SECONDARY_PYTHON must name an interpreter whose torch, transformers, "
    "and huggingface-hub match the secondary pin. "
    "The CUDA parent is not that runtime."
)
_PROBE = (
    "import importlib.metadata, json\n"
    "names = ('torch', 'transformers', 'huggingface-hub')\n"
    "print(json.dumps({name: importlib.metadata.version(name) for name in names}))\n"
)
_META_KEYS = (
    "categories",
    "error_type",
    "raw",
    "reason",
    "refusal",
    "sample_id_matched",
    "severity",
)


def _runner_error(message: str, *, stage: str) -> Exception:
    from .dev_runner import DevRunnerError

    return DevRunnerError(message, stage=stage)


def _versions_match(found: Mapping[str, Any]) -> bool:
    return all(
        isinstance(found.get(name), str) and found.get(name) == expected
        for name, expected in REQUIRED_VERSIONS.items()
    )


def _probe_interpreter(executable: str) -> bool:
    try:
        completed = subprocess.run(
            [executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if completed.returncode != 0:
        return False
    try:
        found = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False
    return isinstance(found, dict) and _versions_match(found)


def select_secondary_runtime() -> str | None:
    """Return None for an in-process CPU pin, or a matching interpreter path.

    A mismatch with no ``HS_SECONDARY_PYTHON`` refuses before any row is judged.
    The probe uses importlib.metadata only and does not import torch.
    """
    try:
        assert_library_versions(REQUIRED_VERSIONS)
    except UnavailableError:
        pass
    else:
        return None
    executable = os.environ.get(SECONDARY_PYTHON_ENV, "").strip()
    if not executable:
        raise _runner_error(_MISSING_RUNTIME, stage="protocol")
    if not _probe_interpreter(executable):
        raise _runner_error(_INTERPRETER_ERROR, stage="protocol")
    return executable


def _public_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    for key in _META_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if key == "categories" and isinstance(value, list) and all(isinstance(item, str) for item in value):
            cleaned[key] = list(value)
        elif key in {"refusal", "sample_id_matched"} and isinstance(value, bool):
            cleaned[key] = value
        elif value is None or isinstance(value, str):
            cleaned[key] = value
    return cleaned


def _write(stdout: TextIO, payload: Mapping[str, Any]) -> None:
    stdout.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    stdout.flush()


def serve(adapter: Any, stdin: TextIO, stdout: TextIO) -> None:
    """Score one JSON request per line. The reply does not echo the prompt."""
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write(stdout, {"ok": False, "reason": "invalid_message"})
            return
        if not isinstance(message, dict):
            _write(stdout, {"ok": False, "reason": "invalid_message"})
            return
        op = message.get("op")
        if op == "unload":
            try:
                adapter.unload()
            except Exception:
                _write(stdout, {"ok": False, "reason": "unload"})
                return
            _write(stdout, {"ok": True, "unloaded": True})
            return
        if op != "judge":
            _write(stdout, {"ok": False, "reason": "invalid_message"})
            return
        sample_id = message.get("sample_id")
        prompt = message.get("prompt")
        response = message.get("response")
        if not isinstance(sample_id, str) or not isinstance(prompt, str) or not isinstance(response, str):
            _write(stdout, {"ok": False, "reason": "invalid_message"})
            return
        try:
            verdict = adapter.judge_one(prompt, response, prompt_id=sample_id)
        except UnavailableError:
            _write(stdout, {"ok": False, "reason": "runtime"})
            return
        except Exception:
            _write(stdout, {"ok": False, "reason": "runtime"})
            return
        _write(stdout, {
            "label": verdict.label if isinstance(verdict.label, str) else None,
            "metadata": _public_metadata(verdict.metadata),
            "ok": True,
            "prompt_id": verdict.prompt_id,
            "value": verdict.value,
        })


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(ROOT)
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if previous == "" else root + os.pathsep + previous
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def configure_stdio_utf8(stdin: TextIO, stdout: TextIO) -> None:
    """Use UTF-8 on the worker pipes. Locale cp1252 cannot encode Devanagari."""
    for stream in (stdin, stdout):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        reconfigure(encoding="utf-8", errors="strict")


class PinnedCpuSecondaryProcess:
    """Parent side of the pinned CPU secondary judge. Weights stay in the child."""

    def __init__(self, executable: str, *, popen: Callable[..., Any] | None = None) -> None:
        if not isinstance(executable, str) or executable.strip() == "":
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        opener = popen or subprocess.Popen
        try:
            self._proc = opener(
                [executable, "-m", "src.eval.d8_secondary_runtime"],
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="strict",
                env=_child_env(),
            )
        except OSError:
            raise _runner_error(_PROCESS_ERROR, stage="runtime") from None
        self._closed = False

    def _exchange(self, message: Mapping[str, Any]) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None or proc.poll() is not None:
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        try:
            proc.stdin.write(json.dumps(dict(message), ensure_ascii=False) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        except Exception:
            raise _runner_error(_PROCESS_ERROR, stage="runtime") from None
        if not isinstance(line, str) or line.strip() == "":
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            raise _runner_error(_PROCESS_ERROR, stage="runtime") from None
        if not isinstance(payload, dict):
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        return payload

    def judge_one(self, prompt: str, response: str, **kwargs: Any) -> JudgeVerdict:
        prompt_id = str(kwargs.get("prompt_id", ""))
        payload = self._exchange({
            "op": "judge",
            "prompt": prompt,
            "response": response,
            "sample_id": prompt_id,
        })
        if payload.get("ok") is not True or payload.get("prompt_id") != prompt_id:
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        value = payload.get("value")
        if isinstance(value, bool) or not (value is None or isinstance(value, (int, float))):
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        label = payload.get("label")
        if not (label is None or isinstance(label, str)):
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise _runner_error(_PROCESS_ERROR, stage="runtime")
        numeric = None if value is None else float(value)
        return JudgeVerdict(prompt_id, numeric, label, _public_metadata(metadata))

    def unload(self) -> None:
        if self._closed:
            return
        error: Exception | None = None
        try:
            payload = self._exchange({"op": "unload"})
            if payload.get("ok") is not True or payload.get("unloaded") is not True:
                error = _runner_error(_PROCESS_ERROR, stage="runtime")
        except Exception as exc:
            error = exc if type(exc).__name__ == "DevRunnerError" else _runner_error(_PROCESS_ERROR, stage="runtime")
        finally:
            self._close()
        if error is not None:
            raise error

    def _close(self) -> None:
        proc = self._proc
        self._proc = None
        self._closed = True
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            if stream is None:
                continue
            try:
                stream.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            killer = getattr(proc, "kill", None)
            if callable(killer):
                try:
                    killer()
                except Exception:
                    pass


def main() -> int:
    """Worker entry. Construction checks pins and does not load weights."""
    from .secondary_adapter import Qwen3GuardSecondaryAdapter

    configure_stdio_utf8(sys.stdin, sys.stdout)
    try:
        pins = json.loads(SECONDARY_PROTOCOL.read_text(encoding="utf-8"))
        adapter = Qwen3GuardSecondaryAdapter(pins)
    except (OSError, json.JSONDecodeError, UnavailableError):
        _write(sys.stdout, {"ok": False, "reason": "pins"})
        return 1
    serve(adapter, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
