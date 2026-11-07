"""Pinned Qwen3Guard secondary adapter. Import does not load weights or open a network.

Unrecognized model, revision, template, or decoding pins refuse before any backend
load. Controversial parser output stays unresolved and is never rewritten as safe.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .judges import JudgeVerdict, UnavailableError
from .secondary_contract import parse_qwen3guard_response

MODEL_ID = "Qwen/Qwen3Guard-Gen-0.6B"
MODEL_REVISION = "fada3b2f655b89601929198343c94cd2f64d93cc"
CHAT_TEMPLATE_SHA256 = "b4230b556815baa1982c0b1eacb181ce965483ee20b1f6e4d9f375294900e02b"
TOKENIZER_CONFIG_SHA256 = "934e187433b06c5d28519e0347dd293084cf28621f00a86ae7e24e42f8b53e81"
MODEL_CARD_SHA256 = "3a52042f467261243d949d86a779f08ac8ff9da30ce47fd636db6e1c9c1b7635"
PROTOCOL_ID = "hs-qwen3guard-secondary-v1"
PARSER_ID = "src.eval.secondary_contract.parse_qwen3guard_response"
DECODING = {
    "do_sample": False,
    "max_new_tokens": 128,
    "system_prompt": None,
    "seed": 42,
}
REQUIRED_VERSIONS = {
    "torch": "2.15.0+cpu",
    "transformers": "5.22.2",
    "huggingface-hub": "2.4.0",
}
_PIN_ERROR = "unrecognized or incomplete secondary pins"
_SEVERITY = {
    "safe": "Safe",
    "unsafe": "Unsafe",
    "secondary_controversial": "Controversial",
}


def require_selected_pins(pins: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached copy of the selected protocol, or refuse before load."""
    if not isinstance(pins, Mapping):
        raise UnavailableError(_PIN_ERROR)
    decoding = pins.get("decoding")
    versions = pins.get("required_versions")
    if not isinstance(decoding, Mapping) or not isinstance(versions, Mapping):
        raise UnavailableError(_PIN_ERROR)
    if set(decoding) != set(DECODING) or set(versions) != set(REQUIRED_VERSIONS):
        raise UnavailableError(_PIN_ERROR)
    for key, expected in DECODING.items():
        if decoding.get(key) != expected or type(decoding.get(key)) is not type(expected):
            raise UnavailableError(_PIN_ERROR)
    for key, expected in REQUIRED_VERSIONS.items():
        if versions.get(key) != expected or type(versions.get(key)) is not str:
            raise UnavailableError(_PIN_ERROR)
    identity = {
        "protocol_id": PROTOCOL_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "model_card_sha256": MODEL_CARD_SHA256,
        "role": "secondary_only",
        "dtype": "float32",
        "device": "cpu",
        "parser": PARSER_ID,
        "may_write_primary_metrics": False,
    }
    for key, expected in identity.items():
        if pins.get(key) != expected or type(pins.get(key)) is not type(expected):
            raise UnavailableError(_PIN_ERROR)
    return {
        **identity,
        "decoding": dict(DECODING),
        "required_versions": dict(REQUIRED_VERSIONS),
    }


def assert_library_versions(required: Mapping[str, str]) -> dict[str, str]:
    """Compare installed distribution versions. This does not import torch."""
    found: dict[str, str] = {}
    for name, expected in required.items():
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise UnavailableError("secondary runtime library pins differ") from exc
        if found[name] != expected:
            raise UnavailableError("secondary runtime library pins differ")
    return found


def template_sha256(template: Any) -> str:
    if not isinstance(template, str) or template == "":
        raise UnavailableError("secondary chat template missing")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def assert_chat_template(template: Any, expected: str) -> str:
    digest = template_sha256(template)
    if digest != expected:
        raise UnavailableError("secondary chat template differs from selected pin")
    return digest


@dataclass(frozen=True)
class SecondaryRequest:
    sample_id: str
    rendered: str
    do_sample: bool
    max_new_tokens: int
    seed: int


@dataclass(frozen=True)
class SecondaryCompletion:
    sample_id: str
    text: str


def _completion(item: Any) -> tuple[str, str] | None:
    if isinstance(item, Mapping):
        sample_id = item.get("sample_id")
        text = item.get("text")
    else:
        sample_id = getattr(item, "sample_id", None)
        text = getattr(item, "text", None)
    if not isinstance(sample_id, str) or not isinstance(text, str):
        return None
    return sample_id, text


def _failure(sample_id: str, reason: str, **extra: Any) -> JudgeVerdict:
    metadata = {"reason": reason, "sample_id_matched": False, **extra}
    return JudgeVerdict(sample_id, None, "judge_failure", metadata)


def _verdict(sample_id: str, raw: str, pins: Mapping[str, Any], template_digest: str | None) -> JudgeVerdict:
    parsed = parse_qwen3guard_response(raw)
    metadata: dict[str, Any] = {
        "severity": _SEVERITY.get(parsed.label),
        "categories": list(parsed.categories),
        "refusal": parsed.refusal,
        "raw": raw,
        "model_id": pins["model_id"],
        "model_revision": pins["model_revision"],
        "chat_template_sha256": template_digest,
        "parser": pins["parser"],
        "sample_id_matched": True,
        "reason": None if parsed.label != "judge_failure" else "invalid_secondary_output",
    }
    return JudgeVerdict(sample_id, parsed.value, parsed.label, metadata)


class PinnedLocalBackend:
    """Local CPU loader. Imported libraries stay inside ``open``, not module import."""

    def __init__(self, pins: Mapping[str, Any]) -> None:
        self.pins = dict(pins)
        self.model: Any = None
        self.tokenizer: Any = None
        self.torch: Any = None
        self.chat_template: str | None = None
        self.library_versions: dict[str, str] | None = None
        self.template_sha256: str | None = None
        self.tokenizer_config_sha256: str | None = None
        self._released = False
        try:
            self._open()
        except BaseException:
            try:
                self.unload()
            except Exception:
                pass
            raise

    def _open(self) -> None:
        pins = self.pins
        self.library_versions = assert_library_versions(pins["required_versions"])
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=pins["model_id"],
            filename="tokenizer_config.json",
            revision=pins["model_revision"],
            local_files_only=True,
        )
        with open(config_path, "rb") as handle:
            config_bytes = handle.read()
        config_digest = hashlib.sha256(config_bytes).hexdigest()
        if config_digest != pins["tokenizer_config_sha256"]:
            raise UnavailableError("tokenizer config hash differs from selected pin")
        self.tokenizer_config_sha256 = config_digest
        payload = json.loads(config_bytes.decode("utf-8"))
        self.template_sha256 = assert_chat_template(
            payload.get("chat_template"), pins["chat_template_sha256"],
        )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        random.seed(pins["decoding"]["seed"])
        torch.manual_seed(pins["decoding"]["seed"])
        tokenizer = AutoTokenizer.from_pretrained(
            pins["model_id"],
            revision=pins["model_revision"],
            local_files_only=True,
        )
        self.chat_template = tokenizer.chat_template
        assert_chat_template(self.chat_template, pins["chat_template_sha256"])
        self.tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            pins["model_id"],
            revision=pins["model_revision"],
            dtype=torch.float32,
            local_files_only=True,
        ).to("cpu").eval()
        self.model = model

    def apply_chat_template(self, messages: Sequence[Mapping[str, str]], tokenize: bool = False) -> str:
        if tokenize is not False:
            raise UnavailableError("secondary render must stay detokenized")
        if self.tokenizer is None:
            raise UnavailableError("secondary backend is unloaded")
        rendered = self.tokenizer.apply_chat_template(list(messages), tokenize=False)
        if not isinstance(rendered, str) or rendered == "":
            raise UnavailableError("secondary render failed")
        return rendered

    def complete(self, requests: Sequence[SecondaryRequest]) -> list[SecondaryCompletion]:
        if self._released or self.model is None or self.tokenizer is None or self.torch is None:
            raise UnavailableError("secondary backend is unloaded")
        torch = self.torch
        results: list[SecondaryCompletion] = []
        for request in requests:
            if request.do_sample is not False or request.max_new_tokens != 128 or request.seed != 42:
                raise UnavailableError("secondary decoding pins differ")
            random.seed(request.seed)
            torch.manual_seed(request.seed)
            inputs = self.tokenizer([request.rendered], return_tensors="pt")
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=request.max_new_tokens,
                )
            prompt_len = int(inputs["input_ids"].shape[1])
            new_ids = generated[0][prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            if not isinstance(text, str):
                raise UnavailableError("secondary decode failed")
            results.append(SecondaryCompletion(request.sample_id, text))
            del inputs, generated
        return results

    def unload(self) -> None:
        model = self.model
        tokenizer = self.tokenizer
        torch = self.torch
        self.model = None
        self.tokenizer = None
        self.torch = None
        self._released = True
        del model, tokenizer
        import gc
        gc.collect()
        if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


class Qwen3GuardSecondaryAdapter:
    """Selected secondary judge. ``backend`` injects a fake transport for tests."""

    name = MODEL_ID
    role = "secondary"

    def __init__(self, pins: Mapping[str, Any], *, backend: Any = None) -> None:
        self.pins = require_selected_pins(pins)
        self._backend = backend
        self._loaded = False
        self._cancel = False
        self._held = False
        self.unload_status: str | None = None
        self.library_versions: dict[str, str] | None = None
        self.template_sha256: str | None = None
        self.tokenizer_config_sha256: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel

    def cancel(self) -> None:
        self._cancel = True

    def load(self) -> None:
        if self.unload_status is not None:
            raise UnavailableError("secondary adapter is unloaded")
        if self._cancel:
            raise UnavailableError("secondary adapter is cancelled")
        if self._loaded:
            return
        if self._backend is None:
            self._backend = PinnedLocalBackend(self.pins)
            self.library_versions = self._backend.library_versions
            self.tokenizer_config_sha256 = self._backend.tokenizer_config_sha256
        self.template_sha256 = assert_chat_template(
            _backend_template(self._backend), self.pins["chat_template_sha256"],
        )
        self._loaded = True
        self._held = True

    def judge_one(self, prompt: str, response: str, **kwargs: Any) -> JudgeVerdict:
        prompt_id = str(kwargs.get("prompt_id", ""))
        return self.judge_many([(prompt_id, prompt, response)])[0]

    def judge_many(self, pairs: Sequence[tuple[str, str, str]]) -> list[JudgeVerdict]:
        if self._cancel:
            return [_failure(str(pid), "cancelled") for pid, _, _ in pairs]
        if self.unload_status is not None:
            return [_failure(str(pid), "unloaded") for pid, _, _ in pairs]
        prepared: list[tuple[str, str, str]] = []
        blocked: dict[str, JudgeVerdict] = {}
        seen: set[str] = set()
        duplicates: set[str] = set()
        for prompt_id, prompt, response in pairs:
            sample_id = str(prompt_id)
            if sample_id in seen:
                duplicates.add(sample_id)
            seen.add(sample_id)
            if not isinstance(prompt, str) or not isinstance(response, str):
                blocked[sample_id] = _failure(sample_id, "invalid_input")
            else:
                prepared.append((sample_id, prompt, response))
        for sample_id in duplicates:
            blocked[sample_id] = _failure(sample_id, "duplicate_sample_id")
        requests: list[SecondaryRequest] = []
        order: list[str] = []
        sendable = [item for item in prepared if item[0] not in duplicates]
        if sendable:
            self.load()
            for sample_id, prompt, response in sendable:
                order.append(sample_id)
                requests.append(SecondaryRequest(
                    sample_id=sample_id,
                    rendered=_render(self._backend, prompt, response, self.pins),
                    do_sample=self.pins["decoding"]["do_sample"],
                    max_new_tokens=self.pins["decoding"]["max_new_tokens"],
                    seed=self.pins["decoding"]["seed"],
                ))
        by_id: dict[str, str] = {}
        mismatched = False
        try:
            completions = self._backend.complete(requests) if requests else []
        except Exception as exc:
            error_type = type(exc).__name__
            return [
                blocked.get(str(pid)) or _failure(str(pid), "judge_call_failed", error_type=error_type)
                for pid, _, _ in pairs
            ]
        returned: list[str] = []
        for item in completions:
            fields = _completion(item)
            if fields is None:
                mismatched = True
                break
            sample_id, text = fields
            returned.append(sample_id)
            if sample_id in by_id:
                mismatched = True
            by_id[sample_id] = text
        if returned != [request.sample_id for request in requests] or set(by_id) != set(order):
            mismatched = True
        verdicts: list[JudgeVerdict] = []
        for prompt_id, _, _ in pairs:
            sample_id = str(prompt_id)
            if sample_id in blocked:
                verdicts.append(blocked[sample_id])
                continue
            if mismatched or sample_id not in by_id:
                verdicts.append(_failure(sample_id, "sample_id_mismatch"))
                continue
            verdicts.append(_verdict(sample_id, by_id[sample_id], self.pins, self.template_sha256))
        return verdicts

    def unload(self) -> str:
        if self.unload_status is not None:
            return self.unload_status
        backend = self._backend
        try:
            if backend is not None and callable(getattr(backend, "unload", None)):
                backend.unload()
        except Exception:
            self.unload_status = "error"
            self._loaded = False
            self._backend = None
            self._held = False
            raise UnavailableError("secondary unload failed") from None
        self.unload_status = "cancelled" if self._cancel else "success"
        self._loaded = False
        self._backend = None
        self._held = False
        return self.unload_status


def _backend_template(backend: Any) -> Any:
    template = getattr(backend, "chat_template", None)
    if template is None:
        tokenizer = getattr(backend, "tokenizer", None)
        template = getattr(tokenizer, "chat_template", None)
    return template


def _render(backend: Any, prompt: str, response: str, pins: Mapping[str, Any]) -> str:
    if pins["decoding"]["system_prompt"] is not None:
        raise UnavailableError("secondary system prompt is not selected")
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    renderer = getattr(backend, "apply_chat_template", None)
    if renderer is None and getattr(backend, "tokenizer", None) is not None:
        renderer = backend.tokenizer.apply_chat_template
    if not callable(renderer):
        raise UnavailableError("secondary render failed")
    rendered = renderer(messages, tokenize=False)
    if not isinstance(rendered, str) or rendered == "":
        raise UnavailableError("secondary render failed")
    return rendered


def secondary_row_problems(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reparse stored completions. Report mismatches; do not rewrite rows."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        return ["rows"]
    problems: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"row {index} type")
            continue
        sample_id = row.get("id")
        raw = row.get("raw_completion")
        if not isinstance(sample_id, str) or sample_id == "":
            problems.append(f"row {index} id")
        elif sample_id in seen:
            problems.append(f"row {index} duplicate id")
        else:
            seen.add(sample_id)
        if not isinstance(raw, str):
            problems.append(f"row {index} raw")
            continue
        parsed = parse_qwen3guard_response(raw)
        if row.get("label") != parsed.label or row.get("value") != parsed.value:
            problems.append(f"row {index} parse")
        if row.get("refusal") != parsed.refusal:
            problems.append(f"row {index} refusal")
        categories = row.get("categories")
        if not isinstance(categories, list) or tuple(categories) != parsed.categories:
            problems.append(f"row {index} categories")
        if row.get("severity") != _SEVERITY.get(parsed.label):
            problems.append(f"row {index} severity")
        if ((row.get("label") == "secondary_controversial" or parsed.label == "secondary_controversial")
                and (row.get("value") is not None or parsed.value is not None)):
            problems.append(f"row {index} controversial value")
        if parsed.label == "safe" and parsed.value != 0.0:
            problems.append(f"row {index} safe value")
        if parsed.label == "unsafe" and parsed.value != 1.0:
            problems.append(f"row {index} unsafe value")
    return problems
