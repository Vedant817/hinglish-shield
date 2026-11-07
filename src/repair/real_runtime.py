"""In-process Tiny Aya load, proposed direction, and dev generation ledger.

Real weights steer with CircuitKIT ActivationSteering.steer, one forward per
new token. Injected fixtures keep the labeled residual hook and are not runs.
Gates run before prompt reads and before any model factory. A missing pin,
a missing original fp32 parity credit, or a dirty source tree fails closed.
Nothing here invents ASR, a chat-template hash, or a refusal mechanism.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.curate.reviewed_variants import (
    REVIEWED_VARIANT_SOURCE,
    ReviewedVariantError,
    verify_ledger_lineage,
)
from src.eval.dev_runner import (
    MAX_NEW_TOKENS,
    REQUIRED_SEED,
    assign_category,
    generation_flags,
    user_only_messages,
)
from src.eval.judges import WildGuardPrimaryJudge

from .errors import RepairError
from .fixture_model import TINY_AYA_ID
from .jsonutil import content_sha256, sha256_file, strict_json_bytes, write_json
from .membership import (
    INDEX_FIELDS,
    ITEM_HS_LABELS,
    ITEM_LANGS,
    assert_prompt_path_allowed,
    assert_repair_membership,
    load_authority,
    load_index,
    load_prompts,
    require_both_classes,
)
from .protocol import load_protocol, load_provenance, load_sweep_config
from .selection import decide_arm
from .direction import extract_direction
from .tensor_fixture import DeterministicTensorFixture
from .vectors import add_scaled, l2_norm, random_norm_matched

ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "e71d5e17fe6e2abfde47ec9a7fbaf31649601b78"
PINNED_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
ORIGINAL_CREDIT_REASON = "original_fp32_short_and_4196_long"
EVIDENCED_LOAD_DTYPE = "float32"
DECLARED_SCORE_DTYPE = "bfloat16"
DECLARED_ARM = "candidate_alpha_1"
LAYER_NAME = "blocks.0.hook_resid_post"
DEFAULT_SOURCE = Path(r"C:\Users\vedan\Documents\Resources\Code\hinglish-shield-scratch\CircuitKIT-source")
STEERING_GIT_PATH = "src/circuitkit/applications/steering/steering.py"
SHA40 = __import__("re").compile(r"[0-9a-f]{40}")
SHA256 = __import__("re").compile(r"[0-9a-f]{64}")
NODE_NAME = __import__("re").compile(r"A\d+\.\d+|MLP\s+\d+")
RUNTIME_KEYS = frozenset({
    "model_id",
    "model_revision",
    "chat_template_sha256",
    "load_dtype",
    "load_dtype_evidence",
    "declared_score_dtype",
    "score_dtype_evidenced",
    "seed",
    "method",
    "declared_arm",
    "decoding",
    "judge_model_id",
    "judge_revision",
    "circuit_scores",
    "circuitkit_source",
})
DECODING_KEYS = frozenset({"do_sample", "max_new_tokens", "system_prompt", "temperature", "top_p"})
REVIEW_PIN_FIELDS = (
    "reviewed_lineage_sha256",
    "reviewed_evidence_sha256",
    "review_sheet_sha256",
    "canonical_source_ids_sha256",
)


def _exact_int(value: Any, expected: int, message: str) -> None:
    if type(value) is not int or value != expected:
        raise RepairError(message, stage="provenance")


def load_runtime_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RepairError("missing real runtime config", stage="provenance")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != RUNTIME_KEYS:
        raise RepairError("real runtime config fields are wrong", stage="provenance")
    if payload.get("model_id") != TINY_AYA_ID:
        raise RepairError("real repair model id is not the pinned Tiny Aya id", stage="provenance")
    revision = payload.get("model_revision")
    if revision != PINNED_REVISION:
        raise RepairError("model revision is not the pinned Tiny Aya revision", stage="provenance")
    template = payload.get("chat_template_sha256")
    if not isinstance(template, str) or SHA256.fullmatch(template) is None:
        raise RepairError("unpinned chat template", stage="provenance")
    if payload.get("score_dtype_evidenced") is not False:
        raise RepairError("config cannot assert dtype evidence", stage="provenance")
    if payload.get("load_dtype") != EVIDENCED_LOAD_DTYPE:
        raise RepairError("load dtype is not the evidenced fp32 dtype", stage="provenance")
    if payload.get("load_dtype_evidence") != ORIGINAL_CREDIT_REASON:
        raise RepairError("load dtype evidence is not the original fp32 gate", stage="provenance")
    if payload.get("declared_score_dtype") != DECLARED_SCORE_DTYPE:
        raise RepairError("declared score dtype is not bfloat16", stage="provenance")
    _exact_int(payload.get("seed"), REQUIRED_SEED, "unpinned seed: seed 42 is required")
    if payload.get("method") != "base":
        raise RepairError("unpinned method: base is required", stage="provenance")
    if payload.get("declared_arm") != DECLARED_ARM:
        raise RepairError("declared arm is not candidate_alpha_1", stage="provenance")
    decoding = payload.get("decoding")
    if not isinstance(decoding, dict) or set(decoding) != DECODING_KEYS:
        raise RepairError("unpinned decoding", stage="provenance")
    if decoding.get("do_sample") is not False:
        raise RepairError("unpinned decoding: greedy do_sample false is required", stage="provenance")
    _exact_int(decoding.get("max_new_tokens"), MAX_NEW_TOKENS, "unpinned decoding: max_new_tokens 256 is required")
    if decoding.get("system_prompt") is not None:
        raise RepairError("unpinned decoding: custom system prompt is refused", stage="provenance")
    if decoding.get("temperature") is not None or decoding.get("top_p") is not None:
        raise RepairError("unpinned decoding: temperature and top_p must be null", stage="provenance")
    if payload.get("judge_model_id") != "allenai/wildguard":
        raise RepairError("unpinned judge_model_id: allenai/wildguard is required", stage="provenance")
    judge_revision = payload.get("judge_revision")
    if judge_revision in ("", None):
        judge_revision = None
    elif not isinstance(judge_revision, str) or SHA40.fullmatch(judge_revision) is None:
        raise RepairError("unpinned judge revision", stage="provenance")
    source = payload.get("circuitkit_source")
    if source in ("", None):
        source_path = DEFAULT_SOURCE
    else:
        if not isinstance(source, str):
            raise RepairError("circuitkit source path is unreadable", stage="circuitkit")
        source_path = Path(source)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "model_id": TINY_AYA_ID,
        "model_revision": PINNED_REVISION,
        "chat_template_sha256": template,
        "load_dtype": EVIDENCED_LOAD_DTYPE,
        "declared_score_dtype": DECLARED_SCORE_DTYPE,
        "score_dtype_evidenced": False,
        "seed": REQUIRED_SEED,
        "method": "base",
        "declared_arm": DECLARED_ARM,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "system_prompt": None,
            "temperature": None,
            "top_p": None,
        },
        "judge_model_id": "allenai/wildguard",
        "judge_revision": judge_revision,
        "circuit_scores": payload.get("circuit_scores"),
        "circuitkit_source": source_path,
    }


def require_original_credit(credit: Mapping[str, Any], *, load_dtype: str) -> dict[str, Any]:
    """Original short-plus-4196 fp32 credit only. Variant and fixture credit do not load weights."""
    if credit.get("credit") is not True or credit.get("reason") != ORIGINAL_CREDIT_REASON:
        raise RepairError(
            "CircuitKIT weight load requires recomputed original fp32 short and 4196 credit",
            stage="parity",
        )
    original = credit.get("original")
    if not isinstance(original, dict) or original.get("status") != "pass":
        raise RepairError(
            "CircuitKIT weight load requires recomputed original fp32 short and 4196 credit",
            stage="parity",
        )
    if credit.get("ignored_ingestion_json") is not True:
        raise RepairError("parity credit was not recomputed from the artifact files", stage="parity")
    if load_dtype != EVIDENCED_LOAD_DTYPE:
        raise RepairError(
            f"dtype evidence covers float32 only. Requested dtype {load_dtype} has no recomputed parity evidence",
            stage="parity",
        )
    return {
        "reason": ORIGINAL_CREDIT_REASON,
        "evidenced_dtype": EVIDENCED_LOAD_DTYPE,
        "declared_score_dtype": DECLARED_SCORE_DTYPE,
        "score_dtype_evidenced": False,
        "variant_used": False,
    }


def require_circuit_scores(scores: Any) -> dict[str, float]:
    if not isinstance(scores, dict) or not scores:
        raise RepairError("caller-supplied node scores are missing", stage="circuitkit")
    cleaned: dict[str, float] = {}
    for name, value in scores.items():
        if not isinstance(name, str) or NODE_NAME.fullmatch(name) is None:
            raise RepairError(
                "circuit score name does not match a pinned ActivationSteering node",
                stage="circuitkit",
            )
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RepairError("circuit score is not finite", stage="circuitkit")
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            raise RepairError("circuit score is not finite", stage="circuitkit")
        cleaned[name] = number
    return cleaned


def load_runtime_index(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RepairError("missing repair index", stage="membership")
    try:
        payload = strict_json_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepairError("repair index is unreadable", stage="membership") from exc
    if not isinstance(payload, list) or not payload:
        raise RepairError("repair index is empty", stage="membership")
    derived_shape = any(
        isinstance(item, dict)
        and ("review_status" in item or item.get("source_dataset") == REVIEWED_VARIANT_SOURCE)
        for item in payload
    )
    if derived_shape:
        return _load_derived_index(payload)
    return load_index(path)


def _load_derived_index(payload: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Index loader that keeps review_status for lineage admission."""
    from .membership import _token
    from src.eval.dev_runner import is_lower_hex

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in payload:
        if not isinstance(raw, dict):
            raise RepairError("repair index row is unreadable", stage="membership")
        if any(key in raw for key in ("prompt", "response", "completion", "text")):
            raise RepairError("repair index must stay metadata-only", stage="membership")
        extra = set(raw) - set(INDEX_FIELDS) - {"review_status"}
        if extra or any(field not in raw for field in INDEX_FIELDS):
            raise RepairError("repair index field set is wrong", stage="membership")
        row = {field: _token(raw[field], field) for field in INDEX_FIELDS}
        if row["split"] not in {"calibration", "dev"}:
            raise RepairError(f"repair index refuses split {row['split']}", stage="membership")
        if row["lang"] not in ITEM_LANGS or row["hs_label"] not in ITEM_HS_LABELS:
            raise RepairError("canonical label mismatch", stage="membership")
        if not is_lower_hex(row["prompt_hash"], 64):
            raise RepairError("missing prompt_hash", stage="membership")
        if row["sample_id"] in seen:
            raise RepairError("duplicate repair index id", stage="membership")
        seen.add(row["sample_id"])
        if raw.get("source_dataset") == REVIEWED_VARIANT_SOURCE or "review_status" in raw:
            if raw.get("review_status") != "reviewed":
                raise RepairError("unreviewed index row", stage="membership")
            row["review_status"] = "reviewed"
        rows.append(row)
    return rows


def admit_rows(
    rows: Sequence[Mapping[str, str]],
    authority: Mapping[str, Any],
    *,
    allowed_splits: frozenset[str],
    reviewed_lineage_path: Path | None,
    review_sheet_path: Path | None,
    reviewed_pins: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Canonical and reviewed-variant admission. Does not open prompts or build a model."""
    if not rows:
        raise RepairError("repair index is empty", stage="membership")
    derived = [row for row in rows if row.get("source_dataset") == REVIEWED_VARIANT_SOURCE]
    canonical = [row for row in rows if row.get("source_dataset") != REVIEWED_VARIANT_SOURCE]
    if canonical:
        assert_repair_membership(canonical, authority, allowed_splits=allowed_splits)
    lineage_report = None
    if derived:
        if reviewed_lineage_path is None or review_sheet_path is None or not reviewed_pins:
            raise RepairError("reviewed lineage contract is required", stage="membership")
        if any(field not in reviewed_pins for field in REVIEW_PIN_FIELDS):
            raise RepairError("reviewed lineage contract is required", stage="membership")
        if any("review_status" not in row for row in derived):
            raise RepairError("reviewed lineage contract is required", stage="membership")
        try:
            lineage_report = verify_ledger_lineage(
                lineage_bytes=reviewed_lineage_path.read_bytes(),
                review_sheet_bytes=review_sheet_path.read_bytes(),
                index_rows=derived,
                authority=authority,
                pins=reviewed_pins,
            )
        except ReviewedVariantError as exc:
            raise RepairError(str(exc), stage="membership") from None
        except OSError as exc:
            raise RepairError("reviewed lineage contract is required", stage="membership") from exc
    elif reviewed_lineage_path is not None or review_sheet_path is not None or reviewed_pins:
        raise RepairError("canonical rows cannot be relabelled with reviewed lineage", stage="membership")
    return {"derived": len(derived), "canonical": len(canonical), "lineage": lineage_report}


def _git_bytes(repo: Path, args: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout


def inspect_circuitkit_source(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False,
        "path": str(path),
        "expected": SOURCE_SHA,
        "head": None,
        "dirty": None,
        "steering_sha256": None,
        "steering_matches_head": False,
        "reason": "CircuitKIT source is missing",
    }
    if not path.is_dir():
        return report
    code, head = _git_bytes(path, ["rev-parse", "HEAD"])
    head = head.strip()
    report["head"] = head if SHA40.fullmatch(head) else None
    if code != 0 or report["head"] != SOURCE_SHA:
        report["reason"] = "CircuitKIT source SHA does not match the pin"
        return report
    code, porcelain = _git_bytes(path, ["status", "--porcelain", "--untracked-files=normal"])
    if code != 0:
        report["reason"] = "CircuitKIT worktree status is unreadable"
        return report
    report["dirty"] = bool(porcelain.strip())
    steering = path / STEERING_GIT_PATH
    if not steering.is_file():
        report["reason"] = "ActivationSteering source file is missing"
        return report
    raw = steering.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    report["steering_sha256"] = hashlib.sha256(raw).hexdigest()
    hashed = subprocess.run(
        ["git", "-C", str(path), "hash-object", STEERING_GIT_PATH],
        check=False,
        capture_output=True,
        text=True,
    )
    show_hash = subprocess.run(
        ["git", "-C", str(path), "rev-parse", f"HEAD:{STEERING_GIT_PATH}"],
        check=False,
        capture_output=True,
        text=True,
    )
    report["steering_matches_head"] = (
        hashed.returncode == 0
        and show_hash.returncode == 0
        and hashed.stdout.strip() == show_hash.stdout.strip()
    )
    if report["dirty"]:
        report["reason"] = "CircuitKIT worktree is dirty"
        return report
    if not report["steering_matches_head"]:
        report["reason"] = "ActivationSteering bytes do not match the pinned commit"
        return report
    report["ok"] = True
    report["reason"] = None
    return report


def repair_source_provenance(root: Path | None = None) -> dict[str, Any]:
    root = root or ROOT
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    try:
        dirty_raw = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
        dirty: bool | None = bool(dirty_raw.strip())
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:src/repair/real_runtime.py"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except (OSError, subprocess.CalledProcessError):
        committed = None
    head_code, head = _git_bytes(root, ["rev-parse", "HEAD"])
    return {
        "repair_source_sha256": hashlib.sha256(source).hexdigest(),
        "git_dirty": dirty,
        "source_matches_commit": committed == source,
        "git_head": head.strip() if head_code == 0 else None,
    }


def _require_clean(code: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    if code.get("git_dirty") is not False or code.get("source_matches_commit") is not True:
        raise RepairError(
            "real run requires committed repair source and a clean git worktree",
            stage="provenance",
        )
    if source.get("ok") is not True:
        raise RepairError(str(source.get("reason") or "CircuitKIT source check failed"), stage="circuitkit")


def assert_import_order_state() -> None:
    if "transformer_lens" in sys.modules and "circuitkit" not in sys.modules:
        raise RepairError(
            "transformer_lens was imported before circuitkit",
            stage="circuitkit",
        )


def load_tiny_aya(pins: Mapping[str, Any], source: Path) -> Any:
    """Load the pinned revision. Imports circuitkit before transformer_lens."""
    assert_import_order_state()
    src = source / "src"
    if not src.is_dir():
        raise RepairError("CircuitKIT source is missing", stage="circuitkit")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import circuitkit  # noqa: F401
    import circuitkit.backends._tl_compat  # noqa: F401
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformer_lens import HookedTransformer
    except ImportError as exc:
        raise RepairError(f"target runtime missing torch, transformers, or transformer_lens: {exc.__class__.__name__}", stage="runtime") from None
    if "circuitkit" not in sys.modules or sys.modules["circuitkit"].__name__ != "circuitkit":
        raise RepairError("circuitkit import did not stay ahead of transformer_lens", stage="circuitkit")
    model_id = str(pins["model_id"])
    revision = str(pins["model_revision"])
    if model_id != TINY_AYA_ID or revision != PINNED_REVISION:
        raise RepairError("model revision is not the pinned Tiny Aya revision", stage="provenance")
    dtype = getattr(torch, EVIDENCED_LOAD_DTYPE)
    tokenizer = None
    hf_model = None
    tl_model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, str) or hashlib.sha256(template.encode("utf-8")).hexdigest() != pins["chat_template_sha256"]:
            raise RepairError("chat template hash mismatch", stage="provenance")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        hf_model.eval()
        torch.manual_seed(REQUIRED_SEED)
        signature = __import__("inspect").signature(HookedTransformer.from_pretrained)
        kwargs: dict[str, Any] = {"hf_model": hf_model, "device": "cpu", "dtype": dtype}
        if "tokenizer" in signature.parameters:
            kwargs["tokenizer"] = tokenizer
        tl_model = HookedTransformer.from_pretrained(model_id, **kwargs)
        tl_model.eval()
        return TinyAyaHandle(
            tl_model=tl_model,
            hf_model=hf_model,
            tokenizer=tokenizer,
            torch_module=torch,
            pins=pins,
        )
    except RepairError:
        _release(tl_model, hf_model)
        raise
    except BaseException as exc:
        _release(tl_model, hf_model)
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RepairError(f"Tiny Aya load failed: {exc.__class__.__name__}", stage="runtime") from None


def _release(*models: Any) -> None:
    for model in models:
        if model is not None:
            del model
    gc.collect()


class TinyAyaHandle:
    """Real weight handle. real_weights is true only after from_pretrained returns."""

    uses_tensors = True
    real_weights = True
    weights_downloaded = True
    not_tiny_aya = False
    model_id = TINY_AYA_ID

    def __init__(self, *, tl_model: Any, hf_model: Any, tokenizer: Any, torch_module: Any, pins: Mapping[str, Any]) -> None:
        self.model = tl_model
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.pins = dict(pins)
        self.unloaded = False
        self._installed = 0

    def add_hook(self, name: str, fn: Any, dir: str = "fwd") -> None:
        self.model.add_hook(name, fn, dir=dir)
        self._installed += 1

    def reset_hooks(self) -> None:
        reset = getattr(self.model, "reset_hooks", None)
        if not callable(reset):
            raise RepairError("model has no reset_hooks", stage="runtime")
        reset()
        self._installed = 0

    def hook_count(self) -> int:
        hook_dict = getattr(self.model, "hook_dict", None)
        if isinstance(hook_dict, dict):
            return sum(len(value) for value in hook_dict.values())
        return self._installed

    def encode_user(self, prompt: str) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            user_only_messages(prompt),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        return [int(token) for token in encoded["input_ids"][0].tolist()]

    def eos_ids(self) -> set[int]:
        found: set[int] = set()
        for candidate in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(getattr(self.model, "tokenizer", None), "eos_token_id", None),
        ):
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                found.add(candidate)
        return found

    def pad_id(self) -> int | None:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if isinstance(pad, int) and not isinstance(pad, bool):
            return pad
        return None

    def step(self, token_ids: Sequence[int]) -> list[float]:
        torch = self.torch
        ids = torch.tensor([list(token_ids)], dtype=torch.long)
        with torch.inference_mode():
            out = self.model(ids)
        logits = out[0, -1] if getattr(out, "ndim", 0) == 3 else out
        return [float(value) for value in logits.detach().float().cpu().tolist()]

    def protocol_activation(self, token_ids: Sequence[int], layer_name: str) -> list[float]:
        torch = self.torch
        captured: dict[str, Any] = {}

        def hook_fn(activation: Any, hook: Any) -> None:
            captured["act"] = activation.detach()
            return None

        self.add_hook(layer_name, hook_fn, dir="fwd")
        try:
            ids = torch.tensor([list(token_ids)], dtype=torch.long)
            with torch.inference_mode():
                self.model(ids)
        finally:
            self.reset_hooks()
        if "act" not in captured:
            raise RepairError("protocol hook did not see a residual", stage="runtime")
        act = captured["act"]
        if getattr(act, "ndim", 0) != 3:
            raise RepairError("protocol residual is not position-resolved", stage="runtime")
        last = _last_nonpad(list(token_ids), self.pad_id())
        return [float(value) for value in act[0, last].float().cpu().tolist()]

    def unload(self) -> None:
        self.model = None
        self.hf_model = None
        self.tokenizer = None
        self.unloaded = True
        gc.collect()
        torch = self.torch
        if torch is not None and bool(getattr(torch.cuda, "is_available", lambda: False)()):
            torch.cuda.empty_cache()


def _last_nonpad(token_ids: Sequence[int], pad_id: int | None) -> int:
    if not token_ids:
        raise RepairError("missing token ids", stage="runtime")
    if pad_id is None:
        return len(token_ids) - 1
    for index in range(len(token_ids) - 1, -1, -1):
        if int(token_ids[index]) != pad_id:
            return index
    raise RepairError("no nonpad token", stage="runtime")


def _argmax(logits: Sequence[float]) -> int:
    best = 0
    for index, value in enumerate(logits):
        if value > logits[best]:
            best = index
    return best


def install_alpha_hook(model: Any, vector: Sequence[float], alpha: float, layer_name: str, counter: dict[str, int]) -> None:
    if getattr(model, "real_weights", False):
        raise RepairError("real weights refuse the residual hook", stage="runtime")
    if getattr(model, "uses_tensors", False):
        torch = model.torch

        def hook_fn(activation: Any, hook: Any) -> Any:
            counter["n"] += 1
            vector_t = torch.tensor(list(vector), device=activation.device, dtype=activation.dtype)
            while vector_t.ndim < activation.ndim:
                vector_t = vector_t.unsqueeze(0)
            return activation + vector_t * activation.new_tensor(float(alpha))
    else:
        def hook_fn(activation: Any, hook: Any) -> list[float]:
            counter["n"] += 1
            return add_scaled(list(activation), vector, alpha)

    model.add_hook(layer_name, hook_fn, dir="fwd")


def generate_greedy(
    model: Any,
    token_ids: Sequence[int],
    *,
    vector: Sequence[float] | None,
    alpha: float | None,
    layer_name: str,
    hooks: bool,
    max_new_tokens: int,
    eos_ids: set[int],
) -> dict[str, Any]:
    """Fixture autoregressive loop. Real weights must call ActivationSteering.steer instead."""
    if hooks and getattr(model, "real_weights", False):
        raise RepairError("real weights refuse the residual hook", stage="runtime")
    if model.hook_count() != 0:
        raise RepairError("forward found a leftover hook", stage="runtime")
    counter = {"n": 0}
    installed = False
    try:
        if hooks:
            if vector is None or alpha is None:
                raise RepairError("hooked arm is missing alpha", stage="runtime")
            install_alpha_hook(model, vector, alpha, layer_name, counter)
            installed = True
            if model.hook_count() < 1:
                raise RepairError("effective hook count is 0", stage="runtime")
        ids = [int(token) for token in token_ids]
        new_ids: list[int] = []
        for _ in range(max_new_tokens):
            logits = model.step(ids)
            if not isinstance(logits, list) or not logits:
                raise RepairError("model step did not return logits", stage="runtime")
            nxt = _argmax(logits)
            new_ids.append(nxt)
            ids.append(nxt)
            if nxt in eos_ids:
                break
        if hooks and counter["n"] < 1:
            raise RepairError("effective hook count is 0", stage="runtime")
        flags = generation_flags(new_ids, eos_ids, max_new_tokens=max_new_tokens)
        return {
            "new_token_ids": new_ids,
            "hook_count": counter["n"] if hooks else 0,
            "eos": flags["eos"],
            "truncated": flags["truncated"],
            "new_token_count": flags["new_token_count"],
        }
    finally:
        if installed:
            model.reset_hooks()
            if model.hook_count() != 0:
                raise RepairError("hooks survived cleanup", stage="runtime")


def generate_with_activation_steering(
    bundle: Any,
    token_ids: Sequence[int],
    *,
    circuit_scores: Mapping[str, float],
    steering_vectors: Mapping[str, Sequence[float]],
    coefficient: float,
    credit: Mapping[str, Any],
    max_new_tokens: int,
    eos_ids: set[int],
) -> dict[str, Any]:
    """One ActivationSteering.steer forward per new token. Does not install the residual hook."""
    if getattr(bundle, "real_weights", False) is not True:
        raise RepairError("ActivationSteering.steer is the real-weight path", stage="runtime")
    if not steering_vectors:
        raise RepairError("real weights refuse the residual hook; ActivationSteering vectors are missing", stage="circuitkit")
    if bundle.hook_count() != 0:
        raise RepairError("forward found a leftover hook", stage="runtime")
    from .circuitkit_backend import steer_coefficient

    ids = [int(token) for token in token_ids]
    new_ids: list[int] = []
    counts: list[int] = []
    prepared = {
        name: _torch_tensor(bundle, list(values), dtype_name="float32")
        for name, values in steering_vectors.items()
    }
    try:
        for _ in range(max_new_tokens):
            result = steer_coefficient(
                bundle.model,
                circuit_scores,
                prepared,
                _torch_tensor(bundle, [ids], dtype_name="long"),
                float(coefficient),
                credit=credit,
            )
            if bundle.hook_count() != 0:
                raise RepairError("hooks survived cleanup", stage="runtime")
            count = result.get("effective_hook_count")
            if type(count) is not int or count < 1:
                raise RepairError("effective hook count is 0", stage="runtime")
            counts.append(count)
            nxt = _argmax(_last_logits(result.get("output")))
            new_ids.append(nxt)
            ids.append(nxt)
            if nxt in eos_ids:
                break
    finally:
        if bundle.hook_count() != 0:
            bundle.reset_hooks()
    if not counts:
        raise RepairError("effective hook count is 0", stage="runtime")
    flags = generation_flags(new_ids, eos_ids, max_new_tokens=max_new_tokens)
    return {
        "new_token_ids": new_ids,
        "hook_count": min(counts),
        "steer_calls": len(new_ids),
        "library_call": "ActivationSteering.steer",
        "eos": flags["eos"],
        "truncated": flags["truncated"],
        "new_token_count": flags["new_token_count"],
    }


def _torch_tensor(bundle: Any, data: Any, *, dtype_name: str) -> Any:
    torch = getattr(bundle, "torch", None)
    if torch is None:
        raise RepairError("ActivationSteering.steer needs the loaded torch runtime", stage="runtime")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise RepairError("ActivationSteering.steer needs the loaded torch runtime", stage="runtime")
    return torch.tensor(data, dtype=dtype)


def _last_logits(output: Any) -> list[float]:
    if isinstance(output, list):
        row: Any = output
        while isinstance(row, list) and row and isinstance(row[0], list):
            row = row[-1]
        if not isinstance(row, list) or not row:
            raise RepairError("ActivationSteering.steer returned no logits", stage="runtime")
        return [_finite_float(value) for value in row]
    current = output
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    if hasattr(current, "float"):
        current = current.float()
    if hasattr(current, "cpu"):
        current = current.cpu()
    ndim = getattr(current, "ndim", None)
    if ndim == 3:
        current = current[0, -1]
    elif ndim == 2:
        current = current[-1]
    elif ndim != 1:
        raise RepairError("ActivationSteering.steer returned no logits", stage="runtime")
    values = current.tolist() if hasattr(current, "tolist") else None
    if not isinstance(values, list) or not values:
        raise RepairError("ActivationSteering.steer returned no logits", stage="runtime")
    return [_finite_float(value) for value in values]


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _label(injected: bool) -> dict[str, Any]:
    if injected:
        return {
            "artifact_kind": "fixture",
            "split": "example",
            "not_tiny_aya": True,
            "weights_executed_by_this_command": False,
            "runtime_verified_on_pretrained_weights": False,
        }
    return {
        "artifact_kind": "run",
        "split": "dev",
        "not_tiny_aya": False,
        "weights_executed_by_this_command": True,
        "runtime_verified_on_pretrained_weights": True,
    }


def _common_stamp(payload: dict[str, Any], *, injected: bool, pins: Mapping[str, Any], credit: Mapping[str, Any]) -> None:
    label = _label(injected)
    payload.update(label)
    payload["model_id"] = TINY_AYA_ID if not injected else payload.get("model_id", "fixture/injected")
    payload["model_revision"] = pins["model_revision"] if not injected else "not-loaded"
    payload["chat_template_sha256"] = pins["chat_template_sha256"] if not injected else None
    payload["dtype"] = pins["load_dtype"]
    payload["declared_score_dtype"] = DECLARED_SCORE_DTYPE
    payload["score_dtype_evidenced"] = False
    payload["load_dtype_evidence"] = credit.get("reason")
    payload["not_wildguard"] = True
    payload["not_a_scored_test"] = True
    payload["proven_refusal_mechanism"] = False
    payload["proposed_protocol"] = True
    payload["primary_asr"] = None
    payload["metrics_eligible"] = False
    payload["factory_injected"] = injected


def locate_with_model(
    *,
    protocol_path: Path,
    provenance_path: Path,
    runtime_config_path: Path,
    index_path: Path,
    manifest_path: Path,
    prompts_path: Path,
    out_path: Path,
    parity_run: Path | None,
    circuit_scores: Mapping[str, float] | None = None,
    circuitkit_source: Path | None = None,
    reviewed_lineage_path: Path | None = None,
    review_sheet_path: Path | None = None,
    reviewed_pins: Mapping[str, str] | None = None,
    model_factory: Callable[..., Any] | None = None,
    prompt_loader: Callable[..., list[dict[str, Any]]] | None = None,
    code_check: Callable[[Path], Mapping[str, Any]] | None = None,
    source_check: Callable[[Path], Mapping[str, Any]] | None = None,
    allow_unpinned_manifest: bool = False,
) -> dict[str, Any]:
    if allow_unpinned_manifest:
        raise RepairError("unpinned manifests are fixture-only", stage="membership")
    protocol = load_protocol(protocol_path)
    if protocol["layer_name"] != LAYER_NAME or protocol["layer_index"] != 0:
        raise RepairError("repair protocol layer is not blocks.0.hook_resid_post", stage="protocol")
    provenance = load_provenance(provenance_path)
    from .parity import locate_credit

    credit = locate_credit(parity_run, accept_variant=provenance["accept_preregistered_bf16_variant"])
    if credit.get("credit") is not True:
        raise RepairError(
            "CircuitKIT locate refused: parity credit is missing, failed, or incomplete",
            stage="parity",
        )
    runtime = load_runtime_config(runtime_config_path)
    dtype_evidence = require_original_credit(credit, load_dtype=runtime["load_dtype"])
    scores = require_circuit_scores(circuit_scores if circuit_scores is not None else runtime["circuit_scores"])
    authority = load_authority(manifest_path, require_pinned_hash=True)
    try:
        rows = load_runtime_index(index_path)
    except RepairError:
        raise
    admit_rows(
        rows,
        authority,
        allowed_splits=frozenset({"calibration"}),
        reviewed_lineage_path=reviewed_lineage_path,
        review_sheet_path=review_sheet_path,
        reviewed_pins=reviewed_pins,
    )
    require_both_classes(rows, stage="direction")
    source_path = circuitkit_source or runtime["circuitkit_source"]
    source_report = (source_check or inspect_circuitkit_source)(source_path)
    code = (code_check or repair_source_provenance)(ROOT)
    _require_clean(code, source_report)
    injected = model_factory is not None
    factory = model_factory or load_tiny_aya
    bundle = None
    fatal: BaseException | None = None
    try:
        bundle = factory(runtime, source_path)
        assert_prompt_path_allowed(prompts_path)
        prompts = (prompt_loader or load_prompts)(prompts_path, rows)
        records = _protocol_records(bundle, prompts, protocol)
        direction = extract_direction(
            records,
            layer_index=protocol["layer_index"],
            layer_name=str(protocol["layer_name"]),
            position=protocol["position"],
            target_label=str(protocol["target_hs_label"]),
            source_label=str(protocol["source_hs_label"]),
            protocol_sha256=protocol["sha256"],
            protocol_id=str(protocol["protocol_id"]),
        )
        library = _library_extract(bundle, prompts, scores, credit, injected=injected)
    except BaseException as exc:
        fatal = exc
        direction = None
        library = None
    finally:
        if bundle is not None and callable(getattr(bundle, "unload", None)):
            bundle.unload()
    if isinstance(fatal, KeyboardInterrupt):
        raise fatal
    if fatal is not None:
        if isinstance(fatal, RepairError):
            raise fatal
        raise RepairError(f"locate failed: {fatal.__class__.__name__}", stage="runtime") from None
    assert direction is not None and library is not None
    direction["calibration_members"] = [
        {key: row[key] for key in ("sample_id", "item_id", "family_id", "source_dataset", "source_id", "lang", "hs_label", "split")}
        for row in rows
    ]
    direction["canonical_manifest_sha256"] = authority["sha256"]
    direction["circuitkit_source"] = source_report
    direction["repair_source"] = dict(code)
    direction["activation_steering"] = library
    direction["activation_source"] = "injected_fixture" if injected else "in_process_model"
    direction["position_reduction"] = "last_nonpad_token"
    direction["vector_norm_kind"] = "raw_l2"
    _common_stamp(direction, injected=injected, pins=runtime, credit=dtype_evidence)
    if injected:
        direction["model_id"] = "fixture/injected"
        direction["not_tiny_aya"] = True
    direction["provenance_sha256"] = provenance["sha256"]
    direction["runtime_config_sha256"] = runtime["sha256"]
    direction["content_sha256"] = content_sha256(direction)
    write_json(out_path, direction)
    return direction


def _protocol_records(bundle: Any, prompts: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    layer_name = str(protocol["layer_name"])
    records = []
    for row in prompts:
        tokens = _encode(bundle, row)
        if hasattr(bundle, "protocol_activation"):
            vector = bundle.protocol_activation(tokens, layer_name)
        else:
            vector = _fixture_activation(bundle, tokens)
        if bundle.hook_count() != 0:
            raise RepairError("hooks survived protocol extraction", stage="runtime")
        records.append({
            "sample_id": row["sample_id"],
            "split": row["split"],
            "hs_label": row["hs_label"],
            "layer_index": protocol["layer_index"],
            "position": protocol["position"],
            "vector": vector,
        })
    return records


def _encode(bundle: Any, row: Mapping[str, Any]) -> list[int]:
    if getattr(bundle, "real_weights", False):
        if row.get("fixture_token_ids") is not None:
            raise RepairError("fixture token ids are not valid on a run", stage="runtime")
        return list(bundle.encode_user(row["prompt"]))
    supplied = row.get("fixture_token_ids")
    if not isinstance(supplied, list) or not supplied:
        raise RepairError("injected fixture is missing token ids", stage="runtime")
    return [int(token) for token in supplied]


def _fixture_activation(bundle: Any, token_ids: Sequence[int]) -> list[float]:
    if hasattr(bundle, "protocol_activation"):
        return list(bundle.protocol_activation(token_ids, LAYER_NAME))
    last = token_ids[-1]
    return [float(last), 0.0, 0.0, 1.0]


def _library_extract(
    bundle: Any,
    prompts: Sequence[Mapping[str, Any]],
    scores: Mapping[str, float],
    credit: Mapping[str, Any],
    *,
    injected: bool,
) -> dict[str, Any]:
    """Call ActivationSteering only on a real handle. Injected fixtures record that the call was not run."""
    if not getattr(bundle, "real_weights", False):
        return {
            "call": None,
            "ran": False,
            "reason": "injected fixture does not run ActivationSteering" if injected else "handle is not real weights",
            "not_the_protocol_resid_vector": True,
            "proven_refusal_mechanism": False,
        }
    from .circuitkit_backend import extract_last_token

    source_examples = []
    target_examples = []
    for row in prompts:
        example = {"input_ids": _encode(bundle, row)}
        if row["hs_label"] == "benign":
            source_examples.append(example)
        elif row["hs_label"] == "harm":
            target_examples.append(example)
    extracted = extract_last_token(
        bundle.model,
        scores,
        source_examples,
        target_examples,
        credit=credit,
    )
    vectors = extracted.get("vectors") or {}
    if not isinstance(vectors, Mapping) or not vectors:
        raise RepairError("CircuitKIT returned no steering vectors", stage="circuitkit")
    norms = {}
    as_lists: dict[str, list[float]] = {}
    for name, tensor in vectors.items():
        values = _vector_as_list(tensor)
        as_lists[str(name)] = values
        norms[str(name)] = _l2(values)
    return {
        "call": "ActivationSteering.compute_steering_vector",
        "ran": True,
        "return_all_positions": True,
        "position_reduction": "last_index_of_unpadded_batch",
        "nodes": extracted.get("nodes"),
        "vector_norms": norms,
        "vectors_as_lists": as_lists,
        "not_the_protocol_resid_vector": True,
        "protocol_resid_not_used_for_steering": True,
        "note": "Library nodes are attention or MLP hook points. Steering uses these vectors through ActivationSteering.steer. The protocol residual is recorded and is not the steered vector.",
        "proven_refusal_mechanism": False,
        "source_sha": SOURCE_SHA,
    }


def _vector_as_list(tensor: Any) -> list[float]:
    current = tensor
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    if hasattr(current, "float"):
        current = current.float()
    if hasattr(current, "cpu"):
        current = current.cpu()
    if hasattr(current, "reshape"):
        current = current.reshape(-1)
    if hasattr(current, "tolist"):
        values = current.tolist()
    elif isinstance(current, list):
        values = current
    else:
        raise RepairError("CircuitKIT vector is not numeric", stage="circuitkit")
    if not isinstance(values, list) or not values:
        raise RepairError("CircuitKIT vector is not numeric", stage="circuitkit")
    return [_finite_float(value) for value in values]


def _finite_float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairError("CircuitKIT vector is not numeric", stage="circuitkit")
    number = float(value)
    if number != number or number == float("inf") or number == float("-inf"):
        raise RepairError("CircuitKIT vector is not numeric", stage="circuitkit")
    return number


def _l2(values: Sequence[float]) -> float:
    return sum(value * value for value in values) ** 0.5


def _require_library_vectors(direction: Mapping[str, Any]) -> dict[str, list[float]]:
    library = direction.get("activation_steering")
    if not isinstance(library, dict) or library.get("ran") is not True:
        raise RepairError(
            "real weights refuse the residual hook; ActivationSteering vectors are missing",
            stage="circuitkit",
        )
    raw = library.get("vectors_as_lists")
    if not isinstance(raw, dict) or not raw:
        raise RepairError(
            "real weights refuse the residual hook; ActivationSteering vectors are missing",
            stage="circuitkit",
        )
    cleaned: dict[str, list[float]] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or name == "" or not isinstance(values, list) or not values:
            raise RepairError("CircuitKIT vector is not numeric", stage="circuitkit")
        cleaned[name] = [_finite_float(value) for value in values]
    return cleaned


def run_real_sweep(
    *,
    protocol_path: Path,
    provenance_path: Path,
    runtime_config_path: Path,
    sweep_path: Path,
    direction_path: Path,
    calibration_index_path: Path,
    dev_index_path: Path,
    dev_prompts_path: Path,
    manifest_path: Path,
    out_path: Path,
    parity_run: Path | None,
    judgments_path: Path | None = None,
    circuit_scores: Mapping[str, float] | None = None,
    circuitkit_source: Path | None = None,
    reviewed_lineage_path: Path | None = None,
    review_sheet_path: Path | None = None,
    reviewed_pins: Mapping[str, str] | None = None,
    model_factory: Callable[..., Any] | None = None,
    judge_factory: Callable[..., Any] | None = None,
    prompt_loader: Callable[..., list[dict[str, Any]]] | None = None,
    code_check: Callable[[Path], Mapping[str, Any]] | None = None,
    source_check: Callable[[Path], Mapping[str, Any]] | None = None,
    allow_unpinned_manifest: bool = False,
) -> dict[str, Any]:
    if allow_unpinned_manifest:
        raise RepairError("unpinned manifests are fixture-only", stage="membership")
    from .parity import locate_credit

    protocol = load_protocol(protocol_path)
    provenance = load_provenance(provenance_path)
    sweep_config = load_sweep_config(sweep_path)
    credit = locate_credit(parity_run, accept_variant=provenance["accept_preregistered_bf16_variant"])
    if credit.get("credit") is not True:
        raise RepairError(
            "CircuitKIT locate refused: parity credit is missing, failed, or incomplete",
            stage="parity",
        )
    runtime = load_runtime_config(runtime_config_path)
    dtype_evidence = require_original_credit(credit, load_dtype=runtime["load_dtype"])
    scores = require_circuit_scores(circuit_scores if circuit_scores is not None else runtime["circuit_scores"])
    authority = load_authority(manifest_path, require_pinned_hash=True)
    calibration_rows = load_runtime_index(calibration_index_path)
    dev_rows = load_runtime_index(dev_index_path)
    admit_rows(
        calibration_rows,
        authority,
        allowed_splits=frozenset({"calibration"}),
        reviewed_lineage_path=None,
        review_sheet_path=None,
        reviewed_pins=None,
    )
    admit_rows(
        dev_rows,
        authority,
        allowed_splits=frozenset({"dev"}),
        reviewed_lineage_path=reviewed_lineage_path,
        review_sheet_path=review_sheet_path,
        reviewed_pins=reviewed_pins,
    )
    direction = _load_direction(direction_path, protocol_sha256=protocol["sha256"])
    if direction.get("layer_index") != protocol["layer_index"]:
        raise RepairError("direction layer does not match the proposed protocol", stage="direction")
    vector = direction["vector"]
    if l2_norm(vector) == 0.0:
        raise RepairError("freeze refused: direction norm is zero", stage="direction")
    random_vector = random_norm_matched(vector, seed=7)
    source_path = circuitkit_source or runtime["circuitkit_source"]
    source_report = (source_check or inspect_circuitkit_source)(source_path)
    code = (code_check or repair_source_provenance)(ROOT)
    _require_clean(code, source_report)
    injected = model_factory is not None
    factory = model_factory or load_tiny_aya
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    generations_path = out_dir / "generations.jsonl"
    if generations_path.exists():
        raise RepairError("ledger generations already exist", stage="runtime")
    bundle = None
    fatal: BaseException | None = None
    arm_rows: dict[str, list[dict[str, Any]]] = {}
    prompts: list[dict[str, Any]] = []
    use_library = False
    try:
        bundle = factory(runtime, source_path)
        use_library = getattr(bundle, "real_weights", False) is True
        library_vectors = _require_library_vectors(direction) if use_library else {}
        prompts = (prompt_loader or load_prompts)(dev_prompts_path, dev_rows)
        eos_ids = set(bundle.eos_ids()) if hasattr(bundle, "eos_ids") else set()
        arms = _arm_plan(vector, random_vector)
        for arm in arms:
            produced = []
            steered_vectors = _steered_node_vectors(arm, library_vectors) if use_library and arm["hooks"] else None
            for row in prompts:
                tokens = _encode(bundle, row)
                if steered_vectors is None:
                    generated = generate_greedy(
                        bundle,
                        tokens,
                        vector=arm["vector"],
                        alpha=arm["alpha"],
                        layer_name=str(protocol["layer_name"]),
                        hooks=arm["hooks"],
                        max_new_tokens=MAX_NEW_TOKENS,
                        eos_ids=eos_ids,
                    )
                else:
                    generated = generate_with_activation_steering(
                        bundle,
                        tokens,
                        circuit_scores=scores,
                        steering_vectors=steered_vectors,
                        coefficient=float(arm["alpha"]),
                        credit=credit,
                        max_new_tokens=MAX_NEW_TOKENS,
                        eos_ids=eos_ids,
                    )
                record = {
                    "sample_id": row["sample_id"],
                    "family_id": row["family_id"],
                    "hs_label": row["hs_label"],
                    "arm_id": arm["id"],
                    "hook_count": generated["hook_count"],
                    "new_token_ids": generated["new_token_ids"],
                    "new_token_count": generated["new_token_count"],
                    "eos": generated["eos"],
                    "truncated": generated["truncated"],
                    "status": "ok",
                    "response": None,
                    "library_call": generated.get("library_call"),
                }
                if getattr(bundle, "real_weights", False) and hasattr(bundle, "decode"):
                    record["response"] = bundle.decode(generated["new_token_ids"])
                produced.append(record)
                _append_jsonl(generations_path, record)
            arm_rows[arm["id"]] = produced
        _check_controls(arm_rows)
    except BaseException as exc:
        fatal = exc
    finally:
        if bundle is not None and callable(getattr(bundle, "unload", None)):
            bundle.unload()
    if isinstance(fatal, KeyboardInterrupt):
        raise fatal
    if fatal is not None:
        if isinstance(fatal, RepairError):
            raise fatal
        raise RepairError(f"dev ledger failed: {fatal.__class__.__name__}", stage="runtime") from None
    if bundle is None or not getattr(bundle, "unloaded", False):
        raise RepairError("target phase was not unloaded", stage="runtime")
    judge_report = _judge_after_unload(
        bundle_unloaded=True,
        rows=arm_rows,
        prompts=prompts,
        runtime=runtime,
        injected=injected,
        judge_factory=judge_factory,
        out_dir=out_dir,
    )
    judgments = _read_judgments(judgments_path) if judgments_path is not None else None
    decision = decide_arm(
        dev_rows,
        judgments,
        over_refusal_max_rise=sweep_config["over_refusal_max_rise"],
        capability_max_drop_points=sweep_config["capability_max_drop_points"],
        bootstrap_reps=sweep_config["bootstrap_reps"],
        bootstrap_seed=sweep_config["bootstrap_seed"],
        min_families_for_accept=sweep_config["min_families_for_accept"],
    )
    if judgments is None or decision.get("decision") == "unavailable":
        decision = {**decision, "freeze": False}
    report: dict[str, Any] = {
        "sweep_id": sweep_config["sweep_id"],
        "sweep_sha256": sweep_config["sha256"],
        "protocol_sha256": protocol["sha256"],
        "direction_sha256": direction["content_sha256"],
        "canonical_manifest_sha256": authority["sha256"],
        "arms": {arm_id: _public_arm(rows) for arm_id, rows in arm_rows.items()},
        "alpha_zero_matches_base": True,
        "random_control": {
            "seed": 7,
            "match": "l2_norm",
            "extracted_l2": l2_norm(vector),
            "random_l2": l2_norm(random_vector),
            "norm_gap": abs(l2_norm(random_vector) - l2_norm(vector)),
        },
        "decision": decision,
        "freeze_allowed": bool(decision.get("freeze")),
        "judgments": judgments,
        "dev_members": [
            {key: row[key] for key in ("sample_id", "item_id", "family_id", "source_dataset", "source_id", "lang", "hs_label", "split")}
            for row in dev_rows
        ],
        "layer_index": direction["layer_index"],
        "declared_arm": DECLARED_ARM,
        "vector_sha256": hashlib.sha256(json.dumps(vector).encode("utf-8")).hexdigest(),
        "generations_sha256": sha256_file(generations_path),
        "judge": judge_report,
        "circuitkit_source": source_report,
        "repair_source": dict(code),
        "circuit_scores_present": True,
        "scores_are_not_a_discovered_circuit": True,
        "disclaimer": protocol["disclaimer"],
        "generation": "ActivationSteering.steer" if use_library else "per_step_hook_forward",
        "not_a_single_steer_call": not use_library,
        "library_steer_per_new_token": use_library,
        "library_call": "ActivationSteering.steer" if use_library else None,
    }
    _common_stamp(report, injected=injected, pins=runtime, credit=dtype_evidence)
    if injected:
        report["model_id"] = "fixture/injected"
    report["content_sha256"] = content_sha256(report)
    write_json(out_path, report)
    return report


def _steered_node_vectors(arm: Mapping[str, Any], library_vectors: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Seed 7 norm match per library node. CircuitKIT get_random_baseline_vectors has no seed."""
    if arm["id"] == "random_norm_matched_seed_7":
        return {name: random_norm_matched(values, seed=7) for name, values in library_vectors.items()}
    return {name: list(values) for name, values in library_vectors.items()}


def _arm_plan(vector: Sequence[float], random_vector: Sequence[float]) -> list[dict[str, Any]]:
    return [
        {"id": "base_no_intervention", "hooks": False, "alpha": None, "vector": None},
        {"id": "alpha_zero_with_wrapper", "hooks": True, "alpha": 0.0, "vector": vector},
        {"id": "random_norm_matched_seed_7", "hooks": True, "alpha": 1.0, "vector": random_vector},
        {"id": DECLARED_ARM, "hooks": True, "alpha": 1.0, "vector": vector},
    ]


def _check_controls(arm_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    base = arm_rows["base_no_intervention"]
    zero = arm_rows["alpha_zero_with_wrapper"]
    candidate = arm_rows[DECLARED_ARM]
    random_arm = arm_rows["random_norm_matched_seed_7"]
    if any(row["hook_count"] != 0 for row in base):
        raise RepairError("base forward found a hook", stage="runtime")
    if any(row["hook_count"] < 1 for row in zero):
        raise RepairError("effective hook count is 0", stage="runtime")
    for left, right in zip(base, zero):
        if left["new_token_ids"] != right["new_token_ids"]:
            raise RepairError("alpha zero is not equivalent to the base forward", stage="runtime")
    if any(row["hook_count"] < 1 for row in candidate) or any(row["hook_count"] < 1 for row in random_arm):
        raise RepairError("effective hook count is 0", stage="runtime")
    if all(left["new_token_ids"] == right["new_token_ids"] for left, right in zip(base, candidate)):
        raise RepairError("steering hook did not change the generation", stage="runtime")


def _public_arm(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "hs_label": row["hs_label"],
            "hook_count": row["hook_count"],
            "new_token_count": row["new_token_count"],
            "eos": row["eos"],
            "truncated": row["truncated"],
            "status": row["status"],
        }
        for row in rows
    ]


def _load_direction(path: Path, *, protocol_sha256: str) -> dict[str, Any]:
    from .direction import assert_frozen_direction

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError("missing frozen direction", stage="direction") from exc
    if not isinstance(payload, dict):
        raise RepairError("missing frozen direction", stage="direction")
    if payload.get("content_sha256") != content_sha256(payload):
        raise RepairError("frozen direction hash does not match its contents", stage="direction")
    return assert_frozen_direction(payload, protocol_sha256=protocol_sha256)


def _read_judgments(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RepairError("missing judgments", stage="selection")
    try:
        payload = strict_json_bytes(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RepairError("judgment file is unreadable", stage="selection") from exc
    if not isinstance(payload, dict):
        raise RepairError("judgment file is unreadable", stage="selection")
    return payload


def _judge_after_unload(
    *,
    bundle_unloaded: bool,
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
    prompts: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    injected: bool,
    judge_factory: Callable[..., Any] | None,
    out_dir: Path,
) -> dict[str, Any]:
    if not bundle_unloaded:
        raise RepairError("judge construction requires the target to be unloaded", stage="runtime")
    samples_path = out_dir / "samples.jsonl"
    revision = runtime.get("judge_revision")
    report: dict[str, Any] = {
        "judge_constructed": False,
        "judge_unloaded": True,
        "judge_status": "unavailable",
        "judge_reason": "unpinned judge revision" if revision is None else None,
        "primary_asr": None,
        "metrics_eligible": False,
    }
    if revision is None and judge_factory is None:
        for arm_rows in rows.values():
            for row in arm_rows:
                sample = dict(row)
                sample["category"] = None
                sample["response"] = None
                _append_jsonl(samples_path, sample)
        report["samples_sha256"] = sha256_file(samples_path) if samples_path.is_file() else None
        return report
    judge = None
    fatal: BaseException | None = None
    try:
        if judge_factory is not None:
            judge = judge_factory(runtime)
        else:
            judge = WildGuardPrimaryJudge(
                model_spec="hf:" + str(runtime["judge_model_id"]),
                judge_model_args={"revision": revision},
            )
        report["judge_constructed"] = True
        by_prompt = {row["sample_id"]: row for row in prompts}
        for arm_rows in rows.values():
            for row in arm_rows:
                sample = dict(row)
                sample["category"] = None
                text = row.get("response")
                if row["status"] != "ok" or not isinstance(text, str):
                    sample["category"] = None
                    sample["judge_reason"] = "missing response text"
                    _append_jsonl(samples_path, sample)
                    continue
                verdict = judge.judge_one(by_prompt[row["sample_id"]]["prompt"], text, prompt_id=row["sample_id"])
                metadata = dict(getattr(verdict, "metadata", None) or {})
                sample["category"] = assign_category(
                    status="ok",
                    truncated=bool(row["truncated"]),
                    judge_label=getattr(verdict, "label", None),
                    fields=metadata,
                )
                sample["judge_label"] = getattr(verdict, "label", None)
                _append_jsonl(samples_path, sample)
    except BaseException as exc:
        fatal = exc
    finally:
        if judge is not None and callable(getattr(judge, "unload", None)):
            try:
                judge.unload()
                report["judge_unloaded"] = True
            except BaseException as exc:
                report["judge_unloaded"] = False
                report["judge_release_error"] = exc.__class__.__name__
                if fatal is None:
                    fatal = exc
    if isinstance(fatal, KeyboardInterrupt):
        raise fatal
    if fatal is not None:
        report["judge_status"] = "unavailable"
        report["judge_reason"] = fatal.__class__.__name__
        report["primary_asr"] = None
        report["metrics_eligible"] = False
        return report
    report["judge_status"] = "judged_ineligible"
    report["judge_reason"] = "capability and category review are not complete"
    report["primary_asr"] = None
    report["metrics_eligible"] = False
    if samples_path.is_file():
        report["samples_sha256"] = sha256_file(samples_path)
    return report


def assert_accept_bindings(decision: Mapping[str, Any], *, min_families: int) -> None:
    if decision.get("decision") != "accept":
        return
    harm = decision.get("harm_improvement")
    low = harm.get("low") if isinstance(harm, dict) else None
    if isinstance(low, bool) or not isinstance(low, (int, float)) or not float(low) > 0.0:
        raise RepairError("accept refused: harm gate is missing", stage="freeze")
    if decision.get("over_refusal_gate") != "pass" or decision.get("capability_gate") != "pass":
        raise RepairError("accept refused: preservation gate is missing", stage="freeze")
    n_families = decision.get("n_families")
    minimum = decision.get("min_families_for_accept")
    if type(n_families) is not int or type(minimum) is not int:
        raise RepairError("accept refused: family count is missing", stage="freeze")
    if minimum != min_families or n_families < min_families:
        raise RepairError("accept refused: required family count was not met", stage="freeze")
    if decision.get("claim_level") != "eligible" or decision.get("deployable_claim") is not True:
        raise RepairError("accept refused: claim level is not eligible", stage="freeze")
    if decision.get("freeze") is not True:
        raise RepairError("accept refused: freeze bit does not match the gates", stage="freeze")


def deterministic_arm_records() -> list[dict[str, Any]]:
    """Replay the stand-in. Same tokens every process. Not pretrained weights."""
    model = DeterministicTensorFixture()
    vector = [20.0, 0.0, 0.0, 0.0]
    random_vector = random_norm_matched(vector, seed=7)
    tokens = [1, 4]
    recorded = []
    for arm in _arm_plan(vector, random_vector):
        generated = generate_greedy(
            model,
            tokens,
            vector=arm["vector"],
            alpha=arm["alpha"],
            layer_name=LAYER_NAME,
            hooks=arm["hooks"],
            max_new_tokens=4,
            eos_ids=set(),
        )
        if model.hook_count() != 0:
            raise RepairError("hooks survived the tensor fixture", stage="runtime")
        recorded.append({
            "arm_id": arm["id"],
            "new_token_ids": generated["new_token_ids"],
            "hook_count": generated["hook_count"],
        })
    return recorded


def verify_tensor_fixture_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("content_sha256") != content_sha256(dict(payload)):
        raise RepairError("tensor fixture hash does not match its contents", stage="freeze")
    if payload.get("artifact_kind") != "fixture" or payload.get("split") != "example":
        raise RepairError("tensor fixture label is wrong", stage="freeze")
    if payload.get("primary_asr") is not None or payload.get("metrics_eligible") is not False:
        raise RepairError("tensor fixture invented metrics", stage="freeze")
    if payload.get("not_pretrained_weights") is not True or payload.get("not_tiny_aya") is not True:
        raise RepairError("tensor fixture label is wrong", stage="freeze")
    fresh = deterministic_arm_records()
    if payload.get("arms") != fresh:
        raise RepairError("fresh process tensor fixture does not match the manifest", stage="freeze")
    base = fresh[0]["new_token_ids"]
    if fresh[1]["new_token_ids"] != base:
        raise RepairError("alpha zero is not equivalent to the base forward", stage="freeze")
    if fresh[3]["new_token_ids"] == base:
        raise RepairError("steering hook did not change the generation", stage="freeze")
    if any(row["hook_count"] < 1 for row in fresh[1:]):
        raise RepairError("effective hook count is 0", stage="freeze")
    return {
        "verified": True,
        "verification_pathway": "tensor_fixture",
        "weights_reloaded": False,
        "runtime_verified_on_pretrained_weights": False,
        "not_pretrained_weights": True,
        "artifact_kind": "fixture",
        "split": "example",
        "primary_asr": None,
        "metrics_eligible": False,
        "alpha_zero_matches_base": True,
        "not_tiny_aya": True,
        "not_wildguard": True,
        "not_a_scored_test": True,
    }


def write_tensor_fixture_evidence(out_dir: Path) -> dict[str, Any]:
    """Write a fixture ledger from the deterministic stand-in. primary_asr stays null."""
    recorded = deterministic_arm_records()
    base = recorded[0]["new_token_ids"]
    if recorded[1]["new_token_ids"] != base:
        raise RepairError("alpha zero is not equivalent to the base forward", stage="runtime")
    if recorded[3]["new_token_ids"] == base:
        raise RepairError("steering hook did not change the generation", stage="runtime")
    source_report = inspect_circuitkit_source(DEFAULT_SOURCE)
    env = _env_versions()
    payload = {
        "artifact_kind": "fixture",
        "split": "example",
        "primary_asr": None,
        "metrics_eligible": False,
        "not_tiny_aya": True,
        "not_wildguard": True,
        "not_pretrained_weights": True,
        "tensor_kind": "deterministic_last_token_function",
        "proven_refusal_mechanism": False,
        "proposed_protocol": True,
        "runtime_verified_on_pretrained_weights": False,
        "weights_executed_by_this_command": False,
        "arms": recorded,
        "alpha_zero_matches_base": True,
        "circuitkit_source": source_report,
        "environment": env,
        "model_id": "fixture/deterministic-tensor",
        "layer_name": LAYER_NAME,
    }
    payload["content_sha256"] = content_sha256(payload)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "fixture_results.json", payload)
    return payload


def _env_versions() -> dict[str, Any]:
    """Record what this interpreter can import. A missing module stays unavailable."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    packages = {
        "torch": "torch",
        "transformer_lens": "transformer-lens",
        "huggingface_hub": "huggingface-hub",
    }
    for module_name, distribution in packages.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            versions[module_name] = {"imported": False, "error": type(exc).__name__}
            continue
        try:
            distribution_version = version(distribution)
        except PackageNotFoundError:
            distribution_version = None
        versions[module_name] = {
            "imported": True,
            "module_version": getattr(module, "__version__", None),
            "distribution_version": distribution_version,
        }
    return versions


def loader_import_order() -> list[str]:
    """Statement order of imports inside load_tiny_aya. Used by the contract test."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_tiny_aya")
    found: list[str] = []

    def visit(statement: ast.AST) -> None:
        if isinstance(statement, ast.Import):
            found.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            found.append(statement.module or "")
        elif isinstance(statement, ast.Try):
            for child in statement.body:
                visit(child)

    for statement in function.body:
        visit(statement)
    return found
