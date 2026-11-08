"""Target tokenizer identity capture for the private Kaggle notebook.

Downloads tokenizer metadata only. No weights, model imports, prompt files,
generation, or scoring. Never writes or logs the secret value. The exact
chat-template string is stored so the run can be audited without guessing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

MODEL_ID = "CohereLabs/tiny-aya-global"
PINNED_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
SCHEMA_VERSION = "target_identity_v2"
TOKENIZER_FILES = ("special_tokens_map.json", "tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)
MAX_FILE_BYTES = 64 * 1024**2
_SECRET = re.compile(r"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _redact(value: str) -> str:
    return _SECRET.sub("[redacted]", value)[:500]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select_chat_template(
    config: Any, tokenizer_json: Any
) -> tuple[str | None, str | None, str | None]:
    """Return (name, source file, exact template string) from downloaded bytes."""
    for source, payload in (
        ("tokenizer_config.json", config),
        ("tokenizer.json", tokenizer_json),
    ):
        if not isinstance(payload, Mapping):
            continue
        value = payload.get("chat_template")
        if isinstance(value, str) and value.strip():
            return None, source, value
        if isinstance(value, Mapping):
            name = "default" if isinstance(value.get("default"), str) else next(iter(value), None)
            template = value.get(name) if isinstance(name, str) else None
            if isinstance(template, str) and template.strip():
                return name, source, template
        if isinstance(value, list):
            entries = [
                entry
                for entry in value
                if isinstance(entry, Mapping)
                and isinstance(entry.get("template"), str)
                and entry["template"].strip()
            ]
            if entries:
                chosen = next(
                    (entry for entry in entries if entry.get("name") == "default"),
                    entries[0],
                )
                name = chosen.get("name") if isinstance(chosen.get("name"), str) else None
                return name, source, chosen["template"]
    return None, None, None


def tokenizer_file_names(names: Iterable[str]) -> list[str]:
    """Keep root-level tokenizer files only. No repository subdirectories."""
    selected: set[str] = set()
    allowed = set(TOKENIZER_FILES) | set(OPTIONAL_TOKENIZER_FILES)
    for name in names:
        if not isinstance(name, str) or "/" in name or "\\" in name:
            continue
        if name in allowed:
            selected.add(name)
    return sorted(selected)


def collect_identity(
    out_dir: Path,
    *,
    source_commit: str,
    source_sha256: str,
    notebook_run_id: str | None = None,
) -> dict[str, Any]:
    """Resolve the model revision, hash tokenizer files, and pin the template.

    The token is read from the environment by huggingface_hub. Only a presence
    boolean is recorded. A failure writes a status artifact and copies no
    tokenizer asset, so a later phase cannot mistake it for a pinned identity.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "setup-smoke",
        "split": "example",
        "status": "unavailable",
        "model_id": MODEL_ID,
        "requested_revision": PINNED_REVISION,
        "model_revision": None,
        "revision_matches_pin": False,
        "tokenizer_files": {},
        "tokenizer_config_sha256": None,
        "chat_template": None,
        "chat_template_sha256": None,
        "chat_template_name": None,
        "chat_template_source": None,
        "hf_token_present": bool((os.environ.get("HF_TOKEN") or "").strip()),
        "secret_status": "present" if (os.environ.get("HF_TOKEN") or "").strip() else "missing",
        "weights_loaded": False,
        "model_imported": False,
        "generation_performed": False,
        "metrics_eligible": False,
        "primary_asr": None,
        "source_commit": source_commit,
        "source_commit_role": "repository_data_pin",
        "source_sha256": source_sha256,
        "notebook_run_id": notebook_run_id,
        "failure_phase": None,
        "error_type": None,
        "error": None,
    }
    if not report["hf_token_present"]:
        report["failure_phase"] = "secret_missing"
        _write_json(out / "identity.json", report)
        return report
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=True)
        info = api.model_info(MODEL_ID, revision=PINNED_REVISION, token=True)
        resolved = getattr(info, "sha", None)
        report["model_revision"] = resolved
        if not isinstance(resolved, str) or resolved != PINNED_REVISION:
            report["failure_phase"] = "revision_unavailable"
            raise RuntimeError("resolved model revision does not match the pinned revision")
        report["revision_matches_pin"] = True
        names = tokenizer_file_names(
            [getattr(sibling, "rfilename", "") for sibling in (info.siblings or [])]
        )
        missing = [name for name in TOKENIZER_FILES if name not in names]
        if missing:
            report["failure_phase"] = "tokenizer_asset_missing"
            raise RuntimeError("missing tokenizer assets: " + ", ".join(missing))
        payloads: dict[str, Any] = {}
        hashes: dict[str, dict[str, Any]] = {}
        for name in names:
            path = hf_hub_download(
                repo_id=MODEL_ID, filename=name, revision=resolved, token=True
            )
            data = Path(path).read_bytes()
            if len(data) > MAX_FILE_BYTES:
                report["failure_phase"] = "tokenizer_asset_too_large"
                raise RuntimeError("tokenizer asset exceeds the metadata size limit")
            hashes[name] = {"sha256": sha256_bytes(data), "size": len(data)}
            if name in {"tokenizer_config.json", "tokenizer.json"}:
                payloads[name] = json.loads(data.decode("utf-8"))
            if name == "tokenizer_config.json":
                (out / "tokenizer_config.json").write_bytes(data)
        report["tokenizer_files"] = hashes
        report["tokenizer_config_sha256"] = hashes["tokenizer_config.json"]["sha256"]
        template_name, template_source, template = select_chat_template(
            payloads.get("tokenizer_config.json"), payloads.get("tokenizer.json")
        )
        if template is None:
            report["failure_phase"] = "chat_template_missing"
            raise RuntimeError("tokenizer assets do not contain a chat template")
        report.update(
            {
                "status": "metadata_retrieved",
                "chat_template": template,
                "chat_template_sha256": sha256_bytes(template.encode("utf-8")),
                "chat_template_name": template_name,
                "chat_template_source": template_source,
            }
        )
    except Exception as exc:
        report["status"] = "unavailable"
        if report["failure_phase"] is None:
            report["failure_phase"] = "download_failed"
        report["error_type"] = type(exc).__name__
        report["error"] = _redact(str(exc))
        (out / "tokenizer_config.json").unlink(missing_ok=True)
        report["tokenizer_files"] = {}
        report["tokenizer_config_sha256"] = None
    _write_json(out / "identity.json", report)
    return report
