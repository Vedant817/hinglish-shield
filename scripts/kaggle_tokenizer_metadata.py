"""Metadata-only identity fetch for the existing private Kaggle notebook.

No weights, model imports, prompt files, generation or metrics. Never log a secret.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL_ID = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
URL = f"https://huggingface.co/{MODEL_ID}/raw/{REVISION}/tokenizer_config.json"
MAX_METADATA_BYTES = 2 * 1024 * 1024


def notebook_document(script_source: str, source_commit: str) -> dict:
    """A deterministic private-job document with papermill's required kernel."""
    digest = hashlib.sha256(script_source.encode("utf-8")).hexdigest()
    code = (
        "from pathlib import Path\nimport hashlib\n"
        f"source = {script_source!r}\n"
        "namespace = {'__name__': 'hs_tokenizer_metadata_job'}\n"
        "exec(compile(source, 'kaggle_tokenizer_metadata.py', 'exec'), namespace)\n"
        f"assert hashlib.sha256(source.encode()).hexdigest() == {digest!r}\n"
        "report = namespace['collect_identity'](Path('/kaggle/working/hs_tokenizer_metadata'), "
        f"source_commit={source_commit!r}, source_sha256={digest!r})\n"
        "print({'artifact_kind': 'setup-smoke', 'status': report['status'], "
        "'weights_loaded': False, 'generation_performed': False})\n"
    )
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}},
            "cells": [{"id": "metadata-only-fetch", "cell_type": "code", "metadata": {},
                       "source": code, "execution_count": None, "outputs": []}]}


def kernel_request() -> dict:
    """Build minimal metadata; never inherit a GPU machine shape."""
    return {"id": "vedant817/hinglish-shield-gpu-parity", "title": "hinglish-shield-gpu-parity",
            "code_file": "tokenizer_metadata.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}


class MetadataRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        destination = urlparse(newurl)
        if redirected is not None and (destination.hostname != "huggingface.co" or destination.scheme != "https"):
            redirected.remove_header("Authorization")
        return redirected


def existing_secret() -> str:
    credential = os.environ.get("HF_TOKEN")
    if not credential:
        from kaggle_secrets import UserSecretsClient
        credential = UserSecretsClient().get_secret("HF_TOKEN")
    if not isinstance(credential, str) or not credential:
        raise ValueError("existing secret is unavailable")
    return credential


def collect_identity(out: Path, *, source_commit: str, source_sha256: str,
                     credential_provider=existing_secret, open_url=None) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "artifact_kind": "setup-smoke", "split": "example", "status": "unavailable",
        "model_id": MODEL_ID, "model_revision": REVISION, "source_url": URL,
        "source_commit": source_commit, "script_sha256": source_sha256,
        "weights_loaded": False, "generation_performed": False,
        "metrics_eligible": False, "primary_asr": None,
    }
    try:
        credential = credential_provider()
        request = Request(URL, headers={"Authorization": "Bearer " + credential})
        opener = open_url or build_opener(MetadataRedirect()).open
        with opener(request, timeout=30) as response:
            raw = response.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("metadata is oversized")
        config = json.loads(raw)
        template = config.get("chat_template") if isinstance(config, dict) else None
        if not isinstance(template, str) or not template:
            raise ValueError("checkpoint chat template is not a nonempty string")
        # Save only the checkpoint asset, never credentials or request headers.
        (out / "tokenizer_config.json").write_bytes(raw)
        report.update(status="metadata_retrieved",
                      tokenizer_config_sha256=hashlib.sha256(raw).hexdigest(),
                      chat_template_sha256=hashlib.sha256(template.encode("utf-8")).hexdigest())
    except HTTPError as exc:
        report.update(error_type=type(exc).__name__, http_status=exc.code)
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    (out / "identity.json").write_bytes((json.dumps(report, indent=2) + "\n").encode())
    return report
