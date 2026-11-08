"""Synthetic HTTP metadata contracts, never model or account access."""
import hashlib
import io
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from scripts.kaggle_tokenizer_metadata import URL, MetadataRedirect, collect_identity, kernel_request, notebook_document


def test_metadata_only_hashes_exact_raw_asset_and_template(tmp_path):
    raw = json.dumps({"chat_template": "user: {{ messages }}\n"}).encode()
    def fetch(request, timeout):
        assert request.full_url == URL and timeout == 30
        assert request.get_header("Authorization") == "Bearer fixture-credential"
        return io.BytesIO(raw)
    out = tmp_path / "metadata"
    result = collect_identity(out, source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=lambda: "fixture-credential", open_url=fetch)
    assert result["status"] == "metadata_retrieved"
    assert result["auth_mode"] == "existing_secret" and result["secret_status"] == "present"
    assert "failure_phase" not in result
    assert result["tokenizer_config_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["chat_template_sha256"] == hashlib.sha256(b"user: {{ messages }}\n").hexdigest()
    assert result["weights_loaded"] is False and result["generation_performed"] is False
    assert "fixture-credential" not in (out / "identity.json").read_text()
    with pytest.raises(FileExistsError):
        collect_identity(out, source_commit="a" * 40, source_sha256="b" * 64)


def test_http_failure_is_unavailable_and_does_not_echo_provider_text(tmp_path):
    def fetch(*args, **kwargs):
        raise HTTPError(URL, 403, "fixture-credential sensitive provider text", {}, None)
    result = collect_identity(tmp_path / "denied", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=lambda: "fixture-credential", open_url=fetch)
    assert result["status"] == "unavailable" and result["http_status"] == 403
    assert result["failure_phase"] == "http_failure" and result["auth_mode"] == "existing_secret"
    assert "fixture-credential" not in json.dumps(result)


def test_connection_error_after_secret_is_http_failure_not_secret_missing(tmp_path):
    def fetch(*args, **kwargs):
        raise ConnectionError("fixture-credential sensitive transport text")
    result = collect_identity(tmp_path / "reset", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=lambda: "fixture-credential", open_url=fetch)
    assert result["failure_phase"] == "http_failure" and result["error_type"] == "ConnectionError"
    assert result["secret_status"] == "present"
    assert "fixture-credential" not in json.dumps(result)


def test_missing_secret_never_starts_http_request(tmp_path):
    def missing():
        raise RuntimeError("fixture secret provider details")
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP started without the existing credential")
    result = collect_identity(tmp_path / "missing", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=missing, open_url=forbidden)
    assert result["status"] == "unavailable" and result["error_type"] == "RuntimeError"
    assert result["failure_phase"] == "secret_missing" and result["public_fetch"] == "not_attempted"
    assert "without Authorization" in result["note"]
    assert "provider details" not in json.dumps(result)


def test_public_fallback_sends_no_authorization_and_can_pin(tmp_path):
    raw = json.dumps({"chat_template": "public-template"}).encode()
    def missing():
        raise RuntimeError("missing")
    def fetch(request, timeout):
        assert timeout == 30
        assert request.get_header("Authorization") is None
        return io.BytesIO(raw)
    result = collect_identity(tmp_path / "public", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=missing, open_url=fetch, public_fallback=True)
    assert result["status"] == "metadata_retrieved" and result["auth_mode"] == "unauthenticated_public"
    assert result["secret_status"] == "missing"
    assert "failure_phase" not in result and "error_type" not in result
    assert "fetched without Authorization" in result["note"]
    assert result["chat_template_sha256"] == hashlib.sha256(b"public-template").hexdigest()


def test_public_fallback_http_401_stays_secret_missing(tmp_path):
    def missing():
        raise RuntimeError("missing")
    def fetch(*args, **kwargs):
        raise HTTPError(URL, 401, "fixture-credential", {}, None)
    result = collect_identity(tmp_path / "anon401", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=missing, open_url=fetch, public_fallback=True)
    assert result["status"] == "unavailable" and result["failure_phase"] == "secret_missing"
    assert result["public_failure_phase"] == "http_failure" and result["public_http_status"] == 401
    assert "fixture-credential" not in json.dumps(result)


@pytest.mark.parametrize("destination", ["https://cdn.example/config", "http://huggingface.co/config"])
def test_authorization_is_not_forwarded_to_a_different_host_or_plain_http(destination):
    req = Request(URL, headers={"Authorization": "Bearer fixture-credential"})
    redirected = MetadataRedirect().redirect_request(req, None, 302, "", {}, destination)
    assert redirected.get_header("Authorization") is None


@pytest.mark.parametrize("template", [None, "", {}, []])
def test_missing_or_ambiguous_template_is_not_pinned(tmp_path, template):
    result = collect_identity(tmp_path / "invalid", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=lambda: "fixture-credential",
                              open_url=lambda *args, **kwargs: io.BytesIO(json.dumps({"chat_template": template}).encode()))
    assert result["status"] == "unavailable"
    assert "chat_template_sha256" not in result


def test_job_packaging_has_explicit_papermill_kernel_and_no_accelerator_shape():
    import ast
    import nbformat
    doc = notebook_document("# fixture source\n", "a" * 40)
    nbformat.validate(doc)
    assert doc["metadata"]["kernelspec"]["name"] == "python3"
    assert doc["metadata"]["kernelspec"]["language"] == "python"
    assert doc["cells"][0]["outputs"] == [] and doc["cells"][0]["execution_count"] is None
    ast.parse(doc["cells"][0]["source"])
    request = kernel_request()
    assert request["is_private"] is True
    assert request["enable_gpu"] is False and request["enable_tpu"] is False
    assert request["enable_internet"] is True
    assert "machine_shape" not in request and "accelerator" not in request
    assert "public_fallback=True" in doc["cells"][0]["source"]
