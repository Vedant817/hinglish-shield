"""Synthetic HTTP metadata contracts, never model or account access."""
import hashlib
import io
import json
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from scripts.kaggle_tokenizer_metadata import URL, MetadataRedirect, collect_identity


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
    assert "fixture-credential" not in json.dumps(result)


def test_missing_secret_never_starts_http_request(tmp_path):
    def missing():
        raise RuntimeError("fixture secret provider details")
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP started without the existing credential")
    result = collect_identity(tmp_path / "missing", source_commit="a" * 40, source_sha256="b" * 64,
                              credential_provider=missing, open_url=forbidden)
    assert result["status"] == "unavailable" and result["error_type"] == "RuntimeError"
    assert "provider details" not in json.dumps(result)


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
