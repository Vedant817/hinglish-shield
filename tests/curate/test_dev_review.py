import csv
import hashlib
import json

import pytest

from scripts.prepare_dev_review import prepare


def inputs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frozen": True, "families": {
        "f-dev": {"split": "dev", "members": [{"source_dataset": "fixture", "source_id": "dev"}]},
        "f-test": {"split": "test", "members": [{"source_dataset": "fixture", "source_id": "test"}]},
    }}))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    tmp_path.joinpath("manifest.json.sha256").write_text(digest)
    source = tmp_path / "source.csv"
    source.write_text("review_id,source,source_id,family_id,split,prompt\na,fixture,dev,f-dev,dev,review me\nb,fixture,test,f-test,test,DO NOT EXPORT\n")
    return source, manifest, tmp_path / "review.csv"


def test_review_excludes_test_and_preserves_human_edits(tmp_path):
    source, manifest, output = inputs(tmp_path)
    assert prepare(source, manifest, output)["test_rows"] == 0
    rows = list(csv.DictReader(output.open()))
    assert [r["review_id"] for r in rows] == ["a"]
    assert "DO NOT EXPORT" not in output.read_text()
    output.write_text("human edits")
    with pytest.raises(FileExistsError):
        prepare(source, manifest, output)
    assert output.read_text() == "human edits"


def test_review_rejects_manifest_drift(tmp_path):
    source, manifest, output = inputs(tmp_path)
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        prepare(source, manifest, output)
    assert not output.exists()


def test_review_rejects_split_laundering(tmp_path):
    source, manifest, output = inputs(tmp_path)
    source.write_text(source.read_text().replace("f-test,test", "f-test,dev"))
    with pytest.raises(ValueError, match="membership"):
        prepare(source, manifest, output)
    assert not output.exists()


def test_review_rejects_member_laundering_under_valid_dev_family(tmp_path):
    source, manifest, output = inputs(tmp_path)
    source.write_text(source.read_text().replace("fixture,test,f-test,test", "fixture,test,f-dev,dev"))
    with pytest.raises(ValueError, match="membership"):
        prepare(source, manifest, output)
    assert not output.exists()
