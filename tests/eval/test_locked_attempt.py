"""Synthetic local admission tests. Never read or generate real test prompts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

import pytest
import yaml

from src.eval import run_baseline as runner
from src.repair import freeze

REAL_VERIFY_MEASURED = freeze.verify_measured


@pytest.fixture
def locked(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_utc_now", lambda: datetime(2026, 11, 10, 18, 30, tzinfo=timezone.utc))
    monkeypatch.setattr(runner, "_git_full_commit", lambda: "c" * 40)
    # No runtime evidence exists here. The CPU verifier is isolated to exercise
    # admission and scheduling with synthetic arm bytes, not authorize real use.
    monkeypatch.setattr(freeze, "verify_measured", lambda arm: {"verified": True})
    membership = tmp_path / runner.FROZEN_MANIFEST
    membership.parent.mkdir(parents=True)
    raw = json.dumps({"frozen": True, "families": {"synthetic": {
        "split": "test", "members": [{"source_dataset": "fixture", "source_id": "one"}]
    }}}).encode()
    membership.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(runner, "FROZEN_MANIFEST_SHA256", digest)
    Path(str(membership) + ".sha256").write_text(digest)
    (membership.parent / "development_exposures.json").write_text('{"entries": []}')
    arm = tmp_path / "freeze.json"
    arm.write_text(json.dumps({"artifact_kind": "run", "not_tiny_aya": False,
        "model_id": "CohereLabs/tiny-aya-global", "producer": {"model_revision": "a" * 40}}))
    cfg = {"model_id": "CohereLabs/tiny-aya-global", "model_revision": "a" * 40,
        "splits": ["test"], "method_arms": ["base", "refuse_everything", "frozen_repair"],
        "seed": 42, "datasets": {"dataset_hash": "b" * 64,
        "split_manifest": str(runner.FROZEN_MANIFEST)},
        "frozen_repair": {"manifest": "freeze.json", "sha256": hashlib.sha256(arm.read_bytes()).hexdigest()}}
    config_path = tmp_path / "test_locked.yaml"

    def save():
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    save()
    return tmp_path, cfg, config_path, save


def execute(locked, callback=lambda *args: "generated", **kwargs):
    return runner._execute_locked_test(locked[2], run_id=kwargs.get("run_id", "synthetic-run"),
        i_understand=kwargs.get("i_understand", True), generate=callback)


def assert_refused(locked, match):
    called = []
    with pytest.raises(ValueError, match=match):
        execute(locked, lambda *args: called.append(True))
    assert called == []
    assert not (locked[0] / "runs" / runner.ATTEMPT_FILENAME).exists()


def test_reservation_is_synced_and_hash_bound_before_callback(locked, monkeypatch):
    synced = []
    original = runner.os.fsync
    monkeypatch.setattr(runner.os, "fsync", lambda fd: (original(fd), synced.append(fd)))

    def callback(cfg, arm, record):
        assert synced
        disk = json.loads((locked[0] / "runs" / runner.ATTEMPT_FILENAME).read_text())
        assert record == disk
        assert record["run_id"] == "synthetic-run"
        assert record["frozen_repair_sha256"] == cfg["frozen_repair"]["sha256"]
        assert record["config_sha256"] == hashlib.sha256(locked[2].read_bytes()).hexdigest()
        assert record["split_manifest_sha256"] == runner.FROZEN_MANIFEST_SHA256
        assert record["full_commit"] == "c" * 40
        assert record["model_revision"] == arm["producer"]["model_revision"]
        return "generated"

    assert execute(locked, callback) == "generated"
    with pytest.raises(ValueError, match="already reserved"):
        execute(locked, run_id="different-run")


@pytest.mark.parametrize("instant,blocked", [
    (datetime(2026, 11, 10, 18, 29, 59, tzinfo=timezone.utc), True),
    (datetime(2026, 11, 10, 18, 30, tzinfo=timezone.utc), False),
])
def test_exact_ist_embargo_boundary(locked, monkeypatch, instant, blocked):
    monkeypatch.setattr(runner, "_utc_now", lambda: instant)
    if blocked:
        assert_refused(locked, "embargo")
    else:
        assert execute(locked) == "generated"


@pytest.mark.parametrize("field,value,match", [
    ("model_revision", None, "model_revision"),
    ("splits", ["dev"], "test split"),
    ("method_arms", ["base"], "three frozen arms"),
    ("frozen_repair", None, "frozen_repair.manifest"),
    ("frozen_repair", {"manifest": "missing.json", "sha256": "d" * 64}, "verification failed"),
    ("frozen_repair", {"manifest": "freeze.json", "sha256": None}, "frozen_repair.sha256"),
    ("frozen_repair", {"manifest": "freeze.json", "sha256": "d" * 64}, "byte hash mismatch"),
])
def test_config_refusals(locked, field, value, match):
    locked[1][field] = value
    locked[3]()
    assert_refused(locked, match)


@pytest.mark.parametrize("field,value", [("artifact_kind", "fixture"), ("not_tiny_aya", True),
    ("producer", {"model_revision": "e" * 40}), ("model_id", "other")])
def test_repair_identity_refusals_even_with_matching_hash(locked, field, value):
    arm = locked[0] / "freeze.json"
    payload = json.loads(arm.read_text())
    payload[field] = value
    arm.write_text(json.dumps(payload))
    locked[1]["frozen_repair"]["sha256"] = hashlib.sha256(arm.read_bytes()).hexdigest()
    locked[3]()
    assert_refused(locked, "matching real-model arm")


def test_real_verifier_rejects_unverified_arm(locked, monkeypatch):
    monkeypatch.setattr(freeze, "verify_measured", REAL_VERIFY_MEASURED)
    assert_refused(locked, "hash does not match")


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_corrupt_repair_with_matching_byte_pin_refuses(locked, contents):
    arm = locked[0] / "freeze.json"
    arm.write_text(contents)
    locked[1]["frozen_repair"]["sha256"] = hashlib.sha256(arm.read_bytes()).hexdigest()
    locked[3]()
    assert_refused(locked, "verification failed")


@pytest.mark.parametrize("run_id", ["", "../other", "x" * 129, None])
def test_invalid_attempt_id_refuses(locked, run_id):
    with pytest.raises(ValueError, match="valid run_id"):
        execute(locked, run_id=run_id)
    assert not (locked[0] / "runs").exists()


def test_missing_source_commit_refuses(locked, monkeypatch):
    monkeypatch.setattr(runner, "_git_full_commit", lambda: None)
    assert_refused(locked, "full source commit")


def test_manifest_change_during_admission_refuses(locked, monkeypatch):
    def commit():
        (locked[0] / runner.FROZEN_MANIFEST).write_text("{}")
        return "c" * 40
    monkeypatch.setattr(runner, "_git_full_commit", commit)
    assert_refused(locked, "changed during admission")


def test_missing_acknowledgment_does_not_reserve(locked):
    with pytest.raises(ValueError, match="i-understand-single-shot"):
        execute(locked, i_understand=False)
    assert not (locked[0] / "runs").exists()


@pytest.mark.parametrize("field,value", [("dataset_hash", None), ("split_manifest", None),
    ("split_manifest", "missing.json")])
def test_dataset_refusals(locked, field, value):
    locked[1]["datasets"][field] = value
    locked[3]()
    assert_refused(locked, "test split")


@pytest.mark.parametrize("mutation,match", [
    ("manifest", "committed pin"), ("checksum", "checksum-verified"),
    ("exposure_missing", "exposure ledger is missing"),
    ("exposure_invalid", "unreadable or invalid"),
    ("exposure_member", "development-exposed"),
    ("manifest_alias", "canonical frozen"),
])
def test_frozen_membership_refusals(locked, mutation, match):
    membership = locked[0] / runner.FROZEN_MANIFEST
    exposure = membership.parent / "development_exposures.json"
    if mutation == "manifest":
        membership.write_text('{"frozen": true, "families": {"different": {}}}')
    elif mutation == "checksum":
        Path(str(membership) + ".sha256").write_text("wrong")
    elif mutation == "exposure_missing":
        exposure.unlink()
    elif mutation == "exposure_invalid":
        exposure.write_text("{")
    elif mutation == "exposure_member":
        exposure.write_text(json.dumps({"entries": [{"family_id": "renamed",
            "source_dataset": "fixture", "source_id": "one"}]}))
    elif mutation == "manifest_alias":
        alias = locked[0] / "alias.json"
        alias.write_bytes(membership.read_bytes())
        Path(str(alias) + ".sha256").write_text(runner.FROZEN_MANIFEST_SHA256)
        locked[1]["datasets"]["split_manifest"] = "alias.json"
        locked[3]()
    assert_refused(locked, match)


@pytest.mark.parametrize("history", ['{"artifact_kind":"run","split":"test"}', '{', '[]'])
def test_prior_or_unreadable_history_refuses(locked, history):
    path = locked[0] / "runs" / "prior" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(history)
    assert_refused(locked, "prior test run record")


@pytest.mark.parametrize("contents", ["", "{", '{"status":"reserved"}'])
def test_existing_partial_or_complete_reservation_refuses(locked, contents):
    path = locked[0] / "runs" / runner.ATTEMPT_FILENAME
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    with pytest.raises(ValueError, match="already reserved"):
        execute(locked)
    assert path.read_text() == contents


def test_failed_generation_consumes_attempt(locked):
    def failure(*args):
        raise RuntimeError("synthetic generation failure")
    with pytest.raises(RuntimeError, match="generation failure"):
        execute(locked, failure)
    with pytest.raises(ValueError, match="already reserved"):
        execute(locked)


def test_concurrent_starts_only_one_callback(locked, monkeypatch):
    barrier = threading.Barrier(2)
    def commit():
        barrier.wait(timeout=10)
        return "c" * 40
    monkeypatch.setattr(runner, "_git_full_commit", commit)
    called = []
    def attempt(i):
        try:
            return execute(locked, lambda *args: called.append(i), run_id=f"run-{i}")
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(attempt, [1, 2]))
    assert len(called) == 1
    assert sum("reservation failed" in str(value) for value in outcomes) == 1


def test_fsync_failure_consumes_attempt_without_generation(locked, monkeypatch):
    def failure(fd):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(runner.os, "fsync", failure)
    called = []
    with pytest.raises(ValueError, match="reservation failed"):
        execute(locked, lambda *args: called.append(True))
    assert called == []
    assert (locked[0] / "runs" / runner.ATTEMPT_FILENAME).exists()


def test_dry_check_and_stub_do_not_consume_attempt(locked, monkeypatch):
    monkeypatch.setattr(runner, "_dataset_missing", lambda cfg: [])
    judges = locked[0] / "judges.yaml"
    judges.write_text("{}")
    args = ["--config", str(locked[2]), "--judges-config", str(judges),
        "--split", "test", "--i-understand-single-shot"]
    assert runner.main(args + ["--dry-check"]) == 0
    assert runner.main(args) == 3
    assert not (locked[0] / "runs" / runner.ATTEMPT_FILENAME).exists()
