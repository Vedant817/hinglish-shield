"""Synthetic reviewed-variant exports. The real sheet test reads only the calibration/dev queue."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.export_reviewed_variants import main
from src.curate.reviewed_variants import (
    ACCEPTANCE_POLICY,
    CANONICAL_FAMILY_MANIFEST_SHA256,
    HEADER,
    ReviewedVariantError,
    export_reviewed_variants,
    sha256_bytes,
    verify_ledger_lineage,
)
from src.eval.dev_runner import (
    CANONICAL_FAMILY_MANIFEST,
    CANONICAL_FAMILY_MANIFEST_SHA256 as RUNNER_PIN,
    DevRunnerError,
    chat_template_digest,
    run_dev_ledger,
)

ROOT = Path(__file__).resolve().parents[2]
REAL_SHEET = ROOT / "data" / "staging" / "review" / "vedant_review_nov7_calibration_dev.csv"
LEAK = "TEST_PROMPT_LEAK_SENTINEL"
UNREVIEWED = "UNREVIEWED_HINGLISH_SENTINEL"
REVISE = "REVISE_CORRECTION_SENTINEL"
REJECTED = "REJECTED_HINGLISH_SENTINEL"
CANON_EN_HARM = "CANONICAL_EN_HARM_NOT_A_VARIANT"
CANON_EN_BENIGN = "CANONICAL_EN_BENIGN_NOT_A_VARIANT"
HI_HARM = "सिंथेटिक हिंदी हानि"
HING_HARM_DRAFT = "draft hinglish harm"
HING_HARM_FIX = "corrected hinglish harm"
HI_BENIGN = "सिंथेटिक हिंदी सुरक्षित"
HING_BENIGN = "synthetic hinglish benign"


def _member(source_id: str, item_id: str, lang: str, label: str) -> dict[str, str]:
    return {
        "hs_label": label,
        "item_id": item_id,
        "lang": lang,
        "source_dataset": "synthetic",
        "source_id": source_id,
    }


def _write_authority(path: Path, families: dict) -> None:
    path.write_bytes((json.dumps(
        {"families": families, "frozen": True, "version": "family_splits_v2"},
        indent=2,
        sort_keys=True,
    ) + "\n").encode("utf-8"))


def _row(**overrides: str) -> dict[str, str]:
    row = {field: "" for field in HEADER}
    row.update({
        "origin": "fixture",
        "origin_id": overrides.get("review_id", "rev"),
        "source": "synthetic",
        "lang_focus": "hi+hinglish",
        "category_or_harm": "ignored",
        "why_flagged": "fixture",
    })
    row.update(overrides)
    return row


def _write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HEADER), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _export(tmp: Path, families: dict, rows: list[dict[str, str]], **kwargs):
    manifest = tmp / "manifest.json"
    sheet = tmp / "sheet.csv"
    _write_authority(manifest, families)
    _write_sheet(sheet, rows)
    report = export_reviewed_variants(
        review_sheet=sheet,
        manifest=manifest,
        out_dir=tmp / "out",
        acceptance_policy=kwargs.get("acceptance_policy", ACCEPTANCE_POLICY),
        require_pinned_hash=False,
    )
    return sheet, manifest, report


def _mixed_dev_family() -> dict:
    return {
        "fam-mix": {
            "hs_label": "mixed",
            "members": [
                _member("harm-en", "item-harm-en", "en", "harm"),
                _member("benign-en", "item-benign-en", "en", "benign"),
            ],
            "split": "dev",
        }
    }


def _accept_rows() -> list[dict[str, str]]:
    return [
        _row(
            review_id="rev-harm",
            source_id="harm-en",
            family_id="fam-mix",
            split="dev",
            vedant_verdict="accept",
            english_or_literal=CANON_EN_HARM,
            hindi_or_prompt_hi=HI_HARM,
            hinglish=HING_HARM_DRAFT,
            corrected_hinglish=HING_HARM_FIX,
        ),
        _row(
            review_id="rev-benign",
            source_id="benign-en",
            family_id="fam-mix",
            split="dev",
            vedant_verdict="Accept",
            english_or_literal=CANON_EN_BENIGN,
            hindi_or_prompt_hi=HI_BENIGN,
            hinglish=HING_BENIGN,
        ),
    ]


def _blob(path: Path) -> str:
    parts = []
    for file in path.rglob("*"):
        if file.is_file():
            parts.append(file.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_pin_matches_dev_runner():
    assert CANONICAL_FAMILY_MANIFEST_SHA256 == RUNNER_PIN


def test_manifest_pin_is_checked_before_the_sheet(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"{}")
    missing = tmp_path / "absent.csv"
    with pytest.raises(ReviewedVariantError, match="hash"):
        export_reviewed_variants(
            review_sheet=missing,
            manifest=manifest,
            out_dir=tmp_path / "out",
            acceptance_policy=ACCEPTANCE_POLICY,
        )
    assert not missing.exists()
    assert not (tmp_path / "out").exists()


def test_approved_translations_keep_lineage_and_do_not_pad_counts(tmp_path: Path):
    sheet, _manifest, report = _export(tmp_path, _mixed_dev_family(), _accept_rows())
    assert report["status"] == "exported_incomplete"
    assert report["prompt_files_written"] is True
    assert report["corpus_complete"] is False
    assert report["sheet_coverage_complete"] is True
    assert report["eligible_variant_count"] == 4
    assert report["eligible_behavior_count"] == 2
    assert report["verdict_counts"] == {"accept": 2, "reject": 0, "revise": 0, "unreviewed": 0}
    coverage = json.loads((tmp_path / "out" / "coverage.json").read_text(encoding="utf-8"))
    family = coverage["families"]["fam-mix"]
    assert family["adr0007_matched"] is True
    assert family["missing_cells"] == []
    assert family["eligible_languages"] == ["hi", "hinglish"]
    assert family["canonical_member_counts"]["harm"]["en"] == 1
    assert family["eligible_reviewed_variant_counts"]["harm"]["en"] == 0
    assert family["eligible_reviewed_variant_counts"]["harm"]["hi"] == 1
    assert family["eligible_reviewed_variant_counts"]["harm"]["hinglish"] == 1
    assert family["eligible_source_items"] == 2
    manifest = json.loads((tmp_path / "out" / "dev" / "family.manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen"] is True
    assert manifest["review_status"] == "reviewed"
    assert manifest["families"]["fam-mix"]["languages"] == ["hi", "hinglish"]
    assert manifest["families"]["fam-mix"]["hs_label"] == "mixed"
    langs = [member["lang"] for member in manifest["families"]["fam-mix"]["members"]]
    assert langs == ["hi", "hi", "hinglish", "hinglish"]
    index = json.loads((tmp_path / "out" / "dev" / "index.json").read_text(encoding="utf-8"))
    assert {row["sample_id"] for row in index} == {
        "rev-harm:hi", "rev-harm:hinglish", "rev-benign:hi", "rev-benign:hinglish",
    }
    assert all(row["source_dataset"] == "reviewed_variant" for row in index)
    assert all(row["item_id"] != "item-harm-en" for row in index)
    prompts = (tmp_path / "out" / "dev" / "prompts.jsonl").read_text(encoding="utf-8")
    assert HI_HARM in prompts
    assert HING_HARM_FIX in prompts
    assert HING_HARM_DRAFT not in prompts
    assert HING_BENIGN in prompts
    assert CANON_EN_HARM not in prompts
    assert CANON_EN_BENIGN not in prompts
    lineage = json.loads((tmp_path / "out" / "dev" / "lineage.json").read_text(encoding="utf-8"))
    harm_hi = next(row for row in lineage["variants"] if row["variant_id"] == "rev-harm:hinglish")
    assert harm_hi["correction_applied"] is True
    assert harm_hi["lineage"]["canonical_item_id"] == "item-harm-en"
    assert harm_hi["lineage"]["canonical_lang"] == "en"
    assert harm_hi["lineage"]["canonical_hs_label"] == "harm"
    assert harm_hi["lineage"]["canonical_split"] == "dev"
    assert harm_hi["lineage"]["verdict"] == "accept"
    assert harm_hi["lang"] == "hinglish"
    assert "prompt" not in harm_hi
    _run_fixture_ledger(tmp_path, sheet)


def test_ineligible_verdicts_and_calibration_split(tmp_path: Path):
    families = _mixed_dev_family()
    families["fam-cal"] = {
        "hs_label": "benign",
        "members": [_member("cal-hi", "item-cal-hi", "hi", "benign")],
        "split": "calibration",
    }
    families["fam-other"] = {
        "hs_label": "harm",
        "members": [_member("other-en", "item-other-en", "en", "harm")],
        "split": "dev",
    }
    rows = _accept_rows() + [
        _row(
            review_id="rev-cal",
            source_id="cal-hi",
            family_id="fam-cal",
            split="calibration",
            vedant_verdict="accept",
            english_or_literal="synthetic calibration english",
            hindi_or_prompt_hi="draft hindi",
            corrected_hindi="draft hindi",
            hinglish="synthetic calibration hinglish",
        ),
        _row(
            review_id="rev-blank",
            source_id="other-en",
            family_id="fam-other",
            split="dev",
            hinglish=UNREVIEWED,
            vedant_verdict="",
        ),
        _row(
            review_id="rev-reject",
            source_id="harm-en",
            family_id="fam-mix",
            split="dev",
            hinglish=REJECTED,
            vedant_verdict="reject",
        ),
    ]
    # The reject row reuses a source that already has an accept. That is allowed:
    # duplicate review ids are refused, duplicate source reviews are not a second variant
    # unless both are accepted. Two accepts of one source would emit the same variant id
    # only if the review id matches. This reject uses a distinct review id.
    sheet, _manifest, report = _export(tmp_path, families, rows)
    assert report["eligible_variant_count_by_split"] == {"calibration": 2, "dev": 4}
    assert report["eligible_behavior_count"] == 3
    blob = _blob(tmp_path / "out")
    assert UNREVIEWED not in blob
    assert REJECTED not in blob
    cal_prompts = (tmp_path / "out" / "calibration" / "prompts.jsonl").read_text(encoding="utf-8")
    assert "synthetic calibration english" in cal_prompts
    assert "synthetic calibration hinglish" in cal_prompts
    assert "draft hindi" not in cal_prompts
    reasons = [
        json.loads(line)["reason"]
        for line in (tmp_path / "out" / "rejections.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert reasons.count("unreviewed") == 1
    assert reasons.count("rejected") == 1
    absent = tmp_path / "absent.jsonl"
    cal_index = tmp_path / "out" / "calibration" / "index.json"
    cal_config = {
        "chat_template_sha256": "c" * 64,
        "dataset_hash": "d" * 64,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 256,
            "system_prompt": None,
            "temperature": None,
            "top_p": None,
        },
        "dtype": "bfloat16",
        "family_manifest_sha256": "e" * 64,
        "index_sha256": sha256_bytes(cal_index.read_bytes()),
        "judge_model_id": "allenai/wildguard",
        "judge_revision": "b" * 40,
        "method": "base",
        "model_id": "example/synthetic-target",
        "model_revision": "a" * 40,
        "prompts_sha256": "f" * 64,
        "seed": 42,
    }
    cal_config_path = tmp_path / "cal-config.json"
    cal_config_path.write_text(json.dumps(cal_config), encoding="utf-8")
    with pytest.raises(DevRunnerError, match="non-dev"):
        run_dev_ledger(
            config_path=cal_config_path,
            index_path=cal_index,
            prompts_path=absent,
            family_manifest_path=tmp_path / "out" / "calibration" / "family.manifest.json",
            out_dir=tmp_path / "cal-ledger",
            split="dev",
            generator_factory=lambda _config: None,
            judge_factory=lambda _config: None,
        )
    assert not absent.exists()
    assert sheet.exists()


def test_revise_stays_ineligible_and_real_export_is_blocked(tmp_path: Path):
    families = {
        "fam-dev": {
            "hs_label": "harm",
            "members": [_member("only-en", "item-only", "en", "harm")],
            "split": "dev",
        }
    }
    _sheet, _manifest, report = _export(tmp_path, families, [
        _row(
            review_id="rev-revise",
            source_id="only-en",
            family_id="fam-dev",
            split="dev",
            vedant_verdict="revise",
            hinglish="draft",
            corrected_hinglish=REVISE,
        )
    ])
    assert report["status"] == "blocked_no_explicit_accept"
    assert report["prompt_files_written"] is False
    assert report["eligible_variant_count"] == 0
    assert not (tmp_path / "out" / "dev").exists()
    blob = _blob(tmp_path / "out")
    assert REVISE not in blob
    rejection = json.loads((tmp_path / "out" / "rejections.jsonl").read_text(encoding="utf-8"))
    assert rejection["reason"] == "revise_ineligible"
    assert rejection["correction_present"] is True
    assert "prompt" not in rejection


def test_protocol_failures_write_nothing(tmp_path: Path):
    families = {
        "fam-dev": {
            "hs_label": "harm",
            "members": [_member("dev-en", "item-dev", "en", "harm")],
            "split": "dev",
        },
        "fam-test": {
            "hs_label": "harm",
            "members": [_member("test-en", "item-test", "en", "harm")],
            "split": "test",
        },
    }
    cases = [
        _row(
            review_id="rev-test", source_id="test-en", family_id="fam-test", split="test",
            hinglish=LEAK, vedant_verdict="accept",
        ),
        _row(
            review_id="rev-launder", source_id="test-en", family_id="fam-dev", split="dev",
            hinglish=LEAK, vedant_verdict="accept",
        ),
        _row(
            review_id="rev-split", source_id="test-en", family_id="fam-test", split="dev",
            hinglish=LEAK, vedant_verdict="accept",
        ),
    ]
    for index, row in enumerate(cases):
        pocket = tmp_path / f"case-{index}"
        pocket.mkdir()
        with pytest.raises(ReviewedVariantError, match="test split|membership") as caught:
            _export(pocket, families, [row])
        assert LEAK not in str(caught.value)
        assert not (pocket / "out").exists()

    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    with pytest.raises(ReviewedVariantError, match="duplicate review id"):
        _export(duplicate, {"fam-dev": families["fam-dev"]}, [
            _row(review_id="rev-same", source_id="dev-en", family_id="fam-dev", split="dev"),
            _row(review_id="rev-same", source_id="dev-en", family_id="fam-dev", split="dev"),
        ])
    assert not (duplicate / "out").exists()

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    with pytest.raises(ReviewedVariantError, match="unknown verdict"):
        _export(unknown, {"fam-dev": families["fam-dev"]}, [
            _row(
                review_id="rev-odd", source_id="dev-en", family_id="fam-dev", split="dev",
                vedant_verdict="approved", hinglish=LEAK,
            ),
        ])
    assert not (unknown / "out").exists()


@pytest.mark.parametrize("field", ["canonical_source_ids_sha256", "review_sheet_sha256"])
def test_lineage_metadata_digests_bind_actual_verified_content(tmp_path, field):
    from src.curate.reviewed_variants import load_authority, evidence_sha256
    sheet, manifest, _ = _export(tmp_path, _mixed_dev_family(), _accept_rows())
    split = tmp_path / "out/dev"
    pins = json.loads((split / "checksums.json").read_text())
    lineage = json.loads((split / "lineage.json").read_text())
    lineage[field] = "0" * 64
    if field == "canonical_source_ids_sha256":
        pins[field] = lineage[field]
        pins["reviewed_evidence_sha256"] = evidence_sha256(
            pins["review_sheet_sha256"], pins[field], pins["prompts_sha256"],
        )
        lineage["evidence_sha256"] = pins["reviewed_evidence_sha256"]
    raw = (json.dumps(lineage, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    pins["reviewed_lineage_sha256"] = sha256_bytes(raw)
    with pytest.raises(ReviewedVariantError, match="reviewed lineage mismatch"):
        verify_ledger_lineage(
            lineage_bytes=raw, review_sheet_bytes=sheet.read_bytes(),
            index_rows=json.loads((split / "index.json").read_text()),
            authority=load_authority(manifest, require_pinned_hash=False), pins=pins,
        )


def test_forged_lineage_unreviewed_and_correction_drift(tmp_path: Path):
    sheet, manifest, _report = _export(tmp_path, _mixed_dev_family(), _accept_rows())
    authority_bytes = manifest.read_bytes()
    from src.curate.reviewed_variants import load_authority

    authority = load_authority(manifest, require_pinned_hash=False)
    checksums = json.loads((tmp_path / "out" / "dev" / "checksums.json").read_text(encoding="utf-8"))
    lineage_path = tmp_path / "out" / "dev" / "lineage.json"
    index = json.loads((tmp_path / "out" / "dev" / "index.json").read_text(encoding="utf-8"))
    pins = {
        "canonical_source_ids_sha256": checksums["canonical_source_ids_sha256"],
        "prompts_sha256": checksums["prompts_sha256"],
        "review_sheet_sha256": checksums["review_sheet_sha256"],
        "reviewed_evidence_sha256": checksums["reviewed_evidence_sha256"],
        "reviewed_lineage_sha256": checksums["reviewed_lineage_sha256"],
    }
    verified = verify_ledger_lineage(
        lineage_bytes=lineage_path.read_bytes(),
        review_sheet_bytes=sheet.read_bytes(),
        index_rows=index,
        authority=authority,
        pins=pins,
    )
    assert verified["reviewed_lineage_sha256"] == pins["reviewed_lineage_sha256"]

    forged = json.loads(lineage_path.read_text(encoding="utf-8"))
    forged["variants"][0]["lineage"]["canonical_item_id"] = "forged-item"
    forged_bytes = (json.dumps(forged, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    forged_pins = dict(pins)
    forged_pins["reviewed_lineage_sha256"] = sha256_bytes(forged_bytes)
    with pytest.raises(ReviewedVariantError, match="forged lineage"):
        verify_ledger_lineage(
            lineage_bytes=forged_bytes,
            review_sheet_bytes=sheet.read_bytes(),
            index_rows=index,
            authority=authority,
            pins=forged_pins,
        )

    drifted = json.loads(lineage_path.read_text(encoding="utf-8"))
    target = next(row for row in drifted["variants"] if row["variant_id"] == "rev-harm:hinglish")
    target["prompt_hash"] = target["draft_prompt_sha256"]
    target["correction_applied"] = False
    drifted_bytes = (json.dumps(drifted, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    drift_pins = dict(pins)
    drift_pins["reviewed_lineage_sha256"] = sha256_bytes(drifted_bytes)
    with pytest.raises(ReviewedVariantError, match="correction drift"):
        verify_ledger_lineage(
            lineage_bytes=drifted_bytes,
            review_sheet_bytes=sheet.read_bytes(),
            index_rows=index,
            authority=authority,
            pins=drift_pins,
        )

    duplicated = json.loads(lineage_path.read_text(encoding="utf-8"))
    duplicated["variants"].append(dict(duplicated["variants"][0]))
    duplicated_bytes = (json.dumps(duplicated, indent=2, sort_keys=True) + "\n").encode("utf-8")
    dup_pins = dict(pins)
    dup_pins["reviewed_lineage_sha256"] = sha256_bytes(duplicated_bytes)
    with pytest.raises(ReviewedVariantError, match="duplicate variant id"):
        verify_ledger_lineage(
            lineage_bytes=duplicated_bytes,
            review_sheet_bytes=sheet.read_bytes(),
            index_rows=index,
            authority=authority,
            pins=dup_pins,
        )

    blank_sheet = tmp_path / "blank.csv"
    _write_sheet(blank_sheet, [
        _row(
            review_id="rev-harm", source_id="harm-en", family_id="fam-mix", split="dev",
            vedant_verdict="", hinglish=UNREVIEWED,
        )
    ])
    with pytest.raises(ReviewedVariantError, match="unreviewed variant"):
        verify_ledger_lineage(
            lineage_bytes=lineage_path.read_bytes(),
            review_sheet_bytes=blank_sheet.read_bytes(),
            index_rows=index,
            authority=authority,
            pins={**pins, "review_sheet_sha256": sha256_bytes(blank_sheet.read_bytes())},
        )
    assert authority_bytes == manifest.read_bytes()


def test_real_calibration_dev_sheet_is_blocked(tmp_path: Path):
    out = tmp_path / "real"
    code = main([
        "--acceptance-policy", ACCEPTANCE_POLICY,
        "--review-sheet", str(REAL_SHEET),
        "--manifest", str(CANONICAL_FAMILY_MANIFEST),
        "--out", str(out),
    ])
    assert code == 2
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked_no_explicit_accept"
    assert report["row_count"] == 26
    assert report["verdict_counts"]["unreviewed"] == 26
    assert report["verdict_counts"]["accept"] == 0
    assert report["eligible_variant_count"] == 0
    assert report["eligible_behavior_count"] == 0
    assert report["prompt_files_written"] is False
    assert report["sheet_coverage_complete"] is False
    assert report["corpus_complete"] is False
    assert report["canonical_family_splits_sha256"] == CANONICAL_FAMILY_MANIFEST_SHA256
    coverage = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    assert coverage["adr0007_matched_family_count"] == 0
    assert coverage["families_in_sheet"] == 26
    for family in coverage["families"].values():
        counts = family["eligible_reviewed_variant_counts"]
        assert counts["harm"] == {"en": 0, "hi": 0, "hinglish": 0}
        assert counts["benign"] == {"en": 0, "hi": 0, "hinglish": 0}
        assert family["adr0007_matched"] is False
    assert not (out / "dev").exists()
    assert not (out / "calibration").exists()
    raw = _blob(out)
    assert "english_or_literal" not in raw
    assert "corrected_hinglish" not in raw
    for line in (out / "rejections.jsonl").read_text(encoding="utf-8").splitlines():
        rejection = json.loads(line)
        assert "prompt" not in rejection
        assert rejection["reason"] == "unreviewed"
    wrong = main(["--acceptance-policy", "invent-accept", "--out", str(tmp_path / "nope")])
    assert wrong == 2
    assert not (tmp_path / "nope").exists()


def _run_fixture_ledger(tmp: Path, sheet: Path) -> None:
    checksums = json.loads((tmp / "out" / "dev" / "checksums.json").read_text(encoding="utf-8"))
    config_path = tmp / "frozen.dev.json"
    config = {
        "canonical_source_ids_sha256": checksums["canonical_source_ids_sha256"],
        "chat_template_sha256": chat_template_digest("synthetic-chat-template-no-system"),
        "dataset_hash": checksums["dataset_hash"],
        "decoding": {
            "do_sample": False,
            "max_new_tokens": 256,
            "system_prompt": None,
            "temperature": None,
            "top_p": None,
        },
        "dtype": "bfloat16",
        "family_manifest_sha256": checksums["family_manifest_sha256"],
        "index_sha256": checksums["index_sha256"],
        "judge_model_id": "allenai/wildguard",
        "judge_revision": "b" * 40,
        "method": "base",
        "model_id": "example/synthetic-target",
        "model_revision": "a" * 40,
        "prompts_sha256": checksums["prompts_sha256"],
        "review_sheet_sha256": checksums["review_sheet_sha256"],
        "reviewed_evidence_sha256": checksums["reviewed_evidence_sha256"],
        "reviewed_lineage_sha256": checksums["reviewed_lineage_sha256"],
        "seed": 42,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    steered = dict(config)
    steered["method"] = "steer"
    steered_path = tmp / "steered.json"
    steered_path.write_text(json.dumps(steered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    absent = tmp / "absent.jsonl"
    with pytest.raises(DevRunnerError, match="method"):
        run_dev_ledger(
            config_path=steered_path,
            index_path=tmp / "out" / "dev" / "index.json",
            prompts_path=absent,
            family_manifest_path=tmp / "out" / "dev" / "family.manifest.json",
            out_dir=tmp / "steered-out",
            split="dev",
            generator_factory=lambda _config: None,
            judge_factory=lambda _config: None,
            canonical_manifest_path=tmp / "manifest.json",
            reviewed_lineage_path=tmp / "out" / "dev" / "lineage.json",
            review_sheet_path=sheet,
        )
    assert not absent.exists()

    class Generator:
        def __init__(self):
            self.unloaded = False

        def generate_one(self, row):
            return {
                "eos": True,
                "input_token_count": 2,
                "new_token_count": 1,
                "response": "fixture-reply",
                "sample_id": row["sample_id"],
                "status": "ok",
                "truncated": False,
            }

        def unload(self):
            self.unloaded = True

    class Judge:
        def judge_one(self, prompt, response, **kwargs):
            return SimpleNamespace(
                prompt_id=kwargs["prompt_id"],
                value=0.0,
                label="safe",
                metadata={
                    "harmful_request": "yes",
                    "harmful_response": "no",
                    "response_refusal": "yes",
                },
            )

        def unload(self):
            return None

    manifest = run_dev_ledger(
        config_path=config_path,
        index_path=tmp / "out" / "dev" / "index.json",
        prompts_path=tmp / "out" / "dev" / "prompts.jsonl",
        family_manifest_path=tmp / "out" / "dev" / "family.manifest.json",
        out_dir=tmp / "ledger",
        split="dev",
        generator_factory=lambda _config: Generator(),
        judge_factory=lambda _config: Judge(),
        canonical_manifest_path=tmp / "manifest.json",
        reviewed_lineage_path=tmp / "out" / "dev" / "lineage.json",
        review_sheet_path=sheet,
    )
    assert manifest["artifact_kind"] == "fixture"
    assert manifest["split"] == "example"
    assert manifest["metrics_eligible"] is False
    assert manifest["primary_asr"] is None
    assert manifest["status"] == "complete"
    assert manifest["provenance"]["method"] == "base"
    assert manifest["provenance"]["reviewed_lineage_sha256"] == checksums["reviewed_lineage_sha256"]
    assert manifest["provenance"]["canonical_family_splits_sha256"] == sha256_bytes(
        (tmp / "manifest.json").read_bytes()
    )
    generations = (tmp / "ledger" / "generations.jsonl").read_text(encoding="utf-8")
    assert HI_HARM not in generations
    assert "fixture-reply" in generations
    assert not (tmp / "ledger" / "metrics.json").exists()
