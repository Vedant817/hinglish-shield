"""Reviewed translation variants with verified lineage back to canonical members.

Eligible prompts require an explicit human accept under
``explicit-human-accept-v1``. Blank, revise, and reject rows stay ineligible.
Corrected text is copied only from the labeled correction cells. This module
does not invent translations, labels, or coverage.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.curate.family_splits import canon

ACCEPTANCE_POLICY = "explicit-human-accept-v1"
REVIEWED_VARIANT_SOURCE = "reviewed_variant"
CANONICAL_FAMILY_MANIFEST_SHA256 = (
    "334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531"
)
HEADER = (
    "review_id",
    "origin",
    "origin_id",
    "source",
    "source_id",
    "family_id",
    "split",
    "lang_focus",
    "english_or_literal",
    "hindi_or_prompt_hi",
    "hinglish",
    "category_or_harm",
    "why_flagged",
    "vedant_verdict",
    "corrected_hindi",
    "corrected_hinglish",
    "vedant_notes",
)
IDENTITY_FIELDS = (
    "review_id",
    "origin",
    "origin_id",
    "source",
    "source_id",
    "family_id",
    "split",
    "vedant_verdict",
)
LANGS = ("en", "hi", "hinglish")
LABELS = ("harm", "benign")
FAMILY_LABELS = frozenset({"harm", "benign", "mixed"})
ADMITTED_SPLITS = frozenset({"calibration", "dev"})
CANONICAL_SPLITS = frozenset({"calibration", "dev", "test"})
DRAFT_FIELDS = {
    "en": "english_or_literal",
    "hi": "hindi_or_prompt_hi",
    "hinglish": "hinglish",
}
CORRECTION_FIELDS = {"hi": "corrected_hindi", "hinglish": "corrected_hinglish"}
PROMPT_KEYS = frozenset({
    "prompt",
    "text",
    "response",
    "completion",
    "english",
    "hindi",
    "hinglish",
    "english_or_literal",
    "hindi_or_prompt_hi",
    "corrected_hindi",
    "corrected_hinglish",
})
INDEX_FIELDS = (
    "sample_id",
    "item_id",
    "family_id",
    "source_dataset",
    "source_id",
    "lang",
    "hs_label",
    "split",
    "prompt_hash",
    "review_status",
)
LINEAGE_FIELDS = (
    "canonical_family_id",
    "canonical_hs_label",
    "canonical_item_id",
    "canonical_lang",
    "canonical_source_dataset",
    "canonical_source_id",
    "canonical_split",
    "origin",
    "origin_id",
    "review_id",
    "verdict",
)


class ReviewedVariantError(Exception):
    """Fail-closed export error. Messages must not contain prompt text."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_dataset_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    """Match the dev ledger dataset bytes: sorted sample_id, three fields, trailing newline."""
    lines: list[str] = []
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        payload = {
            "item_id": row["item_id"],
            "prompt": row["prompt"],
            "sample_id": row["sample_id"],
        }
        lines.append(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return ("\n".join(lines) + "\n").encode("utf-8")


def evidence_sha256(review_sheet_sha: str, source_ids_sha: str, prompts_sha: str) -> str:
    material = f"{review_sheet_sha}\n{source_ids_sha}\n{prompts_sha}\n".encode("utf-8")
    return sha256_bytes(material)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "" or value.strip() != value:
        raise ReviewedVariantError(f"missing {label}")
    if any(char in value for char in "\r\n\t "):
        raise ReviewedVariantError(f"missing {label}")
    return value


def _verdict(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReviewedVariantError("unknown verdict")
    text = value.strip().lower()
    if text not in {"", "accept", "revise", "reject"}:
        raise ReviewedVariantError("unknown verdict")
    return text


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReviewedVariantError("invalid review cell")
    return value.strip()


def _reject_prompt_keys(payload: Any) -> None:
    if isinstance(payload, dict):
        if PROMPT_KEYS & set(payload):
            raise ReviewedVariantError("lineage must stay metadata-only")
        for value in payload.values():
            _reject_prompt_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            _reject_prompt_keys(value)


def load_authority(path: Path, *, require_pinned_hash: bool = True) -> dict[str, Any]:
    """Read family-split metadata. Prompt-like member keys are refused."""
    if not path.is_file():
        raise ReviewedVariantError("missing canonical family manifest")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if require_pinned_hash and digest != CANONICAL_FAMILY_MANIFEST_SHA256:
        raise ReviewedVariantError("canonical family manifest hash mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ReviewedVariantError("canonical family manifest is unreadable") from None
    if not isinstance(payload, dict):
        raise ReviewedVariantError("canonical family manifest is unreadable")
    if payload.get("version") != "family_splits_v2":
        raise ReviewedVariantError("canonical family manifest version mismatch")
    if payload.get("frozen") is not True:
        raise ReviewedVariantError("canonical family manifest is not frozen")
    families_in = payload.get("families")
    if not isinstance(families_in, dict) or not families_in:
        raise ReviewedVariantError("canonical family manifest is missing families")
    by_source: dict[tuple[str, str], dict[str, str]] = {}
    families: dict[str, dict[str, Any]] = {}
    for family_id, family in families_in.items():
        if not isinstance(family_id, str) or family_id.strip() == "" or not isinstance(family, dict):
            raise ReviewedVariantError("canonical family manifest is missing families")
        split = family.get("split")
        if split not in CANONICAL_SPLITS:
            raise ReviewedVariantError("canonical family split is unsupported")
        family_label = family.get("hs_label")
        if family_label not in FAMILY_LABELS:
            raise ReviewedVariantError("canonical family label is not authoritative")
        members_in = family.get("members")
        if not isinstance(members_in, list) or not members_in:
            raise ReviewedVariantError("canonical family manifest is missing families")
        members: list[dict[str, str]] = []
        for member in members_in:
            if not isinstance(member, dict):
                raise ReviewedVariantError("canonical family manifest is missing families")
            if PROMPT_KEYS & set(member):
                raise ReviewedVariantError("canonical family manifest is metadata-only")
            source_dataset = _token(member.get("source_dataset"), "canonical source_dataset")
            source_id = _token(member.get("source_id"), "canonical source_id")
            item_id = _token(member.get("item_id"), "canonical item_id")
            lang = _token(member.get("lang"), "canonical lang")
            hs_label = _token(member.get("hs_label"), "canonical hs_label")
            if lang not in LANGS or hs_label not in LABELS:
                raise ReviewedVariantError("canonical label is not authoritative")
            key = (source_dataset, source_id)
            if key in by_source:
                raise ReviewedVariantError("duplicate canonical member")
            record = {
                "family_hs_label": family_label,
                "family_id": family_id,
                "hs_label": hs_label,
                "item_id": item_id,
                "lang": lang,
                "source_dataset": source_dataset,
                "source_id": source_id,
                "split": split,
            }
            by_source[key] = record
            members.append({
                "hs_label": hs_label,
                "item_id": item_id,
                "lang": lang,
                "source_dataset": source_dataset,
                "source_id": source_id,
            })
        families[family_id] = {
            "hs_label": family_label,
            "members": members,
            "split": split,
        }
    if not by_source:
        raise ReviewedVariantError("canonical family manifest is missing families")
    return {"by_source": by_source, "families": families, "sha256": digest}


def parse_sheet(raw: bytes) -> list[dict[str, Any]]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ReviewedVariantError("review sheet must be UTF-8 without BOM")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ReviewedVariantError("review sheet is unreadable") from None
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if list(reader.fieldnames or []) != list(HEADER):
        raise ReviewedVariantError("review header mismatch")
    rows: list[dict[str, Any]] = []
    for row in reader:
        if row.get(None):
            raise ReviewedVariantError("review header mismatch")
        rows.append(dict(row))
    if not rows:
        raise ReviewedVariantError("missing review rows")
    return rows


def bind_identities(
    identities: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve source, family, and split before any language cell is read."""
    seen: set[str] = set()
    bound: list[dict[str, Any]] = []
    by_source = authority["by_source"]
    for row in identities:
        review_id = _token(row.get("review_id"), "review_id")
        if ":" in review_id:
            raise ReviewedVariantError("missing review_id")
        if review_id in seen:
            raise ReviewedVariantError("duplicate review id")
        seen.add(review_id)
        origin = _token(row.get("origin"), "origin")
        origin_id = _token(row.get("origin_id"), "origin_id")
        source, source_id = canon(str(row.get("source") or ""), str(row.get("source_id") or ""))
        family_id = _token(row.get("family_id"), "family_id")
        split = _token(row.get("split"), "split")
        verdict = _verdict(row.get("vedant_verdict"))
        found = by_source.get((source, source_id))
        if not isinstance(found, dict):
            raise ReviewedVariantError("canonical membership mismatch")
        if found["family_id"] != family_id or found["split"] != split:
            raise ReviewedVariantError("canonical membership mismatch")
        if split == "test" or found["split"] == "test":
            raise ReviewedVariantError("test split is refused")
        if found["split"] not in ADMITTED_SPLITS:
            raise ReviewedVariantError("canonical membership mismatch")
        if found["family_hs_label"] in LABELS and found["family_hs_label"] != found["hs_label"]:
            if found["family_hs_label"] != "mixed":
                raise ReviewedVariantError("canonical label mismatch")
        bound.append({
            "member": found,
            "origin": origin,
            "origin_id": origin_id,
            "review_id": review_id,
            "verdict": verdict,
        })
    return bound


def _variant_record(
    info: Mapping[str, Any],
    *,
    lang: str,
    text: str,
    draft: str,
    correction_applied: bool,
) -> dict[str, Any]:
    member = info["member"]
    review_id = info["review_id"]
    variant_id = f"{review_id}:{lang}"
    draft_hash = sha256_text(draft) if draft else None
    return {
        "correction_applied": correction_applied,
        "draft_prompt_sha256": draft_hash,
        "family_id": member["family_id"],
        "hs_label": member["hs_label"],
        "item_id": f"reviewed:{variant_id}",
        "lang": lang,
        "lineage": {
            "canonical_family_id": member["family_id"],
            "canonical_hs_label": member["hs_label"],
            "canonical_item_id": member["item_id"],
            "canonical_lang": member["lang"],
            "canonical_source_dataset": member["source_dataset"],
            "canonical_source_id": member["source_id"],
            "canonical_split": member["split"],
            "origin": info["origin"],
            "origin_id": info["origin_id"],
            "review_id": review_id,
            "verdict": "accept",
        },
        "prompt": text,
        "prompt_hash": sha256_text(text),
        "review_status": "reviewed",
        "sample_id": variant_id,
        "source_dataset": REVIEWED_VARIANT_SOURCE,
        "source_id": variant_id,
        "split": member["split"],
        "variant_id": variant_id,
    }


def build_variants(
    rows: Sequence[Mapping[str, Any]],
    bound: Sequence[Mapping[str, Any]],
    acceptance_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if acceptance_policy != ACCEPTANCE_POLICY:
        raise ReviewedVariantError("acceptance policy is required")
    if len(rows) != len(bound):
        raise ReviewedVariantError("missing review rows")
    variants: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row, info in zip(rows, bound):
        member = info["member"]
        base = {
            "canonical_hs_label": member["hs_label"],
            "canonical_lang": member["lang"],
            "eligible": False,
            "family_id": member["family_id"],
            "review_id": info["review_id"],
            "split": member["split"],
        }
        verdict = info["verdict"]
        if verdict == "":
            rejections.append({**base, "reason": "unreviewed"})
            continue
        if verdict == "reject":
            rejections.append({**base, "reason": "rejected"})
            continue
        if verdict == "revise":
            correction_present = any(_cell(row.get(field)) for field in CORRECTION_FIELDS.values())
            rejections.append({
                **base,
                "correction_present": correction_present,
                "reason": "revise_ineligible",
            })
            continue
        emitted = False
        for lang in LANGS:
            draft = _cell(row.get(DRAFT_FIELDS[lang]))
            correction = _cell(row.get(CORRECTION_FIELDS[lang])) if lang in CORRECTION_FIELDS else ""
            if correction and correction != draft:
                text = correction
                applied = True
            else:
                text = draft
                applied = False
            if lang == member["lang"] and not applied:
                continue
            if text == "":
                rejections.append({**base, "lang": lang, "reason": "missing_text"})
                continue
            variant = _variant_record(
                info, lang=lang, text=text, draft=draft, correction_applied=applied,
            )
            if variant["variant_id"] in seen_ids:
                raise ReviewedVariantError("duplicate variant id")
            seen_ids.add(variant["variant_id"])
            variants.append(variant)
            emitted = True
        if not emitted:
            rejections.append({**base, "reason": "accept_without_variant"})
    return variants, rejections


def _empty_grid() -> dict[str, dict[str, int]]:
    return {label: {lang: 0 for lang in LANGS} for label in LABELS}


def _union(left: Mapping[str, Mapping[str, int]], right: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, bool]]:
    return {
        label: {lang: (left[label][lang] + right[label][lang]) > 0 for lang in LANGS}
        for label in LABELS
    }


def coverage_report(
    authority: Mapping[str, Any],
    bound: Sequence[Mapping[str, Any]],
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for variant in variants:
        grouped[variant["family_id"]].append(variant)
    family_ids = sorted({info["member"]["family_id"] for info in bound})
    families: dict[str, Any] = {}
    matched = 0
    for family_id in family_ids:
        meta = authority["families"][family_id]
        canonical = _empty_grid()
        for member in meta["members"]:
            canonical[member["hs_label"]][member["lang"]] += 1
        eligible = _empty_grid()
        group = grouped.get(family_id, [])
        for variant in group:
            eligible[variant["hs_label"]][variant["lang"]] += 1
        present = _union(canonical, eligible)
        missing = [
            f"{label}.{lang}"
            for label in LABELS
            for lang in LANGS
            if not present[label][lang]
        ]
        is_matched = not missing
        matched += int(is_matched)
        languages = [lang for lang in LANGS if any(variant["lang"] == lang for variant in group)]
        families[family_id] = {
            "adr0007_matched": is_matched,
            "canonical_family_label": meta["hs_label"],
            "canonical_member_counts": canonical,
            "canonical_split": meta["split"],
            "eligible_languages": languages,
            "eligible_reviewed_variant_counts": eligible,
            "eligible_source_items": len({
                (
                    variant["lineage"]["canonical_source_dataset"],
                    variant["lineage"]["canonical_source_id"],
                )
                for variant in group
            }),
            "missing_cells": missing,
            "union_present": present,
        }
    return {
        "adr": "0007",
        "adr0007_matched_family_count": matched,
        "families": families,
        "families_in_sheet": len(family_ids),
        "schema": "reviewed_coverage_v1",
        "sheet_coverage_complete": len(family_ids) > 0 and matched == len(family_ids),
        "unit": "semantic_family",
    }


def _source_id_bytes(variants: Sequence[Mapping[str, Any]]) -> bytes:
    lines: list[str] = []
    ordered = sorted(
        variants,
        key=lambda variant: (
            variant["lineage"]["canonical_source_dataset"],
            variant["lineage"]["canonical_source_id"],
            variant["variant_id"],
        ),
    )
    for variant in ordered:
        lineage = variant["lineage"]
        lines.append("|".join([
            lineage["canonical_source_dataset"],
            lineage["canonical_source_id"],
            lineage["canonical_item_id"],
            lineage["canonical_family_id"],
            lineage["canonical_split"],
            lineage["canonical_lang"],
            lineage["canonical_hs_label"],
            variant["variant_id"],
        ]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _public_variant(variant: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "correction_applied": variant["correction_applied"],
        "draft_prompt_sha256": variant["draft_prompt_sha256"],
        "family_id": variant["family_id"],
        "hs_label": variant["hs_label"],
        "item_id": variant["item_id"],
        "lang": variant["lang"],
        "lineage": dict(variant["lineage"]),
        "prompt_hash": variant["prompt_hash"],
        "review_status": variant["review_status"],
        "sample_id": variant["sample_id"],
        "source_dataset": variant["source_dataset"],
        "source_id": variant["source_id"],
        "split": variant["split"],
        "variant_id": variant["variant_id"],
    }


def _index_row(variant: Mapping[str, Any]) -> dict[str, str]:
    return {field: variant[field] for field in INDEX_FIELDS}


def _family_manifest(variants: Sequence[Mapping[str, Any]], *, authority_sha: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for variant in variants:
        grouped[variant["family_id"]].append(variant)
    families: dict[str, Any] = {}
    for family_id, group in sorted(grouped.items()):
        labels = {variant["hs_label"] for variant in group}
        family_label = next(iter(labels)) if len(labels) == 1 else "mixed"
        members = []
        for variant in sorted(group, key=lambda item: (LANGS.index(item["lang"]), item["variant_id"])):
            members.append({
                "hs_label": variant["hs_label"],
                "item_id": variant["item_id"],
                "lang": variant["lang"],
                "source_dataset": variant["source_dataset"],
                "source_id": variant["source_id"],
            })
        families[family_id] = {
            "hs_label": family_label,
            "languages": [lang for lang in LANGS if any(variant["lang"] == lang for variant in group)],
            "members": members,
            "split": group[0]["split"],
        }
    return {
        "acceptance_policy": ACCEPTANCE_POLICY,
        "canonical_family_splits_sha256": authority_sha,
        "families": families,
        "frozen": True,
        "review_status": "reviewed",
        "schema": "reviewed_family_manifest_v1",
    }


def _lineage_payload(
    variants: Sequence[Mapping[str, Any]],
    *,
    authority_sha: str,
    sheet_sha: str,
    source_sha: str,
    prompts_sha: str,
    evidence: str,
    split: str,
) -> dict[str, Any]:
    return {
        "acceptance_policy": ACCEPTANCE_POLICY,
        "canonical_family_splits_sha256": authority_sha,
        "canonical_source_ids_sha256": source_sha,
        "evidence_sha256": evidence,
        "frozen": True,
        "prompts_sha256": prompts_sha,
        "review_sheet_sha256": sheet_sha,
        "schema": "reviewed_lineage_v1",
        "split": split,
        "variants": [_public_variant(variant) for variant in sorted(variants, key=lambda item: item["variant_id"])],
    }


def _write_exclusive(path: Path, raw: bytes) -> None:
    if path.exists():
        raise ReviewedVariantError("output directory already exists")
    path.write_bytes(raw)


def _write_split(
    directory: Path,
    variants: Sequence[Mapping[str, Any]],
    *,
    authority_sha: str,
    sheet_sha: str,
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=False)
    split = variants[0]["split"]
    prompt_rows = [
        {"item_id": variant["item_id"], "prompt": variant["prompt"], "sample_id": variant["sample_id"]}
        for variant in variants
    ]
    prompt_raw = canonical_dataset_bytes(prompt_rows)
    source_raw = _source_id_bytes(variants)
    source_sha = sha256_bytes(source_raw)
    prompts_sha = sha256_bytes(prompt_raw)
    evidence = evidence_sha256(sheet_sha, source_sha, prompts_sha)
    index_raw = _json_bytes([_index_row(variant) for variant in sorted(variants, key=lambda item: item["sample_id"])])
    manifest_raw = _json_bytes(_family_manifest(variants, authority_sha=authority_sha))
    lineage_raw = _json_bytes(_lineage_payload(
        variants,
        authority_sha=authority_sha,
        sheet_sha=sheet_sha,
        source_sha=source_sha,
        prompts_sha=prompts_sha,
        evidence=evidence,
        split=split,
    ))
    checksums = {
        "canonical_family_splits_sha256": authority_sha,
        "canonical_source_ids_sha256": source_sha,
        "dataset_hash": prompts_sha,
        "family_manifest_sha256": sha256_bytes(manifest_raw),
        "index_sha256": sha256_bytes(index_raw),
        "prompts_sha256": prompts_sha,
        "review_sheet_sha256": sheet_sha,
        "reviewed_evidence_sha256": evidence,
        "reviewed_lineage_sha256": sha256_bytes(lineage_raw),
        "split": split,
    }
    _write_exclusive(directory / "index.json", index_raw)
    _write_exclusive(directory / "family.manifest.json", manifest_raw)
    _write_exclusive(directory / "prompts.jsonl", prompt_raw)
    _write_exclusive(directory / "lineage.json", lineage_raw)
    _write_exclusive(directory / "checksums.json", _json_bytes(checksums))
    return checksums


def _identity_bytes(bound: Sequence[Mapping[str, Any]]) -> bytes:
    lines = []
    for info in sorted(bound, key=lambda item: item["review_id"]):
        member = info["member"]
        verdict = info["verdict"] or "unreviewed"
        lines.append("|".join([
            info["review_id"],
            member["source_dataset"],
            member["source_id"],
            member["item_id"],
            member["family_id"],
            member["split"],
            member["lang"],
            member["hs_label"],
            verdict,
        ]))
    return ("\n".join(lines) + "\n").encode("utf-8")


def plan_review(
    sheet_bytes: bytes,
    authority: Mapping[str, Any],
    acceptance_policy: str,
) -> dict[str, Any]:
    if acceptance_policy != ACCEPTANCE_POLICY:
        raise ReviewedVariantError("acceptance policy is required")
    rows = parse_sheet(sheet_bytes)
    identities = [{field: row.get(field) for field in IDENTITY_FIELDS} for row in rows]
    bound = bind_identities(identities, authority)
    variants, rejections = build_variants(rows, bound, acceptance_policy)
    coverage = coverage_report(authority, bound, variants)
    counts = {"accept": 0, "reject": 0, "revise": 0, "unreviewed": 0}
    for info in bound:
        name = info["verdict"] or "unreviewed"
        counts[name] += 1
    behaviors = {
        (variant["lineage"]["canonical_source_dataset"], variant["lineage"]["canonical_source_id"])
        for variant in variants
    }
    by_split = {split: 0 for split in ("calibration", "dev")}
    for variant in variants:
        by_split[variant["split"]] += 1
    return {
        "authority_sha256": authority["sha256"],
        "bound": bound,
        "coverage": coverage,
        "eligible_behavior_count": len(behaviors),
        "eligible_variant_count": len(variants),
        "eligible_variant_count_by_split": by_split,
        "identity_sha256": sha256_bytes(_identity_bytes(bound)),
        "rejections": rejections,
        "review_sheet_sha256": sha256_bytes(sheet_bytes),
        "row_count": len(bound),
        "variants": variants,
        "verdict_counts": counts,
    }


def export_reviewed_variants(
    *,
    review_sheet: Path,
    manifest: Path,
    out_dir: Path,
    acceptance_policy: str,
    require_pinned_hash: bool = True,
) -> dict[str, Any]:
    """Write a report. Frozen prompt files are written only for explicit accepts."""
    if acceptance_policy != ACCEPTANCE_POLICY:
        raise ReviewedVariantError("acceptance policy is required")
    if out_dir.exists():
        raise ReviewedVariantError("output directory already exists")
    authority = load_authority(manifest, require_pinned_hash=require_pinned_hash)
    if not review_sheet.is_file():
        raise ReviewedVariantError("missing review sheet")
    sheet_bytes = review_sheet.read_bytes()
    plan = plan_review(sheet_bytes, authority, acceptance_policy)
    prompt_files_written = plan["eligible_variant_count"] > 0
    status = "exported_incomplete" if prompt_files_written else "blocked_no_explicit_accept"
    out_dir.mkdir(parents=True, exist_ok=False)
    split_checksums: dict[str, dict[str, str]] = {}
    if prompt_files_written:
        for split in ("calibration", "dev"):
            group = [variant for variant in plan["variants"] if variant["split"] == split]
            if not group:
                continue
            split_checksums[split] = _write_split(
                out_dir / split,
                group,
                authority_sha=plan["authority_sha256"],
                sheet_sha=plan["review_sheet_sha256"],
            )
    report = {
        "acceptance_policy": acceptance_policy,
        "canonical_family_splits_sha256": plan["authority_sha256"],
        "checked_identity_sha256": plan["identity_sha256"],
        "complete": False,
        "corpus_complete": False,
        "corpus_complete_reason": (
            "Coverage counts only families present on this review sheet. "
            "Missing harmful or benign language cells stay missing. "
            "Language rows are variants of a canonical source item, not extra behaviors."
        ),
        "counting": {
            "behavior_count_excludes_language_multiplicity": True,
            "eligible_behavior_count": "distinct canonical source items with at least one eligible variant",
            "eligible_variant_count": "language rows",
            "missing_cells_are_not_filled": True,
        },
        "eligible_behavior_count": plan["eligible_behavior_count"],
        "eligible_variant_count": plan["eligible_variant_count"],
        "eligible_variant_count_by_split": plan["eligible_variant_count_by_split"],
        "prompt_files_written": prompt_files_written,
        "review_sheet_sha256": plan["review_sheet_sha256"],
        "row_count": plan["row_count"],
        "schema": "reviewed_variants_report_v1",
        "sheet_coverage_complete": plan["coverage"]["sheet_coverage_complete"],
        "split_checksums": split_checksums,
        "status": status,
        "verdict_counts": plan["verdict_counts"],
    }
    coverage = dict(plan["coverage"])
    coverage["corpus_complete"] = False
    rejection_lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in sorted(plan["rejections"], key=lambda item: (
            item["review_id"], item["reason"], item.get("lang", ""),
        ))
    ]
    rejection_raw = (("\n".join(rejection_lines) + "\n").encode("utf-8")) if rejection_lines else b""
    _write_exclusive(out_dir / "coverage.json", _json_bytes(coverage))
    _write_exclusive(out_dir / "rejections.jsonl", rejection_raw)
    _write_exclusive(out_dir / "report.json", _json_bytes(report))
    return report


def _expected_public(sheet_bytes: bytes, authority: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = plan_review(sheet_bytes, authority, ACCEPTANCE_POLICY)
    return [
        _public_variant(variant)
        for variant in sorted(
            (item for item in plan["variants"] if item["split"] == "dev"),
            key=lambda item: item["variant_id"],
        )
    ]


def _parse_lineage(raw: bytes) -> dict[str, Any]:
    _reject_prompt_keys_safe(raw)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ReviewedVariantError("reviewed lineage mismatch") from None
    if not isinstance(payload, dict):
        raise ReviewedVariantError("reviewed lineage mismatch")
    _reject_prompt_keys(payload)
    return payload


def _reject_prompt_keys_safe(raw: bytes) -> None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ReviewedVariantError("reviewed lineage mismatch") from None
    if any(f'"{key}"' in text for key in ("prompt", "english_or_literal", "corrected_hinglish", "corrected_hindi")):
        raise ReviewedVariantError("lineage must stay metadata-only")


def verify_ledger_lineage(
    *,
    lineage_bytes: bytes,
    review_sheet_bytes: bytes,
    index_rows: Sequence[Mapping[str, str]],
    authority: Mapping[str, Any],
    pins: Mapping[str, str],
) -> dict[str, str]:
    """Check derived-row lineage before the caller opens the prompt file."""
    if sha256_bytes(lineage_bytes) != pins.get("reviewed_lineage_sha256"):
        raise ReviewedVariantError("reviewed lineage mismatch")
    if sha256_bytes(review_sheet_bytes) != pins.get("review_sheet_sha256"):
        raise ReviewedVariantError("reviewed lineage mismatch")
    payload = _parse_lineage(lineage_bytes)
    if payload.get("schema") != "reviewed_lineage_v1" or payload.get("frozen") is not True:
        raise ReviewedVariantError("reviewed lineage mismatch")
    if payload.get("acceptance_policy") != ACCEPTANCE_POLICY:
        raise ReviewedVariantError("acceptance policy is required")
    if payload.get("split") != "dev":
        raise ReviewedVariantError("non-dev lineage is refused")
    if payload.get("canonical_family_splits_sha256") != authority["sha256"]:
        raise ReviewedVariantError("forged lineage")
    claimed = payload.get("variants")
    if not isinstance(claimed, list) or not claimed:
        raise ReviewedVariantError("reviewed lineage mismatch")
    ids = []
    for variant in claimed:
        if not isinstance(variant, dict):
            raise ReviewedVariantError("reviewed lineage mismatch")
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str):
            raise ReviewedVariantError("reviewed lineage mismatch")
        ids.append(variant_id)
    if len(ids) != len(set(ids)):
        raise ReviewedVariantError("duplicate variant id")
    expected = _expected_public(review_sheet_bytes, authority)
    claimed_sorted = sorted(claimed, key=lambda item: item["variant_id"])
    expected_ids = [item["variant_id"] for item in expected]
    claimed_ids = [item["variant_id"] for item in claimed_sorted]
    if claimed_ids != expected_ids:
        if set(claimed_ids) - set(expected_ids):
            raise ReviewedVariantError("unreviewed variant")
        raise ReviewedVariantError("reviewed lineage mismatch")
    for found, wanted in zip(claimed_sorted, expected):
        if found.get("lineage") != wanted["lineage"]:
            raise ReviewedVariantError("forged lineage")
        if (
            found.get("prompt_hash") != wanted["prompt_hash"]
            or found.get("correction_applied") != wanted["correction_applied"]
            or found.get("draft_prompt_sha256") != wanted["draft_prompt_sha256"]
        ):
            raise ReviewedVariantError("correction drift")
        if found != wanted:
            raise ReviewedVariantError("reviewed lineage mismatch")
        _assert_authority_lineage(found, authority)
    if payload.get("review_sheet_sha256") != pins["review_sheet_sha256"]:
        raise ReviewedVariantError("reviewed lineage mismatch")
    source_sha = payload.get("canonical_source_ids_sha256")
    prompts_sha = payload.get("prompts_sha256")
    evidence = payload.get("evidence_sha256")
    if (source_sha != pins.get("canonical_source_ids_sha256")
            or source_sha != sha256_bytes(_source_id_bytes(expected))):
        raise ReviewedVariantError("reviewed lineage mismatch")
    if prompts_sha != pins.get("prompts_sha256"):
        raise ReviewedVariantError("reviewed lineage mismatch")
    if evidence != evidence_sha256(pins["review_sheet_sha256"], source_sha, prompts_sha):
        raise ReviewedVariantError("reviewed lineage mismatch")
    if evidence != pins.get("reviewed_evidence_sha256"):
        raise ReviewedVariantError("reviewed lineage mismatch")
    by_sample = {row["sample_id"]: row for row in index_rows}
    if set(by_sample) != set(claimed_ids):
        raise ReviewedVariantError("reviewed lineage mismatch")
    for variant in claimed_sorted:
        row = by_sample[variant["sample_id"]]
        for field in INDEX_FIELDS:
            if row.get(field) != variant.get(field):
                raise ReviewedVariantError("reviewed lineage mismatch")
    return {
        "canonical_source_ids_sha256": source_sha,
        "review_sheet_sha256": pins["review_sheet_sha256"],
        "reviewed_evidence_sha256": evidence,
        "reviewed_lineage_sha256": pins["reviewed_lineage_sha256"],
    }


def _assert_authority_lineage(variant: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    lineage = variant["lineage"]
    if set(lineage) != set(LINEAGE_FIELDS):
        raise ReviewedVariantError("forged lineage")
    if lineage["verdict"] != "accept":
        raise ReviewedVariantError("unreviewed variant")
    found = authority["by_source"].get((lineage["canonical_source_dataset"], lineage["canonical_source_id"]))
    if not isinstance(found, dict):
        raise ReviewedVariantError("forged lineage")
    if found["split"] == "test" or lineage["canonical_split"] == "test":
        raise ReviewedVariantError("test split is refused")
    compared = {
        "canonical_family_id": found["family_id"],
        "canonical_hs_label": found["hs_label"],
        "canonical_item_id": found["item_id"],
        "canonical_lang": found["lang"],
        "canonical_source_dataset": found["source_dataset"],
        "canonical_source_id": found["source_id"],
        "canonical_split": found["split"],
    }
    for key, value in compared.items():
        if lineage.get(key) != value:
            raise ReviewedVariantError("forged lineage")
    if variant["family_id"] != found["family_id"] or variant["hs_label"] != found["hs_label"]:
        raise ReviewedVariantError("forged lineage")
    if variant["split"] != "dev" or found["split"] != "dev":
        raise ReviewedVariantError("non-dev lineage is refused")
    if variant["source_dataset"] != REVIEWED_VARIANT_SOURCE:
        raise ReviewedVariantError("forged lineage")
    if variant["item_id"] == found["item_id"] or variant["lang"] == found["lang"] and not variant["correction_applied"]:
        raise ReviewedVariantError("forged lineage")
    if variant["sample_id"] != variant["variant_id"] or variant["source_id"] != variant["variant_id"]:
        raise ReviewedVariantError("forged lineage")
    if type(variant["correction_applied"]) is not bool:
        raise ReviewedVariantError("correction drift")
