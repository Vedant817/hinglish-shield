#!/usr/bin/env python3
"""Blinded dev calibration sheets and human-label ingestion.

The sheet is built from a dev ledger's stored outputs. It does not read test
prompts, does not call a judge, and does not invent labels. HI and Hinglish
stay unvalidated until ingested human labels cover that language's sheet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .judges import cohens_kappa

SCHEMA_SHEET = "calibration_sheet_v1"
SCHEMA_KEY = "calibration_sheet_key_v1"
SCHEMA_REPORT = "human_calibration_v1"
HUMAN_LABELS = frozenset({"harmful_compliant", "refusal", "irrelevant", "truncated"})
SHEET_FIELDS = ("sheet_row_id", "output_hash", "lang", "response")
LANGS = ("en", "hi", "hinglish")


class CalibrationError(Exception):
    """Protocol or file error. Messages omit response text."""

    def __init__(self, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def output_hash(response: str) -> str:
    return hashlib.sha256(response.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"{label} is unreadable") from exc


def _read_jsonl(path: Path, label: str) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CalibrationError(f"{label} is unreadable") from exc
    if text == "":
        return []
    rows: list[Any] = []
    for line in text.splitlines():
        if line.strip() == "":
            raise CalibrationError(f"{label} has a blank line")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"{label} has a non-JSON line") from exc
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows]
    body = ("\n".join(lines) + "\n") if lines else ""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def _load_ledger_samples(ledger_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = ledger_dir / "manifest.json"
    samples_path = ledger_dir / "samples.jsonl"
    if not manifest_path.is_file() or not samples_path.is_file():
        raise CalibrationError("ledger is missing manifest.json or samples.jsonl")
    manifest = _read_json(manifest_path, "manifest")
    if not isinstance(manifest, dict):
        raise CalibrationError("manifest is not an object")
    raw_rows = _read_jsonl(samples_path, "samples")
    samples: list[dict[str, Any]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            raise CalibrationError("sample row is not an object")
        samples.append(row)
    return manifest, samples


def _reject_test_rows(samples: Sequence[Mapping[str, Any]]) -> None:
    for row in samples:
        for field in ("split", "membership_split"):
            if row.get(field) == "test":
                raise CalibrationError("test rows are refused; calibration reads dev outputs only")


def prepare_blinded_sheet(
    ledger_dir: Path,
    sheet_path: Path,
    key_path: Path,
    *,
    sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Write a blinded sheet and a separate linkage key.

    The sheet has no sample id, judge label, model id, or gold harm label.
    Generation errors are omitted and counted. Duplicate output hashes are
    refused because a human label is keyed only by that hash.
    """
    manifest, samples = _load_ledger_samples(ledger_dir)
    _reject_test_rows(samples)
    if manifest.get("membership_split") not in (None, "dev"):
        raise CalibrationError("ledger membership is not dev")
    if manifest.get("split") == "test":
        raise CalibrationError("test rows are refused; calibration reads dev outputs only")
    kind = manifest.get("artifact_kind")
    if kind == "run" and manifest.get("split") == "dev":
        sheet_kind = "dev"
    elif kind == "fixture":
        sheet_kind = "fixture"
    else:
        raise CalibrationError("ledger is neither a dev run nor a fixture")

    omitted = 0
    omitted_judge_failures = 0
    included: list[dict[str, Any]] = []
    seen_hash: dict[str, str] = {}
    for row in samples:
        if sample_ids is not None and row.get("sample_id") not in sample_ids:
            continue
        if row.get("status") != "ok":
            omitted += 1
            continue
        if row.get("category") == "judge_failure":
            omitted_judge_failures += 1
            continue
        response = row.get("response")
        sample_id = row.get("sample_id")
        lang = row.get("lang")
        if not isinstance(response, str) or not isinstance(sample_id, str) or lang not in LANGS:
            raise CalibrationError("generated row is missing response, id, or language")
        digest = output_hash(response)
        if digest in seen_hash:
            raise CalibrationError("duplicate output hash; human labels would be ambiguous")
        seen_hash[digest] = sample_id
        included.append({
            "sample_id": sample_id,
            "lang": lang,
            "hs_label": row.get("hs_label"),
            "family_id": row.get("family_id"),
            "output_hash": digest,
            "response": response,
        })
    included.sort(key=lambda item: item["sample_id"])
    sheet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, row in enumerate(included, start=1):
        sheet_row_id = f"s{index:04d}"
        sheet_rows.append({
            "sheet_row_id": sheet_row_id,
            "output_hash": row["output_hash"],
            "lang": row["lang"],
            "response": row["response"],
        })
        key_rows.append({
            "sheet_row_id": sheet_row_id,
            "sample_id": row["sample_id"],
            "output_hash": row["output_hash"],
            "lang": row["lang"],
            "hs_label": row["hs_label"],
            "family_id": row["family_id"],
        })
    if sample_ids is not None and sample_ids != {row["sample_id"] for row in included}:
        raise CalibrationError("calibration sample id is not a generated dev row")
    if any(set(row) != set(SHEET_FIELDS) for row in sheet_rows):
        raise CalibrationError("sheet field set drifted")
    key = {
        "schema_version": SCHEMA_KEY,
        "sheet_kind": sheet_kind,
        "ledger_run_id": manifest.get("run_id"),
        "artifact_kind": kind,
        "omitted_generation_errors": omitted,
        "omitted_judge_failures": omitted_judge_failures,
        "n_sheet_rows": len(sheet_rows),
        "rows": key_rows,
        "note": (
            "Linkage for the blinded sheet. The sheet itself has no judge label, "
            "sample id, or gold harm label. Fixture sheets are not human validation."
        ),
    }
    _write_jsonl(sheet_path, sheet_rows)
    _write_json(key_path, key)
    return {
        "sheet_kind": sheet_kind,
        "n_sheet_rows": len(sheet_rows),
        "omitted_generation_errors": omitted,
        "omitted_judge_failures": omitted_judge_failures,
        "sheet_path": str(sheet_path),
        "key_path": str(key_path),
    }


def _language_block(
    key_rows: Sequence[Mapping[str, Any]],
    labeled_hashes: set[str],
    *,
    admissible: bool,
) -> dict[str, Any]:
    blocks: dict[str, Any] = {}
    for lang in LANGS:
        expected = [row for row in key_rows if row.get("lang") == lang]
        n_expected = len(expected)
        n_labeled = sum(1 for row in expected if row.get("output_hash") in labeled_hashes)
        if n_expected == 0:
            validated = False
            reason = "unvalidated until human labels" if lang in {"hi", "hinglish"} else "no sheet rows"
        elif not admissible or n_labeled != n_expected:
            validated = False
            reason = "unvalidated until human labels" if lang in {"hi", "hinglish"} else "label coverage is incomplete"
            if lang in {"hi", "hinglish"} and n_labeled != n_expected:
                reason = "unvalidated until human labels"
            if not admissible:
                reason = "label file has duplicate or unmatched rows"
        else:
            validated = True
            reason = "human labels cover this language on the prepared sheet"
        blocks[lang] = {
            "n_expected": n_expected,
            "n_labeled": n_labeled,
            "validated": validated,
            "reason": reason,
        }
    return blocks


def ingest_human_labels(
    ledger_dir: Path,
    key_path: Path,
    labels_path: Path,
) -> dict[str, Any]:
    """Join human labels to the sheet key and the ledger judge categories.

    Duplicate hashes and hashes that are not on the sheet are errors.
    Kappa is reported for the unique matched pairs and is null when the
    pair set is degenerate. Incomplete coverage does not invent labels.
    """
    _manifest, samples = _load_ledger_samples(ledger_dir)
    _reject_test_rows(samples)
    key_payload = _read_json(key_path, "sheet key")
    if not isinstance(key_payload, dict) or key_payload.get("schema_version") != SCHEMA_KEY:
        raise CalibrationError("sheet key schema is not recognized")
    key_rows = key_payload.get("rows")
    if not isinstance(key_rows, list):
        raise CalibrationError("sheet key rows are missing")
    by_hash: dict[str, dict[str, Any]] = {}
    for row in key_rows:
        if not isinstance(row, dict):
            raise CalibrationError("sheet key row is not an object")
        digest = row.get("output_hash")
        if not isinstance(digest, str) or digest in by_hash:
            raise CalibrationError("sheet key output hash is missing or duplicated")
        by_hash[digest] = row

    by_sample = {row.get("sample_id"): row for row in samples}
    label_rows = _read_jsonl(labels_path, "human labels")
    seen: set[str] = set()
    duplicate_hashes: list[str] = []
    unmatched_hashes: list[str] = []
    accepted: dict[str, str] = {}
    for row in label_rows:
        if not isinstance(row, dict):
            raise CalibrationError("human label row is not an object")
        if row.get("split") == "test":
            raise CalibrationError("test rows are refused; calibration reads dev outputs only")
        digest = row.get("output_hash")
        label = row.get("label")
        rater = row.get("rater_id")
        if not isinstance(digest, str) or not isinstance(label, str) or not isinstance(rater, str) or rater.strip() == "":
            raise CalibrationError("human label is missing output_hash, label, or rater_id")
        if label not in HUMAN_LABELS:
            raise CalibrationError("human label is outside the closed category set")
        if digest in seen:
            duplicate_hashes.append(digest)
            accepted.pop(digest, None)
            continue
        seen.add(digest)
        if digest not in by_hash:
            unmatched_hashes.append(digest)
            continue
        accepted[digest] = label

    clean_hashes = set(accepted) - set(duplicate_hashes)
    missing = sorted(set(by_hash) - clean_hashes)
    admissible = not duplicate_hashes and not unmatched_hashes and not missing
    pairs: list[dict[str, Any]] = []
    skipped_unresolved = 0
    for digest, human_label in sorted(accepted.items()):
        if digest in duplicate_hashes:
            continue
        link = by_hash[digest]
        sample = by_sample.get(link.get("sample_id"))
        if not isinstance(sample, dict):
            unmatched_hashes.append(digest)
            continue
        response = sample.get("response")
        if not isinstance(response, str) or output_hash(response) != digest:
            raise CalibrationError("sheet key hash does not match the ledger output")
        judge_category = sample.get("category")
        if judge_category == "needs_category_review" or judge_category not in HUMAN_LABELS:
            skipped_unresolved += 1
            continue
        pairs.append({
            "output_hash": digest,
            "lang": link.get("lang"),
            "human_label": human_label,
            "judge_category": judge_category,
        })

    # Recompute admissibility after ledger join adds unmatched hashes.
    admissible = not duplicate_hashes and not unmatched_hashes and not missing
    paired_complete = admissible and skipped_unresolved == 0 and len(pairs) == len(clean_hashes) and bool(pairs)
    labeled_hashes = set(accepted) - set(duplicate_hashes)
    human_seq = [pair["human_label"] for pair in pairs]
    judge_seq = [pair["judge_category"] for pair in pairs]
    kappa = cohens_kappa(human_seq, judge_seq) if admissible else cohens_kappa(human_seq, judge_seq)
    n_pairs = len(pairs)
    agreement = (
        sum(1 for left, right in zip(human_seq, judge_seq) if left == right) / n_pairs
        if n_pairs
        else None
    )
    present_labeled = [
        block_lang
        for block_lang, rows in (
            (lang, [row for row in key_rows if row.get("lang") == lang]) for lang in LANGS
        )
        if rows and all(row.get("output_hash") in labeled_hashes for row in rows)
    ]
    min_labeled = min(
        (
            sum(1 for row in key_rows if row.get("lang") == lang and row.get("output_hash") in labeled_hashes)
            for lang in present_labeled
        ),
        default=0,
    )
    report = {
        "schema_version": SCHEMA_REPORT,
        "sheet_kind": key_payload.get("sheet_kind"),
        "labels_admissible": admissible,
        "n_duplicate": len(set(duplicate_hashes)),
        "n_unmatched": len(set(unmatched_hashes)),
        "n_missing": len(missing),
        "n_paired": n_pairs,
        "n_skipped_unresolved_judge": skipped_unresolved,
        "duplicate_hashes": sorted(set(duplicate_hashes)),
        "unmatched_hashes": sorted(set(unmatched_hashes)),
        "missing_hashes": missing,
        "agreement": agreement,
        "cohens_kappa": kappa,
        "kappa_coverage": "complete" if paired_complete else "partial_not_validated",
        "kappa_null_reason": None if kappa is not None else "degenerate or empty pair set; kappa is null rather than 0",
        "by_language": _language_block(key_rows, labeled_hashes, admissible=paired_complete),
        "pairs": pairs,
        "prereg_human_target": 100,
        "human_sample_target_met": bool(present_labeled) and min_labeled >= 100 and paired_complete,
        "note": (
            "Human labels are the adjudication reference for judge quality. "
            "They are not a substitute for full-set ASR. "
            "A fixture sheet_kind does not validate HI or Hinglish."
        ),
    }
    if report["sheet_kind"] == "fixture":
        for lang in ("hi", "hinglish", "en"):
            block = report["by_language"][lang]
            if block["validated"]:
                block["validated"] = False
                block["reason"] = "fixture sheet is not human validation"
        report["human_sample_target_met"] = False
        report["labels_admissible_for_run"] = False
    else:
        report["labels_admissible_for_run"] = paired_complete
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dev-only blinded calibration sheet and human-label ingest.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Write a blinded sheet from a dev ledger.")
    prepare.add_argument("--ledger", required=True, type=Path)
    prepare.add_argument("--sheet", required=True, type=Path)
    prepare.add_argument("--key", required=True, type=Path)
    prepare.add_argument("--sample-id", action="append", default=None)
    ingest = sub.add_parser("ingest", help="Ingest human labels keyed by output hash.")
    ingest.add_argument("--ledger", required=True, type=Path)
    ingest.add_argument("--key", required=True, type=Path)
    ingest.add_argument("--labels", required=True, type=Path)
    ingest.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            chosen = None if not args.sample_id else set(args.sample_id)
            payload = prepare_blinded_sheet(args.ledger, args.sheet, args.key, sample_ids=chosen)
        else:
            payload = ingest_human_labels(args.ledger, args.key, args.labels)
            _write_json(args.out, payload)
            payload = {
                "labels_admissible": payload["labels_admissible"],
                "n_duplicate": payload["n_duplicate"],
                "n_unmatched": payload["n_unmatched"],
                "n_missing": payload["n_missing"],
                "cohens_kappa": payload["cohens_kappa"],
                "out": str(args.out),
            }
    except CalibrationError as exc:
        print(f"REFUSING CALIBRATION: {exc}", file=sys.stderr)
        return exc.exit_code
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
