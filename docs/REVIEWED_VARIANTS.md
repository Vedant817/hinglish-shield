# Reviewed translation variants

This path freezes language variants that a human has explicitly accepted. It is the lineage contract the dev ledger requires before it will read a translated prompt. It does not score models, and it does not claim that Hinglish judging is validated.

## Current sheet

The exporter reads `data/staging/review/vedant_review_nov7_calibration_dev.csv` and the pinned manifest `data/splits/family_splits_v2.json` (SHA-256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`). The mixed queue `vedant_review_nov7.csv` is not an input.

As of this document, all 26 verdicts on that calibration/dev sheet are blank. The command therefore writes a rejection and coverage report and writes no eligible prompt files. Exit code 2 means the export is blocked. That block is the honest state until Vedant fills `vedant_verdict`.

## Command

```text
python scripts/export_reviewed_variants.py --acceptance-policy explicit-human-accept-v1 --out runs/reviewed-variants/<id>
```

`--acceptance-policy` must be `explicit-human-accept-v1`. Any other policy exits 2 and writes nothing. `--out` must be a new directory. The command refuses to overwrite it.

Checked interpreter for unit tests: `C:/Users/vedan/Documents/Resources/Code/hinglish-shield-scratch/.venv-ci/Scripts/python.exe`. That environment has no torch. This command does not load Tiny Aya or WildGuard.

## Acceptance policy

`explicit-human-accept-v1` admits a language cell only when `vedant_verdict` is `accept` after trimming and lowercasing. The tokens `revise` and `reject` stay ineligible. A blank cell stays `unreviewed` and ineligible. Any other token fails the export before files are written.

On an accepted row, `corrected_hindi` or `corrected_hinglish` replaces that language's draft when the correction is non-empty and differs from the draft. An empty correction leaves the draft. A correction that only repeats the draft is not a change. There is no corrected-English column. The exporter never fills a blank cell with generated text.

The original canonical language is not copied into a new variant unless a real correction changed it. The new row uses `source_dataset` `reviewed_variant`, `item_id` `reviewed:<review_id>:<lang>`, and a lineage object that names the canonical item, source, family, split, original language, and original label. The translated language stays on the variant. It is not written onto the canonical member.

Labels come from the pinned member. `category_or_harm` on the sheet is not label authority.

## What gets checked first

The manifest hash is checked before the review sheet is opened. Each row's source, source id, family, and split are then checked against manifest metadata before any language cell is turned into a prompt. A test member refuses the whole export, including when the row names a calibration or dev family. Duplicate review ids and duplicate variant ids refuse the export. Unknown verdicts refuse the export. Those failures leave no output directory.

## Files

When at least one variant is eligible, the command writes `calibration/` and `dev/` only for splits that have eligible rows:

- `index.json`: dev-ledger index fields, metadata only
- `family.manifest.json`: variants grouped by family, with `languages` listing the eligible languages in that family
- `prompts.jsonl`: `sample_id`, `item_id`, `prompt`
- `lineage.json`: `reviewed_lineage_v1`, frozen, with the acceptance policy and hashes
- `checksums.json`: file hashes for the ledger config

`report.json`, `coverage.json`, and `rejections.jsonl` are always written when the export gets past membership checks. They contain ids, splits, labels, counts, and reason codes. They do not contain prompt text.

Reason codes include `unreviewed`, `rejected`, `revise_ineligible`, `missing_text`, and `accept_without_variant`. `revise_ineligible` records `correction_present` as a boolean and drops the correction text.

## Coverage

ADR 0007 asks for matched harmful and benign near-neighbors in EN, HI, and Hinglish inside a semantic family. The coverage file reports, for each family that appears on the sheet:

- canonical member counts by label and language, from the pinned manifest
- eligible reviewed-variant counts by label and language
- the union of those cells, and the missing cells

A cell is present only when its count is greater than zero. Unreviewed text does not fill a cell. Three language rows of one canonical item are one behavior. `eligible_behavior_count` counts distinct canonical source items that have at least one eligible variant. `eligible_variant_count` counts language rows. Those two numbers are not added together.

`sheet_coverage_complete` is true only when every family on the sheet has all six harmful/benign by EN/HI/Hinglish cells present in the union. `corpus_complete` stays false. This report does not describe families that are absent from the sheet, and it is not an ASR denominator.

The dev ledger reads only the `dev/` files. A calibration index still has `split` `calibration`, so the dev ledger refuses it. See `docs/DEV_RUNNER.md` for the hash pins. Copy `checksums.json` into `frozen.dev.json` and pass `--reviewed-lineage` and `--review-sheet`. The runner recomputes the accepted dev variants from the sheet and checks lineage before it opens `prompts.jsonl`.

Fixture factories remain `artifact_kind` `fixture` and `split` `example`. A synthetic canonical manifest cannot authorize `artifact_kind` `run`. Method `base` is unchanged.

## Still unresolved

- Vedant has not entered accepts, revisions, or rejects. No real eligible variant file exists.
- Hinglish evaluation stays unvalidated until a fluent human judge-calibration sample exists (ADR 0007 Decision 2). This export does not create that sample.
- A dev ledger that uses these files still needs a real Tiny Aya and WildGuard runtime before any generation exists. This repository's checked light environment does not load those models.
- Preservation scoring and primary ASR remain withheld by the dev ledger.
