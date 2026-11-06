# Dataset card draft: Hinglish Shield translation staging

Status: DRAFT for human review. Not a locked test set. Not an official XSTest / HarmBench / OR-Bench / Aya score.

## Summary

Staging translations for Track 2 Hinglish Shield. English seeds translated to Hindi (Devanagari) and Hinglish (Roman casual texting). Separate Aya Hindi romanized Hinglish pass.

## Sources and licenses

| source | path | license (from seeds/SOURCES.md) |
| --- | --- | --- |
| HarmBench behaviors | data/seeds/harmbench | MIT |
| XSTest | data/seeds/xstest | CC-BY-4.0 |
| OR-Bench | data/seeds/or_bench | CC-BY-4.0 |
| Aya red-teaming HI | data/seeds/aya_redteaming/aya_hin.jsonl | Apache-2.0 |

Respect upstream terms. Tiny Aya / research use constraints still apply downstream.

## Staging files

| file | n | notes |
| --- | --- | --- |
| data/staging/translations/batch01.csv | 30 | EN+HI+Hinglish from EN seeds |
| data/staging/translations/aya_hinglish/aya_hinglish_sample50.csv | 50 | HI->Hinglish review sample |
| data/staging/review/vedant_review_nov7.csv | 59 | 9 batch01 FLAG + 50 Aya sample |

## Translation columns (batch01)

id, source, source_id, english, hindi, hinglish, label, category, qc_notes

- `label` / `category`: kept as metadata (orchestrator decision).
- `label` values: `harmful` | `benign` (maps to HS `hs_label` harm|benign on export).

## Aya Hinglish columns

id, source (=aya_redteaming_hi), source_id, prompt_hi, literal_en, hinglish, harm_category, global_or_local, method, redaction_notes, qc_notes

### Method (honest)

`method=llm_authored_by_agent`: Data Engineer A agent authored Roman Hinglish from the Hindi prompt plus Aya `literal_translation`. Not rule-based IAST transliteration. Not native-verified. Rows with operational weapons/CSAM detail were excluded upstream; some operational violence how-to source_ids were replaced (see aya_hinglish/NOTES.md).

## CuratorKIT export mapping

Pinned CuratorKIT 0.1.21 `DataSample` (vendor/CuratorKIT/curatorkit/schema.py):

- Audit prompts use `task_type=prompt_only`, `instruction=<prompt>`, empty `output`.
- Project fields live in `metadata`: item_id, hs_label, source_dataset, source_id, category, lang, split, variant_of, qc_notes.
- One CuratorKIT row per language variant. Shared `variant_of` / family id across en/hi/hinglish.
- Machine-readable map: `data/staging/translations/curatorkit_export_mapping.json`
- Helper: `python -m src.curate.export_to_curatorkit --csv ... --out-hs ... --out-ck ...`

Project curated EN exports already use hs_eval fields:
item_id, prompt, hs_label, source_dataset, source_id, category, lang, split, variant_of, curatorkit_id
(see data/curated/hs_harm_en/train/hs_eval.jsonl when present).

CuratorKIT run also writes manifest.json, rejected.jsonl, lexsi_provenance.json, checksums.txt (Approaches Deep Dive / README).

## Integrity notes

- Custom adaptations, not official benchmark leaderboard scores.
- XSTest unsafe contrasts are harmful, never benign.
- batch01 and aya sample are pre semantic-family-split review samples.
- No model inference used to author these translations.
- Do not publish unfiltered dangerous completions in a public demo.

## Open for Vedant / Lead Architect

- Accept/correct FLAG rows in the Nov 7 review sheet.
- Confirm export mapping before CuratorKIT scale export.

## Family splits (Protocol / prereg)

- Unit: semantic family. Ratios calibration 20% / dev 30% / test 50%, seed 42 (research/baseline-prereg 8b6af17).
- DE B owns `data/splits/family_splits_v2.json` and `src/curate/check_family_splits.py`.
- Staging CSVs include `family_id` and `split` columns filled by lookup on `(source, source_id)`. Blank until the manifest exists.
- `translate_seeds.py` samples only manifest-assigned families and refuses unassigned or frozen `test` rows.
- Check: `make check-family-splits-translations` (skips until DE B artifacts land; then runs DE B checker).

## Family split fill (v2)

Filled from DE B `data/splits/family_lookup.csv` (manifest `family_splits_v2.json`, seed 42).
Do not invent local family IDs. XSTest is never in calibration under v2.

batch01 row counts after fill: calibration 2 / dev 4 / test 24 (24/30 are test-split; 0 unmapped).
Aya sample50 and the Vedant review sheet are filled the same way.
Default `translate_seeds.py sample` draws calibration/dev only; `--include-test-families` is required to touch test (for completing language variants; test remains frozen after commit).
