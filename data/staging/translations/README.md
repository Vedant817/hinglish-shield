# Staging translations (human-review queue)

Canonical location per Engineering Standards: `data/staging/` for translations and review queues.
Authorized seeds remain at `data/seeds/` (SOURCES.md / MANIFEST.json). Proposed `data/raw/` alias not created yet to avoid breaking other agents.

Task: Translate and transliterate English seeds into Hindi (Devanagari) and Hinglish (Roman, casual texting).
Notion Done Criteria: each item has EN, HI, Hinglish variants. Due 2026-11-07.

## Integrity (required reading)

- These rows are custom adaptations for Hinglish Shield, not official benchmark scores.
- XSTest unsafe contrast items are harmful evaluation prompts, not benign.
- batch01 is a hand-reviewed sample before semantic-family split and CuratorKIT export. Not a locked final test set.
- No model was run to produce these translations. QC is script heuristics plus planned human review.

## Schema (`batchXX.csv`)

| column | meaning |
| --- | --- |
| id | batch-local id (`b01-001`) |
| source | `xstest` / `harmbench` / `or_bench_toxic` / `or_bench_hard` |
| source_id | upstream id (`BehaviorID`, XSTest `id`, or `row:N` for OR-Bench) |
| english | English seed prompt |
| hindi | Devanagari Hindi |
| hinglish | Roman code-mixed casual Hinglish |
| label | `harmful` or `benign` |
| category | source-specific category tag |
| qc_notes | free-text QC / FLAG notes |

## Tooling

```bash
cd /workspace/hinglish-shield
python3 -m src.curate.translate_seeds write-batch01
python3 -m src.curate.translate_seeds sample --n 30 --out data/staging/translations/batch02_todo.csv --prefix b02
python3 -m src.curate.translate_seeds qc --csv data/staging/translations/batch01.csv --report data/staging/translations/batch01_QC.md
```

Hinglish style target: casual WhatsApp/texting mix (yaar/bhai particles OK), roughly 18% to 60% Latin tokens.

## Family_id and split

Columns `family_id` and `split` are filled from DE B `data/splits/family_splits_v2.json` via `src/curate/family_split_lookup.py`.
Left blank when the manifest is not present yet. Do not invent local family IDs.
Future `translate_seeds.py sample` requires the manifest and skips frozen test families.

Note: after v2 fill, **24/30** batch01 rows are test-split (2 calibration / 4 dev / 24 test).
