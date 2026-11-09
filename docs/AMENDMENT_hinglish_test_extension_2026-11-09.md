# Pre-registration amendment: Hinglish extension of the locked test (proposal)

Status: **PROPOSED, not approved**. Recorded 2026-11-09, before Nov 11 and
before any Hinglish on the test split is generated. This file proposes how a
Hinglish extension to the locked test would be built if Vedant approves it.
No test-split Hinglish rows exist, and none are generated under this
proposal. The frozen test split stays untouched until Nov 11 and the frozen
arm.

## Why

The frozen test split contains only English and Hindi (hi: 442 harm rows).
A Hinglish safety audit without Hinglish test rows cannot measure whether
Hinglish prompts get refused at the rate English and Hindi ones do, which
is the central question of this project. The Hinglish extension is
therefore required for calibration/dev generation (in-notebook, on the
pinned Qwen2.5-7B-Instruct) and for the locked test (this amendment).

Two generator attempts were made before the Kaggle stage: NLLB-200's
tokenizer has no `hin_Latn` target (verified against the pinned model);
Qwen2.5-1.5B-Instruct on CPU produced no code-mixed Hinglish rows that pass
the both-language check (0 of 40 kept), so the CPU generation path was
abandoned. A user instruction also confirmed: pinned translation model is
not to be replaced by regex transliteration or hand-written strings.

## Proposal (build only if approved, starting Nov 11 or later)

1. Families: the same family ids as the frozen test split (from
   `data/splits/family_splits_v2.json`), same pairing as calibration/dev.
2. Same generator as the calibration/dev Hinglish build: the pinned
   `Qwen2.5-1.5B-Instruct` checkpoint at revision `989aa7980e...` (full
   pin in `data/curated/hs_harm_hinglish/lineage.json`), same system and
   user prompt, greedy decoding, seed 42 documented.
3. The same three automated checks (Latin-script ratio, both-language
   token presence, refusal/empty detector) and the same drop policy. Rows
   that fail are counted, not hand-fixed.
4. Every test Hinglish row labeled `unreviewed machine translation`, with
   `metrics_eligible: false` and `primary_asr: null` until Vedant's
   review-sheet verdicts admit them.
5. New curated files (never touching the frozen test files):
   `data/curated/hs_harm_hinglish/test/hs_eval.jsonl`, pinned by its own
   SHA-256 and family-manifest hash in the amendment that introduces them.
6. The amendment that introduces the files records: the proposal hash
   below, the generated file hashes, the counts, the lineage file, the
   review sheet, coverage of drop counts, and Vedant's approval.
7. Until then the locked test stays as-is (English plus Hindi), and the
   locked-test command in `docs/LOCKED_TEST_PREP.md` does not change its
   config: extending it would be a separate frozen-config decision.

## Proposal hash

proposal_sha256: 3bfcf861efe3ff465d315c925f7970a48e46490eb01d89751236b90cbea470b0
The SHA-256 above is the SHA-256 of this file's bytes up to and including
the line before `## Proposal hash`, so the proposal is content-addressed.
It commits the experiment design, not any result.
