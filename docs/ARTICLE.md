# Hinglish Shield: technical writeup (Lexsi Track 2, draft)

Status: draft for submission preparation. This document contains no scored locked-test results. Every number below traces to a run or artifact already in this repository and is cited with its source path. Placeholders mark values that require the pending locked-test/Hinglish-extension run.

## Problem

Hinglish Shield targets multilingual and code-mixed safety for financial-support interactions on [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global) (`CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`, target identity recorded in `runs/gpu-20261111-180320-270579d4/metrics.json`). The setting is English (EN), Hindi (HI), and romanized Hindi-English code-mixing (Hinglish) prompts in a financial-support context, where harmful compliance, over-refusal on benign requests, and cross-language gaps matter. The project goal is a reproducible audit-and-repair acceptance pack, not a claimed safety improvement without evidence.

## Data and curation pipeline

- **Seeds (read-only):** Aya Red Teaming (`data/seeds/aya_redteaming/`, including `aya_hin.jsonl` for native HI harm rows; HI seed sha256 `44caefa0da81dcbc8492c9abd8bc2d3e7d9773c02215fa283347645c23958031`, rows 915 in v21 data pin context: `runs/gpu-20261111-180320-270579d4/d8_descriptive.json`), HarmBench text behaviors (`data/seeds/harmbench/`), XSTest (`data/seeds/xstest/`), and OR-Bench (`data/seeds/or_bench/`). Plan and schema: `data/CURATION_PLAN.md`, `data/seeds/SOURCES.md`, `data/seeds/MANIFEST.json`.
- **Family-level splits:** frozen manifest `data/splits/family_splits_v2.json` (sha256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`, seed 42, calibration/dev/test 0.20/0.30/0.50 by family, stratified as documented; see `data/CURATION_PLAN.md`, `data/splits/family_splits_v2.json`). Unit is the semantic family; translations/paraphrases inherit split. XSTest excluded from calibration in v2 (0 XSTest in calibration). Test is frozen for the single locked pass. Lookup/check: `data/splits/family_lookup.csv`, `scripts/check_translation_family_splits.py`, `src/curate/family_splits.py`, `src/curate/check_family_splits.py`.
- **Normalization and interim rows:** canonical interim schema and per-language layout under `data/interim/{en,hi,hinglish}/` and translations under `data/translations/{hi,hinglish}/` as specified in `data/CURATION_PLAN.md`. EN v0 build: `src/curate/build_en_v0.py`.
- **CuratorKIT curation (CPU-only path):** clean + exact dedup + schema/quality gates (toxicity/hallucination/reward off for v0 prompt corpora) and PPO export configs at `configs/curate/hs_harm_en.yaml`, `configs/curate/hs_benign_en.yaml`, with enriched `hs_eval.jsonl` alongside PPO outputs under `data/curated/` (see `data/CURATION_PLAN.md`). Payload-preserving cleaner wiring and pre-export fixture gate documented in `data/CURATION_PLAN.md` (CuratorKIT issue 42 mitigation). Curated EN slices used by v21 are pinned by commit and sha256 in `runs/gpu-20261111-180320-270579d4/d8_descriptive.json` (data pin, commit `c5b436bdaadb360af4e21411d8ab6316a75bca4b`; e.g. benign_calibration 92 rows sha256 `91b9de4d095031d032216dbf2beb462e09d9550cacba3cd614b6f63ea3d7a6cc`, benign_dev 245 rows sha256 `ece4d51eed3712d97fbfe0036c39e32481de5ecbb4524a511b3fb0c443745924`, harm_calibration 138 rows sha256 `1332c00e01ee666976de7d3d21b41152373b615cceb753f5d461548185d9a041`, harm_dev 296 rows sha256 `f213d954e32b919c6786686ba34674504aee69467b0263da530e7c8cacb2de24`).
- **Hinglish (calibration/dev, descriptive context):** unreviewed machine translations generated and pinned as artifacts in v21 (`runs/gpu-20261111-180320-270579d4/hs_eval_hinglish_calibration.jsonl`, 51 rows, sha256 `f8e1a0beacd4bf3e42e5faac8a3b2eb16c77797f827a89bd8e7eeb10b005301b`; `runs/gpu-20261111-180320-270579d4/hs_eval_hinglish_dev.jsonl`, 67 rows, sha256 `7b6ff1931eef95c6ac87fe5b1ad03c036ae8b3b2221e9ca80bfe0f89b68e0b8b`; lineage/journal `runs/gpu-20261111-180320-270579d4/hinglish_generation.json`, `runs/gpu-20261111-180320-270579d4/hinglish_status.json`). These rows are labeled unreviewed machine translation and are not human labels (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`).
- **HI rows in v21 descriptive eval:** bound from frozen family manifest positionally to pinned `data/seeds/aya_redteaming/aya_hin.jsonl` (literal translations, not human-reviewed), labeled unreviewed translation (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`, harm_calibration_hi 196 rows, harm_dev_hi 277 rows, manifest sha256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`).

## Preregistered protocol

- **Preregistration:** `docs/research/PREREG_baseline.md` (H1 baseline language gap, hs-h1-baseline-v1, status not measured as scored primary H1 in this draft) and frozen config `configs/research/baseline_prereg.yaml`. Protocol deviations D1-D10 and parity/guard/provenance rules are recorded in `docs/research/PREREG_baseline.md`.
- **Evaluation plan and configs:** `docs/EVAL_PLAN.md`, `configs/eval/baseline.yaml` (dev preliminary path), `configs/eval/test_locked.yaml` (single-shot locked test, prepared/pinned; not run). Locked-test preparation and guards: `docs/LOCKED_TEST_PREP.md`, `src/eval/run_baseline.py`. Primary locked pass is frozen EN+HI; Hinglish test extension is descriptive secondary per approved amendment `docs/AMENDMENT_hinglish_test_extension_2026-11-09.md` (proposal sha256 `3bfcf861efe3ff465d315c925f7970a48e46490eb01d89751236b90cbea470b0`).
- **Splits and integrity:** family-grouped calibration/dev/test per D3 and manifest above; test frozen and single-shot gated. Exposure/holdout integrity context recorded in `docs/INTEGRITY_LOG.md`, `data/splits/development_exposures.json` (see repo records); no locked test executed.
- **Judges (pinned in v21 artifacts):** WildGuard primary (`allenai/wildguard`, revision `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`) and Qwen3Guard secondary (`Qwen/Qwen3Guard-Gen-0.6B`, revision `fada3b2f655b89601929198343c94cd2f64d93cc`, chat_template_sha256 `b4230b556815baa1982c0b1eacb181ce965483ee20b1f6e4d9f375294900e02b`) as recorded in `runs/gpu-20261111-180320-270579d4/metrics.json` and `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`.
- **Metrics discipline:** no invented numbers; run artifacts carry `metrics_eligible` and `primary_asr` fields (false/null where descriptive or fixture). Descriptive judge rates are labeled "automated-judge only, human calibration pending" (`runs/gpu-20261111-180320-270579d4/d8_status.json`, `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`).

## Lexsi libraries: runtime-verified vs not

All versions below refer to pinned/project-recorded library versions and repo artifacts.

| Library | Runtime-verified in this repo | Not runtime-verified / not claimed |
|---|---|---|
| **CuratorKIT 0.1.21** (`runs/setup-smoke-2026-11-06/manifest.json`: import_ok true) | CPU clean/dedup/export pipeline wiring and configs (`src/curate/build_en_v0.py`, `configs/curate/hs_harm_en.yaml`, `configs/curate/hs_benign_en.yaml`, `data/CURATION_PLAN.md`); package import/setup smoke (`runs/setup-smoke-2026-11-06/manifest.json`); real-package fixture comparisons retained under `runs/curatorkit-attributes-fixture-2026-11-07-r3/`, `runs/curatorkit-closing-attributes-fixture-2026-11-07/`, `runs/curatorkit-quoted-attributes-fixture-2026-11-07/`. | No LLM/API generation in curation v0; no claim beyond recorded CPU curation/smoke/fixture evidence. |
| **AuditKIT 1.1.3** (`runs/setup-smoke-2026-11-06/manifest.json`: import_ok true, echo_resolve_ok true) | Package import/smoke and adapter/contract integration in project code and notebook judging paths (`runs/setup-smoke-2026-11-06/manifest.json`; judging recorded in `runs/gpu-20261111-180320-270579d4/d8_descriptive.json`, `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json` with WildGuard/Qwen3Guard rendered/parsed as recorded). Real-model WildGuard fixture via strict adapter: `runs/wildguard-fixture-20261107T141335276834Z/` (artifact_kind fixture, split example, 2 English rows, both safe responses; metrics null per `runs/wildguard-fixture-20261107T141335276834Z/metrics.json`; environment `runs/wildguard-fixture-20261107T141335276834Z/environment.json`). | Fixture is not judge accuracy, not an unsafe-response positive control, not multilingual calibration, not ASR (`runs/wildguard-fixture-20261107T141335276834Z/manifest.json`, `runs/wildguard-fixture-20261107T141335276834Z/metrics.json`). No human-calibrated ASR claim. |
| **CircuitKIT 0.2.0** (`runs/setup-smoke-2026-11-06/manifest.json`: import_ok true) | Package import/smoke and offline compat subset (25 passed, real-weight KL parity not run) recorded in `runs/setup-smoke-2026-11-06/manifest.json`; provisional bf16 locate/steering executed in Kaggle runs producing locate artifacts (v18: `runs/gpu-20261110-011925-83c016e0/locate_direction.json`, `runs/gpu-20261110-011925-83c016e0/repair_freeze.json`; v21: `runs/gpu-20261111-180320-270579d4/locate_direction.json`, `runs/gpu-20261111-180320-270579d4/locate_status.json`) using ActivationSteering difference-of-means at MLP 0 / `blocks.0.hook_mlp_out` (provisional, no original fp32 credit). | Original fp32 short+long parity credit incomplete (`runs/gpu-20261111-180320-270579d4/metrics.json`: original_parity status incomplete; `runs/kaggle-parity-v8-2026-11-07/parity_original_sequential.json`: ran false, status incomplete; `runs/gpu-20261108-234246-dd685561/parity_original_sequential.json`: ran false, status incomplete). bf16 variant results are provisional and not original parity credit (`runs/kaggle-parity-v8-2026-11-07/parity_variant_bf16.json`, `runs/gpu-20261111-180320-270579d4/metrics.json`). No weight-backed Locate/Repair acceptance claim. |
| **SafeTune 0.1.12** (`runs/setup-smoke-2026-11-06/manifest.json`: import_ok true, apis_ok include SafeGradTrainer/RefusalDirectionModel/apply_resta) | Package import/smoke and API availability checks in setup (`runs/setup-smoke-2026-11-06/manifest.json`). | No SafeTune LoRA/fine-tune or weight-editing repair run recorded as an accepted scored runtime artifact in this repo. The executed repair experiment in v21 used provisional CircuitKIT ActivationSteering (`runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`, `runs/gpu-20261110-011925-83c016e0/repair_freeze.json`), not a SafeTune training/fine-tune run. |

## D8 automated-judge dev baseline (descriptive)

Source run: `runs/gpu-20261111-180320-270579d4/` (archive sha256 `a0d8c02fabf31ad4e1240bc00510aba4ad35db71dc583236beb3b6d68985c0ca`, `runs/gpu-20261111-180320-270579d4/metrics.json`; run_id `20261111-180320-270579d4`, `runs/gpu-20261111-180320-270579d4/d8_status.json`). Split: calibration+dev, descriptive only. Status: ok, 1362 rows (`runs/gpu-20261111-180320-270579d4/d8_status.json`); languages en 771, hi 473, hinglish 118 (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`, `runs/gpu-20261111-180320-270579d4/metrics.json`). `artifact_kind` descriptive, `scored_baseline` false, `metrics_eligible` false, `primary_asr` null, `human_calibration` pending, label "automated-judge only, human calibration pending" (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`). HI rows are unreviewed translation; Hinglish rows are unreviewed machine translation (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`).

Generation/judging context (as recorded): target `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324`, greedy, max_new_tokens 256, dtype float16_cuda; WildGuard primary and Qwen3Guard secondary as pinned above (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`, `runs/gpu-20261111-180320-270579d4/metrics.json`).

Selected per-slice descriptive rates (WildGuard / Qwen3Guard unsafe_rate, counts from `runs/gpu-20261111-180320-270579d4/d8_descriptive.json`):
- benign_calibration (n 92): WG 0.05434782608695652 (5/92), Qwen 0.10869565217391304 (10/92)
- benign_dev (n 245): WG 0.0163265306122449 (4/245), Qwen 0.07346938775510205 (18/245)
- harm_calibration (n 138): WG 0.2028985507246377 (28/138), Qwen 0.2028985507246377 (28/138)
- harm_dev (n 296): WG 0.12162162162162163 (36/296), Qwen 0.15202702702702703 (45/296)
- harm_calibration_hi (n 196): WG 0.00510204081632653 (1/196), Qwen 0.10714285714285714 (21/196)
- harm_dev_hi (n 277): WG 0.0036101083032490976 (1/277), Qwen 0.061371841155234655 (17/277)
- harm_calibration_hing (n 51): WG 0.0392156862745098 (2/51), Qwen 0.21568627450980393 (11/51)
- harm_dev_hing (n 67): WG 0.07462686567164178 (5/67), Qwen 0.13432835820895522 (9/67)

These are descriptive automated-judge rates, not human-calibrated ASR.

## CircuitKIT Locate and steering repair

- **Provisional locate (v18/v21, calibration fit):** difference-of-means direction via CircuitKIT ActivationSteering on calibration harm vs benign prompt texts, window_tokens 16, node MLP 0 / hook `blocks.0.hook_mlp_out` (nearest supported node for protocol layer 0, not a discovered `blocks.0.hook_resid_post` circuit), declared score 1.0 as routing pin, dtype bfloat16, provisional true, original_fp32_credit false (`runs/gpu-20261110-011925-83c016e0/locate_direction.json`, `runs/gpu-20261111-180320-270579d4/locate_direction.json`, `runs/gpu-20261111-180320-270579d4/locate_status.json`: l2_norm 3.0015764236450195, status ok).
- **Controls (v18 freeze artifact):** base no-intervention ran (n_check 18); alpha_zero_with_wrapper matched base (max_abs_logit_diff 0.0, ran true); candidate_alpha_1 changed logits (max_abs_logit_diff 4.9375, ran true); random_norm_matched_seed_7 changed logits (max_abs_logit_diff 3.765625, norm_gap 1.588622413365271e-07, random_l2_norm 3.0015769004821777, ran true) (`runs/gpu-20261110-011925-83c016e0/repair_freeze.json`).
- **Frozen provisional repair arm:** `runs/gpu-20261110-011925-83c016e0/repair_freeze.json` (schema `provisional_bf16_freeze_v1`, alpha 1.0, direction_sha256 `d5ec351a001fd36a05d267bc8ebe9bb1cf9e4f44720fdc0b85492ebc2b23c8c1`, content_sha256 `6fc30b183cd5e89676b6cf403646a4c54f567af408fe51d1563ad7cf4a6dbf88`, provisional true, metrics_eligible false, primary_asr null, no circuit/deployable/safetune superiority claims per artifact).
- **Parity context:** original fp32 parity incomplete in recorded runs; provisional bf16 variant is not original fp32 credit (`runs/gpu-20261111-180320-270579d4/metrics.json`, `runs/kaggle-parity-v8-2026-11-07/parity_original_sequential.json`, `runs/kaggle-parity-v8-2026-11-07/parity_variant_bf16.json`, `runs/gpu-20261108-234246-dd685561/parity_original_sequential.json`, `runs/gpu-20261108-234246-dd685561/parity_variant_bf16.json`).

## Dev repair evaluation (v21): honest null result (no safety claim)

Source: `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`, `runs/gpu-20261111-180320-270579d4/dev_repair_status.json`, `runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md`, `runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.json`. Dev split only. 885 dev rows (benign 245, harm 640; en 541, hi 277, hinglish 67), 4 arms x 885 generation and judge rows (`runs/gpu-20261111-180320-270579d4/dev_repair_status.json`, `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`). Automated judges only, human calibration pending, hi/hinglish unreviewed machine translations, metrics_eligible false, primary_asr null.

- **Alpha-zero byte-match control:** alpha_zero matched base byte-for-byte on all 885 dev rows (responses and token ids identical; mismatch count 0 as recorded in run evaluation/log context and consistent with identical alpha_zero/base rates in `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`). Alpha-zero hook installed with alpha 0.0 (`runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`, arm alpha_zero hook_count 13991).
- **Candidate vs random vs base (overlapping CIs):** Wilson 95% intervals for `candidate_alpha_1`, `random_norm_matched_seed_7`, and `base` overlap under both WildGuard (primary) and Qwen3Guard (secondary) for harm unsafe rate and benign over-refusal (benign unsafe rate per plan). No separation observed; point estimates show no consistent improvement. No safety or circuit-superiority claim.

Pooled dev rates (Wilson 95%, judge-scored denominators; `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`):

| Arm | WG harm unsafe (640) | WG benign over-refusal/unsafe (245) | Qwen harm unsafe (640) | Qwen benign over-refusal/unsafe (245) |
|---|---:|---:|---:|---:|
| base | 0.0671875 [0.05026070966858178, 0.08927901333287597] (43/640) | 0.0163265306122449 [0.006366914749716127, 0.041219443555740684] (4/245) | 0.1078125 [0.0860820819600923, 0.13422286559718533] (69/640) | 0.07346938775510205 [0.0469751586685129, 0.11313264253756505] (18/245) |
| candidate_alpha_1 | 0.0703125 [0.05296158477046814, 0.09279084781220653] (45/640) | 0.02040816326530612 [0.008747889362498967, 0.046875714864430776] (5/245) | 0.121875 [0.09876120615180303, 0.14950093452095065] (78/640) | 0.06938775510204082 [0.043770302719909, 0.108300252564696] (17/245) |
| random_norm_matched_seed_7 | 0.0734375 [0.05567340431560121, 0.09629173784829036] (47/640) | 0.00816326530612245 [0.002241523839785705, 0.029270342622725227] (2/245) | 0.115625 [0.0931117947974897, 0.14272492671283016] (74/640) | 0.06938775510204082 [0.043770302719909, 0.108300252564696] (17/245) |
| alpha_zero | 0.0671875 [0.05026070966858178, 0.08927901333287597] (43/640) | 0.0163265306122449 [0.006366914749716127, 0.041219443555740684] (4/245) | 0.1078125 [0.0860820819600923, 0.13422286559718533] (69/640) | 0.07346938775510205 [0.0469751586685129, 0.11313264253756505] (18/245) |

Per-language breakdown and small-cell flags (hinglish harm n 67 per arm; benign dev EN only, n 245): `runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md`, `runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.json`.

## Locked test and Hinglish extension (placeholders, not run)

Prepared but not executed. Notebook v22 saved with `--no-run`: <https://www.kaggle.com/code/vedant817/hinglish-shield-gpu-parity-and-baseline> (kernel `vedant817/hinglish-shield-gpu-parity-and-baseline`). No locked-test or extension numbers exist.

| Item | Placeholder |
|---|---|
| Locked test run id | {{LOCKED_TEST_RUN_ID}} |
| Locked test split / config | {{LOCKED_TEST_SPLIT}} (`configs/eval/test_locked.yaml`, arms `base, refuse_everything, frozen_repair`) |
| ASR (base, EN) | {{LOCKED_TEST_ASR_BASE_EN}} |
| ASR (base, HI) | {{LOCKED_TEST_ASR_BASE_HI}} |
| ASR (base, pooled EN+HI) | {{LOCKED_TEST_ASR_BASE_POOLED}} |
| ASR (refuse_everything, EN) | {{LOCKED_TEST_ASR_REFUSE_EVERYTHING_EN}} |
| ASR (refuse_everything, HI) | {{LOCKED_TEST_ASR_REFUSE_EVERYTHING_HI}} |
| ASR (refuse_everything, pooled EN+HI) | {{LOCKED_TEST_ASR_REFUSE_EVERYTHING_POOLED}} |
| ASR (frozen_repair, EN) | {{LOCKED_TEST_ASR_FROZEN_REPAIR_EN}} |
| ASR (frozen_repair, HI) | {{LOCKED_TEST_ASR_FROZEN_REPAIR_HI}} |
| ASR (frozen_repair, pooled EN+HI) | {{LOCKED_TEST_ASR_FROZEN_REPAIR_POOLED}} |
| Over-refusal (base, EN/HI/pooled) | {{LOCKED_TEST_OVER_REFUSAL_BASE_EN}}, {{LOCKED_TEST_OVER_REFUSAL_BASE_HI}}, {{LOCKED_TEST_OVER_REFUSAL_BASE_POOLED}} |
| Over-refusal (refuse_everything, EN/HI/pooled) | {{LOCKED_TEST_OVER_REFUSAL_REFUSE_EVERYTHING_EN}}, {{LOCKED_TEST_OVER_REFUSAL_REFUSE_EVERYTHING_HI}}, {{LOCKED_TEST_OVER_REFUSAL_REFUSE_EVERYTHING_POOLED}} |
| Over-refusal (frozen_repair, EN/HI/pooled) | {{LOCKED_TEST_OVER_REFUSAL_FROZEN_REPAIR_EN}}, {{LOCKED_TEST_OVER_REFUSAL_FROZEN_REPAIR_HI}}, {{LOCKED_TEST_OVER_REFUSAL_FROZEN_REPAIR_POOLED}} |
| Judge agreement / kappa (locked test) | {{LOCKED_TEST_KAPPA}} |
| Wilson CIs / uncertainty (locked test) | {{LOCKED_TEST_WILSON_CI_SUMMARY}} |
| Locked test artifact / checkpoint | {{LOCKED_TEST_ARTIFACT_PATH}} |
| Hinglish extension run id / artifact | {{HINGLISH_EXTENSION_RUN_ID}}, {{HINGLISH_EXTENSION_ARTIFACT_PATH}} |
| Hinglish extension rows (generated/kept/dropped) | {{HINGLISH_EXTENSION_ROWS_GENERATED}}, {{HINGLISH_EXTENSION_ROWS_KEPT}}, {{HINGLISH_EXTENSION_ROWS_DROPPED}} |
| Hinglish extension rates (base/refuse_everything/frozen_repair) | {{HINGLISH_EXTENSION_RATE_BASE}}, {{HINGLISH_EXTENSION_RATE_REFUSE_EVERYTHING}}, {{HINGLISH_EXTENSION_RATE_FROZEN_REPAIR}} |
| Hinglish extension judge agreement / CIs | {{HINGLISH_EXTENSION_KAPPA}}, {{HINGLISH_EXTENSION_WILSON_CI_SUMMARY}} |

All Hinglish extension outputs are unreviewed machine translation, `metrics_eligible` false, `primary_asr` null, descriptive secondary only (approved amendment `docs/AMENDMENT_hinglish_test_extension_2026-11-09.md`). Frozen EN+HI locked-test configuration remains the primary H1 result.

## Limitations

- Automated judges only; human calibration pending. No human-calibrated ASR claim.
- HI and Hinglish rows used in descriptive/dev contexts are unreviewed (literal or machine) translations, not human labels.
- Dev repair result is dev-split only, descriptive, and null; no safety-effect or circuit-superiority claim.
- CircuitKIT locate/repair evidence is provisional bf16 (no original fp32 parity credit). Original fp32 parity remains incomplete in recorded artifacts.
- Locked test and Hinglish extension have not been run; placeholders above are not results.
- Fixture/smoke artifacts (e.g. `runs/_example/metrics.json`, `runs/wildguard-fixture-20261107T141335276834Z/metrics.json`) are not ASR or scored baselines.

## Reproducibility

- Repository: <https://github.com/Vedant817/hinglish-shield>
- Key docs: `README.md`, `docs/RESULTS.md`, `docs/GPU_RUNBOOK.md`, `docs/KAGGLE_AGENT.md`, `docs/LOCKED_TEST_PREP.md`, `docs/EVAL_PLAN.md`, `docs/research/PREREG_baseline.md`, `docs/AMENDMENT_hinglish_test_extension_2026-11-09.md`, `data/CURATION_PLAN.md`
- Frozen split/manifest: `data/splits/family_splits_v2.json` (sha256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`)
- Model/identity (v21): `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324`, chat_template_sha256 `2147c474f41b522b7296f74692b9f31e45f49a31465960576a9d321339c5915a` (`runs/gpu-20261111-180320-270579d4/metrics.json`)
- Key run ids/artifacts: v21 dev eval `runs/gpu-20261111-180320-270579d4/` (metrics `runs/gpu-20261111-180320-270579d4/metrics.json`, dev eval `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`, d8 `runs/gpu-20261111-180320-270579d4/d8_descriptive.json`); v18 locate/freeze `runs/gpu-20261110-011925-83c016e0/` (`repair_freeze.json`, `locate_direction.json`); parity/guard evidence `runs/kaggle-parity-v8-2026-11-07/`, `runs/gpu-20261108-234246-dd685561/`, `runs/kaggle-parity-v12-2026-11-08/`; WildGuard fixture `runs/wildguard-fixture-20261107T141335276834Z/`; setup smoke `runs/setup-smoke-2026-11-06/`.
- Prepared locked-test notebook (not run): v22 `--no-run`, <https://www.kaggle.com/code/vedant817/hinglish-shield-gpu-parity-and-baseline>
- Validation: `python scripts/smoke.py` and `pytest tests/` (see `docs/RESULTS.md` / PR records for recent counts; no scores invented).

No em dashes. All cited numbers are read from the referenced artifact paths.
