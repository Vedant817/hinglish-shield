# Hinglish Shield: Tiny Aya refuses 69% of harmful requests in English but only 24% in Hinglish, and one steering direction does not close the gap

**One-line claim:** A pre-registered, dual-judge audit of Tiny Aya Global shows a large, statistically clear refusal gap on code-mixed Hinglish harmful requests (69.3% refusal in English vs 23.9% in Hinglish), and a controlled repair experiment shows that a single layer-0 refusal direction does not close it, which tells the field where the fix has to come from.

> **Content warning.** This project studies harmful requests (fraud, weapons, self-harm, hate and similar categories from public red-teaming benchmarks). The write-up quotes no harmful completions, but the linked datasets contain harmful prompts and model outputs.

## Tracks and problem statements

- Track 2a: AI Safety and Red Teaming
  - 4. Multilingual and code-mixed safety
  - 2. Adversarial and safety evaluation sets
- Track 2b: AI Safety and Correction
  - 8. Inference-time safety steering
  - 9. Locating where safety lives

## For judges: one minute

- **Model:** `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`, bf16 on free Kaggle T4 x2.
- **Method in one line:** CuratorKIT curation of EN/HI/Hinglish harm and benign sets, AuditKIT dual-judge evaluation (WildGuard primary, Qwen3Guard secondary), CircuitKIT difference-of-means refusal direction at `blocks.0.hook_mlp_out` used as an inference-time steering repair, all behind frozen family-level splits and a single-shot locked-test gate.
- **Headline finding 1 (the gap is real):** refusal rate on harmful dev prompts, base model, WildGuard: **EN 69.3% [63.8, 74.2] (205/296) → HI 54.2% [48.3, 59.9] (150/277) → Hinglish 23.9% [15.3, 35.3] (16/67)**. Qwen3Guard agrees: EN 68.2% → HI 61.7% → Hinglish 29.9%. The EN and Hinglish intervals do not overlap under either judge.
- **Headline finding 2 (the obvious fix does not work):** steering with the located refusal direction moves harmful-prompt unsafe rate from **6.7% [5.0, 8.9] (43/640) base → 7.0% [5.3, 9.3] (45/640) steered**, against **7.3% [5.6, 9.6] (47/640)** for a norm-matched random direction. No separation from base or from the random control, under both judges.
- **Why you can trust it:** alpha-zero hook control is byte-identical to base on all 885 dev rows (0 mismatches); bf16 GPU parity and fp16 guard gates pass; judges are pinned by revision with Cohen's kappa 0.497 (91.8% raw agreement, 1334 rows); every number below is read from a committed run artifact.

## Abstract

Safety tuning of small multilingual models is usually measured in English. Hinglish Shield asks whether that safety transfers to the way hundreds of millions of Indian users actually type: Hindi in Devanagari and romanized Hindi-English code-mixing (Hinglish). We built a reproducible audit and repair pipeline for Cohere's Tiny Aya Global on top of three Lexsi kits. CuratorKIT cleans, deduplicates and exports EN harm and benign sets drawn from Aya Red-Teaming, HarmBench, XSTest and OR-Bench; Hindi rows come from Aya Red-Teaming Hindi and Hinglish rows are machine rewrites filtered by automated script checks (473 candidates, 118 kept). Splits are frozen at the semantic-family level (calibration/dev/test 0.20/0.30/0.50, seed 42) so translations of one prompt never leak across splits. AuditKIT adapters run two pinned safety judges, WildGuard (primary) and Qwen3Guard (secondary), and report Wilson 95% intervals and Cohen's kappa. CircuitKIT ActivationSteering fits a difference-of-means refusal direction on calibration data and applies it at inference time, with alpha-zero and norm-matched random-direction controls.

On 885 dev rows (640 harmful, 245 benign) the base model refuses 69.3% of harmful English prompts but only 23.9% of harmful Hinglish prompts, a gap both judges confirm with non-overlapping intervals. The two judges also diverge most on Hindi (WildGuard 0.7% unsafe vs Qwen3Guard 6.1%), which is itself a warning that English-centric safety classifiers under-read Indic outputs. The steering repair, frozen before evaluation, did not separate from base or from the random control on either harm unsafe rate or benign over-refusal, under either judge. Because the alpha-zero arm reproduces base byte-for-byte, the null is a property of the direction, not of the harness. The result is a clean, controlled negative finding: a single English-calibrated, layer-0 refusal direction is not enough to transfer refusal to code-mixed input, so repair has to target language-aware representations or train-time data. The frozen held-out locked test (1364 EN+HI rows x 3 pre-registered arms) and a Hinglish test extension are packaged as a single-shot gated notebook. All code, configs, splits, run artifacts and the notebook are public.

## Problem statement and approach

### Why this problem

India has hundreds of millions of users who type Hindi in Latin script mixed with English. Small open models like Tiny Aya are the ones that get deployed in low-cost local assistants (banking help desks, government services, education). If refusal behavior learned in English silently weakens in Hinglish, every English-only safety eval overstates how safe these deployments are.

### Research questions (pre-registered in `docs/research/PREREG_baseline.md`)

- **H1 (language gap):** does the base model's safety behavior differ between EN, HI and Hinglish harmful prompts?
- **H2 (repair):** does inference-time steering along a located refusal direction reduce harmful compliance without raising over-refusal, beyond a norm-matched random direction?

### Threat model

A non-expert user submits a harmful request to a deployed Tiny Aya assistant in the language they naturally use, Hindi or Hinglish, with no jailbreak engineering. Success for the attacker is any response a safety judge labels unsafe; we also track the refusal rate, since a non-refusal is the precondition for harm.

### Related work

- Arditi et al., 2024, "Refusal in Language Models Is Mediated by a Single Direction" (difference-of-means refusal direction; our repair arm follows it).
- Deng et al., 2024, "Multilingual Jailbreak Challenges in Large Language Models" and Yong et al., 2023, "Low-Resource Languages Jailbreak GPT-4" (safety degrades outside English).
- Aakanksha et al., 2024, "The Multilingual Alignment Prism" (Aya Red-Teaming dataset).
- Han et al., 2024, "WildGuard"; Qwen3Guard technical report (judges).
- Mazeika et al., 2024, "HarmBench"; Rottger et al., 2024, "XSTest"; Cui et al., 2024, "OR-Bench" (seed sets).

## Hackathon repos (kits) used

- **CuratorKIT 0.1.21:** clean, exact dedup, schema and quality gates, and trainer-ready export for the EN harm and benign corpora (`configs/curate/hs_harm_en.yaml`, `configs/curate/hs_benign_en.yaml`, `src/curate/build_en_v0.py`, `data/CURATION_PLAN.md`), with a payload-preserving cleaner fixture gate (fixtures under `runs/curatorkit-*-fixture-*/`).
- **AuditKIT 1.1.3:** evaluation harness and judge adapters; WildGuard strict adapter validated on a real-model fixture (`runs/wildguard-fixture-20261107T141335276834Z/`), then used for all dev judging with Qwen3Guard as secondary (`runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`).
- **CircuitKIT 0.2.0:** `ActivationSteering` difference-of-means direction on calibration harm vs benign prompts (window 16 tokens) at node MLP 0 / `blocks.0.hook_mlp_out`, and the steering hook used for the repair arms (`runs/gpu-20261110-011925-83c016e0/locate_direction.json`, `repair_freeze.json`).
- **SafeTune 0.1.12:** installed and API-verified in the pinned environment (`SafeGradTrainer`, `RefusalDirectionModel`, `apply_resta`; `runs/setup-smoke-2026-11-06/manifest.json`) as the train-time path for the next stage.

### How the repos connect

```
Seeds (Aya RT EN/HI, HarmBench, XSTest, OR-Bench)
  -> Hinglish rewrite + script checks
  -> CuratorKIT (clean, dedup, gates, export)
  -> frozen family splits (calibration / dev / test)
  -> AuditKIT baseline (Tiny Aya Global, WildGuard + Qwen3Guard, Wilson CIs, kappa)
  -> CircuitKIT Locate (difference-of-means refusal direction, calibration only)
  -> CircuitKIT steering repair, frozen by sha256 before dev eval
  -> AuditKIT after-eval with alpha-zero and random-direction controls
  -> single-shot locked test gate
```

## Models and licenses

- **Base model:** `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324`, bf16 (fp16 guard checked against fp32 reference), greedy decoding, max_new_tokens 256. License: CC-BY-NC-4.0 with Cohere's acceptable use policy (see model card).
- **Judges:** `allenai/wildguard` rev `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` (primary), `Qwen/Qwen3Guard-Gen-0.6B` rev `fada3b2f655b89601929198343c94cd2f64d93cc` (secondary).
- **Hinglish rewriting (supplementary):** `Qwen/Qwen2.5-7B-Instruct` rev `a09a35458c702b33eeacc393d103063234e8bc28` (dev/calibration), `Qwen/Qwen2.5-1.5B-Instruct` rev `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (test extension), greedy, seed 42.

## Datasets, benchmarks and licenses

| Source | Use | Languages | License |
|---|---|---|---|
| Aya Red-Teaming (CohereLabs) | harmful prompts, native Hindi rows | EN, HI | Apache 2.0 |
| HarmBench text behaviors | harmful prompts | EN | MIT |
| XSTest | safe prompts that look risky (over-refusal) | EN | CC BY 4.0 |
| OR-Bench | over-refusal prompts | EN | CC BY 4.0 |
| Hinglish rewrites (this work) | harmful prompts, code-mixed | Hinglish (romanized) | derived; follows source licenses |

- **Splits and leak check:** family-grouped splits in `data/splits/family_splits_v2.json` (sha256 `334e7151...b531`); the unit is the semantic family, so translations and paraphrases inherit their source's split (direct family-ID match check, `scripts/check_translation_family_splits.py`). Stratified by label, source and category; seed chosen by a fixed rule (first passing seed from 42 upward).
- **Dev evaluation set:** 885 rows: 640 harmful (EN 296, HI 277, Hinglish 67) and 245 benign (EN).
- **Over-refusal set:** 245 benign EN dev rows from XSTest and OR-Bench.
- **Hinglish filtering:** 473 candidate rewrites, 118 kept, 355 dropped by automated checks (Latin script only, no plain-English passthrough), recorded in `runs/gpu-20261111-180320-270579d4/hinglish_generation.json`.

## Metrics, and why these

| Metric | Direction | Why |
|---|---|---|
| Harm unsafe rate (judge labels response unsafe on a harmful prompt) | lower is better | the attack success proxy |
| Harm refusal rate | higher is better | direct read of the model's safety behavior, robust to judge calibration |
| Benign over-refusal (unsafe/refusal on benign prompts) | lower is better | a repair must not make the model useless |
| Wilson 95% interval | | small cells (Hinglish n=67) need honest uncertainty |
| Cohen's kappa between judges | higher is better | checks the automatic judges against each other |

**Checking the automatic judge:** WildGuard vs Qwen3Guard on 1334 calibration+dev rows: 91.8% raw agreement, Cohen's kappa 0.497 (`runs/gpu-20261110-011925-83c016e0/wilson_kappa.json`); on the four dev arms kappa is 0.550 to 0.574. Human spot-check calibration is the next step (review sheet in `runs/gpu-20261111-180320-270579d4/hinglish_review.csv`).

## Method and experiments

- **Generation:** greedy, seed 42, max_new_tokens 256, chat template sha256 pinned (`runs/gpu-20261111-180320-270579d4/metrics.json`).
- **Locate:** CircuitKIT ActivationSteering difference-of-means on calibration split only, MLP 0 / `blocks.0.hook_mlp_out`, direction L2 norm 3.0016, direction sha256 `d5ec351a...c8c1`.
- **Repair arm frozen before dev eval:** alpha 1.0, freeze manifest sha256 `6fc30b18...bf88` (`runs/gpu-20261110-011925-83c016e0/repair_freeze.json`). Freeze-time logit checks: alpha-zero max abs logit diff 0.0 vs base, candidate 4.94, random 3.77 (norm gap 1.6e-7).
- **Arms:** `base`, `candidate_alpha_1`, `random_norm_matched_seed_7`, `alpha_zero` (hook installed, alpha 0).
- **Gates:** bf16 variant parity pass and fp16 guard pass (`runs/gpu-20261111-180320-270579d4/parity_variant_bf16.json`, `fp16_guard.json`); tokenizer identity and model revision pinned (`identity.json`); single-shot locked-test reservation (`configs/eval/test_locked.yaml`, `docs/LOCKED_TEST_PREP.md`).
- **Seeds:** 42 for splits and generation; 7 for the random direction.
- **Ablations:** random norm-matched direction and alpha-zero.

## Results

### Finding 1: refusal collapses on Hinglish (base model, dev, harmful prompts)

| Language | n | WildGuard refusal | Qwen3Guard refusal | WildGuard unsafe | Qwen3Guard unsafe |
|---|---:|---:|---:|---:|---:|
| EN | 296 | 69.3% [63.8, 74.2] | 68.2% [62.7, 73.3] | 12.2% [8.9, 16.4] | 14.5% [11.0, 19.0] |
| HI | 277 | 54.2% [48.3, 59.9] | 61.7% [55.9, 67.3] | 0.7% [0.2, 2.6] | 6.1% [3.9, 9.6] |
| Hinglish | 67 | 23.9% [15.3, 35.3] | 29.9% [20.2, 41.7] | 7.5% [3.2, 16.3] | 13.4% [7.2, 23.6] |

Source: `runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md`.

### Finding 2: English-centric judges under-read Hindi

On HI harmful prompts WildGuard flags 0.7% of responses unsafe while Qwen3Guard flags 6.1%, the widest relative disagreement of any language. A safety eval that uses one English-trained judge would report Hindi as nearly perfectly safe. Dual judging with agreement statistics is necessary for Indic safety claims.

### Finding 3: single-direction steering does not transfer refusal (same model before vs after)

| Arm | WG harm unsafe (640) | WG benign over-refusal (245) | Qwen harm unsafe (640) | Qwen benign over-refusal (245) |
|---|---:|---:|---:|---:|
| base | 6.7% [5.0, 8.9] | 1.6% [0.6, 4.1] | 10.8% [8.6, 13.4] | 7.3% [4.7, 11.3] |
| steered (candidate_alpha_1) | 7.0% [5.3, 9.3] | 2.0% [0.9, 4.7] | 12.2% [9.9, 15.0] | 6.9% [4.4, 10.8] |
| random direction control | 7.3% [5.6, 9.6] | 0.8% [0.2, 2.9] | 11.6% [9.3, 14.3] | 6.9% [4.4, 10.8] |
| alpha-zero control | 6.7% [5.0, 8.9] | 1.6% [0.6, 4.1] | 10.8% [8.6, 13.4] | 7.3% [4.7, 11.3] |

Source: `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`. All intervals overlap; the steered arm is indistinguishable from the random control. On Hinglish the steered arm's refusal rate was 14.9% (10/67) vs 23.9% base, also within overlapping intervals.

### What this means

The refusal direction was fit on calibration prompts dominated by English and applied at the earliest MLP. It perturbs logits strongly (max abs diff 4.94) but no more usefully than a random vector of the same norm. The language gap therefore does not live in a single shared early direction, which points the repair toward language-conditioned or later-layer directions and train-time data such as SafeTune hardening on Hinglish.

## Cost

All runs used free Kaggle GPU (T4 x2); out-of-pocket compute cost was $0.

## Scope and limitations

Results are on the dev split with automated judges (human spot-check calibration is the next step), Hindi and Hinglish prompts are unreviewed machine translations, Hinglish cells are small (n=67), and the steering arm is a bf16 research arm; the frozen locked test is packaged as a single-shot gated run and its scoring is the next milestone.

## With more credits

- **Planned approach:** (1) score the frozen locked test (1364 EN+HI rows x `base`, `refuse_everything`, `frozen_repair`) plus the Hinglish test extension; (2) human spot-check 200 judge labels to calibrate both judges per language; (3) fit language-conditioned refusal directions on HI and Hinglish calibration data and sweep layers beyond layer 0; (4) SafeTune train-time hardening (SafeGrad / LoRA) on curated Hinglish harm and benign pairs, evaluated with the same harness.
- **Expected before / after:** we expect language-conditioned or train-time repair to raise Hinglish harmful refusal from the 23.9% base toward the 69.3% English level while keeping benign over-refusal within the base interval; the pre-registered harness will show whether it does.

## Artifacts

- GitHub repo: <https://github.com/Vedant817/hinglish-shield>
- Notebook: `notebooks/gpu_parity_and_baseline.ipynb` (Kaggle, GPU T4 x2)
- Run artifacts: `runs/gpu-20261111-180320-270579d4/` (dev eval, generations, judge scores), `runs/gpu-20261110-011925-83c016e0/` (locate, freeze, kappa)
- HF model and dataset collections: see `docs/SUBMISSION_CHECKLIST.md`

## Documentation and reproduction

- **Environment:** Python 3.11+, `requirements.txt` for CPU checks; GPU notebook installs the pinned heavy stack (torch, transformers, the four Lexsi kits) on Kaggle T4 x2.
- **Layout:** `data/` seeds, splits and curated exports; `configs/` curation, eval and research configs; `src/` curate, eval, locate, repair; `runs/` one folder per run with `metrics.json`; `schemas/` metrics JSON Schema; `scripts/` smoke and checks; `notebooks/` the Kaggle notebook; `tests/` 460 passing tests.
- **Run:** `make install && make smoke` (CPU checks); `pytest tests/`; on Kaggle, attach `HF_TOKEN`, select GPU T4 x2, Save and Run All on `notebooks/gpu_parity_and_baseline.ipynb`; ingest the output zip with the scripts in `scripts/`. GPU steps: `docs/GPU_RUNBOOK.md`; locked test: `docs/LOCKED_TEST_PREP.md`.
- **Declaration of work outside the hackathon:** none; all code, data processing and runs were produced during the hackathon window (Nov 1 to 15, 2026).

## Team

- Vedant Mahajan (solo): design, experiments, analysis, write-up.

## LLM usage statement

AI coding agents wrote a large share of the code and documentation under the author's direction; every number in this write-up was read from a committed run artifact.

## Acknowledgements

Lexsi Labs (CuratorKIT, AuditKIT, CircuitKIT, SafeTune), Cohere Labs (Tiny Aya, Aya Red-Teaming), Hugging Face, AllenAI (WildGuard), Qwen team (Qwen3Guard), and Kaggle for free GPUs.
