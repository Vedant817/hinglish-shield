# Hinglish Shield pitch (Lexsi Track 2)

## One line
Hinglish Shield is a reproducible audit and repair acceptance pack for multilingual/code mixed financial support safety on Tiny Aya Global (EN/HI/Hinglish), with preregistered splits, pinned judges, and honest null results rather than unverified safety claims.

## Problem
Financial support prompts in English, Hindi, and romanized Hinglish can trigger uneven compliance across languages. Without frozen splits, pinned judges, and controls, cross language safety claims are hard to reproduce.

## Approach
- Frozen family grouped splits (`data/splits/family_splits_v2.json`, sha256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`), single shot locked test gate.
- Automated judges pinned (WildGuard primary `allenai/wildguard` rev `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`, Qwen3Guard secondary `Qwen/Qwen3Guard-Gen-0.6B` rev `fada3b2f655b89601929198343c94cd2f64d93cc`), recorded in `runs/gpu-20261111-180320-270579d4/metrics.json`.
- Lexsi libs runtime verified where evidenced: CuratorKIT 0.1.21 (CPU curation/smoke/fixtures), AuditKIT 1.1.3 (import/smoke + real model WildGuard fixture `runs/wildguard-fixture-20261107T141335276834Z/`), CircuitKIT 0.2.0 (import/smoke; provisional bf16 locate only), SafeTune 0.1.12 (import/smoke only).
- Provisional locate/steering via CircuitKIT ActivationSteering at MLP 0 / `blocks.0.hook_mlp_out` (v18 freeze `runs/gpu-20261110-011925-83c016e0/repair_freeze.json`, direction sha256 `d5ec351a001fd36a05d267bc8ebe9bb1cf9e4f44720fdc0b85492ebc2b23c8c1`); original fp32 parity incomplete.
- Dev eval only on v21 (`runs/gpu-20261111-180320-270579d4/`): 885 dev rows, alpha zero byte match 0 mismatches, candidate/random/base CIs overlap under both judges, kappa modest; no safety/circuit superiority claim.

## Evidence (dev, artifact grounded)
From `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json` (WildGuard harm unsafe, 640):
base 0.0671875 [0.05026070966858178, 0.08927901333287597] (43/640); candidate_alpha_1 0.0703125 [0.05296158477046814, 0.09279084781220653] (45/640); random_norm_matched_seed_7 0.0734375 [0.05567340431560121, 0.09629173784829036] (47/640); alpha_zero 0.0671875 [0.05026070966858178, 0.08927901333287597] (43/640). Benign over refusal overlaps similarly. HI/hinglish are unreviewed translations/machine translations; automated judges only, human calibration pending.

## What is pending (placeholders)
Locked test (EN+HI, preregistered arms base/refuse_everything/frozen_repair, `configs/eval/test_locked.yaml`) and Hinglish extension (descriptive only, per `docs/AMENDMENT_hinglish_test_extension_2026-11-09.md`) are prepared as v22 `--no-run` on Kaggle (<https://www.kaggle.com/code/vedant817/hinglish-shield-gpu-parity-and-baseline>) but not run. All locked/extension numbers are placeholders: {{LOCKED_TEST_ASR_BASE_POOLED}}, {{LOCKED_TEST_KAPPA}}, {{HINGLISH_EXTENSION_RATE_BASE}}, etc (see `docs/ARTICLE.md`).

## 3 minute demo script outline
1. 0:00-0:30 Problem and scope: EN/HI/Hinglish financial support, why splits and judges matter. Show `data/splits/family_splits_v2.json` and judge pins in `runs/gpu-20261111-180320-270579d4/metrics.json`.
2. 0:30-1:00 Data/curation: seeds, CuratorKIT pipeline, D8 descriptive baseline (`runs/gpu-20261111-180320-270579d4/d8_descriptive.json`) with unreviewed translation labels.
3. 1:00-1:30 Locate/repair: provisional CircuitKIT locate at MLP 0, controls and freeze (`runs/gpu-20261110-011925-83c016e0/repair_freeze.json`), parity status incomplete.
4. 1:30-2:00 Honest dev null: alpha zero byte match, overlapping CIs across arms, no safety claim (`runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`).
5. 2:00-3:00 Next steps and acceptance: locked test (EN+HI) and Hinglish extension prepared but pending (v22 `--no-run`), placeholders only, reproducibility paths listed.

## Call to action
Evaluate on the frozen locked test with human calibration when available; treat dev null as the measured result until scored locked test artifacts exist.
