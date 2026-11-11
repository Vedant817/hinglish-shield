# Submission checklist (Lexsi Track 2)

## Ready (artifact grounded)
- [x] Repo clean structure, tests passing baseline recorded (recent full suite: 460 passed, 7 subtests passed per `docs/RESULTS.md`/PR history).
- [x] Frozen family splits: `data/splits/family_splits_v2.json` (sha256 `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`).
- [x] Preregistered protocol: `docs/EVAL_PLAN.md`, `configs/eval/test_locked.yaml`, `docs/research/PREREG_baseline.md`, `docs/LOCKED_TEST_PREP.md`.
- [x] Hinglish extension amendment approved: `docs/AMENDMENT_hinglish_test_extension_2026-11-09.md` (proposal sha256 `3bfcf861efe3ff465d315c925f7970a48e46490eb01d89751236b90cbea470b0`).
- [x] Model pinned: `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324` (`runs/gpu-20261111-180320-270579d4/metrics.json`).
- [x] Judges pinned: WildGuard `allenai/wildguard` rev `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`, Qwen3Guard `Qwen/Qwen3Guard-Gen-0.6B` rev `fada3b2f655b89601929198343c94cd2f64d93cc` (`runs/gpu-20261111-180320-270579d4/metrics.json`).
- [x] D8 descriptive baseline (dev): `runs/gpu-20261111-180320-270579d4/d8_descriptive.json` (automated judges only, unreviewed translations labeled).
- [x] Dev repair eval (honest null): `runs/gpu-20261111-180320-270579d4/dev_repair_eval.json`, `dev_eval_language_breakdown.md` (alpha zero byte match, overlapping CIs, no safety claim).
- [x] Provisional locate/freeze: `runs/gpu-20261110-011925-83c016e0/repair_freeze.json`, `locate_direction.json` (bf16 provisional, parity incomplete).
- [x] Lexsi runtime verification recorded: CuratorKIT 0.1.21, AuditKIT 1.1.3, CircuitKIT 0.2.0, SafeTune 0.1.12 (`runs/setup-smoke-2026-11-06/manifest.json`); fixtures in `runs/wildguard-fixture-20261107T141335276834Z/`, `runs/curatorkit-*-fixture-*/`.
- [x] Kaggle notebook prepared for locked test + Hinglish extension (v22 `--no-run`): <https://www.kaggle.com/code/vedant817/hinglish-shield-gpu-parity-and-baseline> (kernel `vedant817/hinglish-shield-gpu-parity-and-baseline`).
- [x] Docs: `docs/ARTICLE.md`, `docs/PITCH.md`, `docs/RESULTS.md`, `docs/README.md` updated with placeholders and honest framing.

## Needs Vedant
- [ ] Run v22 locked test on Kaggle (EN+HI) with preregistered arms `base, refuse_everything, frozen_repair` per `configs/eval/test_locked.yaml` and `docs/LOCKED_TEST_PREP.md`. Capture run id and checkpoint zip.
- [ ] Run Hinglish test extension (descriptive secondary) per `docs/AMENDMENT_hinglish_test_extension_2026-11-09.md` (generator pin `Qwen/Qwen2.5-1.5B-Instruct` rev `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, greedy seed 42). Ensure metrics_eligible false, primary_asr null, rows counted/dropped per policy.
- [ ] Human review verdicts (calibration) if available; update docs with human-calibrated results when present (do not guess).
- [ ] Publish artifacts to HuggingFace if intended (HF publishing) per project plan.
- [ ] Actual submission to Lexsi Track 2 (attach/run artifacts and filled placeholders).

## Placeholders to fill from locked/extension run
All must be filled from artifacts (no hand typed values). See `docs/ARTICLE.md` for full list:
{{LOCKED_TEST_RUN_ID}}, {{LOCKED_TEST_SPLIT}}, {{LOCKED_TEST_ASR_BASE_EN}}, {{LOCKED_TEST_ASR_BASE_HI}}, {{LOCKED_TEST_ASR_BASE_POOLED}}, {{LOCKED_TEST_ASR_REFUSE_EVERYTHING_EN}}, {{LOCKED_TEST_ASR_REFUSE_EVERYTHING_HI}}, {{LOCKED_TEST_ASR_REFUSE_EVERYTHING_POOLED}}, {{LOCKED_TEST_ASR_FROZEN_REPAIR_EN}}, {{LOCKED_TEST_ASR_FROZEN_REPAIR_HI}}, {{LOCKED_TEST_ASR_FROZEN_REPAIR_POOLED}}, {{LOCKED_TEST_OVER_REFUSAL_BASE_EN}}, {{LOCKED_TEST_OVER_REFUSAL_BASE_HI}}, {{LOCKED_TEST_OVER_REFUSAL_BASE_POOLED}}, {{LOCKED_TEST_OVER_REFUSAL_REFUSE_EVERYTHING_EN}}, {{LOCKED_TEST_OVER_REFUSAL_REFUSE_EVERYTHING_HI}}, {{LOCKED_TEST_OVER_REFUSAL_REFUSE_EVERYTHING_POOLED}}, {{LOCKED_TEST_OVER_REFUSAL_FROZEN_REPAIR_EN}}, {{LOCKED_TEST_OVER_REFUSAL_FROZEN_REPAIR_HI}}, {{LOCKED_TEST_OVER_REFUSAL_FROZEN_REPAIR_POOLED}}, {{LOCKED_TEST_KAPPA}}, {{LOCKED_TEST_WILSON_CI_SUMMARY}}, {{LOCKED_TEST_ARTIFACT_PATH}}, {{HINGLISH_EXTENSION_RUN_ID}}, {{HINGLISH_EXTENSION_ARTIFACT_PATH}}, {{HINGLISH_EXTENSION_ROWS_GENERATED}}, {{HINGLISH_EXTENSION_ROWS_KEPT}}, {{HINGLISH_EXTENSION_ROWS_DROPPED}}, {{HINGLISH_EXTENSION_RATE_BASE}}, {{HINGLISH_EXTENSION_RATE_REFUSE_EVERYTHING}}, {{HINGLISH_EXTENSION_RATE_FROZEN_REPAIR}}, {{HINGLISH_EXTENSION_KAPPA}}, {{HINGLISH_EXTENSION_WILSON_CI_SUMMARY}}.
