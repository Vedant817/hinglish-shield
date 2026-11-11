# Results

This document records descriptive, automated-judge results for Hinglish Shield. It does not claim a scored baseline, human-calibrated attack success rate (ASR), safety improvement, or circuit superiority. No locked test has been run or scored.

## Dev repair evaluation (v21, run 20261111-180320-270579d4, dev split only)

**Verdict: clean null on dev (automated judges only).** No safety claim.

- Run: `runs/gpu-20261111-180320-270579d4/` (ingested). Dev split only. 885 dev rows (benign 245, harm 640; by language en 541, hi 277, hinglish 67). Benign dev rows are English only.
- Arms (4, 885/885 generation and judge rows each): `base`, `candidate_alpha_1` (frozen v18 provisional bf16 steering arm, alpha 1.0, MLP 0 / `blocks.0.hook_mlp_out` equivalent, direction `d5ec351a001fd36a05d267bc8ebe9bb1cf9e4f44720fdc0b85492ebc2b23c8c1`, freeze `6fc30b183cd5e89676b6cf403646a4c54f567af408fe51d1563ad7cf4a6dbf88`, `CohereLabs/tiny-aya-global` rev `af89d219b53ed9b13b8a4645f9c8028973510324`), `random_norm_matched_seed_7`, `alpha_zero`.
- Hook control: `alpha_zero` matched `base` byte-for-byte on all 885 dev rows (0 mismatches in responses and token ids).
- Comparison: Wilson 95% confidence intervals for `candidate_alpha_1`, `random_norm_matched_seed_7`, and `base` overlap substantially under both judges (WildGuard primary, Qwen3Guard secondary) on harm unsafe rate and benign over-refusal (benign unsafe rate per plan; benign refusal rate recorded alongside in source artifacts). No separation is observed; point estimates show no consistent improvement for the candidate over the random control or base.
- Inter-judge agreement (Cohen's kappa, rows where both judges gave a binary scored verdict, per arm all-dev): base 0.550 (n=867), candidate 0.560 (n=869), random 0.574 (n=874), alpha_zero 0.556 (n=866). Modest agreement; automated-judge labels are not human calibration.

### Pooled dev rates (WildGuard primary, Wilson 95%, judge-scored denominators)

| Arm | Harm unsafe rate | Benign over-refusal (unsafe rate) | Benign refusal rate |
|---|---:|---:|---:|
| base | 0.0672 [0.0503, 0.0893] (43/640) | 0.0163 [0.0064, 0.0412] (4/245) | 0.1796 [0.1366, 0.2325] (44/245) |
| candidate_alpha_1 | 0.0703 [0.0530, 0.0928] (45/640) | 0.0204 [0.0087, 0.0469] (5/245) | 0.1755 [0.1330, 0.2280] (43/245) |
| random_norm_matched_seed_7 | 0.0734 [0.0557, 0.0963] (47/640) | 0.0082 [0.0022, 0.0293] (2/245) | 0.2000 [0.1547, 0.2545] (49/245) |
| alpha_zero | 0.0672 [0.0503, 0.0893] (43/640) | 0.0163 [0.0064, 0.0412] (4/245) | 0.1796 [0.1366, 0.2325] (44/245) |

Per-language breakdown (including small hinglish harm cells, n=67 per arm): see `runs/gpu-20261111-180320-270579d4/dev_eval_language_breakdown.md`.

### Caveats (dev repair)

- Automated judges only (WildGuard primary, Qwen3Guard secondary); human calibration is pending.
- Hindi and Hinglish rows are unreviewed machine translations.
- Dev split only. `metrics_eligible` is false and `primary_asr` is null in ingested artifacts.
- The frozen candidate is a provisional bf16 research arm chosen on calibration logit effect only. It is not original fp32 parity credit.
- No safety-effect or circuit-superiority claim follows from these results.

## Locked test (frozen EN+HI primary, single shot) - PLACEHOLDER, NOT YET RUN

Status: **not run / no results**. No locked-test numbers are available. Do not treat placeholders as results.

Prepared (not executed): a single Kaggle Save & Run All notebook pass (GPU T4 x2, `HF_TOKEN`) is saved as version 22 (`--no-run`) at <https://www.kaggle.com/code/vedant817/hinglish-shield-gpu-parity-and-baseline>. It is prepared to run the frozen EN+HI locked test once with preregistered arms from `configs/eval/test_locked.yaml` (`base`, `refuse_everything`, `frozen_repair`) behind the existing `locked_test_gate`, then the Hinglish test extension as descriptive secondary, with WildGuard and Qwen3Guard judges (kappa and Wilson CIs) and a checkpoint zip. No execution has been started.

| Item | Value |
|---|---|
| Run id | PLACEHOLDER (not run) |
| Split | test (frozen EN+HI primary) |
| Arms | `base`, `refuse_everything`, `frozen_repair` (preregistered) |
| Primary ASR / harm unsafe (per arm, by language, pooled) | PLACEHOLDER (not run) |
| Over-refusal / benign rates (per arm, by language, pooled) | PLACEHOLDER (not run) |
| Judge agreement (kappa, Wilson CIs) | PLACEHOLDER (not run) |
| Artifact / checkpoint zip | PLACEHOLDER (not run) |
| Notes | Automated judges only; human calibration pending. No safety or circuit-superiority claim. |

## Hinglish test extension (descriptive secondary) - PLACEHOLDER, NOT YET RUN

Status: **not run / no results**. No extension numbers are available. Do not treat placeholders as results.

Per the approved amendment: generated in-notebook from frozen test-split HI harm families with pinned generator (`Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, greedy, seed 42) and same automated checks/drop policy; every row is unreviewed machine translation, `metrics_eligible` false, `primary_asr` null, descriptive secondary only. The frozen EN+HI locked-test configuration remains the primary H1 result.

| Item | Value |
|---|---|
| Run id | PLACEHOLDER (not run) |
| Rows generated / kept / dropped | PLACEHOLDER (not run) |
| Per-arm descriptive rates (harm, hinglish) | PLACEHOLDER (not run) |
| Artifact | PLACEHOLDER (not run) |
