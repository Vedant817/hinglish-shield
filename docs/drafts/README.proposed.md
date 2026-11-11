# Hinglish Shield: proposed status text

Local reversible draft. Do not replace the public README or publish without review.

Tiny Aya Global financial-support safety audit and gated repair acceptance pack for English, Hindi and Hinglish.

Measured results: **Not run**. No eligible dev ASR, multilingual judge calibration, original fp32 parity pass or measured repair acceptance is claimed. Do not copy parity KL values here. They are not in a `metrics.json` with `artifact_kind: run`.

Verified scope includes CuratorKIT preservation fixtures, English AuditKIT/WildGuard checkpoint load/parsing, CircuitKIT tensor-contract checks and bounded dependency evidence. The CPU bf16 files under `runs/kaggle-parity-v8-2026-11-07/` are a provisional variant, not ASR. SafeTune import evidence is not training evidence.

Target chat-template pin: **Not run**. `runs/tokenizer-metadata-v11-2026-11-08/` has `failure_phase` `secret_missing` and an unauthenticated HTTP 401. The Kaggle CLI cannot attach `HF_TOKEN`. D8 allows a future dev baseline to be automated-judge only, labeled "automated-judge only, human calibration pending". That baseline has not started. Unreviewed rows are not human labels.

Next prerequisites: an attached existing `HF_TOKEN` in the Kaggle editor, the tokenizer pin, explicit translation accepts with lineage, and human calibration before any human-label claim. The corrected holdout candidate is accepted only as an inactive artifact. Locked evaluation cannot run before Nov 11 IST or before the remaining gates pass. The frozen arm is a provisional bf16 research arm chosen only on its calibration logit effect; it carries no safety-effect or circuit-superiority claim.

Evidence anchors: `runs/dev-readiness-2026-11-07/`, `runs/secondary-adapter-fixture-local-20261107-parent/`, and `runs/postmerge-readiness-2026-11-07/`. These are readiness/fixture evidence, not eligible benchmark results. Results remain **Not run** until admission writes traceable run-kind metrics and a matching aggregation sidecar.
