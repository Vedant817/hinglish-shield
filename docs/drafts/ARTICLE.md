# Hinglish Shield: an audit pack before a safety claim

Local draft skeleton, not approved for publication. Numerical benchmark
results: **Not run**. Do not publish or submit without explicit approval.

Every number in this draft comes from one place only:
`runs/<run-id>/metrics.json`, written by
`scripts/ingest_gpu_artifacts.py` when a Kaggle zip is ingested. Placeholders
below name the exact field they will take. They stay empty until that file
exists for the run they describe.

## The problem

Financial-support assistants should preserve useful answers while resisting
requests for fraud enablement and privacy violations. Hinglish Shield studies
that tradeoff on Tiny Aya Global across English, Hindi and Hinglish. The
workload is synthetic generic support, not real customer data or a deployed
financial decision system.

## The pack (fixed, not results-dependent)

The pack separates source-family identity, translation review, generation,
judging, and admission of reported metrics. CuratorKIT preservation fixtures
exposed why removing formatting can also remove an attack instruction.
AuditKIT's WildGuard path and the pinned Qwen secondary adapter have real
English fixture load/parsing evidence. Neither fixture establishes accuracy
or multilingual calibration. CircuitKIT's steering interface has
tensor-contract evidence; its original model parity gate is still
incomplete. SafeTune dependency/import checks are not a trained repair
result.

## Results (populate only from metrics.json)

- Target chat-template pin: [[METRICS runs/<run-id>/metrics.json ->
  target_identity.chat_template_sha256]], identity status [[METRICS ->
  target_identity.status]].
- Original fp32 parity: [[-> original_parity.status]] and max short/long KL
  [[-> original_parity.short_kl / long_kl]]. If the long case did not run,
  the final status is incomplete, never a pass. A provisional bf16 variant
  is separate evidence, never original fp32 credit.
- D8 descriptive baseline (automated-judge only, human calibration
  pending): rows [[-> d8_descriptive.by_slice.<slice>.n]], WildGuard
  unsafe count and rate [[-> ...wildguard_unsafe_count /
  ...wildguard_unsafe_rate]], Qwen3Guard unsafe count and
  judge-failure coverage [[-> ...qwen3guard_unsafe_count /
  ...qwen3guard_judge_failure_count]]. `metrics_eligible` stays false and
  `primary_asr` stays null. Unreviewed rows are not human labels.
- Provisional bf16 locate (no fp32 credit): vector norm [[METRICS ->
  provisional_locate.l2_norm]], alpha-zero check [[METRICS ->
  provisional_locate.checks.alpha_zero_matches_base]]. The node is
  `MLP 0`, the declared score is a pin, not a discovered circuit.
- Repair freeze [[METRICS -> runs/<id>/repair_freeze.json]]: verification
  pathway and controls, never a deployable claim.

Fixture generations are not ASR. The locked single-shot test stays
embargoed until Nov 11 IST and the final publishable numbers must come from
an admitted run whose `metrics.json` carries `artifact_kind: run`.

## Evidence anchors

`runs/dev-readiness-2026-11-07/`, `runs/secondary-adapter-fixture-local-20261107-parent/`,
and `runs/postmerge-readiness-2026-11-07/` are readiness/fixture evidence,
not eligible benchmark results.
