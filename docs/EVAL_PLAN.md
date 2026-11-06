# Eval plan (Track 2 baseline)

Times and deadlines in IST unless noted. No invented metric numbers. Every claim needs a `runs/<id>/metrics.json` with `artifact_kind: run`.

## Claimed tasks (Eval Engineer)

1. Set up WildGuard (primary) + one LLM judge (secondary) + human sample for Cohen kappa.
2. Baseline eval per language with AuditKIT on HS-Harm / HS-Benign (ASR, StrongREJECT, over-refusal, capability, language gap).
3. After-eval + leaderboard (later; depends on repair).

Baseline metrics due Nov 8 IST. Today (Nov 6) is Setup.

## Protocol steps for baseline

1. Freeze datasets under `data/curated/hs_harm` and `data/curated/hs_benign` with a `dataset_hash`.
2. Pin `model_id` + `model_revision` and judges in `configs/eval/baseline.yaml` / `configs/eval/judges.yaml`.
3. RUNTIME-VERIFY WildGuard via AuditKIT `GuardJudge(profile="wildguard")` on a tiny fixture batch (fixture artifacts only).
4. Run per language (EN, HI, Hinglish) on frozen config; write one metrics.json per run row (or clearly keyed multi-language rows).
5. Fill Protocol fields: `run_id`, `full_commit`, `config_hash`, `model_revision`, `dataset_hash`, `method`, languages, `split`, `attack`, `query_budget`, `seed`, counts (`n_intended`, `n_generated`, `n_judged`, `n_errors`, `harmful_count`), scores + CIs, `judge_version`, latency/throughput if measured, `artifact_uri`.
6. Nulls allowed only with `null_reasons`. Never placeholder scores.
7. Headline: `language_gap = ASR(Hinglish) - ASR(EN)` on the same method/split.

## StrongREJECT package note

Use the maintained repo **dsbowen/strong_reject**.

Do **not** use deprecated **alexandrasouly/strongreject**.

Store the mean in `metrics.strongreject_mean` (legacy `metrics.strongreject` kept for older readers).

## Kappa plan

- Primary automatic: WildGuard.
- Secondary automatic: one LLM judge (model pinned in `configs/eval/judges.yaml` after compute/approval).
- Human: stratified sample by language and attack family; size set after `n_judged` is known (target about 100).
- Report agreement and Cohen kappa via `src.eval.kappa_report` / `cohens_kappa`. Degenerate all-one-class sets return null kappa, not 0.0.
- Human labels are the adjudication reference for judge quality, not a substitute for full-set ASR.

## Integrity

- `runs/_example` is a **fixture** (`artifact_kind: fixture`). Not a benchmark run.
- Mocks elsewhere must be named fixtures; Space `MOCK_*` constants are placeholders until replaced by real metrics.json loads.
- `python3 src/eval/run_baseline.py` refuses to invent numbers when datasets are missing.

## Local stubs added

- `src/eval/judges.py` (interfaces; UnavailableError until RUNTIME-VERIFIED)
- `src/eval/metrics_io.py` (load/validate)
- `src/eval/run_baseline.py` (CLI stub)
- `configs/eval/judges.yaml`, `configs/eval/baseline.yaml`
- Expanded `schemas/metrics.schema.json`
