# Hinglish Shield run log

Format per iteration: what, result, next. Times in IST. No em dashes.

## 2026-11-06

- what: Create public repo skeleton (Makefile, metrics schema, LOG.md, src layout, data/seeds placeholder).
- result: Starter files on GitHub main; `make smoke` target added.
- next: Data Engineer copies seeds onto main; ML Engineer lib smoke after registration confirmed.

- 2026-11-06 (IST): Canonical clone is /workspace/hinglish-shield (see WORKSPACE.md). Scratch venvs at /workspace/hinglish-shield-scratch. Seed PR #1 open.

- what: Expand metrics schema to Experiment Protocol fields; add eval stubs (judges, metrics_io, run_baseline), configs/eval/*.yaml, docs/EVAL_PLAN.md; mark runs/_example as artifact_kind=fixture with null metrics.
- result: Schema + example validate; smoke passes with jsonschema; baseline CLI refuses when curated datasets missing (no invented scores). StrongREJECT note: dsbowen/strong_reject only.
- next: Data freeze HS-Harm/HS-Benign; RUNTIME-VERIFY WildGuard GuardJudge after HF gate; open eval/metrics-schema-stubs PR for Lead Architect review.

- what: ML Research Lead wrote H1 baseline preregistration (docs/research/PREREG_baseline.md, configs/research/baseline_prereg.yaml) and GPU options memo (docs/research/compute_options.md) on branch research/baseline-prereg.
- result: Prereg frozen locally, not measured. Split ratio and baseline-split discrepancies with CURATION_PLAN.md and configs/eval/baseline.yaml recorded, not overridden. No compute spent.
- next: Lead + Data Engineer decide family-grouped split; pin model revision and chat template hash at run time; Vedant picks compute option.
- Research: logged prereg deviation D1 (free tier only, fp16 on T4, fp16 vs bf16 logit guard on 20 calibration prompts). Next: Lexsi/ML Engineer pick Colab or Kaggle on an existing account.
- what: Prereg fix 2+3 (local, unpushed). Keep baseline.yaml splits=[dev] for Nov 8 preliminary H1. Add configs/eval/test_locked.yaml for Nov 11 single locked test pass (base + refuse_everything + frozen_repair) with --split test --i-understand-single-shot guard. Align EVAL_PLAN + src/eval/stats.py with prereg ASR denominator (harmful_compliant/n_scored; refusal/irrelevant/truncated in denom; judge_failure out; sensitivity over n_intended) and CI (Wilson 95%; family paired bootstrap 2000 / seed 20261106). Timing for locked test noted as agreed with ML Research Lead (Nov 11).
- result: stats unit tests + harness guard tests on synthetic fixtures; smoke still clean. No test split scored (partitions not frozen).
- next: push/PR when Lexsi/parent allows; freeze partitions before any test_locked run; RUNTIME-VERIFY WildGuard.
