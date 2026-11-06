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

## 2026-11-06 12:57 IST: GPU free-tier notebook + runbook

- what: Added notebooks/gpu_parity_and_baseline.ipynb and docs/GPU_RUNBOOK.md for Kaggle free T4 x2. Original CircuitKIT float32 dual-copy parity is optional with RAM/VRAM preflight. Proposed bf16 sequential variant behind RUN_PARITY_VARIANT=False. fp16 guard vs bf16 CPU before baseline; fail switches to fp32 (D2).
- result: Notebook nbformat-validated. Fixture dry-run OK. No model load on box. No paid compute.
- next: Vedant runs on Kaggle with HF_TOKEN secret; download zip into runs/.

## 2026-11-06 13:05 IST: prereg D4 notebook update

- what: notebooks/gpu_parity_and_baseline.ipynb + docs/GPU_RUNBOOK.md aligned to D4 (99a3829). Default parity sequential fp32 CPU (original gate). Dual-resident pytest off. bf16 variant auto on OOM with floor calibration. fp16 guard: fp32 CPU ref when RAM allows, residual hooks, mean KL/top-1/greedy-32; logit diff logged only.
- result: nbformat validated; gate-helper dry-run with synthetic tensors passed. No model load on box.
- next: Vedant Run All on Kaggle; Lead Architect owns PR.
