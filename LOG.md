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

## 2026-11-06 13:09 IST: Parity copy provenance + long-prompt gate

- what: Notebook records CircuitKIT SHA e71d5e17..., prompts/KL/long-spec SHA-256, and adds the 4196-token sliding-window case to the sequential fp32 original gate (short and long KL both required; missing long => incomplete). Added scripts/verify_parity_copy.py; output in runs/setup-smoke-2026-11-06/parity-copy-verify.txt.
- result: verify script PASS. Gate dry-run: long missing => incomplete; both OK => pass; long bad => fail. nbformat OK.
- next: comment verify output on PR #10; Vedant still runs on Kaggle.
- what: ML Research Lead wrote H1 baseline preregistration (docs/research/PREREG_baseline.md, configs/research/baseline_prereg.yaml) and GPU options memo (docs/research/compute_options.md) on branch research/baseline-prereg.
- result: Prereg frozen locally, not measured. Split ratio and baseline-split discrepancies with CURATION_PLAN.md and configs/eval/baseline.yaml recorded, not overridden. No compute spent.
- next: Lead + Data Engineer decide family-grouped split; pin model revision and chat template hash at run time; Vedant picks compute option.
- Research: logged prereg deviation D1 (free tier only, fp16 on T4, fp16 vs bf16 logit guard on 20 calibration prompts). Next: Lexsi/ML Engineer pick Colab or Kaggle on an existing account.
