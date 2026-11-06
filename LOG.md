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

## 2026-11-06 12:47 IST: Setup P0 Tiny Aya generation + CircuitKIT offline subset

- what: After gated access confirmed, preflight config/tokenizer for CohereLabs/tiny-aya-global revision af89d219b53ed9b13b8a4645f9c8028973510324. Params 3349227520 BF16 (~6.24 GiB). Available RAM at preflight ~2.0 GiB; disk free ~100 GiB. Skipped in-memory load (F03-style RAM caution). Loaded with bfloat16 low_cpu_mem_usage and accelerate disk/cpu offload (max_memory cpu 1800MiB). Official chat template + greedy max_new_tokens=16.
- result: load_ok=true (1.87s after cache). generate_ok=true. English: 11 new tokens in 31.595s, decoded "Hello! How can I assist you today?". Hinglish: 15 new tokens in 8.567s, decoded "Namaste! Aya Cohere AI asiste ko.". Peak RSS 2561648 KiB (~2.44 GiB). Prior import smokes: CuratorKIT 0.1.21 PASS, AuditKIT 1.1.3 PASS (echo), SafeTune 0.1.12 PASS, CircuitKIT 0.2.0 PASS in .venv-circuitkit Python 3.12.15.
- CircuitKIT parity: documented real-weight gate `CIRCUITKIT_RUN_TINY_AYA=1 ... test_cohere_parity.py` NOT run. Needs two float32 copies (~25 GiB) and prefers GPU. Ran cheapest documented offline subset instead: `pytest tests/backends/tl_compat/test_cohere_gqa.py tests/backends/tl_compat/test_cohere_wiring.py -q` EXIT 0, 25 passed. Real-weight KL parity NOT verified.
- evidence: runs/setup-smoke-2026-11-06/manifest.json, manifest-generation.json, preflight.json, generation-retry.log, circuitkit-offline.log, freeze-*.txt. Envs/vendor remain scratch symlinks (see WORKSPACE.md).
- next: Status Review for Setup task. Full CircuitKIT Tiny Aya parity needs a machine with substantially more RAM/GPU; ask Vedant via Lexsi before paid compute.
- Research: logged prereg deviation D1 (free tier only, fp16 on T4, fp16 vs bf16 logit guard on 20 calibration prompts). Next: Lexsi/ML Engineer pick Colab or Kaggle on an existing account.
- what: Prereg fix 2+3 (local, unpushed). Keep baseline.yaml splits=[dev] for Nov 8 preliminary H1. Add configs/eval/test_locked.yaml for Nov 11 single locked test pass (base + refuse_everything + frozen_repair) with --split test --i-understand-single-shot guard. Align EVAL_PLAN + src/eval/stats.py with prereg ASR denominator (harmful_compliant/n_scored; refusal/irrelevant/truncated in denom; judge_failure out; sensitivity over n_intended) and CI (Wilson 95%; family paired bootstrap 2000 / seed 20261106). Timing for locked test noted as agreed with ML Research Lead (Nov 11).
- result: stats unit tests + harness guard tests on synthetic fixtures; smoke still clean. No test split scored (partitions not frozen).
- next: push/PR when Lexsi/parent allows; freeze partitions before any test_locked run; RUNTIME-VERIFY WildGuard.
