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

- what: Handoff merge sweep. Merged #9 research prereg, #7 family splits v2, #6 translations family_id (fixed export_to_curatorkit newline), #8 eval prereg stats + test_locked (fixed failure_mass), #10 GPU notebook D4 (marked ready), #11 setup-smoke. Wrote docs/HANDOFF.md.
- result: main at post-merge tip. make smoke OK in venv; pytest 15 passed, 1 skipped; family-split translation check PASS. No open PRs at handoff time.
- next: Vedant Kaggle GPU run (docs/GPU_RUNBOOK.md); Vedant Hindi/Hinglish review Nov 7; data freeze then Nov 8 dev baseline.

## 2026-11-07: Kaggle launch-readiness repair

- what: Grok 4.7 high reviewed the GPU notebook before launch and found retained model bindings, a non-sequential long forward, an undefined timer, missing D4 provenance and guard/fixture gaps. Repaired lifecycle and fail-closed gates, appended D4 before GPU execution, and aligned runbooks. Historical Nov 6 D4 entries above describe the earlier draft; its cited commit is absent from this clone.
- result: External Windows Python 3.12 environment: make smoke PYTHON=python passed; pytest tests/ -q reported 28 passed, 1 skipped; notebook schema/syntax/empty-output validation passed; translation family-split check passed after restoring LF checkout bytes. Frozen YAML and manifest content is unchanged. Kaggle CLI 2.2.5 plus its missing six dependency installed outside git; existing-account kernels listing succeeded. These are setup checks, not GPU parity or ASR.
- review: Independent Grok review caught CircuitKIT/TransformerLens import order. Fixed all three loaders and install cell, added an AST regression check. Verified copied parity bits against an external checkout of CircuitKIT e71d5e17 with scripts/verify_parity_copy.py (path override only), PASS. Added hosted pipeline-integrity checks; no model downloads in CI.
- next: Repository PR, then private Kaggle preparation. Browser unavailable in this session; owner must attach HF_TOKEN and verify free GPU eligibility. Original short and long KL remain unmeasured.

## 2026-11-07: Judge contract and development review queue

- what: Source-checked AuditKIT 1.1.3 GuardJudge from the pinned PyPI wheel. Reproduced malformed output returning value 0.0 with missing fields. Added a strict local wrapper that records judge_failure, preserved separate classifier fields, and fixed duplicate/unmatched agreement accounting. Prepared a calibration/dev-only human queue without altering the frozen manifest or mixed queue.
- result: Isolated AuditKIT environment pytest tests/ -q: 43 passed, 1 skipped. Real installed GuardJudge was exercised through an explicitly fake transport, not a model. Development queue family check PASS with no test membership. Smoke and existing split checks passed. Authenticated Tiny Aya global/base config retrieval succeeded; WildGuard config returned HTTP 403. No model weights loaded and no ASR computed.
- next: Resolve WildGuard access and Kaggle browser setup; human labels pending. Council round 1 Codex identified additional export, count, artifact-eligibility and single-shot hardening work. Claude review unavailable after repeated API errors; no score invented.

## 2026-11-07: Council count/export repair and artifact ingress

- what: Enforced intended/generated/judge-failure count conservation without clipping or changing the primary ASR denominator. Canonical translation membership now comes from the checksum-verified frozen manifest. Unreviewed rows are not scoring-eligible. variant_of remains a canonical EN item link or null, rather than a family id. Added guarded Kaggle ZIP ingestion, file/CRC/hash checks, fixture-only admission and independent short/long gate recomputation.
- result: Isolated AuditKIT environment pytest tests/ -q reported 81 passed, 1 skipped; smoke and family-split checker passed. These are contract fixtures and integrity checks. Frozen prereg YAML and family manifest unchanged. No GPU parity, judged dev metrics, repair or locked-test result exists.
- compute: Launched a separate private existing-account Kaggle T4 x2 runtime preflight, with a 600-second limit. It reads no credentials, loads no models and records hardware/package metadata only. Kernel vedant817/hinglish-shield-runtime-preflight; output pending at launch.
- next: Ingest hardware evidence, verify the remaining dev runner, get independent scoped review, and open the repository PR. Owner is enabling WildGuard gate access; platform HF secret attachment and human labels remain pending.

- preflight completion: Existing-account private Kaggle run completed. Raw hardware output and log copied to runs/kaggle-preflight-2026-11-07/. Remote source SHA-256 matches scripts/kaggle_runtime_preflight.py. CUDA is available with two Tesla T4 devices. Python/torch/RAM metadata is in runtime_preflight.json. No model weights, secrets, curated data, ASR or parity were involved.

## 2026-11-07: Council round 2 corrections and exposure record

- what: Fixed source-member laundering in review preparation, colliding fallback export IDs, unchecked variant_of links, and strict parsing of AuditKIT-truncated diagnostics. Complete classifier output is now validated by the custom profile parse hook. The source-verified development review rebuild is identical to the committed sheet.
- integrity: An earlier header/schema inspection printed a mixed review-sheet row assigned to test. Recorded family fam_983e49e2566f in data/splits/development_exposures.json and docs/INTEGRITY_LOG.md. No model response or test score exists. The original frozen manifest is unchanged; the locked-test guard blocks a manifest that still includes this exposed family in test. A reviewed holdout resolution is required before final evaluation.
- library runtime: Installed pinned CuratorKIT in a separate environment and reproduced the existing default-versus-patched payload fixture. A new synthetic attributed-formatting fixture exposed a gap in the previous local subclass. Candidate now preserves attributed formatting and matching closing tags; real-package fixture comparisons and source hashes are under runs/curatorkit-attributes-fixture-2026-11-07-r3/. No model inference or upstream publication.
- result: PR-scoped CI-equivalent suite excluding the uncommitted dev-runner work: 95 passed, 1 skipped. Separate dev-runner fixtures remain under integration review. Smoke and family-split checks pass.
- access: Authenticated WildGuard config retrieval now returns HTTP 200 after the owner enabled access. This proves configuration access only; judge weights and inference remain unverified. Kaggle platform secret attachment is still pending.
- next: Fresh scoped review before merging. Actual model parity, WildGuard load/calibration, human labels, scored dev baseline and held-out acceptance remain pending.

## 2026-11-07: Council round 3 follow-up

- what: Reproduced negative-KL acceptance, lost attributes on closing formatting tags, and optional/family-only exposure guards with synthetic regression cases. The focused command initially reported 7 failures. The importer now rejects negative KL without clipping; the cleaner preserves attributed opening and closing tags; the locked-test guard requires the ledger and matches exposed source identities across renamed families. No parity tolerance or frozen membership was changed.
- fixture: runs/curatorkit-closing-attributes-fixture-2026-11-07/fixture_results.json executes installed CuratorKIT and compares against project commit 99c4b35, with source hashes. This is synthetic cleaner evidence, not model safety.
- result: PR-scoped suite excluding the separate uncommitted dev ledger: 104 passed, 1 skipped. Repaired dev-ledger synthetic suite: 33 passed, under separate integration review. Neither suite loads model weights. Codex council review was unavailable due to its usage limit; no score assigned. The user-authorized free DeepSeek fallback will provide an additional independent review.
- resumed review: Codex later completed its queued review of 99c4b35 and assigned patch 8/10, project readiness 3/10, not satisfied. Its additional quoted-attribute literal-tag defect was reproduced in the installed-package fixture and repaired with quote-aware span scanning. Current PR-scoped checks: 107 passed, 1 skipped; smoke passed. Latest candidate source hashes are in runs/curatorkit-quoted-attributes-fixture-2026-11-07/. Earlier fixture directories retain their historical hashes.

## 2026-11-07: Free fallback review follow-up

- what: DeepSeek's round-4 review verified the preceding repairs and identified alias/case comparison in the exposure guard and a stale preflight log hash. Three added alias/case regressions failed before the comparison fix and pass afterward. Both identities now pass through source case-folding and the existing canon alias rules; source IDs retain their canonical case semantics.
- provenance: Verified git blob bytes independently. kaggle.log was normalized to LF by its initial git add before byte-preserving attributes were added. The old manifest described the CRLF download. Corrected files_sha256 to the already committed LF artifact, recorded the historical hash and transformation, and aligned local bytes. A new test checks every preflight file hash plus the script source hash. This is hardware evidence only, with no model result altered.
- validation: Full PR-scoped suite excluding uncommitted dev-runner work: 111 passed, 1 skipped. Reviewers' smaller permitted suite omits the existing family-split test and therefore has a different count; the earlier 107/1 result referred to full PR collection, not that smaller review command. No model weights were loaded.
- bounded recheck: DeepSeek confirmed both alias and provenance repairs, then reproduced a new direct-script import regression in 1b45683. Added an isolated subprocess regression from outside the repository; it failed before the fix. The CLI now bootstraps its repository import path only for direct-script execution. Both direct and module --help work; full PR-scoped suite reports 112 passed, 1 skipped. No scoring or model calls were made.

## 2026-11-07: Dev inference ledger integration

- what: Added a dev-only generation/judging ledger with pinned configuration and dataset hashes, canonical membership validation before prompt access, assistant-only greedy outputs, sequential target/judge lifecycle, durable failure rows and exclusive attempt directories. Injected adapters remain fixture/example. Real ledgers remain metrics-ineligible with primary_asr null and no metrics.json.
- review repairs: Grok's synthetic review caught untracked-source provenance and ignored decoding options. Real runs now require a committed matching runner and clean git worktree; ledgers record the LF source hash and dirty/matching flags. Unknown decoding keys are rejected. Earlier integration repairs cover source laundering, exact integer settings, base-only method and cleanup on failure/cancellation.
- validation: Light CI-equivalent environment: 35 dev-ledger tests passed; combined suite 147 passed, 1 skipped; smoke passed. Models and judges are fake transports in these tests. No torch, model weights or GPU was used.
- limits: Existing canonical EN/HI members only; Hinglish and Hindi benign variants need verified reviewed lineage before admission. Human calibration, numerical guard approval, real adapter runtime, category adjudication, preservation metrics and scored baseline remain incomplete. The locked test runner and frozen membership are unchanged.

## 2026-11-07 11:26 IST: Repair CircuitKIT runtime path

- what: Replaced the unconditional CircuitKIT locate refusal with a gated in-process runtime. After recomputed original fp32 short-plus-4196 credit, a run can load pinned Tiny Aya, extract the proposed last-nonpad residual direction, write steered and unsteered dev rows, and unload the target before the judge. Missing credit, an empty chat template, missing node scores, test or relabeled membership, a fixture artifact, or a dirty source tree still fail closed. Hooks clear on success, error, and cancel. Accept still requires the ADR family floor.
- result: Light interpreter pytest tests/repair: 29 passed. No test prompts and no weight download. External torch 2.15.0+cpu interpreter ran verify_circuitkit_contract exit 0 on its own in-process tensor module (vector L2 4.0, alpha 0 diff 0.0, alpha 1 diff 2.0, source e71d5e17). That script credit is not Locate credit. pip check printed no broken requirements. Installed huggingface-hub is 2.4.0 while the CircuitKIT source declares huggingface-hub>=1.5,<2. Deterministic tensor fixture evidence is runs/repair-deterministic-tensor-2026-11-07/fixture_results.json with artifact_kind fixture, split example, and primary_asr null. verify-freeze on that file returned verified true and weights_reloaded false. from_pretrained was not called.
- next: A real Locate still needs an ingested original parity pass, a measured chat-template hash, caller node scores, a clean repair commit, and a judge revision before any scored dev ledger. No ASR is claimed.

# 2026-11-07 IST: dev aggregation admission review repairs

The first independent PR17 review rejected the admission and release gates.
Repaired language-scoped family pairing without dropping both-arm failures,
calibration pair/count conservation and partial kappa validation, missing benign
language coverage, cross-arm identity recording, and explicit harmful/benign
rate denominators. Published run rows now require a hash-bound aggregation
sidecar. Runtime admission consumes hashed WildGuard fixture producer artifacts
and the native notebook dtype-guard format, with revision and per-prompt checks.
Pack acceptance additionally requires matched harmful and benign multilingual
families; capability scores cannot cover only a favorable family subset.

Validation: the permitted model-free eval, notebook and curation suites passed
locally, and smoke passed. The producer interoperability test uses fake inference.
No weights were loaded, no human labels were supplied and no measured run was
promoted. Independent re-review is pending. Reviewed-variant aggregation
integration and actual prerequisite runtime evidence remain outstanding.
# 2026-11-07 IST: bounded aggregation re-review and follow-up

The independent PR17 re-review accepted the repairs for the original admission
findings and identified a single-arm capability mismatch. Capability aggregation
now uses benign families, matching its documented contract and the release gate.
The WildGuard fixture producer refuses a missing Git identity before importing
torch, and calibration documentation distinguishes paired ledger counts from
the supplied sheet's declared expected size. Focused synthetic regressions and
smoke passed. No model/runtime safety score or human calibration was produced.
