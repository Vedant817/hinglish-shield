# Locked test preparation (prepared, never run)

Status: **not run**. The locked test split stays embargoed until 2026-11-11
00:00 IST. This document records the prepared command and the guards so the
single allowed pass can be executed without re-deciding anything on the day.
It does not authorize running anything. Do not call the single-shot command
during preparation.

## The prepared command

On or after 2026-11-11 00:00 IST, only after every row of the checklist below
passes, run exactly:

```text
python -m src.eval.run_baseline --config configs/eval/test_locked.yaml --split test --i-understand-single-shot
```

`--dry-check` (same flags) verifies the config and guards without scoring and
must be used first. The CLI is still a non-scoring stub; a real scorer must
enter through `src.eval.run_baseline._execute_locked_test` only. The command
is for one single-shot pass on the frozen arms `base`, `refuse_everything`
and `frozen_repair`. It is never for selection, tuning, or a retry.

## Guards (all must hold; any failure refuses the run)

1. Date: now is on or after 2026-11-11 00:00 IST (`TEST_EMBARGO_END` in
   `src/eval/run_baseline.py`).
2. Frozen partitions: `datasets.dataset_hash` is a full SHA-256 and
   `datasets.split_manifest` is the canonical frozen family manifest whose
   bytes match the committed pin, with its checksum sidecar.
3. Exposure ledger: `data/splits/development_exposures.json` exists and no
   exposed family sits in the test split.
4. Frozen repair: `configs/eval/test_locked.yaml` names the frozen repair
   manifest and its exact byte SHA-256 (both are still null today, so the
   guard refuses), and that arm verifies as a real-model arm.
5. Model pin: `model_revision` is a full 40-hex HF revision.
6. Single attempt: `runs/_locked_test_attempt.json` does not exist and no
   prior test run record exists under `runs/`. A reserved attempt is never
   deleted to retry.
7. Arms and seeds: the config runs `base`, `refuse_everything` and
   `frozen_repair` with the same harness, judges and seeds (seed 42).

## What is still missing today (Nov 8)

- The frozen repair manifest and hash in `configs/eval/test_locked.yaml`
  are still null. They get pinned only from a verified real repair arm.
- The locked configuration and the scorer entry point are preparation
  only. Admission has not been granted and no generation, judgement, or
  scoring of the test split has happened.
