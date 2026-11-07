# Bounded verification and correction, Nov 7 IST

This is a model-free engineering audit, `artifact_kind: setup-smoke`, not safety or hackathon acceptance. Nothing was merged. Evidence, exact commands, test logs and SHA-256 receipts are in `runs/model-free-audit-2026-11-07/`.

Corrected `docs/HANDOFF.md` and `docs/KAGGLE_AGENT.md`: only the 26-row calibration/dev sheet is for development. The historical mixed queue contains 33 test rows and is not a development review source. HinGE quality ratings and IndicGuard source labels do not approve finance translations or label our generated responses. The unresolved exposure incident and provisional parity limitations now appear in both guides. Removed the instruction to merge merely on green CI.

## Exact snapshots and outcomes

| Snapshot | Commit | Smoke | Permitted pytest | Split checker |
| --- | --- | --- | --- | --- |
| Clean current main | `6ebc9bf0f067d4164a82af1c3bc9fada55353a13` | exit 0 | 259 passed, 7 subtests passed; exit 0 | PASS, 0 errors; exit 0 |
| Isolated PR21+PR22 integration | `861fc623ceadcbeb0c74ff87aab7ac6a0a5f57af` | exit 0 | 260 passed, 7 subtests passed; exit 0 | PASS, 0 errors; exit 0 |

The integration tree is `1e13eb7fc4f2ae07b90ffacd55e9888ced5dc027`. It combines PR21 `23c53c87e3da39d43f8492d77ea4a1ac3d55ac5e` and PR22 `31d82aaf26cbd647b5d81c58451b0735e14f4c2b` on main. Only their conflicting LOG append was resolved, preserving both entries in PR order. The private verification commit was not pushed and neither PR was merged. The legitimate detached worktree was clean before and after testing. Main was also clean during its Git-context run.

Initial archive exports had four pytest failures each (255 and 256 passes, respectively), caused by missing Git HEAD provenance. Those failure logs are retained. The proper Git-context reruns passed without changing tests or weakening provenance gates. Do not call the archive failures product defects or erase them.

Commands below use the external `.venv-ci` Python 3.12.15, with no torch, offline Hugging Face flags, bytecode disabled, and the preserved `audit_guard.py` on `PYTHONPATH`. The guard refuses prompt-file opens under mixed staging, seeds, interim and curated directories; only the calibration/dev review sheet is allowed. The excluded test module parses mixed staging queues. The explicitly requested split checker runs separately and reports identity/count invariants only. It parses mixed CSVs internally; no mixed/test prompt text was displayed or used for generation, scoring or tuning.

```powershell
$py = 'C:/Users/vedan/Documents/Resources/Code/hinglish-shield-scratch/.venv-ci/Scripts/python.exe'
make smoke "PYTHON=$py"
& $py -B -m pytest -p no:cacheprovider -p audit_guard tests --ignore=tests/curate/test_translation_family_splits.py -q --tb=short --basetemp=<external-unique-directory>
& $py -B scripts/check_translation_family_splits.py
```

Machine-readable command lines, working directories, exit codes and results are in `main-git-tests.json` and `integration-git-tests.json`; archive results are in `tests.json`. The split checker reported no unmapped rows in all three queues. The mixed review queue identity counts are calibration 14, dev 12, test 33. The checker does not establish translation quality.

## Frozen blob comparisons and history scan

All four blobs below match handoff main `3422aee0c7b9014dc82c5bdbd97cc3b3b13ad9da` on current main, both PR heads and the integration tree. Comparisons use `git rev-parse REF:path` and SHA-256 of `git show REF:path`, with no test-prompt inspection.

| File | Identical Git blob |
| --- | --- |
| `data/splits/family_splits_v2.json` | `385d4e8795968203dbcb5f2bd3b88d9dee6a1709` |
| `configs/research/baseline_prereg.yaml` | `e55ed3f1a8d8379b8db385dc616647eae9ec3cde` |
| `configs/eval/baseline.yaml` | `a7ef06db5c74404486df5f4931dff6ac7108fc5c` |
| `configs/eval/test_locked.yaml` | `af427279fb0603111b12b30fb49d3c85d6ac6fb4` |

The manifest SHA-256 remains `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`. Full SHA-256 values are in `frozen_comparison.json`. PR13 through PR22 net diffs and constituent commit scans show no changes to these frozen files. Operational D4/D5/D6 prose changes are explicitly logged in the prereg; they are not frozen YAML edits. No added em dashes were found in authored Markdown diffs, and no tracked model-weight or sensitive credential filenames were found. Known token/private-key patterns did not match scanned non-prompt blobs; 76 prompt payload files were deliberately excluded from content scanning. This is scoped consistency checking, not an assertion that every possible secret is absent.

The repository Markdown metric-claim review found protocol thresholds, labeled planning math, theoretical controls, fixture diagnostics and engineering checks, but no claimed measured benchmark result. Both tracked `metrics.json` files are `artifact_kind: fixture`, `split: example`, with all benchmark fields null. No measured-run metrics file exists in this integration snapshot. Off-repository Notion/article/submission contents were not audited here.

## Runtime evidence and independent review status

PR21 remains open at `23c53c87e3da39d43f8492d77ea4a1ac3d55ac5e`: integrity, GitGuardian and Kilo checks passed; no submitted formal review. The recomputed short bf16 mean KL is `0.0004105035332031548`, above the original `1e-4` threshold. Every short value meets its measured two-floor variant rule and top-1 meets the variant condition. The 4196-token bf16 KL is `4.146346327615902e-06`; its long floor is unavailable, so the combined result remains provisional under D4. Original fp32 short and long did not run. Locate credit remains false. D5/D6 operational deviations and raw reports were preserved; no thresholds or acceptance flags changed after v8. The embedded prereg hash matches the cited source, while the runtime prereg file was absent.

PR22 remains open and draft at `31d82aaf26cbd647b5d81c58451b0735e14f4c2b`: integrity and GitGuardian passed; no submitted formal review. The fixture shows English WildGuard CUDA load and strict parsing only. Both response verdicts are safe. There is no unsafe-response positive control, accuracy estimate, human kappa, multilingual calibration or ASR. Recomputed committed bytes match all 12 PR21 ingestion hashes and all 4 PR22 ingestion hashes. This pass did not rehash the external ZIPs or authenticate the producing machine.

The pre-existing independent runtime-evidence review was mistakenly cancelled by the parent and is interrupted. Its messages contain progress but no final findings/verdict. It is still outstanding and must not count as acceptance. The isolated integration test task is completed, but it is model-free verification and does not replace that review. No other work was cancelled, no unchanged council loop was started, and no merge is authorized by this audit.

## Remaining decisions and next executable step

`fam_983e49e2566f` is a real unresolved holdout-contamination incident, not a warning to waive. A synthetic metadata-only guard probe still returns `test holdout contains a development-exposed family: fam_983e49e2566f`. The original manifest, exposure ledger and guard remain unchanged. The Research Lead must separately review and preregister a prospective holdout resolution; no test generation, scoring or tuning before the single Nov 11 frozen-arm pass, and no pass until that resolution and the remaining gates are satisfied.

Human work remains: explicit finance-translation accepts with lineage, human labels of actual generated English/Hindi/Hinglish responses, secondary-judge/capability choices and the prospective holdout decision. Public source labels cannot fill those decisions. No multilingual judge validation or measured dev baseline is claimed.

The next executable step is to produce the calibration/dev-only coverage and missing-label report using the existing reviewed-variant exporter, keeping unreviewed rows ineligible. Use that to assemble the reviewed multilingual lineage and response-calibration pack, pin the complete pre-run identity, then run preliminary H1 on dev only on Nov 8. Weight-backed Locate/Repair remains gated on original parity credit and the remaining pins. No UI expansion, spending, new accounts, HF publishing or other-repository posts were performed.


Coverage-report command (use a new external output directory; blank labels should produce an unavailable report and exit 2):

```powershell
& $py scripts/export_reviewed_variants.py --review-sheet data/staging/review/vedant_review_nov7_calibration_dev.csv --acceptance-policy explicit-human-accept-v1 --out <new-external-coverage-directory>
```

Packaging check: initial `git diff --check` flagged CRLF in newly generated audit metadata. Those new metadata files were normalized to LF and their receipts refreshed. Existing runtime reports and frozen blobs were not changed.
Retained test logs also contain blank lines at EOF; these were preserved. The final documentation/metadata whitespace check excludes only those log files.
