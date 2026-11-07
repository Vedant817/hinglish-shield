# Independent review of the first inactive v3 candidate

Historical verdict for a8647bd192615b5a0b7e60c3993b5f16a8e87733: REVISE, not acceptance. The correction and focused checks must not retroactively turn this verdict into a pass. No activation or test access is authorized.

+**Revise.** The inactive candidate is not accepted. Membership quarantine is correct, but the hashed manifest still publishes the old test denominator.

The worktree is clean at unpublished local commit `a8647bd192615b5a0b7e60c3993b5f16a8e87733`. No remote contains it. Its parent is `3cd7b127cbb3acb9427d8dacd741ccea6ea23e9f`, then proposal `654235f39dc6257c5d3cdf60c62b0849e9337195`, then main `d23546ffcf6456f61528ce3156ae26854821c654`. This review does not approve config, loader, or test use, and it does not authorize activation.

## Blocker

`data/splits/proposals/family_splits_v3_holdout_candidate.json` keeps `counts`, `realized_ratios`, and `harm_share` byte-identical to v2. `counts.test` still says 2397 families and 2550 items. The split field does not: the only family move is `fam_983e49e2566f` from `test` to `quarantine`, so the population is 2396 test families and 2548 test members, plus 1 quarantined family and 2 quarantined members. Calibration stays 896 families / 905 members. Dev stays 1437 families / 1521 members.

The manifest `candidate_review` block does not say those three summaries are historical v2 snapshots, and it does not state the prospective denominator. `holdout_v3_candidate_review.json` does state 2397 to 2396 and 2550 to 2548, without saying the embedded summaries contradict that. Pinning SHA-256 `1dacf931a14f62d57fb5138b01068efb62125222901dfbe8cf0c99c013fc4ad5` would freeze the old test count.

Smallest repair: keep every family object as it is. Inside manifest `candidate_review`, state that `counts`, `realized_ratios`, and `harm_share` are unchanged v2 historical summaries and are not the candidate denominator. State the prospective denominator there: test 2396/2548, quarantine 1/2, calibration and dev unchanged, replacement 0. Mirror that in `holdout_v3_candidate_review.json`, then recompute the raw SHA-256 and sidecar. Do not flip `frozen`, do not point a config or loader at it, and do not treat the new hash as activated.

## What checked out

Raw file bytes, the git blob, and the sidecar all match `1dacf931a14f62d57fb5138b01068efb62125222901dfbe8cf0c99c013fc4ad5`. The sidecar is `hash`, two spaces, filename, and a trailing LF. The file is LF, has no BOM, and has no duplicate JSON keys.

Header changes are exactly `version` `family_splits_v3_holdout_candidate`, `frozen` false, and new `candidate_review`. `execution_authorized` is false. Status is `inactive_candidate_not_approved_for_use`. The embargo timestamp is `2026-11-11T00:00:00+05:30`. Source manifest version and SHA-256 point at v2.

All 4730 family ids and all 4976 member records remain. There are 4976 unique `(source_dataset, source_id)` pairs, 4976 unique canonical ids, and 4976 unique `item_id`s, with no duplicates in v2 or v3. No member was reassigned. No sibling was dropped or added. The only object difference is `fam_983e49e2566f.split`. Its member records match v2 exactly: `xstest:1` benign `xstest:homonyms`, and `xstest:26` harm `xstest:contrast_homonyms`, both English. Those source ids appear in no other family. Family label remains `mixed`, category `xstest:contrast_homonyms`. Member keys are only `category`, `hs_label`, `item_id`, `lang`, `source_dataset`, and `source_id`. No prompt field is present. No replacement family was selected.

`D7_DISPOSITION_DECISION.md` records disposition-only approval: quarantine the entire family, including both current members and future derivatives, outside calibration, dev, and test, with no replacement. The earlier proposal review at `654235f` remains accept-for-proposal-review only. `holdout_resolution_v3.proposed.json` is still the historical proposal, including `candidate_manifest_written: false`.

These four paths match commit `3422aee0c7b9014dc82c5bdbd97cc3b3b13ad9da` by git blob and raw SHA-256, on main and on this commit:

| Path | Blob | SHA-256 |
| --- | --- | --- |
| `data/splits/family_splits_v2.json` | `385d4e8795968203dbcb5f2bd3b88d9dee6a1709` | `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531` |
| `configs/research/baseline_prereg.yaml` | `e55ed3f1a8d8379b8db385dc616647eae9ec3cde` | `7a328e3e8fe02d480cfec6ccf7fb66aeb3830159a1f99c5536db3a2b3629c56c` |
| `configs/eval/baseline.yaml` | `a7ef06db5c74404486df5f4931dff6ac7108fc5c` | `10c1d49d2422026c8ecfa794ca36894edb5e1b3e395c70e63a8024cf698830a6` |
| `configs/eval/test_locked.yaml` | `af427279fb0603111b12b30fb49d3c85d6ac6fb4` | `9bee35aded7eac72f7a86935dc21afe05a62f76a90e3f83862a62d2958fe631c` |

The v2 sidecar still records `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531`. `development_exposures.json` is absent at `3422aee`. On main and here its blob is `8f6cfaaba357a80611071928d0d7e5220b11e3e3` and its SHA-256 is `aeb467ab5047178470669236d4282941330522ae9470bcfcf30c346bac8285b7`. It still records only `xstest` / `26`. `src/eval/run_baseline.py` blob `b289a4679a08085bb00bd6b03670be34c9c16b94` and `tests/eval/test_run_baseline_guard.py` blob `0ffd07bdad6196ced8eb45858e6bf72b1d685e4c` match main. No config, source, script, or test references the candidate path. `test_locked.yaml` still has `split_manifest: null`.

## Later activation prerequisite

This is source reading only. The guard was not executed, and no test was run. `_validate_test_single_shot` never checks `version`. It inspects only the configured `datasets.split_manifest`. It rejects that file when `frozen` is not true, `families` is missing, or the sidecar disagrees with the raw bytes. It does not look at `data/splits/proposals/`. The exposure check fires only when the ledger identity is still in a family whose split is `test`. Quarantine makes that condition false, so the ledger would not reject this population. A later file at an arbitrary path, with any version, `frozen: true`, and a self-matching sidecar, would pass those clauses. Version and frozen state are therefore not an allowlist. A reviewed pin of version, hash, path, lineage, and the approved population is still required before any use.

Original v2 remains development-exposed. Do not call it wholly unseen. Moving this family is a population change, not a safety gain. Future aliases and derivatives still need a separate lineage review. Every method must use one approved population. The Nov 11 IST embargo, a later freeze, calibration, and one atomic single-shot pass remain required.
