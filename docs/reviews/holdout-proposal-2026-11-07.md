# Independent research review of D7-PROPOSED

Scope: proposal defensibility only. Reviewed local commit 654235f39dc6257c5d3cdf60c62b0849e9337195. No human disposition approval, candidate manifest approval, implementation approval or test authorization is implied.

+**accept-for-proposal-review.** Only the inactive D7-PROPOSED disposition is defensible enough to present. Candidate-manifest approval, config or loader changes, and any use stay gated. This review does not authorize a v3 manifest, a preregistration edit, or test access.

Verified on unpublished commit `654235f39dc6257c5d3cdf60c62b0849e9337195`, whose parent is approved main `d23546ffcf6456f61528ce3156ae26854821c654`. The commit adds the proposal pair and does not change frozen inputs. No `family_splits_v3` file exists. `execution_authorized` is false.

The four frozen paths match `3422aee`, the parent, and this commit:

| Path | Git blob | SHA-256 |
| --- | --- | --- |
| `data/splits/family_splits_v2.json` | `385d4e8795968203dbcb5f2bd3b88d9dee6a1709` | `334e7151edfd9fe958afcded67417b9315d2d26b693b51f91b998c5d9638b531` |
| `configs/research/baseline_prereg.yaml` | `e55ed3f1a8d8379b8db385dc616647eae9ec3cde` | `7a328e3e8fe02d480cfec6ccf7fb66aeb3830159a1f99c5536db3a2b3629c56c` |
| `configs/eval/baseline.yaml` | `a7ef06db5c74404486df5f4931dff6ac7108fc5c` | `10c1d49d2422026c8ecfa794ca36894edb5e1b3e395c70e63a8024cf698830a6` |
| `configs/eval/test_locked.yaml` | `af427279fb0603111b12b30fb49d3c85d6ac6fb4` | `9bee35aded7eac72f7a86935dc21afe05a62f76a90e3f83862a62d2958fe631c` |

The ledger `data/splits/development_exposures.json` is absent at `3422aee`. On the parent and on this commit its blob is `8f6cfaaba357a80611071928d0d7e5220b11e3e3` and its SHA-256 is `aeb467ab5047178470669236d4282941330522ae9470bcfcf30c346bac8285b7`. The v2 sidecar is the same blob from `3422aee` through this commit and records that manifest hash.

Canonical membership for `fam_983e49e2566f` is one frozen test family, label `mixed`, category `xstest:contrast_homonyms`, and exactly two English members: `xstest:1` (`benign`, `xstest:homonyms`) and `xstest:26` (`harm`, `xstest:contrast_homonyms`). Those source ids appear in no other family. Member records have no alias or translation fields. The ledger records only `xstest` / `26`, original split test, with no model response and no test score. `xstest:1` is the benign near-neighbor in the same pre-assigned family, not a second ledger row.

Full-family quarantine with no replacement is prospectively defensible. The preregistered unit is the semantic family, including the benign near-neighbor, and D3 keeps XSTest out of calibration. Dropping only the ledger row would split that pair after exposure. Leaving both out of calibration, dev, and test, and selecting no replacement, avoids a content- or outcome-based substitute. The population change is a logged deviation: the original holdout stays not wholly unseen, removal is not a safety gain, and later arms must share one approved population. v2 currently has no derivative rows for these ids. The proposal correctly keeps any future alias, translation, or lineage in quarantine until a separate canonical-lineage review. The exact new denominator stays a later candidate-review condition.

The draft is inactive, separates this decision from manifest and loader approval, and reserves the disposition for Vedant or the Research Lead. The Nov 11 IST embargo and the frozen-arm, calibration, and single-shot requirements remain. Approval of this proposal still has to be written down before any candidate work, and that write must not silently replace the v2 hash or edit the pinned preregistration.
