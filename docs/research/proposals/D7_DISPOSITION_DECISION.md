# D7 disposition decision

Vedant approved disposition only in this conversation: quarantine the entire exposed family `fam_983e49e2566f`, including both members and all derivatives, outside calibration/dev/test, with no replacement. Approval was received on Nov 7, 2026 IST after the independent proposal review.

Approval scope is the proposed disposition. It does not approve a v3 manifest, config/loader changes, canonical hash replacement, test access or the repair arm. Those still require separate review. The Nov 11 IST embargo and all remaining scientific prerequisites remain binding.

The original cancelled runtime review is not a pass. The separately completed replacement review accepted only its immutable PR21/PR22 runtime-evidence scope. Merges and this disposition decision grant no original fp32 parity, multilingual calibration or ASR acceptance.

A metadata-only candidate has now been prepared for separate review at `data/splits/proposals/family_splits_v3_holdout_candidate.json`. It is deliberately `frozen: false`, has a distinct candidate version, and explicitly denies execution. It retains the affected family identity and exact members with split `quarantine`; it preserves every other family object. No active config points to it and no loader is modified. An eventual eligible manifest/config/loader and its exact byte pins still require their own reviewed activation. This file does not silently freeze or authorize those future bytes.
