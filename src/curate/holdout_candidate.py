"""Metadata-only correction of inactive candidate population summaries.

Does not write files, activate a manifest, or inspect prompt text.
"""
from collections import Counter
from copy import deepcopy

HISTORICAL_FIELDS = ("counts", "realized_ratios", "harm_share")


def clarify_candidate_summaries(candidate: dict, original: dict) -> dict:
    if (candidate.get("frozen") is not False
            or candidate.get("candidate_review", {}).get("execution_authorized") is not False):
        raise ValueError("summary correction requires an inactive candidate")
    for field in HISTORICAL_FIELDS:
        if candidate.get(field) != original.get(field):
            raise ValueError("historical v2 summaries must remain unchanged")
    family_counts, member_counts = Counter(), Counter()
    for family in candidate["families"].values():
        family_counts[family["split"]] += 1
        member_counts[family["split"]] += len(family["members"])
    result = deepcopy(candidate)
    result["candidate_review"]["historical_summary_fields"] = list(HISTORICAL_FIELDS)
    result["candidate_review"]["historical_summary_scope"] = (
        "Unchanged v2 historical snapshots only; not candidate denominators, "
        "ratios or composition. Derive candidate populations from families; "
        "use prospective_population below, never counts.test for this candidate."
    )
    result["candidate_review"]["prospective_population"] = {
        split: {"families": family_counts[split], "members": member_counts[split]}
        for split in sorted(family_counts)
    }
    result["candidate_review"]["replacement_count"] = 0
    return result
