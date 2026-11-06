import pytest

from src.eval.judges import JudgeVerdict, kappa_report


def test_agreement_reports_unmatched_and_failed_rows():
    a = [JudgeVerdict("ok", 1.0), JudgeVerdict("unmatched", 0.0),
         JudgeVerdict("failed", None, "judge_failure")]
    b = [JudgeVerdict("ok", 1.0), JudgeVerdict("failed", 0.0)]
    report = kappa_report(a, b)
    assert report["n_paired"] == 1
    assert report["n_unmatched"] == 1
    assert report["n_skipped"] == 2
    assert report["n_primary"] == 3 and report["n_secondary"] == 2


def test_agreement_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        kappa_report([JudgeVerdict("x", 1.0), JudgeVerdict("x", 0.0)], [])


def test_agreement_excludes_nonfinite_scores_even_with_label():
    report = kappa_report([JudgeVerdict("x", float("nan"), "safe")], [JudgeVerdict("x", 0.0)])
    assert report["n_paired"] == 0 and report["n_skipped"] == 1
    assert report["agreement"] is None
