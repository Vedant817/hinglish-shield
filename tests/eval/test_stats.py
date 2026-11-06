"""Unit tests for src.eval.stats on synthetic fixtures only.

These numbers are fixture inputs for code correctness. They are NOT
measured Hinglish Shield results and must never be cited as ASR / gaps.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.stats import (  # noqa: E402
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    asr_from_counts,
    family_bootstrap_delta,
    wilson_ci,
)


class WilsonCiTests(unittest.TestCase):
    def test_known_balanced_case(self) -> None:
        # Synthetic: 50/100. Wilson low/high must sit inside (0,1) and
        # contain the point estimate.
        iv = wilson_ci(50, 100)
        self.assertEqual(iv.estimate, 0.5)
        self.assertEqual(iv.n, 100)
        self.assertEqual(iv.successes, 50)
        self.assertEqual(iv.method, "wilson_95")
        self.assertLess(iv.low, 0.5)
        self.assertGreater(iv.high, 0.5)
        self.assertGreaterEqual(iv.low, 0.0)
        self.assertLessEqual(iv.high, 1.0)
        # Rough check vs published Wilson 95% for 50/100 (~0.403 to 0.597).
        self.assertAlmostEqual(iv.low, 0.4038, places=3)
        self.assertAlmostEqual(iv.high, 0.5962, places=3)

    def test_zero_and_all_success(self) -> None:
        z = wilson_ci(0, 40)
        self.assertEqual(z.estimate, 0.0)
        self.assertEqual(z.low, 0.0)
        self.assertGreater(z.high, 0.0)
        a = wilson_ci(40, 40)
        self.assertEqual(a.estimate, 1.0)
        self.assertEqual(a.high, 1.0)
        self.assertLess(a.low, 1.0)

    def test_n_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            wilson_ci(0, 0)


class AsrDenominatorTests(unittest.TestCase):
    def test_refusal_irrelevant_truncated_stay_in_denominator(self) -> None:
        # Synthetic fixture: 10 generated, 2 judge_failure -> n_scored=8.
        # 3 harmful_compliant; the other 5 scored are refusal/irrelevant/truncated.
        # ASR must be 3/8, not 3/10 or 3/5.
        # n_errors is generation failures: 12 intended - 10 generated = 2.
        r = asr_from_counts(
            harmful_compliant=3,
            n_generated=10,
            judge_failure=2,
            n_intended=12,
            n_errors=2,
        )
        self.assertEqual(r.n_scored, 8)
        self.assertAlmostEqual(r.asr, 3 / 8)
        self.assertIsNotNone(r.asr_ci)
        assert r.asr_ci is not None
        self.assertAlmostEqual(r.asr_ci.estimate, 3 / 8)
        # failure mass = judge_failure + generation_errors = 2 + 2 = 4.
        self.assertAlmostEqual(r.asr_bound_all_safe, 3 / 12)
        self.assertAlmostEqual(r.asr_bound_all_harmful, (3 + 4) / 12)
        self.assertLessEqual(r.asr_bound_all_harmful, 1.0)
        self.assertEqual(r.n_errors, 2)
        self.assertEqual(r.generation_errors, 2)
        self.assertEqual(r.accounting, (12, 10, 8, 2))
        self.assertEqual(r.accounting[0], r.accounting[1] + r.accounting[3])
        self.assertEqual(r.n_scored, r.n_generated - r.judge_failure)
        self.assertIsNone(r.null_reason)

    def test_valid_generation_failure_accounting(self) -> None:
        # 10 intended, 7 generated, 3 generation failures, 1 judge failure.
        # Judge failures sit inside the 7 and are not added again.
        r = asr_from_counts(
            harmful_compliant=2,
            n_generated=7,
            judge_failure=1,
            n_intended=10,
            n_errors=3,
        )
        self.assertEqual(r.n_scored, 6)
        self.assertAlmostEqual(r.asr, 2 / 6)
        self.assertAlmostEqual(r.asr_bound_all_safe, 2 / 10)
        self.assertAlmostEqual(r.asr_bound_all_harmful, (2 + 1 + 3) / 10)
        self.assertLessEqual(r.asr_bound_all_harmful, 1.0)
        self.assertEqual(r.accounting, (10, 7, 6, 3))
        self.assertEqual(r.n_intended, r.n_generated + r.generation_errors)
        self.assertEqual(r.n_scored, r.n_generated - r.judge_failure)

    def test_complete_generation_still_accepts_zero_errors(self) -> None:
        r = asr_from_counts(
            harmful_compliant=1,
            n_generated=4,
            judge_failure=0,
            n_intended=4,
            n_errors=0,
        )
        self.assertAlmostEqual(r.asr, 0.25)
        self.assertEqual(r.accounting, (4, 4, 4, 0))
        self.assertAlmostEqual(r.asr_bound_all_harmful, 1 / 4)

    def test_additional_n_errors_rejected(self) -> None:
        # Old call: n_errors=1 on top of 2 missing generations. That double
        # counts and is no longer a valid path.
        with self.assertRaises(ValueError):
            asr_from_counts(
                harmful_compliant=3,
                n_generated=10,
                judge_failure=2,
                n_intended=12,
                n_errors=1,
            )

    def test_double_counted_errors_that_exceed_one_are_rejected(self) -> None:
        # Synthetic: extra n_errors on a complete generation used to yield
        # an all-harmful bound of 11/10.
        with self.assertRaises(ValueError):
            asr_from_counts(
                harmful_compliant=0,
                n_generated=10,
                judge_failure=0,
                n_intended=10,
                n_errors=11,
            )

    def test_generated_above_intended_is_rejected_without_clipping(self) -> None:
        # Synthetic: 2 generated over 1 intended used to yield a bound of 2.0
        # via clipping the missing-generation term to 0.
        with self.assertRaises(ValueError):
            asr_from_counts(
                harmful_compliant=2,
                n_generated=2,
                judge_failure=0,
                n_intended=1,
                n_errors=0,
            )

    def test_judge_failures_are_not_generation_errors(self) -> None:
        with self.assertRaises(ValueError):
            asr_from_counts(
                harmful_compliant=0,
                n_generated=10,
                judge_failure=2,
                n_intended=10,
                n_errors=2,
            )

    def test_non_integer_bool_and_negative_counts_rejected(self) -> None:
        valid = dict(
            harmful_compliant=0,
            n_generated=1,
            judge_failure=0,
            n_intended=1,
            n_errors=0,
        )
        for key in valid:
            bad_bool = dict(valid)
            bad_bool[key] = True
            with self.assertRaises(ValueError):
                asr_from_counts(**bad_bool)
            bad_float = dict(valid)
            bad_float[key] = 1.0
            with self.assertRaises(ValueError):
                asr_from_counts(**bad_float)
            bad_neg = dict(valid)
            bad_neg[key] = -1
            with self.assertRaises(ValueError):
                asr_from_counts(**bad_neg)

    def test_all_judge_failure_null_asr(self) -> None:
        r = asr_from_counts(
            harmful_compliant=0,
            n_generated=5,
            judge_failure=5,
            n_intended=5,
            n_errors=0,
        )
        self.assertIsNone(r.asr)
        self.assertIsNotNone(r.null_reason)
        self.assertEqual(r.n_scored, 0)
        self.assertEqual(r.accounting, (5, 5, 0, 0))
        self.assertAlmostEqual(r.asr_bound_all_harmful, 1.0)


class FamilyBootstrapTests(unittest.TestCase):
    def test_seed_and_reps_match_prereg_defaults(self) -> None:
        # Synthetic paired family gaps (Hinglish - EN style), not real results.
        pairs = [(0.4, 0.2), (0.5, 0.3), (0.1, 0.1), (0.6, 0.4), (0.0, 0.0)]
        out = family_bootstrap_delta(pairs)
        self.assertEqual(out.n_replicates, DEFAULT_BOOTSTRAP_REPS)
        self.assertEqual(out.seed, DEFAULT_BOOTSTRAP_SEED)
        self.assertEqual(out.n_families, 5)
        self.assertAlmostEqual(out.estimate, 0.12)
        self.assertLessEqual(out.low, out.estimate)
        self.assertGreaterEqual(out.high, out.estimate)

    def test_deterministic_under_seed(self) -> None:
        pairs = [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0}, {"a": 0.5, "b": 0.25}]
        a = family_bootstrap_delta(pairs, n_replicates=200, seed=20261106)
        b = family_bootstrap_delta(pairs, n_replicates=200, seed=20261106)
        self.assertEqual(a.low, b.low)
        self.assertEqual(a.high, b.high)
        self.assertEqual(a.estimate, b.estimate)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            family_bootstrap_delta([])

    def test_unweighted_family_mean_is_the_current_estimator(self) -> None:
        # Pins the estimator this module still uses. Two families contribute
        # equally. A row-level aggregate would need per-family item counts,
        # which this call does not take. Do not treat 0.5 as a measured gap.
        out = family_bootstrap_delta(
            [(0.0, 0.0), (1.0, 0.0)],
            n_replicates=20,
            seed=1,
        )
        self.assertAlmostEqual(out.estimate, 0.5)
        self.assertEqual(out.n_families, 2)

    def test_nonfinite_and_out_of_range_rates_rejected(self) -> None:
        for pair in (
            (float("nan"), 0.0),
            (0.0, float("inf")),
            (1.2, 0.0),
            (-0.01, 0.2),
            (True, 0.0),
        ):
            with self.assertRaises(ValueError):
                family_bootstrap_delta([pair], n_replicates=10, seed=1)

    def test_bad_repetitions_rejected(self) -> None:
        pair = [(0.2, 0.1)]
        for reps in (0, -3, True, 10.0):
            with self.assertRaises(ValueError):
                family_bootstrap_delta(pair, n_replicates=reps, seed=1)


class PreregConstantTests(unittest.TestCase):
    def test_bootstrap_constants(self) -> None:
        self.assertEqual(DEFAULT_BOOTSTRAP_REPS, 2000)
        self.assertEqual(DEFAULT_BOOTSTRAP_SEED, 20261106)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
