"""Tests for the principled threshold-selection protocols."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mcie import thresholds as T
from mcie.model import Region, SituationTable, classify_degree


def _degrees(values):
    return [Fraction(v).limit_denominator(1000) for v in values]


def test_dtrs_symmetric_losses_give_third_two_thirds():
    r = T.dtrs_thresholds(
        lambda_cc=0, lambda_nc=2, lambda_ac=6, lambda_ca=6, lambda_na=2, lambda_aa=0
    )
    assert r.beta == Fraction(1, 3)
    assert r.alpha == Fraction(2, 3)


def test_dtrs_rejects_empty_neutral_region():
    # Deferral never attractive -> beta >= alpha -> invalid.
    with pytest.raises(ValueError):
        T.dtrs_thresholds(
            lambda_cc=0, lambda_nc=5, lambda_ac=6, lambda_ca=6, lambda_na=5, lambda_aa=0
        )


def test_dtrs_rejects_bad_loss_ordering():
    with pytest.raises(ValueError):
        T.dtrs_thresholds(
            lambda_cc=3, lambda_nc=2, lambda_ac=6, lambda_ca=6, lambda_na=2, lambda_aa=0
        )


def test_quantile_thresholds_partition_and_ordering():
    degs = _degrees(np.linspace(0, 1, 101))
    r = T.quantile_thresholds(degs, lower_quantile="1/3", upper_quantile="9/10")
    assert 0 <= r.beta < r.alpha <= 1
    counts = r.region_counts
    assert counts["AR"] + counts["NR"] + counts["CR"] == len(degs)
    # bottom third allied, top ~decile conflict, given a uniform spread
    assert counts["AR"] > 0 and counts["NR"] > 0 and counts["CR"] > 0


def test_quantile_handles_degenerate_ties():
    # All degrees identical: alpha must be nudged above beta and stay valid.
    degs = _degrees([Fraction(1, 2)] * 20)
    r = T.quantile_thresholds(degs, lower_quantile="1/3", upper_quantile="9/10")
    assert r.beta < r.alpha


def test_fraction_quantile_matches_nearest_rank():
    degs = _degrees([0.1, 0.2, 0.3, 0.4, 0.5])
    assert T._fraction_quantile(degs, Fraction(0)) == Fraction(1, 10)
    assert T._fraction_quantile(degs, Fraction(1)) == Fraction(1, 2)
    # median nearest-rank (ceil of 0.6*5=3) -> 3rd order stat = 0.3
    assert T._fraction_quantile(degs, Fraction(1, 2)) == Fraction(3, 10)


def test_sensitivity_grid_filters_invalid_pairs():
    grid = T.sensitivity_grid(betas=["1/4", "1/2"], alphas=["1/4", "3/4"])
    # (1/4,1/4) dropped (not strictly ordered); (1/2,1/4) dropped
    assert (Fraction(1, 4), Fraction(1, 4)) not in grid
    assert (Fraction(1, 2), Fraction(1, 4)) not in grid
    assert (Fraction(1, 4), Fraction(3, 4)) in grid
    assert all(b < a for b, a in grid)


def test_region_partition_counts_match_classify():
    degs = _degrees([0.05, 0.3, 0.6, 0.9])
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    r = T.region_partition(degs, beta, alpha)
    manual = {"AR": 0, "NR": 0, "CR": 0}
    for d in degs:
        manual[classify_degree(d, beta, alpha).value] += 1
    assert r.region_counts == manual


def test_exact_degrees_skip_empty_common_agenda():
    ratings = np.array([[1, 0], [0, 1], [1, -1]], dtype=np.int8)
    observed = np.array([[True, False], [False, True], [True, True]], dtype=bool)
    st = SituationTable(
        name="tiny",
        ratings=ratings,
        observed=observed,
        agent_ids=("x", "y", "z"),
        issue_ids=("i", "j"),
    )
    # x,y share no observed issue -> skipped; other pairs counted.
    degs = T.exact_pairwise_degrees(st, missing_policy="exclude", skip_empty_pairs=True)
    assert len(degs) == 2
    with pytest.raises(ValueError):
        T.exact_pairwise_degrees(
            st, missing_policy="exclude", skip_empty_pairs=False
        )
