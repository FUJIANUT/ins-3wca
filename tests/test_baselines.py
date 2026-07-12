"""Tests for heuristic baselines and fairness metrics."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mcie import baselines as B
from mcie.model import PairInstance
from mcie.solvers import solve_greedy_flip, solve_grid_dp


def mkpair(a, b, w=None):
    a = np.array(a, dtype=np.int8)
    b = np.array(b, dtype=np.int8)
    if w is None:
        w = np.ones(len(a), dtype=np.int64)
    return PairInstance(
        "a", "b", tuple(f"i{i}" for i in range(len(a))), a, b, np.asarray(w), np.arange(len(a))
    )


def test_csr_ties_greedyflip_on_unit_cost_alliance():
    rng = np.random.default_rng(0)
    beta, alpha = Fraction(1, 5), Fraction(1, 2)
    ties = 0
    for _ in range(300):
        m = int(rng.integers(3, 10))
        # bias toward opposition so more pairs start in the conflict region
        a = rng.choice([-1, 1], size=m).astype(np.int8)
        b = (-a).copy()
        flip = rng.random(m) < 0.4
        b[flip] = rng.choice([-1, 0, 1], size=int(flip.sum())).astype(np.int8)
        p = mkpair(a, b)
        if p.exact_degree() < alpha:
            continue
        gf = solve_greedy_flip(p, beta=beta, alpha=alpha, operators=("mod",))
        csr = B.conflict_score_ranking(p, beta=beta, alpha=alpha, target="AR")
        assert csr.reached
        assert csr.cardinality == int(gf.objective_value)
        ties += 1
    assert ties >= 15  # exercised enough conflict pairs


def test_csr_overshoots_narrow_neutral_band_where_griddp_succeeds():
    # Two issues, both fully opposed, narrow band reachable only by a half-step.
    p = mkpair([1, 1], [-1, -1])  # D0 = 1
    beta, alpha = Fraction(3, 5), Fraction(9, 10)  # NR = (0.6, 0.9); 3/4 is inside
    gd = solve_grid_dp(p, beta=beta, alpha=alpha, operators=("mod",))
    assert gd.status.value == "OPTIMAL" and int(gd.objective_value) == 1
    csr = B.conflict_score_ranking(p, beta=beta, alpha=alpha, target="NR")
    # full-alignment step jumps 1 -> 1/2, skipping the open band
    assert csr.overshoot and not csr.reached


def test_zhai_ratio_reaches_alliance():
    p = mkpair([1, 1, 1, 1, 0, 1], [-1, -1, -1, -1, 0, -1], w=[1, 1, 1, 1, 1, 10])
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    r = B.zhai_cost_ratio(p, beta=beta, alpha=alpha, target="AR")
    assert r.reached and r.region == "AR"


def test_liu_and_random_reach_alliance_when_feasible():
    p = mkpair([1, 1, 1, 1, 0, 1], [-1, -1, -1, -1, 0, -1])
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    for r in (
        B.liu_issue_order(p, beta=beta, alpha=alpha, target="AR"),
        B.random_intervention(p, beta=beta, alpha=alpha, target="AR", seed=3),
    ):
        assert r.reached and r.region == "AR"


def test_liu_genetic_matches_exact_on_small_instances():
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    for m in (8, 12, 16):
        p = mkpair([1] * m, [-1] * m)
        gf = solve_greedy_flip(p, beta=beta, alpha=alpha, operators=("mod",))
        gd = solve_grid_dp(p, beta=beta, alpha=alpha, operators=("mod",))
        ga_ar = B.liu_genetic(p, beta=beta, alpha=alpha, target="AR", seed=1,
                              pop_size=50, generations=120)
        ga_nr = B.liu_genetic(p, beta=beta, alpha=alpha, target="NR", seed=1,
                              pop_size=50, generations=120)
        assert ga_ar.reached and ga_ar.cardinality == int(gf.objective_value)
        assert ga_nr.reached and ga_nr.cardinality == int(gd.objective_value)


def test_liu_genetic_is_deterministic_under_seed():
    p = mkpair([1] * 10, [-1] * 10)
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    r1 = B.liu_genetic(p, beta=beta, alpha=alpha, target="AR", seed=7, pop_size=30, generations=40)
    r2 = B.liu_genetic(p, beta=beta, alpha=alpha, target="AR", seed=7, pop_size=30, generations=40)
    assert r1.cardinality == r2.cardinality and r1.post_degree == r2.post_degree


def test_concession_imbalance_range():
    assert B.concession_imbalance({"a": 0, "b": 0}) == 0.0
    assert B.concession_imbalance({"a": 4, "b": 0}) == 1.0
    assert B.concession_imbalance({"a": 3, "b": 1}) == 0.5


def test_balanced_alignment_keeps_degree_and_reduces_imbalance():
    p = mkpair([1, 1, 1, -1, 0, 1], [-1, -1, -1, 1, 0, -1])  # D0 = 5/6
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    csr = B.conflict_score_ranking(p, beta=beta, alpha=alpha, target="AR")
    order = sorted(range(6), key=lambda t: -abs(int(p.attitudes_a[t]) - int(p.attitudes_b[t])))
    aligned = order[: csr.cardinality]
    bal = B.balanced_alignment(p, aligned, beta=beta, alpha=alpha, target="AR")
    assert bal.reached and bal.region == "AR"
    assert bal.cardinality == csr.cardinality
    assert bal.post_degree == csr.post_degree  # side choice does not change the degree
    assert B.concession_imbalance(bal.sides) <= B.concession_imbalance(csr.sides)
    assert B.concession_imbalance(bal.sides) <= Fraction(1, csr.cardinality)  # near-perfect balance


def test_region_of_uses_observed_mask():
    """Mask-consistent region: unobserved cells must not count as neutral."""
    import numpy as np
    from fractions import Fraction
    from run_extended import region_of
    from mcie.model import Region
    ratings = np.array([[1, 1, 1, 1], [-1, -1, 0, 0]], dtype=np.int8)
    weights = np.ones(4, dtype=np.int64)
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    # all observed: d=(1,1,1/2,1/2) -> D=3/4 -> CR (closed)
    obs_all = np.ones((2, 4), dtype=bool)
    assert region_of(ratings, weights, obs_all, 0, 1, beta, alpha) == Region.CR
    # only first two issues co-observed: d=(1,1) -> D=1 -> CR, and the stored 0s ignored
    obs = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    assert region_of(ratings, weights, obs, 0, 1, beta, alpha) == Region.CR
    # no co-observed issue -> None
    obs_none = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=bool)
    assert region_of(ratings, weights, obs_none, 0, 1, beta, alpha) is None


def test_necessary_issue_uses_full_theta():
    """Corollary 14 necessary test needs Theta=sum over I (bad t necessary iff v_t>g)."""
    from fractions import Fraction
    # agenda: two bad issues v=+3,+1 and one good issue v=-2 (g=2, Theta=2)
    vt = [Fraction(3), Fraction(1), Fraction(-2)]
    B_idx = [t for t, x in enumerate(vt) if x > 0]
    Theta = sum(vt)  # correct: 2
    nec = sum(1 for t in B_idx if sum(vt[s] for s in B_idx if s != t) < Theta)
    # v_0=3>g=2 -> necessary; v_1=1<=g=2 -> not necessary. fraction 1/2.
    assert nec == 1 and len(B_idx) == 2
    # the buggy Theta=sum(B)=4 would give nec=2 (fraction 1.0)
    bad_theta = sum(vt[t] for t in B_idx)
    bad_nec = sum(1 for t in B_idx if sum(vt[s] for s in B_idx if s != t) < bad_theta)
    assert bad_nec == 2  # demonstrates the old bug inflated it to all-necessary
