"""High-value exactness and boundary tests for the MCIE solvers."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
import random
from typing import Mapping, Sequence

import numpy as np
import pytest

from mcie.model import PairInstance, Region, SituationTable, classify_degree
from mcie.solvers import (
    SolveStatus,
    solve_greedy_flip,
    solve_grid_dp,
    solve_pair_ilp,
    solve_safe_ilp,
)
from mcie.solvers.common import build_pair_options, evaluate_pair_selection


BETA = Fraction(1, 4)
ALPHA = Fraction(3, 4)

# Representatives whose absolute difference is the requested doubled distance.
_ATTITUDES = {0: (0, 0), 1: (0, 1), 2: (-1, 1)}


def _pair(
    doubled_distances: Sequence[int], weights: Sequence[int] | None = None
) -> PairInstance:
    attitudes = [_ATTITUDES[value] for value in doubled_distances]
    left, right = zip(*attitudes)
    size = len(doubled_distances)
    return PairInstance(
        agent_a="a",
        agent_b="b",
        issue_ids=tuple(f"i{index}" for index in range(size)),
        attitudes_a=np.asarray(left, dtype=np.int8),
        attitudes_b=np.asarray(right, dtype=np.int8),
        weights=np.asarray(weights if weights is not None else [1] * size),
        source_issue_indices=np.arange(size),
    )


def _brute_pair_cost(
    pair: PairInstance,
    *,
    target: Region,
    operators: Sequence[str],
    costs: Mapping[str, Fraction | int] | None = None,
    epsilon: Fraction = Fraction(0),
) -> Fraction | None:
    """Independent finite-menu oracle, including the nonempty-agenda rule."""
    menus, _ = build_pair_options(
        pair,
        operators=operators,
        full_mod_menu=target == Region.NR,
        costs=costs,
    )
    best: Fraction | None = None
    lower, upper = BETA + epsilon, ALPHA - epsilon
    alliance_threshold = max(Fraction(0), BETA - epsilon)
    for selection in product(*(range(len(menu)) for menu in menus)):
        evaluation = evaluate_pair_selection(
            menus, selection, beta=BETA, alpha=ALPHA
        )
        if not evaluation["valid"]:
            continue
        degree = evaluation["degree"]
        feasible = (
            degree <= alliance_threshold
            if target == Region.AR
            else lower < degree < upper
        )
        if feasible and (best is None or evaluation["cost"] < best):
            best = evaluation["cost"]
    return best


def _assert_matches_oracle(result, expected: Fraction | None) -> None:
    if expected is None:
        assert result.status == SolveStatus.INFEASIBLE, result.to_json()
        assert result.objective_value is None
    else:
        assert result.status == SolveStatus.OPTIMAL, result.to_json()
        assert result.objective_value == expected
        assert result.validation.get("valid") is True


def _safe_table() -> SituationTable:
    # Target a::b is CR.  Both bystander pairs are NR.
    ratings = np.asarray(
        [
            [-1, -1],
            [1, 1],
            [-1, 1],
        ],
        dtype=np.int8,
    )
    return SituationTable(
        name="safe-toy",
        ratings=ratings,
        observed=np.ones_like(ratings, dtype=bool),
        agent_ids=("a", "b", "c"),
        issue_ids=("i0", "i1"),
        weights=np.ones(2, dtype=np.int64),
    )


def _apply_safe_plan(situation: SituationTable, plan) -> tuple[np.ndarray, np.ndarray]:
    ratings = situation.ratings.copy()
    active = np.ones(situation.n_issues, dtype=bool)
    for action in plan:
        if action.kind == "mod":
            ratings[int(action.metadata["agent_index"]), action.issue_index] = int(
                action.metadata["new_rating"]
            )
        elif action.kind in {"del", "rew"}:
            active[action.issue_index] = False
        else:  # pragma: no cover - a solver plan must use a known action kind
            raise AssertionError(action.kind)
    return ratings, active


def _region(
    ratings: np.ndarray,
    active: np.ndarray,
    weights: np.ndarray,
    left: int,
    right: int,
) -> Region:
    assert active.any(), "the test oracle also forbids the empty agenda"
    active_weights = np.asarray(weights, dtype=np.int64)[active]
    doubled_distances = np.abs(
        ratings[left, active].astype(np.int16)
        - ratings[right, active].astype(np.int16)
    )
    degree = Fraction(
        int(np.dot(active_weights, doubled_distances)),
        2 * int(active_weights.sum()),
    )
    return classify_degree(degree, BETA, ALPHA)


def test_exact_source_boundary_and_target_boundaries() -> None:
    # D0=(1+1/2)/2=alpha: equality belongs to CR and must be accepted.
    pair = _pair([2, 1])
    assert pair.exact_degree() == ALPHA

    greedy = solve_greedy_flip(pair, beta=BETA, alpha=ALPHA)
    ar_ilp = solve_pair_ilp(pair, target=Region.AR, beta=BETA, alpha=ALPHA)
    for result in (greedy, ar_ilp):
        assert result.status == SolveStatus.OPTIMAL, result.to_json()
        assert result.objective_value == 1
        assert result.post_degree == BETA  # AR has a closed upper boundary.
        assert result.post_region == Region.AR

    grid = solve_grid_dp(pair, beta=BETA, alpha=ALPHA)
    nr_ilp = solve_pair_ilp(pair, target=Region.NR, beta=BETA, alpha=ALPHA)
    for result in (grid, nr_ilp):
        assert result.status == SolveStatus.OPTIMAL, result.to_json()
        assert result.objective_value == 1
        assert BETA < result.post_degree < ALPHA
        assert result.post_region == Region.NR


@pytest.mark.parametrize(
    ("beta", "alpha"),
    [
        (Fraction(1, 2), Fraction(3, 4)),  # only mod_d1 reaches beta
        (Fraction(1, 4), Fraction(1, 2)),  # only mod_d1 reaches alpha
    ],
)
def test_grid_and_ilp_reject_open_nr_boundaries(beta, alpha) -> None:
    pair = _pair([2])
    grid = solve_grid_dp(pair, beta=beta, alpha=alpha, operators=("mod",))
    ilp = solve_pair_ilp(
        pair, target=Region.NR, beta=beta, alpha=alpha, operators=("mod",)
    )
    assert grid.status == SolveStatus.INFEASIBLE
    assert ilp.status == SolveStatus.INFEASIBLE


def test_empty_agenda_is_never_a_solution() -> None:
    pair = _pair([2])
    results = (
        solve_greedy_flip(
            pair, beta=Fraction(0), alpha=Fraction(1, 2), operators=("del",)
        ),
        solve_grid_dp(
            pair, beta=Fraction(0), alpha=Fraction(1, 2), operators=("del",)
        ),
        solve_pair_ilp(
            pair,
            target=Region.AR,
            beta=Fraction(0),
            alpha=Fraction(1, 2),
            operators=("del",),
        ),
        solve_pair_ilp(
            pair,
            target=Region.NR,
            beta=Fraction(0),
            alpha=Fraction(1, 2),
            operators=("del",),
        ),
    )
    assert all(result.status == SolveStatus.INFEASIBLE for result in results)


def test_greedy_retains_weight_on_modification_deletion_gain_tie() -> None:
    # At beta=0, mod-to-zero and deletion have equal algebraic gain.  Deletion
    # empties this one-issue agenda, so the theorem's retaining tie-break matters.
    pair = _pair([2])
    result = solve_greedy_flip(
        pair,
        beta=Fraction(0),
        alpha=Fraction(1, 2),
        operators=("mod", "del"),
    )
    assert result.status == SolveStatus.OPTIMAL, result.to_json()
    assert result.post_degree == 0
    assert len(result.plan) == 1
    assert result.plan[0].kind == "mod"
    assert result.plan[0].post_weight > 0


def test_source_guard_and_epsilon_validation_are_consistent() -> None:
    non_cr = _pair([1])
    results = (
        solve_greedy_flip(non_cr, beta=BETA, alpha=ALPHA),
        solve_grid_dp(non_cr, beta=BETA, alpha=ALPHA),
        solve_pair_ilp(non_cr, target=Region.AR, beta=BETA, alpha=ALPHA),
    )
    assert all(result.status == SolveStatus.INVALID_SOURCE for result in results)

    cr = _pair([2])
    calls = (
        lambda epsilon: solve_greedy_flip(
            cr, beta=BETA, alpha=ALPHA, epsilon=epsilon
        ),
        lambda epsilon: solve_grid_dp(
            cr, beta=BETA, alpha=ALPHA, epsilon=epsilon
        ),
        lambda epsilon: solve_pair_ilp(
            cr, target=Region.AR, beta=BETA, alpha=ALPHA, epsilon=epsilon
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="epsilon"):
            call(Fraction(-1, 10))
        with pytest.raises(ValueError, match="epsilon"):
            call(Fraction(11, 10))


def test_robust_clipping_and_transition_source_guard() -> None:
    endpoint_source = _pair([2])
    for result in (
        solve_greedy_flip(
            endpoint_source,
            beta=BETA,
            alpha=ALPHA,
            epsilon=BETA,
            transition_robust=True,
        ),
        solve_pair_ilp(
            endpoint_source,
            target=Region.AR,
            beta=BETA,
            alpha=ALPHA,
            epsilon=BETA,
            transition_robust=True,
        ),
    ):
        assert result.status == SolveStatus.OPTIMAL, result.to_json()
        assert result.post_degree == 0  # q_epsilon=max(0,beta-epsilon)

    boundary_source = _pair([2, 1])  # D0=alpha, below alpha+epsilon.
    for result in (
        solve_greedy_flip(
            boundary_source,
            beta=BETA,
            alpha=ALPHA,
            epsilon=Fraction(1, 10),
            transition_robust=True,
        ),
        solve_grid_dp(
            boundary_source,
            beta=BETA,
            alpha=ALPHA,
            epsilon=Fraction(1, 10),
            transition_robust=True,
        ),
        solve_pair_ilp(
            boundary_source,
            target=Region.AR,
            beta=BETA,
            alpha=ALPHA,
            epsilon=Fraction(1, 10),
            transition_robust=True,
        ),
    ):
        assert result.status == SolveStatus.INVALID_SOURCE


def test_greedy_and_pair_ilp_match_exhaustive_ar_oracle() -> None:
    rng = random.Random(20_260_711)
    operator_sets = (("mod",), ("del",), ("mod", "del"), ("rew",))
    checked = 0
    while checked < 30:
        size = rng.randint(1, 5)
        pair = _pair(
            [rng.randrange(3) for _ in range(size)],
            [rng.randint(1, 4) for _ in range(size)],
        )
        if pair.exact_degree() < ALPHA:
            continue
        operators = operator_sets[checked % len(operator_sets)]
        expected = _brute_pair_cost(
            pair, target=Region.AR, operators=operators
        )
        greedy = solve_greedy_flip(
            pair, beta=BETA, alpha=ALPHA, operators=operators
        )
        ilp = solve_pair_ilp(
            pair,
            target=Region.AR,
            beta=BETA,
            alpha=ALPHA,
            operators=operators,
        )
        _assert_matches_oracle(greedy, expected)
        _assert_matches_oracle(ilp, expected)
        checked += 1


def test_grid_dp_and_pair_ilp_match_exhaustive_nr_oracle() -> None:
    rng = random.Random(937)
    operator_sets = (
        ("mod",),
        ("del",),
        ("mod", "del"),
        ("rew",),
        ("mod", "rew"),
    )
    checked = 0
    while checked < 30:
        size = rng.randint(1, 5)
        pair = _pair(
            [rng.randrange(3) for _ in range(size)],
            [rng.randint(1, 4) for _ in range(size)],
        )
        if pair.exact_degree() < ALPHA:
            continue
        operators = operator_sets[checked % len(operator_sets)]
        costs = {
            "mod": Fraction(rng.randint(1, 6), 2),
            "del": Fraction(rng.randint(1, 6), 2),
            "rew": Fraction(rng.randint(1, 6), 2),
        }
        expected = _brute_pair_cost(
            pair, target=Region.NR, operators=operators, costs=costs
        )
        grid = solve_grid_dp(
            pair,
            beta=BETA,
            alpha=ALPHA,
            operators=operators,
            costs=costs,
        )
        ilp = solve_pair_ilp(
            pair,
            target=Region.NR,
            beta=BETA,
            alpha=ALPHA,
            operators=operators,
            costs=costs,
        )
        _assert_matches_oracle(grid, expected)
        _assert_matches_oracle(ilp, expected)
        checked += 1


def test_grid_dp_reports_state_budget_exhaustion() -> None:
    result = solve_grid_dp(
        _pair([2, 2]),
        beta=BETA,
        alpha=ALPHA,
        operators=("mod",),
        max_states=1,
    )
    assert result.status == SolveStatus.RESOURCE_LIMIT
    assert "state limit exceeded" in result.message
    assert result.objective_value is None


def test_safe_ilp_preserves_every_bystander_region() -> None:
    situation = _safe_table()
    original_active = np.ones(situation.n_issues, dtype=bool)
    original_regions = {
        pair: _region(
            situation.ratings, original_active, situation.weights, *pair
        )
        for pair in ((0, 2), (1, 2))
    }
    result = solve_safe_ilp(
        situation,
        agent_a="a",
        agent_b="b",
        beta=BETA,
        alpha=ALPHA,
        target=Region.AR,
        operators=("mod",),
        time_limit=None,
    )
    assert result.status == SolveStatus.OPTIMAL, result.to_json()
    assert result.objective_value == 2
    assert result.validation == {
        "valid": True,
        "target_satisfied": True,
        "safe_violation_count": 0,
    }

    changed_ratings, changed_active = _apply_safe_plan(situation, result.plan)
    assert changed_active.any()
    assert _region(
        changed_ratings, changed_active, situation.weights, 0, 1
    ) == Region.AR
    for pair, original_region in original_regions.items():
        assert (
            _region(changed_ratings, changed_active, situation.weights, *pair)
            == original_region
        )


def test_safe_subset_audit_changes_the_optimum_and_full_delete_is_forbidden() -> None:
    situation = _safe_table()
    all_pairs = solve_safe_ilp(
        situation,
        agent_a="a",
        agent_b="b",
        beta=BETA,
        alpha=ALPHA,
        target=Region.NR,
        operators=("mod",),
        time_limit=None,
    )
    selected_pair = solve_safe_ilp(
        situation,
        agent_a="a",
        agent_b="b",
        beta=BETA,
        alpha=ALPHA,
        target=Region.NR,
        operators=("mod",),
        bystander_pairs=(("b", "c"),),
        time_limit=None,
    )
    assert all_pairs.status == selected_pair.status == SolveStatus.OPTIMAL
    assert all_pairs.objective_value == 2
    assert selected_pair.objective_value == 1
    assert selected_pair.details["audited_bystander_pairs"] == 1

    deletion_only = solve_safe_ilp(
        situation,
        agent_a="a",
        agent_b="b",
        beta=BETA,
        alpha=ALPHA,
        target=Region.AR,
        operators=("del",),
        time_limit=None,
    )
    assert deletion_only.status == SolveStatus.INFEASIBLE
    assert deletion_only.objective_value is None


def test_safe_ilp_requires_explicit_missingness_policy() -> None:
    situation = _safe_table()
    situation.observed[2, 0] = False
    with pytest.raises(ValueError, match="complete table"):
        solve_safe_ilp(
            situation,
            agent_a="a",
            agent_b="b",
            beta=BETA,
            alpha=ALPHA,
            missing_policy="error",
        )

    result = solve_safe_ilp(
        situation,
        agent_a="a",
        agent_b="b",
        beta=BETA,
        alpha=ALPHA,
        missing_policy="neutral",
        time_limit=None,
    )
    assert result.status == SolveStatus.OPTIMAL
    assert result.details["missing_policy"] == "neutral"
