from fractions import Fraction
import random

import numpy as np

from mcie.model import SituationTable
from mcie.solvers import (
    SolveStatus,
    solve_greedy_flip,
    solve_grid_dp,
    solve_pair_ilp,
    solve_safe_ilp,
)


def _pair(a, b, weights=None):
    m = len(a)
    table = SituationTable(
        "random",
        np.asarray([a, b], dtype=np.int8),
        np.ones((2, m), dtype=bool),
        ("a", "b"),
        tuple(f"i{index}" for index in range(m)),
        weights=np.ones(m, dtype=np.int64) if weights is None else np.asarray(weights),
    )
    return table.pair("a", "b")


def _optimum(result):
    if result.status == SolveStatus.OPTIMAL:
        return result.objective_value
    if result.status == SolveStatus.INFEASIBLE:
        return None
    raise AssertionError((result.status, result.message, result.details))


def test_greedyflip_matches_ilp_on_random_t2_instances():
    rng = random.Random(20260711)
    checked = 0
    while checked < 60:
        m = rng.randint(2, 7)
        a = [rng.choice((-1, 0, 1)) for _ in range(m)]
        b = [rng.choice((-1, 0, 1)) for _ in range(m)]
        pair = _pair(a, b, [rng.randint(1, 6) for _ in range(m)])
        if pair.exact_degree() < Fraction(3, 5):
            continue
        operators = rng.choice((('mod',), ('del',), ('mod', 'del')))
        greedy = solve_greedy_flip(
            pair,
            beta=Fraction(3, 10),
            alpha=Fraction(3, 5),
            operators=operators,
        )
        ilp = solve_pair_ilp(
            pair,
            target="AR",
            beta=Fraction(3, 10),
            alpha=Fraction(3, 5),
            operators=operators,
        )
        assert _optimum(greedy) == _optimum(ilp)
        checked += 1


def test_griddp_matches_ilp_on_random_neutrality_instances():
    rng = random.Random(20260712)
    checked = 0
    while checked < 50:
        m = rng.randint(2, 7)
        a = [rng.choice((-1, 0, 1)) for _ in range(m)]
        b = [rng.choice((-1, 0, 1)) for _ in range(m)]
        pair = _pair(a, b, [rng.randint(1, 5) for _ in range(m)])
        if pair.exact_degree() < Fraction(7, 10):
            continue
        operators = rng.choice((('mod',), ('mod', 'del')))
        grid = solve_grid_dp(
            pair,
            beta=Fraction(3, 10),
            alpha=Fraction(7, 10),
            operators=operators,
            max_states=100_000,
        )
        ilp = solve_pair_ilp(
            pair,
            target="NR",
            beta=Fraction(3, 10),
            alpha=Fraction(7, 10),
            operators=operators,
        )
        assert _optimum(grid) == _optimum(ilp)
        checked += 1


def test_all_delete_is_not_a_certificate():
    pair = _pair([1, 1, 1], [-1, -1, -1])
    greedy = solve_greedy_flip(
        pair,
        beta=Fraction(1, 4),
        alpha=Fraction(3, 4),
        operators=("del",),
    )
    ilp = solve_pair_ilp(
        pair,
        target="AR",
        beta=Fraction(1, 4),
        alpha=Fraction(3, 4),
        operators=("del",),
    )
    assert greedy.status == SolveStatus.INFEASIBLE
    assert ilp.status == SolveStatus.INFEASIBLE


def test_robust_ar_optimum_is_monotone_and_matches_ilp():
    pair = _pair([1, 1, 1, 1], [-1, -1, -1, 0])
    costs = []
    for epsilon in (Fraction(0), Fraction(1, 20), Fraction(1, 10)):
        greedy = solve_greedy_flip(
            pair,
            beta=Fraction(2, 5),
            alpha=Fraction(7, 10),
            epsilon=epsilon,
            transition_robust=True,
            operators=("mod", "del"),
        )
        ilp = solve_pair_ilp(
            pair,
            target="AR",
            beta=Fraction(2, 5),
            alpha=Fraction(7, 10),
            epsilon=epsilon,
            transition_robust=True,
            operators=("mod", "del"),
        )
        assert _optimum(greedy) == _optimum(ilp)
        costs.append(float("inf") if greedy.objective_value is None else greedy.objective_value)
    assert costs == sorted(costs)


def test_safe_ilp_matches_t5_cycle_gadget():
    n, k = 6, 3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    m = 2 * n - k
    agents = ["a", "b", "g", *(f"c{i}" for i in range(len(edges)))]
    ratings = np.zeros((len(agents), m), dtype=np.int8)
    ratings[0, :] = 1
    ratings[1, :n] = -1
    ratings[1, n:] = 1
    ratings[2, :] = -1
    for edge_index, edge in enumerate(edges):
        for issue in range(n):
            ratings[3 + edge_index, issue] = 1 if issue in edge else 0
    table = SituationTable(
        "t5-cycle",
        ratings,
        np.ones_like(ratings, dtype=bool),
        tuple(agents),
        tuple(f"i{issue}" for issue in range(m)),
    )
    result = solve_safe_ilp(
        table,
        agent_a="a",
        agent_b="b",
        beta=Fraction(n - k, m),
        alpha=Fraction(2 * n - k + 2, 2 * m),
        operators=("mod", "del", "rew"),
        time_limit=30,
    )
    assert result.status == SolveStatus.OPTIMAL, result.to_json()
    assert result.objective_value == k
    assert result.validation == {
        "valid": True,
        "target_satisfied": True,
        "safe_violation_count": 0,
    }
    chosen = {action.issue_index for action in result.plan}
    assert len(chosen) == k
    assert all(issue < n for issue in chosen)
    assert not any(left in chosen and right in chosen for left, right in edges)
