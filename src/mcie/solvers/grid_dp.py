"""Exact (W,N2) dynamic program for finite-menu neutrality transitions."""

from __future__ import annotations

from fractions import Fraction
from time import perf_counter
from typing import Mapping, Sequence

from ..model import PairInstance, Region, as_fraction, classify_degree, validate_thresholds
from .common import (
    SolveResult,
    SolveStatus,
    build_pair_options,
    evaluate_pair_selection,
    plan_from_selection,
)


def solve_grid_dp(
    pair: PairInstance,
    *,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
    operators: Sequence[str] = ("mod",),
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    epsilon: Fraction | int | float | str = 0,
    transition_robust: bool = False,
    max_states: int = 2_000_000,
    max_scaled_weight: int = 20_000,
    time_limit: float | None = None,
) -> SolveResult:
    """Solve finite-menu MCIE-NR exactly by sparse dynamic programming.

    States use integer ``(W,N2)`` with ``N2=2*sum(w*d)``.  Open boundaries
    are checked by exact Fraction comparisons, including robust tightening.
    """
    started = perf_counter()
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    epsilon_f = as_fraction(epsilon)
    if not 0 <= epsilon_f <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    lower, upper = beta_f + epsilon_f, alpha_f - epsilon_f
    original = pair.exact_degree()
    original_region = classify_degree(original, beta_f, alpha_f)
    source_guard = (
        min(Fraction(1), alpha_f + epsilon_f)
        if transition_robust
        else alpha_f
    )
    if original < source_guard:
        return SolveResult(
            status=SolveStatus.INVALID_SOURCE,
            solver="GridDP",
            target=Region.NR.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            message=f"source degree {original} is below required CR guard {source_guard}",
        )
    if lower >= upper:
        return SolveResult(
            status=SolveStatus.INFEASIBLE,
            solver="GridDP",
            target=Region.NR.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            message="the robust neutrality interval is empty",
            details={"effective_lower": str(lower), "effective_upper": str(upper)},
        )

    menus, integer_weights = build_pair_options(
        pair,
        operators=operators,
        full_mod_menu=True,
        costs=costs,
        max_scaled_weight=max_scaled_weight,
    )
    dp: dict[tuple[int, int], Fraction] = {(0, 0): Fraction(0)}
    parents: list[dict[tuple[int, int], tuple[tuple[int, int], int]]] = []
    states_generated = 1

    for issue_index, menu in enumerate(menus):
        next_dp: dict[tuple[int, int], Fraction] = {}
        parent: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        for state, current_cost in dp.items():
            weight, n2 = state
            for option_index, option in enumerate(menu):
                new_state = (weight + option.post_weight, n2 + option.post_n2)
                new_cost = current_cost + option.cost
                old_cost = next_dp.get(new_state)
                if old_cost is None or new_cost < old_cost:
                    next_dp[new_state] = new_cost
                    parent[new_state] = (state, option_index)
        states_generated += len(next_dp)
        if len(next_dp) > max_states:
            return SolveResult(
                status=SolveStatus.RESOURCE_LIMIT,
                solver="GridDP",
                target=Region.NR.value,
                objective_value=None,
                touched_issue_count=0,
                plan=[],
                original_degree=original,
                post_degree=None,
                original_region=original_region,
                post_region=None,
                runtime_seconds=perf_counter() - started,
                guarantee="exact-if-completed",
                message=(
                    f"state limit exceeded after issue {issue_index}: "
                    f"{len(next_dp)} > {max_states}"
                ),
                details={"states_generated": states_generated},
            )
        if time_limit is not None and perf_counter() - started > time_limit:
            return SolveResult(
                status=SolveStatus.RESOURCE_LIMIT,
                solver="GridDP",
                target=Region.NR.value,
                objective_value=None,
                touched_issue_count=0,
                plan=[],
                original_degree=original,
                post_degree=None,
                original_region=original_region,
                post_region=None,
                runtime_seconds=perf_counter() - started,
                guarantee="exact-if-completed",
                message=f"time limit exceeded after issue {issue_index}",
                details={"states_generated": states_generated},
            )
        parents.append(parent)
        dp = next_dp

    feasible: list[tuple[Fraction, tuple[int, int]]] = []
    for state, cost in dp.items():
        weight, n2 = state
        if weight <= 0:
            continue
        degree = Fraction(n2, 2 * weight)
        if lower < degree < upper:
            feasible.append((cost, state))
    if not feasible:
        return SolveResult(
            status=SolveStatus.INFEASIBLE,
            solver="GridDP",
            target=Region.NR.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            message="no reachable state lies in the open neutrality interval",
            details={
                "effective_lower": str(lower),
                "effective_upper": str(upper),
                "final_states": len(dp),
                "states_generated": states_generated,
                "scaled_total_weight": int(integer_weights.sum()),
            },
        )

    best_cost, best_state = min(feasible, key=lambda item: (item[0], item[1]))
    selected = [0] * len(menus)
    state = best_state
    for issue_index in range(len(menus) - 1, -1, -1):
        previous, option_index = parents[issue_index][state]
        selected[issue_index] = option_index
        state = previous
    if state != (0, 0):
        raise RuntimeError("GridDP backtracking did not return to the origin")

    evaluation = evaluate_pair_selection(menus, selected, beta=beta_f, alpha=alpha_f)
    robust_ok = bool(
        evaluation["valid"] and lower < evaluation["degree"] < upper
    )
    if not robust_ok:
        raise RuntimeError("independent validation rejected the GridDP optimum")
    plan = plan_from_selection(menus, selected)
    return SolveResult(
        status=SolveStatus.OPTIMAL,
        solver="GridDP",
        target=Region.NR.value,
        objective_value=best_cost,
        touched_issue_count=evaluation["touched"],
        plan=plan,
        original_degree=original,
        post_degree=evaluation["degree"],
        original_region=original_region,
        post_region=evaluation["region"],
        runtime_seconds=perf_counter() - started,
        guarantee="exact",
        details={
            "effective_lower": str(lower),
            "effective_upper": str(upper),
            "final_states": len(dp),
            "states_generated": states_generated,
            "scaled_total_weight": int(integer_weights.sum()),
        },
        validation={"valid": True, "robust_target_satisfied": robust_ok},
    )
