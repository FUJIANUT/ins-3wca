"""Exact GreedyFlip implementation for unit-cost alliance transitions."""

from __future__ import annotations

from fractions import Fraction
from time import perf_counter
from typing import Sequence

from ..model import PairInstance, Region, as_fraction, classify_degree, validate_thresholds
from .common import (
    PairOption,
    SolveResult,
    SolveStatus,
    build_pair_options,
    evaluate_pair_selection,
    plan_from_selection,
)


def _excess(option: PairOption, threshold: Fraction) -> Fraction:
    return Fraction(option.post_n2, 2) - threshold * option.post_weight


def solve_greedy_flip(
    pair: PairInstance,
    *,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
    operators: Sequence[str] = ("mod", "del"),
    epsilon: Fraction | int | float | str = 0,
    transition_robust: bool = False,
) -> SolveResult:
    """Return a minimum-cardinality AR explanation or report infeasibility.

    Robust calls recompute every gain at q_epsilon=max(0,beta-epsilon).
    The implementation intentionally exposes only unit touch costs, the exact
    domain of the GreedyFlip theorem.
    """
    started = perf_counter()
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    epsilon_f = as_fraction(epsilon)
    if not 0 <= epsilon_f <= 1:
        raise ValueError("epsilon must lie in [0,1]")
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
            solver="GreedyFlip",
            target=Region.AR.value,
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
    threshold = max(Fraction(0), beta_f - epsilon_f)
    menus, _ = build_pair_options(
        pair, operators=operators, full_mod_menu=False, costs={"mod": 1, "del": 1, "rew": 1}
    )
    initial_excess = sum(_excess(menu[0], threshold) for menu in menus)
    selected = [0] * len(menus)
    if initial_excess <= 0:
        evaluation = evaluate_pair_selection(
            menus, selected, beta=beta_f, alpha=alpha_f
        )
        return SolveResult(
            status=SolveStatus.OPTIMAL,
            solver="GreedyFlip",
            target=Region.AR.value,
            objective_value=Fraction(0),
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=evaluation["degree"],
            original_region=original_region,
            post_region=evaluation["region"],
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            details={"effective_beta": str(threshold), "epsilon": str(epsilon_f)},
            validation={"valid": True, "robust_target_satisfied": True},
        )

    candidates: list[tuple[Fraction, int, int, PairOption]] = []
    for issue_index, menu in enumerate(menus):
        keep_excess = _excess(menu[0], threshold)
        best: tuple[Fraction, int, PairOption] | None = None
        for option_index, option in enumerate(menu[1:], start=1):
            gain = keep_excess - _excess(option, threshold)
            if gain <= 0:
                continue
            # Retaining actions win exact gain ties; IDs make all later ties stable.
            if best is None:
                best = (gain, option_index, option)
            else:
                best_gain, _, best_option = best
                if (
                    gain > best_gain
                    or (
                        gain == best_gain
                        and option.retains_positive_weight
                        and not best_option.retains_positive_weight
                    )
                    or (
                        gain == best_gain
                        and option.retains_positive_weight
                        == best_option.retains_positive_weight
                        and option.action_id < best_option.action_id
                    )
                ):
                    best = (gain, option_index, option)
        if best is not None:
            gain, option_index, option = best
            candidates.append((gain, issue_index, option_index, option))

    candidates.sort(
        key=lambda item: (
            -item[0],
            -int(item[3].retains_positive_weight),
            item[3].issue_id,
            item[3].action_id,
        )
    )
    covered = Fraction(0)
    for gain, issue_index, option_index, _ in candidates:
        selected[issue_index] = option_index
        covered += gain
        if covered >= initial_excess:
            break

    if covered < initial_excess:
        return SolveResult(
            status=SolveStatus.INFEASIBLE,
            solver="GreedyFlip",
            target=Region.AR.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            message="all positive gains are insufficient",
            details={"effective_beta": str(threshold)},
        )

    evaluation = evaluate_pair_selection(menus, selected, beta=beta_f, alpha=alpha_f)
    robust_ok = bool(
        evaluation["valid"] and evaluation["degree"] <= threshold
    )
    if not robust_ok:
        return SolveResult(
            status=SolveStatus.INFEASIBLE,
            solver="GreedyFlip",
            target=Region.AR.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=evaluation.get("degree"),
            original_region=original_region,
            post_region=evaluation.get("region"),
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            message=evaluation.get("reason", "the only algebraic cover empties the agenda"),
            details={"effective_beta": str(threshold)},
            validation=evaluation,
        )

    plan = plan_from_selection(menus, selected)
    return SolveResult(
        status=SolveStatus.OPTIMAL,
        solver="GreedyFlip",
        target=Region.AR.value,
        objective_value=evaluation["cost"],
        touched_issue_count=evaluation["touched"],
        plan=plan,
        original_degree=original,
        post_degree=evaluation["degree"],
        original_region=original_region,
        post_region=evaluation["region"],
        runtime_seconds=perf_counter() - started,
        guarantee="exact",
        details={
            "effective_beta": str(threshold),
            "epsilon": str(epsilon_f),
            "initial_excess": str(initial_excess),
            "covered_gain": str(covered),
        },
        validation={"valid": True, "robust_target_satisfied": robust_ok},
    )
