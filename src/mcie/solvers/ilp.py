"""SciPy/HiGHS MILP models for pair-level MCIE and hard MCIE-Safe."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import gcd
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from ..model import (
    PairInstance,
    Region,
    SituationTable,
    as_fraction,
    classify_degree,
    degree_from_integer_state,
    integerize_weights,
    validate_thresholds,
)
from .common import (
    PlanAction,
    SolveResult,
    SolveStatus,
    build_pair_options,
    evaluate_pair_selection,
    linear_score,
    plan_from_selection,
)


@dataclass(frozen=True)
class _GlobalVariable:
    issue_index: int
    issue_id: str
    action_id: str
    kind: str
    cost: Fraction
    metadata: dict[str, Any]


def _sparse_constraints(
    rows: Sequence[Mapping[int, int | float]],
    lower: Sequence[float],
    upper: Sequence[float],
    n_variables: int,
) -> LinearConstraint:
    data: list[float] = []
    row_indices: list[int] = []
    col_indices: list[int] = []
    for row_index, row in enumerate(rows):
        for column, value in row.items():
            if value:
                row_indices.append(row_index)
                col_indices.append(int(column))
                data.append(float(value))
    matrix = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(rows), n_variables),
    ).tocsr()
    return LinearConstraint(
        matrix,
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


def _milp_options(
    *,
    time_limit: float | None,
    mip_rel_gap: float | None,
    node_limit: int | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"disp": False, "presolve": True}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if mip_rel_gap is not None:
        options["mip_rel_gap"] = float(mip_rel_gap)
    if node_limit is not None:
        options["node_limit"] = int(node_limit)
    return options


def _exact_objective(costs: Sequence[Fraction]) -> tuple[np.ndarray, Fraction]:
    """Scale nonnegative rational costs to a primitive integer objective."""
    lcm = 1
    for cost in costs:
        if cost < 0:
            raise ValueError("MILP action costs must be nonnegative")
        lcm = lcm * cost.denominator // gcd(lcm, cost.denominator)
    integers = [cost.numerator * (lcm // cost.denominator) for cost in costs]
    common = 0
    for value in integers:
        common = gcd(common, value)
    common = common or 1
    integers = [value // common for value in integers]
    if integers and max(integers) > 2**52:
        raise ValueError("scaled rational objective exceeds exact float integer range")
    return np.asarray(integers, dtype=float), Fraction(common, lcm)


def _solver_details(result: Any, *, n_variables: int, n_constraints: int) -> dict[str, Any]:
    details: dict[str, Any] = {
        "backend": "scipy.optimize.milp/HiGHS",
        "n_variables": n_variables,
        "n_constraints": n_constraints,
        "highs_status": int(result.status),
        "highs_message": str(result.message),
    }
    for name in ("mip_gap", "mip_node_count", "mip_dual_bound"):
        value = getattr(result, name, None)
        if value is not None:
            details[name] = float(value)
    return details


def solve_pair_ilp(
    pair: PairInstance,
    *,
    target: str | Region,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
    operators: Sequence[str] = ("mod", "del"),
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    issue_costs: Sequence[Fraction | int | float | str] | None = None,
    epsilon: Fraction | int | float | str = 0,
    transition_robust: bool = False,
    max_scaled_weight: int = 2_000_000,
    time_limit: float | None = None,
    mip_rel_gap: float | None = 0.0,
    node_limit: int | None = None,
) -> SolveResult:
    """Solve a finite-menu single-pair AR or NR instance exactly by MILP."""
    started = perf_counter()
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    epsilon_f = as_fraction(epsilon)
    if not 0 <= epsilon_f <= 1:
        raise ValueError("epsilon must lie in [0,1]")
    target_region = Region(str(target).upper()) if not isinstance(target, Region) else target
    if target_region not in {Region.AR, Region.NR}:
        raise ValueError("target must be AR or NR")
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
            solver="ILP",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact-if-optimal",
            message=f"source degree {original} is below required CR guard {source_guard}",
        )
    lower = beta_f + epsilon_f
    upper = alpha_f - epsilon_f
    alliance_threshold = max(Fraction(0), beta_f - epsilon_f)
    if target_region == Region.NR and lower >= upper:
        return SolveResult(
            status=SolveStatus.INFEASIBLE,
            solver="ILP",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact-if-optimal",
            message="the robust neutrality interval is empty",
        )

    menus, integer_weights = build_pair_options(
        pair,
        operators=operators,
        full_mod_menu=target_region == Region.NR,
        costs=costs,
        issue_costs=issue_costs,
        max_scaled_weight=max_scaled_weight,
    )
    flat_options = [option for menu in menus for option in menu]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for menu in menus:
        offsets.append((cursor, cursor + len(menu)))
        cursor += len(menu)
    n_variables = len(flat_options)
    objective, objective_unit = _exact_objective(
        [option.cost for option in flat_options]
    )

    rows: list[dict[int, int]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for start, stop in offsets:
        rows.append({index: 1 for index in range(start, stop)})
        lower_bounds.append(1)
        upper_bounds.append(1)
    rows.append(
        {index: option.post_weight for index, option in enumerate(flat_options)}
    )
    lower_bounds.append(1)
    upper_bounds.append(np.inf)

    def add_score_constraint(
        threshold: Fraction, constraint_lower: float, constraint_upper: float
    ) -> None:
        rows.append(
            {
                index: linear_score(option, threshold)
                for index, option in enumerate(flat_options)
            }
        )
        lower_bounds.append(constraint_lower)
        upper_bounds.append(constraint_upper)

    if target_region == Region.AR:
        add_score_constraint(alliance_threshold, -np.inf, 0)
    else:
        add_score_constraint(lower, 1, np.inf)
        add_score_constraint(upper, -np.inf, -1)

    constraint = _sparse_constraints(
        rows, lower_bounds, upper_bounds, n_variables
    )
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraint,
        options=_milp_options(
            time_limit=time_limit, mip_rel_gap=mip_rel_gap, node_limit=node_limit
        ),
    )
    details = _solver_details(
        result, n_variables=n_variables, n_constraints=len(rows)
    )
    details["scaled_total_weight"] = int(integer_weights.sum())
    details["objective_unit"] = str(objective_unit)
    if result.x is None:
        status = SolveStatus.INFEASIBLE if result.status == 2 else SolveStatus.ERROR
        return SolveResult(
            status=status,
            solver="ILP",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact-if-optimal",
            message=str(result.message),
            details=details,
        )

    selected: list[int] = []
    for start, stop in offsets:
        selected.append(int(np.argmax(result.x[start:stop])))
    evaluation = evaluate_pair_selection(menus, selected, beta=beta_f, alpha=alpha_f)
    if target_region == Region.AR:
        target_ok = bool(
            evaluation["valid"] and evaluation["degree"] <= alliance_threshold
        )
    else:
        target_ok = bool(
            evaluation["valid"] and lower < evaluation["degree"] < upper
        )
    if not target_ok:
        return SolveResult(
            status=SolveStatus.ERROR,
            solver="ILP",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=evaluation.get("degree"),
            original_region=original_region,
            post_region=evaluation.get("region"),
            runtime_seconds=perf_counter() - started,
            guarantee="invalid-incumbent",
            message="independent exact validator rejected the MILP incumbent",
            details=details,
            validation=evaluation,
        )

    solve_status = (
        SolveStatus.OPTIMAL if result.status == 0 else SolveStatus.FEASIBLE_TIMEOUT
    )
    guarantee = "exact" if result.status == 0 else "validated-incumbent"
    return SolveResult(
        status=solve_status,
        solver="ILP",
        target=target_region.value,
        objective_value=evaluation["cost"],
        touched_issue_count=evaluation["touched"],
        plan=plan_from_selection(menus, selected),
        original_degree=original,
        post_degree=evaluation["degree"],
        original_region=original_region,
        post_region=evaluation["region"],
        runtime_seconds=perf_counter() - started,
        guarantee=guarantee,
        message=str(result.message),
        details=details,
        validation={"valid": True, "robust_target_satisfied": target_ok},
    )


def _pair_state(
    ratings: np.ndarray,
    active: np.ndarray,
    weights: np.ndarray,
    agent_a: int,
    agent_b: int,
) -> tuple[int, int, Fraction]:
    weight = int(weights[active].sum())
    if weight <= 0:
        raise ValueError("empty agenda")
    d2 = np.abs(
        ratings[agent_a, active].astype(np.int16)
        - ratings[agent_b, active].astype(np.int16)
    )
    n2 = int(np.dot(weights[active], d2))
    return weight, n2, degree_from_integer_state(n2, weight)


def solve_safe_ilp(
    situation: SituationTable,
    *,
    agent_a: str | int,
    agent_b: str | int,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
    target: str | Region = Region.AR,
    operators: Sequence[str] = ("mod", "del"),
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    missing_policy: str = "error",
    bystander_pairs: Iterable[tuple[str | int, str | int]] | None = None,
    time_limit: float | None = 300,
    mip_rel_gap: float | None = 0.0,
    node_limit: int | None = None,
) -> SolveResult:
    """Solve hard MCIE-Safe with concrete target-cell modifications.

    The Safe model requires a complete operational table.  ``missing_policy``
    must be ``error`` (all entries observed) or the explicit sensitivity choice
    ``neutral`` (stored zero values are treated as ratings).  The latter is
    recorded in the result and must not be described as complete-case evidence.
    """
    started = perf_counter()
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_region = Region(str(target).upper()) if not isinstance(target, Region) else target
    if target_region not in {Region.AR, Region.NR}:
        raise ValueError("Safe target must be AR or NR")
    if missing_policy not in {"error", "neutral"}:
        raise ValueError("Safe missing_policy must be error or neutral")
    if missing_policy == "error" and not situation.observed.all():
        raise ValueError(
            "Safe ILP needs a complete table; use an explicit neutral sensitivity "
            "variant or prefilter to a complete subtable"
        )
    operator_set = {operator.lower() for operator in operators}
    if operator_set - {"mod", "del", "rew"}:
        raise ValueError(f"unknown operators: {operator_set-{'mod','del','rew'}}")
    costs = dict(costs or {})
    ia, ib = situation.agent_index(agent_a), situation.agent_index(agent_b)
    if ia == ib:
        raise ValueError("the target pair must contain distinct agents")
    ratings = situation.ratings.copy()
    weights = integerize_weights(situation.weights)
    active = np.ones(situation.n_issues, dtype=bool)
    original_weight, _, original = _pair_state(
        ratings, active, weights, ia, ib
    )
    original_region = classify_degree(original, beta_f, alpha_f)
    if original_region != Region.CR:
        return SolveResult(
            status=SolveStatus.INVALID_SOURCE,
            solver="ILP-Safe",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact-if-optimal",
            message="the designated target pair is not initially in CR",
        )

    variables: list[_GlobalVariable] = []
    issue_variables: list[list[int]] = [[] for _ in range(situation.n_issues)]
    for issue in range(situation.n_issues):
        if "mod" in operator_set:
            for source, side in ((ia, "a"), (ib, "b")):
                for new_rating in (-1, 0, 1):
                    if new_rating == int(ratings[source, issue]):
                        continue
                    variable = _GlobalVariable(
                        issue_index=issue,
                        issue_id=situation.issue_ids[issue],
                        action_id=f"mod_{side}_to_{new_rating:+d}",
                        kind="mod",
                        cost=as_fraction(costs.get("mod", 1)),
                        metadata={
                            "agent_index": source,
                            "new_rating": new_rating,
                        },
                    )
                    issue_variables[issue].append(len(variables))
                    variables.append(variable)
        if "del" in operator_set or "rew" in operator_set:
            kind = "del" if "del" in operator_set else "rew"
            variable = _GlobalVariable(
                issue_index=issue,
                issue_id=situation.issue_ids[issue],
                action_id="del" if kind == "del" else "rew_0",
                kind=kind,
                cost=as_fraction(costs.get(kind, 1)),
                metadata={"endpoint": kind == "rew"},
            )
            issue_variables[issue].append(len(variables))
            variables.append(variable)
    if not variables:
        return SolveResult(
            status=SolveStatus.INFEASIBLE,
            solver="ILP-Safe",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact",
            message="no non-keep action is available",
        )

    if bystander_pairs is None:
        audited_pairs = [
            pair
            for pair in combinations(range(situation.n_agents), 2)
            if pair != tuple(sorted((ia, ib)))
        ]
    else:
        audited_pairs = []
        for left, right in bystander_pairs:
            pair = tuple(sorted((situation.agent_index(left), situation.agent_index(right))))
            if pair != tuple(sorted((ia, ib))):
                audited_pairs.append(pair)

    rows: list[dict[int, int]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for variable_ids in issue_variables:
        if variable_ids:
            rows.append({variable_id: 1 for variable_id in variable_ids})
            lower_bounds.append(-np.inf)
            upper_bounds.append(1)
    nonempty_row = {
        variable_id: -int(weights[variable.issue_index])
        for variable_id, variable in enumerate(variables)
        if variable.kind in {"del", "rew"}
    }
    rows.append(nonempty_row)
    lower_bounds.append(1 - original_weight)
    upper_bounds.append(np.inf)

    def pair_linear_data(
        left: int, right: int, threshold: Fraction
    ) -> tuple[int, dict[int, int]]:
        denominator, numerator = threshold.denominator, threshold.numerator
        original_d2 = np.abs(
            ratings[left].astype(np.int16) - ratings[right].astype(np.int16)
        )
        issue_scores = (
            denominator * weights * original_d2 - 2 * numerator * weights
        ).astype(np.int64)
        baseline = int(issue_scores.sum())
        deltas: dict[int, int] = {}
        for variable_id, variable in enumerate(variables):
            issue = variable.issue_index
            if variable.kind in {"del", "rew"}:
                delta = -int(issue_scores[issue])
            else:
                changed_agent = int(variable.metadata["agent_index"])
                if changed_agent not in {left, right}:
                    continue
                left_rating = (
                    int(variable.metadata["new_rating"])
                    if changed_agent == left
                    else int(ratings[left, issue])
                )
                right_rating = (
                    int(variable.metadata["new_rating"])
                    if changed_agent == right
                    else int(ratings[right, issue])
                )
                new_d2 = abs(left_rating - right_rating)
                new_score = (
                    denominator * int(weights[issue]) * new_d2
                    - 2 * numerator * int(weights[issue])
                )
                delta = new_score - int(issue_scores[issue])
            if delta:
                deltas[variable_id] = delta
        return baseline, deltas

    def add_region(
        left: int,
        right: int,
        desired: Region,
        lower_threshold: Fraction,
        upper_threshold: Fraction,
    ) -> None:
        if desired == Region.AR:
            baseline, deltas = pair_linear_data(left, right, lower_threshold)
            rows.append(deltas)
            lower_bounds.append(-np.inf)
            upper_bounds.append(-baseline)
        elif desired == Region.CR:
            baseline, deltas = pair_linear_data(left, right, upper_threshold)
            rows.append(deltas)
            lower_bounds.append(-baseline)
            upper_bounds.append(np.inf)
        else:
            baseline, deltas = pair_linear_data(left, right, lower_threshold)
            rows.append(deltas)
            lower_bounds.append(1 - baseline)
            upper_bounds.append(np.inf)
            baseline, deltas = pair_linear_data(left, right, upper_threshold)
            rows.append(deltas)
            lower_bounds.append(-np.inf)
            upper_bounds.append(-1 - baseline)

    add_region(ia, ib, target_region, beta_f, alpha_f)
    base_regions: dict[tuple[int, int], Region] = {}
    for left, right in audited_pairs:
        _, _, degree = _pair_state(ratings, active, weights, left, right)
        region = classify_degree(degree, beta_f, alpha_f)
        base_regions[(left, right)] = region
        add_region(left, right, region, beta_f, alpha_f)

    n_variables = len(variables)
    constraint = _sparse_constraints(
        rows, lower_bounds, upper_bounds, n_variables
    )
    objective, objective_unit = _exact_objective(
        [variable.cost for variable in variables]
    )
    result = milp(
        c=objective,
        integrality=np.ones(n_variables, dtype=np.int8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraint,
        options=_milp_options(
            time_limit=time_limit, mip_rel_gap=mip_rel_gap, node_limit=node_limit
        ),
    )
    details = _solver_details(
        result, n_variables=n_variables, n_constraints=len(rows)
    )
    details.update(
        {
            "audited_bystander_pairs": len(audited_pairs),
            "missing_policy": missing_policy,
            "objective_unit": str(objective_unit),
        }
    )
    if result.x is None:
        status = SolveStatus.INFEASIBLE if result.status == 2 else SolveStatus.ERROR
        return SolveResult(
            status=status,
            solver="ILP-Safe",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=0,
            plan=[],
            original_degree=original,
            post_degree=None,
            original_region=original_region,
            post_region=None,
            runtime_seconds=perf_counter() - started,
            guarantee="exact-if-optimal",
            message=str(result.message),
            details=details,
        )

    chosen_ids = [index for index, value in enumerate(result.x) if value > 0.5]
    changed_ratings = ratings.copy()
    changed_active = active.copy()
    plan: list[PlanAction] = []
    total_cost = Fraction(0)
    for variable_id in chosen_ids:
        variable = variables[variable_id]
        issue = variable.issue_index
        if variable.kind == "mod":
            changed_ratings[int(variable.metadata["agent_index"]), issue] = int(
                variable.metadata["new_rating"]
            )
        else:
            changed_active[issue] = False
        post_weight = int(weights[issue]) if changed_active[issue] else 0
        target_d2 = abs(
            int(changed_ratings[ia, issue]) - int(changed_ratings[ib, issue])
        )
        plan.append(
            PlanAction(
                issue_index=issue,
                issue_id=variable.issue_id,
                action_id=variable.action_id,
                kind=variable.kind,
                post_weight=post_weight,
                post_n2=post_weight * target_d2,
                cost=variable.cost,
                metadata=variable.metadata,
            )
        )
        total_cost += variable.cost

    if not changed_active.any():
        target_ok = False
        post_degree = None
        post_region = None
        safe_violations = ["empty agenda"]
    else:
        _, _, post_degree = _pair_state(
            changed_ratings, changed_active, weights, ia, ib
        )
        post_region = classify_degree(post_degree, beta_f, alpha_f)
        target_ok = post_region == target_region
        safe_violations: list[str] = []
        for left, right in audited_pairs:
            _, _, degree = _pair_state(
                changed_ratings, changed_active, weights, left, right
            )
            region = classify_degree(degree, beta_f, alpha_f)
            if region != base_regions[(left, right)]:
                safe_violations.append(
                    f"{situation.agent_ids[left]}::{situation.agent_ids[right]}"
                )
    valid = target_ok and not safe_violations
    if not valid:
        return SolveResult(
            status=SolveStatus.ERROR,
            solver="ILP-Safe",
            target=target_region.value,
            objective_value=None,
            touched_issue_count=len(plan),
            plan=plan,
            original_degree=original,
            post_degree=post_degree,
            original_region=original_region,
            post_region=post_region,
            runtime_seconds=perf_counter() - started,
            guarantee="invalid-incumbent",
            message="independent Safe validator rejected the MILP incumbent",
            details=details,
            validation={
                "valid": False,
                "target_satisfied": target_ok,
                "safe_violation_count": len(safe_violations),
                "safe_violations": safe_violations[:20],
            },
        )
    solve_status = (
        SolveStatus.OPTIMAL if result.status == 0 else SolveStatus.FEASIBLE_TIMEOUT
    )
    return SolveResult(
        status=solve_status,
        solver="ILP-Safe",
        target=target_region.value,
        objective_value=total_cost,
        touched_issue_count=len(plan),
        plan=plan,
        original_degree=original,
        post_degree=post_degree,
        original_region=original_region,
        post_region=post_region,
        runtime_seconds=perf_counter() - started,
        guarantee="exact" if result.status == 0 else "validated-incumbent",
        message=str(result.message),
        details=details,
        validation={
            "valid": True,
            "target_satisfied": True,
            "safe_violation_count": 0,
        },
    )
