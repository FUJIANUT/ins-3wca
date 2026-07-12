"""Shared action compilation, result schema, and independent validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
import json
from typing import Any, Mapping, Sequence

import numpy as np

from ..model import (
    PairInstance,
    Region,
    as_fraction,
    classify_degree,
    degree_from_integer_state,
    integerize_weights,
    validate_thresholds,
)


class SolveStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    FEASIBLE_TIMEOUT = "FEASIBLE_TIMEOUT"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    INVALID_SOURCE = "INVALID_SOURCE"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PairOption:
    issue_index: int
    issue_id: str
    action_id: str
    kind: str
    post_weight: int
    post_n2: int
    cost: Fraction
    touched: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def retains_positive_weight(self) -> bool:
        return self.post_weight > 0


@dataclass(frozen=True)
class PlanAction:
    issue_index: int
    issue_id: str
    action_id: str
    kind: str
    post_weight: int
    post_n2: int
    cost: Fraction
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_option(cls, option: PairOption) -> "PlanAction":
        return cls(
            issue_index=option.issue_index,
            issue_id=option.issue_id,
            action_id=option.action_id,
            kind=option.kind,
            post_weight=option.post_weight,
            post_n2=option.post_n2,
            cost=option.cost,
            metadata=option.metadata,
        )


@dataclass
class SolveResult:
    status: SolveStatus
    solver: str
    target: str
    objective_value: Fraction | None
    touched_issue_count: int
    plan: list[PlanAction]
    original_degree: Fraction
    post_degree: Fraction | None
    original_region: Region
    post_region: Region | None
    runtime_seconds: float
    guarantee: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Fraction):
                return f"{value.numerator}/{value.denominator}"
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(item) for item in value]
            return value

        return convert(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def reachable_modification_witnesses(
    attitude_a: int, attitude_b: int
) -> dict[int, dict[str, int | str]]:
    """Choose one deterministic concrete cell update for each reachable distance."""
    attitude_a, attitude_b = int(attitude_a), int(attitude_b)
    candidates: dict[int, list[tuple[int, int, str, int, int]]] = {}
    for replacement in (-1, 0, 1):
        if replacement != attitude_a:
            d2 = abs(replacement - attitude_b)
            candidates.setdefault(d2, []).append(
                (abs(replacement - attitude_a), 0, "a", attitude_a, replacement)
            )
        if replacement != attitude_b:
            d2 = abs(attitude_a - replacement)
            candidates.setdefault(d2, []).append(
                (abs(replacement - attitude_b), 1, "b", attitude_b, replacement)
            )
    witnesses: dict[int, dict[str, int | str]] = {}
    for d2, options in candidates.items():
        _, _, side, old_rating, new_rating = min(options)
        witnesses[d2] = {
            "changed_side": side,
            "old_rating": old_rating,
            "new_rating": new_rating,
            "post_doubled_distance": d2,
        }
    return witnesses


def reachable_doubled_distances(attitude_a: int, attitude_b: int) -> tuple[int, ...]:
    return tuple(sorted(reachable_modification_witnesses(attitude_a, attitude_b)))


def build_pair_options(
    pair: PairInstance,
    *,
    operators: Sequence[str] = ("mod", "del"),
    full_mod_menu: bool = True,
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    issue_costs: Sequence[Fraction | int | float | str] | None = None,
    max_scaled_weight: int = 2_000_000,
) -> tuple[list[list[PairOption]], np.ndarray]:
    """Compile independent per-issue actions to exact integer contributions.

    ``issue_costs``, when given, overrides the cost of every non-keep action on
    issue ``i`` with ``issue_costs[i]``; this supports the heterogeneous-cost
    (Theorem 11) setting where each issue carries its own touch cost. When it is
    None (the default) the per-action-kind ``costs`` mapping is used.
    """
    operator_set = {str(operator).lower() for operator in operators}
    unknown = operator_set - {"mod", "del", "rew"}
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")
    costs = dict(costs or {})
    if issue_costs is not None and len(issue_costs) != len(pair.issue_ids):
        raise ValueError("issue_costs length must match the number of issues")
    integer_weights = integerize_weights(pair.weights, max_total=max_scaled_weight)
    menus: list[list[PairOption]] = []
    for index, (issue_id, attitude_a, attitude_b, weight, d2) in enumerate(
        zip(
            pair.issue_ids,
            pair.attitudes_a,
            pair.attitudes_b,
            integer_weights,
            pair.doubled_distances,
        )
    ):
        weight_i, d2_i = int(weight), int(d2)
        icost = as_fraction(issue_costs[index]) if issue_costs is not None else None

        def _cost(kind: str) -> Fraction:
            return icost if icost is not None else as_fraction(costs.get(kind, 1))

        options = [
            PairOption(
                issue_index=index,
                issue_id=issue_id,
                action_id="keep",
                kind="keep",
                post_weight=weight_i,
                post_n2=weight_i * d2_i,
                cost=Fraction(0),
                touched=False,
            )
        ]
        if "mod" in operator_set:
            witnesses = reachable_modification_witnesses(
                int(attitude_a), int(attitude_b)
            )
            reachable = tuple(sorted(witnesses)) if full_mod_menu else (0,)
            for post_d2 in reachable:
                if post_d2 == d2_i:
                    continue
                if post_d2 not in witnesses:
                    continue
                options.append(
                    PairOption(
                        issue_index=index,
                        issue_id=issue_id,
                        action_id=f"mod_d{post_d2}",
                        kind="mod",
                        post_weight=weight_i,
                        post_n2=weight_i * post_d2,
                        cost=_cost("mod"),
                        touched=True,
                        metadata=witnesses[post_d2],
                    )
                )
        if "del" in operator_set:
            options.append(
                PairOption(
                    issue_index=index,
                    issue_id=issue_id,
                    action_id="del",
                    kind="del",
                    post_weight=0,
                    post_n2=0,
                    cost=_cost("del"),
                    touched=True,
                )
            )
        if "rew" in operator_set and "del" not in operator_set:
            options.append(
                PairOption(
                    issue_index=index,
                    issue_id=issue_id,
                    action_id="rew_0",
                    kind="rew",
                    post_weight=0,
                    post_n2=0,
                    cost=_cost("rew"),
                    touched=True,
                    metadata={"endpoint": True},
                )
            )
        menus.append(options)
    return menus, integer_weights


def linear_score(option: PairOption, threshold: Fraction) -> int:
    """Integer multiple of the linearized excess at one threshold."""
    threshold = as_fraction(threshold)
    return (
        threshold.denominator * option.post_n2
        - 2 * threshold.numerator * option.post_weight
    )


def evaluate_pair_selection(
    menus: Sequence[Sequence[PairOption]],
    selected_indices: Sequence[int],
    *,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
) -> dict[str, Any]:
    """Recompute a candidate from its selected actions, independently of a solver."""
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    if len(menus) != len(selected_indices):
        raise ValueError("one selected option is required per issue")
    weight = 0
    n2 = 0
    cost = Fraction(0)
    touched = 0
    for menu, selected in zip(menus, selected_indices):
        option = menu[int(selected)]
        weight += option.post_weight
        n2 += option.post_n2
        cost += option.cost
        touched += int(option.touched)
    if weight <= 0:
        return {
            "valid": False,
            "reason": "empty agenda",
            "weight": weight,
            "n2": n2,
            "cost": cost,
            "touched": touched,
        }
    degree = degree_from_integer_state(n2, weight)
    return {
        "valid": True,
        "weight": weight,
        "n2": n2,
        "degree": degree,
        "region": classify_degree(degree, beta_f, alpha_f),
        "cost": cost,
        "touched": touched,
    }


def plan_from_selection(
    menus: Sequence[Sequence[PairOption]], selected_indices: Sequence[int]
) -> list[PlanAction]:
    return [
        PlanAction.from_option(menu[int(selected)])
        for menu, selected in zip(menus, selected_indices)
        if menu[int(selected)].touched
    ]
