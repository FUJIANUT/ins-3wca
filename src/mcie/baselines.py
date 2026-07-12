"""Heuristic comparison baselines and fairness metrics for MCIE.

These baselines adapt the *selection heuristics* of neighboring conflict-
resolution methods to the pairwise MCIE task, so that the exact algorithms of
Section 4 can be compared against them on a common footing. They are not the
original group-level methods; each docstring states the adaptation.

The comparison is deliberately honest. For the unit-cost alliance target the
conflict-score-ranking heuristic coincides with the exact GreedyFlip optimum,
because ranking by single-issue conflict equals ranking by alliance gain under
equal weights. The informative gaps appear for (i) heterogeneous costs, where
conflict-blind ranking overpays; (ii) the open neutrality band, where a pure
conflict-reduction heuristic can overshoot; and (iii) concession fairness.

All arithmetic is exact (:class:`fractions.Fraction`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping, Sequence

from .model import PairInstance, Region, as_fraction, validate_thresholds
from .solvers.common import (
    PairOption,
    build_pair_options,
    evaluate_pair_selection,
)


@dataclass
class BaselineResult:
    method: str
    target: str
    reached: bool
    region: str
    cardinality: int
    cost: Fraction
    overshoot: bool
    post_degree: Fraction | None
    sides: dict[str, int] = field(default_factory=dict)  # {'a': .., 'b': ..}

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target": self.target,
            "reached": self.reached,
            "region": self.region,
            "cardinality": self.cardinality,
            "cost": f"{self.cost.numerator}/{self.cost.denominator}",
            "cost_float": float(self.cost),
            "overshoot": self.overshoot,
            "post_degree": None if self.post_degree is None else float(self.post_degree),
            "sides": self.sides,
        }


def _full_align_index(menu: Sequence[PairOption]) -> int | None:
    """Index of the complete-alignment modification (post distance zero)."""
    for idx, option in enumerate(menu):
        if option.kind == "mod" and option.post_n2 == 0:
            return idx
    return None


def _side_of(option: PairOption) -> str | None:
    return option.metadata.get("changed_side") if option.kind == "mod" else None


def _summarize(
    menus: Sequence[Sequence[PairOption]],
    selection: Sequence[int],
    *,
    method: str,
    target: Region,
    beta: Fraction,
    alpha: Fraction,
    overshoot: bool,
) -> BaselineResult:
    ev = evaluate_pair_selection(menus, selection, beta=beta, alpha=alpha)
    region = ev.get("region")
    reached = bool(ev.get("valid") and region == target)
    sides = {"a": 0, "b": 0}
    for menu, sel in zip(menus, selection):
        opt = menu[int(sel)]
        s = _side_of(opt)
        if s in sides:
            sides[s] += 1
    return BaselineResult(
        method=method,
        target=target.value,
        reached=reached,
        region=region.value if region is not None else "EMPTY",
        cardinality=int(ev.get("touched", 0)),
        cost=ev.get("cost", Fraction(0)),
        overshoot=overshoot,
        post_degree=ev.get("degree"),
        sides=sides,
    )


def _greedy_by_order(
    pair: PairInstance,
    order_key,
    *,
    beta: Fraction,
    alpha: Fraction,
    target: Region,
    costs: Mapping[str, Fraction | int | float | str] | None,
    method: str,
    prefer_side: str | None = None,
    issue_costs: Sequence[Fraction | int | float | str] | None = None,
) -> BaselineResult:
    """Full-align issues in a heuristic order until the target region is met."""
    menus, _ = build_pair_options(
        pair, operators=("mod",), full_mod_menu=True, costs=costs, issue_costs=issue_costs
    )
    m = len(menus)
    selection = [0] * m
    order = sorted(range(m), key=order_key)
    overshoot = False
    ev = evaluate_pair_selection(menus, selection, beta=beta, alpha=alpha)
    if ev["region"] == target:
        return _summarize(menus, selection, method=method, target=target,
                          beta=beta, alpha=alpha, overshoot=False)
    for t in order:
        align = _full_align_index(menus[t])
        if align is None:
            continue
        # Choose the concession side if the heuristic requests one and both exist.
        if prefer_side is not None:
            for idx, option in enumerate(menus[t]):
                if option.kind == "mod" and option.post_n2 == 0 and _side_of(option) == prefer_side:
                    align = idx
                    break
        selection[t] = align
        ev = evaluate_pair_selection(menus, selection, beta=beta, alpha=alpha)
        region = ev["region"]
        if region == target:
            break
        if target == Region.NR and region == Region.AR:
            # Stepped from conflict past the open band into alliance: overshoot.
            overshoot = True
            break
    return _summarize(menus, selection, method=method, target=target,
                      beta=beta, alpha=alpha, overshoot=overshoot)


def conflict_score_ranking(
    pair: PairInstance,
    *,
    beta,
    alpha,
    target: str | Region = Region.AR,
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    issue_costs: Sequence[Fraction | int | float | str] | None = None,
    prefer_side: str | None = None,
) -> BaselineResult:
    """Rank issues by single-issue conflict contribution and align greedily.

    This is the natural "largest-disagreement-first" baseline. It is *cost
    blind*: issues are ordered by \\(w_t d_t\\) regardless of action cost.
    """
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_r = Region(str(target).upper()) if not isinstance(target, Region) else target
    menus, _ = build_pair_options(pair, operators=("mod",), full_mod_menu=True)
    order_key = (lambda t: (-menus[t][0].post_n2, menus[t][0].issue_id))
    return _greedy_by_order(pair, order_key, beta=beta_f, alpha=alpha_f,
                            target=target_r, costs=costs, method="conflict_score_ranking",
                            prefer_side=prefer_side, issue_costs=issue_costs)


def zhai_cost_ratio(
    pair: PairInstance,
    *,
    beta,
    alpha,
    target: str | Region = Region.AR,
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    issue_costs: Sequence[Fraction | int | float | str] | None = None,
) -> BaselineResult:
    """Adaptation of Zhai et al.'s cost-sensitive knapsack-style selection.

    Zhai et al. (2023) pick attitude-improvement actions by a benefit-to-cost
    trade-off toward a group alliance level. Adapted to the pairwise task, this
    baseline aligns issues in decreasing order of alliance gain per unit cost.
    It is cost aware but still a greedy ratio rule, not the exact cost DP.
    """
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_r = Region(str(target).upper()) if not isinstance(target, Region) else target
    menus, _ = build_pair_options(pair, operators=("mod",), full_mod_menu=True,
                                  costs=costs, issue_costs=issue_costs)

    def ratio_key(t: int) -> tuple:
        menu = menus[t]
        keep = menu[0]
        align = _full_align_index(menu)
        if align is None:
            return (Fraction(0), menu[0].issue_id)
        gain = Fraction(keep.post_n2 - menu[align].post_n2, 2)  # w_t*d_t (drop in unnormalized numerator/2)
        cost = menu[align].cost or Fraction(1)
        return (-(gain / cost), menu[0].issue_id)

    return _greedy_by_order(pair, ratio_key, beta=beta_f, alpha=alpha_f,
                            target=target_r, costs=costs, method="zhai_cost_ratio",
                            issue_costs=issue_costs)


def liu_issue_order(
    pair: PairInstance,
    *,
    beta,
    alpha,
    target: str | Region = Region.AR,
    costs: Mapping[str, Fraction | int | float | str] | None = None,
) -> BaselineResult:
    """Adaptation of Liu et al.'s independent, issue-by-issue adjustment.

    Liu et al. (2026) process issues independently under a per-issue group
    constraint and search with a genetic algorithm without a global optimum
    certificate. Adapted to the pairwise task, this baseline aligns issues in
    their fixed agenda order until the target region is reached, i.e., without
    the global conflict-based prioritization of the other baselines.
    """
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_r = Region(str(target).upper()) if not isinstance(target, Region) else target
    return _greedy_by_order(pair, (lambda t: t), beta=beta_f, alpha=alpha_f,
                            target=target_r, costs=costs, method="liu_issue_order")


def random_intervention(
    pair: PairInstance,
    *,
    beta,
    alpha,
    target: str | Region = Region.AR,
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    seed: int = 0,
) -> BaselineResult:
    """Align issues in a fixed pseudo-random order until the target is reached."""
    import random

    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_r = Region(str(target).upper()) if not isinstance(target, Region) else target
    m = len(pair.issue_ids)
    rng = random.Random(seed)
    perm = list(range(m))
    rng.shuffle(perm)
    rank = {t: i for i, t in enumerate(perm)}
    return _greedy_by_order(pair, (lambda t: rank[t]), beta=beta_f, alpha=alpha_f,
                            target=target_r, costs=costs, method=f"random_seed{seed}")


def liu_genetic(
    pair: PairInstance,
    *,
    beta,
    alpha,
    target: str | Region = Region.AR,
    costs: Mapping[str, Fraction | int | float | str] | None = None,
    issue_costs: Sequence[Fraction | int | float | str] | None = None,
    pop_size: int = 60,
    generations: int = 120,
    seed: int = 0,
) -> BaselineResult:
    """Faithful genetic-search adaptation of Liu et al.'s adjustment method.

    Liu et al. (2026) search the space of per-issue rating adjustments with a
    genetic algorithm and return a feasible but not certified-optimal plan. This
    baseline mirrors that: a chromosome assigns one menu action per issue; the
    fitness lexicographically prefers reaching the target region, then a smaller
    boundary violation, then a smaller cost. It uses tournament selection,
    uniform crossover, and per-gene mutation, and reports the best individual.
    Unlike the exact algorithms it carries no optimality certificate.
    """
    import random

    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_r = Region(str(target).upper()) if not isinstance(target, Region) else target
    rng = random.Random(seed)
    menus, _ = build_pair_options(
        pair, operators=("mod",), full_mod_menu=True, costs=costs, issue_costs=issue_costs
    )
    m = len(menus)
    sizes = [len(menu) for menu in menus]

    def evaluate(chrom):
        ev = evaluate_pair_selection(menus, chrom, beta=beta_f, alpha=alpha_f)
        if not ev["valid"]:
            return (1, Fraction(10 ** 9), Fraction(10 ** 9)), ev
        d = ev["degree"]
        if target_r == Region.AR:
            penalty = max(Fraction(0), d - beta_f)
        else:
            penalty = max(Fraction(0), d - alpha_f) + max(Fraction(0), beta_f - d)
        feasible = 0 if ev["region"] == target_r else 1
        obj = ev["cost"]
        return (feasible, penalty, obj), ev

    # Initialize: mostly keep, a random minority of modifications.
    def random_chrom():
        return [rng.randrange(sizes[t]) if rng.random() < 0.5 else 0 for t in range(m)]

    population = [random_chrom() for _ in range(pop_size)]
    scored = [(evaluate(c)[0], c) for c in population]
    best_key, best = min(scored, key=lambda x: x[0])

    def tournament():
        a, b = rng.choice(scored), rng.choice(scored)
        return a[1] if a[0] <= b[0] else b[1]

    mut = max(1.0 / m, 0.01)
    for _ in range(generations):
        children = [best]  # elitism
        while len(children) < pop_size:
            p1, p2 = tournament(), tournament()
            child = [p1[t] if rng.random() < 0.5 else p2[t] for t in range(m)]
            for t in range(m):
                if rng.random() < mut:
                    child[t] = rng.randrange(sizes[t])
            children.append(child)
        population = children
        scored = [(evaluate(c)[0], c) for c in population]
        key, cand = min(scored, key=lambda x: x[0])
        if key < best_key:
            best_key, best = key, cand

    _, ev = evaluate(best)
    sides = {"a": 0, "b": 0}
    for menu, sel in zip(menus, best):
        s = _side_of(menu[int(sel)])
        if s in sides:
            sides[s] += 1
    return BaselineResult(
        method="liu_genetic",
        target=target_r.value,
        reached=bool(ev.get("valid") and ev.get("region") == target_r),
        region=ev["region"].value if ev.get("region") is not None else "EMPTY",
        cardinality=int(ev.get("touched", 0)),
        cost=ev.get("cost", Fraction(0)),
        overshoot=False,
        post_degree=ev.get("degree"),
        sides=sides,
    )


# ---------------------------------------------------------------------------
# Fairness: concession balance between the two target agents.
# ---------------------------------------------------------------------------


def concession_imbalance(sides: Mapping[str, int]) -> float:
    """Normalized imbalance |#a - #b| / (#a + #b) in [0, 1]; 0 is balanced."""
    a, b = int(sides.get("a", 0)), int(sides.get("b", 0))
    total = a + b
    return abs(a - b) / total if total else 0.0


def balanced_alignment(
    pair: PairInstance,
    aligned_issue_indices: Sequence[int],
    *,
    beta,
    alpha,
    target: str | Region = Region.AR,
) -> BaselineResult:
    """Balance the concession side of a given complete-alignment issue set.

    For a pure pairwise transition, complete alignment on a disagreeing issue
    \\(t\\) can be placed on either agent's cell (set \\(r_{a^*,t}=r_{b^*,t}\\) or
    the reverse) with identical effect on the pairwise degree and the same unit
    cost. Hence, given the cardinality-optimal issue set, perfect concession
    balance is attainable at no extra cost: the two sides are interchangeable.
    We therefore assign the sides alternately and confirm that the post-degree,
    region, and cardinality are unchanged from a single-sided alignment. (Under
    MCIE-Safe the side choice is *not* free, because it changes bystander
    relations; this balance result is specific to the local problem.)
    """
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    target_r = Region(str(target).upper()) if not isinstance(target, Region) else target
    menus, _ = build_pair_options(pair, operators=("mod",), full_mod_menu=True)
    selection = [0] * len(menus)
    balanced_sides = {"a": 0, "b": 0}
    toggle = 0
    for t in aligned_issue_indices:
        align = _full_align_index(menus[t])
        if align is None:
            continue
        selection[t] = align  # the degree is side-independent; use any align option
        # A disagreeing issue admits both concession sides; assign alternately.
        ra, rb = int(pair.attitudes_a[t]), int(pair.attitudes_b[t])
        if ra != rb:
            balanced_sides["a" if toggle % 2 == 0 else "b"] += 1
            toggle += 1
        else:
            balanced_sides[_side_of(menus[t][align]) or "a"] += 1
    result = _summarize(menus, selection, method="balanced_alignment", target=target_r,
                        beta=beta_f, alpha=alpha_f, overshoot=False)
    # Report the achievable balanced side counts (degree/region/cardinality unchanged).
    result.sides = balanced_sides
    return result
