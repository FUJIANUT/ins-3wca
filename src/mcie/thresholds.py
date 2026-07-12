"""Principled threshold-selection protocols for pairwise conflict regions.

Three threshold-calibration protocols are provided:

1. ``quantile_thresholds`` -- data-driven thresholds read off the empirical
   distribution of pairwise conflict degrees, so that the alliance, neutrality,
   and conflict regions each receive a controlled share of pairs.
2. ``dtrs_thresholds`` -- decision-theoretic (Bayesian minimum-risk) thresholds
   derived from an interpretable 3x2 loss matrix, following the three-way
   decision derivation of Yao applied to the conflict degree as a conflict
   posterior.
3. ``sensitivity_grid`` -- a grid of valid ``(beta, alpha)`` pairs for reporting
   how a designated MCIE outcome moves as the thresholds vary.

All thresholds are returned as exact :class:`fractions.Fraction` values so that
they compose with the exact-arithmetic solvers without introducing binary-float
denominators.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from .model import (
    Region,
    SituationTable,
    as_fraction,
    classify_degree,
    pairwise_degrees,
    validate_thresholds,
)


def _fraction_quantile(values: Sequence[Fraction], q: Fraction) -> Fraction:
    """Exact lower-nearest-rank quantile of a list of Fractions, q in [0,1]."""
    if not values:
        raise ValueError("empty value list")
    if not 0 <= q <= 1:
        raise ValueError("quantile must lie in [0,1]")
    ordered = sorted(values)
    if q == 0:
        return ordered[0]
    if q == 1:
        return ordered[-1]
    # nearest-rank (ceil) index, 1-based -> 0-based
    rank = q * len(ordered)
    idx = int(-(-rank.numerator // rank.denominator)) - 1  # ceil then to 0-based
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def exact_pairwise_degrees(
    situation: SituationTable,
    *,
    missing_policy: str = "exclude",
    skip_empty_pairs: bool = True,
) -> list[Fraction]:
    """All unordered pair degrees as exact Fractions.

    With ``missing_policy='exclude'`` some pairs may share no observed issue;
    those are skipped when ``skip_empty_pairs`` is set, matching the convention
    that an empty common agenda is not classified.
    """
    ids = list(situation.agent_ids)
    degrees: list[Fraction] = []
    for i, agent_a in enumerate(ids):
        for agent_b in ids[i + 1 :]:
            try:
                pair = situation.pair(agent_a, agent_b, missing_policy=missing_policy)
            except ValueError:
                if skip_empty_pairs:
                    continue
                raise
            degrees.append(pair.exact_degree())
    return degrees


@dataclass(frozen=True)
class ThresholdReport:
    beta: Fraction
    alpha: Fraction
    method: str
    n_pairs: int
    region_counts: dict[str, int]
    region_fractions: dict[str, float]
    detail: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "beta": f"{self.beta.numerator}/{self.beta.denominator}",
            "alpha": f"{self.alpha.numerator}/{self.alpha.denominator}",
            "beta_float": float(self.beta),
            "alpha_float": float(self.alpha),
            "method": self.method,
            "n_pairs": self.n_pairs,
            "region_counts": self.region_counts,
            "region_fractions": self.region_fractions,
            "detail": self.detail,
        }


def region_partition(
    degrees: Sequence[Fraction], beta: Fraction, alpha: Fraction
) -> ThresholdReport:
    """Count pairs falling in each region under given thresholds."""
    validate_thresholds(beta, alpha)
    counts = {Region.AR.value: 0, Region.NR.value: 0, Region.CR.value: 0}
    for degree in degrees:
        counts[classify_degree(degree, beta, alpha).value] += 1
    total = max(1, len(degrees))
    fractions = {key: value / total for key, value in counts.items()}
    return ThresholdReport(
        beta=beta,
        alpha=alpha,
        method="partition",
        n_pairs=len(degrees),
        region_counts=counts,
        region_fractions=fractions,
        detail={},
    )


def quantile_thresholds(
    degrees: Sequence[Fraction],
    *,
    lower_quantile: Fraction | float | str = Fraction(1, 3),
    upper_quantile: Fraction | float | str = Fraction(9, 10),
) -> ThresholdReport:
    """Thresholds at chosen quantiles of the empirical degree distribution.

    ``beta`` is placed at ``lower_quantile`` and ``alpha`` at ``upper_quantile``
    of the pairwise degrees.  The defaults (1/3 and 9/10) treat the bottom third
    as alliance and the top decile as conflict, leaving a populated neutral band
    for the middle.  When ties collapse the two quantiles, alpha is nudged to the
    next distinct degree above beta so that 0<=beta<alpha<=1 holds.
    """
    lq, uq = as_fraction(lower_quantile), as_fraction(upper_quantile)
    if not lq < uq:
        raise ValueError("lower_quantile must be below upper_quantile")
    beta = _fraction_quantile(degrees, lq)
    alpha = _fraction_quantile(degrees, uq)
    detail = {
        "lower_quantile": f"{lq.numerator}/{lq.denominator}",
        "upper_quantile": f"{uq.numerator}/{uq.denominator}",
    }
    if not beta < alpha:
        higher = sorted({d for d in degrees if d > beta})
        if higher:
            alpha = higher[0]
            detail["alpha_adjusted"] = "raised to next distinct degree above beta"
        else:
            alpha = min(Fraction(1), beta + Fraction(1, 1000))
            detail["alpha_adjusted"] = "beta at distribution max; alpha nudged"
    report = region_partition(degrees, beta, alpha)
    return ThresholdReport(
        beta=beta,
        alpha=alpha,
        method="quantile",
        n_pairs=report.n_pairs,
        region_counts=report.region_counts,
        region_fractions=report.region_fractions,
        detail=detail,
    )


def dtrs_thresholds(
    *,
    lambda_cc: Fraction | float | str,
    lambda_nc: Fraction | float | str,
    lambda_ac: Fraction | float | str,
    lambda_ca: Fraction | float | str,
    lambda_na: Fraction | float | str,
    lambda_aa: Fraction | float | str,
    degrees: Sequence[Fraction] | None = None,
) -> ThresholdReport:
    """Decision-theoretic thresholds from a 3x2 loss matrix.

    Rows are the three decisions (declare Conflict, defer to Neutral, declare
    Alliance); columns are the two latent states (the pair is essentially in
    Conflict, or essentially in Alliance).  Identifying the conflict degree D
    with the posterior probability of the conflict state, Bayesian minimum-risk
    decision making yields the two-boundary three-way rule with

        alpha = (lambda_ca - lambda_na) / ((lambda_ca - lambda_na) + (lambda_nc - lambda_cc)),
        beta  = (lambda_na - lambda_aa) / ((lambda_na - lambda_aa) + (lambda_ac - lambda_nc)),

    which reproduces Yao's three-way-decision thresholds mapped so that a high
    degree favors declaring conflict.  A ValueError is raised if the loss
    ordering does not yield a valid 0<=beta<alpha<=1.
    """
    lcc, lnc, lac = as_fraction(lambda_cc), as_fraction(lambda_nc), as_fraction(lambda_ac)
    lca, lna, laa = as_fraction(lambda_ca), as_fraction(lambda_na), as_fraction(lambda_aa)

    # Reasonableness of the loss ordering (correct decision is cheapest per state).
    if not (lcc <= lnc <= lac):
        raise ValueError("conflict-state losses must satisfy lambda_cc <= lambda_nc <= lambda_ac")
    if not (laa <= lna <= lca):
        raise ValueError("alliance-state losses must satisfy lambda_aa <= lambda_na <= lambda_ca")

    num_alpha = lca - lna
    den_alpha = (lca - lna) + (lnc - lcc)
    num_beta = lna - laa
    den_beta = (lna - laa) + (lac - lnc)
    if den_alpha <= 0 or den_beta <= 0:
        raise ValueError("degenerate loss matrix: nonpositive threshold denominator")
    alpha = num_alpha / den_alpha
    beta = num_beta / den_beta
    if not 0 <= beta < alpha <= 1:
        raise ValueError(
            f"loss matrix yields invalid thresholds beta={beta}, alpha={alpha}; "
            "the neutral region is empty (increase deferral attractiveness)"
        )
    detail = {
        "lambda": f"CC={lcc},NC={lnc},AC={lac},CA={lca},NA={lna},AA={laa}",
        "note": "alpha,beta from Bayesian minimum-risk 3WD boundaries",
    }
    if degrees is not None:
        report = region_partition(degrees, beta, alpha)
        counts, fracs, n = report.region_counts, report.region_fractions, report.n_pairs
    else:
        counts, fracs, n = {}, {}, 0
    return ThresholdReport(
        beta=beta,
        alpha=alpha,
        method="dtrs",
        n_pairs=n,
        region_counts=counts,
        region_fractions=fracs,
        detail=detail,
    )


def sensitivity_grid(
    *,
    betas: Iterable[Fraction | float | str],
    alphas: Iterable[Fraction | float | str],
) -> list[tuple[Fraction, Fraction]]:
    """Valid (beta, alpha) pairs on the Cartesian grid with beta<alpha."""
    beta_vals = sorted({as_fraction(b) for b in betas})
    alpha_vals = sorted({as_fraction(a) for a in alphas})
    grid: list[tuple[Fraction, Fraction]] = []
    for beta in beta_vals:
        for alpha in alpha_vals:
            if 0 <= beta < alpha <= 1:
                grid.append((beta, alpha))
    return grid
