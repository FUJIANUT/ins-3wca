"""Synthetic scaling experiments validating the complexity claims.

Measures how the exact solvers scale in four dimensions:
  S1  m  -- issue count            (GreedyFlip O(m log m); GridDP O(m^2))
  S2  H  -- weight granularity     (GridDP pseudopolynomial; GreedyFlip flat)
  S3  L  -- action-menu size       (deletion adds the retained-weight axis)
  S4  n  -- agents, for MCIE-Safe  (bystander constraints ~ C(n,2))

All instances are synthetic and deterministic. Runtimes are local wall-clock
medians and are meant to confirm the predicted growth exponents, not to rank
hardware.
"""

from __future__ import annotations

import json
import math
import random
import time
from fractions import Fraction
from pathlib import Path
from statistics import median

import numpy as np

from mcie.model import PairInstance, SituationTable
from mcie.solvers import solve_greedy_flip, solve_grid_dp, solve_pair_ilp, solve_safe_ilp
from mcie.solvers.common import build_pair_options

OUT = Path(__file__).resolve().parent / "results"
BETA, ALPHA = Fraction(1, 4), Fraction(3, 4)


def synth_pair(m, w_range=1, seed=0):
    """All-opposed synthetic pair (D0=1); weights in {1..w_range}."""
    a = np.ones(m, dtype=np.int8)
    b = -np.ones(m, dtype=np.int8)
    if w_range <= 1:
        w = np.ones(m, dtype=np.int64)
    else:
        rng = random.Random(seed)
        w = np.array([rng.randint(1, w_range) for _ in range(m)], dtype=np.int64)
    return PairInstance("A", "B", tuple(f"i{i}" for i in range(m)), a, b, w, np.arange(m))


def timed(fn, repeats=1):
    ts = []
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return median(ts), out


def scaled_H(pair):
    _, iw = build_pair_options(pair, operators=("mod",))
    return int(iw.sum())


def s1_issue_count():
    rows = []
    for m in [64, 128, 256, 512, 1024, 2048, 4096]:
        p = synth_pair(m)
        r = int(3 if m <= 512 else 1)
        tg, _ = timed(lambda: solve_greedy_flip(p, beta=BETA, alpha=ALPHA, operators=("mod",)), r)
        rec = {"m": m, "greedy_ms": round(tg * 1000, 3)}
        if m <= 2048:
            td, _ = timed(lambda: solve_grid_dp(p, beta=BETA, alpha=ALPHA, operators=("mod",),
                                                max_states=20_000_000), 1)
            rec["griddp_ms"] = round(td * 1000, 3)
        ti, _ = timed(lambda: solve_pair_ilp(p, target="AR", beta=BETA, alpha=ALPHA, operators=("mod",)), 1)
        rec["ilp_ar_ms"] = round(ti * 1000, 3)
        rows.append(rec)
        print("  S1", rec, flush=True)
    return rows


def s2_weight_granularity():
    rows = []
    m = 256
    for wK in [1, 4, 16, 64, 256]:
        p = synth_pair(m, w_range=wK, seed=1)
        H = scaled_H(p)
        tg, _ = timed(lambda: solve_greedy_flip(p, beta=BETA, alpha=ALPHA, operators=("mod",)), 3)
        td, res = timed(lambda: solve_grid_dp(p, beta=BETA, alpha=ALPHA, operators=("mod",),
                                              max_states=60_000_000, max_scaled_weight=400_000), 1)
        rec = {"w_range": wK, "H": H, "greedy_ms": round(tg * 1000, 3),
               "griddp_ms": round(td * 1000, 3), "griddp_status": res.status.value}
        rows.append(rec)
        print("  S2", rec, flush=True)
    return rows


def s3_menu_size():
    rows = []
    m = 256
    for ops in [("mod",), ("mod", "del")]:
        p = synth_pair(m, w_range=8, seed=2)
        td, res = timed(lambda: solve_grid_dp(p, beta=BETA, alpha=ALPHA, operators=ops,
                                              max_states=60_000_000, max_scaled_weight=400_000), 1)
        rec = {"operators": "+".join(ops), "griddp_ms": round(td * 1000, 3),
               "status": res.status.value}
        rows.append(rec)
        print("  S3", rec, flush=True)
    return rows


def synth_table(n, m, seed=0):
    rng = np.random.default_rng(seed)
    # two polarized blocs plus noise, so extreme pairs exist
    ratings = rng.integers(-1, 2, size=(n, m)).astype(np.int8)
    ratings[0] = 1          # designated agent A: all support
    ratings[1] = -1         # designated agent B: all oppose  -> D(A,B)=1 in CR
    observed = np.ones((n, m), dtype=bool)
    return SituationTable(
        name=f"synth_n{n}_m{m}",
        ratings=ratings,
        observed=observed,
        agent_ids=tuple(f"g{i}" for i in range(n)),
        issue_ids=tuple(f"t{j}" for j in range(m)),
    )


def s4_agents_safe():
    rows = []
    m = 40
    for n in [20, 40, 80, 120, 160, 200]:
        st = synth_table(n, m, seed=n)
        t = time.perf_counter()
        res = solve_safe_ilp(st, agent_a="g0", agent_b="g1", beta=Fraction(1, 4),
                             alpha=Fraction(3, 4), target="NR", operators=("mod",),
                             missing_policy="error", time_limit=120)
        dt = time.perf_counter() - t
        rec = {"n": n, "bystanders": n * (n - 1) // 2 - 1,
               "n_constraints": res.details.get("n_constraints"),
               "n_variables": res.details.get("n_variables"),
               "status": res.status.value, "wall_s": round(dt, 3)}
        rows.append(rec)
        print("  S4", rec, flush=True)
    return rows


def fit_slope(xs, ys):
    """log-log least-squares slope (growth exponent)."""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys if y > 0]
    lx = lx[: len(ly)]
    n = len(lx)
    if n < 2:
        return None
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return round(num / den, 3) if den else None


def main():
    print("== S1 issue count ==", flush=True)
    s1 = s1_issue_count()
    print("== S2 weight granularity ==", flush=True)
    s2 = s2_weight_granularity()
    print("== S3 menu size ==", flush=True)
    s3 = s3_menu_size()
    print("== S4 agents (Safe) ==", flush=True)
    s4 = s4_agents_safe()

    slopes = {
        "greedy_in_m": fit_slope([r["m"] for r in s1], [r["greedy_ms"] for r in s1]),
        "griddp_in_m": fit_slope([r["m"] for r in s1 if "griddp_ms" in r],
                                 [r["griddp_ms"] for r in s1 if "griddp_ms" in r]),
        "ilp_ar_in_m": fit_slope([r["m"] for r in s1], [r["ilp_ar_ms"] for r in s1]),
        "griddp_in_H": fit_slope([r["H"] for r in s2 if r["H"] > 0], [r["griddp_ms"] for r in s2]),
        "greedy_in_H": fit_slope([r["H"] for r in s2 if r["H"] > 0], [r["greedy_ms"] for r in s2]),
        "safe_constraints_in_n": fit_slope([r["n"] for r in s4], [r["n_constraints"] for r in s4]),
        "safe_time_in_n": fit_slope([r["n"] for r in s4], [r["wall_s"] for r in s4]),
    }
    report = {"S1_m": s1, "S2_weight_H": s2, "S3_menu": s3, "S4_agents_safe": s4,
              "loglog_slopes": slopes}
    (OUT / "scaling_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("SLOPES", json.dumps(slopes), flush=True)
    print("WROTE", OUT / "scaling_results.json", flush=True)


if __name__ == "__main__":
    main()
