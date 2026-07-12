"""Formal experiment driver: principled thresholds + solver matrix + P8 + Safe.

Reproducible end-to-end run producing results/formal_*.json.  Uses exact
rational arithmetic throughout; every returned plan is independently validated
inside each solver.  Safe runs are restricted to attitude modification
(``mod``), the interpretable mediation setting in which only the target agents'
cells change.
"""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from mcie import thresholds as T
from mcie.model import Region, SituationTable
from mcie.solvers import (
    solve_greedy_flip,
    solve_grid_dp,
    solve_pair_ilp,
    solve_safe_ilp,
)

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def top_cr_pairs(situation, degrees_map, alpha, n, missing_policy):
    ranked = sorted(
        (d for d in degrees_map if d[2] >= alpha),
        key=lambda d: (-d[2], d[0], d[1]),
    )
    return [(a, b) for a, b, _ in ranked[:n]]


def all_pair_degrees(situation, missing_policy):
    ids = list(situation.agent_ids)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            try:
                pair = situation.pair(a, b, missing_policy=missing_policy)
            except ValueError:
                continue
            out.append((a, b, pair.exact_degree()))
    return out


def solver_matrix(situation, pairs, beta, alpha, missing_policy):
    """GreedyFlip/ILP on AR and GridDP/ILP on NR, with cross-solver agreement."""
    rows = []
    agree_ar = agree_nr = checked_ar = checked_nr = 0
    for a, b in pairs:
        pair = situation.pair(a, b, missing_policy=missing_policy)
        gf = solve_greedy_flip(pair, beta=beta, alpha=alpha, operators=("mod", "del"))
        ilp_ar = solve_pair_ilp(pair, target="AR", beta=beta, alpha=alpha, operators=("mod", "del"))
        gd = solve_grid_dp(pair, beta=beta, alpha=alpha, operators=("mod",))
        ilp_nr = solve_pair_ilp(pair, target="NR", beta=beta, alpha=alpha, operators=("mod",))
        if gf.status.value == "OPTIMAL" and ilp_ar.status.value == "OPTIMAL":
            checked_ar += 1
            agree_ar += int(gf.objective_value == ilp_ar.objective_value)
        if gd.status.value == "OPTIMAL" and ilp_nr.status.value == "OPTIMAL":
            checked_nr += 1
            agree_nr += int(gd.objective_value == ilp_nr.objective_value)
        rows.append({
            "agent_a": a, "agent_b": b,
            "D0": float(pair.exact_degree()),
            "greedy_AR": _r(gf), "ilp_AR": _r(ilp_ar),
            "griddp_NR": _r(gd), "ilp_NR": _r(ilp_nr),
        })
    return rows, {"checked_ar": checked_ar, "agree_ar": agree_ar,
                  "checked_nr": checked_nr, "agree_nr": agree_nr}


def _r(res):
    return {"status": res.status.value,
            "obj": None if res.objective_value is None else int(res.objective_value),
            "Dpi": float(res.post_degree) if res.post_degree is not None else None,
            "ms": round(res.runtime_seconds * 1000, 2)}


def p8_robustness(situation, pairs, beta, alpha, missing_policy, epsilons):
    """Robust GreedyFlip: check OPT nondecreasing in epsilon and report radii."""
    rows = []
    monotone_violations = 0
    for a, b in pairs:
        pair = situation.pair(a, b, missing_policy=missing_policy)
        seq = []
        prev = -1
        for eps in epsilons:
            res = solve_greedy_flip(pair, beta=beta, alpha=alpha,
                                    operators=("mod", "del"),
                                    epsilon=eps, transition_robust=True)
            obj = None if res.objective_value is None else int(res.objective_value)
            feasible = res.status.value == "OPTIMAL"
            if feasible and obj is not None:
                if prev >= 0 and obj < prev:
                    monotone_violations += 1
                prev = obj
            seq.append({"eps": str(Fraction(eps)), "status": res.status.value, "obj": obj,
                        "Dpi": float(res.post_degree) if res.post_degree is not None else None})
        rows.append({"agent_a": a, "agent_b": b, "D0": float(pair.exact_degree()), "sweep": seq})
    return rows, {"monotone_violations": monotone_violations}


def sensitivity(situation, a, b, grid, missing_policy):
    pair = situation.pair(a, b, missing_policy=missing_policy)
    cells = []
    for beta, alpha in grid:
        gf = solve_greedy_flip(pair, beta=beta, alpha=alpha, operators=("mod", "del"))
        gd = solve_grid_dp(pair, beta=beta, alpha=alpha, operators=("mod",))
        cells.append({"beta": str(beta), "alpha": str(alpha),
                      "AR_obj": None if gf.objective_value is None else int(gf.objective_value),
                      "AR_status": gf.status.value,
                      "NR_obj": None if gd.objective_value is None else int(gd.objective_value),
                      "NR_status": gd.status.value})
    return {"agent_a": a, "agent_b": b, "D0": float(pair.exact_degree()), "grid": cells}


def safe_batch(situation, pairs, regimes, missing_policy):
    rows = []
    for a, b in pairs:
        for beta, alpha, target in regimes:
            t = time.time()
            res = solve_safe_ilp(situation, agent_a=a, agent_b=b, beta=beta, alpha=alpha,
                                 target=target, operators=("mod",),
                                 missing_policy=missing_policy, time_limit=120, mip_rel_gap=0.0)
            rows.append({
                "agent_a": a, "agent_b": b, "beta": str(beta), "alpha": str(alpha),
                "target": target, "status": res.status.value,
                "obj": None if res.objective_value is None else int(res.objective_value),
                "D0": round(float(res.original_degree), 4),
                "Dpi": round(float(res.post_degree), 4) if res.post_degree is not None else None,
                "audited": res.details.get("audited_bystander_pairs"),
                "wall_s": round(time.time() - t, 2),
            })
            print("  safe", a, b, str(beta), str(alpha), target, "->",
                  rows[-1]["status"], rows[-1]["obj"], rows[-1]["wall_s"], "s", flush=True)
    return rows


def main():
    report = {}

    # ---- UNGA ----
    print("== UNGA ==", flush=True)
    unga = SituationTable.load(PROC / "unga_2020_2025_v5_mask")
    unga_full = SituationTable.load(PROC / "unga_2020_2025_v5_neutral")  # complete table for Safe
    degs = all_pair_degrees(unga, "exclude")
    exact = [d for _, _, d in degs]
    q = T.quantile_thresholds(exact, lower_quantile="1/3", upper_quantile="9/10")
    beta_u, alpha_u = q.beta, q.alpha
    print(f"UNGA quantile thresholds beta={beta_u}({float(beta_u):.3f}) alpha={alpha_u}({float(alpha_u):.3f})", flush=True)
    pairs_u = top_cr_pairs(unga, degs, alpha_u, 15, "exclude")
    mat_u, agr_u = solver_matrix(unga, pairs_u, beta_u, alpha_u, "exclude")
    print("UNGA matrix agreement:", agr_u, flush=True)
    p8_u, p8s_u = p8_robustness(unga, pairs_u[:5], beta_u, alpha_u, "exclude",
                                [Fraction(0), Fraction(1, 50), Fraction(1, 20), Fraction(1, 10)])
    grid = T.sensitivity_grid(betas=["1/10", "1/5", "1/3"], alphas=["1/3", "1/2", "2/3"])
    sens_u = sensitivity(unga, pairs_u[0][0], pairs_u[0][1], grid, "exclude")
    # Safe: full complete table, mod-only, two regimes on top-5 pairs
    safe_pairs_u = top_cr_pairs(
        unga_full, all_pair_degrees(unga_full, "error"), Fraction(3, 4), 5, "error"
    )
    safe_u = safe_batch(unga_full, safe_pairs_u,
                        [(Fraction(1, 4), Fraction(3, 4), "NR"),
                         (Fraction(1, 2), Fraction(3, 4), "AR"),
                         (Fraction(1, 4), Fraction(3, 4), "AR")],
                        "error")
    report["unga"] = {
        "thresholds": q.as_dict(), "n_target_pairs": len(pairs_u),
        "solver_matrix": mat_u, "matrix_agreement": agr_u,
        "p8": p8_u, "p8_summary": p8s_u, "sensitivity": sens_u,
        "safe_full_table": {"n_bystanders": unga_full.n_agents * (unga_full.n_agents - 1) // 2,
                            "operators": "mod", "rows": safe_u},
    }

    # ---- UCI ----
    print("== UCI ==", flush=True)
    uci = SituationTable.load(PROC / "uci_1984_mask")
    uci_full = SituationTable.load(PROC / "uci_1984_neutral")
    degs2 = all_pair_degrees(uci, "exclude")
    exact2 = [d for _, _, d in degs2]
    q2 = T.quantile_thresholds(exact2, lower_quantile="1/3", upper_quantile="9/10")
    beta_c, alpha_c = q2.beta, q2.alpha
    print(f"UCI quantile thresholds beta={beta_c}({float(beta_c):.3f}) alpha={alpha_c}({float(alpha_c):.3f})", flush=True)
    pairs_c = top_cr_pairs(uci, degs2, alpha_c, 15, "exclude")
    mat_c, agr_c = solver_matrix(uci, pairs_c, beta_c, alpha_c, "exclude")
    print("UCI matrix agreement:", agr_c, flush=True)
    p8_c, p8s_c = p8_robustness(uci, pairs_c[:5], beta_c, alpha_c, "exclude",
                                [Fraction(0), Fraction(1, 50), Fraction(1, 20), Fraction(1, 10)])
    sens_c = sensitivity(uci, pairs_c[0][0], pairs_c[0][1], grid, "exclude")
    safe_pairs_c = top_cr_pairs(
        uci_full, all_pair_degrees(uci_full, "error"), Fraction(3, 4), 5, "error"
    )
    safe_c = safe_batch(uci_full, safe_pairs_c,
                        [(Fraction(1, 4), Fraction(3, 4), "NR"),
                         (Fraction(1, 4), Fraction(3, 4), "AR")],
                        "error")
    report["uci"] = {
        "thresholds": q2.as_dict(), "n_target_pairs": len(pairs_c),
        "solver_matrix": mat_c, "matrix_agreement": agr_c,
        "p8": p8_c, "p8_summary": p8s_c, "sensitivity": sens_c,
        "safe_full_table": {"n_bystanders": uci_full.n_agents * (uci_full.n_agents - 1) // 2,
                            "operators": "mod", "rows": safe_c},
    }

    (OUT / "formal_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("WROTE", OUT / "formal_results.json", flush=True)


if __name__ == "__main__":
    main()
