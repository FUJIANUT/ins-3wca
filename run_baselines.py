"""Baseline comparison, stratified/random sampling, and fairness experiment.

Compares the exact MCIE algorithms against adapted domain heuristics
(conflict-score ranking, Zhai-style cost ratio, Liu-style issue-order, random)
on stratified and random samples of conflict-region pairs, under the quantile
thresholds. Also reports heterogeneous-cost behavior and concession fairness.

The narrative is honest: for the unit-cost alliance target the conflict-score
heuristic matches the exact optimum, so the informative differences are the
neutrality open-band overshoot, the heterogeneous-cost gap, and fairness.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

from mcie import baselines as B
from mcie import thresholds as T
from mcie.model import Region, SituationTable
from mcie.solvers import solve_greedy_flip, solve_grid_dp, solve_pair_ilp

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"


def issue_cost_vector(issue_ids):
    """Deterministic heterogeneous per-issue costs in {1,..,5}."""
    return [Fraction(1 + (i * 7) % 5) for i in range(len(issue_ids))]


def cr_pairs(situation, alpha, policy):
    ids = list(situation.agent_ids)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            try:
                pair = situation.pair(a, b, missing_policy=policy)
            except ValueError:
                continue
            d = pair.exact_degree()
            if d >= alpha:
                out.append((a, b, d))
    return out


def sample_pairs(cr, n_strata, n_random, seed):
    """Stratified-by-degree-rank plus uniform-random pairs from the CR set."""
    cr_sorted = sorted(cr, key=lambda x: (x[2], x[0], x[1]))
    picks = {}
    if cr_sorted:
        for k in range(n_strata):
            idx = round(k * (len(cr_sorted) - 1) / max(1, n_strata - 1))
            a, b, d = cr_sorted[idx]
            picks[(a, b)] = ("stratified", d)
    rng = random.Random(seed)
    for a, b, d in rng.sample(cr, min(n_random, len(cr))):
        picks.setdefault((a, b), ("random", d))
    return [(a, b, kind, d) for (a, b), (kind, d) in picks.items()]


def run_dataset(name, policy, beta, alpha):
    situation = SituationTable.load(PROC / name)
    cr = cr_pairs(situation, alpha, policy)
    sampled = sample_pairs(cr, n_strata=20, n_random=20, seed=17)
    rows = []
    for a, b, kind, d0 in sampled:
        pair = situation.pair(a, b, missing_policy=policy)
        ic = issue_cost_vector(pair.issue_ids)
        rec = {"a": a, "b": b, "sample": kind, "D0": float(d0),
               "m": len(pair.issue_ids)}

        # --- unit-cost AR: exact GreedyFlip vs heuristics ---
        gf = solve_greedy_flip(pair, beta=beta, alpha=alpha, operators=("mod",))
        opt_ar = int(gf.objective_value) if gf.status.value == "OPTIMAL" else None
        rec["AR_opt"] = opt_ar
        for label, res in [
            ("csr", B.conflict_score_ranking(pair, beta=beta, alpha=alpha, target="AR")),
            ("liu", B.liu_issue_order(pair, beta=beta, alpha=alpha, target="AR")),
            ("rand", B.random_intervention(pair, beta=beta, alpha=alpha, target="AR", seed=5)),
        ]:
            rec[f"AR_{label}"] = {"reached": res.reached, "card": res.cardinality}

        # --- unit-cost NR: exact GridDP vs heuristics (overshoot matters) ---
        gd = solve_grid_dp(pair, beta=beta, alpha=alpha, operators=("mod",),
                           max_states=1_500_000)
        opt_nr = int(gd.objective_value) if gd.status.value == "OPTIMAL" else None
        rec["NR_opt"] = opt_nr
        for label, res in [
            ("csr", B.conflict_score_ranking(pair, beta=beta, alpha=alpha, target="NR")),
            ("liu", B.liu_issue_order(pair, beta=beta, alpha=alpha, target="NR")),
            ("rand", B.random_intervention(pair, beta=beta, alpha=alpha, target="NR", seed=5)),
        ]:
            rec[f"NR_{label}"] = {"reached": res.reached, "card": res.cardinality,
                                  "overshoot": res.overshoot}

        # --- heterogeneous-cost AR: exact ILP-cost vs cost-blind CSR vs Zhai ratio ---
        ilp = solve_pair_ilp(pair, target="AR", beta=beta, alpha=alpha,
                             operators=("mod",), issue_costs=ic)
        opt_cost = float(ilp.objective_value) if ilp.status.value == "OPTIMAL" else None
        rec["ARcost_opt"] = opt_cost
        csr_c = B.conflict_score_ranking(pair, beta=beta, alpha=alpha, target="AR", issue_costs=ic)
        zh_c = B.zhai_cost_ratio(pair, beta=beta, alpha=alpha, target="AR", issue_costs=ic)
        rec["ARcost_csr"] = {"reached": csr_c.reached, "cost": float(csr_c.cost)}
        rec["ARcost_zhai"] = {"reached": zh_c.reached, "cost": float(zh_c.cost)}

        # --- fairness: exact AR mod-only optimum vs balanced side assignment ---
        aligned = [p.issue_index for p in gf.plan] if gf.status.value == "OPTIMAL" else []
        default_sides = {"a": 0, "b": 0}
        for p in (gf.plan if gf.status.value == "OPTIMAL" else []):
            s = p.metadata.get("changed_side")
            if s in default_sides:
                default_sides[s] += 1
        bal = B.balanced_alignment(pair, aligned, beta=beta, alpha=alpha, target="AR")
        rec["fair_default_imbalance"] = B.concession_imbalance(default_sides)
        rec["fair_balanced_imbalance"] = B.concession_imbalance(bal.sides)
        rec["fair_balanced_reached"] = bal.reached and bal.cardinality == (opt_ar or -1)

        rows.append(rec)
    return rows


def aggregate(rows):
    def rate(pred):
        vals = [pred(r) for r in rows if pred(r) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    agg = {}
    for label in ("csr", "liu", "rand"):
        # AR: success and exact-match rate (vs opt)
        ar_reach = [r[f"AR_{label}"]["reached"] for r in rows if r["AR_opt"] is not None]
        ar_match = [r[f"AR_{label}"]["card"] == r["AR_opt"]
                    for r in rows if r["AR_opt"] is not None and r[f"AR_{label}"]["reached"]]
        nr_reach = [r[f"NR_{label}"]["reached"] for r in rows if r["NR_opt"] is not None]
        nr_over = [r[f"NR_{label}"]["overshoot"] for r in rows if r["NR_opt"] is not None]
        nr_match = [r[f"NR_{label}"]["card"] == r["NR_opt"]
                    for r in rows if r["NR_opt"] is not None and r[f"NR_{label}"]["reached"]]
        agg[label] = {
            "AR_success": round(sum(ar_reach) / len(ar_reach), 4) if ar_reach else None,
            "AR_exact_match_when_reached": round(sum(ar_match) / len(ar_match), 4) if ar_match else None,
            "NR_success": round(sum(nr_reach) / len(nr_reach), 4) if nr_reach else None,
            "NR_overshoot": round(sum(nr_over) / len(nr_over), 4) if nr_over else None,
            "NR_exact_match_when_reached": round(sum(nr_match) / len(nr_match), 4) if nr_match else None,
        }
    # cost gap
    ratios_csr, ratios_zhai = [], []
    for r in rows:
        if r["ARcost_opt"] and r["ARcost_opt"] > 0:
            if r["ARcost_csr"]["reached"]:
                ratios_csr.append(r["ARcost_csr"]["cost"] / r["ARcost_opt"])
            if r["ARcost_zhai"]["reached"]:
                ratios_zhai.append(r["ARcost_zhai"]["cost"] / r["ARcost_opt"])
    agg["cost_gap"] = {
        "csr_mean_cost_ratio": round(sum(ratios_csr) / len(ratios_csr), 3) if ratios_csr else None,
        "csr_max_cost_ratio": round(max(ratios_csr), 3) if ratios_csr else None,
        "zhai_mean_cost_ratio": round(sum(ratios_zhai) / len(ratios_zhai), 3) if ratios_zhai else None,
        "zhai_max_cost_ratio": round(max(ratios_zhai), 3) if ratios_zhai else None,
    }
    di = [r["fair_default_imbalance"] for r in rows]
    bi = [r["fair_balanced_imbalance"] for r in rows]
    agg["fairness"] = {
        "default_mean_imbalance": round(sum(di) / len(di), 3) if di else None,
        "balanced_mean_imbalance": round(sum(bi) / len(bi), 3) if bi else None,
        "balanced_always_same_cardinality": all(r["fair_balanced_reached"] for r in rows),
    }
    agg["n_pairs"] = len(rows)
    return agg


def main():
    report = {}
    for key, name, policy in [
        ("unga", "unga_2020_2025_v5_mask", "exclude"),
        ("uci", "uci_1984_mask", "exclude"),
    ]:
        situation = SituationTable.load(PROC / name)
        degs = T.exact_pairwise_degrees(situation, missing_policy=policy, skip_empty_pairs=True)
        q = T.quantile_thresholds(degs, lower_quantile="1/3", upper_quantile="9/10")
        print(f"== {key}: quantile thresholds ({float(q.beta):.3f},{float(q.alpha):.3f}) ==", flush=True)
        rows = run_dataset(name, policy, q.beta, q.alpha)
        agg = aggregate(rows)
        report[key] = {"thresholds": {"beta": str(q.beta), "alpha": str(q.alpha)},
                       "aggregate": agg, "rows": rows}
        print(f"   {key} aggregate: {json.dumps(agg, ensure_ascii=False)}", flush=True)
    (OUT / "baseline_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("WROTE", OUT / "baseline_results.json", flush=True)


if __name__ == "__main__":
    main()
