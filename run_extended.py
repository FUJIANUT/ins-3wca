"""Extended experiments: genetic baseline, side-effect/multiplicity, accuracy band.

E1  Genetic baseline (faithful Liu-style GA) vs the exact optimum across agenda
    sizes, showing where a stochastic search matches and where it degrades.
E2  Population-scale side effects and explanation multiplicity: how many
    bystander pairs a local alliance optimum disturbs, and how many issues are
    necessary in every deletion explanation.
E3  Accuracy band: exhaustive ground truth on small instances and ILP
    optimality certificates on medium ones confirm the polynomial solvers.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from itertools import product
from pathlib import Path
from statistics import median

import numpy as np

from mcie import baselines as B
from mcie import thresholds as T
from mcie.model import Region, SituationTable, PairInstance, classify_degree, integerize_weights
from mcie.solvers import solve_greedy_flip, solve_grid_dp, solve_pair_ilp
from mcie.solvers.common import build_pair_options, evaluate_pair_selection

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results"


# --------------------------------------------------------------------------- E1
def synth_pair(m):
    return PairInstance("a", "b", tuple(f"i{i}" for i in range(m)),
                        np.array([1] * m, dtype=np.int8), np.array([-1] * m, dtype=np.int8),
                        np.ones(m, dtype=np.int64), np.arange(m))


def e1_genetic_sweep():
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    rows = []
    for m in [8, 16, 32, 64, 128, 256]:
        p = synth_pair(m)
        opt = int(solve_greedy_flip(p, beta=beta, alpha=alpha, operators=("mod",)).objective_value)
        best, reached = None, False
        for s in range(3):
            ga = B.liu_genetic(p, beta=beta, alpha=alpha, target="AR", seed=s,
                               pop_size=80, generations=200)
            if ga.reached:
                reached = True
                best = ga.cardinality if best is None else min(best, ga.cardinality)
        rows.append({"m": m, "exact": opt, "ga_best": best, "reached": reached,
                     "ratio": round(best / opt, 3) if best else None})
        print("  E1", rows[-1], flush=True)
    return rows


def e1_uci_real():
    st = SituationTable.load(PROC / "uci_1984_mask")
    degs = T.exact_pairwise_degrees(st, missing_policy="exclude", skip_empty_pairs=True)
    q = T.quantile_thresholds(degs, lower_quantile="1/3", upper_quantile="9/10")
    ids = list(st.agent_ids)
    cr = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            try:
                pr = st.pair(a, b, missing_policy="exclude")
            except ValueError:
                continue
            if pr.exact_degree() >= q.alpha:
                cr.append((a, b))
    rng = random.Random(3)
    sample = rng.sample(cr, min(15, len(cr)))
    matched, reached = 0, 0
    for a, b in sample:
        pr = st.pair(a, b, missing_policy="exclude")
        opt = int(solve_greedy_flip(pr, beta=q.beta, alpha=q.alpha, operators=("mod",)).objective_value)
        ga = B.liu_genetic(pr, beta=q.beta, alpha=q.alpha, target="AR", seed=1,
                           pop_size=60, generations=150)
        reached += int(ga.reached)
        matched += int(ga.reached and ga.cardinality == opt)
    return {"n": len(sample), "reached": reached, "matched_exact": matched,
            "beta": str(q.beta), "alpha": str(q.alpha)}


# --------------------------------------------------------------------------- E2
def sample_cr(st, alpha, policy, n_strata, n_random, seed):
    ids = list(st.agent_ids)
    cr = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            try:
                pr = st.pair(a, b, missing_policy=policy)
            except ValueError:
                continue
            d = pr.exact_degree()
            if d >= alpha:
                cr.append((a, b, d))
    cr_sorted = sorted(cr, key=lambda x: (x[2], x[0], x[1]))
    picks = {}
    for k in range(n_strata):
        idx = round(k * (len(cr_sorted) - 1) / max(1, n_strata - 1))
        picks[(cr_sorted[idx][0], cr_sorted[idx][1])] = True
    rng = random.Random(seed)
    for a, b, _ in rng.sample(cr, min(n_random, len(cr))):
        picks.setdefault((a, b), True)
    return list(picks.keys())


def region_of(ratings, weights, observed, i, j, beta, alpha):
    """Region of pair (i,j) over their commonly observed issues (mask-consistent)."""
    sel = observed[i] & observed[j]
    if not sel.any():
        return None
    d2 = np.abs(ratings[i][sel].astype(np.int16) - ratings[j][sel].astype(np.int16))
    w = weights[sel]
    return classify_degree(Fraction(int(np.dot(w, d2)), 2 * int(w.sum())), beta, alpha)


def e2_side_effects_multiplicity(name, policy, beta, alpha, n_target):
    st = SituationTable.load(PROC / name)
    weights = integerize_weights(st.weights)
    ids = st.agent_ids
    pairs = sample_cr(st, alpha, policy, n_strata=n_target // 2, n_random=n_target // 2, seed=23)
    side_counts, side_fracs, zero_side = [], [], 0
    nec_fracs, nonunique = [], 0
    for a, b in pairs:
        ia, ib = ids.index(a), ids.index(b)
        pair = st.pair(a, b, missing_policy=policy)
        gf = solve_greedy_flip(pair, beta=beta, alpha=alpha, operators=("mod",))
        if gf.status.value != "OPTIMAL":
            continue
        # apply the local optimum to the full table
        changed = st.ratings.copy()
        for act in gf.plan:
            src = int(pair.source_issue_indices[act.issue_index])
            side = act.metadata.get("changed_side")
            agent = ia if side == "a" else ib
            changed[agent, src] = int(act.metadata.get("new_rating"))
        # count bystander pairs (incident to a* or b*) that change region,
        # over each bystander pair's own commonly observed issues
        incident = [j for j in range(st.n_agents) if j not in (ia, ib)]
        moved, usable = 0, 0
        for j in incident:
            for k in (ia, ib):
                before = region_of(st.ratings, weights, st.observed, k, j, beta, alpha)
                after = region_of(changed, weights, st.observed, k, j, beta, alpha)
                if before is None or after is None:
                    continue
                usable += 1
                if before != after:
                    moved += 1
        side_counts.append(moved)
        side_fracs.append(moved / usable if usable else 0.0)
        zero_side += int(moved == 0)
        # multiplicity: deletion-only necessary issues (Corollary 14) and gain ties.
        # Theta = sum over ALL pair issues (includes G's non-positive slack), per Thm 13.
        d = pair.doubled_distances
        bw = integerize_weights(pair.weights)
        vt = [Fraction(int(bw[t]) * int(d[t]), 2) - beta * int(bw[t]) for t in range(len(pair.issue_ids))]
        B_idx = [t for t, x in enumerate(vt) if x > 0]
        G_nonempty = any(x <= 0 for x in vt)  # deletion-only alliance feasible iff G != empty
        if B_idx and G_nonempty:
            Theta = sum(vt)  # over all issues in the pair's agenda I
            nec = sum(1 for t in B_idx if sum(vt[s] for s in B_idx if s != t) < Theta)
            nec_fracs.append(nec / len(B_idx))
        # non-unique mod optimum: gain ties at the alliance boundary
        gains = sorted((Fraction(int(bw[t]) * int(d[t]), 2) for t in range(len(pair.issue_ids))), reverse=True)
        k = int(gf.objective_value)
        nonunique += int(k < len(gains) and k > 0 and gains[k - 1] == gains[k])
    n = len(side_counts)
    nn = len(nec_fracs)
    return {
        "n_pairs": n,
        "side_effect_mean": round(sum(side_counts) / n, 2) if n else None,
        "side_effect_median": median(side_counts) if n else None,
        "side_effect_max": max(side_counts) if n else None,
        "side_effect_frac_mean": round(sum(side_fracs) / n, 4) if n else None,
        "zero_side_effect_rate": round(zero_side / n, 4) if n else None,
        "necessary_issue_n_pairs": nn,
        "necessary_issue_frac_mean": round(sum(nec_fracs) / nn, 4) if nn else None,
        "nonunique_optimum_rate": round(nonunique / n, 4) if n else None,
    }


# --------------------------------------------------------------------------- E3
def brute_force_optimum(pair, beta, alpha, target, full_menu):
    menus, _ = build_pair_options(pair, operators=("mod",), full_mod_menu=full_menu)
    best = None
    for combo in product(*[range(len(menu)) for menu in menus]):
        ev = evaluate_pair_selection(menus, combo, beta=beta, alpha=alpha)
        if ev["valid"] and ev["region"].value == target:
            if best is None or ev["touched"] < best:
                best = ev["touched"]
    return best


def rand_cr_pair(m, rng, beta, alpha, oppose=0.8):
    """Random pair biased toward opposition so the degree lands in CR."""
    for _ in range(400):
        a = np.array([rng.choice([-1, 1]) for _ in range(m)], dtype=np.int8)
        b = np.empty(m, dtype=np.int8)
        for i in range(m):
            b[i] = -a[i] if rng.random() < oppose else rng.choice([-1, 0, 1])
        p = PairInstance("a", "b", tuple(f"i{i}" for i in range(m)), a, b,
                         np.ones(m, dtype=np.int64), np.arange(m))
        if p.exact_degree() >= alpha:
            return p
    return None


def e3_accuracy_band():
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    rng = random.Random(11)
    small = []
    for m in [4, 6, 8, 10]:
        ar_ok = nr_ok = trials = 0
        for _ in range(60):
            p = rand_cr_pair(m, rng, beta, alpha)
            if p is None:
                continue
            trials += 1
            bf_ar = brute_force_optimum(p, beta, alpha, "AR", full_menu=False)
            gf = solve_greedy_flip(p, beta=beta, alpha=alpha, operators=("mod",))
            g_ar = int(gf.objective_value) if gf.status.value == "OPTIMAL" else None
            ar_ok += int(bf_ar == g_ar)
            if m <= 8:
                bf_nr = brute_force_optimum(p, beta, alpha, "NR", full_menu=True)
                gd = solve_grid_dp(p, beta=beta, alpha=alpha, operators=("mod",))
                g_nr = int(gd.objective_value) if gd.status.value == "OPTIMAL" else None
                nr_ok += int(bf_nr == g_nr)
        small.append({"m": m, "trials": trials, "AR_match_bruteforce": ar_ok,
                      "NR_match_bruteforce": nr_ok if m <= 8 else "n/a"})
        print("  E3-small", small[-1], flush=True)
    medium = []
    for m in [20, 50, 100, 200]:
        ar_ok = nr_ok = trials = 0
        for s in range(20):
            p = rand_cr_pair(m, random.Random(1000 + s), beta, alpha)
            if p is None:
                continue
            trials += 1
            gf = solve_greedy_flip(p, beta=beta, alpha=alpha, operators=("mod",))
            ilp_ar = solve_pair_ilp(p, target="AR", beta=beta, alpha=alpha, operators=("mod",))
            if gf.status.value == "OPTIMAL" and ilp_ar.status.value == "OPTIMAL":
                ar_ok += int(int(gf.objective_value) == int(ilp_ar.objective_value))
            gd = solve_grid_dp(p, beta=beta, alpha=alpha, operators=("mod",))
            ilp_nr = solve_pair_ilp(p, target="NR", beta=beta, alpha=alpha, operators=("mod",))
            if gd.status.value == "OPTIMAL" and ilp_nr.status.value == "OPTIMAL":
                nr_ok += int(int(gd.objective_value) == int(ilp_nr.objective_value))
        medium.append({"m": m, "trials": trials, "AR_match_ilp": ar_ok, "NR_match_ilp": nr_ok})
        print("  E3-medium", medium[-1], flush=True)
    return {"small_bruteforce": small, "medium_ilp": medium}


def main():
    print("== E1 genetic sweep ==", flush=True)
    e1 = {"synthetic_sweep": e1_genetic_sweep(), "uci_real": e1_uci_real()}
    print("  E1 uci_real", e1["uci_real"], flush=True)
    print("== E2 side effects and multiplicity ==", flush=True)
    e2 = {}
    for key, name, policy in [("unga", "unga_2020_2025_v5_mask", "exclude"),
                              ("uci", "uci_1984_mask", "exclude")]:
        st = SituationTable.load(PROC / name)
        degs = T.exact_pairwise_degrees(st, missing_policy=policy, skip_empty_pairs=True)
        q = T.quantile_thresholds(degs, lower_quantile="1/3", upper_quantile="9/10")
        e2[key] = e2_side_effects_multiplicity(name, policy, q.beta, q.alpha, n_target=150)
        print("  E2", key, e2[key], flush=True)
    print("== E3 accuracy band ==", flush=True)
    e3 = e3_accuracy_band()
    report = {"E1_genetic": e1, "E2_side_effect_multiplicity": e2, "E3_accuracy": e3}
    (OUT / "extended_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", OUT / "extended_results.json", flush=True)


if __name__ == "__main__":
    main()
