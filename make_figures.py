#!/usr/bin/env python3
"""Generate publication-quality experimental figures (vector PDF) from results JSON.

Outputs to latex/figures/. All numbers are read directly from experiments/results/*.json
so the figures stay faithful to the reported experiments.
"""
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent.parent / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

GREY = "#333333"


def load(name):
    return json.loads((RESULTS / name).read_text())


# ---------------------------------------------------------------- Fig scaling
def fig_scaling():
    d = load("scaling_results.json")
    s1 = d["S1_m"]
    s2 = d["S2_weight_H"]
    fig, (axm, axh) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    # panel (a): runtime vs m (log-log)
    ms = [r["m"] for r in s1]
    gf = [r["greedy_ms"] for r in s1]
    dp = [r.get("griddp_ms") for r in s1]
    ilp = [r.get("ilp_ar_ms") for r in s1]
    ms_dp = [m for m, v in zip(ms, dp) if v is not None]
    dp = [v for v in dp if v is not None]
    axm.loglog(ms, gf, "o-", color="#1f5fbf", label="GreedyFlip (AR)")
    axm.loglog(ms_dp, dp, "s-", color="#c0392b", label="GridDP (NR)")
    axm.loglog(ms, ilp, "^--", color=GREY, label="ILP comparator")
    axm.set_xlabel("issues $m$")
    axm.set_ylabel("time (ms)")
    axm.set_title("(a) scaling in $m$ (equal weight)", fontsize=9)
    axm.legend(loc="upper left")

    # panel (b): runtime vs weight granularity H (log-log)
    H = [r["H"] for r in s2]
    gf2 = [r["greedy_ms"] for r in s2]
    dp2 = [r["griddp_ms"] for r in s2]
    axh.loglog(H, dp2, "s-", color="#c0392b", label="GridDP (NR)")
    axh.loglog(H, gf2, "o-", color="#1f5fbf", label="GreedyFlip (AR)")
    axh.set_xlabel("scaled total weight $H$")
    axh.set_ylabel("time (ms)")
    axh.set_title("(b) scaling in $H$ ($m=256$ fixed)", fontsize=9)
    axh.legend(loc="center left")

    fig.tight_layout()
    fig.savefig(OUT / "fig_scaling.pdf")
    plt.close(fig)
    print("wrote fig_scaling.pdf")


# ------------------------------------------------------------- Fig thresholds
def fig_thresholds():
    tp = load("threshold_protocol.json")
    unga = tp["unga_2020_2025_v5_mask"]
    # protocols to show for UNGA
    order = [
        ("quantile_p33_p90", "Quantile\n$p_{33}/p_{90}$"),
        ("quantile_p50_p95", "Quantile\n$p_{50}/p_{95}$"),
        ("dtrs_symmetric", "DTRS\nsymmetric"),
        ("dtrs_guarded", "DTRS\nguarded"),
    ]
    labels, AR, NR, CR = [], [], [], []
    for key, lab in order:
        if key not in unga:
            continue
        fr = unga[key]["region_fractions"]
        labels.append(lab)
        AR.append(fr["AR"]); NR.append(fr["NR"]); CR.append(fr["CR"])

    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    x = range(len(labels))
    ax.bar(x, AR, color="#2e7d32", label="alliance (AR)")
    ax.bar(x, NR, bottom=AR, color="#f4b400", label="neutrality (NR)")
    ax.bar(x, CR, bottom=[a + n for a, n in zip(AR, NR)], color="#c0392b", label="conflict (CR)")
    for i, cr in enumerate(CR):
        ax.text(i, 1.01, f"CR {cr*100:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("fraction of country pairs")
    ax.set_ylim(0, 1.12)
    ax.set_title("UNGA region partition by threshold protocol", fontsize=9)
    ax.grid(axis="x")
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.34), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_thresholds.pdf")
    plt.close(fig)
    print("wrote fig_thresholds.pdf")


# -------------------------------------------------------------- Fig heuristics
def fig_heuristics():
    d = load("baseline_results.json")
    fig, (axsc, axbar) = plt.subplots(1, 2, figsize=(6.6, 2.8))

    # panel (a): per-pair AR cardinality, heuristics vs exact optimum
    for dset, mk, col in [("unga", "o", "#1f5fbf"), ("uci", "s", "#c0392b")]:
        rows = d[dset]["rows"]
        opt = [r["AR_opt"] for r in rows]
        liu = [r["AR_liu"]["card"] for r in rows]
        rnd = [r["AR_rand"]["card"] for r in rows]
        axsc.scatter(opt, liu, marker=mk, s=16, facecolors="none",
                     edgecolors=col, label=f"issue-order ({dset.upper()})")
        axsc.scatter(opt, rnd, marker=mk, s=10, color=col, alpha=0.35,
                     label=f"random ({dset.upper()})")
    lim = 420
    axsc.plot([0, lim], [0, lim], "--", color=GREY, lw=0.9, label="exact optimum ($y=x$)")
    axsc.set_xlim(0, lim); axsc.set_ylim(0, lim)
    axsc.set_xlabel("exact AR cardinality")
    axsc.set_ylabel("heuristic AR cardinality")
    axsc.set_title("(a) heuristics overshoot the AR optimum", fontsize=9)
    axsc.legend(loc="upper left", fontsize=6.5)

    # panel (b): cost ratio + fairness imbalance bars
    cats = ["cost-blind\nranking", "cost-ratio\n(Zhai)"]
    unga_cg = d["unga"]["aggregate"]["cost_gap"]
    uci_cg = d["uci"]["aggregate"]["cost_gap"]
    unga_vals = [unga_cg["csr_mean_cost_ratio"], unga_cg["zhai_mean_cost_ratio"]]
    uci_vals = [uci_cg["csr_mean_cost_ratio"], uci_cg["zhai_mean_cost_ratio"]]
    import numpy as np
    x = np.arange(len(cats)); w = 0.35
    axbar.bar(x - w/2, unga_vals, w, color="#1f5fbf", label="UNGA")
    axbar.bar(x + w/2, uci_vals, w, color="#c0392b", label="UCI")
    axbar.axhline(1.0, color=GREY, ls="--", lw=0.9)
    axbar.set_xticks(x); axbar.set_xticklabels(cats, fontsize=8)
    axbar.set_ylabel("mean cost / exact optimum")
    axbar.set_ylim(0.9, 1.6)
    axbar.set_title("(b) heterogeneous-cost efficiency", fontsize=9)
    axbar.legend(loc="upper right")
    for xi, v in zip(x - w/2, unga_vals):
        axbar.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for xi, v in zip(x + w/2, uci_vals):
        axbar.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_heuristics.pdf")
    plt.close(fig)
    print("wrote fig_heuristics.pdf")


# ------------------------------------------------------------- Fig robustness
def fig_robustness():
    d = load("formal_results.json")
    fig, (au, ac) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    panels = [
        (au, "unga", "(a) UNGA (650-issue agenda)",
         [("o-", "#1f5fbf"), ("s-", "#3f8fdf"), ("^-", "#7fb3e8")]),
        (ac, "uci", "(b) UCI (16-issue agenda)",
         [("D-", "#c0392b"), ("v-", "#e07050"), ("*-", "#eaa090")]),
    ]
    for ax, dset, title, styles in panels:
        p8 = d[dset]["p8"]
        for idx, (st, col) in enumerate(styles):
            if idx >= len(p8):
                break
            e = p8[idx]
            xs = [float(Fraction(s["eps"])) for s in e["sweep"] if s.get("obj") is not None]
            ys = [s["obj"] for s in e["sweep"] if s.get("obj") is not None]
            if not ys:
                continue
            a = e.get("agent_a", "?"); b = e.get("agent_b", "?")
            ax.plot(xs, ys, st, color=col, label=f"{a}–{b}")
        ax.set_xlabel(r"robustness radius $\epsilon$")
        ax.set_ylabel("min robust AR cardinality")
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
        ax.margins(y=0.25)
    fig.suptitle("Robust explanation size is monotone in $\\epsilon$ (zero violations)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_robustness.pdf")
    plt.close(fig)
    print("wrote fig_robustness.pdf")


if __name__ == "__main__":
    fig_scaling()
    fig_thresholds()
    fig_heuristics()
    fig_robustness()
    print("all figures written to", OUT)
