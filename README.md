# kbs-3wca — Minimal Counterfactual Issue Explanations for three-way conflict analysis

Research code for **Minimal Counterfactual Issue Explanations (MCIE)** in three-way
conflict analysis.

Given a pair of agents currently classified as being in **conflict**, the MCIE problem asks
for a minimum-cardinality or minimum-cost set of actionable issue interventions (attitude
modification, issue deletion, finite-level reweighting) that moves the pair into a requested
**neutrality** or **alliance** region. Regions use the standard three-way boundaries

```
AR (alliance):    D <= beta
NR (neutrality):  beta < D < alpha
CR (conflict):    D >= alpha
```

This repository contains the solvers (GreedyFlip, GridDP, exact ILP, and the bystander-safe
ILP), the data-preparation pipeline for two public voting datasets, the threshold protocols,
the heuristic baselines, and the scripts that regenerate every reported table and figure.

## Repository layout

```
src/mcie/            solver package (model, solvers, data loaders, thresholds, baselines, CLI)
tests/               pytest regression suite (exact-arithmetic checks vs brute force / ILP)
configs/             experiment configurations (datasets, thresholds, runs)
run_formal.py        threshold protocol, cross-solver agreement, robustness, safe study
run_baselines.py     adapted heuristics + concession-fairness comparison
run_scaling.py       synthetic complexity scaling
run_extended.py      genetic baseline, population-scale side effects/multiplicity, accuracy band
make_figures.py      regenerate the experimental figures from results/*.json
results/*.json       summary results consumed by the figures
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest                                   # 41 regression tests
```

Without installing, prefix commands with `PYTHONPATH=src python3 -m mcie.cli`.

## Reproducing the results

```bash
# UCI (public, auto-fetched)
mcie data fetch-uci
mcie data prepare-uci --unknown-policy mask
mcie data prepare-uci --unknown-policy neutral

# UNGA (see "Data" below — download once manually, then register+prepare)
mcie data register-unga --source-file /path/to/2026_02_06_ga_voting.csv
mcie data prepare-unga --nonvoting-policy mask
mcie data prepare-unga --nonvoting-policy neutral

# experiments
python run_formal.py
python run_baselines.py
python run_scaling.py
python run_extended.py
python make_figures.py       # writes fig_*.pdf
```

Reference environment for the reported timings: Apple M4 Max, macOS 26.5, CPython 3.12,
NumPy 2.4, SciPy 1.17 (bundled HiGHS via `scipy.optimize.milp`). All region tests and
degrees use exact `fractions.Fraction` arithmetic, so numeric results are
platform-independent; only wall-clock times depend on hardware.

## Data

The vote data is **not** redistributed in this repository. Obtain each dataset from its
official source and verify it with the SHA-256 digest below.

**UCI Congressional Voting Records** — [dataset page](https://archive.ics.uci.edu/dataset/105/congressional%2Bvoting%2Brecords),
DOI `10.24432/C5C01P`, CC BY 4.0. `mcie data fetch-uci` downloads it automatically.
Expected `house-votes-84.data` SHA-256:
`c87c14110a5ba91d4a1e313ec7392824458152bf071fa5f5452340488337936e`
(435 rows, 16 issues; `y -> +1`, `n -> -1`, `? -> 0`; masked or neutral).

**UN General Assembly voting data** — [UN Dag Hammarskjöld Library record 4060887](https://digitallibrary.un.org/record/4060887/),
Version 5 (February 2026, through 2025-12-30). The UNDL terms permit non-commercial use with
attribution but **prohibit automated downloads without prior approval**, so this repository
neither downloads nor redistributes the CSV. Download it once from the official record in a
browser and register it locally. Expected `2026_02_06_ga_voting.csv` SHA-256:
`9b8ddbee7490d9e31aa1b2f852c0b074ff58afd9e4bb67e629bcddbdde7852f1`
(codes `Y -> +1`, `N -> -1`, `A -> 0` observed; `X -> 0` unobserved under `mask`, observed
under the `neutral` sensitivity variant). The official v5 snapshot yields 193 x 650 for
2020--2025.

Required UN citation:

> United Nations Dag Hammarskjöld Library, General Assembly Voting Data, United Nations,
> version 5 (February 2026), 2026, downloaded from https://digitallibrary.un.org/record/4060887,
> [download date].

## Solver scope

- **GreedyFlip** — exact for unit-cost, single-pair AR instances; keeps a positive-weight
  action on gain ties, rejects the empty agenda, and recomputes gains at `max(0, beta-eps)`
  for robust transitions.
- **GridDP** — exact for finite-menu NR instances; state `(W, N2)` with `N2 = 2*sum(w*d)`,
  all strict boundaries via exact rational comparisons.
- **ILP** — SciPy's bundled HiGHS; open NR boundaries integerized as `>= 1` / `<= -1`; every
  incumbent independently re-applied and validated.
- **ILP-Safe** — concrete target-cell modifications that preserve every audited bystander
  pair's original region; requires a complete table or the declared `neutral` convention.

## License

Released under the MIT License (see `LICENSE`). The UCI and UN datasets retain their own
licenses and are not included here.
