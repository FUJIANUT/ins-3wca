"""Configuration-driven cross-solver benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .model import Region, SituationTable, as_fraction, pairwise_degrees
from .solvers import solve_greedy_flip, solve_grid_dp, solve_pair_ilp


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def run_benchmark(config_path: str | Path) -> tuple[Path, Path]:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text("utf-8"))
    base = config_path.parent
    situation = SituationTable.load(_resolve(base, config["dataset"]))
    beta, alpha = as_fraction(config["beta"]), as_fraction(config["alpha"])
    missing_policy = config.get("missing_policy", "exclude")

    selection = config.get("pair_selection", {"strategy": "top_conflict", "count": 5})
    if "pairs" in selection:
        pairs = [(str(a), str(b)) for a, b in selection["pairs"]]
    else:
        degrees = pairwise_degrees(situation, missing_policy=missing_policy)
        degrees = degrees.loc[degrees["degree"] >= float(alpha)]
        degrees = degrees.sort_values(
            ["degree", "agent_a", "agent_b"], ascending=[False, True, True]
        )
        pairs = list(
            degrees.head(int(selection.get("count", 5)))[["agent_a", "agent_b"]]
            .itertuples(index=False, name=None)
        )
    if not pairs:
        raise ValueError("pair selection found no initially conflicting pair")

    records: list[dict[str, Any]] = []
    for agent_a, agent_b in pairs:
        pair = situation.pair(agent_a, agent_b, missing_policy=missing_policy)
        for run in config["runs"]:
            solver = str(run["solver"]).lower()
            target = Region(str(run.get("target", "AR")).upper())
            kwargs = {
                "beta": beta,
                "alpha": alpha,
                "operators": tuple(run.get("operators", ["mod", "del"])),
                "epsilon": as_fraction(run.get("epsilon", 0)),
                "transition_robust": bool(run.get("transition_robust", False)),
            }
            if solver == "greedy":
                if target != Region.AR:
                    raise ValueError("GreedyFlip supports only AR")
                result = solve_greedy_flip(pair, **kwargs)
            elif solver in {"grid", "grid-dp", "griddp"}:
                if target != Region.NR:
                    raise ValueError("GridDP supports only NR")
                result = solve_grid_dp(
                    pair,
                    **kwargs,
                    max_states=int(run.get("max_states", 2_000_000)),
                    time_limit=run.get("time_limit"),
                )
            elif solver == "ilp":
                result = solve_pair_ilp(
                    pair,
                    target=target,
                    **kwargs,
                    time_limit=run.get("time_limit"),
                    mip_rel_gap=run.get("mip_rel_gap", 0.0),
                )
            else:
                raise ValueError(f"unknown benchmark solver {solver!r}")
            record = {
                "dataset": situation.name,
                "agent_a": agent_a,
                "agent_b": agent_b,
                "usable_issues": len(pair.issue_ids),
                "beta": str(beta),
                "alpha": str(alpha),
                "missing_policy": missing_policy,
                **result.to_dict(),
            }
            records.append(record)

    output_stem = _resolve(base, config.get("output", "../results/benchmark"))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_stem.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_fields = [
        "dataset",
        "agent_a",
        "agent_b",
        "usable_issues",
        "solver",
        "target",
        "status",
        "objective_value",
        "touched_issue_count",
        "original_degree",
        "post_degree",
        "runtime_seconds",
        "guarantee",
    ]
    csv_path = output_stem.with_suffix(".csv")
    pd.DataFrame(records).loc[:, summary_fields].to_csv(csv_path, index=False)
    return jsonl_path, csv_path

