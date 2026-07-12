"""Command-line interface for data preparation, solving, and benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .benchmark import run_benchmark
from .data import fetch_uci, prepare_uci, prepare_unga, register_unga_download
from .data.unga import UNGA_CSV_LINK, UNGA_FILENAME, UNGA_LANDING_PAGE
from .model import Region, SituationTable, as_fraction
from .solvers import solve_greedy_flip, solve_grid_dp, solve_pair_ilp, solve_safe_ilp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"


def _operators(text: str) -> tuple[str, ...]:
    values = tuple(value.strip().lower() for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one operator is required")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mcie")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="acquire or prepare a dataset")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    fetch = data_sub.add_parser("fetch-uci", help="download the official UCI ZIP")
    fetch.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW / "uci")
    fetch.add_argument("--force", action="store_true")

    unga_register = data_sub.add_parser(
        "register-unga", help="register a manually downloaded official UNGA CSV"
    )
    unga_register.add_argument("--source-file", type=Path)
    unga_register.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW / "unga")

    uci = data_sub.add_parser("prepare-uci")
    uci.add_argument("--raw-data", type=Path, default=DEFAULT_RAW / "uci" / "house-votes-84.data")
    uci.add_argument("--output", type=Path)
    uci.add_argument("--unknown-policy", choices=("mask", "neutral"), default="mask")
    uci.add_argument("--no-wide-csv", action="store_true")

    unga = data_sub.add_parser("prepare-unga")
    unga.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW / "unga" / UNGA_FILENAME)
    unga.add_argument("--output", type=Path, default=DEFAULT_PROCESSED / "unga_2020_2025_v5_mask")
    unga.add_argument("--date-from", default="2020-01-01")
    unga.add_argument("--date-to", default="2025-12-31")
    unga.add_argument("--sessions", nargs="*")
    unga.add_argument("--nonvoting-policy", choices=("mask", "neutral"), default="mask")
    unga.add_argument("--min-issue-observed-fraction", type=float, default=0.0)
    unga.add_argument("--issue-manifest", type=Path)
    unga.add_argument("--wide-csv", action="store_true")

    inspect = sub.add_parser("inspect", help="summarize a processed bundle")
    inspect.add_argument("--dataset", type=Path, required=True)

    solve = sub.add_parser("solve", help="solve one designated target pair")
    solve.add_argument("--dataset", type=Path, required=True)
    solve.add_argument("--agent-a", required=True)
    solve.add_argument("--agent-b", required=True)
    solve.add_argument("--solver", choices=("greedy", "grid-dp", "ilp", "safe-ilp"), required=True)
    solve.add_argument("--target", choices=("AR", "NR"), default="AR")
    solve.add_argument("--beta", required=True)
    solve.add_argument("--alpha", required=True)
    solve.add_argument("--epsilon", default="0")
    solve.add_argument("--operators", type=_operators, default=("mod", "del"))
    solve.add_argument("--missing-policy", choices=("exclude", "neutral", "error"), default="exclude")
    solve.add_argument("--transition-robust", action="store_true")
    solve.add_argument("--time-limit", type=float)
    solve.add_argument("--max-states", type=int, default=2_000_000)
    solve.add_argument("--output", type=Path)

    benchmark = sub.add_parser("benchmark", help="run a YAML benchmark config")
    benchmark.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data":
        if args.data_command == "fetch-uci":
            path = fetch_uci(args.raw_dir, force=args.force)
            print(path)
            return 0
        if args.data_command == "register-unga":
            if args.source_file is None:
                print(
                    "UNDL terms prohibit automated downloads without approval.\n"
                    f"Open {UNGA_LANDING_PAGE}\n"
                    f"Download the CSV link manually: {UNGA_CSV_LINK}\n"
                    "Then rerun with --source-file /path/to/2026_02_06_ga_voting.csv"
                )
                return 2
            print(register_unga_download(args.source_file, args.raw_dir))
            return 0
        if args.data_command == "prepare-uci":
            if not args.raw_data.exists():
                fetch_uci(args.raw_data.parent)
            output = args.output or (
                DEFAULT_PROCESSED / f"uci_1984_{args.unknown_policy}"
            )
            situation = prepare_uci(
                args.raw_data,
                output,
                unknown_policy=args.unknown_policy,
                export_wide_csv=not args.no_wide_csv,
            )
            print(json.dumps({"output": str(output), "shape": [situation.n_agents, situation.n_issues], "missing_fraction": situation.missing_fraction}, indent=2))
            return 0
        if args.data_command == "prepare-unga":
            situation = prepare_unga(
                args.raw_csv,
                args.output,
                date_from=args.date_from,
                date_to=args.date_to,
                sessions=args.sessions,
                nonvoting_policy=args.nonvoting_policy,
                min_issue_observed_fraction=args.min_issue_observed_fraction,
                issue_manifest=args.issue_manifest,
                export_wide_csv=args.wide_csv,
            )
            print(json.dumps({"output": str(args.output), "shape": [situation.n_agents, situation.n_issues], "missing_fraction": situation.missing_fraction}, indent=2))
            return 0

    if args.command == "inspect":
        situation = SituationTable.load(args.dataset)
        print(
            json.dumps(
                {
                    "name": situation.name,
                    "shape": [situation.n_agents, situation.n_issues],
                    "missing_fraction": situation.missing_fraction,
                    "rating_counts": {
                        str(value): int((situation.ratings == value).sum())
                        for value in (-1, 0, 1)
                    },
                    "provenance": situation.provenance,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0

    if args.command == "solve":
        situation = SituationTable.load(args.dataset)
        beta, alpha, epsilon = (
            as_fraction(args.beta),
            as_fraction(args.alpha),
            as_fraction(args.epsilon),
        )
        if args.solver == "safe-ilp":
            result = solve_safe_ilp(
                situation,
                agent_a=args.agent_a,
                agent_b=args.agent_b,
                target=args.target,
                beta=beta,
                alpha=alpha,
                operators=args.operators,
                missing_policy=args.missing_policy,
                time_limit=args.time_limit,
            )
        else:
            pair = situation.pair(
                args.agent_a, args.agent_b, missing_policy=args.missing_policy
            )
            common = {
                "beta": beta,
                "alpha": alpha,
                "operators": args.operators,
                "epsilon": epsilon,
                "transition_robust": args.transition_robust,
            }
            if args.solver == "greedy":
                if args.target != "AR":
                    raise ValueError("GreedyFlip supports only AR")
                result = solve_greedy_flip(pair, **common)
            elif args.solver == "grid-dp":
                if args.target != "NR":
                    raise ValueError("GridDP supports only NR")
                result = solve_grid_dp(
                    pair,
                    **common,
                    max_states=args.max_states,
                    time_limit=args.time_limit,
                )
            else:
                result = solve_pair_ilp(
                    pair,
                    target=args.target,
                    **common,
                    time_limit=args.time_limit,
                )
        payload = result.to_json()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0 if result.status.value in {"OPTIMAL", "FEASIBLE_TIMEOUT"} else 1

    if args.command == "benchmark":
        jsonl_path, csv_path = run_benchmark(args.config)
        print(json.dumps({"jsonl": str(jsonl_path), "csv": str(csv_path)}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

