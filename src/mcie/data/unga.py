"""UN General Assembly official dataset registration and preparation.

The UN Digital Library permits non-commercial research reuse with attribution,
but its terms prohibit automated downloads without approval.  This module
therefore never downloads the file.  The researcher downloads the linked CSV
once in a browser and registers it locally with a SHA-256 digest.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..model import SituationTable
from .common import register_local_file, utc_now, write_json


UNGA_LANDING_PAGE = "https://digitallibrary.un.org/record/4060887/"
UNGA_FILENAME = "2026_02_06_ga_voting.csv"
UNGA_CSV_LINK = (
    "https://digitallibrary.un.org/record/4060887/files/"
    "2026_02_06_ga_voting.csv"
)
UNGA_VERSION = "Version 5 (February 2026; data through 2025-12-30)"
UNGA_CITATION = (
    "United Nations Dag Hammarskjöld Library, General Assembly Voting Data, "
    "United Nations, version 5 (February 2026), 2026, downloaded from "
    "https://digitallibrary.un.org/record/4060887, [download date]"
)
UNGA_LICENSE = "Copyright United Nations; non-commercial use with attribution"

VOTE_MAP = {"Y": 1, "N": -1, "A": 0, "X": 0}
EXPECTED_V5_2020_2025 = {
    "n_agents": 193,
    "n_issues": 650,
    "vote_counts": {"A": 12002, "N": 8238, "X": 11361, "Y": 93849},
}
USECOLS = [
    "undl_id",
    "ms_code",
    "ms_name",
    "ms_vote",
    "date",
    "session",
    "resolution",
    "title",
    "agenda_title",
    "subjects",
    "total_yes",
    "total_no",
    "total_abstentions",
    "total_non_voting",
    "total_ms",
    "undl_link",
]


def register_unga_download(
    source_file: str | Path,
    raw_dir: str | Path,
) -> Path:
    """Register a browser-downloaded official CSV; no network call is made."""
    raw_dir = Path(raw_dir)
    destination = raw_dir / UNGA_FILENAME
    registration = register_local_file(source_file, destination)
    header = pd.read_csv(destination, nrows=0).columns.tolist()
    missing_columns = sorted(set(USECOLS) - set(header))
    if missing_columns:
        destination.unlink(missing_ok=True)
        raise ValueError(f"not the expected UNGA v5 CSV; missing {missing_columns}")
    manifest = {
        **registration,
        "landing_page": UNGA_LANDING_PAGE,
        "download_link": UNGA_CSV_LINK,
        "version": UNGA_VERSION,
        "citation": UNGA_CITATION,
        "terms": UNGA_LICENSE,
        "automated_download": False,
    }
    write_json(raw_dir / "unga_raw_manifest.json", manifest)
    return destination


def _normalize_sessions(sessions: Iterable[str | int] | None) -> set[str] | None:
    if sessions is None:
        return None
    normalized = {str(value).strip() for value in sessions if str(value).strip()}
    return normalized or None


def prepare_unga(
    raw_csv: str | Path,
    output_dir: str | Path,
    *,
    date_from: str | None = "2020-01-01",
    date_to: str | None = "2025-12-31",
    sessions: Iterable[str | int] | None = None,
    nonvoting_policy: str = "mask",
    min_issue_observed_fraction: float = 0.0,
    issue_manifest: str | Path | None = None,
    chunksize: int = 100_000,
    export_wide_csv: bool = False,
) -> SituationTable:
    """Create a static country-by-resolution three-valued situation table.

    Y/N/A are support/opposition/abstention and map to +1/-1/0.  X denotes
    non-voting; by default it is stored as zero but marked unobserved.  The
    alternative ``neutral`` policy is explicit and recorded for sensitivity
    analysis.  Pivot gaps (for example, non-members) always remain unobserved.

    The current official v5 file contains more 2020--2025 resolutions than the
    505 reported by Wang et al. (2026).  ``issue_manifest`` can pin an explicit
    set of ``undl_id`` values when an older experimental manifest is recovered.
    """
    if nonvoting_policy not in {"mask", "neutral"}:
        raise ValueError("nonvoting_policy must be mask or neutral")
    if not 0 <= min_issue_observed_fraction <= 1:
        raise ValueError("min_issue_observed_fraction must lie in [0,1]")
    raw_csv = Path(raw_csv)
    if not raw_csv.is_file():
        raise FileNotFoundError(raw_csv)
    session_filter = _normalize_sessions(sessions)

    pinned_issues: set[str] | None = None
    if issue_manifest is not None:
        manifest_frame = pd.read_csv(issue_manifest, dtype=str)
        if "undl_id" not in manifest_frame.columns:
            raise ValueError("issue_manifest must contain an undl_id column")
        pinned_issues = set(manifest_frame["undl_id"].dropna().astype(str))

    selected_chunks: list[pd.DataFrame] = []
    raw_counts: Counter[str] = Counter()
    dtype = {
        "undl_id": str,
        "ms_code": str,
        "ms_name": str,
        "ms_vote": str,
        "session": str,
        "resolution": str,
    }
    for chunk in pd.read_csv(
        raw_csv,
        usecols=USECOLS,
        dtype=dtype,
        parse_dates=["date"],
        chunksize=chunksize,
        low_memory=False,
    ):
        keep = pd.Series(True, index=chunk.index)
        if date_from is not None:
            keep &= chunk["date"] >= pd.Timestamp(date_from)
        if date_to is not None:
            keep &= chunk["date"] <= pd.Timestamp(date_to)
        if session_filter is not None:
            keep &= chunk["session"].astype(str).isin(session_filter)
        if pinned_issues is not None:
            keep &= chunk["undl_id"].astype(str).isin(pinned_issues)
        chunk = chunk.loc[keep].copy()
        if chunk.empty:
            continue
        raw_counts.update(chunk["ms_vote"].fillna("").astype(str))
        selected_chunks.append(chunk)

    if not selected_chunks:
        raise ValueError("the UNGA filters selected no rows")
    votes = pd.concat(selected_chunks, ignore_index=True)
    legal_codes = set(VOTE_MAP)
    actual_codes = set(votes["ms_vote"].dropna().astype(str).unique())
    if not actual_codes.issubset(legal_codes):
        raise ValueError(f"unexpected UNGA vote codes: {actual_codes-legal_codes}")

    votes["undl_id"] = votes["undl_id"].astype(str)
    votes["ms_code"] = votes["ms_code"].astype(str)
    duplicate = votes.duplicated(["ms_code", "undl_id"], keep=False)
    if duplicate.any():
        conflicting = (
            votes.loc[duplicate]
            .groupby(["ms_code", "undl_id"])["ms_vote"]
            .nunique()
        )
        if (conflicting > 1).any():
            raise ValueError("conflicting duplicate country-resolution votes")
        votes = votes.drop_duplicates(["ms_code", "undl_id"], keep="last")

    issue_meta = (
        votes.sort_values(["date", "resolution", "undl_id"])
        .drop_duplicates("undl_id", keep="last")
        .loc[
            :,
            [
                "undl_id",
                "date",
                "session",
                "resolution",
                "title",
                "agenda_title",
                "subjects",
                "total_yes",
                "total_no",
                "total_abstentions",
                "total_non_voting",
                "total_ms",
                "undl_link",
            ],
        ]
        .rename(columns={"undl_id": "issue_id"})
        .reset_index(drop=True)
    )
    issue_ids = issue_meta["issue_id"].astype(str).tolist()
    agents_latest = (
        votes.sort_values("date")
        .drop_duplicates("ms_code", keep="last")
        .loc[:, ["ms_code", "ms_name"]]
        .rename(columns={"ms_code": "agent_id", "ms_name": "agent_name"})
        .sort_values("agent_id")
        .reset_index(drop=True)
    )
    agent_ids = agents_latest["agent_id"].astype(str).tolist()

    votes["rating"] = votes["ms_vote"].map(VOTE_MAP).astype(np.int8)
    votes["is_observed"] = votes["ms_vote"].isin(["Y", "N", "A"])
    if nonvoting_policy == "neutral":
        votes.loc[votes["ms_vote"] == "X", "is_observed"] = True

    rating_frame = votes.pivot(index="ms_code", columns="undl_id", values="rating")
    observed_frame = votes.pivot(
        index="ms_code", columns="undl_id", values="is_observed"
    )
    rating_frame = rating_frame.reindex(index=agent_ids, columns=issue_ids)
    observed_frame = observed_frame.reindex(index=agent_ids, columns=issue_ids)
    ratings = rating_frame.fillna(0).to_numpy(dtype=np.int8)
    observed = observed_frame.fillna(False).to_numpy(dtype=bool)

    coverage = observed.mean(axis=0)
    keep_issues = coverage >= min_issue_observed_fraction
    if not keep_issues.any():
        raise ValueError("coverage filter removed every issue")
    if not keep_issues.all():
        ratings = ratings[:, keep_issues]
        observed = observed[:, keep_issues]
        issue_meta = issue_meta.loc[keep_issues].reset_index(drop=True)
        issue_ids = issue_meta["issue_id"].astype(str).tolist()
        coverage = coverage[keep_issues]
    issue_meta["observed_fraction"] = coverage

    situation = SituationTable(
        name="unga_2020_2025_v5",
        ratings=ratings,
        observed=observed,
        agent_ids=tuple(agent_ids),
        issue_ids=tuple(issue_ids),
        weights=np.ones(len(issue_ids), dtype=np.int64),
        agent_metadata=agents_latest,
        issue_metadata=issue_meta,
        provenance={
            "prepared_at": utc_now(),
            "landing_page": UNGA_LANDING_PAGE,
            "version": UNGA_VERSION,
            "citation": UNGA_CITATION,
            "terms": UNGA_LICENSE,
            "raw_file": str(raw_csv.resolve()),
            "automated_download": False,
            "date_from": date_from,
            "date_to": date_to,
            "sessions": sorted(session_filter) if session_filter else None,
            "nonvoting_policy": nonvoting_policy,
            "ternary_mapping": VOTE_MAP,
            "observed_semantics": {
                "Y": True,
                "N": True,
                "A": True,
                "X": nonvoting_policy == "neutral",
                "pivot_gap": False,
            },
            "raw_vote_counts_after_filter": dict(sorted(raw_counts.items())),
            "min_issue_observed_fraction": min_issue_observed_fraction,
            "issue_manifest": (
                str(Path(issue_manifest).resolve()) if issue_manifest else None
            ),
        },
    )
    is_default_v5_slice = (
        raw_csv.name == UNGA_FILENAME
        and date_from == "2020-01-01"
        and date_to == "2025-12-31"
        and session_filter is None
        and pinned_issues is None
        and min_issue_observed_fraction == 0
    )
    if is_default_v5_slice:
        actual = {
            "n_agents": situation.n_agents,
            "n_issues": situation.n_issues,
            "vote_counts": dict(sorted(raw_counts.items())),
        }
        if actual != EXPECTED_V5_2020_2025:
            raise ValueError(
                "official UNGA v5 2020-2025 validation failed: "
                f"expected {EXPECTED_V5_2020_2025}, got {actual}"
            )
        situation.provenance["snapshot_validation"] = {
            "status": "PASS",
            "expected": EXPECTED_V5_2020_2025,
        }
    situation.save(output_dir, export_wide_csv=export_wide_csv)
    return situation
