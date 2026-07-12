"""UCI 1984 Congressional Voting Records acquisition and ternarization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from ..model import SituationTable
from .common import DownloadSpec, download_file, sha256_file, utc_now, write_json


UCI_SPEC = DownloadSpec(
    name="Congressional Voting Records",
    url=(
        "https://archive.ics.uci.edu/static/public/105/"
        "congressional%2Bvoting%2Brecords.zip"
    ),
    filename="congressional_voting_records.zip",
    landing_page=(
        "https://archive.ics.uci.edu/dataset/105/"
        "congressional%2Bvoting%2Brecords"
    ),
    citation=(
        "Congressional Voting Records [Dataset]. (1987). "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5C01P"
    ),
    license="CC BY 4.0",
)

ISSUES = (
    "handicapped-infants",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missile",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa",
)


def fetch_uci(raw_dir: str | Path, *, force: bool = False) -> Path:
    raw_dir = Path(raw_dir)
    archive, manifest = download_file(UCI_SPEC, raw_dir, force=force)
    with zipfile.ZipFile(archive) as bundle:
        required = {"house-votes-84.data", "house-votes-84.names"}
        if not required.issubset(bundle.namelist()):
            raise ValueError(
                f"UCI archive is missing files: {required-set(bundle.namelist())}"
            )
        for name in sorted(required):
            output = raw_dir / name
            with bundle.open(name) as source, output.open("wb") as target:
                target.write(source.read())
    manifest["extracted_files"] = sorted(required)
    write_json(raw_dir / "uci_raw_manifest.json", manifest)
    return raw_dir / "house-votes-84.data"


def prepare_uci(
    raw_data: str | Path,
    output_dir: str | Path,
    *,
    unknown_policy: str = "mask",
    export_wide_csv: bool = True,
) -> SituationTable:
    """Prepare the 435x16 table.

    ``?`` means a disposition other than yea/nay, not a literal unknown in the
    original codebook.  Under the default ``mask`` policy it is stored as zero
    in the compact matrix but marked unobserved, so pair solvers exclude that
    issue.  ``neutral`` exposes the alternative sensitivity convention in
    which ``?`` becomes an observed three-way neutral rating.
    """
    if unknown_policy not in {"mask", "neutral"}:
        raise ValueError("unknown_policy must be mask or neutral")
    raw_data = Path(raw_data)
    frame = pd.read_csv(
        raw_data,
        header=None,
        names=("party", *ISSUES),
        dtype=str,
        keep_default_na=False,
    )
    if frame.shape != (435, 17):
        raise ValueError(f"expected UCI shape (435,17), got {frame.shape}")
    legal = {"y", "n", "?"}
    observed_codes = set(frame.loc[:, ISSUES].stack().unique())
    if not observed_codes.issubset(legal):
        raise ValueError(f"unexpected UCI vote codes: {observed_codes-legal}")

    raw_votes = frame.loc[:, ISSUES].to_numpy(dtype=str)
    party_counts = frame["party"].str.lower().value_counts().to_dict()
    if party_counts != {"democrat": 267, "republican": 168}:
        raise ValueError(f"unexpected UCI party counts: {party_counts}")
    ratings = np.zeros(raw_votes.shape, dtype=np.int8)
    ratings[raw_votes == "y"] = 1
    ratings[raw_votes == "n"] = -1
    observed = raw_votes != "?"
    if unknown_policy == "neutral":
        observed[:] = True

    agent_ids = tuple(f"rep_{index:04d}" for index in range(1, len(frame) + 1))
    agents = pd.DataFrame(
        {
            "agent_id": agent_ids,
            "row_number": np.arange(1, len(frame) + 1),
            "party": frame["party"].str.lower(),
        }
    )
    issues = pd.DataFrame(
        {
            "issue_id": ISSUES,
            "issue_name": ISSUES,
            "source_column": np.arange(2, 18),
        }
    )
    counts = Counter(raw_votes.ravel().tolist())
    if counts != Counter({"y": 3421, "n": 3147, "?": 392}):
        raise ValueError(f"unexpected UCI vote counts: {dict(counts)}")
    situation = SituationTable(
        name="uci_congressional_votes_1984",
        ratings=ratings,
        observed=observed,
        agent_ids=agent_ids,
        issue_ids=ISSUES,
        weights=np.ones(len(ISSUES), dtype=np.int64),
        agent_metadata=agents,
        issue_metadata=issues,
        provenance={
            "prepared_at": utc_now(),
            "source": UCI_SPEC.__dict__,
            "raw_file": str(raw_data.resolve()),
            "raw_file_sha256": sha256_file(raw_data),
            "unknown_policy": unknown_policy,
            "ternary_mapping": {"y": 1, "n": -1, "?": 0},
            "observed_semantics": {
                "y": True,
                "n": True,
                "?": unknown_policy == "neutral",
            },
            "raw_vote_counts": dict(sorted(counts.items())),
            "party_counts": dict(sorted(party_counts.items())),
        },
    )
    situation.save(output_dir, export_wide_csv=export_wide_csv)
    return situation
