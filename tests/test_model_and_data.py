from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from mcie.data.uci import prepare_uci
from mcie.data.unga import prepare_unga
from mcie.model import Region, SituationTable, classify_degree


def test_region_boundaries_use_paper_notation():
    beta, alpha = Fraction(1, 4), Fraction(3, 4)
    assert classify_degree(beta, beta, alpha) == Region.AR
    assert classify_degree(Fraction(1, 2), beta, alpha) == Region.NR
    assert classify_degree(alpha, beta, alpha) == Region.CR


def test_missing_mask_is_distinct_from_neutral_and_round_trips(tmp_path):
    ratings = np.array([[1, 0, -1], [-1, 0, 1]], dtype=np.int8)
    observed = np.array([[True, False, True], [True, True, True]])
    table = SituationTable(
        "toy",
        ratings,
        observed,
        ("a", "b"),
        ("x", "y", "z"),
    )
    assert table.pair("a", "b", missing_policy="exclude").issue_ids == ("x", "z")
    assert table.pair("a", "b", missing_policy="neutral").issue_ids == (
        "x",
        "y",
        "z",
    )
    output = tmp_path / "bundle"
    table.save(output)
    loaded = SituationTable.load(output)
    assert np.array_equal(loaded.ratings, ratings)
    assert np.array_equal(loaded.observed, observed)


def test_official_uci_snapshot_counts_when_raw_file_is_available(tmp_path):
    raw = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "uci"
        / "house-votes-84.data"
    )
    if not raw.exists():
        import pytest

        pytest.skip("official UCI raw file has not been fetched")
    situation = prepare_uci(raw, tmp_path / "uci", unknown_policy="mask")
    assert situation.ratings.shape == (435, 16)
    assert int((situation.ratings == 1).sum()) == 3421
    assert int((situation.ratings == -1).sum()) == 3147
    assert int((~situation.observed).sum()) == 392


def test_unga_four_codes_and_nonvoting_mask(tmp_path):
    rows = []
    for issue, date in (("100", "2020-01-01"), ("101", "2021-01-01")):
        for code, name, vote in (
            ("AAA", "ALPHA", "Y" if issue == "100" else "A"),
            ("BBB", "BETA", "N" if issue == "100" else "X"),
        ):
            rows.append(
                {
                    "undl_id": issue,
                    "ms_code": code,
                    "ms_name": name,
                    "ms_vote": vote,
                    "date": date,
                    "session": "75",
                    "resolution": f"A/RES/{issue}",
                    "title": f"Resolution {issue}",
                    "agenda_title": "Agenda",
                    "subjects": "TEST",
                    "total_yes": 1,
                    "total_no": 1,
                    "total_abstentions": 0,
                    "total_non_voting": 0,
                    "total_ms": 2,
                    "undl_link": f"https://example.invalid/{issue}",
                }
            )
    raw = tmp_path / "unga.csv"
    pd.DataFrame(rows).to_csv(raw, index=False)
    masked = prepare_unga(raw, tmp_path / "masked", nonvoting_policy="mask")
    assert masked.ratings.tolist() == [[1, 0], [-1, 0]]
    assert masked.observed.tolist() == [[True, True], [True, False]]
    neutral = prepare_unga(raw, tmp_path / "neutral", nonvoting_policy="neutral")
    assert neutral.observed.all()

