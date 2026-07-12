"""Canonical three-valued situation-table model and exact region utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import json
from math import gcd
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


class Region(str, Enum):
    AR = "AR"
    NR = "NR"
    CR = "CR"


def as_fraction(value: Fraction | int | float | str) -> Fraction:
    """Convert user input without inheriting a binary-float denominator."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value), 1)
    return Fraction(str(value))


def validate_thresholds(
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
) -> tuple[Fraction, Fraction]:
    beta_f, alpha_f = as_fraction(beta), as_fraction(alpha)
    if not (0 <= beta_f < alpha_f <= 1):
        raise ValueError(
            f"thresholds must satisfy 0 <= beta < alpha <= 1; got "
            f"beta={beta_f}, alpha={alpha_f}"
        )
    return beta_f, alpha_f


def classify_degree(
    degree: Fraction | int | float | str,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
) -> Region:
    beta_f, alpha_f = validate_thresholds(beta, alpha)
    degree_f = as_fraction(degree)
    if degree_f <= beta_f:
        return Region.AR
    if degree_f < alpha_f:
        return Region.NR
    return Region.CR


def degree_from_integer_state(n2: int, weight: int) -> Fraction:
    """Return D from W and N2=2*sum(w*d)."""
    if weight <= 0:
        raise ValueError("the post-intervention agenda must have positive weight")
    return Fraction(int(n2), 2 * int(weight))


def classify_integer_state(
    n2: int,
    weight: int,
    beta: Fraction | int | float | str,
    alpha: Fraction | int | float | str,
) -> Region:
    return classify_degree(degree_from_integer_state(n2, weight), beta, alpha)


def integerize_weights(
    weights: Sequence[Fraction | int | float | str],
    *,
    max_denominator: int = 1_000_000,
    max_total: int = 2_000_000,
) -> np.ndarray:
    """Convert proportional rational weights to a primitive integer vector.

    Scaling every weight by the same positive constant leaves all conflict
    degrees and region memberships unchanged.  A size guard prevents an
    accidental denominator explosion from silently creating an unusable DP.
    """
    fracs: list[Fraction] = []
    for value in weights:
        frac = as_fraction(value).limit_denominator(max_denominator)
        if frac <= 0:
            raise ValueError(f"weights must be positive, got {frac}")
        fracs.append(frac)

    lcm = 1
    for frac in fracs:
        lcm = lcm * frac.denominator // gcd(lcm, frac.denominator)
        if lcm > max_total * max_denominator:
            raise ValueError("weight denominator LCM is too large for exact scaling")

    ints = [frac.numerator * (lcm // frac.denominator) for frac in fracs]
    common = 0
    for value in ints:
        common = gcd(common, value)
    ints = [value // common for value in ints]
    if sum(ints) > max_total:
        raise ValueError(
            f"scaled total weight {sum(ints)} exceeds max_total={max_total}"
        )
    return np.asarray(ints, dtype=np.int64)


@dataclass(frozen=True)
class PairInstance:
    agent_a: str
    agent_b: str
    issue_ids: tuple[str, ...]
    attitudes_a: np.ndarray
    attitudes_b: np.ndarray
    weights: np.ndarray
    source_issue_indices: np.ndarray

    def __post_init__(self) -> None:
        m = len(self.issue_ids)
        if self.attitudes_a.shape != (m,) or self.attitudes_b.shape != (m,):
            raise ValueError("pair attitude vectors and issue_ids disagree")
        if self.weights.shape != (m,):
            raise ValueError("pair weights and issue_ids disagree")
        if m == 0:
            raise ValueError("a pair instance must contain at least one issue")
        if not np.isin(self.attitudes_a, (-1, 0, 1)).all():
            raise ValueError("attitudes_a contains a non-ternary value")
        if not np.isin(self.attitudes_b, (-1, 0, 1)).all():
            raise ValueError("attitudes_b contains a non-ternary value")
        if np.any(self.weights <= 0):
            raise ValueError("pair weights must be positive")

    @property
    def distances(self) -> np.ndarray:
        return np.abs(self.attitudes_a.astype(np.int16) - self.attitudes_b) / 2.0

    @property
    def doubled_distances(self) -> np.ndarray:
        return np.abs(self.attitudes_a.astype(np.int16) - self.attitudes_b).astype(
            np.int8
        )

    @property
    def degree(self) -> float:
        return float(np.dot(self.weights, self.distances) / self.weights.sum())

    def exact_degree(self) -> Fraction:
        integer_weights = integerize_weights(self.weights)
        n2 = int(np.dot(integer_weights, self.doubled_distances))
        return degree_from_integer_state(n2, int(integer_weights.sum()))


@dataclass
class SituationTable:
    name: str
    ratings: np.ndarray
    observed: np.ndarray
    agent_ids: tuple[str, ...]
    issue_ids: tuple[str, ...]
    weights: np.ndarray | None = None
    agent_metadata: pd.DataFrame | None = None
    issue_metadata: pd.DataFrame | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    FORMAT_VERSION = 1

    def __post_init__(self) -> None:
        self.agent_ids = tuple(str(value) for value in self.agent_ids)
        self.issue_ids = tuple(str(value) for value in self.issue_ids)
        self.ratings = np.asarray(self.ratings, dtype=np.int8)
        self.observed = np.asarray(self.observed, dtype=bool)
        if self.ratings.ndim != 2:
            raise ValueError("ratings must be a two-dimensional matrix")
        if self.observed.shape != self.ratings.shape:
            raise ValueError("observed mask must match ratings")
        n, m = self.ratings.shape
        if len(self.agent_ids) != n or len(self.issue_ids) != m:
            raise ValueError("metadata identifiers do not match matrix dimensions")
        if len(set(self.agent_ids)) != n or len(set(self.issue_ids)) != m:
            raise ValueError("agent_ids and issue_ids must be unique")
        if not np.isin(self.ratings, (-1, 0, 1)).all():
            bad = np.unique(self.ratings[~np.isin(self.ratings, (-1, 0, 1))])
            raise ValueError(f"ratings contains non-ternary values: {bad.tolist()}")
        if self.weights is None:
            self.weights = np.ones(m, dtype=np.int64)
        else:
            self.weights = np.asarray(self.weights)
        if self.weights.shape != (m,) or np.any(self.weights <= 0):
            raise ValueError("weights must be a positive vector of length n_issues")

        if self.agent_metadata is None:
            self.agent_metadata = pd.DataFrame({"agent_id": self.agent_ids})
        if self.issue_metadata is None:
            self.issue_metadata = pd.DataFrame({"issue_id": self.issue_ids})
        if "agent_id" not in self.agent_metadata.columns:
            raise ValueError("agent_metadata must contain agent_id")
        if "issue_id" not in self.issue_metadata.columns:
            raise ValueError("issue_metadata must contain issue_id")
        if tuple(self.agent_metadata["agent_id"].astype(str)) != self.agent_ids:
            raise ValueError("agent_metadata order must match agent_ids")
        if tuple(self.issue_metadata["issue_id"].astype(str)) != self.issue_ids:
            raise ValueError("issue_metadata order must match issue_ids")

    @property
    def n_agents(self) -> int:
        return self.ratings.shape[0]

    @property
    def n_issues(self) -> int:
        return self.ratings.shape[1]

    @property
    def missing_fraction(self) -> float:
        return 1.0 - float(self.observed.mean())

    def agent_index(self, agent: str | int) -> int:
        if isinstance(agent, (int, np.integer)):
            index = int(agent)
            if not 0 <= index < self.n_agents:
                raise IndexError(index)
            return index
        try:
            return self.agent_ids.index(str(agent))
        except ValueError as exc:
            raise KeyError(f"unknown agent_id {agent!r}") from exc

    def pair(
        self,
        agent_a: str | int,
        agent_b: str | int,
        *,
        missing_policy: str = "exclude",
    ) -> PairInstance:
        ia, ib = self.agent_index(agent_a), self.agent_index(agent_b)
        if ia == ib:
            raise ValueError("the target pair must contain two distinct agents")
        both_observed = self.observed[ia] & self.observed[ib]
        if missing_policy == "exclude":
            selected = both_observed
        elif missing_policy == "neutral":
            selected = np.ones(self.n_issues, dtype=bool)
        elif missing_policy == "error":
            if not both_observed.all():
                raise ValueError("target pair contains missing ratings")
            selected = both_observed
        else:
            raise ValueError(
                "missing_policy must be one of: exclude, neutral, error"
            )
        indices = np.flatnonzero(selected)
        if len(indices) == 0:
            raise ValueError("the target pair has no usable common issues")
        return PairInstance(
            agent_a=self.agent_ids[ia],
            agent_b=self.agent_ids[ib],
            issue_ids=tuple(self.issue_ids[i] for i in indices),
            attitudes_a=self.ratings[ia, indices].copy(),
            attitudes_b=self.ratings[ib, indices].copy(),
            weights=np.asarray(self.weights)[indices].copy(),
            source_issue_indices=indices,
        )

    def save(self, directory: str | Path, *, export_wide_csv: bool = False) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "matrix.npz",
            ratings=self.ratings,
            observed=self.observed,
            weights=np.asarray(self.weights),
        )
        self.agent_metadata.to_csv(directory / "agents.csv", index=False)
        self.issue_metadata.to_csv(directory / "issues.csv", index=False)

        validation = {
            "status": "PASS",
            "shape": [self.n_agents, self.n_issues],
            "rating_counts": {
                str(value): int((self.ratings == value).sum())
                for value in (-1, 0, 1)
            },
            "observed_cells": int(self.observed.sum()),
            "unobserved_cells": int((~self.observed).sum()),
            "missing_fraction": self.missing_fraction,
            "checks": {
                "ratings_are_ternary": bool(
                    np.isin(self.ratings, (-1, 0, 1)).all()
                ),
                "mask_matches_matrix": self.observed.shape == self.ratings.shape,
                "agent_ids_unique": len(set(self.agent_ids)) == self.n_agents,
                "issue_ids_unique": len(set(self.issue_ids)) == self.n_issues,
                "weights_positive": bool(np.all(np.asarray(self.weights) > 0)),
            },
        }
        (directory / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        manifest = {
            "format_version": self.FORMAT_VERSION,
            "name": self.name,
            "n_agents": self.n_agents,
            "n_issues": self.n_issues,
            "missing_fraction": self.missing_fraction,
            "rating_values": [-1, 0, 1],
            "files": {
                "matrix": "matrix.npz",
                "agents": "agents.csv",
                "issues": "issues.csv",
                "validation": "validation.json",
            },
            "provenance": self.provenance,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        if export_wide_csv:
            wide = pd.DataFrame(self.ratings, columns=self.issue_ids)
            wide.insert(0, "agent_id", self.agent_ids)
            wide.to_csv(directory / "ratings.csv.gz", index=False, compression="gzip")
        else:
            (directory / "ratings.csv.gz").unlink(missing_ok=True)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "SituationTable":
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
        if manifest.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                f"unsupported bundle format {manifest.get('format_version')}"
            )
        arrays = np.load(directory / manifest["files"]["matrix"])
        agents = pd.read_csv(
            directory / manifest["files"]["agents"], dtype={"agent_id": str}
        )
        issues = pd.read_csv(
            directory / manifest["files"]["issues"], dtype={"issue_id": str}
        )
        return cls(
            name=manifest["name"],
            ratings=arrays["ratings"],
            observed=arrays["observed"],
            weights=arrays["weights"],
            agent_ids=tuple(agents["agent_id"].astype(str)),
            issue_ids=tuple(issues["issue_id"].astype(str)),
            agent_metadata=agents,
            issue_metadata=issues,
            provenance=manifest.get("provenance", {}),
        )


def pairwise_degrees(
    situation: SituationTable,
    *,
    missing_policy: str = "exclude",
    agents: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute all unordered pair degrees for inspection/pair selection."""
    ids = list(agents) if agents is not None else list(situation.agent_ids)
    records: list[dict[str, Any]] = []
    for i, agent_a in enumerate(ids):
        for agent_b in ids[i + 1 :]:
            pair = situation.pair(agent_a, agent_b, missing_policy=missing_policy)
            records.append(
                {
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "degree": pair.degree,
                    "n_issues": len(pair.issue_ids),
                }
            )
    return pd.DataFrame.from_records(records)
