"""Shared, provenance-preserving data utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import requests


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    url: str
    filename: str
    landing_page: str
    citation: str
    license: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def download_file(
    spec: DownloadSpec,
    directory: str | Path,
    *,
    force: bool = False,
    timeout: tuple[int, int] = (15, 120),
) -> tuple[Path, dict[str, Any]]:
    """Download a source that explicitly permits automated retrieval.

    The UN Digital Library is intentionally *not* routed through this helper;
    its terms prohibit automated downloads without prior approval.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.filename
    if destination.exists() and not force:
        return destination, {
            "source": spec.__dict__,
            "downloaded_at": None,
            "reused_existing_file": True,
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    headers = {"User-Agent": "MCIE academic reproducibility pipeline/0.1"}
    with requests.get(
        spec.url, stream=True, timeout=timeout, headers=headers
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)
    manifest = {
        "source": spec.__dict__,
        "downloaded_at": utc_now(),
        "reused_existing_file": False,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    return destination, manifest


def register_local_file(
    source: str | Path,
    destination: str | Path,
    *,
    copy: bool = True,
) -> dict[str, Any]:
    source, destination = Path(source).expanduser(), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        if copy:
            shutil.copy2(source, destination)
        else:
            raise ValueError("copy=False is only valid when source is destination")
    return {
        "registered_at": utc_now(),
        "source_file": str(source.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }

