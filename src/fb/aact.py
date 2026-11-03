"""Load selected AACT flat-file tables straight from the pinned export zip.

The AACT static export is a zip of pipe-delimited text files, one per table
(`studies.txt`, `designs.txt`, ...). Only the tables named in `fb.config.AACT_TABLES`
are ever read; nothing is expanded to disk.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from . import config

AACT_ZIP = config.DATA_RAW / "aact" / f"{config.AACT_SNAPSHOT_DATE}_export.zip"


def list_members(zip_path: Path = AACT_ZIP) -> list[tuple[str, int]]:
    with zipfile.ZipFile(zip_path) as z:
        return [(i.filename, i.file_size) for i in z.infolist()]


def _member_for(z: zipfile.ZipFile, table: str) -> str:
    cands = [n for n in z.namelist() if n.split("/")[-1] == f"{table}.txt"]
    if not cands:
        raise FileNotFoundError(f"{table}.txt not in {z.filename}")
    return cands[0]


def load_table(
    table: str,
    usecols: list[str] | None = None,
    zip_path: Path = AACT_ZIP,
    **read_kwargs,
) -> pd.DataFrame:
    """Read one AACT table as strings (dates parsed by the caller)."""
    if table not in config.AACT_TABLES:
        raise ValueError(f"{table} is not in config.AACT_TABLES; add it deliberately")
    with zipfile.ZipFile(zip_path) as z:
        name = _member_for(z, table)
        with z.open(name) as fh:
            df = pd.read_csv(
                io.TextIOWrapper(fh, encoding="utf-8", newline=""),
                sep="|",
                quotechar='"',
                dtype=str,
                usecols=usecols,
                keep_default_na=False,
                na_values=[""],
                low_memory=False,
                **read_kwargs,
            )
    return df


def header_of(table: str, zip_path: Path = AACT_ZIP) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        name = _member_for(z, table)
        with z.open(name) as fh:
            line = io.TextIOWrapper(fh, encoding="utf-8").readline()
    return [c.strip() for c in line.rstrip("\r\n").split("|")]
