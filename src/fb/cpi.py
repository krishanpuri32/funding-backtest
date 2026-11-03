"""CPI-U annual averages (FRED series CPIAUCSL, annual, average) for deflating
ExPORTER obligations to base-year dollars (plan §5 `F_total`: 2020 dollars).

Fiscal-year rows are deflated with the calendar-year average of the fiscal year's
nominal year (FY2015 -> CY2015 index); NIH FY runs Oct-Sep so this is a one-quarter
offset, documented and applied identically everywhere.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

CPI_DIR = config.DATA_RAW / "cpi"
CPI_PATH = CPI_DIR / "CPIAUCSL_annual.csv"


def parse_cpi_csv(text_or_path) -> dict[int, float]:
    """Parse the FRED csv (`observation_date,CPIAUCSL`) into {year: index}; blank values dropped."""
    df = pd.read_csv(text_or_path, dtype=str)
    date_col = next(c for c in df.columns if "date" in c.lower())
    val_col = next(c for c in df.columns if c != date_col)
    out: dict[int, float] = {}
    for d, v in zip(df[date_col], df[val_col]):
        if pd.isna(v) or str(v).strip() in ("", "."):
            continue
        out[int(str(d)[:4])] = float(v)
    return out


def load_cpi(path: Path = CPI_PATH) -> dict[int, float]:
    return parse_cpi_csv(path)


def deflator(cpi: dict[int, float], year: pd.Series, base_year: int = config.CPI_BASE_YEAR) -> pd.Series:
    """Multiplier that converts nominal dollars of `year` into base-year dollars; NA if year unknown."""
    base = cpi[base_year]
    y = pd.to_numeric(year, errors="coerce")
    return y.map(lambda v: base / cpi[int(v)] if pd.notna(v) and int(v) in cpi else float("nan"))


def to_base_dollars(amount: pd.Series, year: pd.Series, cpi: dict[int, float], base_year: int = config.CPI_BASE_YEAR) -> pd.Series:
    return pd.to_numeric(amount, errors="coerce") * deflator(cpi, year, base_year)
