"""Load NIH ExPORTER Projects (by fiscal year) and the Clinical Studies link file.

Column names have drifted across fiscal years; `COLUMN_ALIASES` maps every known
spelling onto the plan §2 canonical names, and `column_drift_report` lists, per FY,
which canonical columns are present / aliased / missing so the drift is on record.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from . import config

EXPORTER_DIR = config.DATA_RAW / "exporter"
CLINICAL_STUDIES_CSV = EXPORTER_DIR / "ClinicalStudies.csv"

# canonical name -> accepted spellings (upper-cased, compared after stripping)
COLUMN_ALIASES: dict[str, list[str]] = {
    "APPLICATION_ID": ["APPLICATION_ID", "APPL_ID"],
    "CORE_PROJECT_NUM": ["CORE_PROJECT_NUM", "CORE_PROJECT_NUMBER"],
    "FULL_PROJECT_NUM": ["FULL_PROJECT_NUM", "PROJECT_NUMBER", "FULL_PROJECT_NUMBER"],
    "APPLICATION_TYPE": ["APPLICATION_TYPE", "APPL_TYPE_CODE", "APPLICATION_TYPE_CODE"],
    "ACTIVITY": ["ACTIVITY", "ACTIVITY_CODE"],
    "ADMINISTERING_IC": ["ADMINISTERING_IC", "IC", "ADMIN_IC", "ADMINISTERING_IC_CODE"],
    "FUNDING_ICs": ["FUNDING_ICS", "FUNDING_IC"],
    "TOTAL_COST": ["TOTAL_COST", "FY_TOTAL_COST", "TOTAL_COST_AMT"],
    "TOTAL_COST_SUB_PROJECT": ["TOTAL_COST_SUB_PROJECT", "FY_TOTAL_COST_SUB_PROJECTS", "TOTAL_COST_SUBPROJECT"],
    "DIRECT_COST_AMT": ["DIRECT_COST_AMT", "DIRECT_COST"],
    "INDIRECT_COST_AMT": ["INDIRECT_COST_AMT", "INDIRECT_COST"],
    "PROJECT_START": ["PROJECT_START", "PROJECT_START_DATE"],
    "PROJECT_END": ["PROJECT_END", "PROJECT_END_DATE"],
    "BUDGET_START": ["BUDGET_START", "BUDGET_START_DATE"],
    "BUDGET_END": ["BUDGET_END", "BUDGET_END_DATE"],
    "SUPPORT_YEAR": ["SUPPORT_YEAR"],
    "SUFFIX": ["SUFFIX", "SUFFIX_CODE"],
    "FY": ["FY", "FISCAL_YEAR"],
    "SUBPROJECT_ID": ["SUBPROJECT_ID", "SUB_PROJECT_ID"],
    "ORG_NAME": ["ORG_NAME"],
    "FUNDING_MECHANISM": ["FUNDING_MECHANISM"],
    "ARRA_FUNDED": ["ARRA_FUNDED"],
    "AWARD_NOTICE_DATE": ["AWARD_NOTICE_DATE"],
}
PLAN_COLUMNS = [
    "CORE_PROJECT_NUM", "APPLICATION_TYPE", "ACTIVITY", "ADMINISTERING_IC", "TOTAL_COST",
    "DIRECT_COST_AMT", "INDIRECT_COST_AMT", "PROJECT_START", "PROJECT_END", "BUDGET_START",
    "BUDGET_END", "SUPPORT_YEAR", "SUFFIX",
]
# columns actually needed by the feasibility pass (kept narrow to bound memory)
FEASIBILITY_COLUMNS = [
    "APPLICATION_ID", "CORE_PROJECT_NUM", "FULL_PROJECT_NUM", "APPLICATION_TYPE", "ACTIVITY",
    "ADMINISTERING_IC", "TOTAL_COST", "TOTAL_COST_SUB_PROJECT", "DIRECT_COST_AMT",
    "INDIRECT_COST_AMT", "PROJECT_START", "PROJECT_END", "BUDGET_START", "BUDGET_END",
    "SUPPORT_YEAR", "SUFFIX", "FY", "SUBPROJECT_ID", "FUNDING_MECHANISM",
]


def projects_zip(fy: int) -> Path:
    return EXPORTER_DIR / f"RePORTER_PRJ_C_FY{fy}.zip"


def _raw_header(fy: int) -> tuple[str, list[str], str]:
    """Return (member name, raw header list, encoding used)."""
    with zipfile.ZipFile(projects_zip(fy)) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise FileNotFoundError(f"no csv member in {projects_zip(fy)}")
        name = members[0]
        with z.open(name) as fh:
            raw = fh.readline()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            line = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return name, [c.strip().strip('"') for c in line.rstrip("\r\n").split(",")], enc


def resolve_columns(raw_cols: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map raw -> canonical for columns we know; return (mapping, unmapped raw columns)."""
    upper = {c.upper(): c for c in raw_cols}
    mapping: dict[str, str] = {}
    for canon, spellings in COLUMN_ALIASES.items():
        for sp in spellings:
            if sp.upper() in upper:
                mapping[upper[sp.upper()]] = canon
                break
    unmapped = [c for c in raw_cols if c not in mapping]
    return mapping, unmapped


def column_drift_report(fys: range | list[int] | None = None) -> pd.DataFrame:
    """One row per FY: for each plan §2 column, 'present' / 'aliased:<raw>' / 'MISSING'."""
    fys = list(fys) if fys is not None else list(range(config.EXPORTER_FY_FIRST, config.EXPORTER_FY_LAST + 1))
    rows = []
    for fy in fys:
        if not projects_zip(fy).exists():
            rows.append({"FY": fy, "file": "MISSING FILE"})
            continue
        name, cols, enc = _raw_header(fy)
        mapping, unmapped = resolve_columns(cols)
        inv = {v: k for k, v in mapping.items()}
        row = {"FY": fy, "file": name, "encoding": enc, "n_cols": len(cols)}
        for canon in PLAN_COLUMNS:
            if canon in inv:
                raw = inv[canon]
                row[canon] = "present" if raw.upper() == canon.upper() else f"aliased:{raw}"
            else:
                row[canon] = "MISSING"
        row["unmapped_raw_cols"] = ";".join(unmapped)
        rows.append(row)
    return pd.DataFrame(rows)


def load_projects_fy(fy: int, columns: list[str] = FEASIBILITY_COLUMNS) -> pd.DataFrame:
    """Load one FY projects file with canonical column names; missing canon columns -> NA."""
    name, raw_cols, enc = _raw_header(fy)
    mapping, _ = resolve_columns(raw_cols)
    want_raw = [raw for raw, canon in mapping.items() if canon in columns]
    with zipfile.ZipFile(projects_zip(fy)) as z:
        with z.open(name) as fh:
            df = pd.read_csv(
                fh, usecols=want_raw, dtype=str, encoding=enc, keep_default_na=False,
                na_values=[""], low_memory=False,
            )
    df = df.rename(columns=mapping)
    for c in columns:
        if c not in df.columns:
            df[c] = pd.NA
    df["FY_FILE"] = fy
    return df[columns + ["FY_FILE"]]


def load_projects_all(fys: range | list[int] | None = None, columns: list[str] = FEASIBILITY_COLUMNS) -> pd.DataFrame:
    fys = list(fys) if fys is not None else list(range(config.EXPORTER_FY_FIRST, config.EXPORTER_FY_LAST + 1))
    parts = [load_projects_fy(fy, columns) for fy in fys if projects_zip(fy).exists()]
    return pd.concat(parts, ignore_index=True)


def load_clinical_studies(path: Path = CLINICAL_STUDIES_CSV) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False, na_values=[""])
            break
        except UnicodeDecodeError:
            continue
    df.columns = [c.strip().strip('"') for c in df.columns]
    return df


def parse_exporter_date(s: pd.Series) -> pd.Series:
    """ExPORTER date columns drift in *format*: FY2000–FY2005 files use MM/DD/YYYY,
    FY2006 onward use ISO YYYY-MM-DD. Parse both; anything else -> NaT."""
    s = s.astype("string").str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    us = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    return iso.fillna(us)
