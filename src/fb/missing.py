"""Plan §6 missing-value rule, shared by P, F and F₂: indicator + median for continuous
features, an explicit 'missing' level for categorical ones. No outcome is imputed.

The fill is lossless: `<col>_missing` marks the filled cells, so a later phase can
recompute medians inside each CV fold if it chooses to.
"""
from __future__ import annotations

import pandas as pd

MISSING_LEVEL = "missing"


def fill_numeric(df: pd.DataFrame, cols: list[str], medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Add `<c>_missing` (int 0/1) and fill NaN with the median (computed here unless given)."""
    out = df.copy()
    med = dict(medians or {})
    for c in cols:
        v = pd.to_numeric(out[c], errors="coerce")
        if c not in med:
            m = v.median()
            med[c] = float(m) if pd.notna(m) else 0.0
        out[f"{c}_missing"] = v.isna().astype(int)
        out[c] = v.fillna(med[c])
    return out, med


def fill_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c].astype("object")
        out[c] = s.where(s.notna() & s.astype(str).str.strip().ne(""), MISSING_LEVEL).astype(str)
    return out


def coverage_table(raw: pd.DataFrame, numeric: list[str], categorical: list[str], max_levels: int = 20) -> pd.DataFrame:
    """Per feature on the RAW (unfilled) values: n non-missing, % missing, median / IQR or level counts."""
    rows = []
    n = len(raw)
    for c in numeric:
        v = pd.to_numeric(raw[c], errors="coerce")
        nn = int(v.notna().sum())
        rows.append({"feature": c, "kind": "numeric", "n": n, "n_non_missing": nn,
                     "pct_missing": round(100 * (n - nn) / n, 1) if n else None,
                     "summary": (f"median {v.median():.4g} · IQR {v.quantile(.25):.4g}–{v.quantile(.75):.4g}"
                                 f" · min {v.min():.4g} · max {v.max():.4g}") if nn else "—"})
    for c in categorical:
        s = raw[c].astype("object")
        miss = s.isna() | s.astype(str).str.strip().eq("") | s.astype(str).eq(MISSING_LEVEL)
        nn = int((~miss).sum())
        vc = s.where(~miss, MISSING_LEVEL).astype(str).value_counts()
        levels = " · ".join(f"{k} {v}" for k, v in vc.head(max_levels).items())
        if len(vc) > max_levels:
            levels += f" · (+{len(vc) - max_levels} more)"
        rows.append({"feature": c, "kind": "categorical", "n": n, "n_non_missing": nn,
                     "pct_missing": round(100 * (n - nn) / n, 1) if n else None, "summary": levels})
    return pd.DataFrame(rows)


def to_markdown(t: pd.DataFrame) -> str:
    lines = ["| feature | kind | n | non-missing | % missing | summary |", "|---|---|---|---|---|---|"]
    for r in t.itertuples(index=False):
        lines.append(f"| {r.feature} | {r.kind} | {r.n} | {r.n_non_missing} | {r.pct_missing} | {r.summary} |")
    return "\n".join(lines)
