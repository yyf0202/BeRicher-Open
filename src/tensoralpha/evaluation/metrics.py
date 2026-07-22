"""OOF metrics that preserve the daily cross-sectional unit."""

from __future__ import annotations

import pandas as pd


def rank_ic_by_date(
    oof: pd.DataFrame,
    *,
    score_column: str = "score",
    return_column: str = "forward_return",
) -> pd.DataFrame:
    required = {"date", "symbol", score_column, return_column}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF panel missing columns: {sorted(missing)}")
    panel = oof.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    rows = []
    for date, group in panel.groupby("date", sort=True):
        valid = group[[score_column, return_column]].dropna()
        value = valid[score_column].corr(valid[return_column], method="spearman")
        rows.append({"date": date, "rank_ic": value, "observations": len(valid)})
    return pd.DataFrame(rows)


def _summary(panel: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float | int | None]:
    values = daily["rank_ic"].dropna()
    return {
        "observations": int(len(panel)),
        "days": int(panel["date"].nunique()),
        "rank_ic_mean": float(values.mean()) if len(values) else None,
        "rank_ic_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "rank_ic_positive_rate": float((values > 0).mean()) if len(values) else None,
    }


def summarize_oof(oof: pd.DataFrame) -> dict:
    panel = oof.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    daily = rank_ic_by_date(panel)
    annual = {}
    for year, year_panel in panel.groupby(panel["date"].dt.year, sort=True):
        year_daily = daily[daily["date"].dt.year == year]
        annual[str(year)] = _summary(year_panel, year_daily)
    return {"overall": _summary(panel, daily), "annual": annual}
