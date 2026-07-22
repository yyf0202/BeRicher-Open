"""Privacy-preserving aggregate and synthetic showcase for OOF scores."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _year_statistics(scores: pd.Series, dates: pd.Series) -> dict:
    numeric = pd.to_numeric(scores, errors="coerce").dropna()
    return {
        "observations": int(len(numeric)),
        "dates": int(pd.to_datetime(dates).nunique()),
        "score_mean": float(numeric.mean()),
        "score_std": float(numeric.std(ddof=1)),
        **{f"score_q{int(q * 100):02d}": float(numeric.quantile(q)) for q in _QUANTILES},
    }


def _base_summary(
    *, model_family: str, start: pd.Timestamp, end: pd.Timestamp, observations: int, annual: dict
) -> dict:
    return {
        "schema_version": 1,
        "model_family": model_family,
        "derived_from_oof": True,
        "coverage": {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
            "observations": int(observations),
            "years": len(annual),
        },
        "privacy": {
            "aggregate_only": True,
            "contains_security_identifiers": False,
            "contains_prices_or_returns": False,
            "contains_account_data": False,
        },
        "annual": annual,
    }


def derive_oof_showcase(oof: pd.DataFrame, *, model_family: str) -> dict:
    """Summarize an in-memory OOF panel without retaining identifiers."""

    required = {"trade_date", "alpha_score"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF source missing columns: {sorted(missing)}")
    dates = pd.to_datetime(oof["trade_date"].astype(str), format="%Y%m%d")
    annual = {}
    for year in sorted(dates.dt.year.unique()):
        mask = dates.dt.year == year
        annual[str(year)] = _year_statistics(oof.loc[mask, "alpha_score"], dates[mask])
    return _base_summary(
        model_family=model_family,
        start=dates.min(),
        end=dates.max(),
        observations=len(oof),
        annual=annual,
    )


@dataclass(slots=True)
class _StreamingYear:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    dates: set[int] = field(default_factory=set)
    samples: list[np.ndarray] = field(default_factory=list)

    def add(self, values: np.ndarray, date_values: np.ndarray, rng: np.random.Generator) -> None:
        finite = values[np.isfinite(values)]
        self.count += len(finite)
        self.total += float(finite.sum())
        self.total_squared += float(np.square(finite).sum())
        self.dates.update(int(value) for value in np.unique(date_values))
        sample_size = min(5_000, len(finite))
        if sample_size:
            indices = rng.choice(len(finite), size=sample_size, replace=False)
            self.samples.append(finite[indices])

    def summary(self) -> dict:
        mean = self.total / self.count
        variance = (
            (self.total_squared - self.count * mean * mean) / (self.count - 1)
            if self.count > 1
            else 0.0
        )
        sample = np.concatenate(self.samples) if self.samples else np.array([mean])
        return {
            "observations": self.count,
            "dates": len(self.dates),
            "score_mean": mean,
            "score_std": math.sqrt(max(0.0, variance)),
            **{f"score_q{int(q * 100):02d}": float(np.quantile(sample, q)) for q in _QUANTILES},
        }


def derive_oof_showcase_csv(
    path: str | Path,
    *,
    model_family: str,
    chunksize: int = 500_000,
    seed: int = 42,
) -> dict:
    """Stream a large private OOF CSV and retain only aggregate state."""

    rng = np.random.default_rng(seed)
    years: dict[int, _StreamingYear] = defaultdict(_StreamingYear)
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    observations = 0
    for chunk in pd.read_csv(
        path,
        usecols=["trade_date", "alpha_score"],
        chunksize=chunksize,
    ):
        dates = pd.to_datetime(chunk["trade_date"].astype(str), format="%Y%m%d")
        scores = pd.to_numeric(chunk["alpha_score"], errors="coerce").to_numpy(float)
        observations += len(chunk)
        start = dates.min() if start is None else min(start, dates.min())
        end = dates.max() if end is None else max(end, dates.max())
        for year in dates.dt.year.unique():
            mask = dates.dt.year.to_numpy() == year
            years[int(year)].add(
                scores[mask],
                chunk.loc[mask, "trade_date"].to_numpy(),
                rng,
            )
    if start is None or end is None:
        raise ValueError("OOF source is empty")
    annual = {str(year): years[year].summary() for year in sorted(years)}
    return _base_summary(
        model_family=model_family,
        start=start,
        end=end,
        observations=observations,
        annual=annual,
    )


def generate_synthetic_oof(
    summary: dict,
    *,
    dates_per_year: int = 20,
    assets: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate fake identifiers and scores matching annual first moments."""

    if dates_per_year < 1 or assets < 2:
        raise ValueError("dates_per_year and assets must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for year_text, stats in sorted(summary["annual"].items()):
        year = int(year_text)
        dates = pd.bdate_range(f"{year}-01-05", periods=dates_per_year)
        lower = stats["score_q01"]
        upper = stats["score_q99"]
        for fold, date in enumerate(dates):
            scores = rng.normal(stats["score_mean"], max(stats["score_std"], 1e-8), assets)
            scores = np.clip(scores, lower, upper)
            for asset_index, score in enumerate(scores, start=1):
                rows.append(
                    {
                        "date": date,
                        "symbol": f"DEMO{asset_index:04d}",
                        "score": float(score),
                        "fold": fold % 5,
                        "synthetic": True,
                    }
                )
    return pd.DataFrame(rows)
