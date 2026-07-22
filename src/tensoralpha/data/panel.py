"""Canonical long-form market panel and Parquet persistence."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

REQUIRED_MARKET_COLUMNS = (
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)
_NUMERIC_COLUMNS = REQUIRED_MARKET_COLUMNS[2:]


class DataValidationError(ValueError):
    """Raised when market data violates the public panel contract."""


def validate_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a long-form OHLCV panel.

    The returned frame is sorted by ``(date, symbol)`` and owns its data.
    Extra columns are preserved so providers can attach point-in-time fields.
    """

    missing = [column for column in REQUIRED_MARKET_COLUMNS if column not in panel]
    if missing:
        raise DataValidationError(f"missing required columns: {', '.join(missing)}")

    result = panel.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["symbol"] = result["symbol"].astype(str).str.strip()
    if result["symbol"].eq("").any():
        raise DataValidationError("symbol must not be empty")

    for column in _NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")

    if result.duplicated(["date", "symbol"]).any():
        raise DataValidationError("duplicate (date, symbol) rows are not allowed")
    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise DataValidationError("OHLC prices must be positive")
    if (result[["volume", "amount"]] < 0).any().any():
        raise DataValidationError("volume and amount must be non-negative")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise DataValidationError("high must be the largest OHLC value")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise DataValidationError("low must be the smallest OHLC value")

    return result.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)


class ParquetPanelStore:
    """Atomic storage for a canonical market panel."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()

    def write(self, panel: pd.DataFrame) -> None:
        normalized = validate_panel(panel)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        normalized.to_parquet(temporary, index=False)
        os.replace(temporary, self.path)

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"market panel not found: {self.path}")
        return validate_panel(pd.read_parquet(self.path))
