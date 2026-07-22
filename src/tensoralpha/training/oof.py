"""Validation and assembly of fold predictions."""

from __future__ import annotations

import pandas as pd


class OOFValidationError(ValueError):
    """Raised when fold outputs cannot form a clean OOF panel."""


def assemble_oof(folds: list[pd.DataFrame]) -> pd.DataFrame:
    required = {"date", "symbol", "score", "fold"}
    if not folds:
        raise OOFValidationError("at least one fold is required")
    for index, fold in enumerate(folds):
        missing = required.difference(fold.columns)
        if missing:
            raise OOFValidationError(f"fold {index} missing columns: {sorted(missing)}")

    result = pd.concat(folds, ignore_index=True).copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    if result.duplicated(["date", "symbol"]).any():
        raise OOFValidationError("duplicate OOF predictions for (date, symbol)")
    if result["score"].isna().any():
        raise OOFValidationError("OOF score contains missing values")
    result["rank"] = result.groupby("date")["score"].rank(pct=True)
    return result.sort_values(["date", "symbol"], kind="stable").reset_index(drop=True)
