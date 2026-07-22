"""Deterministic Top-N portfolio construction."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class TopNRotationStrategy:
    """Select the highest scores and assign equal target weights."""

    top_n: int = 10
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be positive")
        if not 0 < self.max_weight <= 1:
            raise ValueError("max_weight must be in (0, 1]")

    def target_weights(self, signals: pd.DataFrame) -> dict[str, float]:
        required = {"symbol", "score"}
        missing = required.difference(signals.columns)
        if missing:
            raise ValueError(f"signals missing columns: {sorted(missing)}")
        candidates = signals.dropna(subset=["symbol", "score"]).copy()
        candidates["symbol"] = candidates["symbol"].astype(str)
        candidates = candidates.sort_values(
            ["score", "symbol"], ascending=[False, True], kind="stable"
        ).drop_duplicates("symbol")
        selected = candidates.head(self.top_n)["symbol"].tolist()
        if not selected:
            return {}
        weight = min(1.0 / len(selected), self.max_weight)
        return {symbol: weight for symbol in selected}
