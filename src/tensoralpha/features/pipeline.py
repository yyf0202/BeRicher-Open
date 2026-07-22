"""Compact feature set used by the public Transformer pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tensoralpha.data import validate_panel


@dataclass(frozen=True, slots=True)
class FeaturePipeline:
    """Generate normalized price, momentum, volatility, and liquidity features."""

    feature_names: list[str] = field(
        default_factory=lambda: [
            "return_1d",
            "return_5d",
            "return_20d",
            "close_to_ma5",
            "close_to_ma20",
            "volatility_5d",
            "volatility_20d",
            "intraday_range",
            "close_location",
            "volume_ratio_5d",
            "volume_ratio_20d",
            "amount_rank",
        ]
    )

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        result = validate_panel(panel)
        by_symbol = result.groupby("symbol", sort=False, group_keys=False)
        close = result["close"]
        volume = result["volume"]

        result["return_1d"] = by_symbol["close"].pct_change(1)
        result["return_5d"] = by_symbol["close"].pct_change(5)
        result["return_20d"] = by_symbol["close"].pct_change(20)

        ma5 = by_symbol["close"].transform(lambda values: values.rolling(5).mean())
        ma20 = by_symbol["close"].transform(lambda values: values.rolling(20).mean())
        result["close_to_ma5"] = close / ma5 - 1
        result["close_to_ma20"] = close / ma20 - 1

        one_day_return = result["return_1d"]
        return_groups = one_day_return.groupby(result["symbol"], sort=False)
        result["volatility_5d"] = return_groups.transform(lambda values: values.rolling(5).std())
        result["volatility_20d"] = return_groups.transform(lambda values: values.rolling(20).std())
        result["intraday_range"] = (result["high"] - result["low"]) / close
        spread = (result["high"] - result["low"]).replace(0, np.nan)
        result["close_location"] = (close - result["low"]) / spread - 0.5

        volume_ma5 = by_symbol["volume"].transform(lambda values: values.rolling(5).mean())
        volume_ma20 = by_symbol["volume"].transform(lambda values: values.rolling(20).mean())
        result["volume_ratio_5d"] = volume / volume_ma5 - 1
        result["volume_ratio_20d"] = volume / volume_ma20 - 1
        result["amount_rank"] = result.groupby("date")["amount"].rank(pct=True) - 0.5

        result[list(self.feature_names)] = (
            result[list(self.feature_names)]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(-20.0, 20.0)
            .astype("float32")
        )
        return result


def add_forward_rank_target(panel: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Attach a future return and its same-day cross-sectional percentile rank."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    result = validate_panel(panel)
    future_close = result.groupby("symbol", sort=False)["close"].shift(-horizon)
    result["forward_return"] = future_close / result["close"] - 1.0
    result["target_rank"] = result.groupby("date")["forward_return"].rank(pct=True)
    return result
