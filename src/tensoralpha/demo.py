"""Deterministic synthetic market used by examples and CI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tensoralpha.data import validate_panel


@dataclass(frozen=True, slots=True)
class DemoPanel:
    market: pd.DataFrame
    signals: pd.DataFrame


def generate_demo_panel(days: int = 260, assets: int = 40, seed: int = 42) -> DemoPanel:
    """Create an entirely synthetic panel with a weak, time-varying alpha signal."""

    if days < 3 or assets < 2:
        raise ValueError("demo requires at least 3 days and 2 assets")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-05", periods=days)
    symbols = [f"DEMO{index:04d}" for index in range(1, assets + 1)]
    market_factor = rng.normal(0.0002, 0.008, size=days)
    latent = rng.normal(0.0, 1.0, size=(days, assets))
    latent = 0.65 * latent + 0.35 * np.roll(latent, 1, axis=0)
    noise = rng.normal(0.0, 0.012, size=(days, assets))
    intraday_returns = market_factor[:, None] + noise
    intraday_returns[1:] += 0.003 * latent[:-1]
    intraday_returns = np.clip(intraday_returns, -0.095, 0.095)
    starting_prices = rng.uniform(8.0, 60.0, size=assets)

    market_rows: list[dict] = []
    signal_rows: list[dict] = []
    previous_closes = starting_prices
    for day_index, date in enumerate(dates):
        overnight = rng.normal(0.0, 0.002, size=assets)
        opens = previous_closes * (1.0 + overnight)
        closes = opens * (1.0 + intraday_returns[day_index])
        intraday = rng.uniform(0.003, 0.018, size=assets)
        volumes = rng.lognormal(mean=13.3, sigma=0.45, size=assets)
        for asset_index, symbol in enumerate(symbols):
            close = float(closes[asset_index])
            opening = float(opens[asset_index])
            high = max(opening, close) * (1.0 + float(intraday[asset_index]))
            low = min(opening, close) * (1.0 - float(intraday[asset_index]))
            volume = float(volumes[asset_index])
            market_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": opening,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                }
            )
            signal_rows.append(
                {"date": date, "symbol": symbol, "score": float(latent[day_index, asset_index])}
            )
        previous_closes = closes
    return DemoPanel(
        market=validate_panel(pd.DataFrame(market_rows)),
        signals=pd.DataFrame(signal_rows).sort_values(["date", "symbol"]).reset_index(drop=True),
    )
