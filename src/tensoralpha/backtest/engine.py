"""Event loop that executes yesterday's target at today's open."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tensoralpha.backtest.account import PortfolioState, rebalance_to_weights
from tensoralpha.backtest.broker import BrokerConfig, SimulatedBroker
from tensoralpha.data import validate_panel
from tensoralpha.strategy import TopNRotationStrategy


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    top_n: int = 10
    max_weight: float = 1.0
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.002
    lot_size: int = 100


@dataclass(frozen=True, slots=True)
class BacktestResult:
    nav: pd.DataFrame
    trades: pd.DataFrame
    final_positions: dict[str, int]


class BacktestEngine:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()
        self.strategy = TopNRotationStrategy(
            top_n=self.config.top_n, max_weight=self.config.max_weight
        )
        self.broker = SimulatedBroker(
            BrokerConfig(
                commission_rate=self.config.commission_rate,
                min_commission=self.config.min_commission,
                stamp_tax_rate=self.config.stamp_tax_rate,
                slippage=self.config.slippage,
                lot_size=self.config.lot_size,
            )
        )

    def run(self, panel: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        market = validate_panel(panel)
        required = {"date", "symbol", "score"}
        missing = required.difference(signals.columns)
        if missing:
            raise ValueError(f"signals missing columns: {sorted(missing)}")
        signal_panel = signals.copy()
        signal_panel["date"] = pd.to_datetime(signal_panel["date"]).dt.normalize()
        signal_panel["symbol"] = signal_panel["symbol"].astype(str)
        if signal_panel.duplicated(["date", "symbol"]).any():
            raise ValueError("signals contain duplicate (date, symbol) rows")

        state = PortfolioState(cash=float(self.config.initial_cash))
        pending_weights: dict[str, float] = {}
        nav_rows: list[dict] = []
        trade_rows: list[dict] = []

        for date, day_market in market.groupby("date", sort=True):
            fills = rebalance_to_weights(
                state, pending_weights, day_market, pd.Timestamp(date), self.broker
            )
            trade_rows.extend(
                {
                    "date": fill.date,
                    "symbol": fill.symbol,
                    "side": fill.side,
                    "quantity": fill.quantity,
                    "price": fill.price,
                    "fee": fill.fee,
                    "cash_change": fill.cash_change,
                }
                for fill in fills
            )
            closing_nav = state.nav(day_market, "close")
            nav_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "nav": closing_nav,
                    "cash": state.cash,
                    "positions": len(state.positions),
                }
            )
            day_signals = signal_panel[signal_panel["date"] == pd.Timestamp(date)]
            pending_weights = self.strategy.target_weights(day_signals)

        trade_columns = [
            "date",
            "symbol",
            "side",
            "quantity",
            "price",
            "fee",
            "cash_change",
        ]
        return BacktestResult(
            nav=pd.DataFrame(nav_rows),
            trades=pd.DataFrame(trade_rows, columns=trade_columns),
            final_positions={symbol: pos.quantity for symbol, pos in state.positions.items()},
        )
