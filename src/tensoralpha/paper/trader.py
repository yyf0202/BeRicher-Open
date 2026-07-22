"""Atomic paper-account state and daily target execution."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from tensoralpha.backtest.account import PortfolioState, Position, rebalance_to_weights
from tensoralpha.backtest.broker import BrokerConfig, SimulatedBroker
from tensoralpha.data import validate_panel
from tensoralpha.strategy import TopNRotationStrategy


@dataclass(slots=True)
class PaperState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    pending_weights: dict[str, float] = field(default_factory=dict)
    last_date: str | None = None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class PaperTrader:
    def __init__(
        self,
        directory: Path,
        state: PaperState,
        strategy: TopNRotationStrategy,
        broker_config: BrokerConfig,
    ):
        self.directory = directory
        self.state = state
        self.strategy = strategy
        self.broker_config = broker_config
        self.broker = SimulatedBroker(broker_config)

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        initial_cash: float,
        strategy: TopNRotationStrategy,
        broker_config: BrokerConfig | None = None,
    ) -> PaperTrader:
        target = Path(directory).expanduser().resolve()
        if (target / "state.json").exists():
            raise FileExistsError(f"paper account already exists: {target}")
        config = broker_config or BrokerConfig()
        trader = cls(target, PaperState(float(initial_cash)), strategy, config)
        _atomic_json(
            target / "config.json",
            {
                "initial_cash": float(initial_cash),
                "strategy": asdict(strategy),
                "broker": asdict(config),
            },
        )
        trader.save()
        return trader

    @classmethod
    def load(cls, directory: str | Path) -> PaperTrader:
        target = Path(directory).expanduser().resolve()
        config = json.loads((target / "config.json").read_text(encoding="utf-8"))
        raw = json.loads((target / "state.json").read_text(encoding="utf-8"))
        state = PaperState(
            cash=float(raw["cash"]),
            positions={
                symbol: Position(int(value["quantity"]), str(value["acquired_date"]))
                for symbol, value in raw.get("positions", {}).items()
            },
            pending_weights={
                str(symbol): float(weight)
                for symbol, weight in raw.get("pending_weights", {}).items()
            },
            last_date=raw.get("last_date"),
        )
        return cls(
            target,
            state,
            TopNRotationStrategy(**config["strategy"]),
            BrokerConfig(**config["broker"]),
        )

    def save(self) -> None:
        _atomic_json(
            self.directory / "state.json",
            {
                "cash": self.state.cash,
                "positions": {
                    symbol: position.to_dict()
                    for symbol, position in sorted(self.state.positions.items())
                },
                "pending_weights": self.state.pending_weights,
                "last_date": self.state.last_date,
            },
        )

    def tick(self, market: pd.DataFrame, signals: pd.DataFrame) -> list:
        day_market = validate_panel(market)
        dates = day_market["date"].drop_duplicates()
        if len(dates) != 1:
            raise ValueError("paper tick requires exactly one market date")
        date = pd.Timestamp(dates.iloc[0])
        if self.state.last_date and date <= pd.Timestamp(self.state.last_date):
            raise ValueError("paper ticks must be strictly increasing by date")

        portfolio = PortfolioState(self.state.cash, self.state.positions)
        fills = rebalance_to_weights(
            portfolio, self.state.pending_weights, day_market, date, self.broker
        )
        self.state.cash = portfolio.cash
        self.state.positions = portfolio.positions
        self.state.pending_weights = self.strategy.target_weights(signals)
        self.state.last_date = date.strftime("%Y-%m-%d")
        self.save()
        return fills
