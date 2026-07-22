"""Portfolio state and shared target-weight execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from tensoralpha.backtest.broker import Fill, OrderRejected, SimulatedBroker


@dataclass(slots=True)
class Position:
    quantity: int
    acquired_date: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(slots=True)
class PortfolioState:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def nav(self, market: pd.DataFrame, price_column: str = "close") -> float:
        prices = market.set_index("symbol")[price_column].to_dict()
        value = self.cash
        for symbol, position in self.positions.items():
            if symbol in prices:
                value += position.quantity * float(prices[symbol])
        return float(value)


def _market_flags(row: pd.Series) -> dict[str, bool]:
    return {
        "suspended": bool(row.get("suspended", False)),
        "limit_up": bool(row.get("limit_up", False)),
        "limit_down": bool(row.get("limit_down", False)),
    }


def rebalance_to_weights(
    state: PortfolioState,
    target_weights: dict[str, float],
    market: pd.DataFrame,
    date: pd.Timestamp,
    broker: SimulatedBroker,
) -> list[Fill]:
    """Rebalance at the day's open, selling before buying.

    A position acquired on ``date`` is not sellable until a later date.
    Rejected orders are left unfilled and the portfolio remains valid.
    """

    day = pd.Timestamp(date).normalize()
    rows = market.copy()
    rows["symbol"] = rows["symbol"].astype(str)
    by_symbol = {row["symbol"]: row for _, row in rows.iterrows()}
    opening_nav = state.nav(rows, "open")
    fills: list[Fill] = []

    desired_quantities: dict[str, int] = {}
    for symbol, weight in target_weights.items():
        row = by_symbol.get(symbol)
        if row is None or float(row["open"]) <= 0:
            continue
        desired_quantities[symbol] = int(opening_nav * float(weight) / float(row["open"]))

    for symbol in sorted(list(state.positions)):
        position = state.positions[symbol]
        desired = desired_quantities.get(symbol, 0)
        sell_quantity = max(0, position.quantity - desired)
        if sell_quantity == 0 or pd.Timestamp(position.acquired_date) >= day:
            continue
        row = by_symbol.get(symbol)
        if row is None:
            continue
        try:
            fill = broker.fill_order(
                symbol, "sell", float(row["open"]), sell_quantity, day, **_market_flags(row)
            )
        except OrderRejected:
            continue
        position.quantity -= fill.quantity
        state.cash += fill.cash_change
        fills.append(fill)
        if position.quantity == 0:
            del state.positions[symbol]

    for symbol in sorted(target_weights):
        row = by_symbol.get(symbol)
        if row is None:
            continue
        current = state.positions.get(symbol)
        current_quantity = current.quantity if current else 0
        buy_quantity = max(0, desired_quantities.get(symbol, 0) - current_quantity)
        lot_size = broker.config.lot_size
        buy_quantity = buy_quantity // lot_size * lot_size
        while buy_quantity > 0:
            try:
                fill = broker.fill_order(
                    symbol, "buy", float(row["open"]), buy_quantity, day, **_market_flags(row)
                )
            except OrderRejected:
                break
            if -fill.cash_change <= state.cash + 1e-9:
                state.cash += fill.cash_change
                if current:
                    current.quantity += fill.quantity
                else:
                    state.positions[symbol] = Position(fill.quantity, day.strftime("%Y-%m-%d"))
                fills.append(fill)
                break
            buy_quantity -= lot_size

    return fills
