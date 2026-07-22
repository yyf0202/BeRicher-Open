"""Order validation, A-share lot sizing, fees, and slippage."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class OrderRejected(RuntimeError):
    """Raised when market rules prevent an order from filling."""


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_tax_rate: float = 0.0005
    slippage: float = 0.002
    lot_size: int = 100


@dataclass(frozen=True, slots=True)
class Fill:
    date: pd.Timestamp
    symbol: str
    side: str
    quantity: int
    price: float
    fee: float
    cash_change: float


class SimulatedBroker:
    def __init__(self, config: BrokerConfig | None = None):
        self.config = config or BrokerConfig()

    def fill_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: int,
        date: pd.Timestamp,
        *,
        suspended: bool = False,
        limit_up: bool = False,
        limit_down: bool = False,
    ) -> Fill:
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if suspended:
            raise OrderRejected(f"{symbol} is suspended")
        if side == "buy" and limit_up:
            raise OrderRejected(f"{symbol} is limit-up")
        if side == "sell" and limit_down:
            raise OrderRejected(f"{symbol} is limit-down")
        if price <= 0:
            raise OrderRejected(f"{symbol} has an invalid price")

        lots = int(quantity) // self.config.lot_size
        filled_quantity = lots * self.config.lot_size
        if filled_quantity <= 0:
            raise OrderRejected("quantity is smaller than one trading lot")

        direction = 1.0 if side == "buy" else -1.0
        fill_price = float(price) * (1.0 + direction * self.config.slippage)
        notional = fill_price * filled_quantity
        commission = max(notional * self.config.commission_rate, self.config.min_commission)
        stamp_tax = notional * self.config.stamp_tax_rate if side == "sell" else 0.0
        fee = round(commission + stamp_tax, 10)
        cash_change = -notional - fee if side == "buy" else notional - fee
        return Fill(
            date=pd.Timestamp(date).normalize(),
            symbol=str(symbol),
            side=side,
            quantity=filled_quantity,
            price=fill_price,
            fee=fee,
            cash_change=round(cash_change, 10),
        )
