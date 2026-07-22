import pandas as pd


def _day(
    date: str,
    symbol: str,
    price: float,
    *,
    suspended: bool = False,
    limit_up: bool = False,
    limit_down: bool = False,
) -> dict:
    return {
        "date": pd.Timestamp(date),
        "symbol": symbol,
        "open": price,
        "high": price * 1.01,
        "low": price * 0.99,
        "close": price,
        "volume": 1_000_000.0,
        "amount": price * 1_000_000.0,
        "suspended": suspended,
        "limit_up": limit_up,
        "limit_down": limit_down,
    }


def test_broker_applies_lot_size_commission_and_stamp_tax():
    from tensoralpha.backtest import BrokerConfig, SimulatedBroker

    broker = SimulatedBroker(
        BrokerConfig(
            lot_size=100,
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_tax_rate=0.0005,
            slippage=0.0,
        )
    )

    buy = broker.fill_order("DEMO0001", "buy", 12.0, 155, pd.Timestamp("2024-01-02"))
    sell = broker.fill_order("DEMO0001", "sell", 12.0, 100, pd.Timestamp("2024-01-03"))

    assert buy.quantity == 100
    assert buy.fee == 5.0
    assert buy.cash_change == -1205.0
    assert sell.quantity == 100
    assert sell.fee == 5.6
    assert sell.cash_change == 1194.4


def test_broker_blocks_suspended_and_price_limit_orders():
    from tensoralpha.backtest import BrokerConfig, OrderRejected, SimulatedBroker

    broker = SimulatedBroker(BrokerConfig())

    cases = [
        ("buy", {"suspended": True}, "suspended"),
        ("sell", {"suspended": True}, "suspended"),
        ("buy", {"limit_up": True}, "limit-up"),
        ("sell", {"limit_down": True}, "limit-down"),
    ]
    for side, flags, expected in cases:
        try:
            broker.fill_order(
                "DEMO0001",
                side,
                10.0,
                100,
                pd.Timestamp("2024-01-02"),
                **flags,
            )
        except OrderRejected as error:
            assert expected in str(error)
        else:
            raise AssertionError(f"{side} order with {flags} was accepted")


def test_backtest_uses_previous_day_signal_and_respects_t_plus_one():
    from tensoralpha.backtest import BacktestConfig, BacktestEngine

    panel = pd.DataFrame(
        [
            _day("2024-01-02", "DEMO0001", 10.0),
            _day("2024-01-02", "DEMO0002", 20.0),
            _day("2024-01-03", "DEMO0001", 11.0),
            _day("2024-01-03", "DEMO0002", 19.0),
            _day("2024-01-04", "DEMO0001", 12.0),
            _day("2024-01-04", "DEMO0002", 18.0),
        ]
    )
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "symbol": ["DEMO0001", "DEMO0002", "DEMO0001", "DEMO0002"],
            "score": [1.0, 0.0, 0.0, 1.0],
        }
    )
    engine = BacktestEngine(
        BacktestConfig(
            initial_cash=100_000.0,
            top_n=1,
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            slippage=0.0,
        )
    )

    result = engine.run(panel, signals)

    assert list(result.trades["date"]) == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-04"),
    ]
    assert list(result.trades["side"]) == ["buy", "sell", "buy"]
    assert result.trades.iloc[0]["symbol"] == "DEMO0001"
    assert result.trades.iloc[-1]["symbol"] == "DEMO0002"
    assert result.nav.iloc[0]["cash"] == 100_000.0
    assert result.nav.iloc[-1]["nav"] > 100_000.0


def test_top_n_strategy_returns_normalized_target_weights():
    from tensoralpha.strategy import TopNRotationStrategy

    strategy = TopNRotationStrategy(top_n=2, max_weight=0.6)
    signals = pd.DataFrame({"symbol": ["C", "A", "B"], "score": [0.1, 0.9, 0.8]})

    weights = strategy.target_weights(signals)

    assert weights == {"A": 0.5, "B": 0.5}
