from pathlib import Path

import pandas as pd


def _market(date: str, price_a: float, price_b: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date, date]),
            "symbol": ["DEMO0001", "DEMO0002"],
            "open": [price_a, price_b],
            "high": [price_a * 1.01, price_b * 1.01],
            "low": [price_a * 0.99, price_b * 0.99],
            "close": [price_a, price_b],
            "volume": [1_000_000.0, 1_000_000.0],
            "amount": [price_a * 1_000_000.0, price_b * 1_000_000.0],
        }
    )


def test_paper_trader_persists_state_and_executes_prior_day_target(tmp_path: Path):
    from tensoralpha.backtest import BrokerConfig
    from tensoralpha.paper import PaperTrader
    from tensoralpha.strategy import TopNRotationStrategy

    trader = PaperTrader.create(
        tmp_path / "demo-paper",
        initial_cash=100_000.0,
        strategy=TopNRotationStrategy(top_n=1),
        broker_config=BrokerConfig(
            commission_rate=0.0,
            min_commission=0.0,
            stamp_tax_rate=0.0,
            slippage=0.0,
        ),
    )
    first_signals = pd.DataFrame({"symbol": ["DEMO0001", "DEMO0002"], "score": [1.0, 0.0]})
    trader.tick(_market("2024-01-02", 10.0, 20.0), first_signals)

    restored = PaperTrader.load(tmp_path / "demo-paper")
    restored.tick(
        _market("2024-01-03", 11.0, 19.0),
        pd.DataFrame({"symbol": ["DEMO0001", "DEMO0002"], "score": [0.0, 1.0]}),
    )

    assert restored.state.last_date == "2024-01-03"
    assert restored.state.positions["DEMO0001"].quantity > 0
    assert restored.state.pending_weights == {"DEMO0002": 1.0}
    assert (tmp_path / "demo-paper" / "state.json").exists()
