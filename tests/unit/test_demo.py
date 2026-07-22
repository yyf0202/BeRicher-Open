import pandas as pd


def test_demo_panel_is_deterministic_and_uses_only_synthetic_symbols():
    from tensoralpha.demo import generate_demo_panel

    first = generate_demo_panel(days=40, assets=12, seed=42)
    second = generate_demo_panel(days=40, assets=12, seed=42)

    pd.testing.assert_frame_equal(first.market, second.market)
    pd.testing.assert_frame_equal(first.signals, second.signals)
    assert first.market["symbol"].str.fullmatch(r"DEMO\d{4}").all()
    assert first.market["date"].nunique() == 40
    assert first.market["symbol"].nunique() == 12
    assert set(["date", "symbol", "score"]).issubset(first.signals.columns)


def test_demo_can_run_through_backtest_without_network_or_credentials():
    from tensoralpha.backtest import BacktestConfig, BacktestEngine
    from tensoralpha.demo import generate_demo_panel

    demo = generate_demo_panel(days=30, assets=8, seed=7)
    result = BacktestEngine(BacktestConfig(initial_cash=100_000.0, top_n=3)).run(
        demo.market, demo.signals
    )

    assert len(result.nav) == 30
    assert not result.trades.empty
    assert result.nav["nav"].gt(0).all()


def test_prior_day_signal_predicts_next_day_open_to_close_return():
    from tensoralpha.demo import generate_demo_panel

    demo = generate_demo_panel(days=520, assets=40, seed=42)
    trading_dates = demo.market["date"].drop_duplicates().sort_values().tolist()
    next_trading_date = dict(zip(trading_dates[:-1], trading_dates[1:], strict=True))

    shifted_signals = demo.signals.copy()
    shifted_signals["date"] = shifted_signals["date"].map(next_trading_date)
    shifted_signals = shifted_signals.dropna(subset=["date"])
    aligned = demo.market.merge(shifted_signals, on=["date", "symbol"], validate="one_to_one")
    aligned["open_to_close_return"] = aligned["close"] / aligned["open"] - 1.0

    daily_rank_ic = pd.Series(
        [
            frame["score"].corr(frame["open_to_close_return"], method="spearman")
            for _, frame in aligned.groupby("date", sort=True)
        ]
    )

    assert 0.05 < float(daily_rank_ic.mean()) < 0.30
