import numpy as np
import pandas as pd


def _price_history(days: int = 90) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows = []
    for asset_id, base, slope in [("DEMO0001", 10.0, 0.03), ("DEMO0002", 20.0, -0.01)]:
        for i, date in enumerate(dates):
            close = base + slope * i + np.sin(i / 5) * 0.1
            rows.append(
                {
                    "date": date,
                    "symbol": asset_id,
                    "open": close * 0.998,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000 + i * 100,
                    "amount": close * (1_000_000 + i * 100),
                }
            )
    return pd.DataFrame(rows)


def test_feature_pipeline_produces_finite_past_only_features():
    from tensoralpha.features import FeaturePipeline

    panel = _price_history()
    pipeline = FeaturePipeline()
    result = pipeline.transform(panel)

    assert set(pipeline.feature_names).issubset(result.columns)
    assert np.isfinite(result[pipeline.feature_names].to_numpy()).all()

    changed_future = panel.copy()
    last_day = changed_future["date"].max()
    future_rows = changed_future["date"] == last_day
    changed_future.loc[future_rows, ["open", "high", "low", "close"]] *= 100
    changed_future.loc[future_rows, "amount"] *= 100
    before = pipeline.transform(panel)
    after = pipeline.transform(changed_future)
    historical = before["date"] < last_day
    pd.testing.assert_frame_equal(
        before.loc[historical, ["date", "symbol", *pipeline.feature_names]].reset_index(drop=True),
        after.loc[historical, ["date", "symbol", *pipeline.feature_names]].reset_index(drop=True),
    )


def test_forward_rank_target_uses_future_return_within_each_symbol():
    from tensoralpha.features import add_forward_rank_target

    panel = _price_history(days=10)
    result = add_forward_rank_target(panel, horizon=2)

    first = result[result["symbol"] == "DEMO0001"].sort_values("date").iloc[0]
    closes = panel[panel["symbol"] == "DEMO0001"].sort_values("date")["close"].to_numpy()
    expected_return = closes[2] / closes[0] - 1
    assert np.isclose(first["forward_return"], expected_return)
    assert (
        result.groupby("date")["target_rank"]
        .apply(lambda values: values.dropna().between(0, 1).all())
        .all()
    )
