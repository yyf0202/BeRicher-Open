import numpy as np
import pandas as pd


def test_rank_ic_and_annual_summary_are_cross_sectional():
    from tensoralpha.evaluation import rank_ic_by_date, summarize_oof

    oof = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-02", "2024-01-02", "2024-01-02"]),
            "symbol": ["A", "B", "A", "B"],
            "score": [1.0, 0.0, 0.0, 1.0],
            "forward_return": [0.02, -0.01, -0.03, 0.01],
        }
    )

    daily = rank_ic_by_date(oof)
    summary = summarize_oof(oof)

    assert np.allclose(daily["rank_ic"], [1.0, 1.0])
    assert summary["overall"]["days"] == 2
    assert set(summary["annual"]) == {"2023", "2024"}
    assert summary["annual"]["2023"]["observations"] == 2
