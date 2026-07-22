import pandas as pd


def _private_like_oof() -> pd.DataFrame:
    rows = []
    for year, offset in [(2015, -0.2), (2016, 0.3)]:
        for date in pd.bdate_range(f"{year}-01-05", periods=4):
            for asset in range(5):
                rows.append(
                    {
                        "trade_date": int(date.strftime("%Y%m%d")),
                        "symbol": f"REAL{asset}",
                        "alpha_score": offset + asset / 10,
                    }
                )
    return pd.DataFrame(rows)


def test_oof_showcase_contains_only_aggregate_statistics():
    from tensoralpha.evaluation import derive_oof_showcase

    summary = derive_oof_showcase(_private_like_oof(), model_family="transformer-test")

    assert summary["privacy"]["contains_security_identifiers"] is False
    assert summary["coverage"]["start"] == "2015-01-05"
    assert summary["coverage"]["end"] == "2016-01-08"
    assert summary["coverage"]["observations"] == 40
    assert set(summary["annual"]) == {"2015", "2016"}
    assert "symbols" not in str(summary).lower()


def test_synthetic_oof_is_deterministic_and_does_not_reuse_source_symbols():
    from tensoralpha.evaluation import derive_oof_showcase, generate_synthetic_oof

    summary = derive_oof_showcase(_private_like_oof(), model_family="transformer-test")
    first = generate_synthetic_oof(summary, dates_per_year=3, assets=8, seed=17)
    second = generate_synthetic_oof(summary, dates_per_year=3, assets=8, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert first["symbol"].str.fullmatch(r"DEMO\d{4}").all()
    assert first["synthetic"].all()
    assert first.groupby(first["date"].dt.year).size().to_dict() == {2015: 24, 2016: 24}
