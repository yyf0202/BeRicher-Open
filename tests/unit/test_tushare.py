import pandas as pd


class _FakeDailyClient:
    def daily(self, **kwargs):
        assert kwargs["start_date"] == "20240102"
        assert kwargs["end_date"] == "20240103"
        return pd.DataFrame(
            {
                "trade_date": ["20240103", "20240102"],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "open": [10.1, 10.0],
                "high": [10.3, 10.2],
                "low": [9.9, 9.8],
                "close": [10.2, 10.1],
                "vol": [1000.0, 900.0],
                "amount": [10200.0, 9090.0],
            }
        )


def test_tushare_source_normalizes_provider_columns_without_network():
    from tensoralpha.data.tushare import TushareDailySource

    source = TushareDailySource(token="not-a-real-token", client=_FakeDailyClient())
    result = source.fetch_daily("2024-01-02", "2024-01-03")

    assert list(result["date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert result["symbol"].eq("000001.SZ").all()
    assert result["volume"].tolist() == [90000.0, 100000.0]
