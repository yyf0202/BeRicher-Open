"""Optional Tushare adapter.

Tushare volume is reported in lots and amount in thousands of CNY; this
adapter converts both to shares and CNY respectively.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tensoralpha.data import validate_panel


class TushareDailySource:
    def __init__(self, token: str, client: Any | None = None):
        if not token or not token.strip():
            raise ValueError("Tushare token must be supplied explicitly")
        self._token = token.strip()
        if client is None:
            try:
                import tushare as ts
            except ImportError as error:
                raise RuntimeError(
                    "Install the data extra with: pip install 'tensoralpha[data]'"
                ) from error
            client = ts.pro_api(self._token)
        self._client = client

    def fetch_daily(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = pd.Timestamp(start_date).strftime("%Y%m%d")
        end = pd.Timestamp(end_date).strftime("%Y%m%d")
        frame = self._client.daily(start_date=start, end_date=end)
        if frame is None or frame.empty:
            raise RuntimeError(f"Tushare returned no daily rows for {start}..{end}")
        result = frame.rename(
            columns={
                "trade_date": "date",
                "ts_code": "symbol",
                "vol": "volume",
            }
        ).copy()
        result["date"] = pd.to_datetime(result["date"], format="%Y%m%d")
        result["volume"] = pd.to_numeric(result["volume"]) * 100.0
        result["amount"] = pd.to_numeric(result["amount"]) * 1000.0
        return validate_panel(result)
