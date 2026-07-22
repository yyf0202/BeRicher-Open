from pathlib import Path

import pandas as pd
import pytest


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
            "symbol": ["DEMO0002", "DEMO0002", "DEMO0001"],
            "open": [11.0, 10.0, 20.0],
            "high": [11.5, 10.5, 21.0],
            "low": [10.5, 9.5, 19.0],
            "close": [11.2, 10.2, 20.5],
            "volume": [1200.0, 1000.0, 800.0],
            "amount": [13440.0, 10200.0, 16400.0],
        }
    )


def test_validate_panel_sorts_rows_and_normalizes_types():
    from tensoralpha.data import validate_panel

    result = validate_panel(_panel())

    assert list(result[["date", "symbol"]].itertuples(index=False, name=None)) == [
        (pd.Timestamp("2024-01-02"), "DEMO0001"),
        (pd.Timestamp("2024-01-02"), "DEMO0002"),
        (pd.Timestamp("2024-01-03"), "DEMO0002"),
    ]
    assert result["close"].dtype == "float64"


def test_validate_panel_rejects_duplicate_date_symbol_rows():
    from tensoralpha.data import DataValidationError, validate_panel

    duplicated = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicate"):
        validate_panel(duplicated)


def test_parquet_store_round_trip(tmp_path: Path):
    from tensoralpha.data import ParquetPanelStore, validate_panel

    store = ParquetPanelStore(tmp_path / "market.parquet")
    expected = validate_panel(_panel())
    store.write(expected)

    actual = store.read()

    pd.testing.assert_frame_equal(actual, expected)
