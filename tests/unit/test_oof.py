import numpy as np
import pandas as pd
import pytest


def test_purged_walk_forward_split_keeps_validation_after_training_with_gap():
    from tensoralpha.training import PurgedWalkForwardSplit

    dates = pd.bdate_range("2020-01-01", periods=240)
    splitter = PurgedWalkForwardSplit(n_splits=3, purge_days=5, min_train_days=80)
    folds = list(splitter.split(dates))

    assert len(folds) == 3
    for train_idx, validation_idx in folds:
        assert train_idx.max() + 5 < validation_idx.min()
        assert set(train_idx).isdisjoint(validation_idx)


def test_assemble_oof_rejects_duplicate_predictions():
    from tensoralpha.training import OOFValidationError, assemble_oof

    fold = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "symbol": ["DEMO0001"],
            "score": [0.2],
            "fold": [0],
        }
    )

    with pytest.raises(OOFValidationError, match="duplicate"):
        assemble_oof([fold, fold.assign(fold=1)])


def test_assemble_oof_sorts_and_rank_normalizes_each_day():
    from tensoralpha.training import assemble_oof

    fold_a = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
            "symbol": ["DEMO0002", "DEMO0001"],
            "score": [4.0, 2.0],
            "fold": [0, 0],
        }
    )
    fold_b = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04", "2024-01-04"]),
            "symbol": ["DEMO0001", "DEMO0002"],
            "score": [1.0, 3.0],
            "fold": [1, 1],
        }
    )

    result = assemble_oof([fold_b, fold_a])

    assert list(result["date"].unique()) == [
        np.datetime64("2024-01-03"),
        np.datetime64("2024-01-04"),
    ]
    assert result.groupby("date")["rank"].min().eq(0.5).all()
    assert result.groupby("date")["rank"].max().eq(1.0).all()
