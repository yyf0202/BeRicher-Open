import numpy as np
import pandas as pd


def _training_panel(days: int = 36, assets: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    dates = pd.bdate_range("2021-01-04", periods=days)
    for symbol_index in range(assets):
        symbol = f"DEMO{symbol_index + 1:04d}"
        for day_index, date in enumerate(dates):
            feature = rng.normal() + symbol_index * 0.2
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "feature_a": feature,
                    "feature_b": day_index / days,
                    "target_rank": (symbol_index + 1) / assets,
                }
            )
    return pd.DataFrame(rows)


def test_run_purged_oof_trains_each_fold_and_returns_unique_predictions(tmp_path):
    from tensoralpha.models import TransformerConfig
    from tensoralpha.training import (
        PurgedWalkForwardSplit,
        TrainingConfig,
        run_purged_oof,
    )

    result = run_purged_oof(
        _training_panel(),
        feature_names=["feature_a", "feature_b"],
        target_name="target_rank",
        sequence_length=4,
        splitter=PurgedWalkForwardSplit(n_splits=2, purge_days=2, min_train_days=16),
        model_config=TransformerConfig(
            input_dim=2,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        ),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=16,
            learning_rate=1e-3,
            device="cpu",
            seed=3,
        ),
        artifact_dir=tmp_path / "folds",
    )

    assert not result.empty
    assert not result.duplicated(["date", "symbol"]).any()
    assert set(result["fold"]) == {0, 1}
    assert result["rank"].between(0, 1).all()
    assert (tmp_path / "folds" / "fold_0" / "metadata.json").exists()
    assert (tmp_path / "folds" / "fold_1" / "weights.pt").exists()
