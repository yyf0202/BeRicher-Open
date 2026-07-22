from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _featured_panel(days: int = 18) -> pd.DataFrame:
    rows = []
    for symbol_index, symbol in enumerate(["DEMO0001", "DEMO0002"]):
        for day_index, date in enumerate(pd.bdate_range("2024-01-02", periods=days)):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "feature_a": float(day_index),
                    "feature_b": float(symbol_index),
                    "target_rank": float((day_index + symbol_index) % 2),
                }
            )
    return pd.DataFrame(rows)


def test_sequence_dataset_never_crosses_symbol_boundaries():
    from tensoralpha.training import PanelSequenceDataset

    dataset = PanelSequenceDataset(
        _featured_panel(),
        feature_names=["feature_a", "feature_b"],
        target_name="target_rank",
        sequence_length=5,
    )

    features, target, metadata = dataset[0]

    assert features.shape == (5, 2)
    assert target.ndim == 0
    assert metadata.symbol == "DEMO0001"
    assert torch.equal(features[:, 1], torch.zeros(5))


def test_trainer_fits_and_predicts_cpu_sequences():
    from tensoralpha.models import TransformerConfig, TransformerRanker
    from tensoralpha.training import ModelTrainer, PanelSequenceDataset, TrainingConfig

    torch.manual_seed(7)
    dataset = PanelSequenceDataset(
        _featured_panel(),
        feature_names=["feature_a", "feature_b"],
        target_name="target_rank",
        sequence_length=5,
    )
    model = TransformerRanker(
        TransformerConfig(
            input_dim=2,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        )
    )
    trainer = ModelTrainer(model, TrainingConfig(epochs=1, batch_size=8, device="cpu"))

    history = trainer.fit(dataset)
    predictions = trainer.predict(dataset)

    assert len(history) == 1
    assert np.isfinite(history[0].loss)
    assert predictions.shape == (len(dataset),)
    assert np.isfinite(predictions).all()


def test_model_artifact_round_trip_preserves_predictions(tmp_path: Path):
    from tensoralpha.inference import ModelArtifact
    from tensoralpha.models import TransformerConfig, TransformerRanker

    torch.manual_seed(11)
    model = TransformerRanker(
        TransformerConfig(
            input_dim=2,
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        )
    ).eval()
    features = torch.randn(4, 6, 2)
    expected = model(features).detach()

    artifact = ModelArtifact.save(
        tmp_path / "model",
        model,
        feature_names=["feature_a", "feature_b"],
        sequence_length=6,
    )
    restored, metadata = artifact.load()

    torch.testing.assert_close(restored(features), expected)
    assert metadata.feature_names == ["feature_a", "feature_b"]
    assert metadata.sequence_length == 6
