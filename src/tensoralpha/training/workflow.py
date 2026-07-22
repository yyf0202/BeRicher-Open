"""End-to-end purged K-fold training and OOF prediction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from tensoralpha.inference import ModelArtifact
from tensoralpha.models import TransformerConfig, TransformerRanker
from tensoralpha.training.dataset import PanelSequenceDataset
from tensoralpha.training.oof import assemble_oof
from tensoralpha.training.split import PurgedWalkForwardSplit
from tensoralpha.training.trainer import ModelTrainer, TrainingConfig


def run_purged_oof(
    panel: pd.DataFrame,
    *,
    feature_names: list[str],
    target_name: str,
    sequence_length: int,
    splitter: PurgedWalkForwardSplit,
    model_config: TransformerConfig,
    training_config: TrainingConfig,
    artifact_dir: str | Path,
) -> pd.DataFrame:
    """Train one fresh model per expanding fold and assemble OOF predictions."""

    if model_config.input_dim != len(feature_names):
        raise ValueError("model input_dim must match feature_names")
    working = panel.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    unique_dates = pd.Index(working["date"].unique()).sort_values()
    artifact_root = Path(artifact_dir).expanduser().resolve()
    fold_frames: list[pd.DataFrame] = []

    for fold, (train_indices, validation_indices) in enumerate(splitter.split(unique_dates)):
        train_dates = {pd.Timestamp(value) for value in unique_dates[train_indices]}
        validation_dates = {pd.Timestamp(value) for value in unique_dates[validation_indices]}
        train_dataset = PanelSequenceDataset(
            working,
            feature_names=feature_names,
            target_name=target_name,
            sequence_length=sequence_length,
            target_dates=train_dates,
        )
        validation_dataset = PanelSequenceDataset(
            working,
            feature_names=feature_names,
            target_name=target_name,
            sequence_length=sequence_length,
            target_dates=validation_dates,
        )
        fold_config = replace(training_config, seed=training_config.seed + fold)
        torch.manual_seed(fold_config.seed)
        model = TransformerRanker(model_config)
        trainer = ModelTrainer(model, fold_config)
        trainer.fit(train_dataset)
        scores = trainer.predict(validation_dataset)
        fold_frames.append(
            pd.DataFrame(
                {
                    "date": [item.date for item in validation_dataset.metadata],
                    "symbol": [item.symbol for item in validation_dataset.metadata],
                    "score": scores,
                    "fold": fold,
                }
            )
        )
        ModelArtifact.save(
            artifact_root / f"fold_{fold}",
            model,
            feature_names=feature_names,
            sequence_length=sequence_length,
        )
    return assemble_oof(fold_frames)
