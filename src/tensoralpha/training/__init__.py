"""Purged time-series splitting, datasets, training, and OOF assembly."""

from tensoralpha.training.dataset import PanelSequenceDataset, SequenceMetadata
from tensoralpha.training.oof import OOFValidationError, assemble_oof
from tensoralpha.training.split import PurgedWalkForwardSplit
from tensoralpha.training.trainer import EpochMetrics, ModelTrainer, TrainingConfig
from tensoralpha.training.workflow import run_purged_oof

__all__ = [
    "EpochMetrics",
    "ModelTrainer",
    "OOFValidationError",
    "PanelSequenceDataset",
    "PurgedWalkForwardSplit",
    "SequenceMetadata",
    "TrainingConfig",
    "assemble_oof",
    "run_purged_oof",
]
