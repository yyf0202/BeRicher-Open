"""Small, deterministic CPU/GPU training loop for the public model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tensoralpha.training.dataset import PanelSequenceDataset


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    device: str = "cpu"
    seed: int = 42


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    loss: float


def _collate(batch):
    features, targets, metadata = zip(*batch, strict=False)
    return torch.stack(features), torch.stack(targets), metadata


class ModelTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig | None = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        self.model.to(self.device)

    def fit(self, dataset: PanelSequenceDataset) -> list[EpochMetrics]:
        torch.manual_seed(self.config.seed)
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=_collate,
            generator=generator,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.MSELoss()
        history: list[EpochMetrics] = []
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            weighted_loss = 0.0
            sample_count = 0
            for features, targets, _ in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(features)
                loss = loss_function(predictions, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                weighted_loss += float(loss.detach()) * len(targets)
                sample_count += len(targets)
            history.append(EpochMetrics(epoch, weighted_loss / sample_count))
        return history

    def predict(self, dataset: PanelSequenceDataset) -> np.ndarray:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )
        self.model.eval()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for features, _, _ in loader:
                values = self.model(features.to(self.device)).cpu().numpy()
                outputs.append(values)
        return np.concatenate(outputs)
