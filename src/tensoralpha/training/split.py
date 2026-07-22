"""Deterministic expanding-window splits with a purge gap."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PurgedWalkForwardSplit:
    n_splits: int = 5
    purge_days: int = 20
    min_train_days: int = 252

    def split(self, dates: pd.Index | list) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        unique_dates = pd.Index(pd.to_datetime(dates)).drop_duplicates().sort_values()
        if self.n_splits < 1 or self.purge_days < 0 or self.min_train_days < 1:
            raise ValueError("invalid split configuration")
        remaining = len(unique_dates) - self.min_train_days - self.purge_days
        if remaining < self.n_splits:
            raise ValueError("not enough dates for requested splits")

        fold_size = remaining // self.n_splits
        for fold in range(self.n_splits):
            validation_start = self.min_train_days + self.purge_days + fold * fold_size
            validation_end = (
                len(unique_dates) if fold == self.n_splits - 1 else validation_start + fold_size
            )
            train_end = validation_start - self.purge_days
            yield (
                np.arange(0, train_end, dtype=np.int64),
                np.arange(validation_start, validation_end, dtype=np.int64),
            )
