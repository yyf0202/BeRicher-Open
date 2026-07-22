"""Per-security rolling sequences with explicit sample metadata."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class SequenceMetadata:
    date: pd.Timestamp
    symbol: str


class PanelSequenceDataset(Dataset):
    """Build rolling sequences without crossing security boundaries."""

    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        feature_names: list[str],
        target_name: str,
        sequence_length: int,
        target_dates: set[pd.Timestamp] | None = None,
    ):
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        required = {"date", "symbol", target_name, *feature_names}
        missing = required.difference(panel.columns)
        if missing:
            raise ValueError(f"panel missing columns: {sorted(missing)}")
        self.feature_names = list(feature_names)
        self.target_name = target_name
        self.sequence_length = sequence_length
        normalized_dates = (
            {pd.Timestamp(date).normalize() for date in target_dates}
            if target_dates is not None
            else None
        )
        self._samples: list[tuple[np.ndarray, float, SequenceMetadata]] = []

        ordered = panel.copy()
        ordered["date"] = pd.to_datetime(ordered["date"]).dt.normalize()
        ordered["symbol"] = ordered["symbol"].astype(str)
        ordered = ordered.sort_values(["symbol", "date"], kind="stable")
        for symbol, group in ordered.groupby("symbol", sort=False):
            features = group[self.feature_names].to_numpy(dtype=np.float32, copy=True)
            targets = pd.to_numeric(group[target_name], errors="coerce").to_numpy()
            dates = group["date"].to_numpy()
            for end in range(sequence_length - 1, len(group)):
                date = pd.Timestamp(dates[end]).normalize()
                if normalized_dates is not None and date not in normalized_dates:
                    continue
                target = float(targets[end])
                if not np.isfinite(target):
                    continue
                sequence = features[end - sequence_length + 1 : end + 1]
                if not np.isfinite(sequence).all():
                    continue
                self._samples.append((sequence.copy(), target, SequenceMetadata(date, str(symbol))))
        if not self._samples:
            raise ValueError("no valid sequences were produced")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        features, target, metadata = self._samples[index]
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.float32), metadata

    @property
    def metadata(self) -> list[SequenceMetadata]:
        return [sample[2] for sample in self._samples]
