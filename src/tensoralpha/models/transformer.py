"""Sequence Transformer for one-score-per-security ranking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    input_dim: int
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 256

    def __post_init__(self) -> None:
        if self.input_dim < 1:
            raise ValueError("input_dim must be positive")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] > self.encoding.shape[0]:
            raise ValueError("sequence exceeds max_sequence_length")
        return values + self.encoding[: values.shape[1]].to(values.dtype)


class TransformerRanker(nn.Module):
    """Encode a feature sequence and emit a scalar cross-sectional score."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.position = SinusoidalPositionEncoding(config.d_model, config.max_sequence_length)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, config.num_layers)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, sequence, input_dim]")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected input_dim={self.config.input_dim}, got {features.shape[-1]}"
            )
        hidden = self.position(self.input_projection(features))
        hidden = self.encoder(hidden)
        return self.head(self.output_norm(hidden[:, -1])).squeeze(-1)
