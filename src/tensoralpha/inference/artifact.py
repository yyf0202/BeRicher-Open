"""Portable Transformer metadata and weights."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tensoralpha.models import TransformerConfig, TransformerRanker


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    model: dict[str, int | float]
    feature_names: list[str]
    sequence_length: int


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    directory: Path

    @classmethod
    def save(
        cls,
        directory: str | Path,
        model: TransformerRanker,
        *,
        feature_names: list[str],
        sequence_length: int,
    ) -> ModelArtifact:
        target = Path(directory).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        metadata = ModelMetadata(
            model=model.config.to_dict(),
            feature_names=list(feature_names),
            sequence_length=int(sequence_length),
        )
        metadata_path = target / "metadata.json"
        metadata_tmp = target / "metadata.json.tmp"
        metadata_tmp.write_text(
            json.dumps(asdict(metadata), indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(metadata_tmp, metadata_path)
        weights_tmp = target / "weights.pt.tmp"
        torch.save(model.state_dict(), weights_tmp)
        os.replace(weights_tmp, target / "weights.pt")
        return cls(target)

    def load(self, device: str = "cpu") -> tuple[TransformerRanker, ModelMetadata]:
        raw = json.loads((self.directory / "metadata.json").read_text(encoding="utf-8"))
        metadata = ModelMetadata(
            model=raw["model"],
            feature_names=list(raw["feature_names"]),
            sequence_length=int(raw["sequence_length"]),
        )
        model = TransformerRanker(TransformerConfig(**metadata.model))
        state = torch.load(self.directory / "weights.pt", map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        return model, metadata
