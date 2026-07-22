"""Cross-sectional signal diagnostics."""

from tensoralpha.evaluation.metrics import rank_ic_by_date, summarize_oof
from tensoralpha.evaluation.showcase import (
    derive_oof_showcase,
    derive_oof_showcase_csv,
    generate_synthetic_oof,
)

__all__ = [
    "derive_oof_showcase",
    "derive_oof_showcase_csv",
    "generate_synthetic_oof",
    "rank_ic_by_date",
    "summarize_oof",
]
