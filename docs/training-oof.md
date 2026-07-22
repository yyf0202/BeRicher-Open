# Transformer training and OOF

## Features and labels

`FeaturePipeline` uses trailing returns, moving-average distance, realized volatility, intraday price location, volume ratios, and daily amount rank. Warm-up gaps and non-finite values are replaced with neutral zero after clipping.

`add_forward_rank_target()` computes a per-security forward return and ranks that return within the same date. The target is never part of model input.

## Sequences

`PanelSequenceDataset` groups by symbol before creating rolling windows, so a sequence can never cross security boundaries. Every sample carries its target date and symbol as metadata.

## Purged walk-forward validation

`PurgedWalkForwardSplit` uses an expanding training window. A configurable gap separates the end of training from each validation interval. The purge must cover at least the maximum forward-label horizon and any additional leakage boundary required by the experiment.

Each fold:

1. receives a fresh Transformer initialized from the configured seed;
2. trains only on that fold's training target dates;
3. predicts only its validation target dates;
4. saves strict architecture metadata and weights;
5. contributes unique `(date, symbol)` scores to `assemble_oof()`.

`assemble_oof()` rejects duplicate predictions and computes a daily percentile rank. OOF means out-of-fold model prediction; it does not mean that every historical date is necessarily covered, because initial training and purge periods have no validation prediction.

## Reproducibility

- Record the data snapshot and adjustment convention.
- Fix the seed, sequence length, horizon, split count, purge, and minimum training history.
- Do not compare in-sample replay with clean OOF metrics.
- Treat score diagnostics separately from trading performance after costs.
