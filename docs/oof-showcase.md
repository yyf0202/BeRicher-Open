# Privacy-preserving OOF showcase

The private research snapshot contains per-security Transformer K-fold OOF scores. Those rows may encode provider identifiers and are not distributed.

The public showcase has three privacy-separated artifacts:

1. `transformer_11y_oof_summary.json` stores coverage, yearly prediction/date counts, and score mean, standard deviation, and selected quantiles.
2. `transformer_11y_oof_synthetic.csv` samples new scores from each year's aggregate distribution using fixed seed 42 and identifiers `DEMO0001` onward.
3. `stko_11y_oos_profile.json` stores aggregate strategy settings, full-run metrics, and 55 normalized-NAV checkpoints from the authoritative research log. It contains no security-level rows.

The source score is a cross-sectional percentile rank computed within each trading day. A score of 0 is the day's lowest rank, 1 is the highest, and 0.90 means the prediction ranks above roughly 90% of that day's eligible predictions. It is not a point balance, probability, price, or return. This rank construction explains the stable mean near 0.5 and standard deviation near 0.2887. The showcase contains no returns, so it cannot and must not be interpreted as an IC or portfolio-performance report.

The checked-in [`oof_profile.svg`](assets/oof_profile.svg) renders annual prediction counts, the median daily rank, the middle 50% band (P25–P75), and the middle 90% band (P05–P95) directly from the aggregate JSON. The separate [`stko_11y_oos.svg`](assets/stko_11y_oos.svg) renders the exact logged research checkpoints and clearly identifies its Purged K-Fold OOS, modeled-cost, checkpoint-resolution, and not-live-performance boundaries.

Regenerate the two README charts with:

```bash
python scripts/render_showcase.py
```

To also regenerate the deterministic synthetic execution illustration:

```bash
python scripts/render_showcase.py --include-synthetic
```

The SVG renderer is dependency-free, deterministic, and does not embed timestamps or local paths.

To build the same form from your own OOF file:

```bash
python scripts/generate_oof_showcase.py private_oof.csv --output examples/data
```

The script streams only `trade_date` and `alpha_score`, stores aggregate state, and never writes input identifiers.
