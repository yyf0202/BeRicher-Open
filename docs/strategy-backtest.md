# Strategy and event backtest

## Portfolio construction

`TopNRotationStrategy` sorts scores descending, resolves ties by symbol, selects `top_n`, and assigns equal weights subject to an optional per-position cap. It contains no execution logic.

## Execution model

`BacktestEngine` advances one trading date at a time:

1. Execute the target saved on the previous date at today's open.
2. Sell before buying so proceeds can fund the rebalance.
3. Mark the account at today's close.
4. Convert today's scores into the next pending target.

The broker applies:

- board-lot rounding (100 shares by default);
- T+1 sellability for newly acquired positions;
- suspension rejection;
- limit-up rejection for buys and limit-down rejection for sells;
- side-aware slippage;
- minimum commission and proportional commission;
- sell-side stamp tax.

## Deliberate simplifications

The public engine fills accepted orders at a slippage-adjusted open and does not model intraday queue position, partial fills, market impact, corporate actions, borrow, or broker connectivity. These assumptions must be revisited before using the software for capacity or execution research.

## Required inputs

Market data uses the canonical panel. Signals use `date`, `symbol`, and `score`. The score dated T is not executed until T+1.

## Published research profile

The README's STK-O curve is separate from the executable synthetic demo. It contains 55 exact normalized-NAV checkpoints from the documented 2015-04-07 through 2026-03-16 Purged K-Fold OOS run after modeled costs. The published maximum drawdown is the statistic from the original daily run; it is not recomputed from sparse checkpoints. Security-level predictions, prices, orders, holdings, accounts, and model files are intentionally excluded.

Run `python scripts/render_showcase.py` to rebuild the README research chart, or add `--include-synthetic` to also rebuild the deterministic demo chart. The research profile is historical evidence, not live performance or an investment promise.
