# Architecture

TensorAlpha uses narrow interfaces and a one-way data flow. Runtime modules do not depend on scripts or repository layout, and importing a module never starts network, training, or file-writing work.

## Layers

1. `tensoralpha.data` validates the canonical market panel and persists it atomically.
2. `tensoralpha.features` generates features from current and past observations. Forward returns are created only as labels.
3. `tensoralpha.training` converts each security into rolling sequences, performs expanding-window purged splits, trains fresh models, and assembles unique OOF predictions.
4. `tensoralpha.inference` stores architecture metadata separately from weights and restores models strictly.
5. `tensoralpha.evaluation` computes daily cross-sectional diagnostics and privacy-preserving aggregates.
6. `tensoralpha.strategy` converts one day's scores into target weights.
7. `tensoralpha.backtest` executes the previous day's target through a market-rule-aware broker.
8. `tensoralpha.paper` persists the same target execution across daily invocations.

## Time semantics

For trading date T:

1. Pending targets generated after T-1 close execute at T open.
2. The portfolio is marked at T close.
3. Scores available at T close become target weights for T+1.

No order generated from T information is allowed to fill on T. A position acquired on T is not sellable until a later trading date.

## Boundaries

- Network access exists only in explicit provider calls.
- Credentials live only in process environment or untracked local configuration.
- Runtime output stays below `artifacts/` by default.
- Raw data, weights, state, and reports are never Python package resources.
- Scripts are thin entry points; business behavior lives in `src/tensoralpha` and is tested directly.
