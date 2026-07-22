# Offline paper trading

Paper trading persists the same rebalance semantics used by the backtest. It is designed for repeatable daily simulation, not live order placement.

## Create an account

```bash
tensoralpha paper-create --account artifacts/paper/demo --initial-cash 1000000 --top-n 10
```

The directory contains:

- `config.json`: immutable strategy and broker parameters;
- `state.json`: cash, positions, pending target weights, and last processed date.

Both files are runtime artifacts and are ignored by Git.

## Advance one day

```bash
tensoralpha paper-tick --account artifacts/paper/demo --market one_day.csv --signals one_day_signals.csv
```

One invocation must contain exactly one market date. Dates must increase strictly. The previous pending target executes first, then the new signal becomes the next pending target. State is written through a temporary file and atomic replacement.

## Safety boundary

The paper module has no broker credentials, order API, email integration, scheduler installation, or data-repository synchronization. Those concerns must remain outside the core package and require a separate security review.
