# Contributing

1. Create a focused branch and keep runtime artifacts outside Git.
2. Add a failing test before changing behavior.
3. Keep data, features, training, execution, and persistence concerns separated.
4. Run the complete local gate:

```bash
ruff check .
ruff format --check .
pytest -q
python scripts/check_release.py .
python -m build
```

Pull requests should state the behavioral change, leakage assumptions, tests, and any impact on transaction-cost or time semantics. New data providers must document units, adjustment rules, point-in-time availability, and redistribution constraints.

Never submit credentials, personal paths, raw licensed data, model weights, account state, holdings, orders, or trades.
