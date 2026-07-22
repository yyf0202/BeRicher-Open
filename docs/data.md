# Data contract

## Canonical panel

Market data is a long-form DataFrame with one row per `(date, symbol)`:

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `date` | datetime | trading day | normalized session date |
| `symbol` | string | — | provider security identifier |
| `open/high/low/close` | float | CNY | positive unadjusted or consistently adjusted prices |
| `volume` | float | shares | traded shares |
| `amount` | float | CNY | traded notional |

Optional execution flags are `suspended`, `limit_up`, and `limit_down`. Extra point-in-time columns are preserved.

`validate_panel()` rejects missing fields, duplicate keys, non-positive prices, negative volume/amount, and inconsistent high/low values. It returns data sorted by `(date, symbol)`.

## Local storage

`ParquetPanelStore` writes through a temporary file and atomic replacement. Generated data belongs in `data/`, which is ignored by Git.

## Tushare

Install the optional provider and set `TENSORALPHA_TUSHARE_TOKEN` in the process environment. The adapter converts Tushare lots to shares and thousands of CNY to CNY.

This repository does not redistribute Tushare data. Users are responsible for provider authorization, rate limits, field availability, and redistribution terms.

## Synthetic data

`tensoralpha.demo.generate_demo_panel()` uses a fixed pseudo-random process. Its `DEMO` identifiers, prices, volumes, signals, and returns are synthetic and are safe for tests, examples, and CI.
