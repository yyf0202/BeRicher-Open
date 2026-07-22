"""Create privacy-preserving OOF aggregate and synthetic example files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensoralpha.evaluation import derive_oof_showcase_csv, generate_synthetic_oof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="private OOF CSV; never copied")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-family", default="transformer_v3_k5")
    parser.add_argument("--dates-per-year", type=int, default=20)
    parser.add_argument("--assets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = derive_oof_showcase_csv(args.source, model_family=args.model_family, seed=args.seed)
    synthetic = generate_synthetic_oof(
        summary,
        dates_per_year=args.dates_per_year,
        assets=args.assets,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transformer_11y_oof_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    synthetic.to_csv(args.output / "transformer_11y_oof_synthetic.csv", index=False)
    print(f"Wrote aggregate and synthetic showcase to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
