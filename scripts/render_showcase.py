"""Regenerate the README OOF and research-profile SVG assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensoralpha.backtest import BacktestConfig, BacktestEngine
from tensoralpha.demo import generate_demo_panel
from tensoralpha.visualization import (
    load_research_oos_profile,
    render_backtest_svg,
    render_oof_profile_svg,
    render_research_oos_profile_svg,
)

DEFAULT_SHOWCASE_DAYS = 2_610


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("examples/data/transformer_11y_oof_summary.json"),
    )
    parser.add_argument(
        "--research-profile",
        type=Path,
        default=Path("examples/data/stko_11y_oos_profile.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--days", type=int, default=DEFAULT_SHOWCASE_DAYS)
    parser.add_argument("--assets", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=10)
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    render_oof_profile_svg(summary, args.output / "oof_profile.svg")

    research_profile = load_research_oos_profile(args.research_profile)
    render_research_oos_profile_svg(research_profile, args.output / "stko_11y_oos.svg")

    if args.include_synthetic:
        demo = generate_demo_panel(args.days, args.assets, args.seed)
        result = BacktestEngine(BacktestConfig(initial_cash=1_000_000.0, top_n=args.top_n)).run(
            demo.market, demo.signals
        )
        render_backtest_svg(result.nav, args.output / "synthetic_backtest.svg")
    print(f"Wrote reproducible showcase SVGs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
