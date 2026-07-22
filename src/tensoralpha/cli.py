"""Single command-line interface for the public TensorAlpha workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from tensoralpha.backtest import BacktestConfig, BacktestEngine, BrokerConfig
from tensoralpha.data import ParquetPanelStore
from tensoralpha.demo import generate_demo_panel
from tensoralpha.features import FeaturePipeline, add_forward_rank_target
from tensoralpha.inference import ModelArtifact
from tensoralpha.models import TransformerConfig, TransformerRanker
from tensoralpha.paper import PaperTrader
from tensoralpha.settings import Settings
from tensoralpha.strategy import TopNRotationStrategy
from tensoralpha.training import (
    ModelTrainer,
    PanelSequenceDataset,
    PurgedWalkForwardSplit,
    TrainingConfig,
    run_purged_oof,
)
from tensoralpha.visualization import render_backtest_svg


def _read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, parse_dates=["date"])
    raise ValueError(f"unsupported tabular format: {source.suffix}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensoralpha",
        description="Transformer research, OOF, backtesting, and paper trading",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a complete synthetic backtest")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--days", type=int, default=260)
    demo.add_argument("--assets", type=int, default=40)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--top-n", type=int, default=10)

    train = subparsers.add_parser("train", help="train one Transformer model")
    train.add_argument("--market", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--sequence-length", type=int, default=60)
    train.add_argument("--horizon", type=int, default=1)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int, default=42)

    oof = subparsers.add_parser("oof", help="train purged folds and write OOF scores")
    oof.add_argument("--market", type=Path, required=True)
    oof.add_argument("--output", type=Path, required=True)
    oof.add_argument("--sequence-length", type=int, default=60)
    oof.add_argument("--horizon", type=int, default=1)
    oof.add_argument("--folds", type=int, default=5)
    oof.add_argument("--purge-days", type=int, default=20)
    oof.add_argument("--min-train-days", type=int, default=756)
    oof.add_argument("--epochs", type=int, default=20)
    oof.add_argument("--batch-size", type=int, default=512)
    oof.add_argument("--device", default="cpu")
    oof.add_argument("--seed", type=int, default=42)

    backtest = subparsers.add_parser("backtest", help="run event-driven backtest")
    backtest.add_argument("--market", type=Path, required=True)
    backtest.add_argument("--signals", type=Path, required=True)
    backtest.add_argument("--output", type=Path, required=True)
    backtest.add_argument("--initial-cash", type=float, default=1_000_000.0)
    backtest.add_argument("--top-n", type=int, default=10)

    paper_create = subparsers.add_parser("paper-create", help="create an offline paper account")
    paper_create.add_argument("--account", type=Path, required=True)
    paper_create.add_argument("--initial-cash", type=float, default=1_000_000.0)
    paper_create.add_argument("--top-n", type=int, default=10)

    paper_tick = subparsers.add_parser("paper-tick", help="advance one paper-trading day")
    paper_tick.add_argument("--account", type=Path, required=True)
    paper_tick.add_argument("--market", type=Path, required=True)
    paper_tick.add_argument("--signals", type=Path, required=True)

    fetch = subparsers.add_parser("fetch-data", help="download Tushare daily data")
    fetch.add_argument("--start", required=True)
    fetch.add_argument("--end", required=True)
    fetch.add_argument("--output", type=Path, required=True)

    run_config = subparsers.add_parser("run-config", help="execute a checked-in JSON profile")
    run_config.add_argument("config", type=Path)
    return parser


def _prepare_training_panel(path: Path, horizon: int):
    pipeline = FeaturePipeline()
    market = _read_frame(path)
    featured = pipeline.transform(market)
    training_panel = add_forward_rank_target(featured, horizon=horizon)
    return pipeline, training_panel


def _small_public_model(input_dim: int) -> TransformerConfig:
    return TransformerConfig(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    )


def _run_demo(args) -> int:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    demo = generate_demo_panel(args.days, args.assets, args.seed)
    result = BacktestEngine(BacktestConfig(initial_cash=1_000_000.0, top_n=args.top_n)).run(
        demo.market, demo.signals
    )
    demo.market.to_csv(output / "market.csv", index=False)
    demo.signals.to_csv(output / "signals.csv", index=False)
    result.nav.to_csv(output / "backtest_nav.csv", index=False)
    render_backtest_svg(result.nav, output / "backtest_nav.svg")
    result.trades.to_csv(output / "backtest_trades.csv", index=False)
    final_nav = float(result.nav.iloc[-1]["nav"])
    _write_json(
        output / "summary.json",
        {
            "synthetic": True,
            "days": args.days,
            "assets": args.assets,
            "seed": args.seed,
            "trades": len(result.trades),
            "final_nav": final_nav,
            "total_return": final_nav / 1_000_000.0 - 1.0,
        },
    )
    print(f"Synthetic demo written to {output}")
    return 0


def _run_train(args) -> int:
    pipeline, panel = _prepare_training_panel(args.market, args.horizon)
    dataset = PanelSequenceDataset(
        panel,
        feature_names=pipeline.feature_names,
        target_name="target_rank",
        sequence_length=args.sequence_length,
    )
    model = TransformerRanker(_small_public_model(len(pipeline.feature_names)))
    trainer = ModelTrainer(
        model,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ),
    )
    history = trainer.fit(dataset)
    ModelArtifact.save(
        args.output,
        model,
        feature_names=pipeline.feature_names,
        sequence_length=args.sequence_length,
    )
    _write_json(args.output / "training_history.json", {"epochs": [asdict(x) for x in history]})
    return 0


def _run_oof(args) -> int:
    pipeline, panel = _prepare_training_panel(args.market, args.horizon)
    output = args.output.expanduser().resolve()
    result = run_purged_oof(
        panel,
        feature_names=pipeline.feature_names,
        target_name="target_rank",
        sequence_length=args.sequence_length,
        splitter=PurgedWalkForwardSplit(
            n_splits=args.folds,
            purge_days=args.purge_days,
            min_train_days=args.min_train_days,
        ),
        model_config=_small_public_model(len(pipeline.feature_names)),
        training_config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
        ),
        artifact_dir=output / "models",
    )
    output.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output / "oof_predictions.parquet", index=False)
    return 0


def _run_backtest(args) -> int:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = BacktestEngine(BacktestConfig(initial_cash=args.initial_cash, top_n=args.top_n)).run(
        _read_frame(args.market), _read_frame(args.signals)
    )
    result.nav.to_csv(output / "nav.csv", index=False)
    result.trades.to_csv(output / "trades.csv", index=False)
    return 0


def _run_paper_create(args) -> int:
    PaperTrader.create(
        args.account,
        initial_cash=args.initial_cash,
        strategy=TopNRotationStrategy(top_n=args.top_n),
        broker_config=BrokerConfig(),
    )
    return 0


def _run_paper_tick(args) -> int:
    trader = PaperTrader.load(args.account)
    fills = trader.tick(_read_frame(args.market), _read_frame(args.signals))
    print(f"Executed {len(fills)} fills")
    return 0


def _run_fetch(args) -> int:
    from tensoralpha.data.tushare import TushareDailySource

    settings = Settings.from_env()
    source = TushareDailySource(settings.require_tushare_token())
    ParquetPanelStore(args.output).write(source.fetch_daily(args.start, args.end))
    return 0


def _run_config(args) -> int:
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    command = payload.get("command")
    arguments = payload.get("arguments", {})
    if not isinstance(command, str) or command == "run-config":
        raise ValueError("config command must name a non-recursive CLI command")
    if not isinstance(arguments, dict):
        raise ValueError("config arguments must be a JSON object")
    forwarded = [command]
    for name, value in arguments.items():
        option = "--" + str(name).replace("_", "-")
        if isinstance(value, bool):
            if value:
                forwarded.append(option)
            continue
        if value is None:
            continue
        forwarded.extend([option, str(value)])
    return main(forwarded)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        if argv is not None:
            return int(error.code)
        raise
    handlers = {
        "demo": _run_demo,
        "train": _run_train,
        "oof": _run_oof,
        "backtest": _run_backtest,
        "paper-create": _run_paper_create,
        "paper-tick": _run_paper_tick,
        "fetch-data": _run_fetch,
        "run-config": _run_config,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
