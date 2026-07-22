import json
from pathlib import Path


def test_cli_help_lists_complete_workflow(capsys):
    from tensoralpha.cli import main

    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    for command in ["demo", "train", "oof", "backtest", "paper-create", "paper-tick"]:
        assert command in output


def test_demo_command_writes_reusable_artifacts(tmp_path: Path):
    from tensoralpha.cli import main

    assert (
        main(
            [
                "demo",
                "--output",
                str(tmp_path),
                "--days",
                "20",
                "--assets",
                "6",
                "--seed",
                "5",
            ]
        )
        == 0
    )

    expected = {
        "market.csv",
        "signals.csv",
        "backtest_nav.csv",
        "backtest_nav.svg",
        "backtest_trades.csv",
        "summary.json",
    }
    assert expected.issubset(path.name for path in tmp_path.iterdir())
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["synthetic"] is True
    assert summary["days"] == 20
    chart = (tmp_path / "backtest_nav.svg").read_text(encoding="utf-8")
    assert "Synthetic Backtest" in chart


def test_run_config_executes_the_same_cli_contract(tmp_path: Path):
    from tensoralpha.cli import main

    config_path = tmp_path / "demo.json"
    output_path = tmp_path / "configured-output"
    config_path.write_text(
        json.dumps(
            {
                "command": "demo",
                "arguments": {
                    "output": str(output_path),
                    "days": 12,
                    "assets": 4,
                    "seed": 3,
                    "top_n": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    assert main(["run-config", str(config_path)]) == 0
    assert (output_path / "summary.json").exists()
