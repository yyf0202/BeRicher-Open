import json
import runpy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def _oof_summary() -> dict:
    annual = {}
    for offset, year in enumerate(range(2015, 2027)):
        annual[str(year)] = {
            "observations": 400_000 + offset * 75_000,
            "score_q05": 0.05 + offset * 0.0002,
            "score_q25": 0.25 + offset * 0.0001,
            "score_q50": 0.50,
            "score_q75": 0.75 - offset * 0.0001,
            "score_q95": 0.95 - offset * 0.0002,
        }
    return {
        "annual": annual,
        "coverage": {
            "start": "2015-04-07",
            "end": "2026-04-10",
            "observations": sum(row["observations"] for row in annual.values()),
        },
    }


def test_oof_profile_svg_is_deterministic(tmp_path: Path):
    from tensoralpha.visualization import render_oof_profile_svg

    first = tmp_path / "first.svg"
    second = tmp_path / "nested" / "second.svg"

    render_oof_profile_svg(_oof_summary(), first)
    render_oof_profile_svg(_oof_summary(), second)

    assert first.read_bytes() == second.read_bytes()
    svg = first.read_text(encoding="utf-8")
    for label in [
        "OOF Coverage &amp; Daily Rank Scores",
        "Annual OOF predictions",
        "Daily rank score: 0 = lowest, 1 = highest",
        "Middle 90% (P05–P95)",
        "Middle 50% (P25–P75)",
        "Median daily rank",
    ]:
        assert label in svg
    assert "2015" in svg and "2026" in svg


def test_oof_profile_rejects_non_finite_data_without_partial_file(tmp_path: Path):
    from tensoralpha.visualization import VisualizationError, render_oof_profile_svg

    summary = _oof_summary()
    summary["annual"]["2020"]["score_q50"] = float("nan")
    output = tmp_path / "profile.svg"

    with pytest.raises(VisualizationError, match="finite"):
        render_oof_profile_svg(summary, output)

    assert not output.exists()


def test_backtest_svg_contains_synthetic_nav_and_drawdown(tmp_path: Path):
    from tensoralpha.visualization import render_backtest_svg

    nav = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=12),
            "nav": [
                100.0,
                101.0,
                100.5,
                102.0,
                103.0,
                102.0,
                104.0,
                105.0,
                104.0,
                106.0,
                107.0,
                108.0,
            ],
        }
    )
    output = tmp_path / "backtest.svg"

    render_backtest_svg(nav, output)

    svg = output.read_text(encoding="utf-8")
    for label in ["Synthetic Backtest", "Illustrative only", "Normalized NAV", "Drawdown"]:
        assert label in svg
    assert svg.count("<polyline") >= 2


def test_ten_year_backtest_svg_names_exact_coverage(tmp_path: Path):
    from tensoralpha.visualization import render_backtest_svg

    dates = pd.bdate_range("2015-01-05", periods=2_610)
    nav = pd.DataFrame({"date": dates, "nav": 100.0 + pd.Series(range(len(dates)))})
    output = tmp_path / "ten_year_backtest.svg"

    render_backtest_svg(nav, output)

    svg = output.read_text(encoding="utf-8")
    for label in [
        "10-Year Synthetic Backtest",
        "2015-01-05 to 2025-01-03",
        "Illustrative only",
        "Not historical performance",
    ]:
        assert label in svg


def test_backtest_svg_rejects_non_positive_nav(tmp_path: Path):
    from tensoralpha.visualization import VisualizationError, render_backtest_svg

    nav = pd.DataFrame(
        {"date": pd.bdate_range("2025-01-02", periods=3), "nav": [100.0, 0.0, 101.0]}
    )

    with pytest.raises(VisualizationError, match="positive"):
        render_backtest_svg(nav, tmp_path / "backtest.svg")


def test_showcase_script_writes_oof_and_research_readme_assets(tmp_path: Path):
    repository = Path(__file__).resolve().parents[2]
    summary_path = tmp_path / "summary.json"
    output = tmp_path / "assets"
    summary_path.write_text(json.dumps(_oof_summary()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "render_showcase.py"),
            "--summary",
            str(summary_path),
            "--output",
            str(output),
            "--days",
            "20",
            "--assets",
            "6",
        ],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "oof_profile.svg").exists()
    assert (output / "stko_11y_oos.svg").exists()
    assert not (output / "synthetic_backtest.svg").exists()


def test_showcase_default_and_checked_asset_cover_ten_years():
    repository = Path(__file__).resolve().parents[2]
    script = runpy.run_path(str(repository / "scripts" / "render_showcase.py"))

    assert script["_build_parser"]().parse_args([]).days == 2_610

    svg = (repository / "docs" / "assets" / "synthetic_backtest.svg").read_text(encoding="utf-8")
    for label in ["10-Year Synthetic Backtest", "2015-01-05", "2025-01-03"]:
        assert label in svg


def test_showcase_can_optionally_render_synthetic_asset(tmp_path: Path):
    repository = Path(__file__).resolve().parents[2]
    output = tmp_path / "assets"

    result = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "render_showcase.py"),
            "--output",
            str(output),
            "--include-synthetic",
            "--days",
            "20",
            "--assets",
            "6",
        ],
        cwd=repository,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "synthetic_backtest.svg").exists()
