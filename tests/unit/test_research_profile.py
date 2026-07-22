import copy
import json
import re
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPOSITORY / "examples" / "data" / "stko_11y_oos_profile.json"


def _profile() -> dict:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_checked_profile_matches_authoritative_oos_run():
    profile = _profile()
    coverage = profile["evaluation"]["coverage"]
    metrics = profile["metrics"]
    series = profile["nav_series"]
    checkpoints = series["checkpoints"]

    assert profile["strategy"]["display_name"] == "TensorAlpha STK-O"
    assert profile["strategy"]["components"] == [
        {"name": "ME", "weight": 0.4},
        {"name": "CE_Liq", "weight": 0.2},
        {"name": "V46ME S43", "weight": 0.4},
    ]
    assert profile["evaluation"]["method"] == "Purged K-Fold OOS"
    assert coverage == {
        "start": "2015-04-07",
        "end": "2026-03-16",
        "trading_days": 2_659,
    }
    assert metrics["total_return"] == pytest.approx(3.2436752757)
    assert metrics["cagr"] == pytest.approx(0.1468131762)
    assert metrics["max_drawdown"] == pytest.approx(-0.5587568167)
    assert metrics["sharpe"] == pytest.approx(0.5894039057)
    assert series["checkpoint_stride_trading_days"] == 50
    assert len(checkpoints) == series["checkpoint_count"] == 55
    assert checkpoints[0] == {
        "day": 1,
        "date": "2015-04-07",
        "normalized_nav": 1.0,
    }
    assert checkpoints[-1] == {
        "day": 2659,
        "date": "2026-03-16",
        "normalized_nav": 4.24367528,
    }
    assert [row["day"] for row in checkpoints] == [1, *range(50, 2651, 50), 2659]
    assert [row["date"] for row in checkpoints] == sorted(row["date"] for row in checkpoints)


def test_checked_profile_contains_no_security_or_private_runtime_fields():
    profile = _profile()
    prohibited_keys = {
        "symbol",
        "ts_code",
        "security",
        "price",
        "trade",
        "position",
        "account",
        "model_path",
        "local_path",
        "credential",
        "token",
        "timestamp",
    }

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key.casefold()
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert prohibited_keys.isdisjoint(keys(profile))


def test_research_profile_svg_is_deterministic_and_honest(tmp_path: Path):
    from tensoralpha.visualization import render_research_oos_profile_svg

    first = tmp_path / "first.svg"
    second = tmp_path / "nested" / "second.svg"

    render_research_oos_profile_svg(_profile(), first)
    render_research_oos_profile_svg(_profile(), second)

    assert first.read_bytes() == second.read_bytes()
    svg = first.read_text(encoding="utf-8")
    for label in [
        "TensorAlpha · STK-O 11-Year OOS Profile",
        "Purged K-Fold OOS",
        "50-trading-day checkpoints",
        "After modeled costs",
        "Historical research · Not live performance",
        "Cumulative return",
        "CAGR",
        "Full-run max drawdown",
        "Sharpe",
        "0.5894",
        "original daily run",
        "2015-04-07",
        "2026-03-16",
    ]:
        assert label in svg
    assert svg.count("<polyline") == 1
    assert "Checkpoint drawdown" not in svg
    points = re.search(r'<polyline points="([^"]+)"', svg)
    assert points is not None
    assert len(points.group(1).split()) == 55
    assert first.read_bytes() == (REPOSITORY / "docs" / "assets" / "stko_11y_oos.svg").read_bytes()


@pytest.mark.parametrize("bad_key", ["symbol", "model_path", "token"])
def test_research_profile_rejects_prohibited_fields(tmp_path: Path, bad_key: str):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile[bad_key] = "private-value"
    output = tmp_path / "profile.svg"

    with pytest.raises(VisualizationError, match="prohibited"):
        render_research_oos_profile_svg(profile, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "bad_key",
    ["security_codes", "positions", "modelPaths", "apiToken", "tradeRecords"],
)
def test_research_profile_rejects_prohibited_field_variants(tmp_path: Path, bad_key: str):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile[bad_key] = []

    with pytest.raises(VisualizationError, match="prohibited|unknown"):
        render_research_oos_profile_svg(profile, tmp_path / "profile.svg")


@pytest.mark.parametrize(
    "sensitive_value",
    [
        "D:" + r"\private\workspace\model.pt",
        "/" + "home/research/private/model.pt",
        "600000.SH",
        "2026-04-18T23:20:00",
        "gh" + "p_" + "A" * 36,
    ],
)
def test_research_profile_rejects_sensitive_string_values(tmp_path: Path, sensitive_value: str):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile["strategy"]["selection_basis"] = sensitive_value

    with pytest.raises(VisualizationError, match="sensitive value"):
        render_research_oos_profile_svg(profile, tmp_path / "profile.svg")


@pytest.mark.parametrize(
    "missing_section", ["schema_version", "strategy", "universe", "execution", "privacy"]
)
def test_research_profile_requires_full_public_contract(tmp_path: Path, missing_section: str):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile.pop(missing_section)

    with pytest.raises(VisualizationError, match="missing|required"):
        render_research_oos_profile_svg(profile, tmp_path / "profile.svg")


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("evaluation", "coverage", "trading_days"), 2659.9),
        (("nav_series", "checkpoint_count"), True),
        (("nav_series", "checkpoints", 0, "day"), 1.5),
        (("nav_series", "checkpoints", 0, "date"), "2015-04-07T00:00:00"),
    ],
)
def test_research_profile_rejects_noncanonical_counts_and_dates(
    tmp_path: Path, path: tuple[object, ...], bad_value: object
):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    target = profile
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = bad_value

    with pytest.raises(VisualizationError, match="integer|YYYY-MM-DD|sensitive value"):
        render_research_oos_profile_svg(profile, tmp_path / "profile.svg")


@pytest.mark.parametrize(
    ("metric", "bad_value"),
    [
        ("total_return", 2.0),
        ("cagr", 0.5),
        ("max_drawdown", -0.1),
        ("sharpe", 2.0),
    ],
)
def test_research_profile_rejects_non_authoritative_metrics(
    tmp_path: Path, metric: str, bad_value: float
):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile["metrics"][metric] = bad_value

    with pytest.raises(VisualizationError, match="authoritative"):
        render_research_oos_profile_svg(profile, tmp_path / "profile.svg")


def test_research_profile_requires_authoritative_logged_series(tmp_path: Path):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile["nav_series"]["kind"] = "interpolated"

    with pytest.raises(VisualizationError, match="authoritative logged checkpoints"):
        render_research_oos_profile_svg(profile, tmp_path / "profile.svg")


def test_research_profile_loader_rejects_duplicate_json_keys(tmp_path: Path):
    from tensoralpha.visualization import VisualizationError, load_research_oos_profile

    profile_path = tmp_path / "duplicate.json"
    profile_path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")

    with pytest.raises(VisualizationError, match="duplicate JSON key"):
        load_research_oos_profile(profile_path)


def test_research_profile_rejects_inconsistent_final_nav(tmp_path: Path):
    from tensoralpha.visualization import (
        VisualizationError,
        render_research_oos_profile_svg,
    )

    profile = copy.deepcopy(_profile())
    profile["nav_series"]["checkpoints"][-1]["normalized_nav"] = 9.0
    output = tmp_path / "profile.svg"

    with pytest.raises(VisualizationError, match="total return"):
        render_research_oos_profile_svg(profile, output)

    assert not output.exists()


def test_readmes_distinguish_stko_research_from_live_performance():
    english = (REPOSITORY / "README.md").read_text(encoding="utf-8")
    chinese = (REPOSITORY / "README.zh-CN.md").read_text(encoding="utf-8")

    for text in (english, chinese):
        assert "STK-O" in text
        assert "Purged K-Fold OOS" in text
        assert "docs/assets/stko_11y_oos.svg" in text
    assert "not live performance" in english.casefold()
    assert "standalone V46ME" in english
    assert "不是实盘业绩" in chinese
    assert "单独的 V46ME" in chinese
