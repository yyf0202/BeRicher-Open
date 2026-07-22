"""Deterministic, dependency-free SVG charts for public research artifacts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import pandas as pd

_INK = "#172033"
_MUTED = "#667085"
_GRID = "#D9DEE8"
_SURFACE = "#FAFAF8"
_BLUE = "#2367C9"
_BLUE_DARK = "#164B96"
_BLUE_LIGHT = "#DCEAFF"
_BLUE_MID = "#8CB9F2"
_GOLD = "#C28A16"
_GOLD_LIGHT = "#F4E4B6"
_ORANGE = "#D66B2C"
_ORANGE_LIGHT = "#F7DDCF"


class VisualizationError(ValueError):
    """Raised when chart input cannot produce an honest visualization."""


def _write_svg(output: str | Path, svg: str) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(svg, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _fmt_count(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def _points(values: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in values)


def _text(x: float, y: float, label: object, *, css: str = "label", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" class="{css}" '
        f'text-anchor="{anchor}">{escape(str(label))}</text>'
    )


def _base_svg(title: str, description: str, body: Sequence[str], *, height: int) -> str:
    css = f"""
    .title {{ font: 700 28px Inter, Segoe UI, sans-serif; fill: {_INK}; }}
    .subtitle {{ font: 400 14px Inter, Segoe UI, sans-serif; fill: {_MUTED}; }}
    .panel-title {{ font: 650 16px Inter, Segoe UI, sans-serif; fill: {_INK}; }}
    .label {{ font: 500 12px Inter, Segoe UI, sans-serif; fill: {_INK}; }}
    .muted {{ font: 400 11px Inter, Segoe UI, sans-serif; fill: {_MUTED}; }}
    .mono {{ font: 600 12px ui-monospace, SFMono-Regular, Consolas, monospace; fill: {_INK}; }}
    .grid {{ stroke: {_GRID}; stroke-width: 1; shape-rendering: crispEdges; }}
    .axis {{ stroke: {_MUTED}; stroke-width: 1.2; shape-rendering: crispEdges; }}
    """.strip()
    mark = (
        f'<g aria-label="TensorAlpha mark" transform="translate(1137 43)">'
        f'<circle cx="0" cy="-9" r="6" fill="{_BLUE}"/>'
        f'<circle cx="9" cy="0" r="6" fill="{_GOLD}"/>'
        f'<circle cx="0" cy="9" r="6" fill="{_BLUE_MID}"/>'
        f'<circle cx="-9" cy="0" r="6" fill="{_GOLD_LIGHT}" stroke="{_GOLD}"/>'
        "</g>"
    )
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" role="img">',
            f"<title>{escape(title)}</title>",
            f"<desc>{escape(description)}</desc>",
            f"<style>{css}</style>",
            f'<rect width="1200" height="{height}" fill="{_SURFACE}"/>',
            mark,
            *body,
            "</svg>",
            "",
        ]
    )


def _validated_annual(summary: Mapping[str, Any]) -> list[tuple[int, dict[str, float]]]:
    annual = summary.get("annual")
    if not isinstance(annual, Mapping) or not annual:
        raise VisualizationError("annual OOF data must be a non-empty mapping")
    required = ("observations", "score_q05", "score_q25", "score_q50", "score_q75", "score_q95")
    rows: list[tuple[int, dict[str, float]]] = []
    for year_key, raw in annual.items():
        if not isinstance(year_key, str) or len(year_key) != 4 or not year_key.isdigit():
            raise VisualizationError(f"invalid OOF year key: {year_key!r}")
        if not isinstance(raw, Mapping):
            raise VisualizationError(f"OOF year {year_key} must contain a mapping")
        values: dict[str, float] = {}
        for field in required:
            if field not in raw:
                raise VisualizationError(f"OOF year {year_key} is missing {field}")
            try:
                value = float(raw[field])
            except (TypeError, ValueError) as error:
                raise VisualizationError(
                    f"OOF year {year_key} field {field} must be numeric"
                ) from error
            if not math.isfinite(value):
                raise VisualizationError(f"OOF year {year_key} field {field} must be finite")
            values[field] = value
        if values["observations"] <= 0:
            raise VisualizationError(f"OOF year {year_key} observations must be positive")
        quantiles = [values[field] for field in required[1:]]
        if quantiles != sorted(quantiles) or quantiles[0] < 0 or quantiles[-1] > 1:
            raise VisualizationError(
                f"OOF year {year_key} score quantiles must be ordered in [0, 1]"
            )
        rows.append((int(year_key), values))
    return sorted(rows)


def render_oof_profile_svg(summary: Mapping[str, Any], output: str | Path) -> Path:
    """Render annual aggregate OOF coverage and score bands to a stable SVG."""

    rows = _validated_annual(summary)
    years = [year for year, _ in rows]
    coverage = summary.get("coverage") if isinstance(summary.get("coverage"), Mapping) else {}
    total = float(coverage.get("observations", sum(row["observations"] for _, row in rows)))
    if not math.isfinite(total) or total <= 0:
        raise VisualizationError("coverage observations must be finite and positive")
    start = escape(str(coverage.get("start", years[0])))
    end = escape(str(coverage.get("end", years[-1])))

    left, right = 92.0, 1144.0
    width = right - left
    step = width / len(rows)
    x_positions = [left + step * (index + 0.5) for index in range(len(rows))]
    body = [
        _text(58, 52, "TensorAlpha · OOF Coverage & Daily Rank Scores", css="title"),
        _text(
            58, 78, f"{_fmt_count(total)} aggregate predictions · {start} to {end}", css="subtitle"
        ),
        _text(58, 118, "Annual OOF predictions", css="panel-title"),
        _text(270, 118, "Calendar-year counts; 2015 and 2026 are partial", css="muted"),
    ]

    bar_top, bar_bottom = 142.0, 314.0
    max_observations = max(row["observations"] for _, row in rows)
    axis_max = math.ceil(max_observations / 100_000) * 100_000
    for fraction in (0.0, 0.5, 1.0):
        y = bar_bottom - fraction * (bar_bottom - bar_top)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        body.append(
            _text(left - 12, y + 4, _fmt_count(axis_max * fraction), css="muted", anchor="end")
        )
    body.append(
        f'<line x1="{left}" y1="{bar_bottom}" x2="{right}" y2="{bar_bottom}" class="axis"/>'
    )
    for x, (year, row) in zip(x_positions, rows, strict=True):
        height = row["observations"] / axis_max * (bar_bottom - bar_top)
        body.append(
            f'<rect x="{x - step * 0.31:.1f}" y="{bar_bottom - height:.1f}" '
            f'width="{step * 0.62:.1f}" height="{height:.1f}" rx="2" fill="{_GOLD_LIGHT}" '
            f'stroke="{_GOLD}"/>'
        )
        body.append(_text(x, bar_bottom + 22, year, css="muted", anchor="middle"))

    score_top, score_bottom = 414.0, 628.0
    body.extend(
        [
            _text(
                58,
                382,
                "Daily rank score: 0 = lowest, 1 = highest",
                css="panel-title",
            ),
        ]
    )
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = score_bottom - value * (score_bottom - score_top)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        body.append(_text(left - 12, y + 4, f"{value:.2f}", css="muted", anchor="end"))

    def y_score(value: float) -> float:
        return score_bottom - value * (score_bottom - score_top)

    q95 = [(x, y_score(row["score_q95"])) for x, (_, row) in zip(x_positions, rows, strict=True)]
    q05 = [(x, y_score(row["score_q05"])) for x, (_, row) in zip(x_positions, rows, strict=True)]
    q75 = [(x, y_score(row["score_q75"])) for x, (_, row) in zip(x_positions, rows, strict=True)]
    q25 = [(x, y_score(row["score_q25"])) for x, (_, row) in zip(x_positions, rows, strict=True)]
    median = [(x, y_score(row["score_q50"])) for x, (_, row) in zip(x_positions, rows, strict=True)]
    body.append(
        f'<polygon points="{_points([*q95, *reversed(q05)])}" fill="{_BLUE_LIGHT}" '
        f'stroke="{_BLUE_MID}" stroke-dasharray="5 4"/>'
    )
    body.append(
        f'<polygon points="{_points([*q75, *reversed(q25)])}" fill="{_BLUE_MID}" '
        f'fill-opacity="0.62" stroke="{_BLUE}"/>'
    )
    body.append(
        f'<polyline points="{_points(median)}" fill="none" stroke="{_BLUE_DARK}" stroke-width="2.4"/>'
    )
    for x, y in median:
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{_SURFACE}" stroke="{_BLUE_DARK}"/>'
        )
    body.extend(
        [
            f'<rect x="520" y="367" width="20" height="10" fill="{_BLUE_LIGHT}" stroke="{_BLUE_MID}"/>',
            _text(548, 376, "Middle 90% (P05–P95)", css="muted"),
            f'<rect x="704" y="367" width="20" height="10" fill="{_BLUE_MID}" stroke="{_BLUE}"/>',
            _text(732, 376, "Middle 50% (P25–P75)", css="muted"),
            f'<line x1="900" y1="372" x2="924" y2="372" stroke="{_BLUE_DARK}" stroke-width="2.4"/>',
            _text(932, 376, "Median daily rank", css="muted"),
            _text(
                58,
                682,
                "Source: aggregate OOF statistics only · No securities, prices, returns, or account data",
                css="muted",
            ),
        ]
    )
    svg = _base_svg(
        "TensorAlpha OOF Coverage & Daily Rank Scores",
        "Annual aggregate OOF prediction counts and daily rank-score quantile bands from 2015 through 2026.",
        body,
        height=710,
    )
    return _write_svg(output, svg)


_PROHIBITED_RESEARCH_WORDS = {
    "account",
    "accounts",
    "credential",
    "credentials",
    "local",
    "modelpath",
    "local_path",
    "model_path",
    "path",
    "paths",
    "position",
    "positions",
    "price",
    "prices",
    "security",
    "securities",
    "symbol",
    "symbols",
    "timestamp",
    "timestamps",
    "token",
    "tokens",
    "trade",
    "trades",
    "ts_code",
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "strategy",
    "evaluation",
    "universe",
    "execution",
    "metrics",
    "nav_series",
    "privacy",
}
_SENSITIVE_RESEARCH_VALUES = (
    re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s\"']+"),
    re.compile(r"(?<![\w])/(?:home|Users)/[^\s\"']+"),
    re.compile(r"(?<!\d)\d{6}(?:\.(?:SH|SZ|BJ))?(?!\d)", re.IGNORECASE),
    re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
)


def _field_words(field: object) -> set[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(field))
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", expanded).casefold()
    return {normalized, *normalized.split("_")}


def _reject_sensitive_research_content(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            prohibited = _field_words(key).intersection(_PROHIBITED_RESEARCH_WORDS)
            if prohibited:
                raise VisualizationError(
                    f"research profile contains prohibited field: {sorted(prohibited)[0]}"
                )
            _reject_sensitive_research_content(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            _reject_sensitive_research_content(child)
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in _SENSITIVE_RESEARCH_VALUES
    ):
        raise VisualizationError("research profile contains a sensitive value")


def _require_exact_fields(value: Mapping[str, Any], context: str, required: set[str]) -> None:
    fields = set(value)
    missing = sorted(required - fields)
    unknown = sorted(fields - required)
    if missing:
        raise VisualizationError(f"{context} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise VisualizationError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _required_mapping(parent: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise VisualizationError(f"research profile {field} must be a mapping")
    return value


def _strict_integer(parent: Mapping[str, Any], field: str, context: str) -> int:
    value = parent.get(field)
    if type(value) is not int:
        raise VisualizationError(f"{context} {field} must be an integer")
    return value


def _strict_number(parent: Mapping[str, Any], field: str, context: str) -> float:
    if field not in parent:
        raise VisualizationError(f"{context} is missing required field: {field}")
    raw = parent[field]
    if type(raw) not in (int, float) or not math.isfinite(raw):
        raise VisualizationError(f"{context} {field} must be a finite number")
    return float(raw)


def _strict_date(raw: object, context: str) -> date:
    if not isinstance(raw, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw) is None:
        raise VisualizationError(f"{context} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(raw)
    except ValueError as error:
        raise VisualizationError(f"{context} must use a valid YYYY-MM-DD date") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise VisualizationError(f"research profile contains duplicate JSON key: {key}")
        value[key] = child
    return value


def load_research_oos_profile(path: str | Path) -> Mapping[str, Any]:
    """Load and validate a research profile without accepting duplicate JSON keys."""

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except VisualizationError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise VisualizationError(f"unable to load research profile: {source.name}") from error
    if not isinstance(payload, Mapping):
        raise VisualizationError("research profile JSON must contain an object")
    _validated_research_profile(payload)
    return payload


def _validated_research_profile(
    profile: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, float], list[tuple[int, date, float]], int]:
    if not isinstance(profile, Mapping):
        raise VisualizationError("research profile must be a mapping")
    _reject_sensitive_research_content(profile)
    _require_exact_fields(profile, "research profile", _TOP_LEVEL_FIELDS)
    if _strict_integer(profile, "schema_version", "research profile") != 1:
        raise VisualizationError("research profile schema_version must be 1")

    strategy = _required_mapping(profile, "strategy")
    evaluation = _required_mapping(profile, "evaluation")
    coverage = _required_mapping(evaluation, "coverage")
    universe = _required_mapping(profile, "universe")
    execution = _required_mapping(profile, "execution")
    metrics_raw = _required_mapping(profile, "metrics")
    series = _required_mapping(profile, "nav_series")
    privacy = _required_mapping(profile, "privacy")
    _require_exact_fields(
        strategy,
        "research profile strategy",
        {"display_name", "model_family", "selection_basis", "components"},
    )
    _require_exact_fields(evaluation, "research profile evaluation", {"method", "coverage"})
    _require_exact_fields(coverage, "research profile coverage", {"start", "end", "trading_days"})
    _require_exact_fields(
        universe,
        "research profile universe",
        {
            "minimum_circulating_market_value_cny",
            "minimum_daily_amount_cny",
            "maximum_holdings_per_sector",
        },
    )
    _require_exact_fields(
        execution,
        "research profile execution",
        {
            "top_n",
            "signal_timing",
            "slippage_rate",
            "commission_rate",
            "stamp_tax_rate",
        },
    )
    _require_exact_fields(
        metrics_raw,
        "research profile metrics",
        {"total_return", "cagr", "max_drawdown", "sharpe"},
    )
    _require_exact_fields(
        series,
        "research profile nav_series",
        {
            "kind",
            "checkpoint_stride_trading_days",
            "checkpoint_count",
            "checkpoints",
        },
    )
    _require_exact_fields(privacy, "research profile privacy", {"level", "omitted"})

    display_name = str(strategy.get("display_name", "")).strip()
    method = str(evaluation.get("method", "")).strip()
    if display_name != "TensorAlpha STK-O" or method != "Purged K-Fold OOS":
        raise VisualizationError("research profile requires the documented STK-O evaluation")
    if strategy.get("model_family") != "V4.6 Transformer stacking":
        raise VisualizationError("research profile model family is inconsistent")
    if (
        not isinstance(strategy.get("selection_basis"), str)
        or not strategy["selection_basis"].strip()
    ):
        raise VisualizationError("research profile selection basis is required")
    components = strategy.get("components")
    expected_components = [("ME", 0.4), ("CE_Liq", 0.2), ("V46ME S43", 0.4)]
    if not isinstance(components, list) or len(components) != len(expected_components):
        raise VisualizationError("research profile requires the documented STK-O components")
    parsed_components: list[tuple[object, float]] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise VisualizationError(f"research profile component {index} must be a mapping")
        _require_exact_fields(component, f"research profile component {index}", {"name", "weight"})
        parsed_components.append(
            (
                component.get("name"),
                _strict_number(component, "weight", f"research profile component {index}"),
            )
        )
    if parsed_components != expected_components:
        raise VisualizationError("research profile STK-O components or weights are inconsistent")

    start = _strict_date(coverage.get("start"), "research profile coverage start")
    end = _strict_date(coverage.get("end"), "research profile coverage end")
    trading_days = _strict_integer(coverage, "trading_days", "research profile coverage")
    stride = _strict_integer(
        series, "checkpoint_stride_trading_days", "research profile nav_series"
    )
    declared_count = _strict_integer(series, "checkpoint_count", "research profile nav_series")
    if (start.isoformat(), end.isoformat(), trading_days, stride, declared_count) != (
        "2015-04-07",
        "2026-03-16",
        2_659,
        50,
        55,
    ):
        raise VisualizationError("research profile coverage or checkpoint contract is inconsistent")

    if (
        _strict_integer(
            universe,
            "minimum_circulating_market_value_cny",
            "research profile universe",
        ),
        _strict_integer(universe, "minimum_daily_amount_cny", "research profile universe"),
        _strict_integer(universe, "maximum_holdings_per_sector", "research profile universe"),
    ) != (3_000_000_000, 50_000_000, 2):
        raise VisualizationError("research profile universe definition is inconsistent")
    if _strict_integer(execution, "top_n", "research profile execution") != 10:
        raise VisualizationError("research profile execution top_n is inconsistent")
    if execution.get("signal_timing") != "T close signal, T+1 open execution":
        raise VisualizationError("research profile signal timing is inconsistent")
    expected_costs = {"slippage_rate": 0.002, "commission_rate": 0.0003, "stamp_tax_rate": 0.0005}
    for field, expected in expected_costs.items():
        actual = _strict_number(execution, field, "research profile execution")
        if not math.isclose(actual, expected, abs_tol=1e-12):
            raise VisualizationError(f"research profile execution {field} is inconsistent")
    if privacy.get("level") != "aggregate checkpoints only":
        raise VisualizationError("research profile privacy level is inconsistent")
    omitted = privacy.get("omitted")
    if (
        not isinstance(omitted, list)
        or not omitted
        or not all(isinstance(item, str) and item.strip() for item in omitted)
    ):
        raise VisualizationError("research profile privacy omissions are required")

    total_return = _strict_number(metrics_raw, "total_return", "research profile metrics")
    cagr = _strict_number(metrics_raw, "cagr", "research profile metrics")
    max_drawdown = _strict_number(metrics_raw, "max_drawdown", "research profile metrics")
    sharpe = _strict_number(metrics_raw, "sharpe", "research profile metrics")
    authoritative_metrics = {
        "total_return": 3.2436752757,
        "cagr": 0.1468131762,
        "max_drawdown": -0.5587568167,
        "sharpe": 0.5894039057,
    }
    actual_metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }
    if any(
        not math.isclose(actual_metrics[field], expected, abs_tol=5e-10)
        for field, expected in authoritative_metrics.items()
    ):
        raise VisualizationError("research profile metrics do not match the authoritative run")
    if total_return <= -1 or cagr <= -1 or not -1 <= max_drawdown <= 0:
        raise VisualizationError("research profile metrics are outside valid ranges")

    if series.get("kind") != "authoritative_logged_checkpoints":
        raise VisualizationError("research profile requires authoritative logged checkpoints")

    raw_checkpoints = series.get("checkpoints")
    if not isinstance(raw_checkpoints, Sequence) or isinstance(raw_checkpoints, (str, bytes)):
        raise VisualizationError("research profile checkpoints must be a sequence")
    if len(raw_checkpoints) != declared_count:
        raise VisualizationError("research profile checkpoint count is inconsistent")
    checkpoints: list[tuple[int, date, float]] = []
    for index, raw in enumerate(raw_checkpoints):
        if not isinstance(raw, Mapping):
            raise VisualizationError(f"research checkpoint {index} must be a mapping")
        _require_exact_fields(
            raw, f"research checkpoint {index}", {"day", "date", "normalized_nav"}
        )
        day = _strict_integer(raw, "day", f"research checkpoint {index}")
        nav = _strict_number(raw, "normalized_nav", f"research checkpoint {index}")
        checkpoint_date = _strict_date(raw.get("date"), f"research checkpoint {index} date")
        if nav <= 0:
            raise VisualizationError(f"research checkpoint {index} NAV must be positive")
        checkpoints.append((day, checkpoint_date, nav))

    days = [row[0] for row in checkpoints]
    dates = [row[1] for row in checkpoints]
    expected_days = [1, *range(stride, trading_days, stride)]
    if expected_days[-1] != trading_days:
        expected_days.append(trading_days)
    if days != expected_days or dates != sorted(dates) or len(set(dates)) != len(dates):
        raise VisualizationError("research profile checkpoints must follow the declared stride")
    if dates[0] != start or dates[-1] != end:
        raise VisualizationError("research profile checkpoints must match coverage dates")
    if not math.isclose(checkpoints[0][2], 1.0, abs_tol=1e-12):
        raise VisualizationError("research profile must start at normalized NAV 1.0")
    if not math.isclose(checkpoints[-1][2] - 1.0, total_return, abs_tol=1e-7):
        raise VisualizationError("research profile final NAV is inconsistent with total return")

    labels = {
        "display_name": display_name,
        "method": method,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }
    metrics = {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }
    return labels, metrics, checkpoints, stride


def render_research_oos_profile_svg(profile: Mapping[str, Any], output: str | Path) -> Path:
    """Render a validated aggregate OOS strategy profile to a stable SVG."""

    labels, metrics, checkpoints, stride = _validated_research_profile(profile)
    trading_days = checkpoints[-1][0]
    left, right = 92.0, 1144.0
    chart_top, chart_bottom = 210.0, 492.0
    width = right - left
    nav_values = [row[2] for row in checkpoints]
    nav_low = max(0.0, math.floor(min(nav_values) * 2.0) / 2.0 - 0.5)
    nav_high = math.ceil(max(nav_values) * 2.0) / 2.0
    if nav_high - nav_low < 1.0:
        nav_high = nav_low + 1.0

    def x_day(day: int) -> float:
        return left + (day - 1) / (trading_days - 1) * width

    def y_nav(nav: float) -> float:
        return chart_bottom - (nav - nav_low) / (nav_high - nav_low) * (chart_bottom - chart_top)

    chart_name = f"TensorAlpha · {labels['display_name'].removeprefix('TensorAlpha ')}"
    body = [
        _text(58, 52, f"{chart_name} 11-Year OOS Profile", css="title"),
        _text(
            58,
            78,
            f"{labels['method']} · {labels['start']} to {labels['end']} · "
            f"{stride}-trading-day checkpoints · After modeled costs",
            css="subtitle",
        ),
    ]
    kpis = [
        ("Cumulative return", f"{metrics['total_return']:+.2%}"),
        ("CAGR", f"{metrics['cagr']:+.2%}"),
        ("Full-run max drawdown", f"{metrics['max_drawdown']:.2%}"),
        ("Sharpe", f"{metrics['sharpe']:.4f}"),
    ]
    for index, (label, value) in enumerate(kpis):
        x = 58.0 + index * 273.0
        body.extend(
            [
                f'<rect x="{x:.1f}" y="100" width="250" height="58" rx="6" '
                f'fill="#FFFFFF" stroke="{_GRID}"/>',
                _text(x + 14, 122, label, css="muted"),
                _text(x + 14, 147, value, css="panel-title"),
            ]
        )
    body.extend(
        [
            _text(58, 184, "Historical research · Not live performance", css="panel-title"),
            _text(1144, 184, "Normalized NAV", css="muted", anchor="end"),
        ]
    )

    tick_values = [nav_low + index * (nav_high - nav_low) / 4.0 for index in range(5)]
    for value in tick_values:
        y = y_nav(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        body.append(_text(left - 12, y + 4, f"{value:.2f}", css="muted", anchor="end"))
    if nav_low <= 1.0 <= nav_high:
        body.append(
            f'<line x1="{left}" y1="{y_nav(1.0):.1f}" x2="{right}" '
            f'y2="{y_nav(1.0):.1f}" stroke="{_MUTED}" stroke-dasharray="5 4"/>'
        )
    nav_points = [(x_day(day), y_nav(nav)) for day, _, nav in checkpoints]
    body.append(
        f'<polyline points="{_points(nav_points)}" fill="none" stroke="{_BLUE}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    tick_indices = sorted({round(index * (len(checkpoints) - 1) / 5) for index in range(6)})
    for index in tick_indices:
        day, date, _ = checkpoints[index]
        x = x_day(day)
        body.append(_text(x, 516, date.strftime("%Y"), css="muted", anchor="middle"))
    body.extend(
        [
            _text(
                58,
                552,
                f"{len(checkpoints)} exact logged NAV checkpoints · No interpolation · "
                "No security-level records",
                css="muted",
            ),
            _text(
                58,
                574,
                "Full-run max drawdown is from the original daily run, not the checkpoint series.",
                css="muted",
            ),
            _text(
                58,
                596,
                "Historical research only · Modeled execution costs · Not investment advice",
                css="muted",
            ),
        ]
    )
    svg = _base_svg(
        f"{chart_name} 11-Year OOS Profile",
        "Normalized NAV checkpoints and full-run aggregate metrics from the documented Purged "
        "K-Fold OOS research evaluation.",
        body,
        height=620,
    )
    return _write_svg(output, svg)


def _validated_nav(nav: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(nav, pd.DataFrame) or not {"date", "nav"}.issubset(nav.columns):
        raise VisualizationError("backtest data must contain date and nav columns")
    if len(nav) < 2:
        raise VisualizationError("backtest data must contain at least two observations")
    frame = nav.loc[:, ["date", "nav"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    if frame["date"].isna().any() or not frame["date"].is_monotonic_increasing:
        raise VisualizationError("backtest dates must be valid and increasing")
    values = frame["nav"].to_numpy(dtype=float)
    if not all(math.isfinite(value) for value in values):
        raise VisualizationError("backtest NAV values must be finite")
    if not all(value > 0 for value in values):
        raise VisualizationError("backtest NAV values must be positive")
    return frame


def render_backtest_svg(nav: pd.DataFrame, output: str | Path) -> Path:
    """Render normalized synthetic NAV and drawdown to a stable SVG."""

    frame = _validated_nav(nav)
    normalized = frame["nav"] / float(frame["nav"].iloc[0])
    drawdown = normalized / normalized.cummax() - 1.0
    total_return = float(normalized.iloc[-1] - 1.0)
    max_drawdown = float(drawdown.min())
    start_date = frame["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = frame["date"].iloc[-1].strftime("%Y-%m-%d")
    elapsed_years = round((frame["date"].iloc[-1] - frame["date"].iloc[0]).days / 365.2425)
    period_prefix = f"{elapsed_years}-Year " if elapsed_years >= 2 else ""
    chart_title = f"{period_prefix}Synthetic Backtest"

    left, right = 92.0, 1144.0
    width = right - left
    x_positions = [left + width * index / (len(frame) - 1) for index in range(len(frame))]
    body = [
        _text(58, 52, f"TensorAlpha · {chart_title}", css="title"),
        _text(
            58,
            78,
            f"Illustrative only · {start_date} to {end_date} · Deterministic synthetic data · "
            "Not historical performance",
            css="subtitle",
        ),
        _text(58, 112, f"Total return  {total_return:+.1%}", css="mono"),
        _text(270, 112, f"Max drawdown  {max_drawdown:.1%}", css="mono"),
        _text(58, 150, "Normalized NAV", css="panel-title"),
    ]

    nav_top, nav_bottom = 172.0, 382.0
    nav_min, nav_max = float(normalized.min()), float(normalized.max())
    span = max(nav_max - nav_min, 0.02)
    nav_low = min(nav_min, 1.0) - span * 0.12
    nav_high = max(nav_max, 1.0) + span * 0.12

    def y_nav(value: float) -> float:
        return nav_bottom - (value - nav_low) / (nav_high - nav_low) * (nav_bottom - nav_top)

    for fraction in (0.0, 0.5, 1.0):
        value = nav_low + fraction * (nav_high - nav_low)
        y = y_nav(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        body.append(_text(left - 12, y + 4, f"{value:.2f}", css="muted", anchor="end"))
    body.append(
        f'<line x1="{left}" y1="{y_nav(1.0):.1f}" x2="{right}" y2="{y_nav(1.0):.1f}" '
        f'stroke="{_MUTED}" stroke-dasharray="5 4"/>'
    )
    body.append(
        f'<polyline points="{_points(list(zip(x_positions, map(y_nav, normalized), strict=True)))}" '
        f'fill="none" stroke="{_BLUE}" stroke-width="2.6"/>'
    )

    dd_top, dd_bottom = 468.0, 610.0
    dd_low = min(max_drawdown * 1.15, -0.01)

    def y_dd(value: float) -> float:
        return dd_bottom - (value - dd_low) / (0.0 - dd_low) * (dd_bottom - dd_top)

    body.append(_text(58, 444, "Drawdown", css="panel-title"))
    for fraction in (0.0, 0.5, 1.0):
        value = dd_low * (1.0 - fraction)
        y = y_dd(value)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        body.append(_text(left - 12, y + 4, f"{value:.1%}", css="muted", anchor="end"))
    dd_points = list(zip(x_positions, map(y_dd, drawdown), strict=True))
    area_points = [(x_positions[0], y_dd(0.0)), *dd_points, (x_positions[-1], y_dd(0.0))]
    body.append(f'<polygon points="{_points(area_points)}" fill="{_ORANGE_LIGHT}" stroke="none"/>')
    body.append(
        f'<polyline points="{_points(dd_points)}" fill="none" stroke="{_ORANGE}" stroke-width="2.2"/>'
    )

    tick_count = min(6, len(frame))
    tick_indices = sorted(
        {round(index * (len(frame) - 1) / (tick_count - 1)) for index in range(tick_count)}
    )
    for index in tick_indices:
        x = x_positions[index]
        label = frame["date"].iloc[index].strftime("%Y-%m-%d")
        body.append(_text(x, 636, label, css="muted", anchor="middle"))
    body.append(
        _text(
            58,
            664,
            "Source: TensorAlpha deterministic demo · Synthetic data only · Research software, not investment advice",
            css="muted",
        )
    )
    svg = _base_svg(
        f"TensorAlpha {chart_title}",
        f"Normalized synthetic NAV and drawdown from {start_date} through {end_date}.",
        body,
        height=690,
    )
    return _write_svg(output, svg)
