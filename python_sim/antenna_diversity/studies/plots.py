"""Deterministic figures rendered exclusively from normalized study tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


FIGURE_KEYS = (
    "baseline_rankings", "direction_heatmaps", "power_curves", "robustness_curves",
    "combined_error_distributions", "multi_jammer_comparison", "full_chain_validation",
    "cn0_validation",
)
_MODES = ("single", "egc", "mvdr")
_REQUIRED_TABLES = (
    "baseline_rankings", "direction_scan", "power_scan", "robustness_scans",
    "combined_errors", "multi_jammer", "full_chain_validation", "full_chain_cn0",
)


def plot_study_tables(tables: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, Figure]:
    """Build the eight fixed research figures from saved table rows only."""

    _require_tables(tables)
    return {
        "baseline_rankings": _baseline(tables["baseline_rankings"]),
        "direction_heatmaps": _directions(tables["direction_scan"]),
        "power_curves": _power(tables["power_scan"]),
        "robustness_curves": _robustness(tables["robustness_scans"]),
        "combined_error_distributions": _combined(tables["combined_errors"]),
        "multi_jammer_comparison": _multi_jammer(tables["multi_jammer"]),
        "full_chain_validation": _full_chain(tables["full_chain_validation"]),
        "cn0_validation": _cn0(tables["full_chain_cn0"], tables["full_chain_validation"]),
    }


def save_study_figures(figures: Mapping[str, Figure], output_dir: Path) -> dict[str, Path]:
    """Save fixed-name PNG evidence at 150 dpi in the supplied directory."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key in FIGURE_KEYS:
        if key not in figures:
            raise ValueError(f"missing required figure: {key}")
        path = directory / f"{key}.png"
        _save_figure_atomic(figures[key], path)
        paths[key] = path
    return paths


def _save_figure_atomic(figure: Figure, destination: Path) -> None:
    """Render one PNG to a same-directory temporary file before publishing it."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
        figure.savefig(temporary, dpi=150, format="png")
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _require_tables(tables: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    for name in _REQUIRED_TABLES:
        if name not in tables:
            raise ValueError(f"missing required table: {name}")


def _combination(row: Mapping[str, object]) -> str:
    raw = row.get("active_antennas", ())
    return "+".join(str(value) for value in raw) if isinstance(raw, Sequence) and not isinstance(raw, str) else str(raw)


def _finite(value: object) -> float | None:
    try:
        candidate = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return candidate if np.isfinite(candidate) else None


def _groups(rows: Sequence[Mapping[str, object]], *fields: str) -> dict[tuple[str, ...], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(_combination(row) if field == "combination" else str(row.get(field, "")) for field in fields)].append(row)
    return dict(sorted(grouped.items()))


def _empty_axis(axis: Any, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")


def _baseline(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ordered = _ranked_rows(rows)
    finite_rows = [row for row in ordered if _finite(row.get("p10_resolvable_db")) is not None and _finite(row.get("median_resolvable_db")) is not None]
    labels = [str(row.get("display_label", _combination(row))) for row in finite_rows]
    p10 = [_finite(row.get("p10_resolvable_db")) for row in finite_rows]
    median = [_finite(row.get("median_resolvable_db")) for row in finite_rows]
    if labels:
        y = np.arange(len(labels))
        axis.barh(y - 0.18, p10, 0.35, label="Resolvable p10")
        axis.barh(y + 0.18, median, 0.35, label="Resolvable median")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.legend()
    else:
        _empty_axis(axis, "No baseline rows")
    axis.set(xlabel="Output SINR (dB)", title="Baseline rankings")
    return figure


def _directions(rows: Sequence[Mapping[str, object]]) -> Figure:
    _validate_direction_targets(rows)
    grouped = _groups(rows, "combination", "mode")
    panels = list(grouped.items()) or [((), [])]
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False, layout="constrained")
    finite_values = [_finite(row.get("output_sinr_db")) for row in rows]
    values = [value for value in finite_values if value is not None]
    lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    for axis, (key, group) in zip(axes.flat, panels, strict=True):
        cells = {( _finite(row.get("elevation_deg")), _finite(row.get("azimuth_deg"))): _finite(row.get("output_sinr_db")) for row in group}
        azimuths = sorted({az for _, az in cells if az is not None})
        elevations = sorted({el for el, _ in cells if el is not None})
        if azimuths and elevations:
            data = np.array([[cells.get((el, az), np.nan) for az in azimuths] for el in elevations], dtype=float)
            axis.imshow(data, origin="lower", aspect="auto", vmin=lo, vmax=hi,
                        extent=(min(azimuths), max(azimuths), min(elevations), max(elevations)))
            reference = group[0]
            target_az = _required_finite(reference, "target_azimuth_deg")
            target_el = _required_finite(reference, "target_elevation_deg")
            axis.plot(target_az, target_el, "w*", markeredgecolor="black", label="Target")
            boundary = plt.Circle((target_az, target_el), 15.0, fill=False, color="white", linestyle="--", label="15° boundary")
            axis.add_patch(boundary)
            axis.legend(loc="best")
        else:
            _empty_axis(axis, "No finite direction rows")
        axis.set(title=" ".join(key) or "Direction scan", xlabel="Azimuth (deg)", ylabel="Elevation (deg)")
    return figure


def _power(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for (combination, mode), group in _groups(rows, "combination", "mode").items():
        points = _finite_points(group, "power_dbm", "output_sinr_db")
        if points:
            axis.plot(*zip(*points), marker="o", label=f"{mode.upper()}({combination})")
    if axis.lines:
        axis.legend()
    else:
        _empty_axis(axis, "No finite power rows")
    axis.set(xlabel="Interferer power (dBm)", ylabel="Output SINR (dB)", title="Power sweep")
    return figure


def _robustness(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for (combination, factor), group in _groups(rows, "combination", "factor").items():
        points = _finite_points(group, "normalized_severity", "mvdr_minus_egc_sinr_db", fallback_x="absolute_normalized_severity")
        if points:
            axis.plot(*zip(*points), marker="o", label=f"Δ MVDR−EGC {combination} {factor}")
    if axis.lines:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.legend()
    else:
        _empty_axis(axis, "No finite robustness rows")
    axis.set(xlabel="Normalized severity", ylabel="SINR delta (dB)", title="Robustness curves")
    return figure


def _combined(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    grouped = _groups(rows, "level")
    labels, data = [], []
    for (level,), group in grouped.items():
        values = [_finite(row.get("output_sinr_db", row.get("mvdr_output_sinr_db"))) for row in group]
        finite = [value for value in values if value is not None]
        if finite:
            labels.append(level)
            data.append(finite)
    if data:
        if all(len(set(values)) == 1 for values in data):
            axis.plot(range(1, len(data) + 1), [values[0] for values in data], "o")
        else:
            axis.boxplot(data, tick_labels=labels)
    else:
        _empty_axis(axis, "No finite combined-error rows")
    axis.set(xlabel="Combined-error level", ylabel="Output SINR (dB)", title="Combined error distributions")
    return figure


def _multi_jammer(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ordered = [row for row in sorted(rows, key=lambda row: (str(row.get("template", "")), _combination(row), str(row.get("mode", "")))) if _finite(row.get("output_sinr_db")) is not None]
    labels = [f"{row.get('template', '')}\n{_combination(row)}" for row in ordered]
    values = [_finite(row.get("output_sinr_db")) for row in ordered]
    if labels:
        axis.bar(np.arange(len(labels)), values)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    else:
        _empty_axis(axis, "No multi-jammer rows")
    axis.set(ylabel="Output SINR (dB)", title="Multi-jammer comparison")
    return figure


def _full_chain(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), layout="constrained")
    modes = [mode for mode in _MODES if any(str(row.get("mode")) == mode for row in rows)]
    if modes:
        acquired = [np.mean([bool(row.get("acquired")) for row in rows if row.get("mode") == mode]) for mode in modes]
        tracking = [np.mean([bool(row.get("tracking")) for row in rows if row.get("mode") == mode]) for mode in modes]
        x = np.arange(len(modes))
        axes[0].bar(x - 0.2, acquired, 0.4, label="Acquired")
        axes[0].bar(x + 0.2, tracking, 0.4, label="Tracking")
        axes[0].set_xticks(x, [mode.upper() for mode in modes])
        axes[0].legend()
        sinr, scalar_modes = [], []
        for mode in modes:
            values = [_finite(row.get("analytic_output_sinr_db")) for row in rows if row.get("mode") == mode]
            finite = [value for value in values if value is not None]
            if finite:
                scalar_modes.append(mode)
                sinr.append(float(np.mean(finite)))
        if sinr:
            scalar_x = np.arange(len(scalar_modes))
            axes[1].bar(scalar_x, sinr)
            axes[1].set_xticks(scalar_x, [mode.upper() for mode in scalar_modes])
        else:
            _empty_axis(axes[1], "No finite analytic SINR rows")
    else:
        _empty_axis(axes[0], "No full-chain rows")
        _empty_axis(axes[1], "No scalar validation rows")
    axes[0].set(ylabel="Rate", ylim=(0, 1.05), title="Capture and tracking rates")
    axes[1].set(ylabel="analytic_output_sinr_db (dB)", title="Analytic validation")
    return figure


def _cn0(rows: Sequence[Mapping[str, object]], validation_rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for (case_id, mode, run), group in _groups(rows, "case_id", "mode", "run_index").items():
        points = _finite_points(group, "time_ms", "cn0_db_hz")
        if points:
            axis.plot(*zip(*points), label=f"{mode.upper()} {case_id} run {run}")
    if axis.lines:
        axis.legend()
    else:
        unavailable = sorted({str(row.get("mode", "")) for row in validation_rows if not bool(row.get("tracking"))})
        _empty_axis(axis, "C/N₀ unavailable: " + (", ".join(unavailable) or "no finite epoch rows"))
    axis.set(xlabel="Time (ms)", ylabel="C/N₀ (dB-Hz)", title="C/N₀ validation")
    return figure


def _ranked_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    ranks: list[int] = []
    for row in rows:
        rank = row.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("baseline ranking requires a positive integer rank")
        ranks.append(rank)
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        raise ValueError("baseline ranking ranks must be contiguous one-based integers")
    return sorted(rows, key=lambda row: int(row["rank"]))


def _finite_points(
    rows: Sequence[Mapping[str, object]], x_field: str, y_field: str, *, fallback_x: str | None = None,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = _finite(row.get(x_field, row.get(fallback_x) if fallback_x else None))
        y = _finite(row.get(y_field))
        if x is not None and y is not None:
            points.append((x, y))
    return sorted(points)


def _required_finite(row: Mapping[str, object], field: str) -> float:
    value = _finite(row.get(field))
    if value is None:
        raise ValueError(f"direction row requires finite {field}")
    return value


def _validate_direction_targets(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        _required_finite(row, "target_azimuth_deg")
        _required_finite(row, "target_elevation_deg")
