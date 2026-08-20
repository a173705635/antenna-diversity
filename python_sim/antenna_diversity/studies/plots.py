"""Deterministic figures rendered exclusively from normalized study tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.image import NonUniformImage
import numpy as np

from antenna_diversity.array.geometry import CORNER_ORDER


FIGURE_KEYS = (
    "baseline_rankings", "direction_heatmaps", "power_curves", "robustness_curves",
    "combined_error_distributions", "multi_jammer_comparison", "full_chain_validation",
    "cn0_validation",
    "element_patterns", "polarization_losses", "coupling_matrices",
    "blockage_comparison",
)
_MODES = ("single", "egc", "mvdr")
_REQUIRED_TABLES = (
    "baseline_rankings", "direction_scan", "power_scan", "robustness_scans",
    "combined_errors", "multi_jammer", "full_chain_validation", "full_chain_cn0",
    "antenna_diagnostics",
)


def plot_study_tables(tables: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, Figure]:
    """Build the twelve fixed research figures from saved table rows only."""

    _require_tables(tables)
    figures: dict[str, Figure] = {}
    try:
        figures["baseline_rankings"] = _baseline(tables["baseline_rankings"])
        figures["direction_heatmaps"] = _directions(tables["direction_scan"])
        figures["power_curves"] = _power(tables["power_scan"])
        figures["robustness_curves"] = _robustness(tables["robustness_scans"])
        figures["combined_error_distributions"] = _combined(tables["combined_errors"])
        figures["multi_jammer_comparison"] = _multi_jammer(tables["multi_jammer"])
        figures["full_chain_validation"] = _full_chain(tables["full_chain_validation"])
        figures["cn0_validation"] = _cn0(
            tables["full_chain_cn0"], tables["full_chain_validation"]
        )
        diagnostics = tables["antenna_diagnostics"]
        figures["element_patterns"] = _diagnostic_heatmaps(
            diagnostics, "pattern_grid", "response_db", "Element pattern response",
            "Response (dB)",
        )
        figures["polarization_losses"] = _polarization_losses(diagnostics)
        figures["coupling_matrices"] = _coupling_matrices(diagnostics)
        figures["blockage_comparison"] = _blockage_comparison(diagnostics)
    except BaseException:
        for figure in figures.values():
            plt.close(figure)
        raise
    return figures


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


@contextmanager
def _close_on_error(figure: Figure) -> Iterator[None]:
    try:
        yield
    except BaseException:
        plt.close(figure)
        raise


def _cell_edges(
    coordinates: Sequence[float], *, lower: float | None = None,
    upper: float | None = None,
) -> tuple[float, float]:
    if len(coordinates) == 1:
        left, right = coordinates[0] - 0.5, coordinates[0] + 0.5
    else:
        left = coordinates[0] - (coordinates[1] - coordinates[0]) / 2.0
        right = coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2.0
    if lower is not None:
        left = max(lower, left)
    if upper is not None:
        right = min(upper, right)
    return left, right


def _baseline(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5), layout="constrained")
    with _close_on_error(figure):
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
    with _close_on_error(figure):
        finite_values = [_finite(row.get("output_sinr_db")) for row in rows]
        values = [value for value in finite_values if value is not None]
        lo, hi = (min(values), max(values)) if values else (0.0, 1.0)
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        for axis, (key, group) in zip(axes.flat, panels, strict=True):
            cells: dict[tuple[float, float], float | None] = {}
            for row in group:
                elevation = _required_finite(row, "elevation_deg")
                azimuth = _required_finite(row, "azimuth_deg")
                coordinate = (elevation, azimuth)
                if coordinate in cells:
                    raise ValueError(
                        "duplicate direction coordinate: "
                        f"azimuth_deg={azimuth}, elevation_deg={elevation}"
                    )
                cells[coordinate] = _finite(row.get("output_sinr_db"))
            azimuths = sorted({azimuth for _, azimuth in cells})
            elevations = sorted({elevation for elevation, _ in cells})
            if azimuths and elevations:
                data = np.array([[cells.get((el, az), np.nan) for az in azimuths] for el in elevations], dtype=float)
                azimuth_extent = _cell_edges(azimuths)
                elevation_extent = _cell_edges(elevations, lower=0.0, upper=90.0)
                image = NonUniformImage(
                    axis, interpolation="nearest",
                    extent=(*azimuth_extent, *elevation_extent),
                )
                image.set_clim(lo, hi)
                image.set_data(azimuths, elevations, data)
                axis.add_image(image)
                reference = group[0]
                target_az = _required_finite(reference, "target_azimuth_deg")
                target_el = _required_finite(reference, "target_elevation_deg")
                axis.plot(target_az, target_el, "w*", markeredgecolor="black", label="Target")
                boundary = plt.Circle((target_az, target_el), 15.0, fill=False, color="white", linestyle="--", label="15° boundary")
                axis.add_patch(boundary)
                axis.legend(loc="best")
                axis.set_xlim(*azimuth_extent)
                axis.set_ylim(*elevation_extent)
            else:
                _empty_axis(axis, "No finite direction rows")
            axis.set(title=" ".join(key) or "Direction scan", xlabel="Azimuth (deg)", ylabel="Elevation (deg)")
    return figure


def _power(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    with _close_on_error(figure):
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
    with _close_on_error(figure):
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
    with _close_on_error(figure):
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
    with _close_on_error(figure):
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
    with _close_on_error(figure):
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
    with _close_on_error(figure):
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


def _diagnostic_heatmaps(
    rows: Sequence[Mapping[str, object]],
    record_kind: str,
    value_field: str,
    title: str,
    colorbar_label: str,
) -> Figure:
    selected = [row for row in rows if row.get("record_kind") == record_kind]
    views = ("true", "assumed")
    figure, axes = plt.subplots(
        len(views), len(CORNER_ORDER), figsize=(4 * len(CORNER_ORDER), 7.5),
        squeeze=False, layout="constrained",
    )
    with _close_on_error(figure):
        values = [
            value for row in selected
            if (value := _finite(row.get(value_field))) is not None
        ]
        lo, hi = _finite_limits(values)
        images = []
        for row_index, view in enumerate(views):
            for column_index, element_name in enumerate(CORNER_ORDER):
                axis = axes[row_index, column_index]
                group = [
                    row for row in selected
                    if row.get("model_view") == view
                    and row.get("element_name") == element_name
                ]
                image = _saved_grid_image(axis, group, value_field, lo, hi)
                if image is None:
                    _empty_axis(axis, "No finite saved rows")
                else:
                    images.append(image)
                axis.set(
                    title=f"{view.title()} {element_name}",
                    xlabel="Azimuth (deg)", ylabel="Elevation (deg)",
                )
        if images:
            figure.colorbar(
                images[0], ax=axes.ravel().tolist(), shrink=0.82,
                label=colorbar_label,
            )
        figure.suptitle(title)
    return figure


def _saved_grid_image(
    axis: Any,
    rows: Sequence[Mapping[str, object]],
    value_field: str,
    lo: float,
    hi: float,
):
    cells: dict[tuple[float, float], float] = {}
    for row in rows:
        azimuth = _finite(row.get("azimuth_deg"))
        elevation = _finite(row.get("elevation_deg"))
        value = _finite(row.get(value_field))
        if azimuth is None or elevation is None or value is None:
            continue
        coordinate = (elevation, azimuth)
        if coordinate in cells:
            raise ValueError(
                f"duplicate antenna diagnostic coordinate: {coordinate}"
            )
        cells[coordinate] = value
    azimuths = sorted({azimuth for _, azimuth in cells})
    elevations = sorted({elevation for elevation, _ in cells})
    if not azimuths or not elevations:
        return None
    data = np.array([
        [cells.get((elevation, azimuth), np.nan) for azimuth in azimuths]
        for elevation in elevations
    ], dtype=float)
    image = NonUniformImage(
        axis,
        interpolation="nearest",
        extent=(*_cell_edges(azimuths), *_cell_edges(elevations, lower=0.0, upper=90.0)),
    )
    image.set_clim(lo, hi)
    image.set_data(azimuths, elevations, data)
    axis.add_image(image)
    axis.set_xlim(*_cell_edges(azimuths))
    axis.set_ylim(*_cell_edges(elevations, lower=0.0, upper=90.0))
    return image


def _polarization_losses(rows: Sequence[Mapping[str, object]]) -> Figure:
    figure, axis = plt.subplots(figsize=(11, 5), layout="constrained")
    with _close_on_error(figure):
        selected = [
            row for row in rows
            if row.get("record_kind") == "source_match"
            and _finite(row.get("polarization_match_db")) is not None
            and row.get("element_name") in CORNER_ORDER
            and isinstance(row.get("source_name"), str)
        ]
        labels = sorted({
            (str(row["source_name"]), str(row["element_name"]))
            for row in selected
        })
        views = ("true", "assumed")
        if labels:
            x = np.arange(len(labels), dtype=float)
            width = 0.38
            for offset, view in zip((-width / 2.0, width / 2.0), views, strict=True):
                lookup = {
                    (str(row["source_name"]), str(row["element_name"])):
                    _finite(row.get("polarization_match_db"))
                    for row in selected if row.get("model_view") == view
                }
                positions, values = [], []
                for index, label in enumerate(labels):
                    value = lookup.get(label)
                    if value is not None:
                        positions.append(x[index] + offset)
                        values.append(value)
                if values:
                    axis.bar(positions, values, width, label=view.title())
            axis.set_xticks(x, [f"{source}\n{element}" for source, element in labels])
            if axis.patches:
                axis.legend()
        else:
            _empty_axis(axis, "No finite saved polarization rows")
        axis.set(
            ylabel="Polarization match (dB)",
            title="Saved source polarization match",
        )
    return figure


def _blockage_comparison(
    rows: Sequence[Mapping[str, object]],
) -> Figure:
    views = ("true", "assumed")
    states = (("unblocked", "pattern_grid"), ("blocked", "blockage_grid"))
    selected = [
        row for row in rows
        if row.get("record_kind") in {record_kind for _, record_kind in states}
    ]
    figure, axes = plt.subplots(
        len(views) * len(states), len(CORNER_ORDER),
        figsize=(4 * len(CORNER_ORDER), 13), squeeze=False, layout="constrained",
    )
    with _close_on_error(figure):
        values = [
            value for row in selected
            if (value := _finite(row.get("response_db"))) is not None
        ]
        lo, hi = _finite_limits(values)
        images = []
        for view_index, view in enumerate(views):
            for state_index, (state_label, record_kind) in enumerate(states):
                row_index = view_index * len(states) + state_index
                for column_index, element_name in enumerate(CORNER_ORDER):
                    axis = axes[row_index, column_index]
                    group = [
                        row for row in selected
                        if row.get("record_kind") == record_kind
                        and row.get("model_view") == view
                        and row.get("element_name") == element_name
                    ]
                    image = _saved_grid_image(
                        axis, group, "response_db", lo, hi
                    )
                    if image is None:
                        _empty_axis(axis, "No finite saved rows")
                    else:
                        images.append(image)
                    axis.set(
                        title=f"{view.title()} {element_name} {state_label}",
                        xlabel="Azimuth (deg)", ylabel="Elevation (deg)",
                    )
        if images:
            figure.colorbar(
                images[0], ax=axes.ravel().tolist(), shrink=0.82,
                label="Element response (dB)",
            )
        figure.suptitle("Saved unblocked versus blocked element response")
    return figure


def _coupling_matrices(rows: Sequence[Mapping[str, object]]) -> Figure:
    selected = [row for row in rows if row.get("record_kind") == "coupling_entry"]
    views = ("true", "assumed")
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    with _close_on_error(figure):
        matrices = {
            view: _saved_coupling_matrix(selected, view)
            for view in views
        }
        magnitude_values = [
            value for matrix in matrices.values() for value in matrix[0].ravel()
            if np.isfinite(value)
        ]
        phase_values = [
            value for matrix in matrices.values() for value in matrix[1].ravel()
            if np.isfinite(value)
        ]
        magnitude_limits = _finite_limits(magnitude_values)
        phase_limits = _finite_limits(phase_values)
        magnitude_images = []
        phase_images = []
        for column, view in enumerate(views):
            magnitude, phase = matrices[view]
            magnitude_image = axes[0, column].imshow(
                magnitude, vmin=magnitude_limits[0], vmax=magnitude_limits[1],
                interpolation="nearest",
            )
            phase_image = axes[1, column].imshow(
                phase, vmin=phase_limits[0], vmax=phase_limits[1],
                interpolation="nearest", cmap="twilight",
            )
            magnitude_images.append(magnitude_image)
            phase_images.append(phase_image)
            for row_index, label in enumerate(("Magnitude (dB)", "Phase (deg)")):
                axis = axes[row_index, column]
                axis.set_xticks(range(len(CORNER_ORDER)), CORNER_ORDER)
                axis.set_yticks(range(len(CORNER_ORDER)), CORNER_ORDER)
                axis.set(
                    xlabel="Source", ylabel="Receiver",
                    title=f"{view.title()} {label}",
                )
        figure.colorbar(
            magnitude_images[0], ax=axes[0, :].tolist(), shrink=0.8,
            label="Magnitude (dB)",
        )
        figure.colorbar(
            phase_images[0], ax=axes[1, :].tolist(), shrink=0.8,
            label="Phase (deg)",
        )
        figure.suptitle("Saved mutual-coupling matrices")
    return figure


def _saved_coupling_matrix(
    rows: Sequence[Mapping[str, object]], view: str,
) -> tuple[np.ndarray, np.ndarray]:
    size = len(CORNER_ORDER)
    magnitude = np.full((size, size), np.nan, dtype=float)
    phase = np.full((size, size), np.nan, dtype=float)
    indices = {name: index for index, name in enumerate(CORNER_ORDER)}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("model_view") != view:
            continue
        receiver = row.get("receiver")
        source = row.get("source")
        real = _finite(row.get("response_real"))
        imaginary = _finite(row.get("response_imag"))
        magnitude_db = _finite(row.get("response_db"))
        if (
            receiver not in indices or source not in indices
            or real is None or imaginary is None or magnitude_db is None
        ):
            continue
        key = (str(receiver), str(source))
        if key in seen:
            raise ValueError(f"duplicate coupling diagnostic entry: {key}")
        seen.add(key)
        row_index, column_index = indices[str(receiver)], indices[str(source)]
        magnitude[row_index, column_index] = magnitude_db
        phase[row_index, column_index] = float(np.rad2deg(np.angle(complex(real, imaginary))))
    return magnitude, phase


def _finite_limits(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    lo, hi = min(finite), max(finite)
    return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)


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
