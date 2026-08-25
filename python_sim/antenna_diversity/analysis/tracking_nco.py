"""Reproducible float/fixed Carrier NCO comparison helpers."""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from antenna_diversity.config import SPEED_OF_LIGHT_MPS, SimulationConfig
from antenna_diversity.models import SimulationResults
from antenna_diversity.pipeline import run_end_to_end


L1_CARRIER_HZ = 1_575.42e6
CSV_COLUMNS = (
    "nco_case",
    "branch",
    "mode",
    "word_bits",
    "clock_hz",
    "frequency_lsb_hz",
    "l1_velocity_lsb_mm_s",
    "final_fcw",
    "fcw_unique_count",
    "fcw_jump_count",
    "final_command_hz",
    "final_actual_hz",
    "quantization_error_mean_hz",
    "quantization_error_rms_hz",
    "quantization_error_max_abs_hz",
    "final_doppler_error_hz",
    "pll_error_rms_rad",
    "dll_error_rms",
    "prompt_magnitude_mean",
    "final_cn0_db_hz",
)


@dataclass(frozen=True)
class TrackingNCOComparison:
    """Comparison rows plus the full deterministic pipeline runs."""

    rows: tuple[dict[str, object], ...]
    runs: dict[str, SimulationResults]


def _validate_sweep(
    clock_hz: float,
    fixed_word_bits: tuple[int, ...],
) -> tuple[float, tuple[int, ...]]:
    clock = np.asarray(clock_hz)
    if (
        not np.isscalar(clock_hz)
        or clock.ndim != 0
        or not np.issubdtype(clock.dtype, np.number)
        or np.issubdtype(clock.dtype, np.complexfloating)
        or isinstance(clock_hz, (bool, np.bool_))
        or not bool(np.isfinite(clock))
        or float(clock) <= 0.0
    ):
        raise ValueError("clock_hz must be a finite positive real scalar")
    if not isinstance(fixed_word_bits, tuple) or not fixed_word_bits:
        raise ValueError("word_bits must be a nonempty tuple")
    if any(
        isinstance(bits, (bool, np.bool_))
        or not isinstance(bits, (int, np.integer))
        or not 2 <= int(bits) <= 62
        for bits in fixed_word_bits
    ):
        raise ValueError("word_bits must contain integers in [2, 62]")
    normalized_bits = tuple(int(bits) for bits in fixed_word_bits)
    if len(set(normalized_bits)) != len(normalized_bits):
        raise ValueError("word_bits must not contain duplicates")
    return float(clock), normalized_bits


def _tracking_row(
    case_name: str,
    branch_name: str,
    result: SimulationResults,
) -> dict[str, object]:
    cfg = result.config
    branch_key = branch_name.lower()
    tracking = result.branches[branch_key].tracking
    is_fixed = cfg.tracking.carrier_nco_mode == "fixed"
    lsb = (
        cfg.tracking.carrier_nco_clock_hz
        / 2**cfg.tracking.carrier_nco_word_bits
        if is_fixed
        else 0.0
    )
    identity: dict[str, object] = {
        "nco_case": case_name,
        "branch": branch_key,
        "mode": cfg.tracking.carrier_nco_mode,
        "word_bits": (
            cfg.tracking.carrier_nco_word_bits if is_fixed else ""
        ),
        "clock_hz": cfg.tracking.carrier_nco_clock_hz if is_fixed else "",
        "frequency_lsb_hz": lsb,
        "l1_velocity_lsb_mm_s": (
            1000.0 * SPEED_OF_LIGHT_MPS / L1_CARRIER_HZ * lsb
        ),
    }
    if not tracking.get("success", False):
        row = {
            **identity,
            **{name: "" for name in CSV_COLUMNS[7:]},
        }
        return row

    command = np.asarray(
        tracking["carrier_frequency_command_hz"], dtype=float
    )
    actual = np.asarray(tracking["carrier_frequency_hz"], dtype=float)
    error = np.asarray(
        tracking["carrier_frequency_quantization_error_hz"], dtype=float
    )
    pll = np.asarray(tracking["pll_error_rad"], dtype=float)
    dll = np.asarray(tracking["dll_error"], dtype=float)
    prompt = np.asarray(tracking["prompt"], dtype=complex)
    cn0 = np.asarray(tracking["cn0_db_hz"], dtype=float)
    fcw = (
        np.asarray(tracking["carrier_fcw"], dtype=np.int64)
        if is_fixed
        else np.array([], dtype=np.int64)
    )
    row = {
        **identity,
        "final_fcw": int(fcw[-1]) if is_fixed else "",
        "fcw_unique_count": int(np.unique(fcw).size) if is_fixed else "",
        "fcw_jump_count": (
            int(np.count_nonzero(np.diff(fcw))) if is_fixed else ""
        ),
        "final_command_hz": float(command[-1]),
        "final_actual_hz": float(actual[-1]),
        "quantization_error_mean_hz": float(np.mean(error)),
        "quantization_error_rms_hz": float(
            np.sqrt(np.mean(error**2))
        ),
        "quantization_error_max_abs_hz": float(np.max(np.abs(error))),
        "final_doppler_error_hz": float(
            actual[-1] - cfg.signal.doppler_hz
        ),
        "pll_error_rms_rad": float(np.sqrt(np.mean(pll**2))),
        "dll_error_rms": float(np.sqrt(np.mean(dll**2))),
        "prompt_magnitude_mean": float(np.mean(np.abs(prompt))),
        "final_cn0_db_hz": (
            float(cn0[-1]) if np.isfinite(cn0[-1]) else ""
        ),
    }
    for value in row.values():
        if isinstance(value, (int, float, np.number)) and not np.isfinite(
            value
        ):
            raise ValueError("comparison rows must contain finite scalars")
    return row


def compare_tracking_ncos(
    cfg: SimulationConfig,
    *,
    clock_hz: float,
    fixed_word_bits: tuple[int, ...],
) -> TrackingNCOComparison:
    """Run float and fixed NCO cases from identical deterministic inputs."""

    clock, word_bits = _validate_sweep(clock_hz, fixed_word_bits)
    cases: tuple[tuple[str, int | None], ...] = (
        ("float", None),
        *((f"fixed_{bits}", bits) for bits in word_bits),
    )
    runs: dict[str, SimulationResults] = {}
    rows: list[dict[str, object]] = []
    for case_name, bits in cases:
        run_cfg = deepcopy(cfg)
        run_cfg.plot.enable = False
        run_cfg.plot.show = False
        run_cfg.tracking.carrier_nco_mode = (
            "float" if bits is None else "fixed"
        )
        run_cfg.tracking.carrier_nco_clock_hz = clock
        if bits is not None:
            run_cfg.tracking.carrier_nco_word_bits = bits
        run_cfg.validate()
        runs[case_name] = run_end_to_end(run_cfg)

    for case_name, _ in cases:
        result = runs[case_name]
        rows.extend(
            _tracking_row(case_name, branch_name, result)
            for branch_name in result.config.diversity.modes
        )
    return TrackingNCOComparison(rows=tuple(rows), runs=runs)


def plot_tracking_nco_comparison(
    comparison: TrackingNCOComparison,
) -> dict[str, plt.Figure]:
    """Plot actual carrier frequency and quantization error by branch."""

    reference = next(iter(comparison.runs.values()))
    branches = tuple(
        name.lower() for name in reference.config.diversity.modes
    )
    frequency_figure, frequency_axes = plt.subplots(
        len(branches),
        1,
        figsize=(9.0, 2.8 * len(branches)),
        sharex=True,
        squeeze=False,
    )
    error_figure, error_axes = plt.subplots(
        len(branches),
        1,
        figsize=(9.0, 2.8 * len(branches)),
        sharex=True,
        squeeze=False,
    )
    for branch_index, branch_name in enumerate(branches):
        frequency_axis = frequency_axes[branch_index, 0]
        error_axis = error_axes[branch_index, 0]
        unavailable: list[str] = []
        for case_name, result in comparison.runs.items():
            tracking = result.branches[branch_name].tracking
            if not tracking.get("success", False):
                unavailable.append(case_name)
                continue
            actual = np.asarray(
                tracking["carrier_frequency_hz"], dtype=float
            )
            quantization_error = np.asarray(
                tracking["carrier_frequency_quantization_error_hz"],
                dtype=float,
            )
            epochs = np.arange(actual.size)
            frequency_axis.plot(epochs, actual, label=case_name)
            error_axis.plot(epochs, quantization_error, label=case_name)

        frequency_axis.axhline(
            reference.config.signal.doppler_hz,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="true_doppler",
        )
        if unavailable:
            note = f"acquisition unavailable: {', '.join(unavailable)}"
            frequency_axis.text(
                0.01, 0.04, note, transform=frequency_axis.transAxes
            )
            error_axis.text(0.01, 0.04, note, transform=error_axis.transAxes)
        frequency_axis.set_title(branch_name.upper())
        frequency_axis.set_ylabel("Frequency (Hz)")
        frequency_axis.grid(True, alpha=0.3)
        frequency_axis.legend()
        error_axis.set_title(branch_name.upper())
        error_axis.set_ylabel("Error (Hz)")
        error_axis.grid(True, alpha=0.3)
        if error_axis.lines:
            error_axis.legend()

    frequency_axes[-1, 0].set_xlabel("Tracking epoch")
    error_axes[-1, 0].set_xlabel("Tracking epoch")
    frequency_figure.tight_layout()
    error_figure.tight_layout()
    return {
        "tracking_nco_frequency": frequency_figure,
        "tracking_nco_quantization_error": error_figure,
    }


def save_tracking_nco_comparison(
    comparison: TrackingNCOComparison,
    output_dir: Path,
) -> dict[str, Path]:
    """Write the comparison CSV and two independent PNG figures."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "tracking_nco_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_COLUMNS,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(comparison.rows)

    figures = plot_tracking_nco_comparison(comparison)
    frequency_path = destination / "tracking_nco_frequency.png"
    error_path = destination / "tracking_nco_quantization_error.png"
    try:
        figures["tracking_nco_frequency"].savefig(
            frequency_path, dpi=160
        )
        figures["tracking_nco_quantization_error"].savefig(
            error_path, dpi=160
        )
    finally:
        for figure in figures.values():
            plt.close(figure)
    return {
        "csv": csv_path,
        "frequency_figure": frequency_path,
        "quantization_error_figure": error_path,
    }
