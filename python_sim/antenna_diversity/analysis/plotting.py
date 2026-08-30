"""Matplotlib diagnostics for every major receiver stage."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from antenna_diversity.analysis.spatial import response_grid_db
from antenna_diversity.array.steering import steering_vector
from antenna_diversity.config import PhoneArrayConfig
from antenna_diversity.models import SimulationResults


_STANDARD_BRANCH_ORDER = ("single", "egc", "mvdr", "blind_null")


def _ordered_branch_names(results: SimulationResults) -> tuple[str, ...]:
    if not results.branches:
        raise ValueError("results must contain at least one combined branch")
    ordered = tuple(
        mode for mode in _STANDARD_BRANCH_ORDER if mode in results.branches
    )
    extras = tuple(mode for mode in results.branches if mode not in ordered)
    return ordered + extras


def _calculate_spectrum(
    x: np.ndarray,
    fs_hz: float,
    n_fft: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(x).reshape(-1)[:n_fft]
    window = 0.54 - 0.46 * np.cos(
        2.0 * np.pi * np.arange(n_fft) / max(1, n_fft - 1)
    )
    spectrum = np.fft.fftshift(np.fft.fft(samples * window))
    frequency_mhz = (
        np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / fs_hz)) / 1e6
    )
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(spectrum), np.finfo(float).eps)
    )
    return frequency_mhz, magnitude_db


def _plot_spectrum(axis, x: np.ndarray, fs_hz: float, title: str) -> None:
    samples = np.asarray(x).reshape(-1)
    n_fft = min(65_536, samples.size)
    frequency_mhz, magnitude_db = _calculate_spectrum(
        samples, fs_hz, n_fft
    )
    axis.plot(frequency_mhz, magnitude_db, linewidth=0.8)
    axis.grid(True)
    axis.set_xlabel("Frequency (MHz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.set_title(title)


def _enabled_interferers(results: SimulationResults):
    cfg = results.config
    if isinstance(cfg.array, PhoneArrayConfig):
        return tuple(
            interferer for interferer in cfg.interferers if interferer.enable
        )
    return (cfg.jammer,) if cfg.jammer.enable else ()


def _mark_spectrum_directions(
    axis,
    results: SimulationResults,
    frequency_shift_hz: float,
) -> None:
    axis.axvline(
        (frequency_shift_hz + results.truth["doppler_hz"]) / 1e6,
        color="green",
        linestyle="--",
        label="Target Doppler",
        alpha=0.75,
        zorder=1,
    )
    for index, interferer in enumerate(_enabled_interferers(results), start=1):
        axis.axvline(
            (frequency_shift_hz + interferer.offset_hz) / 1e6,
            color="red",
            linestyle="--",
            label=f"Interferer {index}",
            alpha=0.75,
            zorder=1,
        )


def plot_spectra(results: SimulationResults) -> Figure:
    """Plot antenna input, ADC output, and DFE output spectra."""

    figure, axes = plt.subplots(
        3, 1, figsize=(10, 9), constrained_layout=True
    )
    _plot_spectrum(
        axes[0],
        results.scene_input_v[0],
        results.config.signal.fs_rf_hz,
        "Antenna 1 input",
    )
    _mark_spectrum_directions(axes[0], results, results.config.signal.if_hz)
    _plot_spectrum(
        axes[1],
        results.adc_codes[0],
        results.config.signal.fs_rf_hz,
        "ADC output",
    )
    _mark_spectrum_directions(axes[1], results, results.config.signal.if_hz)
    _plot_spectrum(
        axes[2],
        results.dfe_channels[0],
        results.dfe_info["fs_out_hz"],
        "DFE output",
    )
    _mark_spectrum_directions(axes[2], results, 0.0)
    axes[0].legend(loc="best")
    figure.suptitle("RF and DFE chain")
    return figure


def plot_combined_spectra(results: SimulationResults) -> Figure:
    """Compare absolute spectra after each available combining branch."""

    mode_names = _ordered_branch_names(results)

    combined: dict[str, np.ndarray] = {}
    for mode in mode_names:
        values = np.asarray(results.branches[mode].combined)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(
                "each combined branch must be a nonempty one-dimensional array"
            )
        combined[mode] = values

    n_fft = min(
        65_536,
        *(values.size for values in combined.values()),
    )
    fs_hz = results.dfe_info["fs_out_hz"]
    spectra = {
        mode: _calculate_spectrum(values, fs_hz, n_fft)
        for mode, values in combined.items()
    }
    all_magnitudes = np.concatenate(
        [magnitude_db for _, magnitude_db in spectra.values()]
    )
    y_min = float(np.min(all_magnitudes))
    y_max = float(np.max(all_magnitudes))
    y_padding = max(1.0, 0.03 * (y_max - y_min))

    figure, axes = plt.subplots(
        len(mode_names),
        1,
        figsize=(10, 3 * len(mode_names)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    data_axes = axes[:, 0]
    for axis, mode in zip(data_axes, mode_names, strict=True):
        frequency_mhz, magnitude_db = spectra[mode]
        axis.plot(
            frequency_mhz,
            magnitude_db,
            linewidth=0.8,
            zorder=3,
        )
        _mark_spectrum_directions(axis, results, 0.0)
        axis.set_xlim(frequency_mhz[0], frequency_mhz[-1])
        axis.set_ylim(y_min - y_padding, y_max + y_padding)
        axis.set_ylabel("Magnitude (dB)")
        axis.set_title(f"{mode.upper()} combined output")
        axis.grid(True)
    data_axes[-1].set_xlabel("Frequency (MHz)")
    data_axes[0].legend(loc="best")
    figure.suptitle("Spectra after antenna combining")
    return figure


def _plot_legacy_array_response(results: SimulationResults) -> Figure:
    cfg = results.config
    angles_deg = np.arange(-90.0, 90.0 + 0.25, 0.25)
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for mode, branch in results.branches.items():
        weights = branch.combining["weights"]
        response = np.array(
            [
                abs(
                    np.vdot(
                        weights,
                        steering_vector(
                            cfg.array.positions_m,
                            angle_deg,
                            cfg.signal.wavelength_m,
                        ),
                    )
                )
                for angle_deg in angles_deg
            ]
        )
        response_db = 20.0 * np.log10(np.maximum(response, 1e-6))
        axis.plot(angles_deg, response_db, label=mode.upper())
    axis.axvline(
        cfg.array.desired_angle_deg,
        color="green",
        linestyle="--",
        label="Desired",
    )
    if cfg.jammer.enable:
        axis.axvline(
            cfg.jammer.angle_deg,
            color="red",
            linestyle="--",
            label="Jammer",
        )
    axis.grid(True)
    axis.set_xlabel("Angle (deg)")
    axis.set_ylabel("Response (dB)")
    axis.set_ylim(-80, 10)
    axis.set_title("Nominal array response")
    axis.legend(loc="best")
    return figure


def _plot_phone_array_response(results: SimulationResults) -> Figure:
    cfg = results.config
    mode_names = _ordered_branch_names(results)
    azimuths = np.arange(0.0, 360.0 + 2.0, 2.0)
    elevations = np.arange(0.0, 90.0 + 1.0, 1.0)
    grids = {
        mode: response_grid_db(
            cfg,
            results.branches[mode].combining["weights"],
            azimuths,
            elevations,
        )
        for mode in mode_names
    }
    figure, axes = plt.subplots(
        1,
        len(grids),
        figsize=(5 * len(grids), 5),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, (mode, grid) in zip(axes[0], grids.items(), strict=True):
        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=(0.0, 360.0, 0.0, 90.0),
            vmin=-60.0,
            vmax=0.0,
        )
        if float(np.min(grid)) < -3.0 < float(np.max(grid)):
            axis.contour(
                azimuths,
                elevations,
                grid,
                levels=[-3.0],
                colors="white",
                linewidths=1.0,
            )
        target = cfg.signal.true_direction
        axis.plot(
            target.azimuth_deg,
            target.elevation_deg,
            marker="*",
            color="green",
            linestyle="none",
            markersize=12,
            label="Target",
        )
        for index, interferer in enumerate(
            _enabled_interferers(results), start=1
        ):
            axis.plot(
                interferer.direction.azimuth_deg,
                interferer.direction.elevation_deg,
                marker="x",
                color="red",
                linestyle="none",
                markersize=9,
                markeredgewidth=2,
                label=f"Interferer {index}",
            )
        axis.set_xlabel("Azimuth (deg)")
        axis.set_ylabel("Elevation (deg)")
        axis.set_title(f"{mode.upper()} array response")
        axis.legend(loc="best")
    figure.colorbar(
        image,
        ax=axes[0].tolist(),
        label="Normalized response (dB)",
    )
    figure.suptitle("Phone-array spatial response")
    return figure


def plot_array_response(results: SimulationResults) -> Figure:
    """Plot nominal spatial responses of all combining branches."""

    if isinstance(results.config.array, PhoneArrayConfig):
        return _plot_phone_array_response(results)
    return _plot_legacy_array_response(results)


def plot_acquisition(results: SimulationResults) -> Figure:
    """Plot a 2-D search map and best-Doppler code slice for each branch."""

    if not results.branches:
        raise ValueError("results must contain at least one acquisition branch")
    mode_names = list(results.branches)
    figure, axes = plt.subplots(
        len(mode_names),
        2,
        figsize=(13, 4 * len(mode_names)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, mode in enumerate(mode_names):
        acquisition = results.branches[mode].acquisition
        search_power = acquisition["search_power"]
        doppler_bins = acquisition["doppler_bins_hz"]
        if search_power.shape[0] != doppler_bins.size or search_power.size == 0:
            raise ValueError(f"acquisition dimensions are inconsistent for {mode}")

        peak_scale = max(
            float(acquisition["peak"]),
            float(np.max(search_power)),
            float(np.finfo(float).eps),
        )
        search_db = 10.0 * np.log10(
            np.maximum(search_power / peak_scale, np.finfo(float).eps)
        )
        search_db = np.maximum(search_db, -40.0)
        code_phase_chips = (
            np.arange(search_power.shape[1])
            / acquisition["samples_per_ms"]
            * 1023.0
        )
        peak_chip = (
            acquisition["code_phase_samples"]
            / acquisition["samples_per_ms"]
            * 1023.0
        )

        map_axis = axes[row, 0]
        image = map_axis.imshow(
            search_db,
            origin="lower",
            aspect="auto",
            extent=(
                code_phase_chips[0],
                code_phase_chips[-1],
                doppler_bins[0],
                doppler_bins[-1],
            ),
            vmin=-40,
            vmax=0,
            cmap="viridis",
        )
        map_axis.plot(
            peak_chip,
            acquisition["doppler_hz"],
            "rx",
            markersize=10,
            markeredgewidth=2,
        )
        figure.colorbar(image, ax=map_axis, label="Normalized power (dB)")
        map_axis.set_xlabel("Code phase (chips)")
        map_axis.set_ylabel("Doppler (Hz)")
        map_axis.set_title(f"{mode.upper()} search map")

        best_bin = int(
            np.argmin(np.abs(doppler_bins - acquisition["doppler_hz"]))
        )
        slice_db = 10.0 * np.log10(
            np.maximum(
                search_power[best_bin] / peak_scale,
                np.finfo(float).eps,
            )
        )
        slice_db = np.maximum(slice_db, -40.0)
        slice_axis = axes[row, 1]
        slice_axis.plot(code_phase_chips, slice_db, "b-", linewidth=1)
        slice_axis.plot(
            peak_chip,
            0.0,
            "rx",
            markersize=10,
            markeredgewidth=2,
            label="Primary peak",
        )
        second_sample = acquisition["second_peak_code_phase_samples"]
        if np.isfinite(second_sample):
            second_chip = (
                second_sample / acquisition["samples_per_ms"] * 1023.0
            )
            second_db = 10.0 * np.log10(
                max(
                    acquisition["second_peak"] / peak_scale,
                    np.finfo(float).eps,
                )
            )
            slice_axis.plot(
                second_chip,
                second_db,
                "o",
                color="#f2730c",
                markersize=7,
                label="Second peak",
            )
        status = "PASS" if acquisition["success"] else "FAIL"
        slice_axis.grid(True)
        slice_axis.set_ylim(-40, 3)
        slice_axis.set_xlabel("Code phase (chips)")
        slice_axis.set_ylabel("Normalized power (dB)")
        slice_axis.set_title(
            f"{mode.upper()}: {status}, ratio={acquisition['metric']:.2f}\n"
            f"Doppler={acquisition['doppler_hz']:.0f} Hz, code={peak_chip:.2f} chips"
        )
        slice_axis.legend(loc="best")

    figure.suptitle("FFT acquisition: Doppler and code-phase search")
    return figure


def plot_tracking(results: SimulationResults) -> Figure:
    """Compare carrier estimates and Prompt magnitudes of successful branches."""

    figure, axes = plt.subplots(
        2, 1, figsize=(10, 7), constrained_layout=True
    )
    for mode, branch in results.branches.items():
        tracking = branch.tracking
        if tracking.get("success", False):
            axes[0].plot(
                tracking["carrier_frequency_hz"], label=mode.upper()
            )
            axes[1].plot(np.abs(tracking["prompt"]), label=mode.upper())
    axes[0].axhline(
        results.truth["doppler_hz"],
        color="black",
        linestyle="--",
        label="Truth",
    )
    axes[0].grid(True)
    axes[0].set_ylabel("Doppler (Hz)")
    axes[0].set_title("Carrier tracking")
    axes[0].legend(loc="best")
    axes[1].grid(True)
    axes[1].set_xlabel("Tracking epoch (ms)")
    axes[1].set_ylabel("|Prompt|")
    axes[1].set_title("Prompt correlation magnitude")
    axes[1].legend(loc="best")
    return figure


def plot_cn0(results: SimulationResults) -> Figure:
    """Plot measured tracking C/N0 for every available branch."""

    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    unavailable_messages: list[str] = []
    for mode, branch in results.branches.items():
        tracking = branch.tracking
        if not branch.acquisition.get("success", False):
            unavailable_messages.append(
                f"{mode.upper()}: acquisition failed – C/N₀ unavailable"
            )
            continue
        if not tracking.get("success", False):
            unavailable_messages.append(
                f"{mode.upper()}: tracking unavailable – C/N₀ unavailable"
            )
            continue

        epoch_starts = np.asarray(tracking["epoch_start_samples"])
        cn0_db_hz = np.asarray(tracking["cn0_db_hz"])
        if (
            epoch_starts.ndim != 1
            or cn0_db_hz.ndim != 1
            or epoch_starts.size != cn0_db_hz.size
        ):
            raise ValueError(f"{mode} C/N0 tracking arrays have inconsistent length")
        finite = np.isfinite(cn0_db_hz)
        if not np.any(finite):
            unavailable_messages.append(
                f"{mode.upper()}: tracking unavailable – C/N₀ unavailable"
            )
            continue
        time_ms = epoch_starts / results.config.dfe.fs_out_hz * 1e3
        axis.plot(time_ms[finite], cn0_db_hz[finite], label=mode.upper())

    for index, message in enumerate(unavailable_messages):
        axis.text(
            0.02,
            0.98 - 0.07 * index,
            message,
            transform=axis.transAxes,
            va="top",
        )
    axis.grid(True)
    axis.set_xlabel("Tracking time (ms)")
    axis.set_ylabel("Estimated C/N₀ (dB-Hz)")
    axis.set_title("Tracking C/N₀ comparison")
    if axis.lines:
        axis.legend(loc="best")
    return figure


def plot_results(results: SimulationResults) -> dict[str, Figure]:
    """Create every standard simulation diagnostic figure."""

    return {
        "rf_dfe_spectra": plot_spectra(results),
        "combined_spectra": plot_combined_spectra(results),
        "array_response": plot_array_response(results),
        "acquisition_search": plot_acquisition(results),
        "tracking_comparison": plot_tracking(results),
        "cn0_comparison": plot_cn0(results),
    }


def save_figures(
    figures: dict[str, Figure], output_dir: str | Path
) -> dict[str, Path]:
    """Save named figures as PNG files and return their resolved paths."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, figure in figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=150)
        paths[name] = path.resolve()
    return paths
