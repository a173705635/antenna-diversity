"""Multipath-only delay, direction and tracking-bias figures."""

import matplotlib.pyplot as plt
import numpy as np

from ..metrics import circular_chip_error, tracking_code_bias_chips
from ..plotting import _ordered_modes


def _plot_multipath_delay_profile(results):
    detection = results["multipath_detection"]
    figure, axis = plt.subplots(figsize=(10, 5))
    delays = detection["delay_axis_chips"]
    raw = detection["raw_power"]
    residual = detection["residual_power"]
    if delays.size:
        scale = max(
            float(np.max(raw)),
            float(np.max(residual)),
            np.finfo(float).tiny,
        )
        raw_db = 10.0 * np.log10(np.maximum(raw / scale, np.finfo(float).tiny))
        residual_db = 10.0 * np.log10(
            np.maximum(residual / scale, np.finfo(float).tiny)
        )
        axis.plot(delays, raw_db, label="Raw array correlation")
        axis.plot(delays, residual_db, label="After LOS projection")
        threshold_db = 10.0 * np.log10(
            max(detection["threshold_power"] / scale, np.finfo(float).tiny)
        )
        axis.axhline(threshold_db, color="black", linestyle=":", label="Threshold")
    for index, path in enumerate(results["truth"]["multipath_paths"], start=1):
        axis.axvline(
            path["excess_delay_chips"],
            color="blue",
            linestyle="--",
            label="True path" if index == 1 else None,
        )
    for index, candidate in enumerate(detection["candidates"], start=1):
        axis.axvline(
            candidate["excess_delay_chips"],
            color="red",
            linestyle="-.",
            label="Detected path" if index == 1 else None,
        )
    axis.set(
        title=f"Multipath delay profile: {detection['status']}",
        xlabel="Excess delay (chips)",
        ylabel="Relative power (dB)",
        ylim=(-80.0, 5.0),
    )
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend()
    figure.tight_layout()
    return figure


def _plot_multipath_angle(candidate, index):
    peak = max(float(np.max(candidate["angle_power"])), np.finfo(float).tiny)
    power_db = 10.0 * np.log10(
        np.maximum(candidate["angle_power"] / peak, np.finfo(float).tiny)
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    image = axis.imshow(
        power_db,
        origin="lower",
        aspect="auto",
        extent=(
            candidate["azimuths_deg"][0],
            candidate["azimuths_deg"][-1],
            candidate["elevations_deg"][0],
            candidate["elevations_deg"][-1],
        ),
        vmin=-30.0,
        vmax=0.0,
        cmap="viridis",
    )
    axis.plot(
        candidate["direction"].azimuth_deg,
        candidate["direction"].elevation_deg,
        "rx",
        markersize=10,
        markeredgewidth=2,
        label="Estimate",
    )
    axis.set(
        title=(
            f"Multipath {index} Bartlett map, "
            f"delay={candidate['excess_delay_chips']:.2f} chips"
        ),
        xlabel="Azimuth (deg)",
        ylabel="Elevation (deg)",
    )
    axis.legend()
    figure.colorbar(image, ax=axis, label="Normalized power (dB)")
    figure.tight_layout()
    return figure


def _plot_multipath_correlation_comparison(results):
    figure, axis = plt.subplots(figsize=(10, 5))
    fs_hz = results["dfe_info"]["fs_out_hz"]
    code_rate_hz = results["config"].signal.code_rate_hz
    true_delay = results["truth"]["code_delay_chips"]
    for mode in _ordered_modes(results):
        acquisition = results["branches"][mode]["acquisition"]
        row = int(
            np.argmin(
                np.abs(
                    acquisition["doppler_bins_hz"]
                    - acquisition["doppler_hz"]
                )
            )
        )
        power = acquisition["search_power"][row]
        chips = np.arange(power.size) * code_rate_hz / fs_hz
        offset = circular_chip_error(chips, true_delay)
        selected = np.abs(offset) <= 3.0
        order = np.argsort(offset[selected])
        normalized_db = 10.0 * np.log10(
            np.maximum(power[selected] / np.max(power), np.finfo(float).tiny)
        )
        axis.plot(offset[selected][order], normalized_db[order], label=mode.upper())
    axis.set(
        title="Acquisition correlation around the direct path",
        xlabel="Code delay relative to true LOS (chips)",
        ylabel="Normalized power (dB)",
        xlim=(-3.0, 3.0),
        ylim=(-40.0, 3.0),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def _plot_multipath_tracking_bias(results):
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fs_hz = results["dfe_info"]["fs_out_hz"]
    cfg = results["config"]
    metres_per_chip = 299_792_458.0 / cfg.signal.code_rate_hz
    for mode in _ordered_modes(results):
        tracking = results["branches"][mode]["tracking"]
        bias = tracking_code_bias_chips(
            tracking,
            results["truth"],
            fs_hz,
            cfg.tracking.coherent_ms,
        )
        if bias.size:
            epoch = np.arange(bias.size) + 1
            axes[0].plot(epoch, bias, label=mode.upper())
            axes[1].plot(epoch, bias * metres_per_chip, label=mode.upper())
    axes[0].set(ylabel="Code bias (chips)")
    axes[1].set(xlabel="Epoch", ylabel="Code bias (m)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    figure.suptitle("Tracking code bias relative to true LOS")
    figure.tight_layout()
    return figure


def plot_multipath_results(results):
    figures = {
        "multipath_delay_profile": _plot_multipath_delay_profile(results),
        "multipath_correlation_comparison": _plot_multipath_correlation_comparison(results),
        "multipath_tracking_bias": _plot_multipath_tracking_bias(results),
    }
    for index, candidate in enumerate(
        results["multipath_detection"]["candidates"],
        start=1,
    ):
        figures[f"multipath_angle_path_{index}"] = _plot_multipath_angle(
            candidate,
            index,
        )
    return figures
