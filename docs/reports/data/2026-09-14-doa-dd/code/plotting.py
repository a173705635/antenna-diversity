"""Independent figures for each important stage of the simulation."""

import matplotlib.pyplot as plt
import numpy as np

from antenna import antenna_response, build_antenna_model
from config import CWInterferer, WidebandInterferer
from metrics import circular_chip_error, tracking_code_bias_chips


MODE_ORDER = ("single", "egc", "mvdr", "blind_null", "multipath_lcmv")


def _ordered_modes(results):
    standard = [mode for mode in MODE_ORDER if mode in results["branches"]]
    extras = [mode for mode in results["branches"] if mode not in standard]
    return standard + extras


def _spectrum(x, fs_hz):
    x = x[: min(x.size, 65_536)]
    values = np.fft.fftshift(np.fft.fft(x * np.hanning(x.size)))
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(x.size, 1.0 / fs_hz))
    magnitude = np.abs(values)
    power_db = 20.0 * np.log10(magnitude / np.max(magnitude))
    return frequency_hz, power_db


def _absolute_spectrum(x, fs_hz, n_fft=None):
    samples = np.asarray(x).reshape(-1)
    size = min(samples.size, 65_536) if n_fft is None else n_fft
    samples = samples[:size]
    values = np.fft.fftshift(np.fft.fft(samples * np.hanning(size)))
    frequency_mhz = (
        np.fft.fftshift(np.fft.fftfreq(size, 1.0 / fs_hz)) / 1e6
    )
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(values), np.finfo(float).tiny)
    )
    return frequency_mhz, magnitude_db


def _mark_frequencies(axis, results, rf_domain):
    cfg = results["config"]
    frequency_shift_hz = cfg.signal.if_hz if rf_domain else 0.0
    axis.axvline(
        (frequency_shift_hz + results["truth"]["doppler_hz"]) / 1e6,
        color="green",
        linestyle="--",
        label="Target",
    )
    for index, interferer in enumerate(cfg.interferers, start=1):
        if not interferer.enable:
            continue
        if isinstance(interferer, CWInterferer):
            axis.axvline(
                (frequency_shift_hz + interferer.offset_hz) / 1e6,
                color="red",
                linestyle=":",
                label=f"CW {index}",
            )
        elif isinstance(interferer, WidebandInterferer):
            center_hz = frequency_shift_hz + interferer.center_offset_hz
            half_bandwidth_hz = interferer.bandwidth_hz / 2.0
            axis.axvspan(
                (center_hz - half_bandwidth_hz) / 1e6,
                (center_hz + half_bandwidth_hz) / 1e6,
                color="red",
                alpha=0.12,
                label=f"Wideband {index} band",
            )
            axis.axvline(
                center_hz / 1e6,
                color="red",
                linestyle=":",
                label=f"Wideband {index} center",
            )


def _plot_absolute_spectrum(axis, samples, fs_hz, title, results, rf_domain):
    frequency_mhz, magnitude_db = _absolute_spectrum(samples, fs_hz)
    axis.plot(frequency_mhz, magnitude_db, linewidth=0.8)
    _mark_frequencies(axis, results, rf_domain)
    axis.set(
        title=title,
        xlabel="Frequency (MHz)",
        ylabel="Magnitude (dB)",
    )
    axis.grid(True, alpha=0.3)
    axis.legend()


def _plot_channel_spectra(samples, fs_hz, names, title):
    figure, axis = plt.subplots(figsize=(9, 5))
    for channel, name in zip(samples, names):
        frequency, power = _spectrum(channel, fs_hz)
        axis.plot(frequency / 1e6, power, label=name)
    axis.set(title=title, xlabel="Frequency (MHz)", ylabel="Normalized magnitude (dB)")
    axis.set_ylim(-100, 5)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_rf_spectrum(results):
    return _plot_channel_spectra(
        results["scene_input_v"],
        results["config"].signal.fs_rf_hz,
        results["truth"]["active_antennas"],
        "RF antenna-port spectra",
    )


def plot_dfe_spectrum(results):
    return _plot_channel_spectra(
        results["dfe_channels"],
        results["dfe_info"]["fs_out_hz"],
        results["truth"]["active_antennas"],
        "DFE output spectra",
    )


def plot_combined_spectra(results):
    figure, axis = plt.subplots(figsize=(9, 5))
    for mode in MODE_ORDER:
        if mode in results["branches"]:
            frequency, power = _spectrum(
                results["branches"][mode]["combined"],
                results["dfe_info"]["fs_out_hz"],
            )
            axis.plot(frequency / 1e6, power, label=mode.upper())
    axis.set(
        title="Spectra after antenna combining",
        xlabel="Frequency (MHz)",
        ylabel="Normalized magnitude (dB)",
    )
    axis.set_ylim(-100, 5)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_array_response(results):
    cfg = results["config"]
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    azimuths = np.linspace(0.0, 360.0, 181)
    manifolds = np.column_stack(
        [
            antenna_response(
                model,
                type(cfg.signal.true_direction)(
                    azimuth, cfg.signal.true_direction.elevation_deg
                ),
                cfg.signal.true_polarization,
            )[0]
            for azimuth in azimuths
        ]
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    for mode in MODE_ORDER:
        if mode in results["branches"]:
            weights = results["branches"][mode]["combining"]["weights"]
            response = np.abs(weights.conj() @ manifolds)
            response_db = 20.0 * np.log10(response / np.max(response))
            axis.plot(azimuths, response_db, label=mode.upper())
    axis.axvline(
        cfg.signal.true_direction.azimuth_deg,
        color="green",
        linestyle="--",
        label="Target",
    )
    for index, interferer in enumerate(cfg.interferers, start=1):
        if interferer.enable:
            kind = "CW" if isinstance(interferer, CWInterferer) else "Wideband"
            axis.axvline(
                interferer.direction.azimuth_deg,
                color="red",
                linestyle=":",
                label=f"{kind} {index}",
            )
    for index, path in enumerate(cfg.signal.multipath_paths, start=1):
        if path.enable:
            axis.axvline(
                path.direction.azimuth_deg,
                color="blue",
                linestyle="--",
                label=f"Multipath {index}",
            )
    axis.set(
        title=f"Spatial response at elevation {cfg.signal.true_direction.elevation_deg:.1f}°",
        xlabel="Azimuth (deg)",
        ylabel="Normalized response (dB)",
        xlim=(0, 360),
        ylim=(-60, 5),
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_acquisition(results):
    modes = [mode for mode in MODE_ORDER if mode in results["branches"]]
    columns = min(3, len(modes))
    rows = int(np.ceil(len(modes) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
    )
    for axis, mode in zip(axes.flat, modes):
        acquisition = results["branches"][mode]["acquisition"]
        power_db = 10.0 * np.log10(
            acquisition["search_power"] / np.max(acquisition["search_power"])
        )
        image = axis.imshow(
            power_db,
            origin="lower",
            aspect="auto",
            extent=(
                0,
                acquisition["coherent_samples"] - 1,
                acquisition["doppler_bins_hz"][0],
                acquisition["doppler_bins_hz"][-1],
            ),
            vmin=-30,
            vmax=0,
            cmap="viridis",
        )
        axis.plot(
            acquisition["code_phase_samples"],
            acquisition["doppler_hz"],
            "rx",
            markersize=9,
        )
        axis.set(
            title=f"{mode.upper()}  metric={acquisition['metric']:.2f}",
            xlabel="Code phase (samples)",
            ylabel="Doppler (Hz)",
        )
        figure.colorbar(image, ax=axis, label="Normalized power (dB)")
    for axis in axes.flat[len(modes) :]:
        axis.set_visible(False)
    figure.suptitle("Acquisition search maps")
    figure.tight_layout()
    return figure


def plot_tracking(results):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = (
        ("carrier_frequency_hz", "Carrier frequency (Hz)"),
        ("prompt_magnitude", "Prompt magnitude"),
        ("pll_error_rad", "PLL error (rad)"),
        ("dll_error", "DLL error"),
    )
    for axis, (field, ylabel) in zip(axes.flat, labels):
        for mode in MODE_ORDER:
            branch = results["branches"].get(mode)
            if branch and branch["tracking"]["success"]:
                values = branch["tracking"][field]
                axis.plot(np.arange(values.size) + 1, values, label=mode.upper())
        axis.set(xlabel="Epoch (ms)", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend()
    figure.suptitle("Tracking loops")
    figure.tight_layout()
    return figure


def plot_cn0(results):
    figure, axis = plt.subplots(figsize=(9, 5))
    for mode in MODE_ORDER:
        branch = results["branches"].get(mode)
        if branch and branch["tracking"]["success"]:
            values = branch["tracking"]["cn0_db_hz"]
            axis.plot(np.arange(values.size) + 1, values, label=mode.upper())
    axis.set(
        title="Tracking C/N0",
        xlabel="Epoch (ms)",
        ylabel="C/N0 (dB-Hz)",
    )
    axis.grid(True, alpha=0.3)
    if axis.lines:
        axis.legend()
    figure.tight_layout()
    return figure


def _plot_detailed_rf_dfe(results):
    cfg = results["config"]
    figures = {}
    for index, name in enumerate(results["truth"]["active_antennas"]):
        figure, axes = plt.subplots(3, 1, figsize=(10, 9))
        _plot_absolute_spectrum(
            axes[0],
            results["scene_input_v"][index],
            cfg.signal.fs_rf_hz,
            f"{name} antenna-port input",
            results,
            True,
        )
        _plot_absolute_spectrum(
            axes[1],
            results["adc_codes"][index],
            cfg.signal.fs_rf_hz,
            f"{name} ADC output",
            results,
            True,
        )
        _plot_absolute_spectrum(
            axes[2],
            results["dfe_channels"][index],
            results["dfe_info"]["fs_out_hz"],
            f"{name} DFE output",
            results,
            False,
        )
        figure.suptitle(f"RF to DFE chain: {name}")
        figure.tight_layout()
        figures[f"detailed_rf_dfe_{name}"] = figure
    return figures


def _plot_detailed_combined(results):
    modes = _ordered_modes(results)
    n_fft = min(
        65_536,
        *(results["branches"][mode]["combined"].size for mode in modes),
    )
    spectra = {
        mode: _absolute_spectrum(
            results["branches"][mode]["combined"],
            results["dfe_info"]["fs_out_hz"],
            n_fft,
        )
        for mode in modes
    }
    all_magnitudes = np.concatenate(
        [magnitude_db for _, magnitude_db in spectra.values()]
    )
    y_min = float(np.min(all_magnitudes))
    y_max = float(np.max(all_magnitudes))
    padding = max(1.0, 0.03 * (y_max - y_min))

    figures = {}
    for mode in modes:
        frequency_mhz, magnitude_db = spectra[mode]
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(frequency_mhz, magnitude_db, linewidth=0.8)
        _mark_frequencies(axis, results, False)
        axis.set(
            title=f"{mode.upper()} combined output",
            xlabel="Frequency (MHz)",
            ylabel="Magnitude (dB)",
            xlim=(frequency_mhz[0], frequency_mhz[-1]),
            ylim=(y_min - padding, y_max + padding),
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figures[f"detailed_combined_{mode}"] = figure
    return figures


def _plot_detailed_array_response(results):
    cfg = results["config"]
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    azimuths = np.arange(0.0, 361.0, 2.0)
    elevations = np.arange(0.0, 91.0, 2.0)
    manifolds = np.empty(
        (len(model.active_antennas), elevations.size, azimuths.size),
        dtype=complex,
    )
    direction_type = type(cfg.signal.true_direction)
    for elevation_index, elevation in enumerate(elevations):
        for azimuth_index, azimuth in enumerate(azimuths):
            manifolds[:, elevation_index, azimuth_index] = antenna_response(
                model,
                direction_type(azimuth, elevation),
                cfg.signal.true_polarization,
            )[0]

    figures = {}
    for mode in _ordered_modes(results):
        weights = results["branches"][mode]["combining"]["weights"]
        response = np.abs(
            np.tensordot(weights.conj(), manifolds, axes=(0, 0))
        )
        response_db = 20.0 * np.log10(
            np.maximum(response / np.max(response), np.finfo(float).tiny)
        )

        figure, axis = plt.subplots(figsize=(9, 5))
        image = axis.imshow(
            response_db,
            origin="lower",
            aspect="auto",
            extent=(0.0, 360.0, 0.0, 90.0),
            vmin=-60.0,
            vmax=0.0,
            cmap="viridis",
        )
        if float(np.min(response_db)) < -3.0 < float(np.max(response_db)):
            axis.contour(
                azimuths,
                elevations,
                response_db,
                levels=[-3.0],
                colors="white",
                linewidths=1.0,
            )
        axis.plot(
            cfg.signal.true_direction.azimuth_deg,
            cfg.signal.true_direction.elevation_deg,
            marker="*",
            color="green",
            linestyle="none",
            markersize=12,
            label="Target",
        )
        for index, interferer in enumerate(cfg.interferers, start=1):
            if interferer.enable:
                kind = (
                    "CW"
                    if isinstance(interferer, CWInterferer)
                    else "Wideband"
                )
                axis.plot(
                    interferer.direction.azimuth_deg,
                    interferer.direction.elevation_deg,
                    marker="x",
                    color="red",
                    linestyle="none",
                    markersize=9,
                    markeredgewidth=2,
                    label=f"{kind} {index}",
                )
        for index, path in enumerate(cfg.signal.multipath_paths, start=1):
            if path.enable:
                axis.plot(
                    path.direction.azimuth_deg,
                    path.direction.elevation_deg,
                    marker="o",
                    markerfacecolor="none",
                    color="deepskyblue",
                    linestyle="none",
                    markersize=9,
                    markeredgewidth=2,
                    label=f"Multipath {index}",
                )
        axis.set(
            title=f"{mode.upper()} array response",
            xlabel="Azimuth (deg)",
            ylabel="Elevation (deg)",
        )
        axis.legend()
        figure.colorbar(image, ax=axis, label="Normalized response (dB)")
        figure.tight_layout()
        figures[f"detailed_array_response_{mode}"] = figure
    return figures


def _plot_detailed_acquisition(results):
    figures = {}
    for mode in _ordered_modes(results):
        acquisition = results["branches"][mode]["acquisition"]
        search_power = acquisition["search_power"]
        peak_scale = np.max(search_power)
        search_db = 10.0 * np.log10(
            np.maximum(search_power / peak_scale, np.finfo(float).tiny)
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
        doppler_bins = acquisition["doppler_bins_hz"]
        best_bin = int(np.argmin(np.abs(doppler_bins - acquisition["doppler_hz"])))

        figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        image = axes[0].imshow(
            search_db,
            origin="lower",
            aspect="auto",
            extent=(
                code_phase_chips[0],
                code_phase_chips[-1],
                doppler_bins[0],
                doppler_bins[-1],
            ),
            vmin=-40.0,
            vmax=0.0,
            cmap="viridis",
        )
        axes[0].plot(
            peak_chip,
            acquisition["doppler_hz"],
            "rx",
            markersize=10,
            markeredgewidth=2,
        )
        axes[0].set(
            title=f"{mode.upper()} search map",
            xlabel="Code phase (chips)",
            ylabel="Doppler (Hz)",
        )
        figure.colorbar(
            image,
            ax=axes[0],
            label="Normalized power (dB)",
        )

        slice_db = search_db[best_bin]
        axes[1].plot(code_phase_chips, slice_db, linewidth=1.0)
        axes[1].plot(
            peak_chip,
            0.0,
            "rx",
            markersize=10,
            markeredgewidth=2,
            label="Primary peak",
        )
        second_chip = (
            acquisition["second_peak_code_phase_samples"]
            / acquisition["samples_per_ms"]
            * 1023.0
        )
        second_db = 10.0 * np.log10(
            acquisition["second_peak"] / peak_scale
        )
        axes[1].plot(
            second_chip,
            second_db,
            "o",
            color="#f2730c",
            markersize=7,
            label="Second peak",
        )
        status = "PASS" if acquisition["success"] else "FAIL"
        axes[1].set(
            title=(
                f"{mode.upper()}: {status}, metric={acquisition['metric']:.2f}\n"
                f"Doppler={acquisition['doppler_hz']:.0f} Hz, "
                f"code={peak_chip:.2f} chips"
            ),
            xlabel="Code phase (chips)",
            ylabel="Normalized power (dB)",
            ylim=(-40.0, 3.0),
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        figure.suptitle(f"Detailed acquisition: {mode.upper()}")
        figure.tight_layout()
        figures[f"detailed_acquisition_{mode}"] = figure
    return figures

def plot_detailed_results(results):
    return {
        **_plot_detailed_rf_dfe(results),
        **_plot_detailed_combined(results),
        **_plot_detailed_array_response(results),
        **_plot_detailed_acquisition(results),
    }


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


def plot_results(results):
    return {
        "rf_spectrum": plot_rf_spectrum(results),
        "dfe_spectrum": plot_dfe_spectrum(results),
        "combined_spectra": plot_combined_spectra(results),
        "array_response": plot_array_response(results),
        "acquisition": plot_acquisition(results),
        "tracking": plot_tracking(results),
        "cn0": plot_cn0(results),
    }
