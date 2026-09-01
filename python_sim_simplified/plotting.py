"""Independent figures for each important stage of the simulation."""

import matplotlib.pyplot as plt
import numpy as np

from antenna import antenna_response, build_antenna_model


MODE_ORDER = ("single", "egc", "mvdr", "blind_null")


def _spectrum(x, fs_hz):
    x = x[: min(x.size, 65_536)]
    values = np.fft.fftshift(np.fft.fft(x * np.hanning(x.size)))
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(x.size, 1.0 / fs_hz))
    magnitude = np.abs(values)
    power_db = 20.0 * np.log10(magnitude / np.max(magnitude))
    return frequency_hz, power_db


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
    for index, interferer in enumerate(cfg.interferers):
        if interferer.enable:
            axis.axvline(
                interferer.direction.azimuth_deg,
                color="red",
                linestyle=":",
                label="Interferer" if index == 0 else None,
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
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
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
