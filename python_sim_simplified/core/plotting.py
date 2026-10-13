"""Independent figures for each important stage of the simulation."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .antenna import antenna_response, build_antenna_model
from .config import CWInterferer, WidebandInterferer
from .metrics import circular_chip_error, tracking_code_bias_chips


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
        if cfg.enable_multipath_research and path.enable:
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

    columns = min(2, len(modes))
    rows = int(np.ceil(len(modes) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7 * columns, 3.5 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    legend_handles = None
    legend_labels = None
    for axis, mode in zip(axes.flat, modes):
        frequency_mhz, magnitude_db = spectra[mode]
        axis.plot(
            frequency_mhz,
            magnitude_db,
            linewidth=0.8,
        )
        _mark_frequencies(axis, results, False)
        if legend_handles is None:
            legend_handles, legend_labels = axis.get_legend_handles_labels()
        axis.set(
            title=mode.upper(),
            xlabel="Frequency (MHz)",
            ylabel="Magnitude (dB)",
            xlim=(frequency_mhz[0], frequency_mhz[-1]),
            ylim=(y_min - padding, y_max + padding),
        )
        axis.grid(True, alpha=0.3)
    for axis in axes.flat[len(modes):]:
        axis.set_visible(False)
    figure.suptitle("Spectra after antenna combining")
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    return {"detailed_combined_spectra": figure}


def _array_response_directions(cfg):
    directions = [(cfg.signal.true_direction, "Target", "*", "limegreen")]
    for index, interferer in enumerate(cfg.interferers, start=1):
        if interferer.enable:
            if isinstance(interferer, CWInterferer):
                kind, marker, color = "CW", "x", "red"
            else:
                kind, marker, color = "Wideband", "D", "magenta"
            directions.append((interferer.direction, f"{kind} {index}", marker, color))
    for index, path in enumerate(cfg.signal.multipath_paths, start=1):
        if cfg.enable_multipath_research and path.enable:
            directions.append((path.direction, f"Multipath {index}", "o", "deepskyblue"))
    return directions


def _array_response_legend_label(direction, label, gain_db):
    return (
        f"{label} | az {direction.azimuth_deg:g}°, "
        f"el {direction.elevation_deg:g}° | {gain_db:.1f} dB"
    )


def _spherical_response_coordinates(azimuths, elevations, radius):
    """Azimuth from +X toward +Y; elevation from XY toward +Z, in degrees."""
    azimuth_grid, elevation_grid = np.meshgrid(
        np.deg2rad(azimuths), np.deg2rad(elevations)
    )
    x = radius * np.cos(elevation_grid) * np.cos(azimuth_grid)
    y = radius * np.cos(elevation_grid) * np.sin(azimuth_grid)
    z = radius * np.sin(elevation_grid)
    return x, y, z


def _plot_array_response_spherical(results, mode, azimuths, elevations, response):
    """Fixed unit sphere with dB colors and two opposite, opaque views."""
    cfg = results["config"]
    peak_response = np.max(response)
    response_db = 20.0 * np.log10(
        np.maximum(response / peak_response, np.finfo(float).tiny)
    )
    x, y, z = _spherical_response_coordinates(azimuths, elevations, 1.0)
    color_scale = plt.Normalize(vmin=-40.0, vmax=0.0)
    color_map = plt.get_cmap("turbo")
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    weights = results["branches"][mode]["combining"]["weights"]
    directions = _array_response_directions(cfg)
    # Aim the first camera at the first enabled interferer, then show its antipode.
    view_azimuth = next(
        (item.direction.azimuth_deg for item in cfg.interferers if item.enable),
        cfg.signal.true_direction.azimuth_deg,
    )
    markers = []
    legend_handles = []
    legend_labels = []
    for direction, label, marker, color in directions:
        steering, _ = antenna_response(model, direction, cfg.signal.true_polarization)
        gain = abs(np.vdot(weights, steering)) / peak_response
        gain_db = 20.0 * np.log10(np.maximum(gain, np.finfo(float).tiny))
        legend_labels.append(_array_response_legend_label(direction, label, gain_db))
        legend_handles.append(Line2D(
            [], [], marker=marker, color=color, linestyle="none", markersize=10,
            markeredgecolor=color if marker == "x" else "black", markeredgewidth=1,
        ))
        tip = np.asarray(_spherical_response_coordinates(
            [direction.azimuth_deg], [direction.elevation_deg], 1.03
        )).reshape(3)
        markers.append((tip, marker, color))

    figure = plt.figure(figsize=(14, 8))
    for view_index, (camera_elevation, camera_azimuth) in enumerate(
        ((25.0, view_azimuth), (-25.0, view_azimuth + 180.0)), start=1
    ):
        axis = figure.add_subplot(1, 2, view_index, projection="3d", computed_zorder=False)
        axis.plot_surface(
            x, y, z,
            facecolors=color_map(color_scale(response_db)),
            rcount=elevations.size, ccount=azimuths.size,
            linewidth=0, antialiased=False, shade=False, zorder=1,
        )
        camera = np.asarray(_spherical_response_coordinates(
            [camera_azimuth], [camera_elevation], 1.0
        )).reshape(3)
        for tip, marker, color in markers:
            # Only annotate the visible hemisphere; hidden directions appear in the other view.
            if np.dot(tip, camera) > 0.0:
                axis.plot(
                    [tip[0]], [tip[1]], [tip[2]], marker=marker, color=color,
                    markeredgecolor=color if marker == "x" else "black", markeredgewidth=1,
                    linestyle="none", markersize=11,
                    zorder=11 if marker == "x" else 10,
                )
        axis.set(
            title="Interference-facing view" if view_index == 1 else "Opposite view",
            xlabel="X", ylabel="Y", zlabel="Z",
            xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), zlim=(-1.2, 1.2),
            xticks=(-1, 0, 1), yticks=(-1, 0, 1), zticks=(-1, 0, 1),
        )
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=camera_elevation, azim=camera_azimuth)
        axis.grid(False)
        for coordinate_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coordinate_axis.pane.set_alpha(0.0)
    color_axis = figure.add_axes((0.91, 0.28, 0.015, 0.45))
    figure.colorbar(
        plt.cm.ScalarMappable(norm=color_scale, cmap=color_map),
        cax=color_axis, label="Normalized response (dB)", extend="min",
        ticks=(-40, -30, -20, -10, 0),
    )
    figure.suptitle(
        f"{mode.upper()} spherical response\nFixed radius; color shows gain and nulls", fontsize=16
    )
    figure.legend(
        legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.46, 0.065),
        ncol=2, fontsize=10, frameon=False, columnspacing=3,
    )
    figure.text(0.06, 0.02, "Colors below -40 dB share the darkest shade; listed values are not clipped.", fontsize=10)
    figure.subplots_adjust(left=0.01, right=0.88, bottom=0.18, top=0.83, wspace=0.06)
    return figure


def _plot_array_response_3d(
    results, mode, azimuths, elevations, response_db, peak_response
):
    """Upper-hemisphere response at the model carrier; dB floor is display-only."""
    cfg = results["config"]
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    weights = results["branches"][mode]["combining"]["weights"]
    azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations)
    figure = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111, projection="3d", computed_zorder=False)
    surface = axis.plot_surface(
        azimuth_grid,
        elevation_grid,
        np.maximum(response_db, -60.0),
        cmap="viridis",
        vmin=-60.0,
        vmax=0.0,
        rcount=elevations.size,
        ccount=azimuths.size,
        linewidth=0,
        antialiased=True,
        alpha=0.85,
        zorder=1,
    )
    legend_labels = []
    for direction, label, marker, color in _array_response_directions(cfg):
        steering, _ = antenna_response(model, direction, cfg.signal.true_polarization)
        gain = abs(np.vdot(weights, steering)) / peak_response
        gain_db = 20.0 * np.log10(np.maximum(gain, np.finfo(float).tiny))
        legend_labels.append(_array_response_legend_label(direction, label, gain_db))
        axis.plot(
            [direction.azimuth_deg],
            [direction.elevation_deg],
            [np.clip(gain_db, -60.0, 0.0)],
            marker=marker,
            color=color,
            linestyle="none",
            markersize=10,
            label=label,
            zorder=10,
        )
    axis.set(
        title=f"{mode.upper()} 3D array response",
        xlabel="Azimuth (deg)",
        ylabel="Elevation (deg)",
        zlabel="Normalized response (dB)",
        xlim=(0.0, 360.0),
        ylim=(0.0, 90.0),
        zlim=(-60.0, 0.0),
    )
    axis.view_init(elev=28, azim=-125)
    axis.set_box_aspect((2.0, 1.0, 1.0))
    axis.legend(labels=legend_labels, loc="upper left", fontsize=9)
    figure.colorbar(surface, ax=axis, shrink=0.65, pad=0.12, label="Normalized response (dB)")
    figure.subplots_adjust(left=0.03, right=0.9, bottom=0.08, top=0.9)
    return figure


def _plot_detailed_array_response(results):
    cfg = results["config"]
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    azimuths = np.arange(0.0, 361.0, 2.0)
    sphere_elevations = np.arange(-90.0, 91.0, 2.0)
    upper_hemisphere = sphere_elevations >= 0.0
    elevations = sphere_elevations[upper_hemisphere]
    manifolds = np.empty(
        (len(model.active_antennas), sphere_elevations.size, azimuths.size),
        dtype=complex,
    )
    direction_type = type(cfg.signal.true_direction)
    for elevation_index, elevation in enumerate(sphere_elevations):
        for azimuth_index, azimuth in enumerate(azimuths):
            manifolds[:, elevation_index, azimuth_index] = antenna_response(
                model,
                direction_type(azimuth, elevation),
                cfg.signal.true_polarization,
            )[0]

    figures = {}
    for mode in _ordered_modes(results):
        weights = results["branches"][mode]["combining"]["weights"]
        sphere_response = np.abs(
            np.tensordot(weights.conj(), manifolds, axes=(0, 0))
        )
        response = sphere_response[upper_hemisphere]
        response_db = 20.0 * np.log10(
            np.maximum(response / np.max(response), np.finfo(float).tiny)
        )

        figure, axis = plt.subplots(figsize=(10, 6.5))
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
        legend_labels = []
        for direction, label, marker, color in _array_response_directions(cfg):
            steering, _ = antenna_response(model, direction, cfg.signal.true_polarization)
            gain = abs(np.vdot(weights, steering)) / np.max(response)
            gain_db = 20.0 * np.log10(np.maximum(gain, np.finfo(float).tiny))
            legend_labels.append(_array_response_legend_label(direction, label, gain_db))
            axis.plot(
                direction.azimuth_deg, direction.elevation_deg,
                marker=marker, color=color, linestyle="none", markersize=10,
                markeredgewidth=1.5, label=label,
                zorder=11 if marker == "x" else 10,
            )
        axis.set(
            title=f"{mode.upper()} array response",
            xlabel="Azimuth (deg)",
            ylabel="Elevation (deg)",
        )
        axis.legend(
            labels=legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.16),
            fontsize=9, frameon=False,
        )
        figure.colorbar(image, ax=axis, label="Normalized response (dB)")
        figure.tight_layout()
        figures[f"detailed_array_response_{mode}"] = figure
        figures[f"detailed_array_response_3d_{mode}"] = _plot_array_response_3d(
            results, mode, azimuths, elevations, response_db, np.max(response)
        )
        figures[f"detailed_array_response_spherical_{mode}"] = _plot_array_response_spherical(
            results, mode, azimuths, sphere_elevations, sphere_response
        )
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
    with plt.rc_context({"figure.max_open_warning": 0}):
        return {
            **_plot_detailed_rf_dfe(results),
            **_plot_detailed_combined(results),
            **_plot_detailed_array_response(results),
            **_plot_detailed_acquisition(results),
        }


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
