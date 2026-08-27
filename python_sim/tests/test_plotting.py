import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import subprocess
import sys

import pytest

from antenna_diversity.analysis.plotting import (
    plot_acquisition,
    plot_array_response,
    plot_combined_spectra,
    plot_cn0,
    plot_results,
    plot_spectra,
    save_figures,
)
from antenna_diversity.config import (
    CWInterferer,
    Direction,
    default_config,
    mate70_default_config,
)
from antenna_diversity.pipeline import run_end_to_end


@pytest.fixture(autouse=True)
def close_figures_after_test():
    yield
    plt.close("all")


def test_acquisition_plot_contains_maps_slices_and_peak_markers(tmp_path):
    """Catch loss of the visual acquisition evidence requested by the user."""
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    figure = plot_acquisition(results)
    paths = save_figures({"acquisition_search": figure}, tmp_path)

    assert len(figure.axes) >= 6
    assert sum(bool(axis.images) for axis in figure.axes) >= 3
    assert sum(len(axis.lines) for axis in figure.axes) >= 9
    assert paths["acquisition_search"].is_file()
    assert paths["acquisition_search"].stat().st_size > 0
    plt.close(figure)


def test_cn0_plot_shows_tracked_modes_and_failed_acquisition_status():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    figure = plot_cn0(results)
    axis = figure.axes[0]

    assert [line.get_label() for line in axis.lines] == [
        "EGC",
        "MVDR",
        "BLIND_NULL",
    ]
    assert axis.get_xlabel() == "Tracking time (ms)"
    assert axis.get_ylabel() == "Estimated C/N₀ (dB-Hz)"
    assert axis.get_title() == "Tracking C/N₀ comparison"
    assert any(
        "SINGLE: acquisition failed" in text.get_text()
        for text in axis.texts
    )
    for line in axis.lines:
        assert np.all(np.isfinite(line.get_ydata()))


def test_cn0_plot_returns_annotated_figure_when_no_branch_tracks():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    for branch in results.branches.values():
        branch.acquisition["success"] = False
        branch.tracking = {
            "success": False,
            "status": "acquisition_failed",
            "prompt": np.array([], dtype=complex),
        }

    figure = plot_cn0(results)
    axis = figure.axes[0]

    assert not axis.lines
    assert len(axis.texts) == 4
    assert all("C/N₀ unavailable" in text.get_text() for text in axis.texts)


def test_cn0_plot_rejects_mismatched_tracking_arrays():
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)
    results.branches["mvdr"].tracking["epoch_start_samples"] = np.array([0])

    with pytest.raises(ValueError, match="mvdr.*length"):
        plot_cn0(results)


def test_plot_results_produces_six_named_figures(tmp_path):
    """Catch omission of a stage-level diagnostic figure from the full report."""
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    figures = plot_results(results)
    paths = save_figures(figures, tmp_path)

    assert set(figures) == {
        "rf_dfe_spectra",
        "combined_spectra",
        "array_response",
        "acquisition_search",
        "tracking_comparison",
        "cn0_comparison",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
    for figure in figures.values():
        plt.close(figure)


def test_combined_spectra_share_axes_and_show_all_modes(tmp_path):
    """Catch omitted branches or independent scales that hide jammer rejection."""
    cfg = default_config()
    cfg.plot.enable = False
    results = run_end_to_end(cfg)

    figure = plot_combined_spectra(results)
    paths = save_figures({"combined_spectra": figure}, tmp_path)
    main_axes = figure.axes

    assert len(main_axes) == 4
    assert [axis.get_title() for axis in main_axes] == [
        "SINGLE combined output",
        "EGC combined output",
        "MVDR combined output",
        "BLIND_NULL combined output",
    ]
    assert all(len(axis.lines) >= 3 for axis in main_axes)
    assert all(
        axis.lines[0].get_zorder()
        > max(line.get_zorder() for line in axis.lines[1:])
        for axis in main_axes
    )
    assert all(axis.get_xlim() == main_axes[0].get_xlim() for axis in main_axes)
    assert all(axis.get_ylim() == main_axes[0].get_ylim() for axis in main_axes)
    assert paths["combined_spectra"].is_file()
    assert paths["combined_spectra"].stat().st_size > 0
    plt.close(figure)


def test_combined_spectra_show_four_modes_in_stable_order():
    results = run_end_to_end(mate70_default_config())
    figure = plot_combined_spectra(results)
    titles = [
        axis.get_title()
        for axis in figure.axes
        if axis.get_ylabel() == "Magnitude (dB)"
    ]
    assert titles == [
        "SINGLE combined output",
        "EGC combined output",
        "MVDR combined output",
        "BLIND_NULL combined output",
    ]


def test_combined_spectra_accept_original_three_mode_configuration():
    cfg = mate70_default_config()
    cfg.diversity.modes = ("single", "egc", "mvdr")
    results = run_end_to_end(cfg)
    figure = plot_combined_spectra(results)
    data_axes = [
        axis
        for axis in figure.axes
        if axis.get_ylabel() == "Magnitude (dB)"
    ]
    assert len(data_axes) == 3


def test_phone_array_response_contains_four_heatmaps_and_direction_markers():
    """Catch dispatch back to the legacy 1-D response or omitted directions."""
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), offset_hz=250e3),
        CWInterferer(Direction(220.0, 20.0), offset_hz=-400e3),
        CWInterferer(Direction(300.0, 30.0), enable=False, offset_hz=600e3),
    )
    results = run_end_to_end(cfg)

    figure = plot_array_response(results)
    image_axes = [axis for axis in figure.axes if axis.images]

    assert len(image_axes) == 4
    assert all(axis.get_xlabel() == "Azimuth (deg)" for axis in image_axes)
    assert all(axis.get_ylabel() == "Elevation (deg)" for axis in image_axes)
    assert all(len(axis.lines) == 3 for axis in image_axes)
    assert all(axis.images[0].get_clim() == (-60.0, 0.0) for axis in image_axes)
    assert all(axis.collections for axis in image_axes[1:])
    colorbar_axes = [
        axis
        for axis in figure.axes
        if axis.get_ylabel() == "Normalized response (dB)"
    ]
    assert len(colorbar_axes) == 1
    plt.close(figure)


def test_phone_array_response_uses_stable_mode_order_with_extras():
    """Catch dict insertion order from changing standard-then-extra panels."""
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    results = run_end_to_end(cfg)
    branches = results.branches
    results.branches = {
        "mvdr": branches["mvdr"],
        "diagnostic": branches["single"],
        "single": branches["single"],
        "egc": branches["egc"],
    }

    figure = plot_array_response(results)
    image_axes = [axis for axis in figure.axes if axis.images]

    assert [axis.get_title() for axis in image_axes] == [
        "SINGLE array response",
        "EGC array response",
        "MVDR array response",
        "DIAGNOSTIC array response",
    ]
    plt.close(figure)


def test_phone_array_response_renders_nonempty_supported_subset():
    """Catch incomplete phone response reports silently rendering fewer panels."""
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    results = run_end_to_end(cfg)
    branches = results.branches
    results.branches = {
        "mvdr": branches["mvdr"],
        "single": branches["single"],
    }

    figure = plot_array_response(results)
    image_axes = [axis for axis in figure.axes if axis.images]
    assert [axis.get_title() for axis in image_axes] == [
        "SINGLE array response",
        "MVDR array response",
    ]


def test_phone_spectra_mark_enabled_interferers_in_each_sampling_domain():
    """Catch singular jammer markers or RF/DFE frequency-domain confusion."""
    cfg = mate70_default_config()
    cfg.plot.enable = False
    cfg.signal.duration_ms = 8
    cfg.tracking.max_epochs = 2
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), offset_hz=250e3),
        CWInterferer(Direction(220.0, 20.0), offset_hz=-400e3),
        CWInterferer(Direction(300.0, 30.0), enable=False, offset_hz=600e3),
    )
    results = run_end_to_end(cfg)

    rf_figure = plot_spectra(results)
    combined_figure = plot_combined_spectra(results)

    rf_target_mhz = (cfg.signal.if_hz + cfg.signal.doppler_hz) / 1e6
    rf_jammers_mhz = [
        (cfg.signal.if_hz + 250e3) / 1e6,
        (cfg.signal.if_hz - 400e3) / 1e6,
    ]
    baseband_target_mhz = cfg.signal.doppler_hz / 1e6
    baseband_jammers_mhz = [0.25, -0.4]
    for axis in rf_figure.axes[:2]:
        assert len(axis.lines) == 4
        assert [line.get_xdata()[0] for line in axis.lines[1:]] == [
            rf_target_mhz,
            *rf_jammers_mhz,
        ]
    dfe_axis = rf_figure.axes[2]
    assert len(dfe_axis.lines) == 4
    assert [line.get_xdata()[0] for line in dfe_axis.lines[1:]] == [
        baseband_target_mhz,
        *baseband_jammers_mhz,
    ]
    for axis in combined_figure.axes:
        assert len(axis.lines) == 4
        assert [line.get_xdata()[0] for line in axis.lines[1:]] == [
            baseband_target_mhz,
            *baseband_jammers_mhz,
        ]
    plt.close(rf_figure)
    plt.close(combined_figure)


@pytest.mark.parametrize("script", ["run_end_to_end.py", "run_legacy_ula.py"])
def test_cli_avoids_show_warning_with_noninteractive_backend(script):
    """Catch calls to plt.show() when the selected backend cannot display a GUI."""
    project_dir = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    completed = subprocess.run(
        [sys.executable, "-W", "error", script],
        cwd=project_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "cn0_comparison:" in completed.stdout
