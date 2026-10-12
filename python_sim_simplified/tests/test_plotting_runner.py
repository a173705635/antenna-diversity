import matplotlib
import os
from pathlib import Path
import subprocess
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.pipeline import run_end_to_end
from core.plotting import plot_detailed_results, plot_multipath_results, plot_results
from core.presets import default_config
from experiments.interference.full_chain import save_results_figures


def test_plot_results_returns_seven_independent_figures():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)

    figures = plot_results(results)

    assert set(figures) == {
        "rf_spectrum",
        "dfe_spectrum",
        "combined_spectra",
        "array_response",
        "acquisition",
        "tracking",
        "cn0",
    }
    assert len({id(figure) for figure in figures.values()}) == 7
    for figure in figures.values():
        assert figure.axes
        plt.close(figure)


def test_detailed_results_put_combined_output_spectra_in_one_figure():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)

    figures = plot_detailed_results(results)
    expected = {
        *(f"detailed_rf_dfe_{name}" for name in cfg.phone.active_antennas),
        "detailed_combined_spectra",
        *(
            f"detailed_array_response_{mode}"
            for mode in results["branches"]
        ),
        *(f"detailed_acquisition_{mode}" for mode in results["branches"]),
    }

    assert set(figures) == expected
    assert len(figures) == 15
    for name in cfg.phone.active_antennas:
        assert len(figures[f"detailed_rf_dfe_{name}"].axes) == 3
    combined_figure = figures["detailed_combined_spectra"]
    visible_axes = [axis for axis in combined_figure.axes if axis.get_visible()]
    assert len(visible_axes) == len(results["branches"])
    for axis, mode in zip(visible_axes, results["branches"]):
        assert axis.get_title() == mode.upper()
        assert axis.get_legend() is None
    assert len(combined_figure.legends) == 1
    for figure in figures.values():
        plt.close(figure)


def test_detailed_combined_spectra_use_absolute_scale_and_mark_frequencies():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)
    figures = plot_detailed_results(results)
    axes = [
        axis
        for axis in figures["detailed_combined_spectra"].axes
        if axis.get_visible()
    ]

    assert all(axis.get_ylim() == axes[0].get_ylim() for axis in axes)
    assert all(axis.get_ylabel() == "Magnitude (dB)" for axis in axes)
    assert all(len(axis.lines) == 4 for axis in axes)
    assert all(len(axis.patches) == 1 for axis in axes)
    for figure in figures.values():
        plt.close(figure)


def test_detailed_spatial_and_acquisition_figures_keep_diagnostics_separate():
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    results = run_end_to_end(cfg)
    figures = plot_detailed_results(results)

    for mode in results["branches"]:
        spatial = figures[f"detailed_array_response_{mode}"]
        spatial_axis = next(axis for axis in spatial.axes if axis.images)
        assert spatial_axis.images[0].get_clim() == (-60.0, 0.0)
        assert len(spatial_axis.lines) == 4
        assert {
            "Target",
            "CW 1",
            "Wideband 2",
            "Multipath 1",
        } <= {line.get_label() for line in spatial_axis.lines}

        acquisition = figures[f"detailed_acquisition_{mode}"]
        assert any(axis.images for axis in acquisition.axes)
        assert any(
            axis.get_xlabel() == "Code phase (chips)"
            for axis in acquisition.axes
        )
    for figure in figures.values():
        plt.close(figure)


def test_save_results_figures_keeps_existing_and_adds_multipath_outputs(tmp_path):
    cfg = default_config()
    cfg.signal.duration_ms = 20
    cfg.tracking.max_epochs = 10
    cfg.plot.show = False
    results = run_end_to_end(cfg)
    stale_overview = tmp_path / "rf_spectrum.png"
    stale_overview.write_bytes(b"stale")

    paths = save_results_figures(results, tmp_path)

    assert set(paths) >= {
        "detailed_rf_dfe_TL",
        "detailed_combined_spectra",
        "detailed_array_response_blind_null",
        "detailed_array_response_multipath_lcmv",
        "detailed_acquisition_egc",
        "detailed_acquisition_multipath_lcmv",
        "tracking",
        "cn0",
        "multipath_delay_profile",
        "multipath_correlation_comparison",
        "multipath_tracking_bias",
    }
    assert not stale_overview.exists()
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in paths.values()
    )


def test_detected_path_gets_an_independent_angle_figure():
    cfg = default_config()
    cfg.interferers = []
    cfg.signal.power_dbm = -95.0
    cfg.signal.duration_ms = 30
    cfg.signal.multipath_paths[0].relative_power_db = -3.0
    cfg.multipath_detection.detection_threshold_db = 3.0
    cfg.tracking.max_epochs = 20
    results = run_end_to_end(cfg)

    figures = plot_multipath_results(results)

    assert set(figures) >= {
        "multipath_delay_profile",
        "multipath_angle_path_1",
        "multipath_correlation_comparison",
        "multipath_tracking_bias",
    }
    assert all(figure.axes for figure in figures.values())
    for figure in figures.values():
        plt.close(figure)


def test_direct_runner_does_not_show_on_noninteractive_backend(tmp_path):
    source = Path(__file__).resolve().parents[1]
    standalone = tmp_path/'python_sim_simplified'
    import shutil
    shutil.copytree(source, standalone,
                    ignore=shutil.ignore_patterns('tests', 'results', 'validation', 'archive', '__pycache__'))
    script = standalone / "experiments" / "interference" / "full_chain.py"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["PYTHONWARNINGS"] = "error"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
