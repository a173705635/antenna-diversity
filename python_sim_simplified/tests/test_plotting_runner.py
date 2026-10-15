import matplotlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from core.config import Direction
from core.pipeline import run_end_to_end
from core.plotting import (
    _plot_array_response_spherical,
    _spherical_response_coordinates,
    plot_detailed_results, plot_results,
)
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
        *(f"detailed_array_response_3d_{mode}" for mode in results["branches"]),
        *(f"detailed_array_response_spherical_{mode}" for mode in results["branches"]),
        *(f"detailed_acquisition_{mode}" for mode in results["branches"]),
    }

    assert set(figures) == expected
    assert len(figures) == 21
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
        assert len(spatial_axis.lines) == 3
        assert {
            "Target",
            "CW 1",
            "Wideband 2",
        } <= {line.get_label() for line in spatial_axis.lines}

        surface = figures[f"detailed_array_response_3d_{mode}"]
        surface_axis = next(axis for axis in surface.axes if axis.name == "3d")
        assert surface_axis.get_zlim() == (-60.0, 0.0)
        assert surface_axis.collections[0].get_clim() == (-60.0, 0.0)
        assert surface_axis.get_xlabel() == "Azimuth (deg)"
        assert surface_axis.get_ylabel() == "Elevation (deg)"
        assert {line.get_label() for line in spatial_axis.lines} == {
            line.get_label() for line in surface_axis.lines
        }

        spherical = figures[f"detailed_array_response_spherical_{mode}"]
        spherical_axes = [axis for axis in spherical.axes if axis.name == "3d"]
        assert len(spherical_axes) == 2
        assert all(axis.get_zlim()[0] < -1 < 1 < axis.get_zlim()[1] for axis in spherical_axes)
        assert spherical.axes[-1].get_ylim() == (-40.0, 0.0)
        assert spherical_axes[1].azim - spherical_axes[0].azim == 180.0
        assert all(not axis.texts for axis in spherical_axes)
        assert [text.get_text() for text in spatial_axis.get_legend().get_texts()] == [
            text.get_text() for text in surface_axis.get_legend().get_texts()
        ]
        for legend in (spatial_axis.get_legend(), surface_axis.get_legend(), spherical.legends[0]):
            labels = [text.get_text() for text in legend.get_texts()]
            assert all("az " in label and "el " in label and "dB" in label for label in labels)
            assert {label.split(" | ")[0] for label in labels} == {
                line.get_label() for line in spatial_axis.lines
            }

        acquisition = figures[f"detailed_acquisition_{mode}"]
        assert any(axis.images for axis in acquisition.axes)
        assert any(
            axis.get_xlabel() == "Code phase (chips)"
            for axis in acquisition.axes
        )
    for figure in figures.values():
        plt.close(figure)


def test_save_results_figures_keeps_interference_outputs(tmp_path):
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
        "detailed_array_response_3d_mvdr",
        "detailed_array_response_3d_blind_null",
        "detailed_array_response_spherical_mvdr",
        "detailed_array_response_spherical_blind_null",
        "detailed_acquisition_egc",
        "tracking",
        "cn0",
    }
    assert not stale_overview.exists()
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in paths.values()
    )


def test_spherical_coordinates_preserve_response_radius_and_direction():
    azimuths = np.array([0, 90, 180, 270, 360])
    elevations = np.array([-90, 0, 90])
    radius = np.array([[0.25]*5, [1, 0.5, 0.1, 0.001, 1], [0.75]*5])
    x, y, z = _spherical_response_coordinates(azimuths, elevations, radius)
    np.testing.assert_allclose(np.sqrt(x*x+y*y+z*z), radius, atol=1e-15)
    np.testing.assert_allclose(x[1], [1, 0, -0.1, 0, 1], atol=1e-15)
    np.testing.assert_allclose(y[1], [0, 0.5, 0, -0.001, 0], atol=1e-15)
    np.testing.assert_allclose(z[[0, 2]], radius[[0, 2]]*np.array([[-1], [1]]))


def test_spherical_surface_uses_inward_db_display_radius(monkeypatch):
    # 0 dB is the outer sphere; -20 dB is halfway inward, not an outward bump.
    surfaces = []
    real_plot_surface = Axes3D.plot_surface

    def capture_surface(axis, x, y, z, **kwargs):
        surfaces.append(np.sqrt(x*x + y*y + z*z))
        return real_plot_surface(axis, x, y, z, **kwargs)

    monkeypatch.setattr(Axes3D, 'plot_surface', capture_surface)
    cfg = default_config()
    results = {'config': cfg, 'branches': {
        'single': {'combining': {'weights': np.array([1., 0., 0., 0.])}}
    }}
    response = np.array([[0.]*5, [4., 2., 0.4, 0.004, 4.], [3.]*5])
    figure = _plot_array_response_spherical(
        results, 'single', np.array([0., 90., 180., 270., 360.]),
        np.array([-90., 0., 90.]), response,
    )
    expected_radius = [
        [0.]*5,
        [1., 0.8494850021680094, 0.5, 0., 1.],
        [0.93753063169585]*5,
    ]
    assert len(surfaces) == 2
    for radius in surfaces:
        visible = np.isfinite(radius)
        np.testing.assert_allclose(radius[visible], np.asarray(expected_radius)[visible], atol=1e-15)
    assert np.all(np.isfinite(surfaces[0]) | np.isfinite(surfaces[1]))
    plt.close(figure)


def test_spherical_direction_ray_follows_exact_response_with_external_icon():
    cfg = default_config()
    cfg.phone.active_antennas = ('TL', 'TR')
    cfg.phone.phase_centers_m = {
        'TL': (0., 0., 0.), 'TR': (cfg.signal.wavelength_m / 2., 0., 0.)
    }
    cfg.signal.true_direction = Direction(60., 0.)
    cfg.interferers = []
    results = {'config': cfg, 'branches': {
        'egc': {'combining': {'weights': np.array([0.5, 0.5])}}
    }}
    figure = _plot_array_response_spherical(
        results, 'egc', np.array([0., 90., 180., 270., 360.]),
        np.array([-90., 0., 90.]),
        np.array([[1.]*5, [0., 1., 0., 1., 0.], [1.]*5]),
    )
    axis = next(axis for axis in figure.axes if axis.name == '3d')
    target = next(line for line in axis.lines if line.get_marker() == '*')
    # The icon shows direction outside the reference sphere, not apparent gain.
    expected_tip = [1.06 / 2., 1.06 * np.sqrt(3.) / 2., 0.]
    np.testing.assert_allclose(np.asarray(target.get_data_3d()).reshape(3),
                               expected_tip, atol=1e-15)
    ray = next(line for line in axis.lines if line.get_label() == '_direction_ray')
    # Analytic -3.01029995664 dB gives the inner radial endpoint, off the grid.
    inner_radius = 0.9247425010840047
    expected_ray = np.array([0.5, np.sqrt(3.) / 2., 0.])[:, None] * [inner_radius, 1.06]
    np.testing.assert_allclose(np.asarray(ray.get_data_3d()), expected_ray, atol=1e-15)
    plt.close(figure)


def test_spherical_reference_encloses_weak_response_and_marks_origin():
    cfg = default_config()
    cfg.interferers = []
    results = {'config': cfg, 'branches': {
        'single': {'combining': {'weights': np.array([1., 0., 0., 0.])}}
    }}
    figure = _plot_array_response_spherical(
        results, 'single', np.array([0., 90., 180., 270., 360.]),
        np.array([-90., 0., 90.]), np.full((3, 5), 1.),
    )
    for axis in (axis for axis in figure.axes if axis.name == '3d'):
        reference = next(line for line in axis.lines if line.get_label() == '_zero_db_reference')
        np.testing.assert_allclose(np.linalg.norm(reference.get_data_3d(), axis=0),
                                   1., atol=1e-15)
        origin = next(line for line in axis.lines if line.get_label() == '_sphere_origin')
        np.testing.assert_array_equal(np.asarray(origin.get_data_3d()).reshape(3), [0., 0., 0.])
    plt.close(figure)


def test_spherical_cutaway_hides_far_surface_that_fills_in_nulls(monkeypatch):
    surfaces = []
    real_plot_surface = Axes3D.plot_surface

    def capture_surface(axis, x, y, z, **kwargs):
        surfaces.append((axis, np.array([x, y, z])))
        return real_plot_surface(axis, x, y, z, **kwargs)

    monkeypatch.setattr(Axes3D, 'plot_surface', capture_surface)
    cfg = default_config()
    cfg.interferers = []
    results = {'config': cfg, 'branches': {
        'single': {'combining': {'weights': np.array([1., 0., 0., 0.])}}
    }}
    figure = _plot_array_response_spherical(
        results, 'single', np.arange(0., 361., 30.),
        np.arange(-90., 91., 30.), np.ones((7, 13)),
    )
    for axis, coordinates in surfaces:
        az, el = np.deg2rad([axis.azim, axis.elev])
        camera = [np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)]
        projected = np.einsum('i,ijk->jk', camera, coordinates)
        assert np.isnan(projected).any()
        assert np.nanmin(projected) >= -1e-15
    plt.close(figure)


def test_spherical_isotropic_response_still_forms_unit_sphere(monkeypatch):
    surfaces = []
    real_plot_surface = Axes3D.plot_surface

    def capture_surface(axis, x, y, z, **kwargs):
        surfaces.append(np.sqrt(x*x + y*y + z*z))
        return real_plot_surface(axis, x, y, z, **kwargs)

    monkeypatch.setattr(Axes3D, 'plot_surface', capture_surface)
    cfg = default_config()
    results = {'config': cfg, 'branches': {
        'single': {'combining': {'weights': np.array([2., 0., 0., 0.])}}
    }}
    figure = _plot_array_response_spherical(
        results, 'single', np.array([0., 90., 180., 270., 360.]),
        np.array([-90., 0., 90.]), np.full((3, 5), 2.),
    )
    for radius in surfaces:
        np.testing.assert_allclose(radius[np.isfinite(radius)], 1., atol=1e-15)
    assert np.all(np.isfinite(surfaces[0]) | np.isfinite(surfaces[1]))
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
