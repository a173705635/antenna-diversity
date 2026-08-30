import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from antenna_diversity.studies.plots import plot_study_tables, save_study_figures
from antenna_diversity.studies.report import build_summary, render_report, verify_report_links
from tests.study_helpers import completed_tables_for_reporting


_ACTIVE_ANTENNAS = ("TL", "TR", "BL", "BR")


def completed_manifest():
    return {
        "schema_version": 1,
        "name": "report-test",
        "status": "completed",
        "git_sha": "0123456789abcdef",
        "config_sha256": "a" * 64,
    }


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_study_plots_have_stable_keys_and_shared_heatmap_limits():
    figures = plot_study_tables(completed_tables_for_reporting())

    assert tuple(figures) == (
        "baseline_rankings", "direction_heatmaps", "power_curves",
        "robustness_curves", "combined_error_distributions",
        "multi_jammer_comparison", "full_chain_validation", "cn0_validation",
        "element_patterns", "polarization_losses", "coupling_matrices",
        "blockage_comparison",
    )
    heatmaps = figures["direction_heatmaps"].axes
    limits = {image.get_clim() for axis in heatmaps for image in axis.images}
    assert len(limits) == 1
    axis = next(axis for axis in heatmaps if axis.images)
    assert axis.lines[0].get_xdata()[0] == pytest.approx(33.0)
    assert axis.lines[0].get_ydata()[0] == pytest.approx(22.0)
    assert axis.patches[0].center == pytest.approx((33.0, 22.0))


def test_antenna_plots_use_saved_rows_only_and_share_comparison_limits(monkeypatch):
    import antenna_diversity.array.phone_model as phone_model

    monkeypatch.setattr(
        phone_model,
        "PhoneAntennaModel",
        lambda *_args, **_kwargs: pytest.fail("plotting must not evaluate antenna model"),
    )

    figures = plot_study_tables(completed_tables_for_reporting())

    assert {
        "element_patterns", "polarization_losses", "coupling_matrices",
        "blockage_comparison",
    } <= set(figures)
    for key in ("element_patterns", "blockage_comparison"):
        limits = {
            image.get_clim()
            for axis in figures[key].axes
            for image in axis.images
        }
        assert len(limits) == 1
    coupling_axes = [axis for axis in figures["coupling_matrices"].axes if axis.images]
    assert len({axis.images[0].get_clim() for axis in coupling_axes[:2]}) == 1
    assert len({axis.images[0].get_clim() for axis in coupling_axes[2:]}) == 1
    pattern_axis = next(axis for axis in figures["element_patterns"].axes if axis.images)
    assert tuple(pattern_axis.images[0]._Ax) == pytest.approx((0.0, 35.0, 210.0))


def test_antenna_plot_builders_ignore_missing_and_nonfinite_optional_values():
    tables = completed_tables_for_reporting()
    rows = list(tables["antenna_diagnostics"])
    rows[0] = {**rows[0], "response_db": float("nan")}
    rows[1] = {**rows[1], "azimuth_deg": None}
    rows.append({"record_kind": "source_match", "polarization_match_db": None})
    tables["antenna_diagnostics"] = tuple(rows)

    figures = plot_study_tables(tables)

    assert all(key in figures for key in (
        "element_patterns", "polarization_losses", "coupling_matrices",
        "blockage_comparison",
    ))


def test_blockage_figure_compares_saved_unblocked_and_blocked_response_values():
    tables = completed_tables_for_reporting()
    diagnostics = tables["antenna_diagnostics"]

    figure = plot_study_tables(tables)["blockage_comparison"]
    axes = {axis.get_title(): axis for axis in figure.axes if axis.images}
    unblocked = axes["True TL unblocked"].images[0].get_array()
    blocked = axes["True TL blocked"].images[0].get_array()
    expected_unblocked = sorted(
        float(row["response_db"])
        for row in diagnostics
        if row["record_kind"] == "pattern_grid"
        and row["model_view"] == "true" and row["element_name"] == "TL"
    )
    expected_blocked = sorted(
        float(row["response_db"])
        for row in diagnostics
        if row["record_kind"] == "blockage_grid"
        and row["model_view"] == "true" and row["element_name"] == "TL"
    )

    assert sorted(np.asarray(unblocked, dtype=float).ravel()) == pytest.approx(
        expected_unblocked
    )
    assert sorted(np.asarray(blocked, dtype=float).ravel()) == pytest.approx(
        expected_blocked
    )


def _rendered_rgba_at(image, x: float, y: float) -> tuple[int, ...]:
    axis = image.axes
    axis.figure.canvas.draw()
    rgba, left, bottom, _ = image.make_image(axis.figure.canvas.get_renderer())
    display_x, display_y = axis.transData.transform((x, y))
    column = int(display_x - left)
    row = int(display_y - bottom)
    return tuple(rgba[row, column])


def test_single_point_direction_heatmap_renders_a_nonzero_physical_cell():
    tables = completed_tables_for_reporting()
    row = {
        **tables["direction_scan"][0],
        "azimuth_deg": 120.0,
        "elevation_deg": 10.0,
        "target_azimuth_deg": 120.0,
        "target_elevation_deg": 10.0,
        "output_sinr_db": 12.0,
    }
    tables["direction_scan"] = (row,)

    figures = plot_study_tables(tables)
    axis = figures["direction_heatmaps"].axes[0]
    image = axis.images[0]

    assert axis.get_xlim() == pytest.approx((119.5, 120.5))
    assert axis.get_ylim() == pytest.approx((9.5, 10.5))
    assert image.get_array().shape == (1, 1)
    assert image.get_array()[0, 0] == pytest.approx(12.0)
    assert image.get_clim() == pytest.approx((11.5, 12.5))
    assert axis.get_xlim()[1] - axis.get_xlim()[0] > 0.0
    assert axis.get_ylim()[1] - axis.get_ylim()[0] > 0.0
    assert axis.lines[0].get_xdata()[0] == pytest.approx(120.0)
    assert axis.lines[0].get_ydata()[0] == pytest.approx(10.0)
    assert axis.patches[0].center == pytest.approx((120.0, 10.0))
    assert axis.patches[0].radius == pytest.approx(15.0)


@pytest.mark.parametrize(
    ("points", "expected_extent"),
    [
        (((120.0, 10.0), (120.0, 20.0)), (119.5, 120.5, 5.0, 25.0)),
        (((90.0, 10.0), (120.0, 10.0)), (75.0, 135.0, 9.5, 10.5)),
    ],
    ids=("single-azimuth-axis", "single-elevation-axis"),
)
def test_direction_heatmap_uses_cell_edges_for_single_axis_grids(
    points, expected_extent,
):
    tables = completed_tables_for_reporting()
    reference = tables["direction_scan"][0]
    tables["direction_scan"] = tuple(
        {
            **reference,
            "case_id": f"direction-{azimuth}-{elevation}",
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "target_azimuth_deg": points[0][0],
            "target_elevation_deg": points[0][1],
            "output_sinr_db": 12.0 + index,
        }
        for index, (azimuth, elevation) in enumerate(points)
    )

    figures = plot_study_tables(tables)
    axis = figures["direction_heatmaps"].axes[0]
    image = axis.images[0]

    assert (*axis.get_xlim(), *axis.get_ylim()) == pytest.approx(expected_extent)
    assert image.get_array().size == 2


@pytest.mark.parametrize(
    ("azimuth", "elevation", "expected_extent"),
    [
        (0.0, 0.0, (-0.5, 0.5, 0.0, 0.5)),
        (359.75, 90.0, (359.25, 360.25, 89.5, 90.0)),
    ],
)
def test_single_point_direction_extent_wraps_azimuth_and_clamps_elevation(
    azimuth, elevation, expected_extent,
):
    tables = completed_tables_for_reporting()
    tables["direction_scan"] = ({
        **tables["direction_scan"][0],
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "target_azimuth_deg": azimuth,
        "target_elevation_deg": elevation,
    },)

    figures = plot_study_tables(tables)
    axis = figures["direction_heatmaps"].axes[0]
    image = axis.images[0]

    assert (*axis.get_xlim(), *axis.get_ylim()) == pytest.approx(expected_extent)
    assert image.get_array().shape == (1, 1)


def test_direction_heatmap_renders_unsorted_nonuniform_centers_at_true_midpoints():
    tables = completed_tables_for_reporting()
    reference = tables["direction_scan"][0]
    coordinates = (
        (40.0, 20.0), (0.0, 10.0), (10.0, 20.0),
        (40.0, 10.0), (0.0, 20.0), (10.0, 10.0),
    )
    values = {0.0: 10.0, 10.0: 20.0, 40.0: 30.0}
    tables["direction_scan"] = tuple(
        {
            **reference,
            "case_id": f"direction-{index}",
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "target_azimuth_deg": 300.0,
            "target_elevation_deg": 80.0,
            "output_sinr_db": values[azimuth],
        }
        for index, (azimuth, elevation) in enumerate(coordinates)
    )

    figures = plot_study_tables(tables)
    axis = figures["direction_heatmaps"].axes[0]
    image = axis.images[0]

    assert image.get_array().tolist() == [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]
    expected = {
        10.0: tuple(image.to_rgba(10.0, bytes=True)),
        20.0: tuple(image.to_rgba(20.0, bytes=True)),
        30.0: tuple(image.to_rgba(30.0, bytes=True)),
    }
    assert _rendered_rgba_at(image, 4.0, 12.0) == expected[10.0]
    assert _rendered_rgba_at(image, 6.0, 12.0) == expected[20.0]
    assert _rendered_rgba_at(image, 24.0, 12.0) == expected[20.0]
    assert _rendered_rgba_at(image, 26.0, 12.0) == expected[30.0]
    assert axis.get_xlim() == pytest.approx((-5.0, 55.0))
    assert axis.get_ylim() == pytest.approx((5.0, 25.0))


def test_direction_heatmap_rejects_duplicate_coordinates_without_leaking_figures():
    tables = completed_tables_for_reporting()
    reference = tables["direction_scan"][0]
    tables["direction_scan"] = (
        reference,
        {**reference, "case_id": "duplicate", "output_sinr_db": 99.0},
    )
    external = plt.figure()
    figures_before = set(plt.get_fignums())

    with pytest.raises(ValueError, match="duplicate direction coordinate"):
        plot_study_tables(tables)

    assert set(plt.get_fignums()) == figures_before
    assert plt.fignum_exists(external.number)


@pytest.mark.parametrize("field", ["azimuth_deg", "elevation_deg"])
def test_direction_heatmap_rejects_nonfinite_axis_coordinates(field):
    tables = completed_tables_for_reporting()
    tables["direction_scan"] = ({
        **tables["direction_scan"][0], field: float("nan"),
    },)

    with pytest.raises(ValueError, match=field):
        plot_study_tables(tables)


class _ExplodingFloat:
    def __float__(self):
        raise RuntimeError("builder exploded")


class _ExplodingString:
    def __str__(self):
        raise RuntimeError("builder exploded")


class _ExplodingBoolean:
    def __bool__(self):
        raise RuntimeError("builder exploded")


def test_first_builder_failure_repeatedly_closes_its_figure_and_preserves_external_one():
    tables = completed_tables_for_reporting()
    tables["baseline_rankings"] = ({
        **tables["baseline_rankings"][0], "rank": None,
    },)
    external = plt.figure()
    figures_before = set(plt.get_fignums())

    for _ in range(20):
        with pytest.raises(ValueError, match="rank"):
            plot_study_tables(tables)

        assert set(plt.get_fignums()) == figures_before
        assert plt.fignum_exists(external.number)


@pytest.mark.parametrize(
    ("table_name", "field", "bad_value"),
    [
        ("direction_scan", "azimuth_deg", _ExplodingFloat()),
        ("power_scan", "power_dbm", _ExplodingFloat()),
        ("robustness_scans", "normalized_severity", _ExplodingFloat()),
        ("combined_errors", "level", _ExplodingString()),
        ("multi_jammer", "template", _ExplodingString()),
        ("full_chain_validation", "acquired", _ExplodingBoolean()),
        ("full_chain_cn0", "case_id", _ExplodingString()),
    ],
    ids=("directions", "power", "robustness", "combined", "multi-jammer", "full-chain", "cn0"),
)
def test_each_later_builder_closes_its_current_figure_on_mid_build_error(
    table_name, field, bad_value,
):
    tables = completed_tables_for_reporting()
    tables[table_name] = ({**tables[table_name][0], field: bad_value},)
    external = plt.figure()
    figures_before = set(plt.get_fignums())

    with pytest.raises(RuntimeError, match="builder exploded"):
        plot_study_tables(tables)

    assert set(plt.get_fignums()) == figures_before
    assert plt.fignum_exists(external.number)


@pytest.mark.parametrize(
    ("record_kind", "field"),
    [
        ("pattern_grid", "response_db"),
        ("source_match", "polarization_match_db"),
        ("coupling_entry", "response_real"),
        ("blockage_grid", "response_db"),
    ],
)
def test_each_antenna_builder_closes_its_partial_figure_on_error(
    record_kind, field,
):
    tables = completed_tables_for_reporting()
    rows = list(tables["antenna_diagnostics"])
    index = next(
        index for index, row in enumerate(rows)
        if row["record_kind"] == record_kind
    )
    rows[index] = {**rows[index], field: _ExplodingFloat()}
    tables["antenna_diagnostics"] = tuple(rows)
    external = plt.figure()
    figures_before = set(plt.get_fignums())

    with pytest.raises(RuntimeError, match="builder exploded"):
        plot_study_tables(tables)

    assert set(plt.get_fignums()) == figures_before
    assert plt.fignum_exists(external.number)


def test_successful_builders_and_generic_save_leave_caller_owned_figures_open(tmp_path: Path):
    external = plt.figure()
    figures = plot_study_tables(completed_tables_for_reporting())

    save_study_figures(figures, tmp_path)

    assert plt.fignum_exists(external.number)
    assert all(plt.fignum_exists(figure.number) for figure in figures.values())


def test_plot_construction_failure_closes_only_partially_built_figures():
    """A later builder failure must release earlier figures, not external ones."""
    tables = completed_tables_for_reporting()
    tables["direction_scan"] = tuple(
        {**row, "target_azimuth_deg": None} for row in tables["direction_scan"]
    )
    external = plt.figure()
    figures_before = set(plt.get_fignums())

    for _ in range(20):
        with pytest.raises(ValueError, match="target_azimuth_deg"):
            plot_study_tables(tables)

        assert set(plt.get_fignums()) == figures_before
        assert plt.fignum_exists(external.number)


def test_report_contains_evidence_boundaries_and_reproduction_links(tmp_path: Path):
    tables = completed_tables_for_reporting()
    manifest = completed_manifest()
    selection = {
        "cases": [{"case_id": "best-001", "active_antennas": ["TL", "BR"], "selection_reasons": ["best"]}],
        "unselected_roles": {"knee": "no crossing"},
    }
    figures = {"baseline_rankings": Path("figures/baseline_rankings.png")}
    table_paths = {name: Path("tables") / f"{name}.csv" for name in tables}
    summary = build_summary(
        tables, manifest, selection, active_antennas=_ACTIVE_ANTENNAS
    )
    report = render_report(summary, manifest, figures, table_paths)

    assert "比较性研究" in report
    assert "不代表真实 Mate 70 产品性能" in report
    assert "analytic_output_sinr_db" in report
    assert "final_cn0_db_hz" in report
    assert "未建模" in report
    assert "配置哈希" in report
    assert "two_jammer_separated" in report
    assert "TL+BR" in report
    assert "12.0" in report
    assert "最佳组合" in report
    assert "全方向" in report
    assert "可分辨区域" in report
    assert summary["antenna_model"] == {
        "source": "parametric",
        "label": "synthetic_phone_corner_v1",
        "measured_pattern_sha256": None,
        "active_antennas": ["TL", "TR", "BL", "BR"],
    }
    assert "手机天线模型证据" in report
    assert "parametric" in report
    assert "单阵元方向图" in report
    assert "极化匹配增益" in report
    assert "极化匹配损耗" not in report
    match_rows = [
        row for row in tables["antenna_diagnostics"]
        if row["record_kind"] == "source_match"
    ]
    assert match_rows
    assert all("polarization_match_db" in row for row in match_rows)
    assert all(float(row["polarization_match_db"]) <= 0.0 for row in match_rows)
    assert "互耦幅相矩阵" in report
    assert "遮挡衰减对比" in report
    assert "相关外部天线噪声" in report
    assert "全波平台散射" in report
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("active_antennas", [("TL", "BR"), ("TL", "TR", "BR")])
def test_summary_uses_resolved_active_antennas_not_four_corner_diagnostic_rows(
    active_antennas,
):
    summary = build_summary(
        completed_tables_for_reporting(),
        completed_manifest(),
        {},
        active_antennas=active_antennas,
    )

    assert summary["antenna_model"]["active_antennas"] == list(active_antennas)


def test_plots_allow_constant_and_failure_only_rows_without_warnings(tmp_path: Path):
    tables = completed_tables_for_reporting()
    tables["direction_scan"] = tuple({**row, "output_sinr_db": 12.0} for row in tables["direction_scan"])
    tables["full_chain_cn0"] = ()
    tables["full_chain_validation"] = tuple({**row, "acquired": False, "tracking": False, "final_cn0_db_hz": None} for row in tables["full_chain_validation"])

    figures = plot_study_tables(tables)
    paths = save_study_figures(figures, tmp_path)

    assert tuple(paths) == tuple(figures)
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_figure_save_failure_keeps_existing_png_and_removes_temp(monkeypatch, tmp_path: Path):
    """A save interruption must not replace a finished PNG with partial bytes."""
    figures = plot_study_tables(completed_tables_for_reporting())
    destination = tmp_path / "baseline_rankings.png"
    destination.write_bytes(b"complete-old-png")

    def partial_then_fail(path, **kwargs):
        Path(path).write_bytes(b"partial-new-png")
        raise RuntimeError("interrupted PNG write")

    monkeypatch.setattr(figures["baseline_rankings"], "savefig", partial_then_fail)

    with pytest.raises(RuntimeError, match="interrupted PNG write"):
        save_study_figures(figures, tmp_path)

    assert destination.read_bytes() == b"complete-old-png"
    assert not list(tmp_path.glob(".baseline_rankings.png.*.tmp"))


def test_summary_rejects_missing_required_table():
    tables = completed_tables_for_reporting()
    tables.pop("power_scan")

    with pytest.raises(ValueError, match="missing required table: power_scan"):
        build_summary(
            tables, completed_manifest(), {}, active_antennas=_ACTIVE_ANTENNAS
        )


def test_reporting_rejects_missing_or_invalid_saved_metadata():
    tables = completed_tables_for_reporting()
    tables["baseline_rankings"] = ({**tables["baseline_rankings"][0], "rank": None},)
    with pytest.raises(ValueError, match="rank"):
        build_summary(
            tables, completed_manifest(), {}, active_antennas=_ACTIVE_ANTENNAS
        )
    with pytest.raises(ValueError, match="rank"):
        plot_study_tables(tables)

    tables = completed_tables_for_reporting()
    tables["direction_scan"] = tuple({key: value for key, value in row.items() if key != "target_azimuth_deg"} for row in tables["direction_scan"])
    with pytest.raises(ValueError, match="target_azimuth_deg"):
        plot_study_tables(tables)


def test_plots_do_not_fabricate_zero_bars_for_unavailable_science_values():
    tables = completed_tables_for_reporting()
    tables["baseline_rankings"] = ({**tables["baseline_rankings"][0], "p10_resolvable_db": None, "median_resolvable_db": float("nan")},)
    tables["multi_jammer"] = ({**tables["multi_jammer"][0], "output_sinr_db": None},)
    tables["full_chain_validation"] = tuple({**row, "analytic_output_sinr_db": None} for row in tables["full_chain_validation"])

    figures = plot_study_tables(tables)

    assert not figures["baseline_rankings"].axes[0].patches
    assert not figures["multi_jammer_comparison"].axes[0].patches
    assert not figures["full_chain_validation"].axes[1].patches


def test_full_chain_plot_includes_blind_rates_without_fake_analytic_bar():
    tables = completed_tables_for_reporting()
    tables["full_chain_validation"] += ({
        "case_id": "best-001",
        "selection_reasons": ["best"],
        "mode": "blind_null",
        "run_index": 0,
        "seed": 7,
        "acquired": True,
        "tracking": True,
        "status": "tracking",
        "acquisition_metric": 3.2,
        "analytic_output_sinr_db": None,
        "final_cn0_db_hz": 36.5,
        "final_doppler_error_hz": 4.0,
    },)

    figure = plot_study_tables(tables)["full_chain_validation"]

    rate_labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    analytic_labels = [tick.get_text() for tick in figure.axes[1].get_xticklabels()]
    assert "BLIND_NULL" in rate_labels
    assert "BLIND_NULL" not in analytic_labels


def test_summary_names_rank_roles_and_validates_knee_evidence():
    tables = completed_tables_for_reporting()
    second = {**tables["baseline_rankings"][0], "active_antennas": ["TR", "BL"], "display_label": "MVDR(TR+BL)", "rank": 2}
    third = {**tables["baseline_rankings"][0], "active_antennas": ["TL", "TR"], "display_label": "MVDR(TL+TR)", "rank": 3}
    tables["baseline_rankings"] = (tables["baseline_rankings"][0], second, third)
    summary = build_summary(
        tables, completed_manifest(), {"cases": [], "unselected_roles": {}},
        active_antennas=_ACTIVE_ANTENNAS,
    )

    assert summary["baseline"]["best"]["rank"] == 1
    assert summary["baseline"]["typical"]["rank"] == 2
    assert summary["baseline"]["worst"]["rank"] == 3
    assert summary["knee"] == {"status": "unavailable", "reason": "not selected"}
    with pytest.raises(ValueError, match="knee"):
        build_summary(
            tables, completed_manifest(),
            {"cases": [{"selection_reasons": ["knee"]}],
             "unselected_roles": {"knee": "no crossing"}},
            active_antennas=_ACTIVE_ANTENNAS,
        )


@pytest.mark.parametrize("rank_values", [(2, 3), (1, 3), (1, 1), (0, 1), (True, 2)])
def test_reporting_requires_contiguous_one_based_ranks(rank_values):
    tables = completed_tables_for_reporting()
    first = {**tables["baseline_rankings"][0], "rank": rank_values[0]}
    second = {**tables["baseline_rankings"][0], "active_antennas": ["TR", "BL"], "rank": rank_values[1]}
    tables["baseline_rankings"] = (first, second)

    with pytest.raises(ValueError, match="rank"):
        build_summary(
            tables, completed_manifest(), {}, active_antennas=_ACTIVE_ANTENNAS
        )
    with pytest.raises(ValueError, match="rank"):
        plot_study_tables(tables)


def test_curve_plots_ignore_partial_and_all_unavailable_coordinates_without_errors():
    tables = completed_tables_for_reporting()
    tables["power_scan"] = (
        {**tables["power_scan"][0], "power_dbm": None},
        tables["power_scan"][1],
        {**tables["power_scan"][2], "output_sinr_db": float("nan")},
    )
    tables["robustness_scans"] = (
        {**tables["robustness_scans"][0], "normalized_severity": None},
        tables["robustness_scans"][1],
        {**tables["robustness_scans"][2], "mvdr_minus_egc_sinr_db": float("nan")},
    )
    tables["full_chain_cn0"] = (
        {**tables["full_chain_cn0"][0], "time_ms": None},
        tables["full_chain_cn0"][1],
        {**tables["full_chain_cn0"][2], "cn0_db_hz": float("nan")},
    )

    partial = plot_study_tables(tables)
    assert tuple(partial) == (
        "baseline_rankings", "direction_heatmaps", "power_curves", "robustness_curves",
        "combined_error_distributions", "multi_jammer_comparison", "full_chain_validation", "cn0_validation",
        "element_patterns", "polarization_losses", "coupling_matrices",
        "blockage_comparison",
    )
    for figure in partial.values():
        plt.close(figure)
    tables["power_scan"] = ({"mode": "mvdr", "active_antennas": ["TL", "BR"], "power_dbm": None, "output_sinr_db": None},)
    tables["robustness_scans"] = ({"factor": "phase", "active_antennas": ["TL", "BR"], "normalized_severity": None, "mvdr_minus_egc_sinr_db": None},)
    tables["full_chain_cn0"] = ({"case_id": "x", "mode": "mvdr", "run_index": 0, "time_ms": None, "cn0_db_hz": None},)
    unavailable = plot_study_tables(tables)
    assert not unavailable["power_curves"].axes[0].lines
    assert not unavailable["robustness_curves"].axes[0].lines
    assert not unavailable["cn0_validation"].axes[0].lines


def test_summary_binds_each_role_reason_to_its_own_selected_case():
    tables = completed_tables_for_reporting()
    best = tables["baseline_rankings"][0]
    second = {**best, "active_antennas": ["TR", "BL"], "display_label": "MVDR(TR+BL)", "rank": 2}
    tables["baseline_rankings"] = (best, second)
    with pytest.raises(ValueError, match="best"):
        build_summary(tables, completed_manifest(), {
            "cases": [
                {"active_antennas": ["TR", "BL"], "selection_reasons": ["best"]},
                {"active_antennas": ["TL", "BR"], "selection_reasons": ["manual"]},
            ],
        }, active_antennas=_ACTIVE_ANTENNAS)
    summary = build_summary(tables, completed_manifest(), {
        "cases": [
            {"active_antennas": ["TL", "BR"], "selection_reasons": ["best"]},
            {"active_antennas": ["TR", "BL"], "selection_reasons": ["worst"]},
            {"active_antennas": ["TL", "BR"], "selection_reasons": ["manual"]},
        ],
    }, active_antennas=_ACTIVE_ANTENNAS)
    assert summary["baseline"]["best"]["combination"] == "TL+BR"


def test_report_link_validation_requires_existing_confined_targets(tmp_path: Path):
    figures = tmp_path / "figures"
    tables = tmp_path / "tables"
    figures.mkdir()
    tables.mkdir()
    (figures / "baseline_rankings.png").write_bytes(b"plot")
    for name in completed_tables_for_reporting():
        (tables / f"{name}.csv").write_text("header\n", encoding="utf-8")
    report = render_report(
        build_summary(
            completed_tables_for_reporting(), completed_manifest(), {},
            active_antennas=_ACTIVE_ANTENNAS,
        ),
        completed_manifest(),
        {"baseline_rankings": Path("figures/baseline_rankings.png")},
        {name: Path("tables") / f"{name}.csv" for name in completed_tables_for_reporting()},
    )

    verify_report_links(report, tmp_path / "report.md")
    with pytest.raises(ValueError, match="escapes report directory"):
        verify_report_links("[bad](../outside.csv)\n", tmp_path / "report.md")
