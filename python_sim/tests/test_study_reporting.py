import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

from antenna_diversity.studies.plots import plot_study_tables, save_study_figures
from antenna_diversity.studies.report import build_summary, render_report, verify_report_links
from tests.study_helpers import completed_tables_for_reporting


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
    )
    heatmaps = figures["direction_heatmaps"].axes
    limits = {image.get_clim() for axis in heatmaps for image in axis.images}
    assert len(limits) == 1
    axis = next(axis for axis in heatmaps if axis.images)
    assert axis.lines[0].get_xdata()[0] == pytest.approx(33.0)
    assert axis.lines[0].get_ydata()[0] == pytest.approx(22.0)
    assert axis.patches[0].center == pytest.approx((33.0, 22.0))


def test_plot_construction_failure_closes_only_partially_built_figures():
    """A later builder failure must release earlier figures, not external ones."""
    tables = completed_tables_for_reporting()
    tables["direction_scan"] = tuple(
        {**row, "target_azimuth_deg": None} for row in tables["direction_scan"]
    )
    external = plt.figure()
    figures_before = set(plt.get_fignums())

    for _ in range(3):
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
    summary = build_summary(tables, manifest, selection)
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
    json.dumps(summary, allow_nan=False)


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
        build_summary(tables, completed_manifest(), {})


def test_reporting_rejects_missing_or_invalid_saved_metadata():
    tables = completed_tables_for_reporting()
    tables["baseline_rankings"] = ({**tables["baseline_rankings"][0], "rank": None},)
    with pytest.raises(ValueError, match="rank"):
        build_summary(tables, completed_manifest(), {})
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


def test_summary_names_rank_roles_and_validates_knee_evidence():
    tables = completed_tables_for_reporting()
    second = {**tables["baseline_rankings"][0], "active_antennas": ["TR", "BL"], "display_label": "MVDR(TR+BL)", "rank": 2}
    third = {**tables["baseline_rankings"][0], "active_antennas": ["TL", "TR"], "display_label": "MVDR(TL+TR)", "rank": 3}
    tables["baseline_rankings"] = (tables["baseline_rankings"][0], second, third)
    summary = build_summary(tables, completed_manifest(), {"cases": [], "unselected_roles": {}})

    assert summary["baseline"]["best"]["rank"] == 1
    assert summary["baseline"]["typical"]["rank"] == 2
    assert summary["baseline"]["worst"]["rank"] == 3
    assert summary["knee"] == {"status": "unavailable", "reason": "not selected"}
    with pytest.raises(ValueError, match="knee"):
        build_summary(tables, completed_manifest(), {"cases": [{"selection_reasons": ["knee"]}], "unselected_roles": {"knee": "no crossing"}})


@pytest.mark.parametrize("rank_values", [(2, 3), (1, 3), (1, 1), (0, 1), (True, 2)])
def test_reporting_requires_contiguous_one_based_ranks(rank_values):
    tables = completed_tables_for_reporting()
    first = {**tables["baseline_rankings"][0], "rank": rank_values[0]}
    second = {**tables["baseline_rankings"][0], "active_antennas": ["TR", "BL"], "rank": rank_values[1]}
    tables["baseline_rankings"] = (first, second)

    with pytest.raises(ValueError, match="rank"):
        build_summary(tables, completed_manifest(), {})
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
    )
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
        })
    summary = build_summary(tables, completed_manifest(), {
        "cases": [
            {"active_antennas": ["TL", "BR"], "selection_reasons": ["best"]},
            {"active_antennas": ["TR", "BL"], "selection_reasons": ["worst"]},
            {"active_antennas": ["TL", "BR"], "selection_reasons": ["manual"]},
        ],
    })
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
        build_summary(completed_tables_for_reporting(), completed_manifest(), {}),
        completed_manifest(),
        {"baseline_rankings": Path("figures/baseline_rankings.png")},
        {name: Path("tables") / f"{name}.csv" for name in completed_tables_for_reporting()},
    )

    verify_report_links(report, tmp_path / "report.md")
    with pytest.raises(ValueError, match="escapes report directory"):
        verify_report_links("[bad](../outside.csv)\n", tmp_path / "report.md")
