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


def test_report_contains_evidence_boundaries_and_reproduction_links(tmp_path: Path):
    tables = completed_tables_for_reporting()
    manifest = completed_manifest()
    selection = {
        "cases": [{"case_id": "best-001", "selection_reasons": ["best"]}],
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


def test_summary_rejects_missing_required_table():
    tables = completed_tables_for_reporting()
    tables.pop("power_scan")

    with pytest.raises(ValueError, match="missing required table: power_scan"):
        build_summary(tables, completed_manifest(), {})


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
