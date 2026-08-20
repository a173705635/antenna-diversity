import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import pytest

import run_spatial_research
from antenna_diversity.studies import resolve_study_config


TABLE_NAMES = (
    "baseline_rankings",
    "direction_scan",
    "power_scan",
    "robustness_scans",
    "combined_errors",
    "multi_jammer",
    "full_chain_validation",
    "full_chain_cn0",
    "antenna_diagnostics",
)
FIGURE_NAMES = (
    "baseline_rankings", "direction_heatmaps", "power_curves",
    "robustness_curves", "combined_error_distributions",
    "multi_jammer_comparison", "full_chain_validation", "cn0_validation",
    "element_patterns", "polarization_losses", "coupling_matrices",
    "blockage_comparison",
)


def _parametric_antenna_model() -> dict[str, object]:
    polarization = {
        "kind": "rhcp", "tilt_deg": 0.0,
        "az_real": 1.0, "az_imag": 0.0, "el_real": 0.0, "el_imag": 0.0,
    }
    patterns = {
        corner: {
            "peak_gain_db": float(index),
            "cosine_exponent": 1.5,
            "back_lobe_gain_db": -18.0,
            "nominal_polarization": polarization,
            "cross_polar_discrimination_db": 24.0,
            "cross_phase_deg": 10.0,
        }
        for index, corner in enumerate(("TL", "TR", "BL", "BR"))
    }
    state = {"pattern_kind": "parametric", "parametric_patterns": patterns}
    return {
        "model_label": "minimal parametric fixture",
        "model_source": "parametric",
        "true_state": state,
        "assumed_state": state,
    }


def _strict_load(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def _completed_study(tmp_path: Path) -> Path:
    completed = tmp_path / "completed"
    checkpoints = completed / "checkpoints"
    tables = completed / "tables"
    figures = completed / "figures"
    checkpoints.mkdir(parents=True)
    tables.mkdir()
    figures.mkdir()
    report = completed / "report.md"
    report.write_text("complete\n", encoding="utf-8")
    for name in TABLE_NAMES:
        (tables / f"{name}.csv").write_text("record\n1\n", encoding="utf-8")
    for name in FIGURE_NAMES:
        (figures / f"{name}.png").write_bytes(b"png")
    report_checkpoint = checkpoints / "report.json"
    report_checkpoint.write_text(json.dumps({
        "stage": "report",
        "payload": {
            "artifacts": {
                "summary": "summary.json",
                "report": "report.md",
                "tables": {name: f"tables/{name}.csv" for name in TABLE_NAMES},
                "figures": {name: f"figures/{name}.png" for name in FIGURE_NAMES},
            },
            "record_counts": {"baseline_rankings": 1},
        },
    }), encoding="utf-8")
    stages = {
        name: {
            "status": "completed",
            "record_count": 1,
            "checkpoint_sha256": "b" * 64,
            "checkpoint_path": f"checkpoints/{name}.json",
            "error": None,
        }
        for name in (
            "baseline", "single_factor", "combined_error", "multi_jammer",
            "selection", "full_chain", "report",
        )
    }
    (completed / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "name": "completed",
        "status": "completed",
        "created_at_utc": "2026-08-14T00:00:00Z",
        "updated_at_utc": "2026-08-14T00:00:01Z",
        "configuration_sha256": "a" * 64,
        "git_sha": None,
        "git_status": "git_unavailable",
        "environment": {},
        "artifacts": {
            "configuration": "resolved-config.json",
            "manifest": "manifest.json",
            "report": "checkpoints/report.json",
        },
        "stages": stages,
    }), encoding="utf-8")
    return completed


def test_cli_forwards_only_explicit_overrides(monkeypatch, tmp_path: Path, capsys):
    captured = {}

    def capture_resolver(config_path=None, preset=None, cli_overrides=None):
        captured["config_path"] = config_path
        captured["preset"] = preset
        captured["cli_overrides"] = dict(cli_overrides or {})
        return object()

    def capture_runner(resolved, resume=False):
        captured["resume"] = resume
        return _completed_study(tmp_path)

    monkeypatch.setattr(run_spatial_research, "resolve_study_config", capture_resolver)
    monkeypatch.setattr(run_spatial_research, "run_research", capture_runner)

    result = run_spatial_research.main([
        "--config", "studies/example_quick.json",
        "--name", "chosen-name",
        "--combinations", "TL+BR",
        "--resume",
    ])

    assert result == 0
    assert captured == {
        "config_path": Path("studies/example_quick.json"),
        "preset": None,
        "cli_overrides": {"name": "chosen-name", "combinations": "TL+BR"},
        "resume": True,
    }
    printed = capsys.readouterr().out.splitlines()
    assert printed
    assert all(Path(line.split(": ", 1)[1]).is_absolute() for line in printed)
    assert any(line.startswith("manifest: ") for line in printed)
    assert any(line.startswith("report: ") for line in printed)
    assert any(line.startswith("table.baseline_rankings: ") for line in printed)
    assert any(line.startswith("figure.baseline_rankings: ") for line in printed)


def test_cli_rejects_unknown_mode_before_running(monkeypatch):
    monkeypatch.setattr(
        run_spatial_research,
        "run_research",
        lambda *args, **kwargs: pytest.fail("runner must not be called"),
    )

    with pytest.raises(SystemExit) as error:
        run_spatial_research.main(["--modes", "mvdr,music"])

    assert error.value.code == 2


def test_cli_turns_configuration_errors_into_parser_errors(monkeypatch):
    def invalid_config(*args, **kwargs):
        raise ValueError("invalid study setting")

    monkeypatch.setattr(run_spatial_research, "resolve_study_config", invalid_config)

    with pytest.raises(SystemExit) as error:
        run_spatial_research.main(["--preset", "quick"])

    assert error.value.code == 2


def test_cli_preserves_unexpected_runtime_traceback(monkeypatch):
    monkeypatch.setattr(run_spatial_research, "resolve_study_config", lambda **kwargs: object())

    def fail_runtime(*args, **kwargs):
        raise RuntimeError("stage exploded")

    monkeypatch.setattr(run_spatial_research, "run_research", fail_runtime)

    with pytest.raises(RuntimeError, match="stage exploded"):
        run_spatial_research.main([])


def test_cli_refuses_zero_exit_for_incomplete_manifest(monkeypatch, tmp_path: Path):
    completed = _completed_study(tmp_path)
    manifest_path = completed / "manifest.json"
    manifest = _strict_load(manifest_path)
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(run_spatial_research, "resolve_study_config", lambda **kwargs: object())
    monkeypatch.setattr(run_spatial_research, "run_research", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="completed"):
        run_spatial_research.main([])


def test_editable_quick_example_resolves_every_visible_axis():
    path = Path(__file__).parents[1] / "studies" / "example_quick.json"

    resolved = resolve_study_config(path, preset="quick")

    assert resolved.study.preset == "quick"
    assert resolved.study.combinations == (("TL", "BR"), ("TL", "TR", "BR"))
    assert resolved.study.spatial_modes == ("egc", "mvdr")
    assert resolved.study.full_chain_modes == ("single", "egc", "mvdr")
    assert resolved.study.azimuths_deg.size
    assert resolved.study.elevations_deg.size
    assert resolved.study.powers_dbm.size
    assert resolved.study.direction_errors_deg.size
    assert resolved.study.position_errors_m.size
    assert resolved.study.amplitude_errors_db.size
    assert resolved.study.phase_errors_deg.size
    assert resolved.study.loading_factors.size


def test_real_minimal_cli_runs_all_stages_and_publishes_complete_evidence(
    tmp_path: Path, capsys,
):
    output_dir = tmp_path / "results"
    config_path = tmp_path / "minimal.json"
    example_path = Path(__file__).parents[1] / "studies" / "example_quick.json"
    data = json.loads(example_path.read_text(encoding="utf-8"))
    data["study"].update({
        "name": "minimal-cli",
        "output_dir": str(output_dir),
        "combinations": [["TL", "BR"]],
        "single_antennas": ["TL"],
        "azimuths_deg": [120.0],
        "elevations_deg": [10.0],
        "powers_dbm": [-95.0, -90.0],
        "direction_errors_deg": [-1.0, 0.0, 1.0],
        "position_errors_m": [-0.001, 0.0, 0.001],
        "amplitude_errors_db": [-0.5, 0.0, 0.5],
        "phase_errors_deg": [-5.0, 0.0, 5.0],
        "loading_factors": [0.001, 0.01, 0.1],
        "combined_error_samples_per_level": 1,
        "full_chain_run_count": 1,
        "automatic_roles": ["best"],
    })
    data["simulation"]["signal"]["duration_ms"] = 8
    data["simulation"]["array"] = {
        "active_antennas": ["TL", "BR"],
        "antenna_model": _parametric_antenna_model(),
    }
    config_path.write_text(json.dumps(data), encoding="utf-8")

    result = run_spatial_research.main(["--config", str(config_path), "--preset", "quick"])

    assert result == 0
    study_dir = output_dir / "minimal-cli"
    manifest = _strict_load(study_dir / "manifest.json")
    assert manifest["status"] == "completed"
    assert set(manifest["stages"]) == {
        "baseline", "single_factor", "combined_error", "multi_jammer",
        "selection", "full_chain", "report",
    }
    assert all(record["status"] == "completed" for record in manifest["stages"].values())

    json_paths = [study_dir / "manifest.json", study_dir / "resolved-config.json", study_dir / "summary.json"]
    json_paths.extend(sorted((study_dir / "checkpoints").glob("*.json")))
    assert len(json_paths) == 10
    for path in json_paths:
        _strict_load(path)
    summary = _strict_load(study_dir / "summary.json")
    assert summary["antenna_model"]["source"] == "parametric"
    assert summary["antenna_model"]["active_antennas"] == ["TL", "BR"]

    table_paths = {name: study_dir / "tables" / f"{name}.csv" for name in TABLE_NAMES}
    rows_by_table = {}
    for name, path in table_paths.items():
        assert path.is_file() and path.stat().st_size > 0
        with path.open(encoding="utf-8", newline="") as stream:
            rows_by_table[name] = list(csv.DictReader(stream))
    for name in TABLE_NAMES:
        if name != "full_chain_cn0":
            assert rows_by_table[name], name
    if not rows_by_table["full_chain_cn0"]:
        assert all(row["tracking"] == "False" for row in rows_by_table["full_chain_validation"])
    assert tuple(rows_by_table["antenna_diagnostics"][0]) == (
        "record_kind", "model_source", "model_label", "model_view",
        "element_name", "receiver", "source", "source_name", "azimuth_deg",
        "elevation_deg", "response_real", "response_imag", "response_db",
        "attenuation_db", "polarization_match_db", "measured_pattern_sha256",
    )

    figure_paths = sorted((study_dir / "figures").glob("*.png"))
    assert len(figure_paths) == 12
    assert all(path.stat().st_size > 0 for path in figure_paths)
    report_path = study_dir / "report.md"
    assert report_path.is_file() and report_path.stat().st_size > 0
    report = report_path.read_text(encoding="utf-8")
    assert "parametric" in report
    assert "minimal parametric fixture" in report
    for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", report):
        assert not Path(target).is_absolute()
        assert (study_dir / target).resolve().is_file()

    validation = rows_by_table["full_chain_validation"]
    assert {row["mode"] for row in validation} == {"single", "egc", "mvdr"}
    for row in validation:
        assert row["status"]
        cn0 = row["final_cn0_db_hz"]
        assert cn0 == "" or math.isfinite(float(cn0))

    printed_paths = [
        Path(line.split(": ", 1)[1])
        for line in capsys.readouterr().out.splitlines()
    ]
    assert len(printed_paths) == 23
    assert all(path.is_absolute() and path.is_file() for path in printed_paths)
