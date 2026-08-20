from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pytest

from antenna_diversity.studies.runner import STAGES, run_research
from tests.study_helpers import resolved_for_test


class _FailOnceStages(dict):
    """Stage double whose target fails once, retaining observed order."""

    def __init__(self, target: str):
        self.calls: list[str] = []
        self.target = target
        self.failed = False
        super().__init__({name: self._stage(name) for name in STAGES})

    def _stage(self, name: str):
        def invoke(context):
            self.calls.append(name)
            if name == self.target and not self.failed:
                self.failed = True
                raise RuntimeError(name)
            return {"stage": name}
        return invoke


def test_runner_executes_each_stage_once_in_fixed_order(monkeypatch, tmp_path: Path):
    """A wrong runner ordering or omitted stage must be observable."""
    calls: list[str] = []
    resolved = resolved_for_test(output_dir=tmp_path, name="runner-test")
    monkeypatch.setattr(
        "antenna_diversity.studies.runner._STAGE_FUNCTIONS",
        {name: lambda context, name=name: calls.append(name) or {"stage": name}
         for name in STAGES},
    )

    output = run_research(resolved)

    assert calls == list(STAGES)
    assert output == tmp_path / resolved.study.name


def test_runner_resume_skips_valid_completed_stages(monkeypatch, tmp_path: Path):
    """A retry must retain valid earlier checkpoints and begin at the failure."""
    resolved = resolved_for_test(output_dir=tmp_path, name="runner-test")
    stages = _FailOnceStages("combined_error")
    monkeypatch.setattr("antenna_diversity.studies.runner._STAGE_FUNCTIONS", stages)
    with pytest.raises(RuntimeError, match="combined_error"):
        run_research(resolved)

    calls_before = tuple(stages.calls)
    run_research(resolved, resume=True)

    assert stages.calls.count("baseline") == 1
    assert stages.calls.count("single_factor") == 1
    assert tuple(stages.calls[:len(calls_before)]) == calls_before


def test_stage_context_exposes_only_immutable_prior_checkpoints(monkeypatch, tmp_path: Path):
    """A stage cannot read or mutate its own or a later stage's checkpoint."""
    seen: dict[str, object] = {}
    resolved = resolved_for_test(output_dir=tmp_path, name="isolated-runner")

    def baseline(context):
        seen["baseline"] = context.prerequisites
        return {"rows": [{"value": 1}]}

    def single_factor(context):
        seen["single_factor"] = context.prerequisites
        with pytest.raises(TypeError):
            context.prerequisites["later"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            context.prerequisites["baseline"]["rows"][0]["value"] = 2  # type: ignore[index]
        return {"rows": []}

    monkeypatch.setattr(
        "antenna_diversity.studies.runner._STAGE_FUNCTIONS",
        {**{name: lambda context, name=name: {"stage": name} for name in STAGES},
         "baseline": baseline, "single_factor": single_factor},
    )

    run_research(resolved)

    assert dict(seen["baseline"]) == {}
    assert tuple(seen["single_factor"]) == ("baseline",)


def test_pending_baseline_cannot_read_a_preexisting_future_report_checkpoint(monkeypatch, tmp_path: Path):
    """A future checkpoint must not give an earlier adapter a read capability."""
    from antenna_diversity.studies.artifacts import StudyStore

    resolved = resolved_for_test(output_dir=tmp_path, name="future-probe")
    store = StudyStore.create(resolved)
    store.write_stage("report", {"future": "must remain unavailable"})
    observed: dict[str, object] = {}

    def baseline(context):
        observed["has_store"] = hasattr(context, "store")
        observed["has_read_stage"] = hasattr(context, "read_stage")
        observed["has_study_dir"] = hasattr(context, "study_dir")
        observed["publication"] = context.publication
        observed["prerequisites"] = context.prerequisites
        return {"stage": "baseline"}

    monkeypatch.setattr(
        "antenna_diversity.studies.runner._STAGE_FUNCTIONS",
        {**{name: lambda context, name=name: {"stage": name} for name in STAGES}, "baseline": baseline},
    )

    run_research(resolved, resume=True)

    assert observed == {
        "has_store": False, "has_read_stage": False, "has_study_dir": False,
        "publication": None, "prerequisites": {},
    }


def test_runner_recovers_from_corrupt_checkpoint_at_that_stage(monkeypatch, tmp_path: Path):
    """A modified checkpoint must be recomputed, while valid predecessors remain skipped."""
    calls: list[str] = []
    resolved = resolved_for_test(output_dir=tmp_path, name="corrupt-runner")
    stages = {name: lambda context, name=name: calls.append(name) or {"stage": name}
              for name in STAGES}
    monkeypatch.setattr("antenna_diversity.studies.runner._STAGE_FUNCTIONS", stages)
    output = run_research(resolved)
    (output / "checkpoints" / "combined_error.json").write_text("{}", encoding="utf-8")

    run_research(resolved, resume=True)

    assert calls.count("baseline") == 1
    assert calls.count("single_factor") == 1
    assert calls.count("combined_error") == 2


def test_real_minimal_analytical_stages_reach_selection(monkeypatch, tmp_path: Path):
    """Runner passes real saved analytical rows into the selection adapter."""
    base = resolved_for_test(output_dir=tmp_path, name="real-runner")
    study = replace(
        base.study, combinations=(("TL", "BR"),), spatial_modes=("mvdr",),
        single_antennas=("TL",), azimuths_deg=[120.0], elevations_deg=[10.0],
        powers_dbm=[-90.0], direction_errors_deg=[0.0], position_errors_m=[0.0],
        amplitude_errors_db=[0.0], phase_errors_deg=[0.0], loading_factors=[base.simulation.diversity.diagonal_loading_factor],
        combined_error_samples_per_level=1, multi_jammer_templates=(), automatic_roles=("best",),
    )
    resolved = type(base).from_parts(study, base.simulation)
    from antenna_diversity.studies.runner import _STAGE_FUNCTIONS
    monkeypatch.setitem(_STAGE_FUNCTIONS, "full_chain", lambda context: {"metrics": [], "cn0": []})
    monkeypatch.setitem(_STAGE_FUNCTIONS, "report", lambda context: {"artifacts": {}, "record_counts": {}})

    output = run_research(resolved)

    selection = __import__("json").loads((output / "checkpoints" / "selection.json").read_text(encoding="utf-8"))["payload"]
    assert selection["selection"]["cases"][0]["selection_reasons"] == ["best"]


def test_report_stage_publishes_nine_linked_tables_and_twelve_figures(monkeypatch, tmp_path: Path):
    """Report publication must verify links after every target is atomically written."""
    from antenna_diversity.studies.runner import _publication_for, _report
    from antenna_diversity.studies.artifacts import StudyStore
    from antenna_diversity.studies.report import verify_report_links
    from tests.study_helpers import completed_tables_for_reporting

    resolved = resolved_for_test(output_dir=tmp_path, name="report-runner")
    simulation = resolved.simulation
    simulation.array.active_antennas = ("TL", "TR", "BR")
    resolved = type(resolved).from_parts(resolved.study, simulation)
    store = StudyStore.create(resolved)
    tables = completed_tables_for_reporting()
    checkpoints = {
        "baseline": {"tables": {"direction": tables["direction_scan"], "power": tables["power_scan"], "antenna_diagnostics": tables["antenna_diagnostics"]}},
        "single_factor": {"tables": {"single_factor": tables["robustness_scans"]}},
        "combined_error": {"tables": {"combined_error": tables["combined_errors"]}},
        "multi_jammer": {"tables": {"multi_jammer": tables["multi_jammer"]}},
        "selection": {"ranking": {"summaries": tables["baseline_rankings"]}, "selection": {"cases": [{"case_id": "best-001", "active_antennas": ["TL", "BR"], "selection_reasons": ["best"]}], "unselected_roles": {"knee": "none"}}},
        "full_chain": {"metrics": tables["full_chain_validation"], "cn0": tables["full_chain_cn0"]},
    }
    context = __import__("antenna_diversity.studies.runner", fromlist=["StageContext"]).StageContext(
        resolved, checkpoints, publication=_publication_for(store.study_dir), manifest=store.manifest,
    )

    payload = _report(context)

    assert set(payload["artifacts"]["tables"]) == set(tables)
    assert len(payload["artifacts"]["tables"]) == 9
    assert len(payload["artifacts"]["figures"]) == 12
    summary = __import__("json").loads(
        (store.study_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["antenna_model"]["active_antennas"] == ["TL", "TR", "BR"]
    verify_report_links((store.study_dir / "report.md").read_text(encoding="utf-8"), store.study_dir / "report.md")


def test_publication_repeatedly_saves_without_leaking_owned_figures(tmp_path: Path):
    """Repeated report publication must not accumulate its twelve owned figures."""
    from antenna_diversity.studies.runner import _publication_for
    from tests.study_helpers import completed_tables_for_reporting

    external = plt.figure()
    figures_before = set(plt.get_fignums())
    try:
        publication = _publication_for(tmp_path)
        for _ in range(3):
            publication.save_figures(completed_tables_for_reporting())

        assert set(plt.get_fignums()) == figures_before
    finally:
        for figure_number in set(plt.get_fignums()) - figures_before:
            plt.close(figure_number)
        plt.close(external)


def test_publication_closes_owned_figures_when_saving_raises(monkeypatch, tmp_path: Path):
    """A failed save must release every figure created by report publication."""
    from antenna_diversity.studies.runner import _publication_for
    from tests.study_helpers import completed_tables_for_reporting

    figures_before = set(plt.get_fignums())

    def fail_save(figures, output_dir):
        raise RuntimeError("figure save failed")

    monkeypatch.setattr("antenna_diversity.studies.runner.save_study_figures", fail_save)
    try:
        with pytest.raises(RuntimeError, match="figure save failed"):
            _publication_for(tmp_path).save_figures(completed_tables_for_reporting())

        assert set(plt.get_fignums()) == figures_before
    finally:
        for figure_number in set(plt.get_fignums()) - figures_before:
            plt.close(figure_number)


def test_report_context_publication_has_only_fixed_publication_actions(monkeypatch, tmp_path: Path):
    """Report adapters get fixed publication actions, never a writable artifact root."""
    observed: dict[str, object] = {}
    resolved = resolved_for_test(output_dir=tmp_path, name="publication-surface")

    def report(context):
        observed["context_attrs"] = set(vars(context))
        observed["publication_attrs"] = {
            name for name in dir(context.publication)
            if not name.startswith("_")
        }
        return {"artifacts": {}, "record_counts": {}}

    monkeypatch.setattr(
        "antenna_diversity.studies.runner._STAGE_FUNCTIONS",
        {**{name: lambda context, name=name: {"stage": name} for name in STAGES}, "report": report},
    )

    run_research(resolved)

    assert observed["context_attrs"] == {"resolved", "prerequisites", "publication", "manifest"}
    assert observed["publication_attrs"] == {"publish_report", "save_figures", "write_summary", "write_tables"}
