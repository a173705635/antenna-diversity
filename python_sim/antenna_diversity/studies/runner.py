"""Sequential, checkpointed orchestration for the spatial research suite."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt

from .analytical import (
    run_baseline,
    run_combined_error_studies,
    run_multi_jammer_studies,
    run_single_factor_studies,
)
from .artifacts import StudyStore, write_csv_atomic, write_json_atomic
from .full_chain import run_selected_full_chain
from .models import ResolvedStudyConfig, SelectionResult, StudyCase
from .plots import plot_study_tables, save_study_figures
from .ranking import select_representative_cases, summarize_and_rank
from .report import build_summary, render_report, verify_report_links


STAGES = (
    "baseline", "single_factor", "combined_error", "multi_jammer",
    "selection", "full_chain", "report",
)


@dataclass(frozen=True)
class StageContext:
    """Inputs available to a stage, limited to immutable earlier checkpoints."""

    resolved: ResolvedStudyConfig
    prerequisites: Mapping[str, object]
    publication: "_Publication" | None = None
    manifest: Mapping[str, object] | None = None


class _Publication:
    """Confined fixed-name report publication; intentionally exposes no paths."""

    __slots__ = ("__root",)

    def __init__(self, root: Path) -> None:
        self.__root = root

    def write_tables(
        self, tables: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> dict[str, Path]:
        table_dir = self.__root / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name in _TABLE_ORDER:
            path = table_dir / f"{name}.csv"
            write_csv_atomic(path, _CSV_FIELDS[name], _csv_rows(tables[name], _CSV_FIELDS[name]))
            paths[name] = path.relative_to(self.__root)
        return paths

    def save_figures(
        self, tables: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> dict[str, Path]:
        figures = plot_study_tables(tables)
        try:
            return {
                name: path.relative_to(self.__root)
                for name, path in save_study_figures(
                    figures, self.__root / "figures"
                ).items()
            }
        finally:
            for figure in figures.values():
                plt.close(figure)

    def write_summary(self, summary: Mapping[str, object]) -> Path:
        path = self.__root / "summary.json"
        write_json_atomic(path, summary)
        return path.relative_to(self.__root)

    def publish_report(self, report: str) -> Path:
        path = self.__root / "report.md"
        verify_report_links(report, path)
        _write_text_atomic(path, report)
        return path.relative_to(self.__root)


def _publication_for(root: Path) -> _Publication:
    return _Publication(root)


def run_research(resolved: ResolvedStudyConfig, resume: bool = False) -> Path:
    """Run every pending stage in canonical order and return the study directory."""

    if set(STAGES) != StudyStore.REQUIRED_STAGES:
        raise RuntimeError("runner and store stage names disagree")
    store = StudyStore.resume(resolved) if resume else StudyStore.create(resolved)
    completed: dict[str, object] = {}
    rerun_from_here = False
    for name in STAGES:
        if not rerun_from_here and store.is_stage_completed(name):
            completed[name] = store.read_stage(name)
            continue
        rerun_from_here = True
        prerequisites = MappingProxyType({
            prior: _read_only(completed[prior]) for prior in STAGES[:STAGES.index(name)]
            if prior in completed
        })
        context = StageContext(
            resolved, prerequisites,
            publication=_publication_for(store.study_dir) if name == "report" else None,
            manifest=_read_only(store.manifest) if name == "report" else None,
        )
        try:
            payload = _STAGE_FUNCTIONS[name](context)
            store.write_stage(name, payload)
        except BaseException as error:
            store.fail_stage(name, error)
            raise
        completed[name] = payload
    store.complete()
    return store.study_dir


def _baseline(context: StageContext) -> dict[str, object]:
    return run_baseline(context.resolved).to_dict()


def _single_factor(context: StageContext) -> dict[str, object]:
    return run_single_factor_studies(context.resolved).to_dict()


def _combined_error(context: StageContext) -> dict[str, object]:
    return run_combined_error_studies(context.resolved).to_dict()


def _multi_jammer(context: StageContext) -> dict[str, object]:
    return run_multi_jammer_studies(context.resolved).to_dict()


def _selection(context: StageContext) -> dict[str, object]:
    baseline = _tables(context.prerequisites, "baseline")
    single = _tables(context.prerequisites, "single_factor")
    combined = _tables(context.prerequisites, "combined_error")
    multi = _tables(context.prerequisites, "multi_jammer")
    direction_rows = _rows(baseline, "direction")
    ranking = summarize_and_rank(direction_rows, context.resolved)
    selection = select_representative_cases(
        ranking, direction_rows,
        (*_rows(single, "single_factor"), *_rows(combined, "combined_error")),
        _rows(multi, "multi_jammer"), context.resolved,
    )
    return {"ranking": ranking.to_dict(), "selection": selection.to_dict()}


def _full_chain(context: StageContext) -> dict[str, object]:
    payload = _mapping(context.prerequisites["selection"], "selection checkpoint")
    selection = _selection_from_payload(_mapping(payload.get("selection"), "selection"))
    return run_selected_full_chain(context.resolved, selection).to_dict()


def _report(context: StageContext) -> dict[str, object]:
    if context.publication is None or context.manifest is None:
        raise ValueError("report stage requires publication context")
    tables = _report_tables(context.prerequisites)
    table_paths = context.publication.write_tables(tables)
    figure_paths = context.publication.save_figures(tables)
    selection = _mapping(_mapping(context.prerequisites["selection"], "selection checkpoint").get("selection"), "selection")
    final_manifest = {**context.manifest, "status": "completed"}
    summary = build_summary(
        tables,
        final_manifest,
        selection,
        active_antennas=context.resolved.simulation.array.active_antennas,
    )
    summary_path = context.publication.write_summary(summary)
    report = render_report(summary, final_manifest, figure_paths, table_paths)
    report_path = context.publication.publish_report(report)
    return {
        "artifacts": {
            "summary": summary_path.as_posix(),
            "report": report_path.as_posix(),
            "tables": {name: path.as_posix() for name, path in table_paths.items()},
            "figures": {name: path.as_posix() for name, path in figure_paths.items()},
        },
        "record_counts": {name: len(rows) for name, rows in tables.items()},
    }


_STAGE_FUNCTIONS: dict[str, Callable[[StageContext], dict[str, object]]] = {
    "baseline": _baseline,
    "single_factor": _single_factor,
    "combined_error": _combined_error,
    "multi_jammer": _multi_jammer,
    "selection": _selection,
    "full_chain": _full_chain,
    "report": _report,
}


_TABLE_ORDER = (
    "baseline_rankings", "direction_scan", "power_scan", "robustness_scans",
    "combined_errors", "multi_jammer", "full_chain_validation", "full_chain_cn0",
    "antenna_diagnostics",
)
_CSV_FIELDS: dict[str, tuple[str, ...]] = {
    "baseline_rankings": ("rank", "active_antennas", "combination_name", "mode", "all_count", "resolvable_count", "p10_all_db", "median_all_db", "minimum_all_db", "maximum_all_db", "p10_resolvable_db", "median_resolvable_db", "minimum_resolvable_db", "maximum_resolvable_db", "white_noise_gain", "maximum_baseline_m", "representative_p10_direction"),
    "direction_scan": ("stage", "case_id", "active_antennas", "simulation_active_antennas", "num_antennas", "mode", "display_label", "maximum_baseline_m", "reference_interferer_power_dbm", "reference_power_dbm", "elevation_index", "azimuth_index", "elevation_deg", "azimuth_deg", "target_azimuth_deg", "target_elevation_deg", "output_sinr_db", "white_noise_gain", "interferer_gain"),
    "power_scan": ("stage", "case_id", "active_antennas", "simulation_active_antennas", "num_antennas", "mode", "display_label", "maximum_baseline_m", "reference_interferer_power_dbm", "reference_power_dbm", "power_index", "power_dbm", "azimuth_deg", "elevation_deg", "output_sinr_db", "white_noise_gain", "interferer_gain"),
    "robustness_scans": ("stage", "case_id", "mvdr_case_id", "egc_case_id", "active_antennas", "num_antennas", "factor", "value", "signed_value", "absolute_normalized_severity", "corner", "axis", "level", "sample_index", "child_seed", "egc_output_sinr_db", "mvdr_output_sinr_db", "mvdr_minus_egc_sinr_db", "true_geometry_unchanged", "true_state_unchanged", "zero_matches_baseline", "default_loading", "realization", "egc_metrics", "mvdr_metrics", "mvdr_metric_record"),
    "combined_errors": ("stage", "case_id", "mvdr_case_id", "egc_case_id", "active_antennas", "num_antennas", "level", "sample_index", "child_seed", "egc_output_sinr_db", "mvdr_output_sinr_db", "mvdr_minus_egc_sinr_db", "true_geometry_unchanged", "true_state_unchanged", "realization", "egc_metrics", "mvdr_metrics", "mvdr_metric_record"),
    "multi_jammer": ("stage", "case_id", "active_antennas", "num_antennas", "mode", "template", "child_seed", "enabled_interferer_count", "output_sinr_db", "worst_interferer_gain", "covariance_rank", "covariance_condition_number", "interferers", "enabled_interferers", "metrics"),
    "full_chain_validation": ("case_id", "stage", "role", "run_index", "seed", "active_antennas", "mode", "acquired", "acquisition_metric", "tracking", "status", "reason", "final_cn0_db_hz", "final_doppler_error_hz", "target_direction_gain", "interferer_direction_gains", "worst_interferer_gain", "analytic_output_sinr_db"),
    "full_chain_cn0": ("case_id", "stage", "role", "run_index", "seed", "active_antennas", "mode", "time_ms", "cn0_db_hz"),
    "antenna_diagnostics": (
        "record_kind", "model_source", "model_label", "model_view",
        "element_name", "receiver", "source", "source_name", "azimuth_deg",
        "elevation_deg", "response_real", "response_imag", "response_db",
        "attenuation_db", "polarization_match_db", "measured_pattern_sha256",
    ),
}


def _report_tables(prerequisites: Mapping[str, object]) -> dict[str, tuple[Mapping[str, object], ...]]:
    baseline = _tables(prerequisites, "baseline")
    single = _tables(prerequisites, "single_factor")
    combined = _tables(prerequisites, "combined_error")
    multi = _tables(prerequisites, "multi_jammer")
    selection = _mapping(prerequisites["selection"], "selection checkpoint")
    ranking = _mapping(selection.get("ranking"), "ranking")
    full = _mapping(prerequisites["full_chain"], "full-chain checkpoint")
    return {
        "baseline_rankings": _rows(ranking, "summaries"),
        "direction_scan": _rows(baseline, "direction"),
        "power_scan": _rows(baseline, "power"),
        "robustness_scans": (*_rows(single, "single_factor"), *_rows(combined, "combined_error")),
        "combined_errors": _rows(combined, "combined_error"),
        "multi_jammer": _rows(multi, "multi_jammer"),
        "full_chain_validation": _rows(full, "metrics"),
        "full_chain_cn0": _rows(full, "cn0"),
        "antenna_diagnostics": _rows(baseline, "antenna_diagnostics"),
    }


def _tables(prerequisites: Mapping[str, object], stage: str) -> Mapping[str, object]:
    return _mapping(_mapping(prerequisites[stage], f"{stage} checkpoint").get("tables"), f"{stage} tables")


def _rows(payload: Mapping[str, object], name: str) -> tuple[Mapping[str, object], ...]:
    value = payload.get(name, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} rows must be a sequence")
    if not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} rows must be mappings")
    return tuple(value)  # type: ignore[return-value]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _selection_from_payload(payload: Mapping[str, object]) -> SelectionResult:
    raw_cases = _rows(payload, "cases")
    cases = tuple(StudyCase(
        case_id=str(row["case_id"]), stage=str(row["stage"]),
        active_antennas=tuple(str(value) for value in row["active_antennas"]),
        mode=str(row["mode"]), seed=row.get("seed") if isinstance(row.get("seed"), int) else None,
        parameters=dict(_mapping(row["parameters"], "case parameters")),
        selection_reasons=tuple(str(value) for value in row.get("selection_reasons", ())),
    ) for row in raw_cases)
    raw_unselected = _mapping(payload.get("unselected_roles", {}), "unselected roles")
    return SelectionResult(cases, {str(key): str(value) for key, value in raw_unselected.items()})


def _read_only(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _read_only(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_read_only(item) for item in value)
    return value


def _csv_rows(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> list[dict[str, object]]:
    return [{field: _csv_value(row.get(field)) for field in fields} for row in rows]


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_text_atomic(destination: Path, text: str) -> None:
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, destination)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
