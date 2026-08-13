"""Resolution of reproducible spatial-study defaults, presets, JSON, and CLI."""

from __future__ import annotations

import copy
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from antenna_diversity.config import CWInterferer, Direction, RFStage, SimulationConfig, default_config

from .models import ErrorLevel, InterfererTemplate, ResolvedStudyConfig, StudyConfig
from .scenarios import parse_combination_spec


PRESET_NAMES = ("quick", "standard", "extended")
_ALL_COMBINATIONS = (
    ("TL", "TR"), ("TL", "BL"), ("TL", "BR"), ("TR", "BL"), ("TR", "BR"), ("BL", "BR"),
    ("TL", "TR", "BL"), ("TL", "TR", "BR"), ("TL", "BL", "BR"), ("TR", "BL", "BR"),
    ("TL", "TR", "BL", "BR"),
)


def _templates() -> tuple[InterfererTemplate, ...]:
    def interferer(azimuth: float, elevation: float, offset: float, phase: float) -> CWInterferer:
        return CWInterferer(Direction(azimuth, elevation), power_dbm=-90.0, offset_hz=offset, initial_phase_rad=phase)
    separated = (interferer(120, 10, 250e3, 0.8), interferer(250, 25, -180e3, 1.1))
    return (
        InterfererTemplate("two_jammer_separated", separated),
        InterfererTemplate("two_jammer_close", (separated[0], interferer(128, 12, -180e3, 1.1))),
        InterfererTemplate("three_jammer_stress", separated + (interferer(315, 40, 420e3, -0.4),)),
    )


def _study_for_preset(preset: str) -> StudyConfig:
    if preset not in PRESET_NAMES:
        raise ValueError(f"unknown preset: {preset}")
    linear_count, loading_count, combined_count, run_count, az_step, el_step, powers = {
        "quick": (5, 5, 50, 3, 30, 15, np.linspace(-110, -80, 4)),
        "standard": (13, 9, 500, 40, 10, 10, np.linspace(-110, -70, 9)),
        "extended": (25, 21, 5000, 200, 5, 5, np.linspace(-115, -65, 21)),
    }[preset]
    roles = ("best", "typical", "worst", "knee")
    if preset != "quick":
        roles += tuple(template.name for template in _templates())
    if preset == "extended":
        roles += ("combined_light", "combined_medium", "combined_heavy")
    return StudyConfig(
        schema_version=1, name="spatial-research", preset=preset, seed=42,
        output_dir=Path("results") / "studies", combinations=_ALL_COMBINATIONS,
        spatial_modes=("egc", "mvdr"), full_chain_modes=("single", "egc", "mvdr"),
        single_antennas=("TL", "TR", "BL", "BR"), ranking_mode="mvdr", exclusion_angle_deg=15.0,
        azimuths_deg=np.arange(0, 360, az_step), elevations_deg=np.arange(0, 91, el_step), powers_dbm=powers,
        direction_errors_deg=np.linspace(-12, 12, linear_count), position_errors_m=np.linspace(-0.012, 0.012, linear_count),
        amplitude_errors_db=np.linspace(-3, 3, linear_count), phase_errors_deg=np.linspace(-30, 30, linear_count),
        loading_factors=np.logspace(-5, 0, loading_count),
        error_levels=(ErrorLevel("light", 1.0, .001, .25, 3.0), ErrorLevel("medium", 3.0, .003, .75, 10.0), ErrorLevel("heavy", 8.0, .010, 2.0, 30.0)),
        combined_error_samples_per_level=combined_count, full_chain_run_count=run_count,
        multi_jammer_templates=_templates(), automatic_roles=roles, manual_cases=(),
    )


def _reject_unknown(data: Mapping[str, object], allowed: set[str], context: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown {context} key: {sorted(unknown)[0]}")


def _require_exact_keys(data: Mapping[str, object], required: set[str], context: str) -> None:
    _reject_unknown(data, required, context)
    missing = required - set(data)
    if missing:
        raise ValueError(f"missing {context} key: {sorted(missing)[0]}")


def _update_dataclass(value: object, updates: Mapping[str, object], context: str) -> object:
    if not is_dataclass(value):
        raise ValueError(f"{context} does not accept nested keys")
    allowed = {field.name for field in fields(value)}
    _reject_unknown(updates, allowed, context)
    values: dict[str, object] = {}
    for key, raw in updates.items():
        current = getattr(value, key)
        if key in {"true_direction", "assumed_direction", "direction"} and isinstance(raw, Mapping):
            _reject_unknown(raw, {"azimuth_deg", "elevation_deg"}, f"{context}.{key}")
            values[key] = Direction(**raw)
        elif key == "interferers":
            if not isinstance(raw, (tuple, list)):
                raise ValueError("simulation.interferers must be a list")
            values[key] = tuple(_interferer(item) for item in raw)
        elif isinstance(raw, Mapping) and current is not None and is_dataclass(current):
            values[key] = _update_dataclass(current, raw, f"{context}.{key}")
        elif isinstance(current, np.ndarray):
            values[key] = np.asarray(raw, dtype=current.dtype)
        elif isinstance(current, Path):
            values[key] = Path(raw)  # type: ignore[arg-type]
        elif isinstance(current, tuple) and key == "stages":
            if not isinstance(raw, (tuple, list)):
                raise ValueError("simulation.rf.stages must be a list")
            stages = []
            for stage in raw:
                if not isinstance(stage, Mapping):
                    raise ValueError("simulation.rf.stages entries must be objects")
                _reject_unknown(stage, {field.name for field in fields(RFStage)}, "simulation.rf.stages")
                stages.append(RFStage(**stage))
            values[key] = tuple(stages)
        else:
            values[key] = copy.deepcopy(raw)
    return replace(value, **values)


def _interferer(raw: object) -> CWInterferer:
    if not isinstance(raw, Mapping):
        raise ValueError("interferer must be an object")
    allowed = {field.name for field in fields(CWInterferer)}
    _reject_unknown(raw, allowed, "interferer")
    if "direction" not in raw:
        raise ValueError("missing interferer key: direction")
    data = dict(raw)
    direction = data.get("direction")
    if not isinstance(direction, Mapping):
        raise ValueError("interferer.direction must be an object")
    _require_exact_keys(direction, {"azimuth_deg", "elevation_deg"}, "interferer.direction")
    data["direction"] = Direction(**direction)
    try:
        return CWInterferer(**data)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"invalid interferer: {error}") from error


def _apply_simulation_updates(simulation: SimulationConfig, updates: Mapping[str, object]) -> SimulationConfig:
    allowed = {"signal", "phone", "array", "rf", "dfe", "diversity", "acquisition", "tracking", "interferers"}
    _reject_unknown(updates, allowed, "simulation")
    values: dict[str, object] = {}
    for name, raw in updates.items():
        if name == "interferers":
            if not isinstance(raw, (list, tuple)):
                raise ValueError("simulation.interferers must be a list")
            values[name] = tuple(_interferer(item) for item in raw)
        else:
            if not isinstance(raw, Mapping):
                raise ValueError(f"simulation.{name} must be an object")
            values[name] = _update_dataclass(getattr(simulation, name), raw, f"simulation.{name}")
    return replace(simulation, **values)


def _study_updates(
    study: StudyConfig,
    updates: Mapping[str, object],
    *,
    allow_cli_modes: bool = False,
) -> StudyConfig:
    allowed = {field.name for field in fields(StudyConfig)}
    if allow_cli_modes:
        allowed.add("modes")
    _reject_unknown(updates, allowed, "study")
    converted: dict[str, object] = {}
    axes = {"azimuths_deg", "elevations_deg", "powers_dbm", "direction_errors_deg", "position_errors_m", "amplitude_errors_db", "phase_errors_deg", "loading_factors"}
    for name, value in updates.items():
        if name == "combinations":
            converted[name] = parse_combination_spec(value)  # type: ignore[arg-type]
        elif name in {"spatial_modes", "full_chain_modes", "single_antennas", "automatic_roles"}:
            converted[name] = tuple(value) if not isinstance(value, str) else tuple(part.strip() for part in value.split(",") if part.strip())
        elif name == "modes":  # CLI shorthand, retained outside the serialized model.
            converted["spatial_modes"] = tuple(value) if not isinstance(value, str) else tuple(part.strip() for part in value.split(",") if part.strip())
        elif name in axes:
            try:
                converted[name] = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"study.{name} must be a real numeric axis") from error
        elif name == "output_dir":
            converted[name] = Path(value)  # type: ignore[arg-type]
        elif name == "error_levels":
            if not isinstance(value, (list, tuple)):
                raise ValueError("study.error_levels must be a list")
            levels = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError("study.error_levels entries must be objects")
                _require_exact_keys(item, {field.name for field in fields(ErrorLevel)}, "study.error_levels")
                try:
                    levels.append(ErrorLevel(**item))
                except TypeError as error:
                    raise ValueError(f"invalid study.error_levels entry: {error}") from error
            converted[name] = tuple(levels)
        elif name == "multi_jammer_templates":
            if not isinstance(value, (list, tuple)):
                raise ValueError("study.multi_jammer_templates must be a list")
            templates = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError("study.multi_jammer_templates entries must be objects")
                _require_exact_keys(item, {field.name for field in fields(InterfererTemplate)}, "study.multi_jammer_templates")
                interferers = item.get("interferers")
                if not isinstance(interferers, (list, tuple)):
                    raise ValueError("study.multi_jammer_templates.interferers must be a list")
                templates.append(InterfererTemplate(str(item.get("name", "")), tuple(_interferer(entry) for entry in interferers)))
            converted[name] = tuple(templates)
        else:
            converted[name] = copy.deepcopy(value)
    return replace(study, **converted)


def resolve_study_config(config_path: Path | None = None, preset: str | None = None, cli_overrides: Mapping[str, object] | None = None) -> ResolvedStudyConfig:
    """Resolve defaults, preset, JSON, and CLI into one validated snapshot."""

    selected_preset = preset or "standard"
    study = _study_for_preset(selected_preset)
    simulation = default_config()
    json_study: Mapping[str, object] = {}
    json_simulation: Mapping[str, object] = {}
    if config_path is not None:
        try:
            data = json.loads(Path(config_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON configuration: {error.msg}") from error
        if not isinstance(data, Mapping):
            raise ValueError("JSON configuration must be an object")
        _reject_unknown(data, {"study", "simulation"}, "JSON")
        json_study = data.get("study", {})
        json_simulation = data.get("simulation", {})
        if not isinstance(json_study, Mapping) or not isinstance(json_simulation, Mapping):
            raise ValueError("JSON study and simulation entries must be objects")
    study = _study_updates(study, json_study)
    simulation = _apply_simulation_updates(simulation, json_simulation)
    if cli_overrides:
        study = _study_updates(study, cli_overrides, allow_cli_modes=True)
    return ResolvedStudyConfig.from_parts(study, simulation)
