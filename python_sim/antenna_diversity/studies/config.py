"""Resolution of reproducible spatial-study defaults, presets, JSON, and CLI."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np

from antenna_diversity.array.antenna_config import AntennaStateConfig, PhoneAntennaConfig
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath, ElementOrientation
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.config import (
    PHONE_CORNER_NAMES,
    CWInterferer,
    Direction,
    PhoneArrayConfig,
    RFStage,
    SimulationConfig,
    default_config,
)

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


def _mapping(raw: object, context: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{context} must be an object with string keys")
    return raw


def _sequence(raw: object, context: str) -> tuple[object, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{context} must be a list")
    return tuple(raw)


def _exact_record(
    raw: object, record_type: type[object], context: str
) -> dict[str, object]:
    data = _mapping(raw, context)
    _require_exact_keys(data, {field.name for field in fields(record_type)}, context)
    return copy.deepcopy(dict(data))


def _construct(record_type: type[object], data: Mapping[str, object], context: str):
    try:
        return record_type(**data)
    except TypeError as error:
        raise ValueError(f"invalid {context}: {error}") from error
    except ValueError as error:
        raise ValueError(f"invalid {context}: {error}") from error


def _is_finite_real(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        candidate = np.asarray(value)
        return (
            candidate.shape == ()
            and np.issubdtype(candidate.dtype, np.number)
            and not np.iscomplexobj(candidate)
            and bool(np.isfinite(float(candidate)))
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _require_real_fields(
    data: Mapping[str, object], names: set[str], context: str
) -> None:
    invalid = next((name for name in names if not _is_finite_real(data[name])), None)
    if invalid is not None:
        raise ValueError(f"{context}.{invalid} must be a finite real number")


def _direction(raw: object, context: str) -> Direction:
    data = _exact_record(raw, Direction, context)
    _require_real_fields(data, {"azimuth_deg", "elevation_deg"}, context)
    value = _construct(Direction, data, context)
    try:
        value.validate()
    except ValueError as error:
        raise ValueError(f"invalid {context}: {error}") from error
    return value


def _polarization(raw: object, context: str) -> WavePolarization:
    data = _exact_record(raw, WavePolarization, context)
    if not isinstance(data["kind"], str):
        raise ValueError(f"{context}.kind must be a string")
    _require_real_fields(
        data,
        {"tilt_deg", "az_real", "az_imag", "el_real", "el_imag"},
        context,
    )
    value = _construct(WavePolarization, data, context)
    try:
        value.jones_vector()
    except ValueError as error:
        raise ValueError(f"invalid {context}: {error}") from error
    return value


def _parametric_pattern(
    raw: object, context: str
) -> ParametricPatternParameters:
    data = _exact_record(raw, ParametricPatternParameters, context)
    data["nominal_polarization"] = _polarization(
        data["nominal_polarization"], f"{context}.nominal_polarization"
    )
    _require_real_fields(
        data,
        {
            "peak_gain_db",
            "cosine_exponent",
            "back_lobe_gain_db",
            "cross_polar_discrimination_db",
            "cross_phase_deg",
        },
        context,
    )
    return _construct(ParametricPatternParameters, data, context)


def _corner_mapping(raw: object, context: str) -> Mapping[str, object]:
    mapping = _mapping(raw, context)
    if set(mapping) != PHONE_CORNER_NAMES:
        raise ValueError(f"{context} must contain exactly TL/TR/BL/BR")
    return mapping


def _parametric_patterns(
    raw: object, context: str
) -> dict[str, ParametricPatternParameters]:
    if isinstance(raw, Mapping) and not raw:
        return {}
    mapping = _corner_mapping(raw, context)
    return {
        name: _parametric_pattern(value, f"{context}.{name}")
        for name, value in mapping.items()
    }


def _orientation_map(raw: object, context: str) -> dict[str, ElementOrientation]:
    mapping = _corner_mapping(raw, context)
    result: dict[str, ElementOrientation] = {}
    for name, value in mapping.items():
        entry_context = f"{context}.{name}"
        data = _exact_record(value, ElementOrientation, entry_context)
        _require_real_fields(data, {"yaw_deg", "pitch_deg", "roll_deg"}, entry_context)
        result[name] = _construct(ElementOrientation, data, entry_context)
    return result


def _attenuation_regions(raw: object, context: str) -> tuple[AttenuationRegion, ...]:
    result = []
    for index, item in enumerate(_sequence(raw, context)):
        entry_context = f"{context}[{index}]"
        data = _exact_record(item, AttenuationRegion, entry_context)
        elements = _sequence(data["elements"], f"{entry_context}.elements")
        if any(not isinstance(name, str) for name in elements):
            raise ValueError(f"{entry_context}.elements must contain strings")
        data["elements"] = tuple(elements)
        _require_real_fields(
            data,
            {
                "center_azimuth_deg",
                "center_elevation_deg",
                "radius_deg",
                "attenuation_db",
                "transition_width_deg",
            },
            entry_context,
        )
        region = _construct(AttenuationRegion, data, entry_context)
        try:
            region.validate()
        except ValueError as error:
            raise ValueError(f"invalid {entry_context}: {error}") from error
        result.append(region)
    return tuple(result)


def _coupling_paths(raw: object, context: str) -> tuple[CouplingPath, ...]:
    result = []
    for index, item in enumerate(_sequence(raw, context)):
        entry_context = f"{context}[{index}]"
        data = _exact_record(item, CouplingPath, entry_context)
        if not isinstance(data["source"], str) or not isinstance(data["receiver"], str):
            raise ValueError(f"{entry_context} endpoints must be strings")
        _require_real_fields(data, {"magnitude_db", "phase_deg"}, entry_context)
        path = _construct(CouplingPath, data, entry_context)
        try:
            path.validate()
        except ValueError as error:
            raise ValueError(f"invalid {entry_context}: {error}") from error
        result.append(path)
    return tuple(result)


def _ideal_antenna_model() -> PhoneAntennaConfig:
    return PhoneAntennaConfig(
        model_label="legacy isotropic",
        model_source="ideal",
        true_state=AntennaStateConfig(),
        assumed_state=AntennaStateConfig(),
    )


def _antenna_state(
    raw: object, base: AntennaStateConfig, context: str
) -> AntennaStateConfig:
    data = _mapping(raw, context)
    _reject_unknown(data, {field.name for field in fields(AntennaStateConfig)}, context)
    updates: dict[str, object] = {}
    for name, value in data.items():
        entry_context = f"{context}.{name}"
        if name == "pattern_kind":
            if not isinstance(value, str):
                raise ValueError(f"{entry_context} must be a string")
            updates[name] = value
        elif name == "parametric_patterns":
            updates[name] = _parametric_patterns(value, entry_context)
        elif name == "measured_pattern_csv":
            if value is not None and not isinstance(value, (str, Path)):
                raise ValueError(f"{entry_context} must be a path string or null")
            updates[name] = None if value is None else Path(value)
        elif name == "measured_pattern_sha256":
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{entry_context} must be a string or null")
            updates[name] = value
        elif name == "orientations":
            updates[name] = _orientation_map(value, entry_context)
        elif name == "attenuation_regions":
            updates[name] = _attenuation_regions(value, entry_context)
        elif name == "coupling_paths":
            updates[name] = _coupling_paths(value, entry_context)
    try:
        state = replace(copy.deepcopy(base), **updates)
    except TypeError as error:
        raise ValueError(f"invalid {context}: {error}") from error
    if state.pattern_kind != "measured":
        try:
            state.validate()
        except ValueError as error:
            raise ValueError(f"invalid {context}: {error}") from error
    return state


def _antenna_model(
    raw: object, base: PhoneAntennaConfig | None, context: str
) -> PhoneAntennaConfig:
    data = _mapping(raw, context)
    _reject_unknown(data, {field.name for field in fields(PhoneAntennaConfig)}, context)
    model = copy.deepcopy(base) if base is not None else _ideal_antenna_model()
    updates: dict[str, object] = {}
    for name, value in data.items():
        entry_context = f"{context}.{name}"
        if name in {"model_label", "model_source"}:
            if not isinstance(value, str):
                raise ValueError(f"{entry_context} must be a string")
            updates[name] = value
        elif name == "true_state":
            updates[name] = _antenna_state(value, model.true_state, entry_context)
        elif name == "assumed_state":
            updates[name] = _antenna_state(value, model.assumed_state, entry_context)
    try:
        model = replace(model, **updates)
    except TypeError as error:
        raise ValueError(f"invalid {context}: {error}") from error
    if model.true_state.pattern_kind != "measured" and model.assumed_state.pattern_kind != "measured":
        try:
            model.validate()
        except ValueError as error:
            raise ValueError(f"invalid {context}: {error}") from error
    return model


def _update_dataclass(value: object, updates: Mapping[str, object], context: str) -> object:
    if not is_dataclass(value):
        raise ValueError(f"{context} does not accept nested keys")
    allowed = {field.name for field in fields(value)}
    _reject_unknown(updates, allowed, context)
    values: dict[str, object] = {}
    for key, raw in updates.items():
        current = getattr(value, key)
        if key in {"true_direction", "assumed_direction", "direction"} and raw is not None:
            values[key] = _direction(raw, f"{context}.{key}")
        elif key in {"true_polarization", "assumed_polarization"}:
            values[key] = _polarization(raw, f"{context}.{key}")
        elif key == "antenna_model" and raw is not None:
            values[key] = _antenna_model(raw, current, f"{context}.{key}")
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
            stages = []
            for index, stage in enumerate(_sequence(raw, "simulation.rf.stages")):
                entry_context = f"simulation.rf.stages[{index}]"
                data = _exact_record(stage, RFStage, entry_context)
                if not isinstance(data["name"], str):
                    raise ValueError(f"{entry_context}.name must be a string")
                _require_real_fields(data, {"gain_db", "noise_figure_db"}, entry_context)
                stages.append(_construct(RFStage, data, entry_context))
            values[key] = tuple(stages)
        else:
            values[key] = copy.deepcopy(raw)
    try:
        return replace(value, **values)
    except TypeError as error:
        raise ValueError(f"invalid {context}: {error}") from error


def _interferer(raw: object) -> CWInterferer:
    raw = _mapping(raw, "interferer")
    allowed = {field.name for field in fields(CWInterferer)}
    _reject_unknown(raw, allowed, "interferer")
    required = allowed - {"polarization"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing interferer key: {sorted(missing)[0]}")
    data = copy.deepcopy(dict(raw))
    data["direction"] = _direction(data["direction"], "interferer.direction")
    if "polarization" in data:
        data["polarization"] = _polarization(data["polarization"], "interferer.polarization")
    else:
        data["polarization"] = WavePolarization()
    if not isinstance(data["enable"], bool):
        raise ValueError("interferer.enable must be a Boolean")
    _require_real_fields(
        data,
        {"power_dbm", "offset_hz", "initial_phase_rad", "start_ms"},
        "interferer",
    )
    if data["stop_ms"] is not None and not _is_finite_real(data["stop_ms"]):
        raise ValueError("interferer.stop_ms must be a finite real number or null")
    return _construct(CWInterferer, data, "interferer")


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
            levels = []
            for index, item in enumerate(_sequence(value, "study.error_levels")):
                entry_context = f"study.error_levels[{index}]"
                data = _exact_record(item, ErrorLevel, entry_context)
                if not isinstance(data["name"], str):
                    raise ValueError(f"{entry_context}.name must be a string")
                _require_real_fields(
                    data,
                    {
                        "direction_sigma_deg",
                        "position_sigma_m",
                        "amplitude_sigma_db",
                        "phase_sigma_deg",
                    },
                    entry_context,
                )
                levels.append(_construct(ErrorLevel, data, entry_context))
            converted[name] = tuple(levels)
        elif name == "multi_jammer_templates":
            templates = []
            for index, item in enumerate(_sequence(value, "study.multi_jammer_templates")):
                entry_context = f"study.multi_jammer_templates[{index}]"
                data = _exact_record(item, InterfererTemplate, entry_context)
                if not isinstance(data["name"], str):
                    raise ValueError(f"{entry_context}.name must be a string")
                interferers = tuple(
                    _interferer(entry)
                    for entry in _sequence(
                        data["interferers"], f"{entry_context}.interferers"
                    )
                )
                templates.append(
                    _construct(
                        InterfererTemplate,
                        {"name": data["name"], "interferers": interferers},
                        entry_context,
                    )
                )
            converted[name] = tuple(templates)
        else:
            converted[name] = copy.deepcopy(value)
    try:
        return replace(study, **converted)
    except TypeError as error:
        raise ValueError(f"invalid study: {error}") from error


def _resolve_measured_state(
    state: AntennaStateConfig, config_dir: Path, context: str
) -> AntennaStateConfig:
    if state.pattern_kind != "measured":
        return copy.deepcopy(state)
    if not isinstance(state.measured_pattern_csv, Path):
        raise ValueError(f"{context} measured pattern CSV path is missing or invalid")
    source = state.measured_pattern_csv
    if not source.is_absolute():
        source = config_dir / source
    source = source.resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {context} measured pattern CSV: {source}") from error
    try:
        raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} measured pattern CSV must be UTF-8") from error
    digest = hashlib.sha256(raw).hexdigest()
    expected = state.measured_pattern_sha256
    if expected is not None:
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)
        ):
            raise ValueError(
                f"{context} measured pattern SHA-256 must contain 64 hexadecimal digits"
            )
        if digest.lower() != expected.lower():
            raise ValueError(
                f"{context} measured pattern CSV SHA-256 does not match configuration"
            )
    return replace(
        state,
        measured_pattern_csv=Path(str(source)),
        measured_pattern_sha256=digest,
    )


def resolve_pattern_resources(
    simulation: SimulationConfig, config_dir: Path
) -> SimulationConfig:
    """Return a detached simulation with measured pattern paths and SHA-256 resolved."""

    if not isinstance(simulation, SimulationConfig):
        raise ValueError("simulation must be a SimulationConfig")
    try:
        directory = Path(config_dir).resolve()
    except (TypeError, ValueError, OSError) as error:
        raise ValueError("config_dir must be a valid path") from error
    resolved = copy.deepcopy(simulation)
    if not isinstance(resolved.array, PhoneArrayConfig):
        return resolved
    model = resolved.array.antenna_model
    if model is None:
        return resolved
    if not isinstance(model, PhoneAntennaConfig):
        raise ValueError("simulation.array.antenna_model must be a PhoneAntennaConfig")
    resolved.array.antenna_model = replace(
        model,
        true_state=_resolve_measured_state(
            model.true_state, directory, "simulation.array.antenna_model.true_state"
        ),
        assumed_state=_resolve_measured_state(
            model.assumed_state,
            directory,
            "simulation.array.antenna_model.assumed_state",
        ),
    )
    try:
        resolved.validate()
    except ValueError as error:
        raise ValueError(f"invalid resolved simulation: {error}") from error
    return resolved


def resolve_study_config(config_path: Path | None = None, preset: str | None = None, cli_overrides: Mapping[str, object] | None = None) -> ResolvedStudyConfig:
    """Resolve defaults, preset, JSON, and CLI into one validated snapshot."""

    selected_preset = preset or "standard"
    study = _study_for_preset(selected_preset)
    simulation = default_config()
    json_study: Mapping[str, object] = {}
    json_simulation: Mapping[str, object] = {}
    config_dir = Path.cwd()
    if config_path is not None:
        source = Path(config_path)
        config_dir = source.resolve().parent
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON configuration: {error.msg}") from error
        except OSError as error:
            raise ValueError(f"cannot read JSON configuration: {source}") from error
        if not isinstance(data, Mapping):
            raise ValueError("JSON configuration must be an object")
        _reject_unknown(data, {"study", "simulation"}, "JSON")
        json_study = data.get("study", {})
        json_simulation = data.get("simulation", {})
        if not isinstance(json_study, Mapping) or not isinstance(json_simulation, Mapping):
            raise ValueError("JSON study and simulation entries must be objects")
    study = _study_updates(study, json_study)
    simulation = _apply_simulation_updates(simulation, json_simulation)
    simulation = resolve_pattern_resources(simulation, config_dir)
    if cli_overrides:
        study = _study_updates(study, cli_overrides, allow_cli_modes=True)
    return ResolvedStudyConfig.from_parts(study, simulation)
