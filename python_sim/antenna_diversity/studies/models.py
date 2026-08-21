"""Immutable records and strict snapshots for spatial studies."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from antenna_diversity.array.antenna_config import AntennaStateConfig, PhoneAntennaConfig
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath, ElementOrientation
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.config import CWInterferer, PhoneArrayConfig, SimulationConfig


class _FrozenDict(dict[str, object]):
    """A recursively frozen JSON-like mapping that remains deepcopy-safe."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        raise TypeError("resolved study mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, object]) -> "_FrozenDict":
        return self


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        frozen = _FrozenDict()
        dict.update(frozen, {key: _freeze(item) for key, item in value.items()})
        return frozen
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _immutable_array(value: object) -> np.ndarray:
    """Detach an axis onto a bytes buffer whose write access cannot be restored."""

    array = np.array(value, copy=True)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _canonical_study(study: StudyConfig) -> StudyConfig:
    """Copy and normalize every public StudyConfig container at the boundary."""

    values = {
        field.name: copy.deepcopy(getattr(study, field.name))
        for field in fields(StudyConfig)
    }
    values["output_dir"] = Path(str(values["output_dir"]))
    values["combinations"] = tuple(tuple(combo) for combo in values["combinations"])
    for name in ("spatial_modes", "full_chain_modes", "single_antennas", "automatic_roles"):
        values[name] = tuple(values[name])
    values["error_levels"] = tuple(
        ErrorLevel(
            name=str(level.name),
            direction_sigma_deg=copy.deepcopy(level.direction_sigma_deg),
            position_sigma_m=copy.deepcopy(level.position_sigma_m),
            amplitude_sigma_db=copy.deepcopy(level.amplitude_sigma_db),
            phase_sigma_deg=copy.deepcopy(level.phase_sigma_deg),
        )
        for level in values["error_levels"]
    )
    values["multi_jammer_templates"] = tuple(
        InterfererTemplate(
            name=str(template.name),
            interferers=tuple(copy.deepcopy(interferer) for interferer in template.interferers),
        )
        for template in values["multi_jammer_templates"]
    )
    values["manual_cases"] = tuple(values["manual_cases"])
    return StudyConfig(**values)


def _canonical_polarization(value: object) -> object:
    if not isinstance(value, WavePolarization):
        return copy.deepcopy(value)
    return WavePolarization(
        kind=copy.deepcopy(value.kind),
        tilt_deg=copy.deepcopy(value.tilt_deg),
        az_real=copy.deepcopy(value.az_real),
        az_imag=copy.deepcopy(value.az_imag),
        el_real=copy.deepcopy(value.el_real),
        el_imag=copy.deepcopy(value.el_imag),
    )


def _canonical_pattern(value: object) -> object:
    if not isinstance(value, ParametricPatternParameters):
        return copy.deepcopy(value)
    return ParametricPatternParameters(
        peak_gain_db=copy.deepcopy(value.peak_gain_db),
        cosine_exponent=copy.deepcopy(value.cosine_exponent),
        back_lobe_gain_db=copy.deepcopy(value.back_lobe_gain_db),
        nominal_polarization=_canonical_polarization(value.nominal_polarization),
        cross_polar_discrimination_db=copy.deepcopy(
            value.cross_polar_discrimination_db
        ),
        cross_phase_deg=copy.deepcopy(value.cross_phase_deg),
    )


def _canonical_orientation(value: object) -> object:
    if not isinstance(value, ElementOrientation):
        return copy.deepcopy(value)
    return ElementOrientation(
        copy.deepcopy(value.yaw_deg),
        copy.deepcopy(value.pitch_deg),
        copy.deepcopy(value.roll_deg),
    )


def _canonical_region(value: object) -> object:
    if not isinstance(value, AttenuationRegion):
        return copy.deepcopy(value)
    return AttenuationRegion(
        tuple(copy.deepcopy(value.elements)),
        copy.deepcopy(value.center_azimuth_deg),
        copy.deepcopy(value.center_elevation_deg),
        copy.deepcopy(value.radius_deg),
        copy.deepcopy(value.attenuation_db),
        copy.deepcopy(value.transition_width_deg),
    )


def _canonical_coupling(value: object) -> object:
    if not isinstance(value, CouplingPath):
        return copy.deepcopy(value)
    return CouplingPath(
        copy.deepcopy(value.source),
        copy.deepcopy(value.receiver),
        copy.deepcopy(value.magnitude_db),
        copy.deepcopy(value.phase_deg),
    )


def _canonical_antenna_state(value: object) -> object:
    if not isinstance(value, AntennaStateConfig):
        return copy.deepcopy(value)
    csv_path = value.measured_pattern_csv
    return AntennaStateConfig(
        pattern_kind=copy.deepcopy(value.pattern_kind),
        parametric_patterns={
            copy.deepcopy(name): _canonical_pattern(parameters)
            for name, parameters in value.parametric_patterns.items()
        },
        measured_pattern_csv=(
            None if csv_path is None else Path(str(csv_path))
        ),
        measured_pattern_sha256=copy.deepcopy(value.measured_pattern_sha256),
        orientations={
            copy.deepcopy(name): _canonical_orientation(orientation)
            for name, orientation in value.orientations.items()
        },
        attenuation_regions=tuple(
            _canonical_region(region) for region in value.attenuation_regions
        ),
        coupling_paths=tuple(
            _canonical_coupling(path) for path in value.coupling_paths
        ),
    )


def _canonical_antenna_model(value: object) -> object:
    if not isinstance(value, PhoneAntennaConfig):
        return copy.deepcopy(value)
    return PhoneAntennaConfig(
        model_label=copy.deepcopy(value.model_label),
        model_source=copy.deepcopy(value.model_source),
        true_state=_canonical_antenna_state(value.true_state),
        assumed_state=_canonical_antenna_state(value.assumed_state),
    )


def _canonical_simulation(simulation: SimulationConfig) -> SimulationConfig:
    """Detach and normalize every antenna-related scientific input."""

    result = copy.deepcopy(simulation)
    result.signal.true_polarization = _canonical_polarization(
        simulation.signal.true_polarization
    )
    result.signal.assumed_polarization = _canonical_polarization(
        simulation.signal.assumed_polarization
    )
    result.interferers = tuple(
        replace(
            copy.deepcopy(interferer),
            polarization=_canonical_polarization(interferer.polarization),
        )
        if isinstance(interferer, CWInterferer)
        else copy.deepcopy(interferer)
        for interferer in simulation.interferers
    )
    if isinstance(result.array, PhoneArrayConfig):
        result.array.antenna_model = _canonical_antenna_model(
            simulation.array.antenna_model
        )
    result.plot.output_dir = Path(str(simulation.plot.output_dir))
    return result


def strict_snapshot(value: object) -> object:
    """Convert supported values to a detached, strict-JSON-compatible snapshot."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: strict_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("snapshot dictionaries must have string keys")
        return {key: strict_snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_snapshot(item) for item in value]
    if isinstance(value, np.ndarray):
        return [strict_snapshot(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return strict_snapshot(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("snapshot floats must be finite")
        return value
    raise ValueError(f"unsupported snapshot value: {type(value).__name__}")


@dataclass(frozen=True)
class ErrorLevel:
    name: str
    direction_sigma_deg: float
    position_sigma_m: float
    amplitude_sigma_db: float
    phase_sigma_deg: float


@dataclass(frozen=True)
class InterfererTemplate:
    name: str
    interferers: tuple[CWInterferer, ...]


@dataclass(frozen=True)
class StudyCase:
    case_id: str
    stage: str
    active_antennas: tuple[str, ...]
    mode: str
    seed: int | None
    parameters: dict[str, object]
    selection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankingResult:
    summaries: tuple[dict[str, object], ...]
    ranked_combinations: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class SelectionResult:
    cases: tuple[StudyCase, ...]
    unselected_roles: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class FullChainRecords:
    metrics: tuple[dict[str, object], ...]
    cn0: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class StageRecords:
    """Normalized strict-JSON rows emitted by one study stage."""

    name: str
    tables: dict[str, tuple[dict[str, object], ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tables": {
                key: [dict(row) for row in rows]
                for key, rows in self.tables.items()
            },
        }


@dataclass(frozen=True)
class CombinedErrorRealization:
    level: str
    sample_index: int
    child_seed: int
    azimuth_error_deg: float
    elevation_error_deg: float
    position_errors_m: dict[str, tuple[float, float, float]]
    amplitude_errors_db: dict[str, float]
    phase_errors_deg: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a detached strict-JSON-compatible realization snapshot."""

        return strict_snapshot(self)  # type: ignore[return-value]


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    name: str
    preset: str
    seed: int
    output_dir: Path
    combinations: tuple[tuple[str, ...], ...]
    spatial_modes: tuple[str, ...]
    full_chain_modes: tuple[str, ...]
    single_antennas: tuple[str, ...]
    ranking_mode: str
    exclusion_angle_deg: float
    azimuths_deg: np.ndarray
    elevations_deg: np.ndarray
    powers_dbm: np.ndarray
    direction_errors_deg: np.ndarray
    position_errors_m: np.ndarray
    amplitude_errors_db: np.ndarray
    phase_errors_deg: np.ndarray
    loading_factors: np.ndarray
    error_levels: tuple[ErrorLevel, ...]
    combined_error_samples_per_level: int
    full_chain_run_count: int
    multi_jammer_templates: tuple[InterfererTemplate, ...]
    automatic_roles: tuple[str, ...]
    manual_cases: tuple[dict[str, object], ...]

    def __post_init__(self) -> None:
        for name in (
            "azimuths_deg", "elevations_deg", "powers_dbm", "direction_errors_deg",
            "position_errors_m", "amplitude_errors_db", "phase_errors_deg", "loading_factors",
        ):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        object.__setattr__(self, "manual_cases", tuple(_freeze(case) for case in self.manual_cases))


def validate_study_config(study: StudyConfig) -> None:
    """Validate the resolved study values required by every later stage."""

    if study.schema_version != 1:
        raise ValueError("study.schema_version must be 1")
    if not isinstance(study.name, str) or not study.name or study.name in {".", ".."} or any(
        character in study.name for character in '<>:"/\\|?*'
    ):
        raise ValueError("study.name must be a nonempty directory-safe name")
    if not isinstance(study.seed, (int, np.integer)):
        raise ValueError("study.seed must be an integer")
    if not isinstance(study.output_dir, Path):
        raise ValueError("study.output_dir must be a Path")
    if not study.combinations or len(set(study.combinations)) != len(study.combinations):
        raise ValueError("study.combinations must be nonempty and unique")
    allowed_antennas = {"TL", "TR", "BL", "BR"}
    if any(
        not isinstance(combo, tuple)
        or not 2 <= len(combo) <= 4
        or len(set(combo)) != len(combo)
        or set(combo) - allowed_antennas
        for combo in study.combinations
    ):
        raise ValueError("study.combinations must be unique two-to-four antenna subsets")
    if (
        not study.single_antennas
        or len(set(study.single_antennas)) != len(study.single_antennas)
        or set(study.single_antennas) - allowed_antennas
    ):
        raise ValueError("study.single_antennas must be a nonempty unique corner subset")
    allowed_modes = {"single", "egc", "mvdr"}
    if not study.spatial_modes or set(study.spatial_modes) - allowed_modes:
        raise ValueError("study.spatial_modes contains an unknown mode")
    if not study.full_chain_modes or set(study.full_chain_modes) - allowed_modes:
        raise ValueError("study.full_chain_modes contains an unknown mode")
    if study.ranking_mode not in study.spatial_modes:
        raise ValueError("study.ranking_mode must be included in study.spatial_modes")
    if not 0.0 <= study.exclusion_angle_deg <= 180.0:
        raise ValueError("study.exclusion_angle_deg must lie in [0, 180]")
    for name in (
        "azimuths_deg", "elevations_deg", "powers_dbm", "direction_errors_deg",
        "position_errors_m", "amplitude_errors_db", "phase_errors_deg", "loading_factors",
    ):
        values = np.asarray(getattr(study, name))
        if values.ndim != 1 or not values.size or np.iscomplexobj(values) or not np.all(np.isfinite(values)):
            raise ValueError(f"study.{name} must be a nonempty finite real axis")
    if np.any(np.diff(study.loading_factors) <= 0) or np.any(study.loading_factors <= 0):
        raise ValueError("study.loading_factors must be positive and strictly increasing")
    if not study.error_levels or len({level.name for level in study.error_levels}) != len(study.error_levels):
        raise ValueError("study.error_levels must be nonempty and uniquely named")
    for level in study.error_levels:
        if not level.name or any(not np.isfinite(value) or value < 0 for value in (
            level.direction_sigma_deg, level.position_sigma_m, level.amplitude_sigma_db, level.phase_sigma_deg,
        )):
            raise ValueError("study.error_levels must contain finite nonnegative values")
    if study.combined_error_samples_per_level < 1 or study.full_chain_run_count < 1:
        raise ValueError("study sample counts must be positive")


@dataclass(frozen=True)
class ResolvedStudyConfig:
    study: StudyConfig
    simulation: SimulationConfig
    config_sha256: str

    def __getattribute__(self, name: str) -> object:
        """Return a working copy of mutable simulation configuration to callers."""

        if name == "simulation":
            return copy.deepcopy(object.__getattribute__(self, name))
        return object.__getattribute__(self, name)

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot({
            "study": self.study,
            "simulation": object.__getattribute__(self, "simulation"),
            "config_sha256": self.config_sha256,
        })  # type: ignore[return-value]

    def scientific_dict(self) -> dict[str, object]:
        snapshot = self.to_dict()
        study = dict(snapshot["study"])  # type: ignore[arg-type]
        study.pop("name")
        study.pop("preset")
        study.pop("output_dir")
        snapshot["study"] = study
        simulation = dict(snapshot["simulation"])  # type: ignore[arg-type]
        plot = dict(simulation["plot"])  # type: ignore[arg-type]
        plot.pop("output_dir")
        plot.pop("enable")
        plot.pop("show")
        simulation["plot"] = plot
        snapshot["simulation"] = simulation
        snapshot.pop("config_sha256")
        return snapshot

    @classmethod
    def from_parts(cls, study: StudyConfig, simulation: SimulationConfig) -> "ResolvedStudyConfig":
        study_copy = _canonical_study(study)
        simulation_copy = _canonical_simulation(simulation)
        validate_study_config(study_copy)
        simulation_copy.validate()
        unhashed = cls(study_copy, simulation_copy, "")
        encoded = json.dumps(unhashed.scientific_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return cls(study_copy, simulation_copy, hashlib.sha256(encoded).hexdigest())
