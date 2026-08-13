"""Immutable records and strict snapshots for spatial studies."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from antenna_diversity.config import CWInterferer, SimulationConfig


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
            object.__setattr__(self, name, np.array(getattr(self, name), copy=True))
        object.__setattr__(self, "manual_cases", copy.deepcopy(tuple(self.manual_cases)))


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
    if any(not 2 <= len(combo) <= 4 or len(set(combo)) != len(combo) or set(combo) - allowed_antennas for combo in study.combinations):
        raise ValueError("study.combinations must be unique two-to-four antenna subsets")
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

    def to_dict(self) -> dict[str, object]:
        return strict_snapshot({"study": self.study, "simulation": self.simulation, "config_sha256": self.config_sha256})  # type: ignore[return-value]

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
        study_copy = copy.deepcopy(study)
        simulation_copy = copy.deepcopy(simulation)
        validate_study_config(study_copy)
        simulation_copy.validate()
        unhashed = cls(study_copy, simulation_copy, "")
        encoded = json.dumps(unhashed.scientific_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return cls(study_copy, simulation_copy, hashlib.sha256(encoded).hexdigest())
