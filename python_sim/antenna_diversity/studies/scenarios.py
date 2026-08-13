"""Stable spatial-study scenario definitions and deterministic perturbations.

The combined-error draw order is azimuth, signed elevation (redrawn alone
until the perturbed true elevation is valid), per-corner XYZ position,
per-corner amplitude, then per-corner phase.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from antenna_diversity.array.geometry import CORNER_ORDER, active_corner_combinations

from .models import CombinedErrorRealization, StudyCase, strict_snapshot

if TYPE_CHECKING:
    from antenna_diversity.config import SimulationConfig
    from .models import ResolvedStudyConfig


_CORNER_INDEX = {name: index for index, name in enumerate(CORNER_ORDER)}


def _canonical_combination(value: Sequence[str]) -> tuple[str, ...]:
    combination = tuple(value)
    if (
        not 2 <= len(combination) <= len(CORNER_ORDER)
        or len(set(combination)) != len(combination)
        or any(name not in _CORNER_INDEX for name in combination)
    ):
        raise ValueError("combination must be a unique two-to-four corner subset")
    return tuple(sorted(combination, key=_CORNER_INDEX.__getitem__))


def parse_combination_spec(value: str | Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    """Parse combination input while preserving list order and canonicalizing rows."""

    if isinstance(value, str):
        if value.strip() == "all":
            return active_corner_combinations()
        raw_combinations = tuple(
            tuple(part.strip() for part in entry.split("+"))
            for entry in value.split(",")
            if entry.strip()
        )
    else:
        if not isinstance(value, Sequence):
            raise ValueError("combination specification must be a sequence")
        try:
            raw_combinations = tuple(tuple(entry) for entry in value)
        except TypeError as error:
            raise ValueError("combination specification entries must be sequences") from error
    if not raw_combinations:
        raise ValueError("combination specification must not be empty")
    combinations = tuple(_canonical_combination(combination) for combination in raw_combinations)
    if len(set(combinations)) != len(combinations):
        raise ValueError("combination specification contains duplicate combinations")
    return combinations


def derive_child_seed(master_seed: int, namespace: str, index: int) -> int:
    """Derive a reproducible unsigned 64-bit seed without Python hash randomization."""

    payload = f"{master_seed}:{namespace}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False)


def stable_case_id(
    stage: str,
    active_antennas: Sequence[str],
    mode: str,
    parameters: dict[str, object],
    seed: int | None,
) -> str:
    """Return a readable case identifier from a canonical strict-JSON payload."""

    payload = strict_snapshot({
        "stage": stage,
        "active_antennas": _canonical_combination(active_antennas),
        "mode": mode,
        "parameters": parameters,
        "seed": seed,
    })
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{stage.replace('_', '-')}-{digest}"


def sample_combined_errors(
    resolved: "ResolvedStudyConfig",
    level_name: str,
    active_antennas: Sequence[str],
    sample_count: int | None = None,
) -> tuple[CombinedErrorRealization, ...]:
    """Sample repeatable zero-mean channel residuals for one error level."""

    combination = _canonical_combination(active_antennas)
    levels = {level.name: level for level in resolved.study.error_levels}
    try:
        level = levels[level_name]
    except KeyError as error:
        raise ValueError(f"unknown combined error level: {level_name}") from error
    count = resolved.study.combined_error_samples_per_level if sample_count is None else sample_count
    if not isinstance(count, int) or count < 1:
        raise ValueError("sample_count must be a positive integer")
    true_direction = resolved.simulation.signal.true_direction
    if true_direction is None:
        raise ValueError("combined error sampling requires a true target direction")
    true_target_elevation_deg = true_direction.elevation_deg

    namespace = f"combined-error:{level_name}:{'+'.join(combination)}"
    realizations = []
    for sample_index in range(count):
        child_seed = derive_child_seed(resolved.study.seed, namespace, sample_index)
        rng = np.random.default_rng(child_seed)
        azimuth_error = float(rng.normal(0.0, level.direction_sigma_deg))
        elevation_error = float(rng.normal(0.0, level.direction_sigma_deg))
        while not 0.0 <= true_target_elevation_deg + elevation_error <= 90.0:
            elevation_error = float(rng.normal(0.0, level.direction_sigma_deg))
        position_errors = {
            corner: tuple(float(rng.normal(0.0, level.position_sigma_m)) for _ in range(3))
            for corner in combination
        }
        amplitude_raw = {
            corner: float(rng.normal(0.0, level.amplitude_sigma_db))
            for corner in combination
        }
        phase_raw = {
            corner: float(rng.normal(0.0, level.phase_sigma_deg))
            for corner in combination
        }
        amplitude_mean = sum(amplitude_raw.values()) / len(combination)
        phase_mean = sum(phase_raw.values()) / len(combination)
        realizations.append(CombinedErrorRealization(
            level=level_name,
            sample_index=sample_index,
            child_seed=child_seed,
            azimuth_error_deg=azimuth_error,
            elevation_error_deg=elevation_error,
            position_errors_m=position_errors,
            amplitude_errors_db={corner: value - amplitude_mean for corner, value in amplitude_raw.items()},
            phase_errors_deg={corner: value - phase_mean for corner, value in phase_raw.items()},
        ))
    return tuple(realizations)


def build_multi_jammer_cases(
    resolved: "ResolvedStudyConfig",
    base_configuration: "SimulationConfig",
    active_antennas: Sequence[str],
    mode: str,
) -> tuple[StudyCase, ...]:
    """Build detached, exact multi-jammer template cases for one array/mode pair."""

    del base_configuration
    combination = _canonical_combination(active_antennas)
    cases = []
    for index, template in enumerate(resolved.study.multi_jammer_templates):
        parameters: dict[str, object] = {
            "template": template.name,
            "interferers": strict_snapshot(template.interferers),
        }
        seed = derive_child_seed(
            resolved.study.seed,
            f"multi-jammer:{template.name}:{'+'.join(combination)}:{mode}",
            index,
        )
        cases.append(StudyCase(
            case_id=stable_case_id("multi_jammer", combination, mode, parameters, seed),
            stage="multi_jammer",
            active_antennas=combination,
            mode=mode,
            seed=seed,
            parameters=parameters,
        ))
    return tuple(cases)
