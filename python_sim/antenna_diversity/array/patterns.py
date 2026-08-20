"""Polarized parametric and measured antenna-element pattern models."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from .polarization import WavePolarization


_REQUIRED_COLUMNS = {
    "element_name",
    "frequency_hz",
    "azimuth_deg",
    "elevation_deg",
    "h_az_real",
    "h_az_imag",
    "h_el_real",
    "h_el_imag",
}
_NUMERIC_COLUMNS = (
    "frequency_hz",
    "azimuth_deg",
    "elevation_deg",
    "h_az_real",
    "h_az_imag",
    "h_el_real",
    "h_el_imag",
)


def _finite_real_scalar(value: object) -> bool:
    """Return whether *value* is one finite, non-Boolean real scalar."""
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        array = np.asarray(value)
        if (
            array.shape != ()
            or np.iscomplexobj(array)
            or not np.issubdtype(array.dtype, np.number)
        ):
            return False
        return bool(np.isfinite(float(array)))
    except (TypeError, ValueError, OverflowError):
        return False


def _validated_frequency(value: object, *, label: str = "frequency_hz") -> float:
    if not _finite_real_scalar(value) or float(value) <= 0.0:
        raise ValueError(f"{label} must be a positive finite real scalar")
    return float(value)


def _frequency_tolerance(frequency_hz: float) -> float:
    return max(1.0, 1e-9 * frequency_hz)


@dataclass(frozen=True)
class LocalDirection:
    """Direction in an antenna element's local spherical coordinate frame."""

    azimuth_deg: float
    elevation_deg: float

    def __post_init__(self) -> None:
        if not _finite_real_scalar(self.azimuth_deg):
            raise ValueError("local direction azimuth must be a finite real scalar")
        if not _finite_real_scalar(self.elevation_deg):
            raise ValueError("local direction elevation must be a finite real scalar")
        elevation = float(self.elevation_deg)
        if not -90.0 <= elevation <= 90.0:
            raise ValueError("local direction elevation must be within [-90, 90]")
        object.__setattr__(self, "azimuth_deg", float(self.azimuth_deg) % 360.0)
        object.__setattr__(self, "elevation_deg", elevation)


class PolarizedElementPattern(Protocol):
    """A receive pattern expressed as a local two-component effective height."""

    def effective_height(
        self,
        element_name: str,
        direction: LocalDirection,
        frequency_hz: float,
    ) -> NDArray[np.complex128]: ...


@dataclass(frozen=True)
class ParametricPatternParameters:
    """Per-element parameters for a smooth polarized receive pattern."""

    peak_gain_db: float = 0.0
    cosine_exponent: float = 1.0
    back_lobe_gain_db: float = -20.0
    nominal_polarization: WavePolarization = WavePolarization("rhcp")
    cross_polar_discrimination_db: float = 30.0
    cross_phase_deg: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "peak_gain_db": self.peak_gain_db,
            "cosine_exponent": self.cosine_exponent,
            "back_lobe_gain_db": self.back_lobe_gain_db,
            "cross_polar_discrimination_db": self.cross_polar_discrimination_db,
            "cross_phase_deg": self.cross_phase_deg,
        }
        if any(not _finite_real_scalar(value) for value in values.values()):
            raise ValueError("parametric pattern fields must be finite real scalars")
        if float(self.cosine_exponent) <= 0.0:
            raise ValueError("cosine_exponent must be positive")
        if float(self.cross_polar_discrimination_db) < 0.0:
            raise ValueError("cross_polar_discrimination_db must be nonnegative")
        if not isinstance(self.nominal_polarization, WavePolarization):
            raise ValueError("nominal_polarization must be a WavePolarization")
        self.nominal_polarization.jones_vector()


@dataclass(frozen=True)
class ParametricElementPattern:
    """Evaluate a cosine-rolloff, finite-backlobe polarized pattern."""

    parameters: Mapping[str, ParametricPatternParameters]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a mapping of element names")
        validated: dict[str, ParametricPatternParameters] = {}
        for element_name, parameters in self.parameters.items():
            if not isinstance(element_name, str) or not element_name:
                raise ValueError("element names must be nonempty strings")
            if not isinstance(parameters, ParametricPatternParameters):
                raise ValueError("each element parameter must be ParametricPatternParameters")
            validated[element_name] = parameters
        object.__setattr__(self, "parameters", MappingProxyType(validated))

    def effective_height(
        self,
        element_name: str,
        direction: LocalDirection,
        frequency_hz: float,
    ) -> NDArray[np.complex128]:
        if not isinstance(element_name, str) or element_name not in self.parameters:
            raise ValueError(f"unknown parametric pattern element: {element_name!r}")
        if not isinstance(direction, LocalDirection):
            raise ValueError("direction must be a LocalDirection")
        _validated_frequency(frequency_hz)
        parameters = self.parameters[element_name]

        front_power = max(np.sin(np.deg2rad(direction.elevation_deg)), 0.0) ** float(
            parameters.cosine_exponent
        )
        relative_gain_db = (
            -np.inf if front_power == 0.0 else 10.0 * np.log10(front_power)
        )
        gain_db = float(parameters.peak_gain_db) + max(
            relative_gain_db,
            float(parameters.back_lobe_gain_db) - float(parameters.peak_gain_db),
        )
        voltage_gain = 10.0 ** (gain_db / 20.0)

        nominal = parameters.nominal_polarization.jones_vector()
        orthogonal = np.array(
            [-np.conj(nominal[1]), np.conj(nominal[0])], dtype=np.complex128
        )
        cross_voltage = 10.0 ** (-float(parameters.cross_polar_discrimination_db) / 20.0)
        cross_phase = np.exp(1j * np.deg2rad(float(parameters.cross_phase_deg)))
        mixture = nominal + cross_voltage * cross_phase * orthogonal
        result = (voltage_gain * mixture / np.linalg.norm(mixture)).astype(np.complex128)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class ScalarPatternAdapter:
    """Adapt a legacy scalar ``complex_gain`` pattern to two components."""

    scalar_pattern: object

    def __post_init__(self) -> None:
        if not callable(getattr(self.scalar_pattern, "complex_gain", None)):
            raise ValueError("scalar_pattern must provide complex_gain")

    def effective_height(
        self,
        element_name: str,
        direction: LocalDirection,
        frequency_hz: float,
    ) -> NDArray[np.complex128]:
        if not isinstance(direction, LocalDirection):
            raise ValueError("direction must be a LocalDirection")
        _validated_frequency(frequency_hz)
        try:
            gain = complex(
                self.scalar_pattern.complex_gain(element_name, direction, float(frequency_hz))
            )
        except Exception as error:
            raise ValueError("scalar pattern could not evaluate the local direction") from error
        if not np.isfinite(gain):
            raise ValueError("scalar pattern gain must be finite")
        result = np.array([gain, 0.0j], dtype=np.complex128)
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class _ElementGrid:
    azimuths_deg: NDArray[np.float64]
    elevations_deg: NDArray[np.float64]
    heights: NDArray[np.complex128]


@dataclass(frozen=True)
class TabulatedElementPattern:
    """A strict regular-grid measured pattern with complex bilinear interpolation."""

    grids: Mapping[str, _ElementGrid]
    frequency_hz: float
    sha256: str

    @classmethod
    def from_csv(
        cls,
        path: Path,
        frequency_hz: float,
        expected_sha256: str | None = None,
    ) -> "TabulatedElementPattern":
        requested_frequency = _validated_frequency(frequency_hz)
        if expected_sha256 is not None and not isinstance(expected_sha256, str):
            raise ValueError("measured pattern CSV SHA-256 must be a hexadecimal string")
        try:
            source = Path(path)
        except (TypeError, ValueError) as error:
            raise ValueError("measured pattern CSV path is invalid") from error
        try:
            raw = source.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read measured pattern CSV: {source}") from error
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ValueError("measured pattern CSV SHA-256 does not match configuration")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("measured pattern CSV must be UTF-8") from error
        reader = csv.DictReader(io.StringIO(text))
        if (
            reader.fieldnames is None
            or len(reader.fieldnames) != len(_REQUIRED_COLUMNS)
            or set(reader.fieldnames) != _REQUIRED_COLUMNS
        ):
            raise ValueError("measured pattern CSV columns do not match the required schema")
        rows = list(reader)
        if not rows:
            raise ValueError("measured pattern CSV must contain data rows")
        grids = _parse_pattern_rows(rows, requested_frequency)
        return cls(MappingProxyType(grids), requested_frequency, digest)

    def effective_height(
        self,
        element_name: str,
        direction: LocalDirection,
        frequency_hz: float,
    ) -> NDArray[np.complex128]:
        if not isinstance(element_name, str) or element_name not in self.grids:
            raise ValueError(f"unknown measured pattern element: {element_name!r}")
        if not isinstance(direction, LocalDirection):
            raise ValueError("direction must be a LocalDirection")
        requested_frequency = _validated_frequency(frequency_hz)
        if abs(requested_frequency - self.frequency_hz) > _frequency_tolerance(
            requested_frequency
        ):
            raise ValueError("measured pattern frequency does not match the requested frequency")
        grid = self.grids[element_name]
        elevation = direction.elevation_deg
        if elevation < grid.elevations_deg[0] or elevation > grid.elevations_deg[-1]:
            raise ValueError("local direction elevation is outside measured pattern coverage")
        elevation_lower, elevation_upper, elevation_fraction = _bounded_neighbors(
            grid.elevations_deg, elevation
        )
        azimuth_lower, azimuth_upper, azimuth_fraction = _periodic_neighbors(
            grid.azimuths_deg, direction.azimuth_deg
        )
        low = (
            (1.0 - azimuth_fraction) * grid.heights[elevation_lower, azimuth_lower]
            + azimuth_fraction * grid.heights[elevation_lower, azimuth_upper]
        )
        high = (
            (1.0 - azimuth_fraction) * grid.heights[elevation_upper, azimuth_lower]
            + azimuth_fraction * grid.heights[elevation_upper, azimuth_upper]
        )
        result = ((1.0 - elevation_fraction) * low + elevation_fraction * high).astype(
            np.complex128
        )
        result.setflags(write=False)
        return result


def _parse_pattern_rows(
    rows: list[dict[str | None, str | None]], requested_frequency_hz: float
) -> dict[str, _ElementGrid]:
    """Validate parsed rows and turn them into complete per-element grids."""
    grouped: dict[str, list[tuple[float, float, float, complex, complex]]] = {}
    for row_number, row in enumerate(rows, start=2):
        if set(row) != _REQUIRED_COLUMNS:
            raise ValueError("measured pattern CSV row does not match the required schema")
        element_name = row["element_name"]
        if not isinstance(element_name, str) or not element_name:
            raise ValueError(f"measured pattern CSV row {row_number} has an invalid element name")
        values: dict[str, float] = {}
        for column in _NUMERIC_COLUMNS:
            raw_value = row[column]
            if isinstance(raw_value, bool):
                raise ValueError(f"measured pattern CSV {column} must be finite")
            try:
                value = float(raw_value)  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    f"measured pattern CSV {column} must be a finite real number"
                ) from error
            if not np.isfinite(value):
                raise ValueError(f"measured pattern CSV {column} must be finite")
            values[column] = value
        elevation = values["elevation_deg"]
        if not -90.0 <= elevation <= 90.0:
            raise ValueError("measured pattern CSV elevation must be within [-90, 90]")
        grouped.setdefault(element_name, []).append(
            (
                values["frequency_hz"],
                values["azimuth_deg"] % 360.0,
                elevation,
                complex(values["h_az_real"], values["h_az_imag"]),
                complex(values["h_el_real"], values["h_el_imag"]),
            )
        )

    result: dict[str, _ElementGrid] = {}
    for element_name, samples in grouped.items():
        frequencies = {sample[0] for sample in samples}
        if len(frequencies) != 1:
            raise ValueError("measured pattern CSV must contain one frequency per element")
        file_frequency = frequencies.pop()
        if abs(file_frequency - requested_frequency_hz) > _frequency_tolerance(
            requested_frequency_hz
        ):
            raise ValueError("measured pattern CSV frequency does not match the requested frequency")

        coordinates: dict[tuple[float, float], tuple[complex, complex]] = {}
        for _, azimuth, elevation, h_az, h_el in samples:
            coordinate = (azimuth, elevation)
            if coordinate in coordinates:
                raise ValueError("measured pattern CSV contains duplicate grid coordinates")
            coordinates[coordinate] = (h_az, h_el)
        azimuths = np.array(sorted({coordinate[0] for coordinate in coordinates}), dtype=np.float64)
        elevations = np.array(sorted({coordinate[1] for coordinate in coordinates}), dtype=np.float64)
        if len(coordinates) != azimuths.size * elevations.size:
            raise ValueError("measured pattern CSV must contain a complete rectangular grid")
        heights = np.empty((elevations.size, azimuths.size, 2), dtype=np.complex128)
        for elevation_index, elevation in enumerate(elevations):
            for azimuth_index, azimuth in enumerate(azimuths):
                try:
                    h_az, h_el = coordinates[(float(azimuth), float(elevation))]
                except KeyError as error:
                    raise ValueError(
                        "measured pattern CSV must contain a complete rectangular grid"
                    ) from error
                heights[elevation_index, azimuth_index] = (h_az, h_el)
        azimuths.setflags(write=False)
        elevations.setflags(write=False)
        heights.setflags(write=False)
        result[element_name] = _ElementGrid(azimuths, elevations, heights)
    return result


def _bounded_neighbors(
    axis: NDArray[np.float64], value: float
) -> tuple[int, int, float]:
    if axis.size == 1 or value <= axis[0]:
        return 0, 0, 0.0
    if value >= axis[-1]:
        last = axis.size - 1
        return last, last, 0.0
    upper = int(np.searchsorted(axis, value, side="right"))
    lower = upper - 1
    return lower, upper, float((value - axis[lower]) / (axis[upper] - axis[lower]))


def _periodic_neighbors(
    axis: NDArray[np.float64], value: float
) -> tuple[int, int, float]:
    if axis.size == 1:
        return 0, 0, 0.0
    upper = int(np.searchsorted(axis, value, side="right"))
    if upper == axis.size:
        lower = axis.size - 1
        upper = 0
        upper_value = axis[0] + 360.0
    else:
        lower = upper - 1
        upper_value = axis[upper]
    lower_value = axis[lower]
    return lower, upper, float((value - lower_value) / (upper_value - lower_value))
