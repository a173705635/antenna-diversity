"""Direction-dependent attenuation and ordered mutual-coupling effects."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


_PHONE_CORNERS = frozenset({"TL", "TR", "BL", "BR"})


def _finite_real_scalar(value: object) -> bool:
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


def _require_finite_fields(fields: dict[str, object], label: str) -> None:
    if any(not _finite_real_scalar(value) for value in fields.values()):
        raise ValueError(f"{label} fields must be finite real scalars")


def _validate_active_antennas(active_antennas: object) -> tuple[str, ...]:
    if not isinstance(active_antennas, tuple):
        raise ValueError("active_antennas must be a unique two-to-four corner subset")
    if (
        not 2 <= len(active_antennas) <= 4
        or any(not isinstance(name, str) for name in active_antennas)
        or len(set(active_antennas)) != len(active_antennas)
        or any(name not in _PHONE_CORNERS for name in active_antennas)
    ):
        raise ValueError("active_antennas must be a unique two-to-four corner subset")
    return active_antennas


@dataclass(frozen=True)
class ElementOrientation:
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_fields(
            {
                "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg,
                "roll_deg": self.roll_deg,
            },
            "orientation",
        )


@dataclass(frozen=True)
class AttenuationRegion:
    elements: tuple[str, ...]
    center_azimuth_deg: float
    center_elevation_deg: float
    radius_deg: float
    attenuation_db: float
    transition_width_deg: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_fields(
            {
                "center_azimuth_deg": self.center_azimuth_deg,
                "center_elevation_deg": self.center_elevation_deg,
                "radius_deg": self.radius_deg,
                "attenuation_db": self.attenuation_db,
                "transition_width_deg": self.transition_width_deg,
            },
            "attenuation region",
        )

    def validate(self) -> None:
        if (
            not isinstance(self.elements, tuple)
            or any(not isinstance(name, str) for name in self.elements)
            or len(set(self.elements)) != len(self.elements)
            or any(name not in _PHONE_CORNERS for name in self.elements)
        ):
            raise ValueError("attenuation region elements must be a unique corner subset")
        if not 0.0 <= float(self.center_elevation_deg) <= 90.0:
            raise ValueError("attenuation region elevation must lie in [0, 90]")
        if float(self.radius_deg) < 0.0:
            raise ValueError("attenuation region radius must be nonnegative")
        if float(self.attenuation_db) < 0.0:
            raise ValueError("attenuation region attenuation must be nonnegative")
        if float(self.transition_width_deg) < 0.0:
            raise ValueError("attenuation region transition width must be nonnegative")


@dataclass(frozen=True)
class CouplingPath:
    source: str
    receiver: str
    magnitude_db: float
    phase_deg: float

    def __post_init__(self) -> None:
        _require_finite_fields(
            {"magnitude_db": self.magnitude_db, "phase_deg": self.phase_deg},
            "coupling path",
        )

    def validate(self) -> None:
        if (
            not isinstance(self.source, str)
            or not isinstance(self.receiver, str)
            or self.source not in _PHONE_CORNERS
            or self.receiver not in _PHONE_CORNERS
        ):
            raise ValueError("coupling path endpoints must be phone corners")
        if self.source == self.receiver:
            raise ValueError("coupling path must not be a self path")


def _direction_vector(azimuth_deg: float, elevation_deg: float) -> NDArray[np.float64]:
    azimuth, elevation = np.deg2rad([azimuth_deg % 360.0, elevation_deg])
    return np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )


def attenuation_voltage(
    regions: tuple[AttenuationRegion, ...],
    active_antennas: tuple[str, ...],
    direction: object,
) -> NDArray[np.float64]:
    """Return per-active-element voltage scale for the requested direction."""
    active_antennas = _validate_active_antennas(active_antennas)
    if not isinstance(regions, tuple) or any(
        not isinstance(region, AttenuationRegion) for region in regions
    ):
        raise ValueError("regions must be a tuple of AttenuationRegion records")
    try:
        azimuth_deg = direction.azimuth_deg
        elevation_deg = direction.elevation_deg
    except AttributeError as error:
        raise ValueError("direction must contain finite real azimuth and elevation") from error
    if (
        not _finite_real_scalar(azimuth_deg)
        or not _finite_real_scalar(elevation_deg)
        or not 0.0 <= float(elevation_deg) <= 90.0
    ):
        raise ValueError("direction must contain finite real azimuth and elevation")
    direction_vector = _direction_vector(float(azimuth_deg), float(elevation_deg))
    attenuation_db = np.zeros(len(active_antennas), dtype=np.float64)
    for region in regions:
        region.validate()
        center_vector = _direction_vector(
            float(region.center_azimuth_deg), float(region.center_elevation_deg)
        )
        dot_product = float(
            np.clip(np.dot(direction_vector, center_vector), -1.0, 1.0)
        )
        cross_norm = float(np.linalg.norm(np.cross(direction_vector, center_vector)))
        separation_deg = float(np.rad2deg(np.arctan2(cross_norm, dot_product)))
        if region.transition_width_deg == 0.0:
            fraction = 1.0 if separation_deg <= region.radius_deg else 0.0
        else:
            inner = region.radius_deg - region.transition_width_deg / 2.0
            outer = region.radius_deg + region.transition_width_deg / 2.0
            if separation_deg <= inner:
                fraction = 1.0
            elif separation_deg >= outer:
                fraction = 0.0
            else:
                phase = np.pi * (separation_deg - inner) / region.transition_width_deg
                fraction = float(0.5 * (1.0 + np.cos(phase)))
        selected = set(region.elements) if region.elements else set(active_antennas)
        for index, element in enumerate(active_antennas):
            if element in selected:
                attenuation_db[index] += fraction * region.attenuation_db
    return 10.0 ** (-attenuation_db / 20.0)


def coupling_matrix(
    paths: tuple[CouplingPath, ...], active_antennas: tuple[str, ...]
) -> NDArray[np.complex128]:
    """Build the ordered active-element coupling matrix."""
    active_antennas = _validate_active_antennas(active_antennas)
    if not isinstance(paths, tuple) or any(
        not isinstance(path, CouplingPath) for path in paths
    ):
        raise ValueError("paths must be a tuple of CouplingPath records")
    seen_paths: set[tuple[str, str]] = set()
    for path in paths:
        path.validate()
        key = (path.source, path.receiver)
        if key in seen_paths:
            raise ValueError("duplicate coupling path")
        seen_paths.add(key)
    matrix = np.eye(len(active_antennas), dtype=np.complex128)
    indices = {name: index for index, name in enumerate(active_antennas)}
    for path in paths:
        if path.source in indices and path.receiver in indices:
            matrix[indices[path.receiver], indices[path.source]] = 10.0 ** (
                path.magnitude_db / 20.0
            ) * np.exp(1j * np.deg2rad(path.phase_deg % 360.0))
    return matrix
