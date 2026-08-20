"""Polarization and coordinate transforms for phone antenna elements."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_real_scalar(value: object) -> bool:
    """Return whether ``value`` is a finite, non-Boolean real scalar."""
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


@dataclass(frozen=True)
class WavePolarization:
    """A normalized Jones vector expressed in the local spherical basis."""

    kind: str = "rhcp"
    tilt_deg: float = 0.0
    az_real: float = 1.0
    az_imag: float = 0.0
    el_real: float = 0.0
    el_imag: float = 0.0

    def jones_vector(self) -> NDArray[np.complex128]:
        """Return the unit-power Jones vector in ``[E_az, E_el]`` order."""
        values = (
            self.tilt_deg,
            self.az_real,
            self.az_imag,
            self.el_real,
            self.el_imag,
        )
        if any(not _finite_real_scalar(value) for value in values):
            raise ValueError("polarization fields must be finite real scalars")
        if self.kind == "rhcp":
            vector = np.array([1.0, -1.0j], dtype=np.complex128)
        elif self.kind == "lhcp":
            vector = np.array([1.0, 1.0j], dtype=np.complex128)
        elif self.kind == "linear":
            tilt = np.deg2rad(float(self.tilt_deg))
            vector = np.array([np.cos(tilt), np.sin(tilt)], dtype=np.complex128)
        elif self.kind == "jones":
            vector = np.array(
                [
                    complex(float(self.az_real), float(self.az_imag)),
                    complex(float(self.el_real), float(self.el_imag)),
                ],
                dtype=np.complex128,
            )
        else:
            raise ValueError("polarization kind must be rhcp, lhcp, linear, or jones")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("polarization Jones vector must be nonzero")
        result = vector / norm
        result.setflags(write=False)
        return result


@dataclass(frozen=True)
class LocalWave:
    """Incident direction and Jones vector represented in an element frame."""

    azimuth_deg: float
    elevation_deg: float
    jones: NDArray[np.complex128]

    def __post_init__(self) -> None:
        if not _finite_real_scalar(self.azimuth_deg) or not _finite_real_scalar(
            self.elevation_deg
        ):
            raise ValueError("local wave angles must be finite real scalars")
        elevation = float(self.elevation_deg)
        if not -90.0 <= elevation <= 90.0:
            raise ValueError("local wave elevation must be within [-90, 90]")
        try:
            jones = np.asarray(self.jones, dtype=np.complex128)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("local wave Jones vector must have two finite components") from error
        if jones.shape != (2,) or not np.all(np.isfinite(jones)):
            raise ValueError("local wave Jones vector must have two finite components")
        if not np.isclose(np.linalg.norm(jones), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("local wave Jones vector must have unit norm")
        result = jones.copy()
        result.setflags(write=False)
        object.__setattr__(self, "azimuth_deg", float(self.azimuth_deg) % 360.0)
        object.__setattr__(self, "elevation_deg", elevation)
        object.__setattr__(self, "jones", result)


def rotation_matrix_ypr(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> NDArray[np.float64]:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` from local to phone axes."""
    angles = (yaw_deg, pitch_deg, roll_deg)
    if any(not _finite_real_scalar(angle) for angle in angles):
        raise ValueError("yaw, pitch, and roll must be finite real scalars")
    yaw, pitch, roll = np.deg2rad(np.asarray(angles, dtype=np.float64))
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


def _spherical_frame(
    azimuth_deg: float, elevation_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return radial, increasing-azimuth, and increasing-elevation unit vectors."""
    if not _finite_real_scalar(azimuth_deg) or not _finite_real_scalar(elevation_deg):
        raise ValueError("wave angles must be finite real scalars")
    elevation = float(elevation_deg)
    if not -90.0 <= elevation <= 90.0:
        raise ValueError("wave elevation must be within [-90, 90]")
    azimuth, elevation_rad = np.deg2rad(
        np.asarray([float(azimuth_deg), elevation], dtype=np.float64)
    )
    cos_azimuth, sin_azimuth = np.cos(azimuth), np.sin(azimuth)
    cos_elevation, sin_elevation = np.cos(elevation_rad), np.sin(elevation_rad)
    unit = np.array(
        [
            cos_elevation * cos_azimuth,
            cos_elevation * sin_azimuth,
            sin_elevation,
        ]
    )
    azimuth_basis = np.array([-sin_azimuth, cos_azimuth, 0.0])
    elevation_basis = np.array(
        [
            -sin_elevation * cos_azimuth,
            -sin_elevation * sin_azimuth,
            cos_elevation,
        ]
    )
    return unit, azimuth_basis, elevation_basis


def _validated_rotation(local_to_phone: ArrayLike) -> NDArray[np.float64]:
    """Validate and return a proper local-to-phone rotation matrix."""
    try:
        rotation = np.asarray(local_to_phone)
    except (TypeError, ValueError) as error:
        raise ValueError("local_to_phone must be a finite 3x3 proper rotation") from error
    if rotation.shape != (3, 3) or np.iscomplexobj(rotation):
        raise ValueError("local_to_phone must be a finite 3x3 proper rotation")
    try:
        rotation = rotation.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("local_to_phone must be a finite 3x3 proper rotation") from error
    if not np.all(np.isfinite(rotation)):
        raise ValueError("local_to_phone must be a finite 3x3 proper rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
        raise ValueError("local_to_phone must be orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("local_to_phone must have determinant +1")
    return rotation


def wave_in_element_frame(
    azimuth_deg: float,
    elevation_deg: float,
    polarization: WavePolarization,
    local_to_phone: ArrayLike,
) -> LocalWave:
    """Transform an incident phone-frame wave into an element's local frame."""
    if not isinstance(polarization, WavePolarization):
        raise ValueError("polarization must be a WavePolarization")
    if not _finite_real_scalar(elevation_deg):
        raise ValueError("external wave elevation must be within [0, 90]")
    external_elevation = float(elevation_deg)
    if not 0.0 <= external_elevation <= 90.0:
        raise ValueError("external wave elevation must be within [0, 90]")
    rotation = _validated_rotation(local_to_phone)
    unit_phone, az_basis_phone, el_basis_phone = _spherical_frame(
        azimuth_deg, external_elevation
    )
    jones_phone = polarization.jones_vector()
    field_phone = jones_phone[0] * az_basis_phone + jones_phone[1] * el_basis_phone
    unit_local = rotation.T @ unit_phone
    field_local = rotation.T @ field_phone
    local_azimuth = float(np.rad2deg(np.arctan2(unit_local[1], unit_local[0])) % 360.0)
    local_z = float(unit_local[2])
    if local_z > 1.0:
        if local_z > 1.0 + 1e-12:
            raise ValueError("rotation produced an invalid local direction")
        local_z = 1.0
    elif local_z < -1.0:
        if local_z < -1.0 - 1e-12:
            raise ValueError("rotation produced an invalid local direction")
        local_z = -1.0
    local_elevation = float(np.rad2deg(np.arcsin(local_z)))
    _, az_basis_local, el_basis_local = _spherical_frame(
        local_azimuth, local_elevation
    )
    local_jones = np.array(
        [np.vdot(az_basis_local, field_local), np.vdot(el_basis_local, field_local)],
        dtype=np.complex128,
    )
    norm = float(np.linalg.norm(local_jones))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("rotation produced an invalid local polarization")
    local_jones /= norm
    return LocalWave(local_azimuth, local_elevation, local_jones)
