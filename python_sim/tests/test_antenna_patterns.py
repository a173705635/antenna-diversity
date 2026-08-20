"""Tests for polarized parametric and measured antenna element patterns."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from antenna_diversity.array.manifold import IsotropicElementPattern
from antenna_diversity.array.patterns import (
    LocalDirection,
    ParametricElementPattern,
    ParametricPatternParameters,
    ScalarPatternAdapter,
    TabulatedElementPattern,
)
from antenna_diversity.array.polarization import WavePolarization


_FREQUENCY_HZ = 1_575.42e6
_HEADER = (
    "element_name,frequency_hz,azimuth_deg,elevation_deg,"
    "h_az_real,h_az_imag,h_el_real,h_el_imag\n"
)


def write_rectangular_pattern_csv(
    directory: Path,
    *,
    azimuths: tuple[float, ...] = (0.0, 180.0),
    elevations: tuple[float, ...] = (-90.0, 90.0),
) -> Path:
    """Write a fixed hand-checked complex grid for interpolation tests."""
    if azimuths != (0.0, 180.0) or elevations != (-90.0, 90.0):
        raise ValueError("test helper supports the fixed two-by-two grid only")
    rows = (
        "TL,1575420000,0,-90,1,1,10,-1\n",
        "TL,1575420000,180,-90,5,3,20,1\n",
        "TL,1575420000,0,90,9,5,30,3\n",
        "TL,1575420000,180,90,13,7,40,5\n",
    )
    path = directory / "pattern.csv"
    path.write_text(_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_parametric_pattern_has_configured_peak_and_back_floor():
    parameters = {
        "TL": ParametricPatternParameters(
            peak_gain_db=3.0,
            cosine_exponent=2.0,
            back_lobe_gain_db=-20.0,
            nominal_polarization=WavePolarization("rhcp"),
            cross_polar_discrimination_db=30.0,
            cross_phase_deg=0.0,
        )
    }
    pattern = ParametricElementPattern(parameters)

    front = pattern.effective_height("TL", LocalDirection(0.0, 90.0), _FREQUENCY_HZ)
    back = pattern.effective_height("TL", LocalDirection(0.0, -90.0), _FREQUENCY_HZ)

    assert np.linalg.norm(front) == pytest.approx(10 ** (3.0 / 20.0))
    assert np.linalg.norm(back) == pytest.approx(10 ** (-20.0 / 20.0))


def test_parametric_pattern_sets_hand_checked_co_and_cross_polarized_heights():
    pattern = ParametricElementPattern(
        {
            "TL": ParametricPatternParameters(
                nominal_polarization=WavePolarization("linear", tilt_deg=0.0),
                cross_polar_discrimination_db=20.0,
                cross_phase_deg=90.0,
            )
        }
    )

    height = pattern.effective_height("TL", LocalDirection(0.0, 90.0), _FREQUENCY_HZ)

    # A 20 dB XPD is a 0.1 voltage cross component.  After unit-norm
    # normalization, the height's cross component is +0.1j/sqrt(1.01).
    # Receive projection uses conjugation, so the cross-polarized response is
    # -0.1j/sqrt(1.01).
    np.testing.assert_allclose(
        height,
        np.array([1.0, 0.1j], dtype=np.complex128) / np.sqrt(1.01),
        atol=1e-12,
    )
    assert np.vdot(height, np.array([1.0, 0.0])) == pytest.approx(1.0 / np.sqrt(1.01))
    assert np.vdot(height, np.array([0.0, 1.0])) == pytest.approx(-0.1j / np.sqrt(1.01))


def test_local_direction_normalizes_azimuth_and_accepts_negative_elevation():
    direction = LocalDirection(-90.0, -45.0)
    assert direction.azimuth_deg == pytest.approx(270.0)
    assert direction.elevation_deg == pytest.approx(-45.0)


@pytest.mark.parametrize("elevation_deg", [-90.1, 90.1])
def test_local_direction_rejects_elevation_outside_local_sphere(elevation_deg):
    with pytest.raises(ValueError, match="elevation"):
        LocalDirection(0.0, elevation_deg)


def test_scalar_pattern_adapter_returns_a_two_component_height():
    height = ScalarPatternAdapter(IsotropicElementPattern()).effective_height(
        "TL", LocalDirection(0.0, 0.0), _FREQUENCY_HZ
    )
    np.testing.assert_allclose(height, np.array([1.0 + 0.0j, 0.0 + 0.0j]))


def test_scalar_pattern_adapter_translates_legacy_lookup_errors_to_value_error():
    class MissingElementPattern:
        def complex_gain(self, element_name, direction, frequency_hz):
            del element_name, direction, frequency_hz
            raise KeyError("TL")

    adapter = ScalarPatternAdapter(MissingElementPattern())
    with pytest.raises(ValueError, match="could not evaluate"):
        adapter.effective_height("TL", LocalDirection(0.0, 0.0), _FREQUENCY_HZ)


def test_tabulated_pattern_interpolates_complex_components_and_wraps_azimuth(tmp_path):
    path = write_rectangular_pattern_csv(tmp_path)
    pattern = TabulatedElementPattern.from_csv(path, _FREQUENCY_HZ)

    at_90 = pattern.effective_height("TL", LocalDirection(90.0, 0.0), _FREQUENCY_HZ)
    at_270 = pattern.effective_height("TL", LocalDirection(270.0, 0.0), _FREQUENCY_HZ)

    # Bilinear average of the four static cells: h_az=(1+5+9+13)/4 +
    # j(1+3+5+7)/4 = 7+4j; h_el=(10+20+30+40)/4 + j(-1+1+3+5)/4 = 25+2j.
    expected = np.array([7.0 + 4.0j, 25.0 + 2.0j], dtype=np.complex128)
    np.testing.assert_allclose(at_90, expected, atol=1e-12)
    np.testing.assert_allclose(at_270, expected, atol=1e-12)


def test_tabulated_pattern_interpolates_across_zero_for_shifted_azimuth_grid(tmp_path):
    path = tmp_path / "shifted-grid.csv"
    path.write_text(
        _HEADER
        + "TL,1575420000,90,-45,2,1,10,-1\n"
        + "TL,1575420000,270,-45,6,5,30,3\n"
        + "TL,1575420000,90,45,2,1,10,-1\n"
        + "TL,1575420000,270,45,6,5,30,3\n",
        encoding="utf-8",
    )
    pattern = TabulatedElementPattern.from_csv(path, _FREQUENCY_HZ)

    at_zero = pattern.effective_height("TL", LocalDirection(0.0, 0.0), _FREQUENCY_HZ)

    # Around the periodic seam, 270 degrees is the -90-degree neighbor of
    # zero and 90 degrees is the +90-degree neighbor, so zero is their mean.
    np.testing.assert_allclose(
        at_zero, np.array([4.0 + 3.0j, 20.0 + 1.0j]), atol=1e-12
    )


def test_tabulated_pattern_verifies_sha256(tmp_path):
    path = write_rectangular_pattern_csv(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    TabulatedElementPattern.from_csv(path, _FREQUENCY_HZ, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        TabulatedElementPattern.from_csv(path, _FREQUENCY_HZ, "0" * 64)


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        (
            _HEADER
            + "TL,1575420000,0,-90,1,0,0,0\n"
            + "TL,1575420000,360,-90,2,0,0,0\n",
            "duplicate",
        ),
        (
            _HEADER
            + "TL,1575420000,0,-90,1,0,0,0\n"
            + "TL,1575420000,180,-90,1,0,0,0\n"
            + "TL,1575420000,0,90,1,0,0,0\n",
            "complete.*grid",
        ),
        (
            _HEADER.replace("h_el_imag", "unknown")
            + "TL,1575420000,0,-90,1,0,0,0\n",
            "columns",
        ),
        (
            _HEADER + "TL,1575420000,0,-90,nan,0,0,0\n",
            "finite",
        ),
    ],
)
def test_tabulated_pattern_rejects_malformed_tables(tmp_path, contents, match):
    path = tmp_path / "malformed.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        TabulatedElementPattern.from_csv(path, _FREQUENCY_HZ)


def test_tabulated_pattern_rejects_missing_column_empty_file_and_frequency_mismatch(tmp_path):
    missing = tmp_path / "missing-column.csv"
    missing.write_text(
        "element_name,frequency_hz,azimuth_deg,elevation_deg,h_az_real,h_az_imag,h_el_real\n"
        "TL,1575420000,0,-90,1,0,0\n",
        encoding="utf-8",
    )
    empty = tmp_path / "empty.csv"
    empty.write_text(_HEADER, encoding="utf-8")
    mismatch = write_rectangular_pattern_csv(tmp_path)

    with pytest.raises(ValueError, match="columns"):
        TabulatedElementPattern.from_csv(missing, _FREQUENCY_HZ)
    with pytest.raises(ValueError, match="data rows"):
        TabulatedElementPattern.from_csv(empty, _FREQUENCY_HZ)
    with pytest.raises(ValueError, match="frequency"):
        TabulatedElementPattern.from_csv(mismatch, _FREQUENCY_HZ + 2.0)


def test_tabulated_pattern_rejects_missing_element_and_out_of_range_elevation(tmp_path):
    pattern = TabulatedElementPattern.from_csv(write_rectangular_pattern_csv(tmp_path), _FREQUENCY_HZ)
    limited = tmp_path / "limited.csv"
    limited.write_text(
        _HEADER
        + "TL,1575420000,0,-45,1,0,0,0\n"
        + "TL,1575420000,180,-45,1,0,0,0\n"
        + "TL,1575420000,0,45,1,0,0,0\n"
        + "TL,1575420000,180,45,1,0,0,0\n",
        encoding="utf-8",
    )
    limited_pattern = TabulatedElementPattern.from_csv(limited, _FREQUENCY_HZ)

    with pytest.raises(ValueError, match="element"):
        pattern.effective_height("TR", LocalDirection(0.0, 0.0), _FREQUENCY_HZ)
    with pytest.raises(ValueError, match="elevation"):
        limited_pattern.effective_height("TL", LocalDirection(0.0, 60.0), _FREQUENCY_HZ)
