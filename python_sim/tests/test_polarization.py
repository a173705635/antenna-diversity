import numpy as np
import pytest

from antenna_diversity.array import (
    WavePolarization,
    rotation_matrix_ypr,
    wave_in_element_frame,
)


def test_rhcp_and_lhcp_are_unit_and_orthogonal():
    rhcp = WavePolarization("rhcp").jones_vector()
    lhcp = WavePolarization("lhcp").jones_vector()
    np.testing.assert_allclose(np.vdot(rhcp, rhcp), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.vdot(lhcp, lhcp), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.vdot(rhcp, lhcp), 0.0, atol=1e-12)


def test_linear_tilt_and_explicit_jones_are_normalized():
    linear = WavePolarization("linear", tilt_deg=90.0).jones_vector()
    explicit = WavePolarization(
        "jones", az_real=2.0, az_imag=0.0, el_real=0.0, el_imag=0.0
    ).jones_vector()
    np.testing.assert_allclose(abs(linear[1]), 1.0, atol=1e-12)
    np.testing.assert_allclose(explicit, [1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1 + 1j, [1.0]])
def test_polarization_rejects_nonfinite_or_nonscalar_fields(value):
    with pytest.raises(ValueError):
        WavePolarization("linear", tilt_deg=value).jones_vector()


def test_identity_frame_preserves_direction_and_jones():
    wave = wave_in_element_frame(120.0, 30.0, WavePolarization("rhcp"), np.eye(3))
    assert wave.azimuth_deg == pytest.approx(120.0)
    assert wave.elevation_deg == pytest.approx(30.0)
    np.testing.assert_allclose(
        wave.jones, WavePolarization("rhcp").jones_vector(), atol=1e-12
    )


def test_positive_yaw_moves_source_to_negative_local_azimuth():
    rotation = rotation_matrix_ypr(90.0, 0.0, 0.0)
    wave = wave_in_element_frame(120.0, 20.0, WavePolarization("linear"), rotation)
    assert wave.azimuth_deg == pytest.approx(30.0)
    assert wave.elevation_deg == pytest.approx(20.0)
