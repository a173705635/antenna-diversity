import numpy as np
import pytest

from antenna_diversity.array import (
    WavePolarization,
    rotation_matrix_ypr,
    wave_in_element_frame,
)
from antenna_diversity.config import mate70_default_config


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


def test_huge_finite_explicit_jones_is_unit_normalized():
    jones = WavePolarization(
        "jones", az_real=1e308, az_imag=-1e308, el_real=-1e308, el_imag=1e308
    ).jones_vector()

    assert np.all(np.isfinite(jones))
    np.testing.assert_allclose(np.vdot(jones, jones), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        jones,
        np.array([1.0 - 1.0j, -1.0 + 1.0j]) / 2.0,
        atol=1e-12,
    )


def test_config_validation_preserves_huge_finite_explicit_jones():
    cfg = mate70_default_config()
    cfg.signal.true_polarization = WavePolarization(
        "jones", az_real=1e308, el_real=1e308
    )

    cfg.validate()
    np.testing.assert_allclose(
        cfg.signal.true_polarization.jones_vector(),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        atol=1e-12,
    )


@pytest.mark.parametrize("value", [np.nan, np.inf, 1 + 1j, [1.0], "30.0"])
def test_polarization_rejects_nonfinite_or_nonscalar_fields(value):
    with pytest.raises(ValueError):
        WavePolarization("linear", tilt_deg=value).jones_vector()


@pytest.mark.parametrize(
    "call",
    [
        lambda: rotation_matrix_ypr("0.0", 0.0, 0.0),
        lambda: wave_in_element_frame(
            "15.0", 10.0, WavePolarization("rhcp"), np.eye(3)
        ),
    ],
)
def test_coordinate_interfaces_reject_string_angles(call):
    with pytest.raises(ValueError):
        call()


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


def test_external_wave_rejects_negative_elevation():
    with pytest.raises(ValueError):
        wave_in_element_frame(15.0, -10.0, WavePolarization("rhcp"), np.eye(3))


def test_positive_pitch_transforms_local_jones_basis():
    rotation = rotation_matrix_ypr(0.0, 90.0, 0.0)
    wave = wave_in_element_frame(
        90.0, 0.0, WavePolarization("linear", tilt_deg=45.0), rotation
    )

    assert wave.azimuth_deg == pytest.approx(90.0)
    assert wave.elevation_deg == pytest.approx(0.0)
    np.testing.assert_allclose(
        wave.jones, np.array([1.0, -1.0]) / np.sqrt(2.0), atol=1e-12
    )
