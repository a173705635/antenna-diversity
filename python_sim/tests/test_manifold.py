import numpy as np
import pytest

from antenna_diversity.array.manifold import (
    IsotropicElementPattern,
    array_manifold,
    direction_unit_vector,
)
from antenna_diversity.array.geometry import select_phase_centers
from antenna_diversity.config import Direction, mate70_default_config


def test_direction_unit_vectors_follow_phone_axes():
    np.testing.assert_allclose(direction_unit_vector(Direction(0.0, 0.0)), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(direction_unit_vector(Direction(90.0, 0.0)), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(direction_unit_vector(Direction(0.0, 90.0)), [0, 0, 1], atol=1e-12)


def test_isotropic_manifold_has_unit_magnitude():
    cfg = mate70_default_config()
    positions = select_phase_centers(
        cfg.phone.phase_centers_m,
        cfg.array.active_antennas,
    )
    response = array_manifold(
        positions,
        Direction(30.0, 60.0),
        cfg.signal.carrier_hz,
        cfg.array.active_antennas,
        IsotropicElementPattern(),
    )

    np.testing.assert_allclose(np.abs(response), np.ones(4), atol=1e-12)


def test_coplanar_array_has_identical_response_at_zenith():
    cfg = mate70_default_config()
    positions = select_phase_centers(
        cfg.phone.phase_centers_m,
        cfg.array.active_antennas,
    )
    response = array_manifold(
        positions,
        Direction(217.0, 90.0),
        cfg.signal.carrier_hz,
    )

    np.testing.assert_allclose(response, np.ones(4), atol=1e-12)


def test_selected_subarray_equals_full_array_slice():
    cfg = mate70_default_config()
    full_positions = select_phase_centers(cfg.phone.phase_centers_m, cfg.array.active_antennas)
    full = array_manifold(full_positions, Direction(120.0, 10.0), cfg.signal.carrier_hz)
    cfg.array.active_antennas = ("TR", "BR")
    selected_positions = select_phase_centers(cfg.phone.phase_centers_m, cfg.array.active_antennas)
    selected = array_manifold(selected_positions, Direction(120.0, 10.0), cfg.signal.carrier_hz)

    np.testing.assert_allclose(selected, full[[1, 3]], atol=1e-12)


@pytest.mark.parametrize(
    ("positions_m", "frequency_hz", "match"),
    [
        (np.zeros(3), 1.0, "shape"),
        (np.zeros((2, 2)), 1.0, "shape"),
        (np.array([[0.0, 0.0, np.nan]]), 1.0, "finite"),
        (np.zeros((1, 3)), 0.0, "positive finite"),
        (np.zeros((1, 3)), np.inf, "positive finite"),
    ],
)
def test_array_manifold_rejects_invalid_positions_and_frequency(
    positions_m, frequency_hz, match
):
    with pytest.raises(ValueError, match=match):
        array_manifold(positions_m, Direction(0.0, 0.0), frequency_hz)


def test_array_manifold_rejects_mismatched_element_names():
    with pytest.raises(ValueError, match="element_names"):
        array_manifold(
            np.zeros((2, 3)),
            Direction(0.0, 0.0),
            1.0,
            element_names=("TL",),
        )
