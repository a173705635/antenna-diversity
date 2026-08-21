import numpy as np
import pytest

from antenna_diversity.array.effects import (
    AttenuationRegion,
    CouplingPath,
    ElementOrientation,
    attenuation_voltage,
    coupling_matrix,
)
from antenna_diversity.config import Direction


def test_coupling_matrix_preserves_requested_corner_order():
    paths = (CouplingPath("TL", "BR", -20.0, 90.0),)

    matrix = coupling_matrix(paths, ("BR", "TL"))

    np.testing.assert_allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == pytest.approx(0.1j)
    assert matrix[1, 0] == 0.0


def test_overlapping_attenuation_regions_add_in_db():
    regions = (
        AttenuationRegion(("TL",), 120.0, 30.0, 20.0, 6.0, 0.0),
        AttenuationRegion(("TL",), 120.0, 30.0, 10.0, 4.0, 0.0),
    )

    voltage = attenuation_voltage(
        regions, ("TL", "BR"), Direction(120.0, 30.0)
    )

    np.testing.assert_allclose(voltage, [10 ** (-10.0 / 20.0), 1.0])


def test_empty_region_element_list_attenuates_every_active_corner():
    regions = (AttenuationRegion((), 30.0, 60.0, 5.0, 8.0),)

    voltage = attenuation_voltage(
        regions, ("BR", "TL", "TR"), Direction(30.0, 60.0)
    )

    np.testing.assert_allclose(voltage, np.full(3, 10 ** (-8.0 / 20.0)))


def test_transition_uses_raised_cosine_centered_on_region_radius():
    regions = (AttenuationRegion(("TL",), 0.0, 0.0, 20.0, 6.0, 10.0),)

    voltage = attenuation_voltage(
        regions, ("TL", "BR"), Direction(20.0, 0.0)
    )

    np.testing.assert_allclose(voltage, [10 ** (-3.0 / 20.0), 1.0])


def test_zero_radius_region_at_exact_center_applies_full_attenuation():
    regions = (AttenuationRegion(("TL",), 0.1, 33.3, 0.0, 6.0),)

    voltage = attenuation_voltage(
        regions, ("TL", "BR"), Direction(0.1, 33.3)
    )

    np.testing.assert_allclose(voltage, [10 ** (-6.0 / 20.0), 1.0])


@pytest.mark.parametrize(
    ("azimuth_deg", "expected_attenuation_db"),
    [
        (10.0, 8.0),
        (15.0, 8.0),
        (20.0, 4.0),
        (25.0, 0.0),
        (30.0, 0.0),
    ],
)
def test_raised_cosine_transition_has_stable_inside_edge_middle_and_outside_values(
    azimuth_deg, expected_attenuation_db
):
    regions = (AttenuationRegion(("TL",), 0.0, 0.0, 20.0, 8.0, 10.0),)

    voltage = attenuation_voltage(
        regions, ("TL", "BR"), Direction(azimuth_deg, 0.0)
    )

    np.testing.assert_allclose(
        voltage, [10 ** (-expected_attenuation_db / 20.0), 1.0], atol=1e-15
    )


def test_coupling_path_with_inactive_endpoint_does_not_change_selected_matrix():
    paths = (
        CouplingPath("TL", "BR", -10.0, 0.0),
        CouplingPath("BR", "TR", -6.0, 45.0),
    )

    matrix = coupling_matrix(paths, ("TR", "TL"))

    np.testing.assert_array_equal(matrix, np.eye(2, dtype=np.complex128))


@pytest.mark.parametrize(
    "active_antennas",
    [
        ("TL", "BR"),
        ("TR", "BR", "TL"),
        ("BL", "TL", "BR", "TR"),
    ],
)
def test_coupling_indices_follow_each_two_to_four_corner_active_order(active_antennas):
    paths = (CouplingPath("TL", "BR", -20.0, 180.0),)

    matrix = coupling_matrix(paths, active_antennas)

    source_index = active_antennas.index("TL")
    receiver_index = active_antennas.index("BR")
    expected = np.eye(len(active_antennas), dtype=np.complex128)
    expected[receiver_index, source_index] = -0.1
    np.testing.assert_allclose(matrix, expected, atol=1e-15)


@pytest.mark.parametrize(
    "value",
    [True, 1.0 + 2.0j, np.array([1.0]), np.nan, np.inf],
)
def test_effect_records_reject_non_real_scalar_numeric_fields(value):
    with pytest.raises(ValueError):
        ElementOrientation(yaw_deg=value)
    with pytest.raises(ValueError):
        AttenuationRegion(("TL",), value, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        CouplingPath("TL", "BR", value, 0.0)


@pytest.mark.parametrize(
    "region",
    [
        AttenuationRegion(("XX",), 0.0, 0.0, 1.0, 1.0),
        AttenuationRegion(("TL", "TL"), 0.0, 0.0, 1.0, 1.0),
        AttenuationRegion(("TL",), 0.0, 0.0, -1.0, 1.0),
        AttenuationRegion(("TL",), 0.0, 0.0, 1.0, -1.0),
        AttenuationRegion(("TL",), 0.0, 0.0, 1.0, 1.0, -1.0),
    ],
)
def test_attenuation_rejects_bad_corner_sets_or_negative_fields(region):
    with pytest.raises(ValueError):
        attenuation_voltage((region,), ("TL", "BR"), Direction(0.0, 0.0))


@pytest.mark.parametrize(
    "active_antennas",
    [("TL",), ("TL", "TL"), ("TL", "XX"), ("TL", "TR", "BL", "BR", "XX")],
)
def test_effect_builders_require_unique_two_to_four_corner_active_subset(active_antennas):
    with pytest.raises(ValueError, match="active_antennas"):
        attenuation_voltage((), active_antennas, Direction(0.0, 0.0))
    with pytest.raises(ValueError, match="active_antennas"):
        coupling_matrix((), active_antennas)


def test_effect_builders_reject_invalid_direction_and_coupling_paths():
    with pytest.raises(ValueError, match="direction"):
        attenuation_voltage((), ("TL", "BR"), Direction(0.0, np.nan))
    with pytest.raises(ValueError, match="self"):
        coupling_matrix(
            (CouplingPath("TL", "TL", -20.0, 0.0),), ("TL", "BR")
        )
    with pytest.raises(ValueError, match="duplicate"):
        coupling_matrix(
            (
                CouplingPath("TL", "BR", -20.0, 0.0),
                CouplingPath("TL", "BR", -10.0, 45.0),
            ),
            ("TL", "BR"),
        )


def test_effect_builders_translate_malformed_corner_containers_to_value_error():
    malformed_corner = np.array(["TL"])

    with pytest.raises(ValueError, match="active_antennas"):
        coupling_matrix((), (malformed_corner, "BR"))
    with pytest.raises(ValueError, match="elements"):
        attenuation_voltage(
            (AttenuationRegion((malformed_corner,), 0.0, 0.0, 1.0, 1.0),),
            ("TL", "BR"),
            Direction(0.0, 0.0),
        )
    with pytest.raises(ValueError, match="endpoints"):
        coupling_matrix(
            (CouplingPath(malformed_corner, "BR", -20.0, 0.0),),
            ("TL", "BR"),
        )
