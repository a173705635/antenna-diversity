from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from antenna_diversity.array.antenna_config import (
    AntennaStateConfig,
    PhoneAntennaConfig,
    resolved_phone_antenna_config,
)
from antenna_diversity.array.effects import (
    AttenuationRegion,
    CouplingPath,
    ElementOrientation,
)
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.config import (
    CWInterferer,
    Direction,
    default_config,
    legacy_ula_config,
    mate70_default_config,
)


CORNERS = ("TL", "TR", "BL", "BR")


def orientations_for_test() -> dict[str, ElementOrientation]:
    return {corner: ElementOrientation() for corner in CORNERS}


def parameters_for_test() -> dict[str, ParametricPatternParameters]:
    return {corner: ParametricPatternParameters() for corner in CORNERS}


def phone_antenna_config_for_test(
    *,
    source: str = "ideal",
    true_state: AntennaStateConfig | None = None,
    assumed_state: AntennaStateConfig | None = None,
) -> PhoneAntennaConfig:
    if true_state is None:
        true_state = AntennaStateConfig()
    if assumed_state is None:
        assumed_state = AntennaStateConfig()
    return PhoneAntennaConfig("test antenna", source, true_state, assumed_state)


def test_default_phone_resolves_to_ideal_antenna_without_changing_old_fields():
    cfg = default_config()

    cfg.validate()

    assert cfg.array.pattern_model == "isotropic"
    assert cfg.array.antenna_model is None


def test_legacy_isotropic_selector_resolves_two_complete_independent_ideal_states():
    cfg = default_config()

    resolved = resolved_phone_antenna_config(cfg.array)

    assert resolved.model_source == "ideal"
    assert resolved.true_state.pattern_kind == "isotropic"
    assert resolved.assumed_state.pattern_kind == "isotropic"
    assert set(resolved.true_state.orientations) == set(CORNERS)
    assert set(resolved.assumed_state.orientations) == set(CORNERS)
    assert resolved.true_state is not resolved.assumed_state
    assert resolved.true_state.orientations is not resolved.assumed_state.orientations


def test_structured_state_requires_exact_four_corner_maps():
    cfg = default_config()
    cfg.array.antenna_model = phone_antenna_config_for_test()
    del cfg.array.antenna_model.true_state.orientations["BR"]

    with pytest.raises(ValueError, match="TL/TR/BL/BR"):
        cfg.validate()


def test_structured_model_is_returned_by_resolver_after_validation():
    cfg = default_config()
    model = phone_antenna_config_for_test()
    cfg.array.antenna_model = model

    cfg.validate()

    assert resolved_phone_antenna_config(cfg.array) is model


def test_default_signal_and_interferer_polarizations_are_rhcp():
    cfg = default_config()

    cfg.validate()

    np.testing.assert_allclose(
        cfg.signal.true_polarization.jones_vector(), [2**-0.5, -1j * 2**-0.5]
    )
    np.testing.assert_allclose(
        cfg.signal.assumed_polarization.jones_vector(), [2**-0.5, -1j * 2**-0.5]
    )
    np.testing.assert_allclose(
        cfg.interferers[0].polarization.jones_vector(), [2**-0.5, -1j * 2**-0.5]
    )


def test_interferer_polarization_extension_preserves_legacy_study_snapshot_schema():
    from antenna_diversity.studies.models import strict_snapshot

    polarization = WavePolarization("linear", tilt_deg=45.0)
    interferer = CWInterferer(
        Direction(120.0, 10.0), polarization=polarization
    )

    assert interferer.polarization is polarization
    assert strict_snapshot(interferer) == {
        "direction": {"azimuth_deg": 120.0, "elevation_deg": 10.0},
        "enable": True,
        "power_dbm": -90.0,
        "offset_hz": 250e3,
        "initial_phase_rad": 0.8,
        "start_ms": 0.0,
        "stop_ms": None,
    }


@pytest.mark.parametrize(
    "polarization",
    [
        WavePolarization("invalid"),
        WavePolarization("jones", az_real=0.0, el_real=0.0),
    ],
)
def test_root_validation_rejects_bad_signal_polarization(polarization):
    cfg = default_config()
    cfg.signal.true_polarization = polarization

    with pytest.raises(ValueError, match="polarization"):
        cfg.validate()


def test_root_validation_rejects_bad_interferer_polarization():
    cfg = default_config()
    cfg.interferers = (
        CWInterferer(
            direction=Direction(120.0, 10.0),
            polarization=WavePolarization("invalid"),
        ),
    )

    with pytest.raises(ValueError, match="polarization"):
        cfg.validate()


@pytest.mark.parametrize("pattern_model", ["parametric", "measured", "unknown"])
def test_structured_selector_conflicts_with_nonlegacy_pattern_selector(pattern_model):
    cfg = default_config()
    cfg.array.pattern_model = pattern_model
    cfg.array.antenna_model = phone_antenna_config_for_test()

    with pytest.raises(ValueError, match="pattern_model.*antenna_model"):
        cfg.validate()


def test_legacy_pattern_selector_must_be_isotropic():
    cfg = default_config()
    cfg.array.pattern_model = "parametric"

    with pytest.raises(ValueError, match="pattern_model"):
        cfg.validate()


@pytest.mark.parametrize("model_source", ["", "isotropic", "unknown", True])
def test_model_source_must_name_supported_provenance(model_source):
    cfg = default_config()
    cfg.array.antenna_model = phone_antenna_config_for_test(source=model_source)

    with pytest.raises(ValueError, match="model_source"):
        cfg.validate()


@pytest.mark.parametrize("pattern_kind", ["", "ideal", "unknown", True])
def test_state_pattern_kind_must_name_supported_pattern(pattern_kind):
    cfg = default_config()
    cfg.array.antenna_model = phone_antenna_config_for_test(
        true_state=AntennaStateConfig(pattern_kind=pattern_kind)
    )

    with pytest.raises(ValueError, match="pattern_kind"):
        cfg.validate()


def test_parametric_state_requires_exact_four_parameter_records():
    cfg = default_config()
    state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=parameters_for_test(),
    )
    del state.parametric_patterns["BR"]
    cfg.array.antenna_model = phone_antenna_config_for_test(
        source="parametric", true_state=state
    )

    with pytest.raises(ValueError, match="parametric.*TL/TR/BL/BR"):
        cfg.validate()


def test_parametric_state_accepts_four_parameter_records():
    cfg = default_config()
    state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=parameters_for_test(),
    )
    cfg.array.antenna_model = phone_antenna_config_for_test(
        source="parametric", true_state=state
    )

    cfg.validate()


def test_nonempty_parametric_mapping_always_uses_exact_corner_names():
    cfg = default_config()
    state = AntennaStateConfig(
        parametric_patterns={"XX": ParametricPatternParameters()}
    )
    cfg.array.antenna_model = phone_antenna_config_for_test(true_state=state)

    with pytest.raises(ValueError, match="parametric.*TL/TR/BL/BR"):
        cfg.validate()


@pytest.mark.parametrize("missing_field", ["measured_pattern_csv", "measured_pattern_sha256"])
def test_measured_state_requires_existing_path_and_digest(tmp_path: Path, missing_field):
    csv_path = tmp_path / "pattern.csv"
    csv_path.write_text("placeholder", encoding="utf-8")
    values = {
        "pattern_kind": "measured",
        "measured_pattern_csv": csv_path,
        "measured_pattern_sha256": "a" * 64,
    }
    values[missing_field] = None
    cfg = default_config()
    cfg.array.antenna_model = phone_antenna_config_for_test(
        source="measured", true_state=AntennaStateConfig(**values)
    )

    with pytest.raises(ValueError, match="measured"):
        cfg.validate()


def test_measured_state_rejects_missing_file_and_bad_sha(tmp_path: Path):
    cfg = default_config()
    missing = AntennaStateConfig(
        pattern_kind="measured",
        measured_pattern_csv=tmp_path / "missing.csv",
        measured_pattern_sha256="a" * 64,
    )
    cfg.array.antenna_model = phone_antenna_config_for_test(
        source="measured", true_state=missing
    )
    with pytest.raises(ValueError, match="exist"):
        cfg.validate()

    source = tmp_path / "pattern.csv"
    source.write_text("placeholder", encoding="utf-8")
    cfg.array.antenna_model.true_state = replace(
        missing,
        measured_pattern_csv=source,
        measured_pattern_sha256="not-a-sha",
    )
    with pytest.raises(ValueError, match="SHA-256"):
        cfg.validate()


def test_nonmeasured_state_does_not_probe_irrelevant_path(tmp_path: Path):
    cfg = default_config()
    state = AntennaStateConfig(measured_pattern_csv=tmp_path / "missing.csv")
    cfg.array.antenna_model = phone_antenna_config_for_test(true_state=state)

    cfg.validate()


@pytest.mark.parametrize(
    "region",
    [
        AttenuationRegion(("XX",), 0.0, 0.0, 10.0, 1.0),
        AttenuationRegion(("TL", "TL"), 0.0, 0.0, 10.0, 1.0),
        AttenuationRegion(("TL",), 0.0, 0.0, -1.0, 1.0),
        AttenuationRegion(("TL",), 0.0, 0.0, 1.0, -1.0),
        AttenuationRegion(("TL",), 0.0, 0.0, 1.0, 1.0, -1.0),
    ],
)
def test_root_validation_rejects_invalid_attenuation_regions(region):
    cfg = default_config()
    state = AntennaStateConfig(attenuation_regions=(region,))
    cfg.array.antenna_model = phone_antenna_config_for_test(true_state=state)

    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize(
    "paths",
    [
        (CouplingPath("TL", "TL", -20.0, 0.0),),
        (
            CouplingPath("TL", "BR", -20.0, 0.0),
            CouplingPath("TL", "BR", -10.0, 90.0),
        ),
        (CouplingPath("XX", "BR", -20.0, 0.0),),
    ],
)
def test_root_validation_rejects_self_duplicate_or_unknown_coupling_paths(paths):
    cfg = default_config()
    state = AntennaStateConfig(coupling_paths=paths)
    cfg.array.antenna_model = phone_antenna_config_for_test(true_state=state)

    with pytest.raises(ValueError):
        cfg.validate()


def test_state_rejects_wrong_record_types_and_non_tuple_effect_collections():
    cfg = default_config()
    state = AntennaStateConfig()
    state.orientations["TL"] = object()
    state.attenuation_regions = []
    state.coupling_paths = []
    cfg.array.antenna_model = phone_antenna_config_for_test(true_state=state)

    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize(
    "orientation",
    [
        ElementOrientation(yaw_deg=0.0),
        ElementOrientation(pitch_deg=0.0),
        ElementOrientation(roll_deg=0.0),
    ],
)
def test_orientation_records_are_accepted_for_all_axes(orientation):
    cfg = default_config()
    state = AntennaStateConfig()
    state.orientations["TL"] = orientation
    cfg.array.antenna_model = phone_antenna_config_for_test(true_state=state)

    cfg.validate()


def test_mate70_config_uses_all_four_ideal_corners():
    cfg = mate70_default_config()

    assert cfg.phone.model_name == "HUAWEI Mate 70"
    assert cfg.phone.length_m == pytest.approx(0.1609)
    assert cfg.phone.width_m == pytest.approx(0.0759)
    assert cfg.array.active_antennas == ("TL", "TR", "BL", "BR")
    np.testing.assert_allclose(
        [cfg.phone.phase_centers_m[name] for name in cfg.array.active_antennas],
        [
            [-0.03795, 0.08045, 0.0],
            [0.03795, 0.08045, 0.0],
            [-0.03795, -0.08045, 0.0],
            [0.03795, -0.08045, 0.0],
        ],
    )


def test_mate70_config_disables_legacy_jammer_when_using_cw_interferer():
    cfg = mate70_default_config()

    assert not cfg.jammer.enable
    assert len(cfg.interferers) == 1
    assert cfg.interferers[0].enable


def test_phone_config_rejects_enabled_legacy_jammer():
    cfg = mate70_default_config()
    cfg.jammer.enable = True

    with pytest.raises(ValueError, match="legacy jammer.*phone"):
        cfg.validate()


def test_legacy_config_rejects_phone_interferers():
    cfg = legacy_ula_config()
    cfg.interferers = (CWInterferer(Direction(120.0, 10.0)),)

    with pytest.raises(ValueError, match="interferers.*legacy"):
        cfg.validate()


def test_phone_channel_response_uses_real_errors_in_active_antenna_order():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("BR", "TL", "TR")
    cfg.array.channel_amplitude_error_db = {
        "TL": 0.0,
        "TR": -20.0,
        "BL": 3.0,
        "BR": 20.0,
    }
    cfg.array.channel_phase_error_deg = {
        "TL": 0.0,
        "TR": 180.0,
        "BL": 45.0,
        "BR": 90.0,
    }

    response = cfg.array.channel_response(estimated=False)

    np.testing.assert_allclose(response, [10.0j, 1.0 + 0.0j, -0.1 + 0.0j], atol=1e-12)


def test_phone_channel_response_uses_estimated_errors_in_active_antenna_order():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("BR", "TL", "TR")
    cfg.array.estimated_channel_amplitude_error_db = {
        "TL": 20.0,
        "TR": 20.0 * np.log10(2.0),
        "BL": -3.0,
        "BR": 20.0 * np.log10(0.5),
    }
    cfg.array.estimated_channel_phase_error_deg = {
        "TL": 180.0,
        "TR": 0.0,
        "BL": 30.0,
        "BR": -90.0,
    }

    response = cfg.array.channel_response(estimated=True)

    np.testing.assert_allclose(response, [-0.5j, -10.0 + 0.0j, 2.0 + 0.0j], atol=1e-12)


def test_phone_config_accepts_arbitrary_unique_two_to_four_corner_subset():
    cfg = mate70_default_config()
    cfg.array.active_antennas = ("TR", "BL")
    cfg.validate()

    assert cfg.array.num_elements == 2
    np.testing.assert_allclose(
        [cfg.phone.phase_centers_m[name] for name in cfg.array.active_antennas],
        [[0.03795, 0.08045, 0.0], [-0.03795, -0.08045, 0.0]],
    )


@pytest.mark.parametrize(
    "active",
    [("TL",), ("TL", "TL"), ("TL", "XX"), ("TL", "TR", "BL", "BR", "TL")],
)
def test_phone_config_rejects_invalid_corner_selections(active):
    cfg = mate70_default_config()
    cfg.array.active_antennas = active

    with pytest.raises(ValueError, match="active_antennas"):
        cfg.validate()


def test_phone_config_rejects_consistent_noncorner_coordinate_and_error_maps():
    cfg = mate70_default_config()
    noncorners = ("XX", "YY", "ZZ", "WW")
    phase_centers = {
        name: (float(index), 0.0, 0.0)
        for index, name in enumerate(noncorners)
    }
    cfg.phone.phase_centers_m = phase_centers
    cfg.phone.assumed_phase_centers_m = phase_centers.copy()
    cfg.array.active_antennas = ("XX", "YY")
    cfg.array.channel_amplitude_error_db = {name: 0.0 for name in noncorners}
    cfg.array.channel_phase_error_deg = {name: 0.0 for name in noncorners}
    cfg.array.estimated_channel_amplitude_error_db = {name: 0.0 for name in noncorners}
    cfg.array.estimated_channel_phase_error_deg = {name: 0.0 for name in noncorners}

    with pytest.raises(ValueError, match="corner-name set"):
        cfg.validate()


@pytest.mark.parametrize("elevation", [-0.1, 90.1])
def test_direction_rejects_elevation_outside_visible_hemisphere(elevation):
    with pytest.raises(ValueError, match="elevation"):
        Direction(30.0, elevation).validate()


def test_phone_config_rejects_more_than_three_interferers():
    cfg = mate70_default_config()
    cfg.interferers = tuple(
        CWInterferer(direction=Direction(index * 30.0, 10.0), enable=False)
        for index in range(4)
    )

    with pytest.raises(ValueError, match="three"):
        cfg.validate()


@pytest.mark.parametrize("count", [0, 3])
def test_phone_config_accepts_zero_to_three_configured_interferers(count):
    cfg = mate70_default_config()
    cfg.interferers = tuple(
        CWInterferer(direction=Direction(index * 30.0, 10.0), enable=False)
        for index in range(count)
    )

    cfg.validate()


def test_phone_config_validates_each_interferer_direction():
    cfg = mate70_default_config()
    cfg.interferers = (CWInterferer(Direction(np.nan, 10.0)),)

    with pytest.raises(ValueError, match="direction.azimuth_deg"):
        cfg.validate()


@pytest.mark.parametrize(
    "direction",
    [
        Direction(120.0 + 1.0j, 10.0),
        Direction(np.array([120.0]), 10.0),
        Direction(120.0, 10.0 + 1.0j),
        Direction(120.0, np.array([10.0])),
    ],
)
def test_phone_config_rejects_non_real_scalar_interferer_directions(direction):
    cfg = mate70_default_config()
    cfg.interferers = (CWInterferer(direction),)

    with pytest.raises(ValueError, match="direction\\.(azimuth|elevation)_deg"):
        cfg.validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"power_dbm": np.nan}, "interferer.power_dbm"),
        ({"offset_hz": np.inf}, "interferer.offset_hz"),
        ({"initial_phase_rad": -np.inf}, "interferer.initial_phase_rad"),
        ({"start_ms": np.nan}, "interferer.start_ms"),
        ({"start_ms": -0.1}, "interferer.start_ms"),
        ({"stop_ms": np.inf}, "interferer.stop_ms"),
        ({"start_ms": 4.0, "stop_ms": 3.9}, "interferer.stop_ms"),
        ({"start_ms": 40.1}, "interferer.start_ms.*signal.duration_ms"),
        ({"stop_ms": 40.1}, "interferer.stop_ms.*signal.duration_ms"),
    ],
)
def test_phone_config_rejects_invalid_interferer_fields(overrides, match):
    cfg = mate70_default_config()
    cfg.interferers = (CWInterferer(Direction(120.0, 10.0), **overrides),)

    with pytest.raises(ValueError, match=match):
        cfg.validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"power_dbm": -90.0 + 1.0j}, "interferer.power_dbm"),
        ({"power_dbm": np.array([-90.0])}, "interferer.power_dbm"),
        ({"offset_hz": 250e3 + 1.0j}, "interferer.offset_hz"),
        ({"offset_hz": np.array([250e3])}, "interferer.offset_hz"),
        ({"initial_phase_rad": 0.8 + 1.0j}, "interferer.initial_phase_rad"),
        ({"initial_phase_rad": np.array([0.8])}, "interferer.initial_phase_rad"),
        ({"start_ms": 1.0 + 1.0j}, "interferer.start_ms"),
        ({"start_ms": np.array([1.0])}, "interferer.start_ms"),
        ({"stop_ms": 2.0 + 1.0j}, "interferer.stop_ms"),
        ({"stop_ms": np.array([2.0])}, "interferer.stop_ms"),
    ],
)
def test_phone_config_rejects_non_real_scalar_interferer_fields(overrides, match):
    cfg = mate70_default_config()
    cfg.interferers = (CWInterferer(Direction(120.0, 10.0), **overrides),)

    with pytest.raises(ValueError, match=match):
        cfg.validate()


def test_phone_config_accepts_interferer_gate_at_signal_duration_boundary():
    cfg = mate70_default_config()
    cfg.interferers = (
        CWInterferer(Direction(120.0, 10.0), start_ms=40.0),
        CWInterferer(Direction(220.0, 20.0), stop_ms=40.0),
    )

    cfg.validate()


def test_legacy_constructor_preserves_original_sampling_and_array():
    cfg = legacy_ula_config()

    assert cfg.array.num_elements == 2
    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.array.desired_angle_deg == -10.0
