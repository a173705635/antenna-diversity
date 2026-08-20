import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from antenna_diversity.array.antenna_config import (
    AntennaStateConfig,
    PhoneAntennaConfig,
)
from antenna_diversity.array.effects import (
    AttenuationRegion,
    CouplingPath,
    ElementOrientation,
)
from antenna_diversity.array.geometry import select_phase_centers
from antenna_diversity.array.manifold import IsotropicElementPattern, array_manifold
from antenna_diversity.array.patterns import (
    ParametricPatternParameters,
    TabulatedElementPattern,
)
from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.array.polarization import WavePolarization
from antenna_diversity.config import (
    Direction,
    SimulationConfig,
    default_config,
    legacy_ula_config,
)


CORNER_NAMES = ("TL", "TR", "BL", "BR")
ZENITH = Direction(0.0, 90.0)
RHCP = WavePolarization("rhcp")


def _parametric_config(
    active_antennas: tuple[str, ...],
    voltage_gain: dict[str, float],
) -> SimulationConfig:
    cfg = default_config()
    cfg.array.active_antennas = active_antennas
    zero_positions = {name: (0.0, 0.0, 0.0) for name in CORNER_NAMES}
    cfg.phone.phase_centers_m = zero_positions.copy()
    cfg.phone.assumed_phase_centers_m = zero_positions.copy()
    parameters = _parametric_parameters(voltage_gain)
    true_state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=parameters,
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="test parametric model",
        model_source="parametric",
        true_state=true_state,
        assumed_state=copy.deepcopy(true_state),
    )
    cfg.validate()
    return cfg


def _parametric_parameters(
    voltage_gain: dict[str, float],
) -> dict[str, ParametricPatternParameters]:
    return {
        name: ParametricPatternParameters(
            peak_gain_db=20.0 * np.log10(voltage_gain[name]),
            back_lobe_gain_db=-100.0,
            nominal_polarization=RHCP,
            cross_polar_discrimination_db=300.0,
        )
        for name in CORNER_NAMES
    }


def _model_for_pattern_kind(pattern_kind: str, tmp_path: Path) -> PhoneAntennaModel:
    if pattern_kind == "isotropic":
        return PhoneAntennaModel.from_config(default_config())
    if pattern_kind == "parametric":
        cfg = _parametric_config(
            ("TL", "TR"), {name: 1.0 for name in CORNER_NAMES}
        )
        return PhoneAntennaModel.from_config(cfg)
    cfg = default_config()
    cfg.array.active_antennas = ("TL", "TR")
    frequency_hz = cfg.signal.carrier_hz
    component = 2.0**-0.5
    pattern_path = tmp_path / "validation-pattern.csv"
    pattern_path.write_text(
        "element_name,frequency_hz,azimuth_deg,elevation_deg,"
        "h_az_real,h_az_imag,h_el_real,h_el_imag\n"
        f"TL,{frequency_hz},0,90,{component},0,0,{-component}\n"
        f"TR,{frequency_hz},0,90,{component},0,0,{-component}\n",
        encoding="utf-8",
    )
    state = AntennaStateConfig(
        pattern_kind="measured",
        measured_pattern_csv=pattern_path,
        measured_pattern_sha256=hashlib.sha256(pattern_path.read_bytes()).hexdigest(),
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="measured validation model",
        model_source="measured",
        true_state=state,
        assumed_state=copy.deepcopy(state),
    )
    return PhoneAntennaModel.from_config(cfg)


def test_phone_antenna_model_is_exported_from_array_package():
    from antenna_diversity.array import PhoneAntennaModel as PublicPhoneAntennaModel

    assert PublicPhoneAntennaModel is PhoneAntennaModel


def test_from_config_validates_a_deep_copy_without_mutating_caller(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = default_config()
    original_validate = SimulationConfig.validate

    def mutating_validate(candidate: SimulationConfig) -> None:
        candidate.random_seed = 99
        original_validate(candidate)

    monkeypatch.setattr(SimulationConfig, "validate", mutating_validate)

    PhoneAntennaModel.from_config(cfg)

    assert cfg.random_seed == 42


def test_from_config_rejects_non_phone_configuration():
    with pytest.raises(ValueError, match="phone-array configuration"):
        PhoneAntennaModel.from_config(legacy_ula_config())


def test_ideal_phone_model_exactly_matches_existing_manifold_and_channel_response():
    cfg = default_config()
    names = cfg.array.active_antennas
    expected_nominal = array_manifold(
        select_phase_centers(cfg.phone.phase_centers_m, names),
        cfg.signal.true_direction,
        cfg.signal.carrier_hz,
        names,
        IsotropicElementPattern(),
    )

    model = PhoneAntennaModel.from_config(cfg)

    np.testing.assert_allclose(
        model.open_circuit_response(
            cfg.signal.true_direction,
            cfg.signal.true_polarization,
            cfg.signal.carrier_hz,
            estimated=False,
        ),
        expected_nominal,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.effective_manifold(
            cfg.signal.true_direction,
            cfg.signal.true_polarization,
            cfg.signal.carrier_hz,
            estimated=False,
        ),
        cfg.array.channel_response(False) * expected_nominal,
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "polarization",
    [
        WavePolarization("rhcp"),
        WavePolarization("lhcp"),
        WavePolarization("linear", tilt_deg=37.0),
        WavePolarization(
            "jones", az_real=0.2, az_imag=-0.3, el_real=0.4, el_imag=0.5
        ),
    ],
)
def test_ideal_open_circuit_is_exactly_polarization_independent(polarization):
    cfg = default_config()
    names = cfg.array.active_antennas
    expected = array_manifold(
        select_phase_centers(cfg.phone.phase_centers_m, names),
        cfg.signal.true_direction,
        cfg.signal.carrier_hz,
        names,
        IsotropicElementPattern(),
    )

    actual = PhoneAntennaModel.from_config(cfg).open_circuit_response(
        cfg.signal.true_direction,
        polarization,
        cfg.signal.carrier_hz,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "active_antennas",
    [
        ("BR", "TL"),
        ("BL", "TR", "TL"),
        ("TR", "BR", "TL", "BL"),
    ],
)
def test_parametric_open_circuit_preserves_arbitrary_active_order(active_antennas):
    voltage_gain = {"TL": 1.0, "TR": 2.0, "BL": 3.0, "BR": 4.0}
    cfg = _parametric_config(active_antennas, voltage_gain)

    response = PhoneAntennaModel.from_config(cfg).open_circuit_response(
        ZENITH, RHCP, cfg.signal.carrier_hz
    )

    np.testing.assert_allclose(
        response,
        [voltage_gain[name] for name in active_antennas],
        rtol=0.0,
        atol=1e-12,
    )


def test_parametric_open_circuit_applies_element_orientation_in_local_frame():
    cfg = _parametric_config(("TL", "TR"), {name: 1.0 for name in CORNER_NAMES})
    cfg.array.antenna_model.true_state.orientations["TR"] = ElementOrientation(
        pitch_deg=90.0
    )

    response = PhoneAntennaModel.from_config(cfg).open_circuit_response(
        ZENITH, RHCP, cfg.signal.carrier_hz
    )

    np.testing.assert_allclose(np.abs(response), [1.0, 1e-5], rtol=0.0, atol=1e-12)


def test_parametric_open_circuit_uses_conjugating_jones_inner_product():
    cfg = _parametric_config(("TL", "TR"), {name: 1.0 for name in CORNER_NAMES})
    state = cfg.array.antenna_model.true_state
    state.parametric_patterns = {
        name: ParametricPatternParameters(
            peak_gain_db=0.0,
            back_lobe_gain_db=-100.0,
            nominal_polarization=RHCP,
            cross_polar_discrimination_db=20.0,
        )
        for name in CORNER_NAMES
    }
    model = PhoneAntennaModel.from_config(cfg)

    matched = model.open_circuit_response(ZENITH, RHCP, cfg.signal.carrier_hz)
    orthogonal = model.open_circuit_response(
        ZENITH, WavePolarization("lhcp"), cfg.signal.carrier_hz
    )

    np.testing.assert_allclose(
        np.abs(orthogonal) / np.abs(matched),
        [0.1, 0.1],
        rtol=0.0,
        atol=1e-12,
    )


def test_open_circuit_multiplies_overlapping_attenuation_in_voltage():
    cfg = _parametric_config(("TL", "TR"), {name: 1.0 for name in CORNER_NAMES})
    cfg.array.antenna_model.true_state.attenuation_regions = (
        AttenuationRegion(("TL",), 0.0, 90.0, 1.0, 6.0),
        AttenuationRegion(("TL",), 0.0, 90.0, 1.0, 14.0),
    )

    response = PhoneAntennaModel.from_config(cfg).open_circuit_response(
        ZENITH, RHCP, cfg.signal.carrier_hz
    )

    np.testing.assert_allclose(np.abs(response), [0.1, 1.0], rtol=0.0, atol=1e-12)


def test_effective_response_is_channel_times_asymmetric_coupling_times_open_circuit():
    cfg = _parametric_config(
        ("BR", "TL"),
        {"TL": 10.0, "TR": 1.0, "BL": 1.0, "BR": 1.0},
    )
    cfg.array.antenna_model.true_state.coupling_paths = (
        CouplingPath("BR", "TL", -6.020599913279624, 90.0),
    )
    cfg.array.channel_amplitude_error_db["BR"] = 6.020599913279624
    cfg.array.channel_phase_error_deg["TL"] = 90.0
    model = PhoneAntennaModel.from_config(cfg)

    open_circuit = model.open_circuit_response(ZENITH, RHCP, cfg.signal.carrier_hz)
    effective = model.effective_manifold(ZENITH, RHCP, cfg.signal.carrier_hz)

    np.testing.assert_allclose(open_circuit, [1.0, 10.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        model.coupling_matrix(False),
        [[1.0, 0.0], [0.5j, 1.0]],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        effective,
        [2.0, -0.5 + 10.0j],
        rtol=0.0,
        atol=1e-12,
    )


def test_estimated_response_uses_only_assumed_geometry_state_and_channel():
    cfg = _parametric_config(("TL", "TR"), {name: 1.0 for name in CORNER_NAMES})
    half_wavelength = cfg.signal.wavelength_m / 2.0
    cfg.phone.assumed_phase_centers_m["TL"] = (0.0, 0.0, half_wavelength)
    cfg.array.antenna_model.assumed_state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=_parametric_parameters(
            {"TL": 2.0, "TR": 3.0, "BL": 1.0, "BR": 1.0}
        ),
        coupling_paths=(CouplingPath("TL", "TR", -6.020599913279624, 0.0),),
    )
    cfg.array.estimated_channel_amplitude_error_db["TL"] = 6.020599913279624
    cfg.array.estimated_channel_amplitude_error_db["TR"] = 12.041199826559248
    model = PhoneAntennaModel.from_config(cfg)

    true_open = model.open_circuit_response(
        ZENITH, RHCP, cfg.signal.carrier_hz, estimated=False
    )
    assumed_open = model.open_circuit_response(
        ZENITH, RHCP, cfg.signal.carrier_hz, estimated=True
    )

    np.testing.assert_allclose(true_open, [1.0, 1.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(assumed_open, [-2.0, 3.0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        model.effective_manifold(
            ZENITH, RHCP, cfg.signal.carrier_hz, estimated=True
        ),
        [-4.0, 8.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.channel_response(False), [1.0, 1.0], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        model.channel_response(True), [2.0, 4.0], rtol=0.0, atol=1e-12
    )


def test_measured_pattern_loads_once_per_runtime_state_not_per_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg = default_config()
    cfg.array.active_antennas = ("TL", "TR")
    frequency_hz = cfg.signal.carrier_hz
    component = 2.0**-0.5
    pattern_path = tmp_path / "phone-pattern.csv"
    pattern_path.write_text(
        "element_name,frequency_hz,azimuth_deg,elevation_deg,"
        "h_az_real,h_az_imag,h_el_real,h_el_imag\n"
        f"TL,{frequency_hz},0,90,{component},0,0,{-component}\n"
        f"TR,{frequency_hz},0,90,{component},0,0,{-component}\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(pattern_path.read_bytes()).hexdigest()
    state = AntennaStateConfig(
        pattern_kind="measured",
        measured_pattern_csv=pattern_path,
        measured_pattern_sha256=digest,
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="measured test model",
        model_source="measured",
        true_state=state,
        assumed_state=copy.deepcopy(state),
    )
    original_read_bytes = Path.read_bytes
    reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == pattern_path:
            reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)

    model = PhoneAntennaModel.from_config(cfg)
    assert reads == 2
    for estimated in (False, True):
        for _ in range(3):
            response = model.open_circuit_response(
                ZENITH, RHCP, frequency_hz, estimated=estimated
            )
            assert np.all(np.isfinite(response))
    assert reads == 2


def test_diagnostics_are_finite_detached_snapshots_without_runtime_objects():
    cfg = _parametric_config(("TR", "BL"), {name: 1.0 for name in CORNER_NAMES})
    model = PhoneAntennaModel.from_config(cfg)

    diagnostics = model.diagnostics(ZENITH, RHCP, cfg.signal.carrier_hz)

    assert set(diagnostics) == {
        "active_antennas",
        "estimated",
        "model_label",
        "model_source",
        "pattern_kind",
        "frequency_hz",
        "direction_azimuth_deg",
        "direction_elevation_deg",
        "polarization_jones",
        "positions_m",
        "spatial_phase",
        "pattern_polarization_voltage",
        "blockage_voltage",
        "open_circuit_response",
        "coupling_matrix",
        "coupled_response",
        "channel_response",
        "effective_manifold",
    }
    assert diagnostics["active_antennas"] == ("TR", "BL")
    assert diagnostics["estimated"] is False
    assert diagnostics["model_label"] == "test parametric model"
    assert diagnostics["model_source"] == "parametric"
    assert diagnostics["pattern_kind"] == "parametric"
    assert diagnostics["frequency_hz"] == cfg.signal.carrier_hz
    assert diagnostics["direction_azimuth_deg"] == 0.0
    assert diagnostics["direction_elevation_deg"] == 90.0
    numeric_keys = (
        "polarization_jones",
        "positions_m",
        "spatial_phase",
        "pattern_polarization_voltage",
        "blockage_voltage",
        "open_circuit_response",
        "coupling_matrix",
        "coupled_response",
        "channel_response",
        "effective_manifold",
    )
    for key in numeric_keys:
        assert isinstance(diagnostics[key], np.ndarray)
        assert np.all(np.isfinite(diagnostics[key]))

    originals = {key: diagnostics[key].copy() for key in numeric_keys}
    for key in numeric_keys:
        diagnostics[key].flat[0] = np.nan
    fresh = model.diagnostics(ZENITH, RHCP, cfg.signal.carrier_hz)
    for key in numeric_keys:
        assert np.all(np.isfinite(fresh[key]))
        np.testing.assert_allclose(fresh[key], originals[key], rtol=0.0, atol=0.0)


def test_diagnostics_decompose_every_response_stage_with_numeric_source_context():
    cfg = _parametric_config(
        ("BR", "TL"),
        {"TL": 10.0, "TR": 1.0, "BL": 1.0, "BR": 1.0},
    )
    cfg.array.antenna_model.true_state.attenuation_regions = (
        AttenuationRegion(("TL",), 0.0, 90.0, 1.0, 6.020599913279624),
    )
    cfg.array.antenna_model.true_state.coupling_paths = (
        CouplingPath("BR", "TL", -6.020599913279624, 90.0),
    )
    cfg.array.channel_amplitude_error_db["BR"] = 6.020599913279624
    cfg.array.channel_phase_error_deg["TL"] = 90.0
    model = PhoneAntennaModel.from_config(cfg)

    diagnostics = model.diagnostics(ZENITH, RHCP, cfg.signal.carrier_hz)

    assert diagnostics["frequency_hz"] == cfg.signal.carrier_hz
    assert diagnostics["direction_azimuth_deg"] == 0.0
    assert diagnostics["direction_elevation_deg"] == 90.0
    np.testing.assert_allclose(
        diagnostics["polarization_jones"],
        [2.0**-0.5, -1.0j * 2.0**-0.5],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["spatial_phase"], [1.0, 1.0], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        diagnostics["pattern_polarization_voltage"],
        [1.0, 10.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["blockage_voltage"], [1.0, 0.5], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        diagnostics["open_circuit_response"],
        diagnostics["spatial_phase"]
        * diagnostics["pattern_polarization_voltage"]
        * diagnostics["blockage_voltage"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["coupled_response"],
        diagnostics["coupling_matrix"] @ diagnostics["open_circuit_response"],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["effective_manifold"],
        diagnostics["channel_response"] * diagnostics["coupled_response"],
        rtol=0.0,
        atol=1e-12,
    )


def test_internal_numeric_state_has_irreversible_read_only_backing():
    cfg = _parametric_config(("TL", "BR"), {name: 1.0 for name in CORNER_NAMES})
    model = PhoneAntennaModel.from_config(cfg)
    true_before = model.effective_manifold(ZENITH, RHCP, cfg.signal.carrier_hz)
    assumed_before = model.effective_manifold(
        ZENITH, RHCP, cfg.signal.carrier_hz, estimated=True
    )
    internal_arrays = (
        model.true_positions,
        model.assumed_positions,
        model.true_channel,
        model.assumed_channel,
        model.true_state.coupling,
        model.assumed_state.coupling,
        *model.true_state.rotations.values(),
        *model.assumed_state.rotations.values(),
    )

    for values in internal_arrays:
        assert values.flags.writeable is False
        with pytest.raises(ValueError):
            values.setflags(write=True)
        with pytest.raises(ValueError):
            values.flat[0] = 123.0

    np.testing.assert_allclose(
        model.effective_manifold(ZENITH, RHCP, cfg.signal.carrier_hz),
        true_before,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        model.effective_manifold(
            ZENITH, RHCP, cfg.signal.carrier_hz, estimated=True
        ),
        assumed_before,
        rtol=0.0,
        atol=0.0,
    )


def test_measured_runtime_grid_storage_cannot_change_model_response(tmp_path: Path):
    model = _model_for_pattern_kind("measured", tmp_path)
    frequency_hz = 1_575.42e6
    true_before = model.effective_manifold(ZENITH, RHCP, frequency_hz)
    assumed_before = model.effective_manifold(
        ZENITH, RHCP, frequency_hz, estimated=True
    )

    for state in (model.true_state, model.assumed_state):
        assert isinstance(state.pattern, TabulatedElementPattern)
        for grid in state.pattern.grids.values():
            for values in (grid.azimuths_deg, grid.elevations_deg, grid.heights):
                assert values.flags.writeable is False
                with pytest.raises(ValueError):
                    values.setflags(write=True)
                with pytest.raises(ValueError):
                    values.flat[0] = 123.0

    np.testing.assert_allclose(
        model.effective_manifold(ZENITH, RHCP, frequency_hz),
        true_before,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        model.effective_manifold(ZENITH, RHCP, frequency_hz, estimated=True),
        assumed_before,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("pattern_kind", ["isotropic", "parametric", "measured"])
def test_public_response_entries_reject_non_direction_objects(
    pattern_kind: str, tmp_path: Path
):
    model = _model_for_pattern_kind(pattern_kind, tmp_path)

    for method in (
        model.open_circuit_response,
        model.effective_manifold,
        model.diagnostics,
    ):
        with pytest.raises(ValueError, match="direction"):
            method(object(), RHCP, 1_575.42e6)


@pytest.mark.parametrize("pattern_kind", ["isotropic", "parametric", "measured"])
@pytest.mark.parametrize(
    "frequency_hz",
    [True, 1.0 + 0.0j, np.array([1.0]), 0.0, -1.0, np.inf, np.nan],
)
def test_public_response_entries_reject_invalid_frequency_for_every_pattern_kind(
    pattern_kind: str, frequency_hz: object, tmp_path: Path
):
    model = _model_for_pattern_kind(pattern_kind, tmp_path)

    for method in (
        model.open_circuit_response,
        model.effective_manifold,
        model.diagnostics,
    ):
        with pytest.raises(ValueError, match="frequency_hz"):
            method(ZENITH, RHCP, frequency_hz)


def test_factory_rejects_nonfinite_derived_channel_without_warning():
    cfg = default_config()
    cfg.array.channel_amplitude_error_db["TL"] = 1e308

    with pytest.raises(ValueError, match="true channel.*finite"):
        PhoneAntennaModel.from_config(cfg)


def test_factory_rejects_nonfinite_derived_coupling_without_warning():
    cfg = default_config()
    state = AntennaStateConfig(
        coupling_paths=(CouplingPath("TL", "TR", 1e308, 0.0),)
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="coupling overflow",
        model_source="ideal",
        true_state=state,
        assumed_state=copy.deepcopy(state),
    )

    with pytest.raises(ValueError, match="coupling matrix.*finite"):
        PhoneAntennaModel.from_config(cfg)


def test_open_circuit_rejects_nonfinite_pattern_voltage_without_warning():
    cfg = _parametric_config(("TL", "TR"), {name: 1.0 for name in CORNER_NAMES})
    cfg.array.antenna_model.true_state.parametric_patterns = {
        name: ParametricPatternParameters(
            peak_gain_db=1e308,
            back_lobe_gain_db=-100.0,
            nominal_polarization=RHCP,
        )
        for name in CORNER_NAMES
    }
    model = PhoneAntennaModel.from_config(cfg)

    with pytest.raises(ValueError, match="pattern/polarization voltage.*finite"):
        model.open_circuit_response(ZENITH, RHCP, cfg.signal.carrier_hz)


@pytest.mark.parametrize("method_name", ["effective_manifold", "diagnostics"])
def test_composed_response_rejects_nonfinite_effective_voltage_without_warning(
    method_name: str,
):
    cfg = _parametric_config(
        ("TL", "TR"), {name: 1e150 for name in CORNER_NAMES}
    )
    cfg.array.channel_amplitude_error_db["TL"] = 6000.0
    model = PhoneAntennaModel.from_config(cfg)

    with pytest.raises(ValueError, match="effective manifold.*finite"):
        getattr(model, method_name)(ZENITH, RHCP, cfg.signal.carrier_hz)


def test_structured_isotropic_state_applies_blockage_and_coupling():
    cfg = default_config()
    cfg.array.active_antennas = ("TL", "TR")
    zero_positions = {name: (0.0, 0.0, 0.0) for name in CORNER_NAMES}
    cfg.phone.phase_centers_m = zero_positions.copy()
    cfg.phone.assumed_phase_centers_m = zero_positions.copy()
    state = AntennaStateConfig(
        attenuation_regions=(
            AttenuationRegion(("TL",), 0.0, 90.0, 1.0, 6.020599913279624),
        ),
        coupling_paths=(
            CouplingPath("TL", "TR", -6.020599913279624, 0.0),
        ),
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="structured isotropic",
        model_source="ideal",
        true_state=state,
        assumed_state=copy.deepcopy(state),
    )
    model = PhoneAntennaModel.from_config(cfg)

    diagnostics = model.diagnostics(ZENITH, RHCP, cfg.signal.carrier_hz)

    np.testing.assert_allclose(
        diagnostics["pattern_polarization_voltage"],
        [1.0, 1.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["blockage_voltage"], [0.5, 1.0], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        diagnostics["open_circuit_response"],
        [0.5, 1.0],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        diagnostics["effective_manifold"],
        [0.5, 1.25],
        rtol=0.0,
        atol=1e-12,
    )
