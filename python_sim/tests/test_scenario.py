import copy

import numpy as np
import pytest

from antenna_diversity.array.antenna_config import (
    AntennaStateConfig,
    PhoneAntennaConfig,
)
from antenna_diversity.array.effects import AttenuationRegion, CouplingPath
from antenna_diversity.array.patterns import ParametricPatternParameters
from antenna_diversity.array.phone_model import PhoneAntennaModel
from antenna_diversity.config import legacy_ula_config, mate70_default_config
from antenna_diversity.gnss.scenario import generate_scenario
from antenna_diversity.studies.models import strict_snapshot


def _unit_parametric_patterns():
    return {
        name: ParametricPatternParameters(
            back_lobe_gain_db=0.0,
            cross_polar_discrimination_db=300.0,
        )
        for name in ("TL", "TR", "BL", "BR")
    }


def _complex_diagnostic(snapshot):
    return np.asarray(snapshot["real"]) + 1j * np.asarray(snapshot["imag"])


def test_scenario_projects_signal_through_both_antennas():
    """Catch loss of code Doppler, array response, or antennas-by-samples shape."""
    cfg = legacy_ula_config()
    cfg.jammer.enable = False

    x, truth = generate_scenario(cfg)

    expected_samples = round(cfg.signal.fs_rf_hz * cfg.signal.duration_ms * 1e-3)
    assert x.shape == (cfg.array.num_elements, expected_samples)
    expected_rate = cfg.signal.code_rate_hz * (
        1 + cfg.signal.doppler_hz / cfg.signal.carrier_hz
    )
    assert truth["code_rate_hz"] == pytest.approx(expected_rate, abs=1e-9)
    ratio = np.mean(x[1] / x[0])
    expected_ratio = (
        truth["desired_steering_effective"][1]
        / truth["desired_steering_effective"][0]
    )
    assert ratio == pytest.approx(expected_ratio, abs=1e-10)


def test_channel_amplitude_and_phase_errors_modify_each_antenna_response():
    """Catch omission of per-channel complex gain errors from scenario generation."""
    cfg = legacy_ula_config()
    cfg.jammer.enable = False
    cfg.array.channel_amplitude_error_db = np.array([0.0, 6.020599913279624])
    cfg.array.channel_phase_error_deg = np.array([0.0, 90.0])

    _, truth = generate_scenario(cfg)

    np.testing.assert_allclose(truth["channel_response"], [1.0, 2.0j], atol=1e-12)


def test_legacy_scenario_preserves_singular_truth_aliases():
    cfg = legacy_ula_config()

    _, truth = generate_scenario(cfg)

    np.testing.assert_allclose(
        truth["desired_steering_nominal"], truth["desired_steering_assumed"]
    )
    np.testing.assert_allclose(
        truth["desired_steering_effective"], truth["desired_steering_true"]
    )
    np.testing.assert_allclose(
        truth["jammer_steering_nominal"], truth["interferer_steering_nominal"][0]
    )
    np.testing.assert_allclose(
        truth["jammer_steering_effective"], truth["interferer_steering_effective"][0]
    )
    np.testing.assert_allclose(
        truth["jammer_component"], truth["interferer_components"][0]
    )


def test_scenario_truth_and_samples_use_shared_phone_antenna_model():
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 1
    cfg.array.active_antennas = ("TL", "TR")
    true_state = AntennaStateConfig(
        pattern_kind="parametric",
        parametric_patterns=_unit_parametric_patterns(),
        attenuation_regions=(
            AttenuationRegion(
                ("TL",),
                cfg.signal.true_direction.azimuth_deg,
                cfg.signal.true_direction.elevation_deg,
                1.0,
                6.020599913279624,
            ),
        ),
        coupling_paths=(CouplingPath("TL", "TR", -6.020599913279624, 90.0),),
    )
    cfg.array.antenna_model = PhoneAntennaConfig(
        model_label="scenario integration",
        model_source="parametric",
        true_state=true_state,
        assumed_state=copy.deepcopy(true_state),
    )
    model = PhoneAntennaModel.from_config(cfg)

    samples, truth = generate_scenario(cfg)

    expected_nominal = model.open_circuit_response(
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    expected_true = model.effective_manifold(
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    expected_assumed = model.effective_manifold(
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.signal.carrier_hz,
        estimated=True,
    )
    np.testing.assert_allclose(truth["desired_steering_nominal"], expected_nominal)
    np.testing.assert_allclose(truth["desired_steering_true"], expected_true)
    np.testing.assert_allclose(truth["desired_steering_assumed"], expected_assumed)
    np.testing.assert_allclose(
        truth["desired_component"],
        expected_true[:, None] * truth["desired_waveform"][None, :],
    )
    np.testing.assert_allclose(
        samples,
        truth["desired_component"] + sum(truth["interferer_components"]),
    )
    assert truth["antenna_model_source"] == "parametric"
    true_diagnostics = model.diagnostics(
        cfg.signal.true_direction,
        cfg.signal.true_polarization,
        cfg.signal.carrier_hz,
        estimated=False,
    )
    assumed_diagnostics = model.diagnostics(
        cfg.signal.assumed_direction,
        cfg.signal.assumed_polarization,
        cfg.signal.carrier_hz,
        estimated=True,
    )
    np.testing.assert_allclose(
        _complex_diagnostic(truth["desired_open_circuit_response_true"]),
        true_diagnostics["open_circuit_response"],
    )
    np.testing.assert_allclose(
        _complex_diagnostic(truth["desired_open_circuit_response_assumed"]),
        assumed_diagnostics["open_circuit_response"],
    )
    np.testing.assert_allclose(
        truth["desired_blockage_voltage_true"],
        true_diagnostics["blockage_voltage"],
    )
    np.testing.assert_allclose(
        _complex_diagnostic(truth["coupling_matrix_true"]),
        true_diagnostics["coupling_matrix"],
    )
    assert np.all(np.isfinite(truth["desired_polarization_match_loss_db_true"]))
    diagnostic_keys = (
        "antenna_model_source",
        "antenna_model_label",
        "desired_open_circuit_response_true",
        "desired_open_circuit_response_assumed",
        "interferer_open_circuit_responses",
        "desired_blockage_voltage_true",
        "desired_blockage_voltage_assumed",
        "interferer_blockage_voltages",
        "coupling_matrix_true",
        "coupling_matrix_assumed",
        "desired_polarization_match_loss_db_true",
        "desired_polarization_match_loss_db_assumed",
        "interferer_polarization_match_losses_db",
        "antenna_model_diagnostics",
    )
    strict_snapshot({key: truth[key] for key in diagnostic_keys})


def test_legacy_ula_steering_arrays_remain_exactly_compatible():
    cfg = legacy_ula_config()
    cfg.signal.duration_ms = 1

    _, truth = generate_scenario(cfg)

    np.testing.assert_allclose(
        truth["desired_steering_nominal"],
        [1.0 + 0.0j, 0.854851454758261 + 0.518872807437126j],
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        truth["jammer_steering_nominal"],
        [1.0 + 0.0j, -0.22909435549432253 - 0.9734042203938922j],
        rtol=0.0,
        atol=1e-15,
    )


def test_ideal_phone_steering_arrays_remain_exactly_compatible():
    cfg = mate70_default_config()
    cfg.signal.duration_ms = 1

    _, truth = generate_scenario(cfg)

    expected_desired = np.array(
        [
            0.9926283618653807 - 0.12119791755822688j,
            0.35613585245170737 - 0.9344341895492136j,
            0.35613585245170737 + 0.9344341895492136j,
            0.9926283618653807 + 0.12119791755822688j,
        ]
    )
    expected_jammer = np.array(
        [
            -0.9666236070492344 - 0.25620070705430753j,
            -0.07761828547735872 - 0.9969831501883847j,
            -0.07761828547735872 + 0.9969831501883847j,
            -0.9666236070492344 + 0.25620070705430753j,
        ]
    )
    for key in (
        "desired_steering_nominal",
        "desired_steering_true",
        "desired_steering_assumed",
    ):
        np.testing.assert_allclose(truth[key], expected_desired, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        truth["jammer_steering_nominal"], expected_jammer, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        truth["jammer_steering_effective"], expected_jammer, rtol=0.0, atol=1e-15
    )
