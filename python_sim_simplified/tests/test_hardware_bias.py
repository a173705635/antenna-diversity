from dataclasses import replace

import numpy as np
import pytest

from antenna import antenna_response, build_antenna_model
from audit_dual_tracking_phase import compare_phases
from config import RFConfig, RFStage
from presets import default_config
from rf import process_rf


def test_antenna_residual_is_separate_from_assumed_response():
    cfg = default_config()
    state = cfg.phone.antenna_model.true_state
    state.amplitude_error_db = {'TR': -1.0}
    state.phase_error_deg = {'TR': 5.0}
    model = build_antenna_model(cfg.phone, cfg.signal.carrier_hz)
    actual, info = antenna_response(model, cfg.signal.true_direction,
                                    cfg.signal.true_polarization)
    assumed, _ = antenna_response(model, cfg.signal.true_direction,
                                  cfg.signal.true_polarization, estimated=True)
    expected = np.ones(len(actual), dtype=complex)
    expected[1] = 10**(-1/20) * np.exp(1j*np.deg2rad(5))
    np.testing.assert_allclose(actual / assumed, expected)
    np.testing.assert_allclose(info['element_error_response'], expected)


def test_rf_error_rotates_and_scales_signal_and_noise_before_adc():
    cfg = RFConfig(stages=(RFStage('unity', 0., 0.),), adc_bits=24,
                   temperature_k=1e12, enable_soft_limiting=False)
    x = np.full((2, 2048), .003+.005j)
    nominal, info = process_rf(x, cfg, 50., 12)
    biased, biased_info = process_rf(x, replace(cfg,
        amplitude_error_db=(0., 20*np.log10(2)), phase_error_deg=(0., 90.)), 50., 12)
    np.testing.assert_array_equal(biased[0], nominal[0])
    # One-code tolerance for separately rounded inputs; also verifies noise rotation.
    np.testing.assert_allclose(biased[1], 2j*nominal[1], atol=1.5, rtol=0)
    assert biased_info['input_noise_power_w'] == info['input_noise_power_w']
    with pytest.raises(ValueError):
        process_rf(x, replace(cfg, phase_error_deg=(0., 1., 2.)), 50., 12)
    with pytest.raises(ValueError):
        process_rf(x, replace(cfg, amplitude_error_db=float('nan')), 50., 12)


def test_raw_comparison_keeps_full_cycle_jump():
    starts = np.arange(8)*10
    first = {'epoch_start_samples': starts, 'carrier_nco_phase_rad': np.zeros(8),
             'carrier_nco_frequency_hz': np.zeros(8)}
    phase = np.r_[np.full(4, .3), np.full(4, .3+6*np.pi)]
    second = dict(first, carrier_nco_phase_rad=phase)
    result = compare_phases(first, second, 10000., 10)
    np.testing.assert_array_equal(result['raw_phase_difference_rad'], phase)
