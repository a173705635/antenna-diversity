import numpy as np

from gnss import _desired_waveform
from presets import default_config


def test_continuous_source_responds_to_subsample_delay():
    cfg = default_config()
    cfg.signal.code_bandwidth_hz = 1.8e6
    cfg.signal.doppler_hz = 0
    n = np.arange(16368)
    cfg.signal.code_delay_chips = 137.251
    first, _, _ = _desired_waveform(cfg, n, n / cfg.signal.fs_rf_hz)
    cfg.signal.code_delay_chips = 137.26
    second, _, _ = _desired_waveform(cfg, n, n / cfg.signal.fs_rf_hz)
    assert np.linalg.norm(first-second)/np.linalg.norm(first) > 0.005
    expected_power = 10 ** ((cfg.signal.power_dbm - 30)/10) * cfg.reference_impedance_ohm
    # Allow the dense-table linear interpolation's small power loss (0.05%).
    assert abs(np.mean(abs(first)**2)/expected_power - 1) < 5e-4


def test_oracle_lcmv_restores_los_reference_for_one_echo():
    from audit_multipath_oracle import run_case

    baseline = run_case()["branches"][-1]
    echo = run_case(echo=True, phase=np.pi)["branches"][-1]
    assert echo["oracle_uses_truth"]
    assert echo["echo_gain"][0] < 1e-12
    assert abs(echo["los_gain"] - 1) < 1e-12
    np.testing.assert_allclose(echo["code_phase_error_m"],
                               baseline["code_phase_error_m"], atol=1e-8)
    assert echo["white_noise_gain"] / .25 < 1.02
