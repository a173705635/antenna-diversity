import numpy as np
import pytest

from l5 import l5q_code, NH20, synchronize_nh


def test_q5_icd_initial_states():
    # IS-GPS-705J Table 3-Ia; rightmost bit is first out; XA starts all ones.
    states = ['1001011001100', '0100011110110', '1111000100011',
              '0011101101010', '0011110110010', '0101010101001',
              '1111110000001', '0110101101000']
    for prn, state in enumerate(states, 1):
        code = l5q_code(prn)
        assert code.shape == (10230,)
        np.testing.assert_array_equal(code[:13], [2*int(b)-1 for b in state[::-1]])
        assert set(code) == {-1, 1}
    with pytest.raises(ValueError):
        l5q_code(0)


def test_nh_search_unknown_phase_frequency_and_offset():
    t = np.arange(40)*.001
    for offset in range(20):
        prompts = NH20[(np.arange(40)+offset) % 20]*np.exp(1j*(2.2+2*np.pi*72.5*t))
        result = synchronize_nh(prompts)
        assert result['offset'] == offset
        assert abs(result['frequency_hz']-72.5) < .01
    with pytest.raises(ValueError):
        synchronize_nh(np.ones(40, dtype=complex))
    with pytest.raises(ValueError):
        synchronize_nh(np.zeros(40, dtype=complex))


def test_l5_rf_dfe_tracking_and_observation():
    from run_gps_spp import build_scene, satellite_setup
    from rf import process_rf
    from dfe import process_dfe
    from acquisition import acquire
    from tracking import track
    from positioning import code_pseudorange
    satellite = satellite_setup()[1]
    scene, cfg = build_scene([satellite], 100, signal_kind='L5Q')
    assert cfg.signal.fs_rf_hz == 33e6 and cfg.dfe.decimation == 1
    adc, _ = process_rf(scene, cfg.rf, cfg.reference_impedance_ohm, 4000)
    baseband, info = process_dfe(adc, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    acq = acquire(baseband[0], info['fs_out_hz'], 2, cfg.signal.code_rate_hz,
                  cfg.acquisition, signal_kind='L5Q')
    assert acq['success']
    assert abs(acq['doppler_hz']-cfg.signal.doppler_hz) <= 125
    hist = track(baseband[0], info['fs_out_hz'], 2, acq, cfg.tracking,
                 cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
    assert hist['success'], hist['status']
    assert hist['nh_sync']['metric'] > 1.5
    assert np.mean(hist['phase_lock_metric'][-30:]) > .95
    times = np.arange(60, 99)*.001
    obs = code_pseudorange(hist, times, 33e6, .001, satellite['coarse_range_m'],
                           code_rate_hz=10.23e6, code_length=10230)
    expected = satellite['range_at_zero_m']-satellite['doppler_hz']*(299792458/1575.42e6)*times
    assert np.sqrt(np.mean((obs-expected)**2)) < 10
