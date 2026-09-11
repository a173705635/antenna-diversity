import numpy as np

from config import TrackingConfig
from gnss import ca_code
from tracking import track


def test_tracked_nco_recovers_phase_not_just_prompt_residual():
    fs = 4.092e6
    n = np.arange(round(fs * .1))
    phase = .7
    x = ca_code(1)[np.floor(np.mod(n * 1.023e6 / fs, 1023)).astype(int)]
    x = x * np.exp(1j * (2*np.pi*1500*n/fs + phase))
    cfg = TrackingConfig()
    result = track(x, fs, 1, {'code_phase_samples': 0, 'doppler_hz': 1500.},
                   cfg, 1575.42e6, 1.023e6)
    predicted = 2*np.pi*1500*result['epoch_start_samples']/fs + phase
    error = np.angle(np.exp(1j*(result['carrier_nco_phase_rad']-predicted)))
    assert np.max(np.abs(error[-30:])) < .01
    assert np.max(np.abs(result['pll_error_rad'][-30:])) < .01


def test_phase_comparison_aligns_physical_time_within_nco_blocks():
    from audit_dual_tracking_phase import compare_phases
    fs = 10000.
    starts = np.arange(10, 101, 10)
    other = starts + 3
    rate = 1500.
    def history(times, phase):
        return {'epoch_start_samples': times,
                'carrier_nco_phase_rad': 2*np.pi*rate*times/fs + phase,
                'carrier_nco_frequency_hz': np.full(times.size, rate)}
    result = compare_phases(history(starts, .2), history(other, -.5), fs, 10)
    assert len(result['phase_difference_rad']) == 9
    assert np.allclose(result['phase_difference_rad'], -.7)


def test_two_chains_have_separate_noise_and_tracking_state(monkeypatch):
    import audit_dual_tracking_phase as audit
    rf_seeds, dfe_calls, acquisitions, tracking_results = [], [], [], []
    original_rf, original_dfe = audit.process_rf, audit.process_dfe
    original_acquire, original_track = audit.acquire, audit.track
    def rf(*args):
        rf_seeds.append(args[-1])
        assert args[0].shape[0] == 1
        return original_rf(*args)
    def dfe(*args):
        dfe_calls.append(args[0].shape)
        return original_dfe(*args)
    def acquire(*args):
        value = original_acquire(*args)
        acquisitions.append(value)
        return value
    def track(*args):
        value = original_track(*args)
        tracking_results.append(value)
        return value
    monkeypatch.setattr(audit, 'process_rf', rf)
    monkeypatch.setattr(audit, 'process_dfe', dfe)
    monkeypatch.setattr(audit, 'acquire', acquire)
    monkeypatch.setattr(audit, 'track', track)
    result = audit.run_case(True)
    assert result['status'] == 'evaluated_not_lock_certified'
    assert rf_seeds == [42, 43] and len(dfe_calls) == 2
    assert len(acquisitions) == 2 and acquisitions[0] is not acquisitions[1]
    assert not np.shares_memory(tracking_results[0]['carrier_nco_phase_rad'],
                                tracking_results[1]['carrier_nco_phase_rad'])
    assert result['all_epochs']['epochs'] > 30


def test_zero_dfe_control_is_not_reported_as_precise_tracking():
    from audit_dual_tracking_phase import run_case
    result = run_case(False)
    assert result['status'] == 'zero_dfe_output'
    assert result['adc_nonzero_samples'] > 0
    assert 'all_epochs' not in result


def test_requested_duration_also_extends_tracking_and_seeds(monkeypatch):
    import audit_dual_tracking_phase as audit
    seen = []
    class StopBeforeGenerating(Exception):
        pass
    def inspect(cfg):
        seen.append(cfg)
        raise StopBeforeGenerating
    monkeypatch.setattr(audit, 'generate_scenario', inspect)
    import pytest
    with pytest.raises(StopBeforeGenerating):
        audit.run_case(duration_ms=1000, seed_base=1000)
    assert seen[0].signal.duration_ms == 1000
    assert seen[0].tracking.max_epochs >= 1000
    assert seen[0].random_seed == 1000


def test_long_study_windows_use_absolute_time_and_flag_invalid_results():
    from audit_dual_tracking_stability import summarize_case
    times = np.arange(.005, 1., .001)
    expected = -.7
    phase = np.full(times.size, expected)
    phase[times < .2] += .5
    case = {'status': 'evaluated_not_lock_certified', 'time_s': times.tolist(),
            'phase_difference_rad': phase.tolist(), 'wavelength_m': .19,
            'ideal_geometry_phase_deg': np.degrees(expected), 'channels': []}
    result = summarize_case(case)
    assert result['post_200ms']['wrapped_error_rms_m'] < 1e-12
    assert result['all_epochs']['wrapped_error_rms_m'] > .001
    assert result['post_200ms']['epochs'] >= 799
    case['phase_difference_rad'][300] = float('nan')
    assert summarize_case(case)['status'] == 'nonfinite_tracking_output'
