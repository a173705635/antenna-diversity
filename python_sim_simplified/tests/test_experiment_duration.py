"""Short recordings must not create out-of-range observations or zero-valued missing results."""
import json
import sys
import numpy as np
import pytest
from experiments.multipath import sdcp_baseline as baseline, sdcp_multipath as multipath
from experiments.multipath.sdcp_support import save_json


@pytest.mark.parametrize('duration', [300, 700, 1500])
def test_baseline_full_analysis_and_verification_with_short_histories(tmp_path, monkeypatch, duration):
    monkeypatch.setattr('core.config.ENABLE_MULTIPATH_RESEARCH', True)
    # Synthetic NCO fixtures exercise orchestration, saving, statistics, plots and
    # independent verification. They are not RF performance simulations.
    # Other tests import positioning; isolate the module registry for this no-positioning run.
    for name in ('core.positioning', 'core.gps_simulation'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(baseline, 'DURATION_MS', duration)
    monkeypatch.setattr(baseline, 'reset_output', lambda name: tmp_path)
    monkeypatch.setattr(baseline, 'generate_scenario',
                        lambda cfg: (np.ones((2, 32), complex), {}))

    def receive(signal, cfg, seed, out, name, hardware=False):
        fs = cfg.signal.fs_rf_hz/cfg.dfe.decimation
        starts = np.arange(duration-5)*round(fs*.001)
        t = starts/fs
        phase = np.zeros(len(t))
        if name.endswith('_B'):
            case = name.split('_', 1)[1][:-2]
            phase = np.deg2rad(baseline.phase_degrees(case, t))
        h = dict(epoch_start_samples=starts, carrier_nco_phase_rad=phase,
                 carrier_nco_frequency_hz=np.zeros(len(t)),
                 phase_lock_metric=np.ones(len(t)), pll_error_rad=np.zeros(len(t)))
        np.savez_compressed(out/(name+'.npz'), **h)
        save_json(out/(name+'.json'), dict(acquisition=dict(success=True), tracking_status='tracking'))
        return h

    monkeypatch.setattr(baseline, 'receive', receive)
    baseline.main()
    summary = json.loads((tmp_path/'summary.json').read_text())
    gap = next(r for r in summary['rows'] if r['case']=='gap')
    if duration == 700:
        assert gap['regions']['event']['truth_error']['mean_mm'] is None
        assert gap['regions']['event']['truth_error']['epochs'] == 0
        assert gap['recovery_status'].startswith('not_evaluated')
    if duration == 300:
        ramp = next(r for r in summary['rows'] if r['case']=='ramp')
        assert ramp['fit_status'].startswith('not_evaluated')
        assert 'ramp_fit' not in ramp
    checks = json.loads((tmp_path/'verification.json').read_text())
    assert checks['records_reconstructed'] == 24
    with np.load(tmp_path/'aligned.npz') as arrays:
        assert arrays['time_s'][-1] < duration/1000


def test_default_baseline_timing_is_unchanged():
    times, intervals = baseline.analysis_timing(1500)
    np.testing.assert_array_equal(times, np.arange(10,1491)*.001)
    assert intervals['tail'] == [1.1, 1.5]


def test_multipath_incomplete_windows_have_no_scores_or_alarms():
    for size in (0, 1, 191, 199):
        ends, score = multipath.scores(np.zeros(size))
        assert len(ends) == len(score) == 0
    for size in (0, 1, 2):
        assert not multipath.confirmed(np.ones(size), .5).any()
    ends, score = multipath.scores(np.full(491, .095))
    np.testing.assert_allclose(score, .095, atol=1e-14)
    assert len(ends) == len(score) == 30


@pytest.mark.parametrize('module,setting,value', [
    ('multipath.sdcp_multipath','DURATION_MS',390), ('multipath.sdcp_baseline','DURATION_MS',10),
    ('gps_spp','DURATION_MS',200), ('gps_l1_l5','DURATION_MS',200),
    ('recorded_iq','DURATION_S',.7), ('spoofing.single_source','DURATION_S',.1),
])
def test_unsupported_duration_is_explained_before_output_reset(monkeypatch, module, setting, value):
    import importlib
    if module.startswith('multipath.'):
        monkeypatch.setattr('core.config.ENABLE_MULTIPATH_RESEARCH', True)
    experiment = importlib.import_module('experiments.'+module)
    monkeypatch.setattr(experiment, setting, value)
    def no_reset(*args):
        pytest.fail('Existing results must not be removed for an unevaluable duration')
    monkeypatch.setattr(experiment, 'reset_output', no_reset)
    with pytest.raises(ValueError, match='needs|need|evaluated'):
        experiment.main()


def test_recorded_positioning_without_epochs_is_not_evaluated():
    from core.recorded_receiver import position_channels
    result = position_channels({}, 6.6e6, [])
    assert result['status'] == 'not_evaluated_no_observation_times'
    assert result['epochs'] == []


def test_single_source_no_post_attack_data_is_not_zero_outage(monkeypatch):
    from experiments.spoofing import single_source
    z = dict(available_time=np.array([.2,.21]))
    nav = dict(position=np.zeros((2,3)), initialization_time=None)
    monkeypatch.setattr(single_source, 'frontend', lambda *args: z)
    monkeypatch.setattr(single_source, 'navigate', lambda *args: nav)
    _, _, score = single_source.evaluate({}, [0,1], dict(attack_start=.3, velocity_east=0.))
    assert score['assessment_status'].startswith('not_evaluated')
    assert score['initialization_status'] == 'not_initialized_in_record'
    assert score['longest_unavailable_s'] is None


@pytest.mark.parametrize('duration', [430, 700])
def test_multipath_full_short_record_analysis(tmp_path, monkeypatch, duration):
    monkeypatch.setattr('core.config.ENABLE_MULTIPATH_RESEARCH', True)
    for name in ('core.positioning', 'core.gps_simulation'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(multipath, 'DURATION_MS', duration)
    monkeypatch.setattr(multipath, 'reset_output', lambda name: tmp_path)
    monkeypatch.setattr(multipath, 'generate_scenario',
                        lambda cfg: (np.ones((2,32),complex), {'desired_waveform':np.ones(32,complex)}))
    from types import SimpleNamespace
    monkeypatch.setattr(multipath, '_multipath_scene',
                        lambda *args: SimpleNamespace(truth={'component':np.zeros((2,32),complex)}))

    def receive(signal, cfg, seed, out, name, hardware=False):
        fs=cfg.signal.fs_rf_hz/cfg.dfe.decimation
        starts=np.arange(duration-5)*round(fs*.001)
        h=dict(epoch_start_samples=starts,carrier_nco_phase_rad=np.zeros(len(starts)),
               carrier_nco_frequency_hz=np.zeros(len(starts)),prompt=np.ones(len(starts),complex),
               phase_lock_metric=np.ones(len(starts)),pll_error_rad=np.zeros(len(starts)))
        np.savez_compressed(out/(name+'.npz'),**h)
        return h

    monkeypatch.setattr(multipath,'receive',receive)
    multipath.main()
    rows=json.loads((tmp_path/'verified_results.json').read_text())
    late=next(r for r in rows if r['case']=='mp_t600ms_p-3dB_d0p3chip_phi90_az110_el15')
    assert late['full_post_event_window'] is False
    if duration==430:
        assert late['detected'] is None
        assert late['assessment_status'].startswith('not_evaluated')
    else:
        assert late['assessment_status']=='evaluated'
    assert json.loads((tmp_path/'verification.json').read_text())['records_verified']==39
