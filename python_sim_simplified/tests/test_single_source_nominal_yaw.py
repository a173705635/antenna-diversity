"""Fixed nominal-yaw perturbation and publication screen contracts."""
import json
import numpy as np
import pytest

from core.recovery_geometry import WAVELENGTH
from experiments import single_source_nominal_yaw as study


def test_positive_yaw_rotates_receiver_baseline_counterclockwise_only():
    record = dict(samples=np.ones((2, 4), complex),
                  nominal=np.ones((2, 1), complex),
                  satellites=np.array([[[3., 4., 5.]]]),
                  coarse=np.array([1.]), codes=np.array([[1.]]))
    direction = np.array([[1., 0., 0.]])
    antennas = np.array([[0., 0., 0.], [WAVELENGTH/4, 0., 0.]])
    shifted, physical = study.receiver_record(record, direction, antennas, 90.)
    np.testing.assert_allclose(shifted['nominal'], [[1.], [1.]], atol=1e-14)
    np.testing.assert_array_equal(physical, record['nominal'])
    for name in ('samples', 'satellites', 'coarse', 'codes'):
        assert shifted[name] is record[name]
    assert shifted is not record
    negative, _ = study.receiver_record(record, direction, antennas, -90.)
    np.testing.assert_allclose(negative['nominal'], shifted['nominal'], atol=1e-14)


def test_yaw_zero_reuses_exact_nominal_and_nonzero_has_opposite_phase_signs():
    antennas = np.array([[0., 0., 0.], [WAVELENGTH/4, 0., 0.]])
    direction = np.array([[0., 1., 0.]])
    record = {'nominal': np.array([[1.], [1j]])}
    zero, physical = study.receiver_record(record, direction, antennas, 0.)
    assert zero['nominal'] is record['nominal']
    assert physical is record['nominal']
    plus, _ = study.receiver_record(record, direction, antennas, 30.)
    minus, _ = study.receiver_record(record, direction, antennas, -30.)
    np.testing.assert_allclose(plus['nominal'][1, 0], np.exp(1j*np.pi/4))
    np.testing.assert_allclose(minus['nominal'][1, 0], np.exp(-1j*np.pi/4))


def test_schedule_has_six_new_rf_paths_and_three_old_baselines():
    schedule = study.path_schedule()
    assert len(schedule) == 9
    assert [yaw for _, yaw in schedule] == [0., -2., 2.]*3
    assert {scene for scene, _ in schedule} == {
        'h0_drop', 'weak', 'genuine_absent'}
    assert sum(yaw != 0 for _, yaw in schedule) == 6


def test_identical_weak_positions_pass_strict_non_degradation():
    time = np.array([.29, .30, .31, .50])
    position = np.column_stack((2*time, np.zeros(len(time))))
    result = study.score_pair('weak', time, position, position, 2., .2, .2)
    assert result['opportunities'] == 3
    assert result['common']['count'] == 3
    assert result['pass']


def test_absent_late_publication_fails_even_at_true_position():
    time = np.array([.30, .49, .50])
    baseline = np.full((3, 2), np.nan)
    candidate = np.array([[np.nan, np.nan], [np.nan, np.nan], [1., 0.]])
    result = study.score_pair('genuine_absent', time, baseline, candidate,
                              2., None, None)
    assert result['candidate_after_half_published'] == 1
    assert not result['pass']


def test_h0_worse_error_and_weak_worse_motion_fail():
    time = np.array([.29, .30, .31, .50])
    baseline = np.column_stack((2*time, np.zeros(len(time))))
    worse = baseline.copy()
    worse[2, 0] += .01
    assert not study.score_pair('h0_drop', time, baseline, worse,
                                2., .1, .1)['pass']
    assert not study.score_pair('weak', time, baseline, baseline,
                                2., .1, .2)['pass']


def test_physical_diagnostic_uses_true_response_and_fit_projector(tmp_path):
    seed = 2026100404
    rng = np.random.default_rng(seed)
    gain = np.exp(1j*np.deg2rad(10.*rng.normal(size=3)))
    gain *= 10**(rng.normal(size=3)/20)
    nominal = np.ones((3, 1), complex)
    np.savez_compressed(tmp_path/'tracking.npz', calibration=gain,
                        calibration_rate=np.zeros(3),
                        startup_relative_amplitude=np.ones(3))
    np.savez_compressed(tmp_path/'trace.npz', trace_nominal=nominal)
    np.savez_compressed(tmp_path/'fit.npz',
                        projector=np.diag([1., 0., 1.])[None])
    physical = study.physical_diagnostics({'nominal': nominal}, seed, tmp_path)
    np.testing.assert_allclose(physical['mismatch_before_scalar'], 0., atol=1e-14)
    np.testing.assert_allclose(physical['mismatch_after_scalar'], 0., atol=1e-14)
    np.testing.assert_allclose(physical['true_star_power_retained'], [[2./3.]])


def test_nonzero_calibration_rate_changes_each_block_physical_diagnostic(tmp_path):
    seed = 2026100404
    rng = np.random.default_rng(seed)
    gain = np.exp(1j*np.deg2rad(10.*rng.normal(size=3)))
    gain *= 10**(rng.normal(size=3)/20)
    rate = np.array([0., 20.*np.pi, 0.])
    nominal = np.ones((3, 1), complex)
    projector = np.array([[.5, .5, 0.], [.5, .5, 0.], [0., 0., 1.]])
    np.savez_compressed(tmp_path/'tracking.npz', calibration=gain,
                        calibration_rate=rate,
                        startup_relative_amplitude=np.ones(3))
    np.savez_compressed(tmp_path/'trace.npz', trace_nominal=nominal)
    np.savez_compressed(tmp_path/'fit.npz',
                        projector=np.repeat(projector[None], 2, axis=0))
    physical = study.physical_diagnostics({'nominal': nominal}, seed, tmp_path)
    midpoint = np.array([.01, .03])
    phase = rate[1]*midpoint
    expected_pre = np.sqrt(abs(np.exp(-1j*phase)-1.)**2/3.)
    expected_retained = (2.+np.cos(phase))/3.
    np.testing.assert_allclose(physical['block_midpoint_s'], midpoint)
    np.testing.assert_allclose(physical['mismatch_before_scalar'][:, 0],
                               expected_pre)
    np.testing.assert_allclose(physical['mismatch_after_scalar'][:, 0],
                               expected_pre*np.sqrt(2./3.))
    np.testing.assert_allclose(physical['true_star_power_retained'][:, 0],
                               expected_retained)


@pytest.mark.parametrize('saved_state', ('fresh', 'receipt', 'science', 'partial'))
def test_runner_reuses_completed_paths_and_preserves_partial_data(
        tmp_path, monkeypatch, saved_state):
    source = {'experiments/single_source_nominal_yaw.py': 'hash'}
    science_hashes = {name: 'hash' for name in study.SCIENCE}
    monkeypatch.setattr(study, 'OUTPUT', tmp_path/'new')
    monkeypatch.setattr(study, 'STAGE47', tmp_path/'old')
    monkeypatch.setattr(study, 'preflight', lambda: ({'cases': {
        str(study.SEED): {scene: {'input_sha256': {'samples.npy': 'hash'},
            'paths': {'gate3': {'output_sha256': science_hashes}}}
            for scene in study.SCENES}}}, source))
    monkeypatch.setattr(study, 'source_hashes', lambda: source)
    monkeypatch.setattr(study, 'source_snapshot', lambda *args: None)
    monkeypatch.setattr(study, 'digest', lambda *args: 'hash')
    monkeypatch.setattr(study, 'plot_scene', lambda *args: None)
    monkeypatch.setattr(study, 'physical_diagnostics', lambda *args: {
        'calibration': np.ones(3), 'calibration_rate': np.zeros(3),
        'startup_relative_amplitude': np.ones(3),
        'block_midpoint_s': np.array([.01]),
        'mismatch_before_scalar': np.zeros((1, 24)),
        'mismatch_after_scalar': np.zeros((1, 24)),
        'true_star_power_retained': np.ones((1, 24))})
    antennas = np.array(list(study.mate70_phase_centers().values()))
    nominal = study.steering(study.sky(study.SEED, 24), antennas, 0.)

    def input_record(folder, settings, prefix, geometry):
        record = {'nominal': nominal, 'samples': np.ones((4, 2)),
                  'satellites': np.ones((2, 24, 3))}
        identity = {'case': {'velocity_east': 2., 'attack_start': .3}}
        return record, identity, {'samples.npy': 'hash'}, prefix, geometry

    time = np.arange(.15, 20.-.01, .01)
    position = np.column_stack((2*time, np.zeros(len(time))))
    navigation = {mode: {'max_relative_motion_error_m': .1,
                         'initial_horizontal_error_m': .1}
                  for mode in ('short', 'original')}
    view = {'observation': time, 'available': time+.02,
            'position': {'short': position, 'original': position},
            'navigation': navigation}
    monkeypatch.setattr(study, 'load_input', input_record)
    monkeypatch.setattr(study, 'path_view', lambda *args: view)
    for scene in study.SCENES:
        folder = study.STAGE47/str(study.SEED)/scene/'gate3'
        folder.mkdir(parents=True)
        (folder/'path_receipt.json').write_text(json.dumps({
            'output_sha256': science_hashes}))

    calls = []

    def run_path(record, case, gate, folder):
        calls.append((folder.parent.name, folder.name, gate,
                      record['samples'], record['satellites']))
        folder.mkdir()
        return {'output_sha256': science_hashes,
                'navigation': navigation}, view

    monkeypatch.setattr(study, 'run_path', run_path)
    target = study.OUTPUT/study.SCENES[0]/'yaw_minus2'
    if saved_state != 'fresh':
        target.mkdir(parents=True)
    if saved_state == 'receipt':
        (target/'path_receipt.json').write_text(json.dumps({
            'input_sha256': {'samples.npy': 'hash'},
            'source_sha256': source, 'yaw_deg': -2.,
            'output_sha256': science_hashes, 'diagnostics_sha256': 'hash',
            'navigation': navigation}))
    recovered = []
    if saved_state == 'science':
        for name in study.SCIENCE:
            (target/name).write_bytes(b'saved science')

        def recover_path(folder, record, case, gate):
            recovered.append(folder)
            return {'output_sha256': science_hashes,
                    'navigation': navigation}, view

        monkeypatch.setattr(study, 'recover_path', recover_path)
    if saved_state == 'partial':
        (target/'tracking.npz').write_bytes(b'preserved partial data')
        with pytest.raises(AssertionError, match='Unreceipted RF path retained'):
            study.main()
        assert (target/'tracking.npz').read_bytes() == b'preserved partial data'
        assert calls == []
        return
    summary = study.main()
    expected = [
        (scene, yaw) for scene in study.SCENES
        for yaw in ('yaw_minus2', 'yaw_plus2')]
    if saved_state != 'fresh':
        expected.remove((study.SCENES[0], 'yaw_minus2'))
    assert len(calls) == len(expected)
    assert [(scene, yaw) for scene, yaw, _, _, _ in calls] == expected
    assert all(gate == 3. for _, _, gate, _, _ in calls)
    assert len(recovered) == int(saved_state == 'science')
    assert all(summary['cases'][scene]['paths']['yaw0']['status'] ==
               'reused_stage47_gate3' for scene in study.SCENES)
