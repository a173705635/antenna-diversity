"""Independent fit and observation-time hard screen for the fixed gate pair."""
import numpy as np
import pytest

from core.single_source import residual_trigger
from core.recovery_geometry import FS
from experiments import single_source_projection_threshold as study


def test_independent_fit_switches_projector_and_weight_at_two():
    model = np.array([[1., 1.], [0., 0.], [1., 1.]], complex)
    prompts = np.array([[1., 2., 1., 2.],
                        [5., 5., 5., 5.],
                        [1., 2., 1., 2.]], complex)
    fitted = study.independent_fit(prompts, model, np.array([0, 1]),
                                   noise=.01, threshold=2.)
    projector = fitted['projector']
    np.testing.assert_allclose(projector, projector.conj().T, atol=1e-12)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
    assert fitted['dimension'] == 2
    response = np.sum(model.conj()*(projector@model), axis=0).real
    np.testing.assert_allclose(fitted['weight'],
                               projector@model/(response+.01)[None, :], atol=1e-12)
    np.testing.assert_allclose(fitted['gain'], response/3., atol=1e-12)
    inactive = study.independent_fit(prompts, model, np.array([0, 1]),
                                     noise=.01, threshold=1e12)
    assert inactive['dimension'] == 3
    np.testing.assert_array_equal(inactive['projector'], np.eye(3))
    np.testing.assert_array_equal(inactive['weight'], model/3.)
    np.testing.assert_array_equal(inactive['gain'], np.ones(2))
    receiver_projector, receiver_raw = residual_trigger(
        prompts, np.repeat(model, 2, axis=1), .01)
    np.testing.assert_allclose(fitted['fitted_projector'],
                               receiver_projector, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted['raw'], receiver_raw, rtol=0, atol=1e-12)


def test_independent_fit_matches_receiver_on_complex_grouped_prompts():
    rng = np.random.default_rng(207)
    model = np.exp(1j*rng.uniform(-2., 2., (3, 24)))
    desired = np.repeat(model, 4, axis=1)
    nuisance = np.array([1., -1j, .4+1j])[:, None]
    prompts = (desired*(1.+.2j) + nuisance*(.7-.4j) +
               .1*(rng.normal(size=desired.shape)+
                   1j*rng.normal(size=desired.shape)))
    fitted = study.independent_fit(prompts, model, np.arange(24),
                                   noise=.03, threshold=2.)
    receiver_projector, receiver_raw = residual_trigger(prompts, desired, .03)
    np.testing.assert_allclose(fitted['fitted_projector'],
                               receiver_projector, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fitted['raw'], receiver_raw, rtol=0, atol=1e-10)


def test_iq_audit_checks_projector_gain_and_actual_stream(tmp_path, monkeypatch):
    rng = np.random.default_rng(91)
    nominal = np.exp(1j*rng.uniform(-2., 2., (3, 4)))
    samples = (rng.normal(size=(3, round(.12*FS)))+
               1j*rng.normal(size=(3, round(.12*FS))))
    record = dict(samples=samples, nominal=nominal,
                  codes=np.zeros((4, 1)))
    prns = np.arange(4)
    templates = np.zeros((4, 4))
    monkeypatch.setattr(study, 'extract_startup',
                        lambda record: (prns, templates, np.ones(3), np.zeros(3)))
    monkeypatch.setattr(study, 'startup_prompts',
                        lambda *args: np.zeros((3, 4, 100), complex))
    monkeypatch.setattr(study, 'estimate_amplitude',
                        lambda *args: (np.ones(3), None))
    monkeypatch.setattr(study, 'local_signal',
                        lambda *args: np.ones(len(args[-1]), complex))
    last = samples[:, round(.10*FS):]
    grouped = (last.reshape(3, 4, 5, 2046).mean(axis=3).sum(axis=2)/
               np.sqrt(5))
    grouped = np.tile(grouped, (1, 4))
    fitted = study.independent_fit(grouped, nominal, prns,
                                   float(np.mean(abs(last)**2)/2046), 2.)
    raw = np.r_[np.full(5, np.nan), fitted['raw']]
    projectors = np.tile(np.eye(3), (6, 1, 1)).astype(complex)
    projectors[5] = fitted['projector']
    dimensions = np.r_[np.full(5, 3), fitted['dimension']]
    trace = dict(trace_nominal=nominal, trace_raw_trigger=raw,
                 trace_projectors=projectors, trace_dimensions=dimensions)
    tracking = dict(startup_prns=prns, calibration=np.ones(3),
                    calibration_rate=np.zeros(3),
                    startup_relative_amplitude=np.ones(3),
                    time=np.array([.10]), gain=fitted['gain'][None, :],
                    ratio=np.array([fitted['raw']]))
    path = tmp_path/'beam_stream.dat'
    stream = np.memmap(path, dtype=np.complex128, mode='w+',
                       shape=(4, samples.shape[1]))
    begin = round(.10*FS)
    stream[:, :begin] = (nominal/3.).conj().T@samples[:, :begin]
    stream[:, begin:] = fitted['weight'].conj().T@last
    stream.flush()
    del stream
    study.audit_blocks(record, tracking, trace, path, 2.)
    altered = dict(trace, trace_projectors=projectors.copy())
    altered['trace_projectors'][5, 0, 0] += .1
    with pytest.raises(AssertionError):
        study.audit_blocks(record, tracking, altered, path, 2.)


def test_screen_counts_edge_gaps_and_observation_boundary():
    time = np.array([.29, .30, .31, .32, .33, .34])
    baseline = np.array([[np.nan, np.nan], [np.nan, np.nan], [1., 0.],
                         [1., 0.], [np.nan, np.nan], [1., 0.]])
    candidate = np.array([[np.nan, np.nan], [1., 0.], [1., 0.],
                          [1., 0.], [1., 0.], [np.nan, np.nan]])
    result = study.pair_score('h0_constant', time, baseline, candidate,
                              velocity_east=0., baseline_motion=2.,
                              candidate_motion=1.)
    assert result['opportunities'] == 5
    assert result['baseline']['published'] == 3
    assert result['candidate']['published'] == 4
    assert result['baseline']['longest_gap_epochs'] == 1
    assert result['candidate']['longest_gap_epochs'] == 1
    assert result['common']['count'] == 2
    assert result['lost'] == 1 and result['new'] == 2
    assert not result['pass']


def test_absent_screen_rejects_new_publication_at_half_second():
    time = np.array([.29, .30, .49, .50, .51])
    baseline = np.full((5, 2), np.nan)
    candidate = baseline.copy()
    candidate[2:4] = [0., 0.]
    result = study.pair_score('genuine_absent', time, baseline, candidate,
                              velocity_east=0., baseline_motion=None,
                              candidate_motion=None)
    assert result['new_after_half_second'] == 1
    assert not result['pass']


def test_fixed_screen_plan_has_four_conditions_and_two_gates():
    assert study.SEED == 2026100401
    assert study.GATES == (3., 2.)
    assert study.SCENES == ('h0_constant', 'h0_drop', 'weak', 'genuine_absent')
    for scene in study.SCENES:
        plan = study.scene_plan(scene)
        assert plan['duration_s'] == 2.
        assert plan['attack_start_s'] == .3
        assert plan['startup_cn0_db_hz'] == 45.
        assert plan['post_0p3s_cn0_db_hz'] == (45. if scene == 'h0_constant' else 30.)
        assert plan['spoof_db'] == -5.
        assert plan['angle_deg'] == 5.
        assert plan['fake_east_m'] == 0.


def test_no_published_baseline_is_no_go_without_metric_comparisons():
    time = np.array([.29, .30, .31, .32])
    none = np.full((4, 2), np.nan)
    result = study.pair_score('weak', time, none, none, 0., None, None)
    assert result['baseline']['published'] == 0
    assert result['baseline']['longest_gap_epochs'] == 3
    assert result['candidate']['longest_gap_epochs'] == 3
    assert not result['pass']
    added = none.copy()
    added[-1] = [0., 0.]
    result = study.pair_score('weak', time, none, added, 0., None, 0.)
    assert result['new'] == 1
    assert not result['pass']


def test_unchanged_h0_passes_but_unchanged_weak_has_no_benefit():
    time = np.array([.29, .30, .31, .32])
    position = np.array([[np.nan, np.nan], [0., 0.], [0., 0.], [0., 0.]])
    h0 = study.pair_score('h0_constant', time, position, position,
                           0., None, None)
    weak = study.pair_score('weak', time, position, position, 0., 1., 1.)
    assert h0['pass']
    assert not weak['pass']
