import numpy as np
import warm_source as ws


def test_shared_observation_and_observed_initialization():
    case = ws.make_case(111001, 'clean', False)
    assert case['warm'].shape == (12, 4, 8)
    for m in (2, 3, 4):
        state = ws.initialize(case['warm'][:, :m])
        assert state['complete']
        assert state['time'] < case['attack_time']
        assert np.allclose(state['template'], case['warm'][:8, :m].mean(axis=0))


def test_truth_not_needed_by_detector():
    case = ws.make_case(111013, 'clean', True)
    warm, current = case['warm'][:, :3], case['current'][:, :3]
    assert np.isfinite(ws.score(warm, current))


def test_common_phase_has_no_projective_change():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 8)) + 1j*rng.normal(size=(4, 8))
    assert np.allclose(abs(ws.unit(a).conj().T @ ws.unit(a)),
                       abs(ws.unit(a*1j).conj().T @ ws.unit(a*1j)))


def test_frozen_score_invariant_to_satellite_carrier_reacquisition():
    case = ws.make_case(111007, 'clean', True)
    warm, current = case['warm'], case['current']
    carrier = np.exp(1j*np.linspace(-2,2,8))
    assert np.isclose(ws.score(warm,current),ws.score(warm,current*carrier))


def test_low_signal_initialization_is_not_silently_accepted():
    for seed in (112001,112013,112037):
        case = ws.make_case(seed,'low_signal',False)
        assert not ws.initialize(case['warm'])['complete']


def test_frozen_replay_exact():
    from warm_source_audit import replay
    assert replay()
