import numpy as np
from navigation_research import candidate_modes, navigate


def run(position, velocity, spatial, *, estimate=False, limit=np.inf, admitted=True):
    count = len(position)
    return navigate(dict(time_s=np.arange(count+1)*.1,
                         position_m=np.zeros((count+1, 3)), velocity_mps=np.zeros((count+1, 3))),
                    position, velocity, spatial, estimate_acceleration=estimate,
                    velocity_limit=limit, position_limit=np.inf, mean_velocity_limit=np.inf,
                    admitted=admitted)


def test_pending_release_and_confirmation_are_distinct():
    scores = [.5, 2, 2, .5, .5, .5, 2, 2]
    np.testing.assert_array_equal(candidate_modes(scores, 1, 3, 2), [-1, -1, 0, -1, -1, 1, -1, 0])
    np.testing.assert_array_equal(candidate_modes(scores, 1, 3, 0), [0, 0, 0, 0, 0, 1, 0, 0])


def test_innovation_is_checked_before_contaminated_update_and_latches():
    v = np.zeros((20, 3)); v[5, 0] = 5
    result = run(np.zeros_like(v), v, np.zeros(20), limit=1.)
    assert result['gnss_accepted'][5]
    assert not result['gnss_accepted'][6:].any()
    np.testing.assert_array_equal(result['state'], 0)
    prefix = run(np.zeros_like(v[:10]), v[:10], np.zeros(10), limit=1.)
    np.testing.assert_array_equal(prefix['state'], result['state'][:11])


def test_unadmitted_and_confirmed_runs_never_resume_updates():
    zeros = np.zeros((10, 3)); spatial = np.zeros(10); spatial[3] = 1
    assert not run(zeros, zeros, spatial, admitted=False)['gnss_accepted'].any()
    assert not run(zeros, zeros, spatial)['gnss_accepted'][4:].any()


def test_effective_acceleration_is_learned_from_clean_velocity():
    t = np.arange(601)*.1; p = np.column_stack((.02*t*t/2, t*0, t*0))
    v = np.column_stack((.02*t, t*0, t*0)); zero = np.zeros((600, 3))
    result = navigate(dict(time_s=t, position_m=p, velocity_mps=v), zero, zero, np.zeros(600),
                      estimate_acceleration=True, velocity_limit=np.inf, position_limit=np.inf,
                      mean_velocity_limit=np.inf)
    assert abs(result['state'][-1, 6]-.02) < .001
    assert np.linalg.norm(result['state'][-1, :3]) < .05


def test_mean_monitor_waits_for_complete_window_then_blocks_current_update():
    n = 20; zero = np.zeros((n+1, 3)); velocity = np.zeros((n, 3)); velocity[:, 0] = .2
    result = navigate(dict(time_s=np.arange(n+1)*.1, position_m=zero, velocity_mps=zero),
                      zero[1:], velocity, np.full(n, -1), estimate_acceleration=False,
                      velocity_limit=np.inf, position_limit=np.inf, mean_velocity_limit=.15)
    assert (result['mode'][:10] == -1).all()
    assert (result['mode'][10:] == 1).all()
    assert not result['gnss_accepted'].any()
