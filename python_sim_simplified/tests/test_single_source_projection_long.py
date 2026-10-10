"""Fixed long-batch plan and preregistered relative-plus-absolute screen."""
import numpy as np

from experiments import single_source_projection_long as study


def positions(east):
    return np.column_stack((east, np.zeros(len(east))))


def test_h0_identical_passes_but_shared_twenty_metres_fails_absolute():
    time = np.array([.29, .30, .31, .50])
    good = positions(np.array([np.nan, 0., 0., 0.]))
    same = study.long_screen('h0_constant', time, good, good, 0., None, None)
    assert same['relative']['pass'] and same['pass']
    wrong = positions(np.array([np.nan, 20., 0., 0.]))
    shared = study.long_screen('h0_constant', time, wrong, wrong,
                                0., None, None)
    assert shared['relative']['pass']
    assert shared['candidate_ge20'] == 1
    assert not shared['pass']


def test_weak_identical_has_no_benefit():
    time = np.array([.29, .30, .31, .50])
    good = positions(np.array([np.nan, 0., 0., 0.]))
    same = study.long_screen('weak', time, good, good, 0., 1., 1.)
    assert not same['relative']['pass']
    assert same['candidate_ge20'] == 0
    assert not same['pass']


def test_absent_late_correct_position_is_still_a_failure():
    time = np.array([.29, .30, .31, .49, .50, .51])
    both = positions(np.array([np.nan, 0., 0., np.nan, 0., np.nan]))
    result = study.long_screen('genuine_absent', time, both, both,
                                0., None, None)
    assert result['relative']['pass']
    assert result['candidate_after_half_published'] == 1
    assert result['candidate_ge20'] == 0
    assert not result['pass']


def test_absent_early_twenty_metres_is_rejected():
    time = np.array([.29, .30, .31, .49, .50])
    both = positions(np.array([np.nan, 20., 0., np.nan, np.nan]))
    result = study.long_screen('genuine_absent', time, both, both,
                                0., None, None)
    assert result['relative']['pass']
    assert result['candidate_ge20'] == 1
    assert not result['pass']


def test_three_new_seeds_share_fixed_twenty_second_geometry():
    assert study.SEEDS == (2026100402, 2026100403, 2026100404)
    assert study.GATES == (3., 2.)
    assert study.SCENES == ('h0_constant', 'h0_drop', 'weak', 'genuine_absent')
    for seed in study.SEEDS:
        for scene in study.SCENES:
            plan = study.long_plan(seed, scene)
            assert plan['seed'] == seed
            assert plan['duration_s'] == 20.
            assert plan['satellites'] == 24
            assert plan['angle_deg'] == 5.
            assert plan['attack_start_s'] == .3
            assert plan['fake_east_m'] == 0.
            assert plan['fake_rate_mps'] == 3.
            assert plan['post_0p3s_cn0_db_hz'] == (
                45. if scene == 'h0_constant' else 30.)


def test_all_twenty_four_paths_run_after_no_go_and_prefix_resets(tmp_path,
                                                                 monkeypatch):
    calls = []
    prefixes = []
    monkeypatch.setattr(study, 'OUTPUT', tmp_path/'long')
    monkeypatch.setattr(study, 'preflight',
                        lambda: ({}, {'core/single_source.py': 'frozen'}))
    monkeypatch.setattr(study, 'source_snapshot', lambda *args: None)
    monkeypatch.setattr(study, 'source_hashes',
                        lambda: {'core/single_source.py': 'frozen'})
    monkeypatch.setattr(study, 'digest', lambda *args: 'hash')
    monkeypatch.setattr(study, 'plot_seed', lambda *args: None)

    def generated(plan, scene, fake_rate):
        record = dict(seed=plan['seed'], scene=scene)
        identity = dict(case={'velocity_east': 2., 'attack_start': .3})
        return record, identity

    def saved(folder, record, identity, first_prefix, first_geometry):
        prefixes.append((record['seed'], first_prefix, first_geometry))
        return {'samples.npy': 'hash'}, record['seed'], record['seed']

    def received(record, case, threshold, folder):
        calls.append((record['seed'], folder.parent.name, threshold, record))
        time = np.arange(.15, 20.-.01, .01)
        position = positions(2.*time)
        scores = dict(max_relative_motion_error_m=.1,
                      initial_horizontal_error_m=.1)
        view = dict(observation=time, available=time+.02,
                    position={'short': position, 'original': position},
                    support={'short': np.zeros(len(time)),
                             'original': np.zeros(len(time))},
                    navigation={'short': scores, 'original': scores})
        return {'output_sha256': {'tracking.npz': 'hash'}}, view

    monkeypatch.setattr(study, 'generate_new_record', generated)
    monkeypatch.setattr(study, 'save_new_input', saved)
    monkeypatch.setattr(study, 'run_path', received)
    summary = study.main()
    assert len(calls) == 24
    assert [threshold for _, _, threshold, _ in calls] == [3., 2.]*12
    for offset in range(0, 24, 2):
        assert calls[offset][3] is calls[offset+1][3]
    assert len(prefixes) == 12
    for offset in (0, 4, 8):
        assert prefixes[offset][1:] == (None, None)
        assert all(prefixes[index][1:] == (prefixes[offset][0],)*2
                   for index in range(offset+1, offset+4))
    assert not summary['screen_pass']
