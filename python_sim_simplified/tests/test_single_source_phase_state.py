"""Unpublished phase state carries only valid output-gated carrier motion."""
import numpy as np

from core.spoofing import single_source
from test_single_source_phase_leave_one_out import phase_record


def observations(n=30):
    geometry, _, _, _ = phase_record(stars=8)
    time = np.arange(n) * .01
    return dict(time=time, available_time=time + .02,
                satellites=np.broadcast_to(geometry['satellites'][0],
                                           (n, 8, 3)).copy(),
                carrier=np.zeros((n, 8)), arc=np.zeros((n, 8), int),
                ranges=np.broadcast_to(np.arange(1, n + 1)[:, None],
                                       (n, 8)).astype(float).copy(),
                lock=np.ones((n, 8)), cn0=np.full((n, 8), 30.),
                gain=np.ones((n, 8)), support=np.full((n, 8), 4.))


def fixed_code_geometry(monkeypatch, consistent_after=27):
    monkeypatch.setattr(single_source, 'smooth_history',
                        lambda z, valid, **kwargs: (z['ranges'].copy(),
                                                     np.zeros_like(z['ranges'])))
    monkeypatch.setattr(single_source, 'position_solution',
                        lambda *args: (np.zeros(3), 1., 0.))
    monkeypatch.setattr(single_source, 'receiver_reference',
                        lambda: (np.zeros(3), np.eye(3)))

    def solve(satellites, ranges, **kwargs):
        epoch = int(ranges[0]) - 1
        east = 40. if len(satellites) == 7 and 3 <= epoch < consistent_after else 2.
        return dict(position_m=np.array([east, 0., 0.]))

    monkeypatch.setattr(single_source, 'solve_position', solve)


def test_four_output_gated_epochs_keep_short_phase_state_and_reopen_without_recovery(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z = observations()
    off = single_source.navigate(z, diagnostics=True, code_blend=0.,
                                 phase_leave_one_out=True)
    on = single_source.navigate(z, diagnostics=True, code_blend=0.,
                                phase_leave_one_out=True,
                                phase_state_continuity=True)
    assert on['diagnostics']['reason'][23:27].tolist() == ['output_gate'] * 4
    assert np.isnan(on['position'][23:27]).all()
    assert np.isfinite(on['diagnostics']['phase_state_position_m'][23:27]).all()
    assert on['diagnostics']['phase_state_unpublished_run_epochs'][23:27].tolist() == [1, 2, 3, 4]
    assert on['diagnostics']['phase_state_short_anchor_index'][26] == 25
    assert not on['diagnostics']['phase_state_short_anchor_published'][26]
    assert on['diagnostics']['phase_ok'][26]
    assert not off['diagnostics']['phase_ok'][26]
    assert np.isfinite(on['position'][27]).all()
    assert not on['diagnostics']['isolated'][27]
    assert np.isnan(off['position'][27]).all()


def test_plain_phase_bridge_reports_its_real_internal_anchor_without_enabling_loo(monkeypatch):
    fixed_code_geometry(monkeypatch)
    monkeypatch.setattr(single_source, 'phase_prediction_leave_one_out',
                        lambda *args: (_ for _ in ()).throw(AssertionError('LOO used')))
    nav = single_source.navigate(observations(), diagnostics=True,
                                 code_blend=0., phase_state_continuity=True)
    assert nav['diagnostics']['phase_ok'][26]
    assert nav['diagnostics']['phase_state_short_anchor_index'][26] == 25
    assert not nav['diagnostics']['phase_state_short_anchor_published'][26]
    assert np.isfinite(nav['position'][27]).all()


def test_initialization_isolation_and_fresh_recovery_never_store_a_provisional_phase(monkeypatch):
    fixed_code_geometry(monkeypatch, consistent_after=0)
    monkeypatch.setattr(single_source, 'position_solution',
                        lambda satellites, ranges, valid:
                        (np.zeros(3), 2000. if int(ranges[0]) == 4 else 1., 0.))
    monkeypatch.setattr(single_source, 'phase_prediction',
                        lambda *args, **kwargs: (np.array([10., 0., 0.]),
                                                  max(0, args[3]-1))
                        if kwargs.get('return_anchor') else np.array([10., 0., 0.]))
    nav = single_source.navigate(observations(16), diagnostics=True,
                                 code_blend=0., phase_state_continuity=True)
    state = nav['diagnostics']['phase_state_position_m']
    assert np.isnan(state[:2]).all()
    assert np.isnan(state[3:13]).all()
    assert nav['diagnostics']['isolated'][3:13].all()
    assert nav['diagnostics']['recovery_confirmed'][13]
    assert nav['position'][13, 0] == state[13, 0] == 2.
    assert nav['candidate_position'][13, 0] == 10.
    assert nav['diagnostics']['phase_state_unpublished_run_epochs'][13] == 0


def test_arc_change_and_time_gap_stop_internal_carrier_bridge(monkeypatch):
    fixed_code_geometry(monkeypatch)
    for change in ('arc', 'gap'):
        z = observations()
        if change == 'arc':
            z['arc'][23] += 1
        else:
            z['time'][23:] += .04
        nav = single_source.navigate(z, diagnostics=True, code_blend=0.,
                                     phase_leave_one_out=True,
                                     phase_state_continuity=True)
        assert not nav['diagnostics']['phase_ok'][23]
        assert np.isnan(nav['diagnostics']['phase_state_position_m'][23]).all()
        assert nav['diagnostics']['phase_state_short_anchor_index'][23] == -1


def test_disabled_flag_keeps_all_original_fields_and_diagnostics_exact(monkeypatch):
    fixed_code_geometry(monkeypatch)
    z = observations()
    default = single_source.navigate(z, diagnostics=True, code_blend=0.,
                                     phase_leave_one_out=True)
    explicit = single_source.navigate(z, diagnostics=True, code_blend=0.,
                                      phase_leave_one_out=True,
                                      phase_state_continuity=False)
    assert default.keys() == explicit.keys()
    assert default['diagnostics'].keys() == explicit['diagnostics'].keys()
    for key in ('position', 'candidate_position', 'selected'):
        np.testing.assert_array_equal(default[key], explicit[key], strict=True)
    for key in default['diagnostics']:
        np.testing.assert_array_equal(default['diagnostics'][key],
                                      explicit['diagnostics'][key], strict=True)
    assert default['initialization_time'] == explicit['initialization_time']


def test_phase_state_prefix_is_unchanged_by_future_observations(monkeypatch):
    fixed_code_geometry(monkeypatch)
    full = single_source.navigate(observations(), diagnostics=True,
                                  code_blend=0., phase_leave_one_out=True,
                                  phase_state_continuity=True)
    prefix = single_source.navigate(observations(26), diagnostics=True,
                                    code_blend=0., phase_leave_one_out=True,
                                    phase_state_continuity=True)
    for key in ('position', 'candidate_position', 'selected'):
        np.testing.assert_array_equal(prefix[key], full[key][:26], strict=True)
    for key in full['diagnostics']:
        np.testing.assert_array_equal(prefix['diagnostics'][key],
                                      full['diagnostics'][key][:26], strict=True)
