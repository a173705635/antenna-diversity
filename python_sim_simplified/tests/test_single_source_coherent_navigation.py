"""Causal 100 ms coherent direction support from saved 5 ms groups."""
import json

import numpy as np
import pytest

from experiments.single_source_coherent_navigation import (
    build_coherent_support, run_navigation_modes, validate_sources,
)
from experiments.single_source_suppression_study import digest


CHIP = 2046
GROUP = 5 * CHIP


def groups(count=20):
    starts = np.arange(count) * GROUP
    return dict(star=np.zeros(count, int), arc=np.zeros(count, int),
                start_sample=starts, end_sample=starts + GROUP,
                available_time=np.full(count, .11), q=np.full(count, 2),
                gain=np.ones(count), on_coh=np.full(count, 4.),
                off_coh=np.ones(count))


def observations(times=(.1,), support=99.):
    times = np.array(times)
    return dict(time=times, available_time=times + .02,
                arc=np.zeros((len(times), 1), int),
                support=np.full((len(times), 1), support),
                ranges=np.ones((len(times), 1)))


def test_complete_100ms_uses_energy_sums_not_ratio_mean():
    table = groups()
    table['on_coh'][-1] = 0.
    table['off_coh'][-1] = 100.
    result = build_coherent_support(observations(), table)
    np.testing.assert_allclose(result['support'][0, 0], 76. / 119.)
    assert result['group_count'][0, 0] == 20
    np.testing.assert_allclose(result['start_time_s'][0, 0], 0.)
    np.testing.assert_allclose(result['end_time_s'][0, 0], .1)
    np.testing.assert_allclose(result['latest_group_available_time_s'][0, 0], .11)


@pytest.mark.parametrize('change', ['missing', 'arc', 'late', 'invalid_gain', 'unaligned'])
def test_incomplete_or_unavailable_100ms_does_not_borrow_original_support(change):
    table = groups()
    if change == 'missing':
        table = {key: values[:-1] for key, values in table.items()}
    elif change == 'arc':
        table['arc'][10] = 1
    elif change == 'late':
        table['available_time'][10] = .13
    elif change == 'invalid_gain':
        table['gain'][10] = .01
    else:
        table['start_sample'] += CHIP
        table['end_sample'] += CHIP
    result = build_coherent_support(observations(), table)
    assert np.isnan(result['support'][0, 0])
    assert result['group_count'][0, 0] == 0


def test_future_groups_cannot_change_past_output_and_arc_is_current():
    base = groups()
    earlier = build_coherent_support(observations(), base)
    later = groups(23)
    later['on_coh'][20:] = 1000.
    later['available_time'][20:] = .14
    full = build_coherent_support(observations((.1, .11, .115)), later)
    np.testing.assert_array_equal(full['support'][0], earlier['support'][0])
    assert np.isnan(full['support'][1, 0])  # .11 requires group ending .11, but it arrives later.
    other_arc = observations()
    other_arc['arc'][0, 0] = 1
    assert np.isnan(build_coherent_support(other_arc, base)['support'][0, 0])


def test_three_navigations_get_independent_support_without_mutating_tracking():
    original = observations((.1, .11), support=4.)
    original['gain'] = np.array([[.2], [.3]])
    original['cn0'] = np.array([[30.], [31.]])
    snapshot = {key: value.copy() for key, value in original.items()}
    noncoherent = np.array([[2.], [3.]])
    coherent = np.array([[5.], [6.]])
    calls = []

    def fake_navigate(record, diagnostics):
        assert diagnostics is True
        assert 'initialization_time' not in record
        for key, value in snapshot.items():
            if key != 'support':
                np.testing.assert_array_equal(record[key], value, err_msg=key)
        calls.append(record['support'].copy())
        return dict(initialization_time=float(record['support'][0, 0]))

    result = run_navigation_modes(original, noncoherent, coherent, fake_navigate)
    assert [result[name]['initialization_time'] for name in
            ('original', '20ms_noncoherent', '100ms_coherent')] == [4., 2., 5.]
    for key, value in snapshot.items():
        np.testing.assert_array_equal(original[key], value, err_msg=key)
    for expected, actual in zip((original['support'], noncoherent, coherent), calls):
        np.testing.assert_array_equal(actual, expected)


def test_source_preflight_rejects_changed_stage15_table(tmp_path):
    stage14 = tmp_path / 'stage14'
    stage15 = tmp_path / 'stage15'
    stage14.mkdir()
    stage15.mkdir()
    inputs = {}
    rows14 = {}
    rows15 = {}
    for name in ('pll_weak', 'genuine_absent', 'control'):
        folder = tmp_path / name
        folder.mkdir()
        for filename in ('samples.npy', 'observations.npz', 'metadata.json'):
            (folder / filename).write_text(name + filename, encoding='utf-8')
        output = stage14 / name
        output.mkdir()
        for kind in ('tracking', 'navigation', 'trace', 'history'):
            (output / f'{kind}.npz').write_text(name + kind, encoding='utf-8')
        table = stage15 / f'{name}.npz'
        table.write_text(name + 'table', encoding='utf-8')
        inputs[name] = (output / 'tracking.npz', folder)
        rows14[name] = dict(input_sha256={filename: digest(folder / filename)
                                          for filename in ('samples.npy', 'observations.npz', 'metadata.json')},
                            **{f'{kind}_sha256': digest(output / f'{kind}.npz')
                               for kind in ('tracking', 'navigation', 'trace', 'history')})
        rows15[name] = dict(table_sha256=digest(table),
                            stage14_trace_sha256=rows14[name]['trace_sha256'])
    (stage14 / 'summary.json').write_text(json.dumps({'cases': rows14}), encoding='utf-8')
    (stage15 / 'summary.json').write_text(json.dumps({
        'stage14_summary_sha256': digest(stage14 / 'summary.json'),
        'cases': rows15}), encoding='utf-8')
    validate_sources(stage14, stage15, inputs)
    (stage15 / 'control.npz').write_text('changed', encoding='utf-8')
    with pytest.raises(AssertionError, match='control.*table'):
        validate_sources(stage14, stage15, inputs)
