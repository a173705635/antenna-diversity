"""Replay saved tracking with one causal 100 ms coherent support candidate.

The numerical threshold remains three, but coherent and noncoherent ratios
are different statistics. This experiment does not calibrate false alarms.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.spoofing.recovery_geometry import FS
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_beam_history_rf import score
from experiments.spoofing.single_source_coherence_diagnosis import (
    CASES, STAGE14, OUTPUT as STAGE15, validate_saved_sources,
)
from experiments.spoofing.single_source_float_window_study import CASE_INPUTS
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, digest, tracking_counts, tracking_summary,
)


OUTPUT = RESULTS / STUDY_OUTPUT / 'coherent_navigation'
GROUP_SAMPLES = 5 * 2046
WINDOW_GROUPS = 20
THRESHOLD = 3.


def build_coherent_support(observations, table):
    """Use twenty unique, available 5 ms groups ending at each observation."""
    shape = observations['support'].shape
    result = dict(support=np.full(shape, np.nan), group_count=np.zeros(shape, int),
                  start_time_s=np.full(shape, np.nan), end_time_s=np.full(shape, np.nan),
                  latest_group_available_time_s=np.full(shape, np.nan))
    groups = {}
    for index in range(len(table['star'])):
        start = int(table['start_sample'][index])
        if start % GROUP_SAMPLES or table['end_sample'][index] != start + GROUP_SAMPLES:
            continue
        key = (int(table['star'][index]), int(table['arc'][index]),
               start)
        assert key not in groups
        groups[key] = index

    for k, time in enumerate(observations['time']):
        end = round(float(time) * FS)
        first = end - WINDOW_GROUPS * GROUP_SAMPLES
        available = observations['available_time'][k]
        for star in range(shape[1]):
            arc = int(observations['arc'][k, star])
            if arc < 0:
                continue
            keys = [(star, arc, first + block * GROUP_SAMPLES)
                    for block in range(WINDOW_GROUPS)]
            if any(key not in groups for key in keys):
                continue
            indices = [groups[key] for key in keys]
            if any(table['available_time'][index] > available for index in indices):
                continue
            if any(table['q'][index] < 2 or
                   table['gain'][index] / len(ANTENNAS) < .05 or
                   not np.isfinite(table['on_coh'][index]) or
                   not np.isfinite(table['off_coh'][index])
                   for index in indices):
                continue
            on = table['on_coh'][indices].sum()
            off = table['off_coh'][indices].sum()
            result['support'][k, star] = on / off
            result['group_count'][k, star] = WINDOW_GROUPS
            result['start_time_s'][k, star] = first / FS
            result['end_time_s'][k, star] = end / FS
            result['latest_group_available_time_s'][k, star] = table['available_time'][indices].max()
    return result


def run_navigation_modes(observations, noncoherent, coherent, navigation_fn=navigate):
    """Run three fresh navigation states, changing only each support array."""
    supports = dict(original=observations['support'],
                    **{'20ms_noncoherent': noncoherent, '100ms_coherent': coherent})
    return {name: navigation_fn({**observations, 'support': support}, diagnostics=True)
            for name, support in supports.items()}


def validate_sources(stage14, stage15, case_inputs):
    """Verify both saved stages and original inputs before replaying navigation."""
    previous = validate_saved_sources(stage14, case_inputs)
    current = json.loads((stage15 / 'summary.json').read_text(encoding='utf-8'))
    assert current['stage14_summary_sha256'] == digest(stage14 / 'summary.json')
    for name in CASES:
        assert digest(stage15 / f'{name}.npz') == current['cases'][name]['table_sha256'], f'{name} table changed'
        assert current['cases'][name]['stage14_trace_sha256'] == previous['cases'][name]['trace_sha256']
    return previous, current


def quality_support_counts(observations, support):
    quality = (np.isfinite(observations['ranges']) & (observations['lock'] >= .5)
               & (observations['cn0'] >= 25.) & (observations['gain'] >= .05))
    return (quality & (support >= THRESHOLD)).sum(axis=1)


def score_mode(navigation, observations, case, counts, support):
    """Keep output opportunities and truth times distinct, as in Stage 14."""
    available = observations['available_time']
    post = available >= case['attack_start']
    steady = available >= .5
    published = np.isfinite(navigation['position']).all(axis=1)
    supported = counts >= 6
    initialization = navigation['initialization_time']
    reasons, reason_counts = np.unique(navigation['diagnostics']['reason'][post],
                                       return_counts=True)
    steady_reasons, steady_reason_counts = np.unique(
        navigation['diagnostics']['reason'][steady], return_counts=True)
    return dict(navigation=score(navigation, observations, case),
                tracking=tracking_summary(
                    tracking_counts({**observations, 'support': support}),
                    available, case['attack_start']),
                initialized_before_evaluation=initialization is not None
                    and initialization < case['attack_start'],
                steady_published=int((steady & published).sum()),
                steady_opportunities=int(steady.sum()),
                last_published_available_time_s=None if not published.any() else
                    float(available[published][-1]),
                post_support_ge6=int((post & supported).sum()),
                steady_support_ge6=int((steady & supported).sum()),
                last_support_ge6_available_time_s=None if not supported.any() else
                    float(available[supported][-1]),
                last_post_support_ge6_available_time_s=None if not (post & supported).any() else
                    float(available[post & supported][-1]),
                post_support_mean=float(counts[post].mean()),
                steady_support_mean=float(counts[steady].mean()),
                post_reasons={str(reason): int(count) for reason, count in
                              zip(reasons, reason_counts)},
                steady_reasons={str(reason): int(count) for reason, count in
                                zip(steady_reasons, steady_reason_counts)})


def main():
    stage14, stage15 = validate_sources(STAGE14, STAGE15, CASE_INPUTS)
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        stage14_summary_sha256=digest(STAGE14 / 'summary.json'),
        stage15_summary_sha256=digest(STAGE15 / 'summary.json'),
        input_sha256={name: stage14['cases'][name]['input_sha256'] for name in CASES},
        stage14_tracking_sha256={name: stage14['cases'][name]['tracking_sha256'] for name in CASES},
        stage15_table_sha256={name: stage15['cases'][name]['table_sha256'] for name in CASES},
        window_ms=100, group_ms=5, groups_per_window=WINDOW_GROUPS,
        numerical_threshold=THRESHOLD)), indent=2), encoding='utf-8')

    reports = {}
    plotted = {}
    for name in CASES:
        _, folder = CASE_INPUTS[name]
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        with np.load(STAGE14 / name / 'tracking.npz') as saved:
            observations = dict(saved)
        with np.load(STAGE14 / name / 'history.npz') as saved:
            noncoherent = saved['ratio'][:, :, 0]
        with np.load(STAGE15 / f'{name}.npz') as saved:
            table = dict(saved)
        coherent = build_coherent_support(observations, table)
        navigations = run_navigation_modes(observations, noncoherent, coherent['support'])
        with np.load(STAGE14 / name / 'navigation.npz') as saved:
            for key in saved.files:
                np.testing.assert_array_equal(navigations['original'][key], saved[key], err_msg=f'{name}.{key}')
        scores = {}
        saved_arrays = {f'coherent_{key}': value for key, value in coherent.items()}
        saved_arrays['noncoherent_support'] = noncoherent
        support_modes = dict(original=observations['support'],
                             **{'20ms_noncoherent': noncoherent,
                                '100ms_coherent': coherent['support']})
        for mode, navigation in navigations.items():
            support = support_modes[mode]
            counts = quality_support_counts(observations, support)
            scores[mode] = score_mode(navigation, observations, metadata['case'],
                                      counts, support)
            saved_arrays[f'{mode}_support_count'] = counts
            for key, value in navigation.items():
                if key == 'diagnostics':
                    for diagnostic, data in value.items():
                        saved_arrays[f'{mode}_diagnostic_{diagnostic}'] = data
                else:
                    saved_arrays[f'{mode}_{key}'] = np.nan if value is None else value
        path = OUTPUT / f'{name}.npz'
        np.savez_compressed(path, **saved_arrays)
        reports[name] = dict(table_sha256=stage15['cases'][name]['table_sha256'],
                             result_sha256=digest(path), modes=scores)
        plotted[name] = (observations['time'], observations['available_time'],
                         observations, support_modes, navigations)

    fig, axes = plt.subplots(2, len(CASES), figsize=(13, 7), sharex='col',
                             constrained_layout=True)
    for column, name in enumerate(CASES):
        times, available, observations, support_modes, navigations = plotted[name]
        metadata = json.loads((CASE_INPUTS[name][1] / 'metadata.json').read_text(encoding='utf-8'))
        truth = np.column_stack((metadata['case']['velocity_east'] * times,
                                 np.zeros(len(times))))
        for mode, navigation in navigations.items():
            counts = quality_support_counts(observations, support_modes[mode])
            error = np.linalg.norm(navigation['position'][:, :2] - truth, axis=1)
            axes[0, column].plot(available, counts, label=mode)
            axes[1, column].plot(available, error, label=mode)
        axes[0, column].set(title=name, ylabel='Quality-gated support count')
        axes[1, column].set(xlabel='Output available time (s)', ylabel='Published error (m)')
        for axis in axes[:, column]:
            axis.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        stage14_summary_sha256=digest(STAGE14 / 'summary.json'),
        stage15_summary_sha256=digest(STAGE15 / 'summary.json'),
        manifest_sha256=digest(OUTPUT / 'manifest.json'), cases=reports)),
        indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
