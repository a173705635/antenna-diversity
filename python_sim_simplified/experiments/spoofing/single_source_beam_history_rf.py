"""Replay weak three-antenna RF or postprocess saved beam-history tracks."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import frontend, navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, INPUT, OUTPUT, digest, tracking_counts, tracking_summary,
)
from experiments.spoofing.single_source_float_window_study import (
    CASE_INPUTS, FORMAL_HORIZONTAL_STD_STRATUM_M, geometry_and_admission, solve_history,
)
from experiments.spoofing.single_source_cv_bridge import cv_bridge, score as cv_score
from experiments.spoofing.single_source_suppression_study import RECEIVER_PLL, SELECTED


ROOT = RESULTS / OUTPUT
DESTINATION = ROOT / 'beam_history_rf'
THRESHOLDS_M = (20., 30., 50.)
BACKEND = DESTINATION / 'backend'
COMBINED = DESTINATION / 'combined'
MODE = 'combined'


def score(navigation, observations, case):
    """Score published positions against truth at the observation epoch."""
    times = observations['time']
    available = observations['available_time']
    published = np.isfinite(navigation['position']).all(axis=1)
    post = available >= case['attack_start']
    truth = np.column_stack((case['velocity_east'] * times, np.zeros(len(times))))
    error = np.linalg.norm(navigation['position'][:, :2] - truth, axis=1)
    missed = np.r_[False, ~published[post], False]
    starts = np.flatnonzero(np.diff(missed.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(missed.astype(int)) == -1)
    tail = available >= available[-1] - .5
    finite = error[post & published]
    return dict(opportunities=int(post.sum()), published=int(published[post].sum()),
                error_ge_20=int((finite >= 20.).sum()),
                error_ge_30=int((finite >= 30.).sum()),
                error_ge_50=int((finite >= 50.).sum()),
                max_error_m=None if not len(finite) else float(finite.max()),
                p95_error_m=None if not len(finite) else float(np.percentile(finite, 95)),
                longest_unavailable_s=0. if not len(starts) else float((ends-starts).max() * .01),
                tail_published=int(published[tail].sum()),
                tail_opportunities=int(tail.sum()),
                initialization_available_time_s=navigation['initialization_time'])


def epoch_diagnostics(navigation, observations, case):
    times = observations['time']
    truth = np.column_stack((case['velocity_east'] * times, np.zeros(len(times))))
    error = np.linalg.norm(navigation['position'][:, :2] - truth, axis=1)
    published = np.isfinite(navigation['position']).all(axis=1)
    result = dict(observation_time=times, available_time=observations['available_time'],
                  horizontal_error_m=error, published=published)
    for threshold in THRESHOLDS_M:
        result[f'error_ge_{int(threshold)}m'] = published & (error >= threshold)
    result.update(tracking_counts(observations))
    return result


def rf_main():
    metadata = json.loads((INPUT / 'metadata.json').read_text(encoding='utf-8'))
    with np.load(INPUT / 'observations.npz') as saved:
        record = dict(saved)
    record['samples'] = np.load(INPUT / 'samples.npy', mmap_mode='r')
    hashes = {name: digest(INPUT / name)
              for name in ('samples.npy', 'observations.npz', 'metadata.json')}
    baseline_tracking = ROOT / 'tracking.npz'
    baseline_navigation = ROOT / 'navigation.npz'
    stage_one = json.loads((ROOT / 'summary.json').read_text(encoding='utf-8'))
    assert hashes == stage_one['input_sha256']
    assert digest(baseline_tracking) == stage_one['tracking_sha256']
    assert digest(baseline_navigation) == stage_one['navigation_sha256']
    with np.load(baseline_tracking) as saved:
        historical_tracking = dict(saved)
    with np.load(baseline_navigation) as saved:
        historical_navigation = dict(saved)
    source_hash = source_hashes()
    DESTINATION.mkdir(exist_ok=True)
    (DESTINATION / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, input=str(INPUT.resolve()), input_sha256=hashes,
        source_sha256=source_hash, baseline_tracking_sha256=digest(baseline_tracking),
        baseline_navigation_sha256=digest(baseline_navigation),
        antennas=ANTENNAS, beam_history_blocks=[1, 5],
        score_truth_time='observation time',
        publication_opportunity='available_time >= attack_start')),
        indent=2), encoding='utf-8')

    reports = {}
    observations_by_mode = {}
    navigation_by_mode = {}
    for name, blocks in (('default', 1), ('five_blocks', 5)):
        print(name, 'frontend start', flush=True)
        observations = frontend(record, ANTENNAS, split_support=True,
                                beam_history_blocks=blocks)
        print(name, 'frontend done; navigation start', flush=True)
        navigation = navigate(observations)
        if blocks == 1:
            for key, value in historical_tracking.items():
                np.testing.assert_array_equal(observations[key], value, err_msg=f'tracking.{key}')
            np.testing.assert_array_equal(navigation['position'], historical_navigation['baseline_position'])
            np.testing.assert_array_equal(navigation['selected'], historical_navigation['baseline_selected'])
            assert navigation['initialization_time'] == stage_one['original']['initialization_available_time_s']
        else:
            original = observations_by_mode['default']
            for key in ('time', 'available_time', 'startup_prns', 'calibration',
                        'calibration_rate', 'satellites', 'ratio'):
                np.testing.assert_array_equal(observations[key], original[key], err_msg=key)
        np.savez_compressed(DESTINATION / f'{name}_tracking.npz', **observations)
        np.savez_compressed(DESTINATION / f'{name}_navigation.npz', **navigation)
        np.savez_compressed(DESTINATION / f'{name}_diagnostics.npz',
                            **epoch_diagnostics(navigation, observations, metadata['case']))
        counts = tracking_counts(observations)
        available = observations['available_time']
        reports[name] = dict(
            score=score(navigation, observations, metadata['case']),
            tracking=tracking_summary(counts, available, metadata['case']['attack_start']),
            tracking_sha256=digest(DESTINATION / f'{name}_tracking.npz'),
            navigation_sha256=digest(DESTINATION / f'{name}_navigation.npz'),
            diagnostics_sha256=digest(DESTINATION / f'{name}_diagnostics.npz'))
        observations_by_mode[name] = observations
        navigation_by_mode[name] = navigation
        print(name, 'navigation done', reports[name]['score'], flush=True)

    assert source_hashes() == source_hash
    (DESTINATION / 'summary.json').write_text(json.dumps(serializable(dict(
        input_sha256=hashes, source_sha256=source_hash,
        baseline_tracking_sha256=digest(baseline_tracking),
        baseline_navigation_sha256=digest(baseline_navigation),
        comparisons=reports)), indent=2, allow_nan=False), encoding='utf-8')
    return reports


def backend_main():
    """Apply unchanged causal float and CV methods to both saved RF tracks."""
    rf_summary = json.loads((DESTINATION / 'summary.json').read_text(encoding='utf-8'))
    metadata_path = INPUT / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert digest(metadata_path) == rf_summary['input_sha256']['metadata.json']
    sources = source_hashes()
    BACKEND.mkdir(exist_ok=True)
    (BACKEND / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope='Offline unchanged all-history float and conditional CV postprocessing',
        source_sha256=sources, input_sha256=rf_summary['input_sha256'],
        rf_summary_sha256=digest(DESTINATION / 'summary.json'),
        metadata_sha256=digest(metadata_path))), indent=2), encoding='utf-8')

    rows = {}
    plotted = {}
    for name in ('default', 'five_blocks'):
        tracking_path = DESTINATION / f'{name}_tracking.npz'
        assert digest(tracking_path) == rf_summary['comparisons'][name]['tracking_sha256']
        with np.load(tracking_path) as saved:
            observations = dict(saved)
        design, code, carrier, valid = geometry_and_admission(observations)
        print(name, 'float history start', flush=True)
        floating = solve_history(design, code, carrier, valid, observations['arc'])
        times = observations['time']
        post = times >= metadata['case']['attack_start']
        truth = np.column_stack((metadata['case']['velocity_east'] * times,
                                 np.zeros(len(times))))
        float_error = np.linalg.norm(floating['position_m'][:, :2] - truth, axis=1)
        formal = floating['formal_horizontal_std_m']
        formal_layer = floating['observable'] & (formal <= FORMAL_HORIZONTAL_STD_STRATUM_M)
        bridge = cv_bridge(times, floating['position_m'], formal)
        bridge_score, bridge_error = cv_score(times, bridge, metadata)
        if name == 'default':
            with np.load(ROOT / 'float_history' / 'baseline_weak.npz') as historical:
                for key, value in floating.items():
                    np.testing.assert_array_equal(value, historical[key], err_msg=f'float.{key}')
                np.testing.assert_array_equal(float_error, historical['horizontal_error_m'])
                np.testing.assert_array_equal(formal_layer, historical['diagnostic_std_le10'])
            with np.load(ROOT / 'cv_bridge' / 'baseline_weak.npz') as historical:
                for key, value in bridge.items():
                    np.testing.assert_array_equal(value, historical[key], err_msg=f'cv.{key}')
                np.testing.assert_array_equal(bridge_error, historical['horizontal_error_m'])

        output = BACKEND / f'{name}.npz'
        np.savez_compressed(output, observation_time=times,
                            post=post, admitted_count=valid.sum(axis=1),
                            float_horizontal_error_m=float_error,
                            float_formal_le10=formal_layer,
                            **{f'float_{key}': value for key, value in floating.items()},
                            cv_horizontal_error_m=bridge_error,
                            **{f'cv_{key}': value for key, value in bridge.items()})
        observable = post & floating['observable']
        accepted = post & formal_layer
        rows[name] = dict(
            tracking_sha256=digest(tracking_path), result_sha256=digest(output),
            observation_time_opportunities=int(post.sum()),
            observable=int(observable.sum()),
            observable_max_error_m=float(float_error[observable].max()),
            formal_le10=int(accepted.sum()),
            formal_le10_max_error_m=float(float_error[accepted].max()),
            cv=bridge_score)
        plotted[name] = (times, floating['observable'], float_error, bridge['status'])
        print(name, 'backend done', rows[name], flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)
    for name, (time, observable, error, status) in plotted.items():
        axes[0].plot(time, np.where(observable, error, np.nan), label=name)
        axes[1].plot(time, (status == 'measured').astype(int)
                     + .5 * (status == 'predicted').astype(int), label=name)
    axes[0].set(ylabel='Observable float error (m)', ylim=(0, 10))
    axes[1].set(xlabel='Observation time (s)', ylabel='CV: 1 measured, 0.5 predicted')
    for axis in axes:
        axis.grid(alpha=.3)
        axis.legend()
    fig.savefig(BACKEND / 'summary.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (BACKEND / 'summary.json').write_text(json.dumps(serializable(dict(
        scope='Offline float/CV diagnostics; no receiver publication',
        source_sha256=sources, input_sha256=rf_summary['input_sha256'],
        rf_summary_sha256=digest(DESTINATION / 'summary.json'),
        metadata_sha256=digest(metadata_path), rows=rows)),
        indent=2, allow_nan=False), encoding='utf-8')
    return rows


def combined_main():
    """Fixed five-block beam plus existing 0.15/1 receive PLL on two RFs."""
    rf_summary = json.loads((DESTINATION / 'summary.json').read_text(encoding='utf-8'))
    pll_summary = json.loads((ROOT / 'pll' / 'summary.json').read_text(encoding='utf-8'))
    float_summary = json.loads((ROOT / 'float_history' / 'summary.json').read_text(encoding='utf-8'))
    selected = json.loads(SELECTED.read_text(encoding='utf-8'))
    absent_tracking, absent_input = CASE_INPUTS['genuine_absent']
    absent_row = next(row for row in selected['selected_rows']
                      if row['after_id'] == absent_tracking.parent.name)
    absent_float_row = next(row for row in float_summary['cases']
                            if row['name'] == 'genuine_absent')
    assert digest(ROOT / 'pll' / 'tracking.npz') == pll_summary['tracking_sha256']
    assert digest(ROOT / 'pll' / 'navigation.npz') == pll_summary['navigation_sha256']
    assert digest(absent_tracking) == absent_float_row['tracking_sha256']
    assert digest(absent_input / 'metadata.json') == absent_float_row['metadata_sha256']
    sources = source_hashes()
    cases = dict(weak=(INPUT, ROOT / 'pll' / 'tracking.npz'),
                 genuine_absent=(absent_input, absent_tracking))
    inputs = {}
    for name, (folder, _) in cases.items():
        inputs[name] = {item: digest(folder / item)
                        for item in ('samples.npy', 'observations.npz', 'metadata.json')}
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        assert inputs[name]['samples.npy'] == metadata['samples_sha256']
        assert inputs[name]['observations.npz'] == metadata['observations_sha256']
    assert inputs['weak'] == rf_summary['input_sha256'] == pll_summary['input_sha256']
    assert inputs['genuine_absent']['samples.npy'] == absent_row['after']['input_sha256']
    COMBINED.mkdir(exist_ok=True)
    (COMBINED / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope='Two saved RFs, five-block beam plus fixed 0.15/1 receive PLL',
        source_sha256=sources, input_sha256=inputs,
        rf_summary_sha256=digest(DESTINATION / 'summary.json'),
        pll_summary_sha256=digest(ROOT / 'pll' / 'summary.json'),
        selected_sha256=digest(SELECTED), receiver_pll=RECEIVER_PLL,
        antennas=ANTENNAS, beam_history_blocks=5)), indent=2), encoding='utf-8')

    results = {}
    for name, (folder, reference_path) in cases.items():
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        with np.load(folder / 'observations.npz') as saved:
            record = dict(saved)
        record['samples'] = np.load(folder / 'samples.npy', mmap_mode='r')
        with np.load(reference_path) as saved:
            reference = dict(saved)
        print(name, 'combined frontend start', flush=True)
        observations = frontend(record, ANTENNAS, split_support=True,
                                receiver_pll=RECEIVER_PLL, beam_history_blocks=5)
        for key in ('time', 'available_time', 'startup_prns', 'calibration',
                    'calibration_rate', 'satellites', 'ratio'):
            np.testing.assert_array_equal(observations[key], reference[key], err_msg=f'{name}.{key}')
        if name == 'weak':
            with np.load(DESTINATION / 'five_blocks_tracking.npz') as old_candidate:
                np.testing.assert_array_equal(observations['beam_history_weights'],
                                              old_candidate['beam_history_weights'])
        print(name, 'combined frontend done; navigation and backend start', flush=True)
        navigation = navigate(observations)
        design, code, carrier, valid = geometry_and_admission(observations)
        floating = solve_history(design, code, carrier, valid, observations['arc'])
        bridge = cv_bridge(observations['time'], floating['position_m'],
                           floating['formal_horizontal_std_m'])
        cv_metrics, cv_error = cv_score(observations['time'], bridge, metadata)
        post = observations['time'] >= metadata['case']['attack_start']
        truth = np.column_stack((metadata['case']['velocity_east'] * observations['time'],
                                 np.zeros(len(post))))
        float_error = np.linalg.norm(floating['position_m'][:, :2] - truth, axis=1)
        observable = post & floating['observable']
        formal = observable & (floating['formal_horizontal_std_m'] <= FORMAL_HORIZONTAL_STD_STRATUM_M)

        np.savez_compressed(COMBINED / f'{name}_tracking.npz', **observations)
        np.savez_compressed(COMBINED / f'{name}_navigation.npz', **navigation)
        np.savez_compressed(COMBINED / f'{name}_diagnostics.npz',
                            **epoch_diagnostics(navigation, observations, metadata['case']))
        np.savez_compressed(COMBINED / f'{name}_backend.npz',
                            observation_time=observations['time'], post=post,
                            float_horizontal_error_m=float_error,
                            **{f'float_{key}': value for key, value in floating.items()},
                            cv_horizontal_error_m=cv_error,
                            **{f'cv_{key}': value for key, value in bridge.items()})
        available = observations['available_time']
        steady = available >= .5
        published = np.isfinite(navigation['position']).all(axis=1)
        results[name] = dict(
            reference_tracking_sha256=digest(reference_path),
            tracking_sha256=digest(COMBINED / f'{name}_tracking.npz'),
            navigation_sha256=digest(COMBINED / f'{name}_navigation.npz'),
            diagnostics_sha256=digest(COMBINED / f'{name}_diagnostics.npz'),
            backend_sha256=digest(COMBINED / f'{name}_backend.npz'),
            navigation=score(navigation, observations, metadata['case']),
            steady_published=int((steady & published).sum()),
            steady_opportunities=int(steady.sum()),
            published_after_0p6s=int((published & (available >= .6)).sum()),
            last_published_available_time_s=None if not published.any()
            else float(available[published][-1]),
            tracking=tracking_summary(tracking_counts(observations), available,
                                      metadata['case']['attack_start']),
            float=dict(opportunities=int(post.sum()), observable=int(observable.sum()),
                       formal_le10=int(formal.sum()),
                       max_observable_error_m=None if not observable.any() else float(float_error[observable].max())),
            cv=cv_metrics)
        print(name, 'combined done', results[name]['navigation'], flush=True)

    return combined_finalize()


def combined_finalize():
    """Summarize saved combined replay without running another RF frontend."""
    manifest = json.loads((COMBINED / 'manifest.json').read_text(encoding='utf-8'))
    assert digest(DESTINATION / 'summary.json') == manifest['rf_summary_sha256'], 'RF summary changed'
    assert digest(ROOT / 'pll' / 'summary.json') == manifest['pll_summary_sha256'], 'PLL summary changed'
    assert digest(SELECTED) == manifest['selected_sha256'], 'selected summary changed'
    pll_summary = json.loads((ROOT / 'pll' / 'summary.json').read_text(encoding='utf-8'))
    selected = json.loads(SELECTED.read_text(encoding='utf-8'))
    absent_tracking, absent_input = CASE_INPUTS['genuine_absent']
    absent_row = next(row for row in selected['selected_rows']
                      if row['after_id'] == absent_tracking.parent.name)
    cases = dict(weak=INPUT, genuine_absent=absent_input)
    metadata_by_case = {}
    for name, folder in cases.items():
        metadata_path = folder / 'metadata.json'
        assert digest(metadata_path) == manifest['input_sha256'][name]['metadata.json'], f'{name} metadata.json changed'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        assert metadata['samples_sha256'] == manifest['input_sha256'][name]['samples.npy']
        assert metadata['observations_sha256'] == manifest['input_sha256'][name]['observations.npz']
        metadata_by_case[name] = metadata
    assert digest(ROOT / 'pll' / 'tracking.npz') == pll_summary['tracking_sha256'], 'PLL tracking changed'
    assert digest(ROOT / 'pll' / 'navigation.npz') == pll_summary['navigation_sha256'], 'PLL navigation changed'
    absent_float_summary = json.loads((ROOT / 'float_history' / 'summary.json').read_text(encoding='utf-8'))
    absent_float_row = next(row for row in absent_float_summary['cases']
                            if row['name'] == 'genuine_absent')
    assert digest(absent_tracking) == absent_float_row['tracking_sha256'], 'absent reference tracking changed'
    assert absent_float_row['metadata_sha256'] == manifest['input_sha256']['genuine_absent']['metadata.json']
    assert absent_row['after']['input_sha256'] == manifest['input_sha256']['genuine_absent']['samples.npy']
    results = {}
    for name, folder in cases.items():
        metadata = metadata_by_case[name]
        with np.load(COMBINED / f'{name}_tracking.npz') as saved:
            observations = dict(saved)
        with np.load(COMBINED / f'{name}_navigation.npz') as saved:
            navigation = dict(saved)
        with np.load(COMBINED / f'{name}_backend.npz') as saved:
            backend = dict(saved)
        available = observations['available_time']
        published = np.isfinite(navigation['position']).all(axis=1)
        steady = available >= .5
        post = backend['observation_time'] >= metadata['case']['attack_start']
        observable = post & backend['float_observable']
        formal = observable & (backend['float_formal_horizontal_std_m'] <= FORMAL_HORIZONTAL_STD_STRATUM_M)
        bridge = {key[3:]: value for key, value in backend.items() if key.startswith('cv_')
                  and key != 'cv_horizontal_error_m'}
        cv_metrics, _ = cv_score(observations['time'], bridge, metadata)
        reference_path = ROOT / 'pll' / 'tracking.npz' if name == 'weak' else absent_tracking
        results[name] = dict(
            reference_tracking_sha256=digest(reference_path),
            tracking_sha256=digest(COMBINED / f'{name}_tracking.npz'),
            navigation_sha256=digest(COMBINED / f'{name}_navigation.npz'),
            diagnostics_sha256=digest(COMBINED / f'{name}_diagnostics.npz'),
            backend_sha256=digest(COMBINED / f'{name}_backend.npz'),
            navigation=score(navigation, observations, metadata['case']),
            steady_published=int((steady & published).sum()),
            steady_opportunities=int(steady.sum()),
            published_after_0p6s=int((published & (available >= .6)).sum()),
            last_published_available_time_s=None if not published.any()
            else float(available[published][-1]),
            tracking=tracking_summary(tracking_counts(observations), available,
                                      metadata['case']['attack_start']),
            float=dict(opportunities=int(post.sum()), observable=int(observable.sum()),
                       formal_le10=int(formal.sum()),
                       max_observable_error_m=None if not observable.any()
                       else float(backend['float_horizontal_error_m'][observable].max())),
            cv=cv_metrics)

    # Historical negative control used the default receive PLL, so it is a
    # guard against false recovery rather than a two-factor comparison.
    with np.load(absent_tracking.parent / 'navigation.npz') as saved:
        old_negative = dict(saved)
    with np.load(absent_tracking) as saved:
        old_negative_tracking = dict(saved)
    negative_reference = dict(position=old_negative['position'],
                              initialization_time=absent_row['after']['initialization_time'])
    references = dict(
        weak_pll_tracking_sha256=digest(ROOT / 'pll' / 'tracking.npz'),
        weak_pll_navigation_sha256=digest(ROOT / 'pll' / 'navigation.npz'),
        negative_tracking_sha256=digest(absent_tracking),
        negative_navigation_sha256=digest(absent_tracking.parent / 'navigation.npz'),
        negative_default_pll_score=score(negative_reference, old_negative_tracking,
                                        metadata_by_case['genuine_absent']['case']))
    with np.load(ROOT / 'pll' / 'navigation.npz') as saved:
        old_pll_navigation = dict(saved)
    with np.load(ROOT / 'pll' / 'tracking.npz') as saved:
        old_pll_tracking = dict(saved)
    references['weak_pll_score'] = score(dict(position=old_pll_navigation['position'],
                                             initialization_time=pll_summary['pll']['initialization_available_time_s']),
                                         old_pll_tracking,
                                         metadata_by_case['weak']['case'])
    (COMBINED / 'summary.json').write_text(json.dumps(serializable(dict(
        rf_source_sha256=manifest['source_sha256'],
        finalizer_source_sha256=source_hashes(),
        input_sha256=manifest['input_sha256'], references=references,
        cases=results)), indent=2, allow_nan=False), encoding='utf-8')
    return results


def main():
    if MODE == 'backend':
        return backend_main()
    if MODE == 'combined':
        return combined_main()
    if MODE == 'combined_finalize':
        return combined_finalize()
    return rf_main()


if __name__ == '__main__':
    main()
