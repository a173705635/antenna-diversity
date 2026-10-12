"""Replay the established causal all-history float model with fixed supports.

Only original versus 100 ms coherent support changes. This offline model uses
30 m code and 0.03 m carrier weights, free current XYZ/clock, and persistent
satellite-arc biases; formal sigma is not an integrity bound or publication.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_coherent_navigation import (
    STAGE14, STAGE15, validate_sources,
)
from experiments.spoofing.single_source_coherent_transfer import (
    INPUT as TRANSFER_INPUT, preflight_input,
)
from experiments.spoofing.single_source_float_window_study import (
    CASE_INPUTS, FORMAL_HORIZONTAL_STD_STRATUM_M, HISTORY_OUTPUT,
    geometry_and_admission, solve_history,
)
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest


STAGE16 = RESULTS / STUDY_OUTPUT / 'coherent_navigation'
STAGE17 = RESULTS / STUDY_OUTPUT / 'coherent_transfer'
OUTPUT = RESULTS / STUDY_OUTPUT / 'coherent_float'
CASES = ('pll_weak', 'genuine_absent', 'control', 'transfer_weak')
SOLVE_FIELDS = ('position_m', 'formal_sigma_xyz_m', 'formal_horizontal_std_m',
                'condition', 'target_null_norm', 'rank', 'state_count',
                'code_count', 'carrier_count', 'arc_bias_count', 'observable')


def solve_pair(observations, coherent_support,
               geometry_fn=geometry_and_admission, solver_fn=solve_history):
    """Build each support admission and solve from an empty prior separately."""
    result = {}
    for name, support in (('original', observations['support']),
                          ('coherent', coherent_support)):
        design, code, carrier, valid = geometry_fn(
            {**observations, 'support': support})
        result[name] = solver_fn(design, code, carrier, valid,
                                 observations['arc']), valid
    return result


def observation_masks(time, available, start):
    """Float post uses observation time; navigation reference uses availability."""
    tail = np.arange(len(time)) >= len(time) - 50
    return time >= start, available >= start, tail


def float_metrics(error, formal_std, mask):
    values = error[mask]
    return dict(count=int(mask.sum()),
                error_ge_20=int((values >= 20.).sum()),
                error_ge_30=int((values >= 30.).sum()),
                error_ge_50=int((values >= 50.).sum()),
                median_error_m=None if not len(values) else float(np.median(values)),
                p95_error_m=None if not len(values) else float(np.percentile(values, 95)),
                max_error_m=None if not len(values) else float(values.max()),
                median_formal_horizontal_std_m=None if not len(values)
                else float(np.median(formal_std[mask])))


def preflight_sources():
    stage14, _ = validate_sources(STAGE14, STAGE15, CASE_INPUTS)
    stage16 = json.loads((STAGE16 / 'summary.json').read_text(encoding='utf-8'))
    assert stage16['stage14_summary_sha256'] == digest(STAGE14 / 'summary.json')
    assert stage16['stage15_summary_sha256'] == digest(STAGE15 / 'summary.json')
    assert stage16['manifest_sha256'] == digest(STAGE16 / 'manifest.json')
    stage16_manifest = json.loads((STAGE16 / 'manifest.json').read_text(encoding='utf-8'))
    for name in CASES[:3]:
        assert stage16['cases'][name]['result_sha256'] == digest(STAGE16 / f'{name}.npz')
        assert stage16['cases'][name]['table_sha256'] == digest(STAGE15 / f'{name}.npz')
        assert stage16_manifest['input_sha256'][name] == stage14['cases'][name]['input_sha256']
        assert stage16_manifest['stage14_tracking_sha256'][name] == stage14['cases'][name]['tracking_sha256']

    stage17 = json.loads((STAGE17 / 'summary.json').read_text(encoding='utf-8'))
    metadata, transfer_hashes = preflight_input(TRANSFER_INPUT)
    assert stage17['input_sha256'] == transfer_hashes
    assert stage17['manifest_sha256'] == digest(STAGE17 / 'manifest.json')
    stage17_manifest = json.loads((STAGE17 / 'manifest.json').read_text(encoding='utf-8'))
    assert stage17_manifest['input_sha256'] == transfer_hashes
    for filename in ('tracking.npz', 'support.npz'):
        assert stage17['output_sha256'][filename] == digest(STAGE17 / filename)

    old_float = json.loads((HISTORY_OUTPUT / 'summary.json').read_text(encoding='utf-8'))
    old_row = next(row for row in old_float['cases'] if row['name'] == 'pll_weak')
    old_result = HISTORY_OUTPUT / 'pll_weak.npz'
    assert old_row['result_sha256'] == digest(old_result)
    assert old_row['metadata_sha256'] == stage14['cases']['pll_weak']['input_sha256']['metadata.json']
    return stage14, stage16, stage17, old_result


def score_pair(observations, metadata, pair):
    time = observations['time']
    available = observations['available_time']
    post, post_available, tail = observation_masks(time, available,
                                                   metadata['case']['attack_start'])
    truth = np.column_stack((metadata['case']['velocity_east'] * time,
                             np.zeros(len(time))))
    scored = {}
    errors = {}
    for mode, (fit, valid) in pair.items():
        error = np.linalg.norm(fit['position_m'][:, :2] - truth, axis=1)
        errors[mode] = error
        observable = fit['observable']
        formal = fit['formal_horizontal_std_m']
        qualifying = observable & (formal <= FORMAL_HORIZONTAL_STD_STRATUM_M)
        measurement = post & observable
        scored[mode] = dict(
            observation_opportunities=int(post.sum()),
            available_opportunities=int(post_available.sum()),
            observable_available=int((post_available & observable).sum()),
            all_observable=float_metrics(error, formal, measurement),
            formal_std_le10=float_metrics(error, formal, post & qualifying),
            tail50_observable=int((tail & observable).sum()),
            tail50_formal_std_le10=int((tail & qualifying).sum()),
            admitted_count_mean=float(valid.sum(axis=1)[post].mean()),
            last_measurement_observation_time_s=None if not measurement.any()
                else float(time[measurement][-1]),
            last_measurement_available_time_s=None if not measurement.any()
                else float(available[measurement][-1]))
    common = post & pair['original'][0]['observable'] & pair['coherent'][0]['observable']
    for mode, (fit, _) in pair.items():
        scored[mode]['common_observable'] = float_metrics(
            errors[mode], fit['formal_horizontal_std_m'], common)
    return scored, errors, post, post_available, tail


def main():
    stage14, stage16, stage17, old_result = preflight_sources()
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_hashes = {name: stage14['cases'][name]['input_sha256'] for name in CASES[:3]}
    input_hashes['transfer_weak'] = stage17['input_sha256']
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources, input_sha256=input_hashes,
        stage14_summary_sha256=digest(STAGE14 / 'summary.json'),
        stage16_summary_sha256=digest(STAGE16 / 'summary.json'),
        stage17_summary_sha256=digest(STAGE17 / 'summary.json'),
        reference_float_summary_sha256=digest(HISTORY_OUTPUT / 'summary.json'),
        reference_float_result_sha256=digest(old_result),
        code_sigma_m=30., carrier_sigma_m=.03,
        formal_horizontal_std_stratum_m=FORMAL_HORIZONTAL_STD_STRATUM_M)),
        indent=2, allow_nan=False), encoding='utf-8')

    reports = {}
    plots = {}
    for name in CASES:
        if name == 'transfer_weak':
            tracking_path = STAGE17 / 'tracking.npz'
            support_path = STAGE17 / 'support.npz'
            metadata_path = TRANSFER_INPUT / 'metadata.json'
        else:
            tracking_path = STAGE14 / name / 'tracking.npz'
            support_path = STAGE16 / f'{name}.npz'
            metadata_path = CASE_INPUTS[name][1] / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        with np.load(tracking_path) as saved:
            observations = dict(saved)
        with np.load(support_path) as saved:
            coherent = saved['coherent_support']
        print(name, 'float start', flush=True)
        pair = solve_pair(observations, coherent)
        if name == 'pll_weak':
            with np.load(old_result) as reference:
                for field in SOLVE_FIELDS:
                    np.testing.assert_array_equal(pair['original'][0][field],
                                                  reference[field], err_msg=field)
        scores, errors, post, post_available, tail = score_pair(
            observations, metadata, pair)
        arrays = dict(observation_time=observations['time'],
                      available_time=observations['available_time'],
                      post_observation=post, post_available=post_available,
                      tail50=tail)
        for mode, (fit, valid) in pair.items():
            arrays.update({f'{mode}_{field}': value for field, value in fit.items()})
            arrays[f'{mode}_admitted_count'] = valid.sum(axis=1)
            arrays[f'{mode}_horizontal_error_m'] = errors[mode]
        result_path = OUTPUT / f'{name}.npz'
        np.savez_compressed(result_path, **arrays)
        reports[name] = dict(tracking_sha256=digest(tracking_path),
                             support_sha256=digest(support_path),
                             metadata_sha256=digest(metadata_path),
                             result_sha256=digest(result_path), modes=scores)
        plots[name] = (observations['time'], errors)
        print(name, 'float done', flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col',
                             constrained_layout=True)
    for axis, name in zip(axes.flat, CASES):
        time, errors = plots[name]
        for mode, values in errors.items():
            axis.plot(time, values, label=mode)
        axis.axhline(20., color='gray', linestyle=':', linewidth=1)
        axis.set(title=name, xlabel='Observation time (s)',
                 ylabel='Observable float error (m)')
        axis.grid(alpha=.3)
    axes[0, 0].legend()
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources, input_sha256=input_hashes,
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'), cases=reports)),
        indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
