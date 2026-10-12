"""Replay saved coherent support with the fixed 0.1 and zero code blends.

Zero removes code from one accepted code/phase mix only. Fresh code still
governs initialization, separation, anchoring, and recovery in navigate.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import navigate
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_coherent_float import (
    CASES, STAGE14, STAGE16, STAGE17, OUTPUT as STAGE18,
    preflight_sources,
)
from experiments.spoofing.single_source_coherent_navigation import (
    quality_support_counts, score_mode,
)
from experiments.spoofing.single_source_coherent_transfer import INPUT as TRANSFER_INPUT
from experiments.spoofing.single_source_float_window_study import CASE_INPUTS
from experiments.spoofing.single_source_suppression_study import OUTPUT as STUDY_OUTPUT, digest


OUTPUT = RESULTS / STUDY_OUTPUT / 'code_blend'
BLENDS = (('.1', .1), ('0', 0.))


def assert_saved_navigation(navigation, saved, prefix='100ms_coherent_'):
    """Check the complete saved Stage 16/17 coherent navigation, diagnostics too."""
    for key, value in navigation.items():
        if key == 'diagnostics':
            for diagnostic, data in value.items():
                np.testing.assert_array_equal(
                    data, saved[f'{prefix}diagnostic_{diagnostic}'],
                    err_msg=diagnostic)
        else:
            np.testing.assert_array_equal(value, saved[f'{prefix}{key}'],
                                          err_msg=key)


def published_metrics(error, mask):
    values = error[mask]
    return dict(count=int(mask.sum()), error_ge_20=int((values >= 20.).sum()),
                error_ge_30=int((values >= 30.).sum()),
                error_ge_50=int((values >= 50.).sum()),
                median_error_m=None if not len(values) else float(np.median(values)),
                p95_error_m=None if not len(values) else float(np.percentile(values, 95)),
                max_error_m=None if not len(values) else float(values.max()))


def main():
    preflight_sources()
    stage17 = json.loads((STAGE17 / 'summary.json').read_text(encoding='utf-8'))
    assert stage17['output_sha256']['navigation.npz'] == digest(STAGE17 / 'navigation.npz')
    stage18 = json.loads((STAGE18 / 'summary.json').read_text(encoding='utf-8'))
    assert stage18['manifest_sha256'] == digest(STAGE18 / 'manifest.json')
    for name in CASES:
        assert stage18['cases'][name]['result_sha256'] == digest(STAGE18 / f'{name}.npz')
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        stage18_summary_sha256=digest(STAGE18 / 'summary.json'),
        stage16_summary_sha256=digest(STAGE16 / 'summary.json'),
        stage17_summary_sha256=digest(STAGE17 / 'summary.json'),
        blend_values={name: value for name, value in BLENDS},
        support='100ms_coherent')), indent=2, allow_nan=False), encoding='utf-8')

    reports = {}
    plots = {}
    for name in CASES:
        if name == 'transfer_weak':
            tracking_path = STAGE17 / 'tracking.npz'
            support_path = STAGE17 / 'support.npz'
            reference_path = STAGE17 / 'navigation.npz'
            metadata_path = TRANSFER_INPUT / 'metadata.json'
        else:
            tracking_path = STAGE14 / name / 'tracking.npz'
            support_path = STAGE16 / f'{name}.npz'
            reference_path = support_path
            metadata_path = CASE_INPUTS[name][1] / 'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        with np.load(tracking_path) as saved:
            tracking = dict(saved)
        with np.load(support_path) as saved:
            support = saved['coherent_support']
        observations = {**tracking, 'support': support}
        navigations = {label: navigate(observations, diagnostics=True,
                                       code_blend=blend)
                       for label, blend in BLENDS}
        with np.load(reference_path) as saved:
            assert_saved_navigation(navigations['.1'], saved)

        counts = quality_support_counts(tracking, support)
        time = tracking['time']
        available = tracking['available_time']
        post = available >= metadata['case']['attack_start']
        truth = np.column_stack((metadata['case']['velocity_east'] * time,
                                 np.zeros(len(time))))
        errors = {label: np.linalg.norm(nav['position'][:, :2] - truth, axis=1)
                  for label, nav in navigations.items()}
        common = post & np.isfinite(navigations['.1']['position']).all(axis=1) \
                      & np.isfinite(navigations['0']['position']).all(axis=1)
        scores = {}
        arrays = dict(observation_time=time, available_time=available,
                      common_published=common, support_count=counts)
        for label, navigation in navigations.items():
            scores[label] = score_mode(navigation, tracking, metadata['case'],
                                       counts, support)
            scores[label]['common_published'] = published_metrics(errors[label], common)
            arrays[f'{label}_horizontal_error_m'] = errors[label]
            for key, value in navigation.items():
                if key == 'diagnostics':
                    for diagnostic, data in value.items():
                        arrays[f'{label}_diagnostic_{diagnostic}'] = data
                else:
                    arrays[f'{label}_{key}'] = np.nan if value is None else value
        path = OUTPUT / f'{name}.npz'
        np.savez_compressed(path, **arrays)
        reports[name] = dict(tracking_sha256=digest(tracking_path),
                             support_sha256=digest(support_path),
                             reference_sha256=digest(reference_path),
                             metadata_sha256=digest(metadata_path),
                             result_sha256=digest(path), modes=scores)
        plots[name] = (available, errors)
        print(name, 'done', flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col',
                             constrained_layout=True)
    for axis, name in zip(axes.flat, CASES):
        available, errors = plots[name]
        for label, values in errors.items():
            axis.plot(available, values, label=f'code_blend={label}')
        axis.axhline(20., color='gray', linestyle=':', linewidth=1)
        axis.set(title=name, xlabel='Output available time (s)',
                 ylabel='Published error (m)')
        axis.grid(alpha=.3)
    axes[0, 0].legend()
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)
    assert source_hashes() == sources
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources,
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        figure_sha256=digest(OUTPUT / 'overview.png'), cases=reports)),
        indent=2, allow_nan=False), encoding='utf-8')
    return reports


if __name__ == '__main__':
    main()
