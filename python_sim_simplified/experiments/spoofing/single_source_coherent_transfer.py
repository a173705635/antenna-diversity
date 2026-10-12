"""Transfer the fixed coherent support candidate to one saved four-second RF.

This is a known difficult simulated IQ, with a different seed and geometry
from the two-second development case. It is not an independent holdout or a
same-seed attack/absence/control safety comparison.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from core.provenance import source_hashes
from core.serialization import serializable
from core.spoofing.single_source import frontend
from experiments.paths import SPOOFING_RESULTS as RESULTS
from experiments.spoofing.single_source_coherence_diagnosis import analyze_groups
from experiments.spoofing.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts, run_navigation_modes,
    score_mode,
)
from experiments.spoofing.single_source_evidence_history import analyze_trace
from experiments.spoofing.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


INPUT = RESULTS / 'weak-ten' / 'inputs' / '116921_weak_30_24_4'
OUTPUT = RESULTS / STUDY_OUTPUT / 'coherent_transfer'


def preflight_input(folder):
    """Check the saved RF and observation records against their metadata."""
    metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    hashes = {name: digest(folder / name)
              for name in ('samples.npy', 'observations.npz', 'metadata.json')}
    assert hashes['samples.npy'] == metadata['samples_sha256']
    assert hashes['observations.npz'] == metadata['observations_sha256']
    return metadata, hashes


def main():
    metadata, hashes = preflight_input(INPUT)
    sources = source_hashes()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'manifest.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources, input_sha256=hashes,
        samples_bytes=(INPUT / 'samples.npy').stat().st_size,
        antennas=ANTENNAS, beam_history_blocks=1, receiver_pll=RECEIVER_PLL,
        support_modes=('original', '20ms_noncoherent', '100ms_coherent'),
        numerical_threshold=3.)), indent=2, allow_nan=False), encoding='utf-8')

    with np.load(INPUT / 'observations.npz') as saved:
        record = dict(saved)
    record['samples'] = np.load(INPUT / 'samples.npy', mmap_mode='r')
    print('frontend start', flush=True)
    observations = frontend(record, ANTENNAS, split_support=True,
                            receiver_pll=RECEIVER_PLL, evidence_trace=True)
    trace_keys = [key for key in observations if key.startswith('trace_')]
    trace = {key: observations[key] for key in trace_keys}
    tracking = {key: value for key, value in observations.items()
                if key not in trace_keys}
    np.savez_compressed(OUTPUT / 'tracking.npz', **tracking)
    np.savez_compressed(OUTPUT / 'trace.npz', **trace)
    print('frontend saved', flush=True)

    history = analyze_trace(observations)
    np.savez_compressed(OUTPUT / 'history.npz', **history)
    groups = analyze_groups(trace)
    np.savez_compressed(OUTPUT / 'groups.npz', **groups)
    coherent = build_coherent_support(observations, groups)
    noncoherent = history['ratio'][:, :, 0]
    navigations = run_navigation_modes(observations, noncoherent,
                                       coherent['support'])

    supports = dict(original=observations['support'],
                    **{'20ms_noncoherent': noncoherent,
                       '100ms_coherent': coherent['support']})
    support_arrays = {f'coherent_{key}': value for key, value in coherent.items()}
    support_arrays['noncoherent_support'] = noncoherent
    navigation_arrays = {}
    scores = {}
    for mode, navigation in navigations.items():
        counts = quality_support_counts(observations, supports[mode])
        support_arrays[f'{mode}_support_count'] = counts
        scores[mode] = score_mode(navigation, observations, metadata['case'],
                                  counts, supports[mode])
        for key, value in navigation.items():
            if key == 'diagnostics':
                for diagnostic, data in value.items():
                    navigation_arrays[f'{mode}_diagnostic_{diagnostic}'] = data
            else:
                navigation_arrays[f'{mode}_{key}'] = np.nan if value is None else value
    np.savez_compressed(OUTPUT / 'support.npz', **support_arrays)
    np.savez_compressed(OUTPUT / 'navigation.npz', **navigation_arrays)
    print('navigation done', flush=True)

    time = observations['time']
    available = observations['available_time']
    truth = np.column_stack((metadata['case']['velocity_east'] * time,
                             np.zeros(len(time))))
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             constrained_layout=True)
    for mode, navigation in navigations.items():
        counts = support_arrays[f'{mode}_support_count']
        error = np.linalg.norm(navigation['position'][:, :2] - truth, axis=1)
        axes[0].plot(available, counts, label=mode)
        axes[1].plot(available, error, label=mode)
    axes[0].set(ylabel='Quality-gated support count')
    axes[1].set(xlabel='Output available time (s)', ylabel='Published error (m)')
    for axis in axes:
        axis.grid(alpha=.3)
    axes[0].legend()
    fig.savefig(OUTPUT / 'overview.png', dpi=150)
    plt.close(fig)

    assert source_hashes() == sources
    files = ('tracking.npz', 'trace.npz', 'history.npz', 'groups.npz',
             'support.npz', 'navigation.npz', 'overview.png')
    (OUTPUT / 'summary.json').write_text(json.dumps(serializable(dict(
        scope=__doc__, source_sha256=sources, input_sha256=hashes,
        manifest_sha256=digest(OUTPUT / 'manifest.json'),
        output_sha256={name: digest(OUTPUT / name) for name in files},
        case=metadata['case'], modes=scores)),
        indent=2, allow_nan=False), encoding='utf-8')
    return scores


if __name__ == '__main__':
    main()
