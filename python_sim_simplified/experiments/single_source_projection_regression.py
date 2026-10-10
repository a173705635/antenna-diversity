"""Two-gate replay of the frozen 8 s, 0 dB, 100 m absent-genuine IQ.

This is a historical regression input shared by both gates, not a new draw.
Truth is used only after receiver and independent IQ/beam auditing finish.
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from core.provenance import source_hashes
from core.serialization import serializable
from experiments.paths import RESULTS
from experiments.single_source_aided_coherent_rf import OUTPUT as STAGE28
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_projection_threshold import (
    OUTPUT as STAGE45, pair_score, run_path,
)
from experiments.single_source_ramp_transition import saved_record
from experiments.single_source_spoof_power import OUTPUT as STAGE26
from experiments.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS/STUDY_OUTPUT/'projection_threshold_regression'
SCENE = 'genuine_absent'
GATES = (3., 2.)
STAGE45_SUMMARY_SHA256 = '2314ace71e8e9fb5fe58bf1e58111ee05adb506f982708d31e9d60e30863c6ae'


def historical_screen(time, baseline, candidate, velocity_east):
    """Keep the relative absent-genuine rule and require absolute safety."""
    relative = pair_score(SCENE, time, baseline, candidate,
                          velocity_east, None, None)
    published = np.isfinite(candidate).all(axis=1)
    later = int(((time >= .5) & published).sum())
    serious = relative['candidate']['error_ge20']
    result = dict(relative=relative, candidate_after_half_published=later,
                  candidate_ge20=serious)
    result['pass'] = bool(relative['pass'] and later == 0 and serious == 0)
    return result


def preflight():
    """Bind this replay to the passed Stage 45 screen and exact old IQ."""
    stages = {name: json.loads((folder/'summary.json').read_text(encoding='utf-8'))
              for name, folder in (('stage45', STAGE45),
                                   ('stage26', STAGE26), ('stage28', STAGE28))}
    screen = stages['stage45']
    assert digest(STAGE45/'summary.json') == STAGE45_SUMMARY_SHA256
    assert screen['screen_pass']
    assert all(screen['cases'][scene]['short_screen']['pass']
               for scene in ('h0_constant', 'h0_drop', 'weak', 'genuine_absent'))
    assert len(screen['source_sha256']) == 96
    current = source_hashes()
    assert set(current)-set(screen['source_sha256']) == {
        'experiments/single_source_projection_regression.py'}
    assert {name: current[name] for name in screen['source_sha256']} == (
        screen['source_sha256'])
    for name, folder in (('stage45', STAGE45), ('stage26', STAGE26),
                         ('stage28', STAGE28)):
        summary = stages[name]
        assert summary['source_snapshot_sha256'] == digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip', summary['source_sha256'])
        assert summary['plan_sha256'] == digest(folder/'plan.json')
        assert summary['manifest_sha256'] == digest(folder/'manifest.json')
    old_input = stages['stage26']['cases'][SCENE]['input_sha256']
    assert stages['stage28']['cases'][SCENE]['input_sha256'] == old_input
    folder = STAGE26/SCENE/'input'
    for name, expected in old_input.items():
        assert digest(folder/name) == expected
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert metadata['samples_sha256'] == old_input['samples.npy']
    assert metadata['observations_sha256'] == old_input['observations.npz']
    case = metadata['case']
    assert (case['duration'], case['stars'], case['antennas'],
            case['spoof_db'], case['fake_east'], case['angle'],
            case['genuine'], case['attack_start'], case['velocity_east'],
            case['fake_rate']) == (8., 24, 4, 0., 100., 5., False, .3, 2., 0.)
    assert metadata['attacked']
    return stages, current, metadata, old_input


def plot_results(views, velocity_east):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for threshold in GATES:
        view = views[threshold]
        time = view['observation']
        truth = np.column_stack((velocity_east*time, np.zeros(len(time))))
        for mode, style in (('short', '-'), ('original', '--')):
            label = f'gate {threshold:g} {mode}'
            position = view['position'][mode]
            error = np.linalg.norm(position[:, :2]-truth, axis=1)
            axes[0].plot(time, view['support'][mode], linestyle=style,
                         linewidth=.7, label=label)
            axes[1].plot(time, error, linestyle=style, linewidth=.7,
                         marker='.', markersize=1, label=label)
    axes[0].set(ylabel='Qualified support count')
    axes[1].set(ylabel='Published horizontal error (m)')
    for axis in axes:
        axis.set(xlabel='Observation time (s)', xlim=(0., views[3.]['observation'][-1]))
        axis.grid(alpha=.25)
        axis.legend(fontsize=7)
    figure.savefig(OUTPUT/'overview.png', dpi=150)
    plt.close(figure)


def main():
    """Run the two current frontends on one immutable Stage 26 IQ record."""
    stages, sources, metadata, input_hashes = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip', sources)
    plan = dict(scope=__doc__, scene=SCENE, gates=GATES,
        input_reference=(STAGE26/SCENE/'input').relative_to(RESULTS.parent).as_posix(),
        input_sha256=input_hashes,
        receiver=dict(processed_antennas=ANTENNAS, receiver_pll=RECEIVER_PLL,
            beam_template_carrier_aiding=True, beam_coherent_ms=5,
            beam_startup_amplitude=True, fast_truncation=True,
            navigation_modes=('original', 'short')),
        hard_screen=('relative absent-genuine rule; gate2 observation >=0.5 s '
                     'zero publications and observation >=0.3 s zero errors >=20 m'),
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                     scipy=scipy.__version__), source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan), indent=2,
        allow_nan=False), encoding='utf-8')
    record, identity = saved_record(STAGE26/SCENE)
    assert identity == metadata
    paths = {}; views = {}
    for threshold in GATES:
        name = f'gate{threshold:g}'
        paths[name], views[threshold] = run_path(
            record, identity['case'], threshold, OUTPUT/name)
        print(name, 'RF, fit audit and navigation saved', flush=True)
    baseline = views[3.]
    candidate = views[2.]
    np.testing.assert_array_equal(baseline['observation'], candidate['observation'])
    np.testing.assert_array_equal(baseline['available'], candidate['available'])
    time = baseline['observation']
    assert int((time >= .3).sum()) == 769
    assert int((time >= .5).sum()) == 749
    velocity = identity['case']['velocity_east']
    short = historical_screen(time, baseline['position']['short'],
                              candidate['position']['short'], velocity)
    original = pair_score(SCENE, time, baseline['position']['original'],
        candidate['position']['original'], velocity, None, None)
    original.pop('pass')
    original['candidate_after_half_published'] = int(((time >= .5) &
        np.isfinite(candidate['position']['original']).all(axis=1)).sum())
    for mode, result in (('short', short), ('original', original)):
        scores = (short['relative'] if mode == 'short' else result)
        scores['baseline_initial_anchor_error_m'] = (
            baseline['navigation'][mode]['initial_horizontal_error_m'])
        scores['candidate_initial_anchor_error_m'] = (
            candidate['navigation'][mode]['initial_horizontal_error_m'])
    plot_results(views, velocity)
    assert source_hashes() == sources
    manifest = dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        reference_summary_sha256={name: digest(folder/'summary.json')
            for name, folder in (('stage45', STAGE45), ('stage26', STAGE26),
                                 ('stage28', STAGE28))},
        input_sha256=input_hashes,
        output_sha256={name: path['output_sha256'] for name, path in paths.items()},
        figure_sha256=digest(OUTPUT/'overview.png'))
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2, allow_nan=False), encoding='utf-8')
    summary = dict(scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'),
        figure_sha256=digest(OUTPUT/'overview.png'),
        input_sha256=input_hashes,
        short_screen=short, original_diagnostic=original,
        screen_pass=short['pass'], paths=paths)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
