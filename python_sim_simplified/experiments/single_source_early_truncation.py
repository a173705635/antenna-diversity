"""Stage 43: saved-IQ replay of optional early receive-track truncation.

Only the receive tracking stop rule changes. The three inputs and their old
scientific outputs are read from the frozen Stage 42/41 experiments.
"""
import gc
import json
from time import perf_counter

import numpy as np

from core.provenance import source_hashes
from core.recovery_geometry import FS
from core.serialization import serializable
from core.single_source import LOOP_HISTORY_FIELDS, frontend
from experiments.paths import RESULTS
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import build_coherent_support
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_long_coherent import source_snapshot
from experiments.single_source_phase_exclusion import verify_source_snapshot
from experiments.single_source_ramp_transition import (
    OUTPUT as STAGE42, saved_record, save_loops,
)
from experiments.single_source_startup_amplitude import load_record, source_folder
from experiments.single_source_startup_amplitude_rf import (
    OUTPUT as STAGE41, replay_navigation,
)
from experiments.single_source_suppression_study import (
    ANTENNAS, OUTPUT as STUDY_OUTPUT, RECEIVER_PLL, digest,
)


OUTPUT = RESULTS/STUDY_OUTPUT/'early_truncation'
CASES = (
    ('weak_off', STAGE42/'weak', STAGE42/'weak'/'off', False),
    ('genuine_absent_on', STAGE42/'genuine_absent',
     STAGE42/'genuine_absent'/'on', True),
    ('h0_114001_on', source_folder('h0_114001'),
     STAGE41/'h0_114001'/'on', True),
)
SCIENTIFIC = ('tracking.npz', 'trace.npz', 'history.npz', 'groups.npz',
              'support.npz', 'navigation.npz')


def compare_arrays(actual, historical):
    """Compare values, dtype, shape and NaN placement for every saved field."""
    assert set(actual) == set(historical)
    for name in actual:
        assert actual[name].dtype == historical[name].dtype, name
        np.testing.assert_array_equal(actual[name], historical[name], err_msg=name)


def saved_arrays(path):
    with np.load(path) as saved:
        return dict(saved)


def verify_inputs():
    historical = {
        'stage42': (STAGE42, json.loads((STAGE42/'summary.json').read_text(
            encoding='utf-8'))),
        'stage41': (STAGE41, json.loads((STAGE41/'summary.json').read_text(
            encoding='utf-8'))),
    }
    current = source_hashes()
    for folder, summary in historical.values():
        assert summary['source_snapshot_sha256'] == digest(folder/'source_snapshot.zip')
        verify_source_snapshot(folder/'source_snapshot.zip', summary['source_sha256'])
        for key, name in (('plan_sha256', 'plan.json'),
                          ('manifest_sha256', 'manifest.json'),
                          ('figure_sha256', 'overview.png')):
            assert summary[key] == digest(folder/name)
    stage42 = historical['stage42'][1]
    for name, sha in stage42['source_sha256'].items():
        if name not in ('core/tracking.py', 'core/single_source.py'):
            assert current[name] == sha, name
    for name, input_folder, reference, _ in CASES:
        if name == 'h0_114001_on':
            expected = historical['stage41'][1]['cases']['h0_114001']['input_sha256']
            reference_hashes = historical['stage41'][1]['cases']['h0_114001'][
                'paths']['on']['output_sha256']
        else:
            scene = 'weak' if name == 'weak_off' else 'genuine_absent'
            expected = stage42['cases'][scene]['input_sha256']
            input_folder = input_folder/'input'
            path = 'off' if name == 'weak_off' else 'on'
            reference_hashes = stage42['cases'][scene]['paths'][path][
                'output_sha256']
        for file, sha in expected.items():
            assert digest(input_folder/file) == sha
        for file in SCIENTIFIC:
            assert digest(reference/file) == reference_hashes[file]
        if 'loops.npz' in reference_hashes:
            assert digest(reference/'loops.npz') == reference_hashes['loops.npz']
    return historical, current


def near_tail_energy_check(observations, history):
    """Rebuild the last eligible history without the five future output rows."""
    count = len(observations['time'])-5
    rows = observations['trace_window_k'] < count
    prefix = {}
    trace_fields = ('trace_window_k', 'trace_star', 'trace_arc',
                    'trace_start_sample', 'trace_available_time', 'trace_prompts')
    for name, value in observations.items():
        if name in trace_fields:
            prefix[name] = value[rows]
        elif (isinstance(value, np.ndarray) and value.ndim
              and len(value) == len(observations['time'])
              and not name.startswith('trace_')):
            prefix[name] = value[:count]
        else:
            prefix[name] = value
    rebuilt = analyze_trace(prefix)
    for name, value in rebuilt.items():
        np.testing.assert_array_equal(value, history[name][:count], err_msg=name)
    return count


def run_case(name, record, identity, reference, amplitude, output):
    output.mkdir(exist_ok=True)
    scratch = output/'beam_stream.dat'
    loops = []
    started = perf_counter()
    observations = frontend(
        record, ANTENNAS, split_support=True, receiver_pll=RECEIVER_PLL,
        evidence_trace=True, temporal_beam_trigger_enabled=False,
        beam_template_carrier_aiding=True, beam_coherent_ms=5,
        beam_startup_amplitude=amplitude, stream_memmap_path=scratch,
        loop_history=loops, fast_truncation=True)
    elapsed_s = perf_counter()-started
    assert scratch.stat().st_size == len(record['codes'])*record['samples'].shape[1]*16
    scratch.unlink()
    tracking = {key: value for key, value in observations.items()
                if not key.startswith('trace_')}
    trace = {key: value for key, value in observations.items()
             if key.startswith('trace_')}
    np.savez_compressed(output/'tracking.npz', **tracking)
    np.savez_compressed(output/'trace.npz', **trace)
    for file, arrays in (('tracking.npz', tracking), ('trace.npz', trace)):
        compare_arrays(arrays, saved_arrays(reference/file))

    loop_counts = save_loops(output, loops)
    if (reference/'loops.npz').exists():
        old_loops = saved_arrays(reference/'loops.npz')
        compare_arrays(saved_arrays(output/'loops.npz'), old_loops)
        assert set(LOOP_HISTORY_FIELDS) <= {
            key.split('_', 2)[-1] for key in old_loops}
    accepted_epochs = loop_counts['accepted_epochs']
    old_full_epoch_lower_bound = 0
    for segment in loops:
        end = int(segment['epoch_start_samples'][-1])+2046
        old_full_epoch_lower_bound += (record['samples'].shape[1]-end)//2046 + len(
            segment['epoch_start_samples'])

    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    support = {f'coherent_{key}': value for key, value in coherent.items()}
    for file, arrays in (('history.npz', history), ('groups.npz', groups),
                         ('support.npz', support)):
        np.savez_compressed(output/file, **arrays)
        compare_arrays(arrays, saved_arrays(reference/file))
    near_tail = near_tail_energy_check(observations, history)
    replay_navigation(tracking, trace, coherent, identity['case'], output)
    compare_arrays(saved_arrays(output/'navigation.npz'),
                   saved_arrays(reference/'navigation.npz'))
    del observations, tracking, trace, history, groups, coherent, loops
    gc.collect()
    return dict(frontend_walltime_s=elapsed_s, accepted_epochs=accepted_epochs,
                old_full_epoch_lower_bound=old_full_epoch_lower_bound,
                near_tail_recomputed_rows=near_tail,
                output_sha256={file: digest(output/file)
                               for file in (*SCIENTIFIC, 'loops.npz')})


def main():
    historical, sources = verify_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_snapshot(OUTPUT/'source_snapshot.zip', sources)
    plan = dict(scope=__doc__, cases=[name for name, *_ in CASES],
                input='saved Stage 42/41 IQ', fast_truncation=True,
                loss_confirm_epochs=5, loss_settle_epochs=100,
                coherent_cn0_window_epochs=100, source_sha256=sources)
    (OUTPUT/'plan.json').write_text(json.dumps(serializable(plan), indent=2,
        allow_nan=False), encoding='utf-8')
    manifest = dict(source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        historical_summary_sha256={name: digest(folder/'summary.json')
                                   for name, (folder, _) in historical.items()})
    (OUTPUT/'manifest.json').write_text(json.dumps(serializable(manifest),
        indent=2, allow_nan=False), encoding='utf-8')
    reports = {}
    for name, input_folder, reference, amplitude in CASES:
        if name == 'h0_114001_on':
            record, identity = load_record('h0_114001')
        else:
            record, identity = saved_record(input_folder)
        result = run_case(name, record, identity, reference, amplitude,
                          OUTPUT/name)
        input_summary = historical['stage41' if name == 'h0_114001_on'
                                   else 'stage42'][1]
        input_case = ('h0_114001' if name == 'h0_114001_on' else
                      'weak' if name == 'weak_off' else 'genuine_absent')
        reports[name] = dict(input_sha256=input_summary['cases'][input_case]
                             ['input_sha256'], result=result)
        del record
        gc.collect()
        print(name, 'scientific arrays matched', flush=True)
    assert source_hashes() == sources
    summary = dict(scope=__doc__, source_sha256=sources,
        source_snapshot_sha256=digest(OUTPUT/'source_snapshot.zip'),
        plan_sha256=digest(OUTPUT/'plan.json'),
        manifest_sha256=digest(OUTPUT/'manifest.json'), cases=reports)
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(summary),
        indent=2, allow_nan=False), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
