"""Fixed odd/even cross-fit beam candidate on frozen Stage 47 IQ.

Each private frontend uses one training-fold projector for every PRN and
receives only the opposite fold. Evidence is computed per pass, then the
original short navigation sees one PRN-wise merged observation table.
"""
import json
from pathlib import Path
from types import FunctionType
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from core.recovery_geometry import FS
from core.single_source import beam, frontend, navigate, receive_measured
from experiments.single_source_coherence_diagnosis import analyze_groups
from experiments.single_source_coherent_navigation import (
    build_coherent_support, quality_support_counts,
)
from experiments.single_source_evidence_history import analyze_trace
from experiments.single_source_phase_state_gate import projection_eligibility
from experiments.single_source_projection_long import long_screen
from experiments.single_source_projection_threshold import independent_fit
from experiments.single_source_ramp_transition import save_loops
from experiments.single_source_startup_amplitude_rf import (
    check_prefix, score_navigation,
)
from experiments.single_source_suppression_study import ANTENNAS, digest
from experiments.single_source_temporal_trigger import navigation_input
from experiments.single_source_template_coherence import WIDTH, group_five_ms
from tools import odd_even_prn_direction as direction
from tools import oracle_phase_transport as oracle
from tools import frozen_weak_h0_counterfactual as paired
from tools.resume_projection_long import (
    check_saved_arrays, check_saved_navigation, read_npz,
)


SEEDS = paired.SEEDS
SCENES = ('h0_drop', 'weak', 'genuine_absent')
START_BLOCK = round(.1 * FS) // WIDTH
BLOCKS = round(20. * FS) // WIDTH
OUTPUT = paired.stage47.OUTPUT / 'cross_fold_prn_navigation'
DIRECTION_SUMMARY_SHA256 = (
    '6135372d8fa53770d8770d5960db9ae78d46ef00c981c89f0ab4ed1718873bbe'
)
DIRECTION_SOURCE_SHA256 = (
    'c899e91536544d9632d6fcf6da70c36c8e7ac93659ff590e52b6095a8e8e2c0e'
)
PASS_FILES = ('tracking.npz', 'trace.npz', 'loops.npz', 'history.npz',
              'groups.npz', 'support.npz', 'fold_fit.npz')
STAR_FIELDS = ('ranges', 'lock', 'cn0', 'carrier', 'gain', 'support', 'arc',
               'support_split', 'split_on_energy', 'split_off_energy')


def fold_beam(block, model, local, prns, train_parity, index, saved_raw):
    """Keep the original all-PRN raw gate; cross-fit only its active weights."""
    original_weight, original_gain, raw = beam(
        block, model, local, None, prns, coherent_ms=5,
        beam_projection_threshold=3.)
    np.testing.assert_allclose(raw, saved_raw, rtol=0, atol=1e-9,
                               err_msg=f'full-PRN raw block {index}')
    antennas = len(block)
    milliseconds = block.shape[1] // 2046
    prompts = np.einsum('mkn,skn->msk',
        block.reshape(antennas, milliseconds, 2046),
        local.conj().reshape(len(prns), milliseconds, 2046)) / 2046
    selected = np.flatnonzero((np.asarray(prns) + 1) % 2 == train_parity)
    eigen = np.full((7, 2), np.nan)
    failure = -2
    if len(selected) == 12:
        grouped = group_five_ms(prompts[:, selected, :])
        desired = np.repeat(model[:, np.asarray(prns)[selected]], 4, axis=1)
        nuisance, eigen, failure = direction.fit_direction(grouped, desired)
    if raw < 3.:
        return original_weight, original_gain, raw, eigen, failure, None
    if failure != -1:
        raise RuntimeError(f'active fold direction undefined: block={index}, '
                           f'parity={train_parity}, step={failure}')
    projector = np.eye(antennas) - np.outer(nuisance, nuisance.conj())
    projected = projector @ model
    response = np.sum(model.conj() * projected, axis=0).real
    weight = projected / (response + .01)[None, :]
    return weight, response / antennas, raw, eigen, failure, projector


def private_frontend(record, saved_fit, train_parity, scratch, loops):
    """Change only private beam and receive globals, not shared core state."""
    next_block = [START_BLOCK]
    eigenvalues = np.full((BLOCKS, 7, 2), np.nan)
    failure_steps = np.full(BLOCKS, -3, int)  # -3: original startup block
    projectors = np.tile(np.eye(len(ANTENNAS)), (BLOCKS, 1, 1)).astype(complex)
    receive_calls = []

    def cross_beam(block, model, local, startup_cov, prns,
                   trigger_history=None, coherent_ms=5,
                   beam_projection_threshold=3.):
        assert trigger_history is None and coherent_ms == 5
        assert beam_projection_threshold == 3.
        index = next_block[0]
        next_block[0] += 1
        weight, gain, raw, eigen, failure, projector = fold_beam(
            block, model, local, prns, train_parity, index,
            saved_fit['raw'][index])
        eigenvalues[index] = eigen
        failure_steps[index] = failure
        if projector is not None:
            projectors[index] = projector
        return weight, gain, raw

    def opposite_receive(signal, prn, acquisition, tracking, **kwargs):
        receive_calls.append(prn)
        if prn % 2 == train_parity:
            return {'segments': []}
        return receive_measured(signal, prn, acquisition, tracking, **kwargs)

    private_globals = dict(frontend.__globals__)
    private_globals.update(beam=cross_beam, receive_measured=opposite_receive)
    receiver = FunctionType(frontend.__code__, private_globals,
                            frontend.__name__, frontend.__defaults__,
                            frontend.__closure__)
    observations = oracle.run_frontend(receiver, record, scratch, loops)
    assert next_block[0] == BLOCKS
    assert receive_calls == list(range(1, len(record['codes']) + 1))
    tracking, trace = oracle.split_observations(observations)
    assert np.all(trace['trace_star'] % 2 != (1 - train_parity))
    np.testing.assert_allclose(trace['trace_raw_trigger'][START_BLOCK:],
                               saved_fit['raw'][START_BLOCK:],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(trace['trace_projectors'], projectors,
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(trace['trace_dimensions'],
        np.where(saved_fit['raw'] >= 3., len(ANTENNAS) - 1,
                 len(ANTENNAS)))
    return observations, dict(eigenvalues=eigenvalues,
                              failure_step=failure_steps,
                              projector=projectors)


def merge_star_fields(odd, even):
    """Merge only per-PRN fields; shared axes must be exactly identical."""
    assert odd.keys() == even.keys()
    stars = odd['ranges'].shape[1]
    odd_stars = np.arange(stars) % 2 == 0
    merged = {}
    for key in odd:
        if key in STAR_FIELDS:
            merged[key] = np.where(odd_stars[None, :], even[key], odd[key])
        else:
            np.testing.assert_array_equal(odd[key], even[key], err_msg=key)
            merged[key] = odd[key]
    return merged


def merge_support(odd, even):
    assert odd.keys() == even.keys()
    stars = odd['support'].shape[1]
    odd_stars = np.arange(stars) % 2 == 0
    return {key: np.where(odd_stars[None, :], even[key], odd[key])
            for key in odd}


def verify_pass_alignment(odd_trace, even_trace, saved_fit, available_time):
    """Two separate projectors may share eligibility only with equal gates."""
    np.testing.assert_array_equal(odd_trace['trace_raw_trigger'],
                                  even_trace['trace_raw_trigger'])
    np.testing.assert_array_equal(odd_trace['trace_dimensions'],
                                  even_trace['trace_dimensions'])
    np.testing.assert_allclose(odd_trace['trace_raw_trigger'][START_BLOCK:],
                               saved_fit['raw'][START_BLOCK:], rtol=0,
                               atol=1e-9)
    odd_eligible, latest, recent = projection_eligibility(available_time,
                                                            odd_trace)
    even_eligible, _, _ = projection_eligibility(available_time, even_trace)
    np.testing.assert_array_equal(odd_eligible, even_eligible)
    return odd_eligible, latest, recent


def preflight():
    frozen = direction.preflight()
    root = Path(__file__).resolve().parents[1]
    assert digest(direction.OUTPUT / 'summary.json') == DIRECTION_SUMMARY_SHA256
    assert digest(root / 'tools/odd_even_prn_direction.py') == DIRECTION_SOURCE_SHA256
    assert digest(direction.OUTPUT / 'source_snapshot.zip') == json.loads((
        direction.OUTPUT / 'summary.json').read_text(encoding='utf-8'))[
            'source_snapshot_sha256']
    with ZipFile(direction.OUTPUT / 'source_snapshot.zip') as source:
        assert source.read('tools/odd_even_prn_direction.py') == (
            root / 'tools/odd_even_prn_direction.py').read_bytes()
    return frozen


def snapshot_source():
    path = OUTPUT / 'source_snapshot.zip'
    root = Path(__file__).resolve().parents[1]
    names = ('tools/cross_fold_prn_navigation.py',
             'tools/odd_even_prn_direction.py')
    if path.exists():
        with ZipFile(path) as archive:
            assert set(archive.namelist()) == set(names)
            for name in names:
                assert archive.read(name) == (root / name).read_bytes()
    else:
        with ZipFile(path, 'x', compression=ZIP_STORED) as archive:
            for name in names:
                archive.writestr(ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)),
                                 (root / name).read_bytes())
    return digest(path)


def load_scene(seed, scene, frozen):
    source = paired.stage47.OUTPUT / str(seed) / scene / 'input'
    hashes = frozen['cases'][str(seed)][scene]['input_sha256']
    assert {name: digest(source / name) for name in paired.INPUT_FILES} == hashes
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    record = read_npz(source / 'observations.npz')
    record['samples'] = np.load(source / 'samples.npy', mmap_mode='r')
    saved = paired.stage47.OUTPUT / str(seed) / scene / 'gate3'
    outputs = frozen['cases'][str(seed)][scene]['paths']['gate3']['output_sha256']
    assert {name: digest(saved / name) for name in outputs} == outputs
    return record, metadata, saved, hashes, outputs


def verified_baseline(saved, metadata):
    """Recompute frozen evidence and both saved navigations, without RF replay."""
    tracking = read_npz(saved / 'tracking.npz')
    trace = read_npz(saved / 'trace.npz')
    coherent = check_saved_arrays(saved, tracking, trace)
    scores, positions, _ = check_saved_navigation(
        saved, tracking, trace, coherent, metadata['case'])
    return scores['short'], positions['short']


def verify_completed_pass(folder, train_parity, input_hashes,
                          saved_fit_sha256, source_sha):
    """Bind a resumed pass to its parity and every actual saved output byte."""
    receipt = json.loads((folder / 'receipt.json').read_text(encoding='utf-8'))
    assert receipt['train_parity'] == train_parity
    assert receipt['source_sha256'] == source_sha
    assert receipt['source_snapshot_sha256'] == digest(
        OUTPUT / 'source_snapshot.zip')
    assert receipt['input_sha256'] == input_hashes
    assert receipt['saved_fit_sha256'] == saved_fit_sha256
    assert {name: digest(folder / name) for name in PASS_FILES} == (
        receipt['output_sha256'])
    return receipt


def run_pass(record, saved_fit, train_parity, folder, input_hashes,
             source_sha):
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        verify_completed_pass(folder, train_parity, input_hashes,
                              saved_fit['sha256'], source_sha)
        return (read_npz(folder / 'tracking.npz'),
                read_npz(folder / 'trace.npz'),
                {key.removeprefix('coherent_'): value for key, value in
                 read_npz(folder / 'support.npz').items()})
    assert not folder.exists(), f'incomplete pass retained: {folder}'
    folder.mkdir(parents=True)
    scratch = folder / 'beam_stream.dat'
    loops = []
    observations, fold_fit = private_frontend(
        record, saved_fit, train_parity, scratch, loops)
    tracking, trace = oracle.split_observations(observations)
    for name, arrays in (('tracking.npz', tracking), ('trace.npz', trace),
                         ('fold_fit.npz', fold_fit)):
        np.savez_compressed(folder / name, **arrays)
    save_loops(folder, loops)
    history = analyze_trace(observations)
    groups = analyze_groups(trace)
    coherent = build_coherent_support(tracking, groups)
    for name, arrays in (('history.npz', history), ('groups.npz', groups),
                         ('support.npz', {f'coherent_{key}': value for key,
                                          value in coherent.items()})):
        np.savez_compressed(folder / name, **arrays)
    scratch.unlink()
    receipt = dict(train_parity=train_parity, source_sha256=source_sha,
                   source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
                   input_sha256=input_hashes,
                   saved_fit_sha256=saved_fit['sha256'],
                   output_sha256={name: digest(folder / name) for name in PASS_FILES})
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return tracking, trace, coherent


def replay_merged_short(tracking, support, eligible, latest, recent, folder):
    inputs = navigation_input(tracking, support['support'])
    navigation = navigate(inputs, diagnostics=True, code_blend=0.,
                          phase_leave_one_out=True,
                          phase_state_continuity=True,
                          phase_state_eligible=eligible)
    check_prefix(tracking, support['support'], eligible, navigation, 'short')
    counts = quality_support_counts(tracking, support['support'])
    arrays = dict(observation_time=tracking['time'],
                  available_time=tracking['available_time'],
                  phase_state_eligible=eligible,
                  latest_complete_block=latest,
                  recent_block_dimensions=recent,
                  short_support_count=counts)
    for key, value in navigation.items():
        if key == 'diagnostics':
            arrays.update({f'short_diagnostic_{name}': data
                           for name, data in value.items()})
        else:
            arrays[f'short_{key}'] = np.nan if value is None else value
    np.savez_compressed(folder / 'navigation.npz', **arrays)
    return navigation


def run_path(seed, scene, frozen, source_sha):
    record, metadata, saved, inputs, saved_outputs = load_scene(seed, scene, frozen)
    baseline_score, baseline_position = verified_baseline(saved, metadata)
    folder = OUTPUT / str(seed) / scene
    receipt_path = folder / 'receipt.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        assert receipt['source_sha256'] == source_sha
        assert receipt['source_snapshot_sha256'] == digest(
            OUTPUT / 'source_snapshot.zip')
        assert receipt['input_sha256'] == inputs
        assert receipt['saved_output_sha256'] == saved_outputs
        for name, parity in (('train_odd', 1), ('train_even', 0)):
            verify_completed_pass(folder / name, parity, inputs,
                                  digest(saved / 'fit.npz'), source_sha)
        assert receipt['pass_receipt_sha256'] == {name: digest(
            folder / name / 'receipt.json') for name in ('train_odd', 'train_even')}
        assert {name: digest(folder / name) for name in ('navigation.npz',
            'merged_tracking.npz', 'merged_support.npz')} == (
                receipt['output_sha256'])
        return receipt['score']
    folder.mkdir(parents=True, exist_ok=True)
    fit = read_npz(saved / 'fit.npz')
    fit['sha256'] = digest(saved / 'fit.npz')
    odd = run_pass(record, fit, 1, folder / 'train_odd', inputs, source_sha)
    even = run_pass(record, fit, 0, folder / 'train_even', inputs, source_sha)
    odd_tracking, odd_trace, odd_support = odd
    even_tracking, even_trace, even_support = even
    tracking = merge_star_fields(odd_tracking, even_tracking)
    support = merge_support(odd_support, even_support)
    eligible, latest, recent = verify_pass_alignment(
        odd_trace, even_trace, fit, tracking['available_time'])
    assert not any((folder / name).exists() for name in (
        'merged_tracking.npz', 'merged_support.npz', 'navigation.npz')), (
            f'incomplete merged path retained: {folder}')
    np.savez_compressed(folder / 'merged_tracking.npz', **tracking)
    np.savez_compressed(folder / 'merged_support.npz', **support)
    navigation = replay_merged_short(tracking, support, eligible, latest,
                                     recent, folder)
    candidate_score = score_navigation(
        tracking, navigation, metadata['case']['velocity_east'])
    score = long_screen(scene, tracking['time'], baseline_position,
                        navigation['position'], metadata['case']['velocity_east'],
                        baseline_score['max_relative_motion_error_m'],
                        candidate_score['max_relative_motion_error_m'])
    receipt = dict(seed=seed, scene=scene, source_sha256=source_sha,
                   source_snapshot_sha256=digest(OUTPUT / 'source_snapshot.zip'),
                   stage47_summary_sha256=oracle.STAGE47_SUMMARY_SHA256,
                   direction_summary_sha256=DIRECTION_SUMMARY_SHA256,
                   input_sha256=inputs, saved_output_sha256=saved_outputs,
                   pass_receipt_sha256={name: digest(folder / name / 'receipt.json')
                                        for name in ('train_odd', 'train_even')},
                   output_sha256={name: digest(folder / name) for name in (
                       'navigation.npz', 'merged_tracking.npz',
                       'merged_support.npz')},
                   baseline_short=baseline_score, candidate_short=candidate_score,
                   score=score)
    receipt_path.write_text(json.dumps(receipt, indent=2,
                                       allow_nan=False), encoding='utf-8')
    return score


def main():
    frozen = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_sha = digest(Path(__file__))
    snapshot_sha = snapshot_source()
    results = {}
    for seed in SEEDS:
        for scene in SCENES:
            result = run_path(seed, scene, frozen, source_sha)
            results[f'{seed}/{scene}'] = result
            print(seed, scene, result['pass'], flush=True)
    summary = dict(stage47_summary_sha256=oracle.STAGE47_SUMMARY_SHA256,
                   direction_summary_sha256=DIRECTION_SUMMARY_SHA256,
                   source_snapshot_sha256=snapshot_sha,
                   all_pass=all(result['pass'] for result in results.values()),
                   results=results)
    assert not (OUTPUT / 'summary.json').exists()
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2,
        allow_nan=False), encoding='utf-8')
    manifest = dict(source_sha256=source_sha,
                    source_snapshot_sha256=snapshot_sha,
                    summary_sha256=digest(OUTPUT / 'summary.json'),
                    receipt_sha256={name: digest(OUTPUT / name / 'receipt.json')
                                    for name in results})
    assert not (OUTPUT / 'manifest.json').exists()
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2),
                                           encoding='utf-8')


if __name__ == '__main__':
    main()
