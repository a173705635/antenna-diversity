"""Offline odd/even captured-PRN nuisance-direction diagnosis on frozen IQ.

All six fits finish before generator truth is read. No beam, receiver, or
navigation state is changed.
"""
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from core.recovery_geometry import FS
from core.single_source_scene import local_signal
from experiments.single_source_complex_response_audit import generator_truth
from experiments.single_source_startup_amplitude import (
    estimate_amplitude, startup_prompts,
)
from experiments.single_source_suppression_study import ANTENNAS, digest
from experiments.single_source_template_coherence import (
    WIDTH, aided_template, extract_startup, group_five_ms,
)
from tools import frozen_weak_h0_counterfactual as paired
from tools import oracle_phase_transport as oracle
from tools.resume_projection_long import read_npz


SEEDS = paired.SEEDS
SCENES = ('weak', 'h0_drop')
FIRST_BLOCK = round(.3 * FS) // WIDTH
LAST_BLOCK = round(20. * FS) // WIDTH
BLOCKS = LAST_BLOCK - FIRST_BLOCK
GAP_RATIO = 1e-10
FROZEN_HELPER_SHA256 = (
    '6c9dccd086ff5aa67f1824428b27b7fd815210bfbf016e7bd5decf18edbb4ea4'
)
PAIRED_SUMMARY_SHA256 = (
    'ecc3f7b4cefd9ea8d3fe3cb3a81e256f157ed57be237f898e5ffeacf8a38eb27'
)
OUTPUT = paired.stage47.OUTPUT / 'odd_even_prn_direction'


def fold_indices(prns):
    """Use actual one-based PRN parity, preserving captured order in each fold."""
    prns = np.asarray(prns)
    return (np.flatnonzero((prns + 1) % 2 == 1),
            np.flatnonzero((prns + 1) % 2 == 0))


def fit_direction(grouped, desired):
    """Original six-step fit, with eigenvalues retained at every update.

    The final vector is undefined when any covariance is non-identifying.
    A NaN final vector never enters a direction score.
    """
    antennas = len(grouped)
    values = np.full((7, 2), np.nan)
    nuisance = None
    remainder = grouped
    for step in range(7):
        eigen, vectors = np.linalg.eigh(remainder @ remainder.conj().T)
        values[step] = eigen[-1], eigen[-2]
        if eigen[-1] <= 0 or eigen[-1] - eigen[-2] <= GAP_RATIO * eigen[-1]:
            return np.full(antennas, np.nan + 0j), values, step
        nuisance = vectors[:, -1]
        if step == 6:
            break
        projector = np.eye(antennas) - np.outer(nuisance, nuisance.conj())
        projected = projector @ desired
        amplitude = np.sum(projected.conj() * grouped, axis=0) / (
            np.sum(desired.conj() * projected, axis=0).real + .01 * antennas)
        remainder = grouped - desired * amplitude
    return nuisance, values, -1


def fit_block(prompts, model, prns):
    """Split before 5 ms grouping; return two directions and seven spectra."""
    directions = np.full((2, len(model)), np.nan + 0j)
    spectra = np.full((2, 7, 2), np.nan)
    failure = np.full(2, -2, int)  # -2: insufficient PRNs; -1: valid; >=0: step
    fold_size = np.zeros(2, int)
    for fold, indices in enumerate(fold_indices(prns)):
        fold_size[fold] = len(indices)
        if len(indices) != 12:
            continue
        grouped = group_five_ms(prompts[:, indices, :])
        desired = np.repeat(model[:, prns[indices]], 4, axis=1)
        directions[fold], spectra[fold], failure[fold] = fit_direction(
            grouped, desired)
    return directions, spectra, failure, fold_size


def block_prompts(record, prns, aided, calibration, rate, index):
    """Repeat the original calibrated 20 ms prompt extraction without a gate."""
    start = index * WIDTH
    samples = np.arange(start, start + WIDTH)
    block = record['samples'][ANTENNAS, start:start + WIDTH] / (
        calibration[:, None] * np.exp(1j * rate[:, None] * samples[None, :] / FS))
    time = samples / FS
    local = np.stack([local_signal(record['codes'][star], template, time)
                      for star, template in zip(prns, aided)])
    milliseconds = WIDTH // 2046
    return np.einsum('mkn,skn->msk',
                     block.reshape(len(ANTENNAS), milliseconds, 2046),
                     local.conj().reshape(len(prns), milliseconds, 2046)) / 2046


def fit_scene(record, saved):
    """Fit every attack-period block using only saved observations and IQ."""
    tracking = read_npz(saved / 'tracking.npz')
    trace = read_npz(saved / 'trace.npz')
    prns, templates, calibration, rate = extract_startup(record)
    np.testing.assert_array_equal(prns, tracking['startup_prns'])
    np.testing.assert_allclose(calibration, tracking['calibration'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rate, tracking['calibration_rate'], rtol=0, atol=1e-12)
    nominal = record['nominal'][ANTENNAS]
    startup = startup_prompts(record, prns, templates, calibration, rate)
    amplitude, _ = estimate_amplitude(startup, nominal, prns)
    np.testing.assert_allclose(amplitude, tracking['startup_relative_amplitude'],
                               rtol=0, atol=1e-12)
    model = amplitude[:, None] * nominal
    np.testing.assert_allclose(model, trace['trace_nominal'], rtol=0, atol=1e-12)
    directions = np.full((BLOCKS, 2, len(ANTENNAS)), np.nan + 0j)
    spectra = np.full((BLOCKS, 2, 7, 2), np.nan)
    failures = np.empty((BLOCKS, 2), int)
    sizes = np.empty((BLOCKS, 2), int)
    aided = [aided_template(template) for template in templates]
    for offset, index in enumerate(range(FIRST_BLOCK, LAST_BLOCK)):
        prompts = block_prompts(record, prns, aided, calibration, rate, index)
        directions[offset], spectra[offset], failures[offset], sizes[offset] = (
            fit_block(prompts, model, prns))
    return dict(direction=directions, eigenvalues=spectra, failure_step=failures,
                fold_prns=sizes, captured_prns=prns, calibration=calibration,
                calibration_rate=rate)


def truth_labels(metadata, nominal, calibration, rate):
    """Construct counterfactual fake steering labels only after all fits."""
    physical, fake = generator_truth(metadata, nominal)
    midpoint = (np.arange(FIRST_BLOCK, LAST_BLOCK) + .5) * WIDTH / FS
    labels = (physical[ANTENNAS] / calibration)[:, None] * (
        fake[ANTENNAS, None] * np.exp(-1j * rate[:, None] * midpoint[None, :]))
    return (labels / np.linalg.norm(labels, axis=0)[None, :]).T


def score_scene(fit, labels):
    valid = np.all(fit['failure_step'] == -1, axis=1)
    direction = fit['direction']
    alignment = abs(np.einsum('bi,bi->b', direction[:, 0].conj(),
                              direction[:, 1])) ** 2
    truth_alignment = abs(np.einsum('bfi,bi->bf', direction.conj(), labels)) ** 2
    alignment[~valid] = np.nan
    truth_alignment[~valid] = np.nan
    return alignment, truth_alignment, valid


def score_pair(weak, h0):
    common = weak['valid'] & h0['valid']
    differences = np.column_stack((weak['alignment'] - h0['alignment'],
                                   weak['truth_alignment'] - h0['truth_alignment']))
    median = np.median(differences[common], axis=0).tolist() if np.any(common) else (
        [None, None, None])
    full_coverage = bool(np.all(weak['valid']) and np.all(h0['valid']))
    return dict(valid_weak=int(weak['valid'].sum()),
                valid_h0=int(h0['valid'].sum()), paired_valid=int(common.sum()),
                median_delta=median,
                pass_rule=bool(full_coverage and all(value > 0 for value in median)))


def preflight():
    frozen = paired.preflight()
    root = Path(__file__).resolve().parents[1]
    assert digest(root / 'tools/frozen_weak_h0_counterfactual.py') == FROZEN_HELPER_SHA256
    assert digest(paired.OUTPUT / 'summary.json') == PAIRED_SUMMARY_SHA256
    return frozen


def load_scene(seed, scene, frozen):
    source = paired.stage47.OUTPUT / str(seed) / scene / 'input'
    expected = frozen['cases'][str(seed)][scene]['input_sha256']
    assert {name: digest(source / name) for name in paired.INPUT_FILES} == expected
    record = read_npz(source / 'observations.npz')
    record['samples'] = np.load(source / 'samples.npy', mmap_mode='r')
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    saved = paired.stage47.OUTPUT / str(seed) / scene / 'gate3'
    outputs = frozen['cases'][str(seed)][scene]['paths']['gate3']['output_sha256']
    assert {name: digest(saved / name) for name in outputs} == outputs
    return record, metadata, saved, expected


def fit_all_scenes(frozen):
    """Finish the complete fixed matrix before any truth-label construction."""
    fits, labels_input, provenance = {}, {}, {}
    for seed in SEEDS:
        for scene in SCENES:
            record, metadata, saved, hashes = load_scene(seed, scene, frozen)
            key = (seed, scene)
            fits[key] = fit_scene(record, saved)
            labels_input[key] = (metadata, record['nominal'])
            provenance[f'{seed}/{scene}'] = dict(input_sha256=hashes,
                saved_output_sha256=frozen['cases'][str(seed)][scene]['paths'][
                    'gate3']['output_sha256'])
    return fits, labels_input, provenance


def score_all_scenes(fits, labels_input):
    scored = {}
    for key, fit in fits.items():
        metadata, nominal = labels_input[key]
        labels = truth_labels(metadata, nominal, fit['calibration'],
                              fit['calibration_rate'])
        alignment, truth_alignment, valid = score_scene(fit, labels)
        scored[key] = dict(alignment=alignment, truth_alignment=truth_alignment,
                           valid=valid)
    return scored


def main():
    frozen = preflight()
    fits, labels_input, provenance = fit_all_scenes(frozen)
    scored = score_all_scenes(fits, labels_input)
    results = {str(seed): score_pair(scored[seed, 'weak'],
                                    scored[seed, 'h0_drop']) for seed in SEEDS}
    assert not OUTPUT.exists(), 'Formal output already exists; do not overwrite'
    OUTPUT.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    snapshot = OUTPUT / 'source_snapshot.zip'
    with ZipFile(snapshot, 'x', compression=ZIP_STORED) as archive:
        name = 'tools/odd_even_prn_direction.py'
        archive.writestr(ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), source)
    outputs = {}
    for key, fit in fits.items():
        name = f'{key[0]}_{key[1]}.npz'
        path = OUTPUT / name
        np.savez_compressed(path, **fit, **scored[key])
        outputs[name] = digest(path)
    summary = dict(stage47_summary_sha256=oracle.STAGE47_SUMMARY_SHA256,
                   paired_summary_sha256=PAIRED_SUMMARY_SHA256,
                   source_snapshot_sha256=digest(snapshot),
                   rule_supported=all(result['pass_rule'] for result in results.values()),
                   results=results)
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2,
        allow_nan=False), encoding='utf-8')
    manifest = dict(stage47_summary_sha256=oracle.STAGE47_SUMMARY_SHA256,
                    paired_summary_sha256=PAIRED_SUMMARY_SHA256,
                    frozen_helper_sha256=FROZEN_HELPER_SHA256,
                    oracle_helper_snapshot_sha256={str(seed): json.loads((
                        paired.stage47.OUTPUT / 'oracle_phase_transport' / str(seed) /
                        'receipt.json').read_text(encoding='utf-8'))[
                            'source_snapshot_sha256'] for seed in SEEDS},
                    inputs=provenance, source_snapshot_sha256=digest(snapshot),
                    source_sha256=digest(Path(__file__)), output_sha256=outputs,
                    summary_sha256=digest(OUTPUT / 'summary.json'))
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2),
                                           encoding='utf-8')


if __name__ == '__main__':
    main()
