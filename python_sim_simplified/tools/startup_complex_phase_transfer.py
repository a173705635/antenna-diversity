"""Frozen-startup complex direction predicts same-boundary beam phase jumps.

Nine saved Stage 47 gate-3 records are read without new RF or navigation.
Generator truth is constructed only after prediction is complete.
"""
import json
from pathlib import Path

import numpy as np

from core.recovery_geometry import FS
from core.single_source_scene import local_signal
from experiments import single_source_projection_long as study
from experiments.single_source_complex_response_audit import generator_truth
from experiments.single_source_suppression_study import ANTENNAS, digest
from experiments.single_source_template_coherence import STARTUP, extract_startup
from tools.single_path_carrier_continuity import STAGE47_SUMMARY_SHA256


SCENES = ('h0_drop', 'weak', 'genuine_absent')
BLOCK_S = .02
FIRST_BOUNDARY_S = .3
REFERENCE_S = .05
EIGEN_RELATIVE_TOL = 1e-10
RESPONSE_ZERO = 1e-12
SOURCES = (
    'core/single_source.py', 'core/recovery_geometry.py',
    'core/array_acquisition.py', 'core/acquisition.py',
    'core/tracking.py', 'core/presets.py',
    'core/single_source_scene.py', 'core/array_processing.py',
    'experiments/single_source_template_coherence.py',
    'experiments/single_source_startup_amplitude.py',
    'experiments/single_source_complex_response_audit.py',
    'experiments/single_source_projection_long.py',
)


def reference_prompts(record, prns, templates, calibration, rate):
    """First 100 ms IQ in one fixed receiver coordinate at 0.05 s.

    Calibration rate is removed at REFERENCE_S only, not once per raw sample.
    The original frontend weights remain in their sample-wise calibrated frame.
    """
    raw = record['samples'][ANTENNAS, :STARTUP]
    scale = calibration[:, None]*np.exp(1j*rate[:, None]*REFERENCE_S)
    referenced = raw/scale
    sample_time = np.arange(STARTUP)/FS
    local = np.stack([local_signal(record['codes'][prn], template, sample_time)
                      for prn, template in zip(prns, templates)])
    milliseconds = STARTUP//2046
    return np.einsum('mkn,skn->msk',
                     referenced.reshape(len(ANTENNAS), milliseconds, 2046),
                     local.conj().reshape(len(prns), milliseconds, 2046))/2046


def startup_directions(prompts, stars, prns):
    """Principal fixed-reference prompt direction per acquired PRN, frozen."""
    antennas = prompts.shape[0]
    directions = np.full((antennas, stars), np.nan+0j)
    acquired = np.zeros(stars, bool)
    degenerate = np.zeros(stars, bool)
    eigenvalues = np.full((stars, antennas), np.nan)
    for column, prn in enumerate(prns):
        acquired[prn] = True
        p = prompts[:, column]
        covariance = p@p.conj().T/p.shape[1]
        values, vectors = np.linalg.eigh(covariance)
        eigenvalues[prn] = values
        degenerate[prn] = (values[-1]-values[-2]
                           <= EIGEN_RELATIVE_TOL*values[-1])
        if not degenerate[prn]:
            directions[:, prn] = vectors[:, -1]
    return directions, acquired, degenerate, eigenvalues


def direction_at_boundary(direction, rate, reference_s, boundary_s):
    """A fixed physical response acquires receiver de-rotation after t_ref."""
    return direction[None]*np.exp(
        -1j*rate[None, :, None]*(boundary_s-reference_s)[:, None, None])


def predict_phase(weights, direction, rate, acquired, degenerate,
                  boundary_indices):
    """Only startup observations and saved actual weights enter prediction."""
    boundary_s = boundary_indices*BLOCK_S
    u = direction_at_boundary(direction, rate, REFERENCE_S, boundary_s)
    before = np.einsum('bms,bms->bs', weights[boundary_indices-1].conj(), u)
    after = np.einsum('bms,bms->bs', weights[boundary_indices].conj(), u)
    missing = np.broadcast_to(~acquired, before.shape).copy()
    ambiguous = np.broadcast_to(acquired & degenerate, before.shape).copy()
    zero = (~missing & ~ambiguous
            & ((np.abs(before) <= RESPONSE_ZERO)
               | (np.abs(after) <= RESPONSE_ZERO)))
    available = ~missing & ~ambiguous & ~zero
    prediction = np.full(before.shape, np.nan)
    prediction[available] = np.angle(after[available]*before[available].conj())
    return dict(prediction=prediction, acquired=acquired,
                degenerate=degenerate, missing=missing,
                ambiguous=ambiguous, zero=zero, available=available,
                before_amplitude=np.abs(before), after_amplitude=np.abs(after),
                boundary_s=boundary_s)


def truth_after_prediction(weights, nominal, physical_gain, calibration, rate,
                           boundary_indices):
    """Counterfactual pure genuine component at each common boundary time."""
    boundary_s = boundary_indices*BLOCK_S
    genuine = nominal[np.array(ANTENNAS)]
    relative = (physical_gain[np.array(ANTENNAS)]/calibration)[None, :, None]
    spatial = relative*np.exp(-1j*boundary_s[:, None, None]*rate[None, :, None])
    spatial = spatial*genuine[None]
    before = np.einsum('bms,bms->bs', weights[boundary_indices-1].conj(),
                       spatial)
    after = np.einsum('bms,bms->bs', weights[boundary_indices].conj(), spatial)
    return np.angle(after*before.conj()), np.abs(before), np.abs(after)


def loss_summary(prediction, truth, available, subset):
    denominator = int(subset.sum())
    selected = available & subset
    count = int(selected.sum())
    if count == 0:
        return dict(opportunities=denominator, predicted=0,
                    prediction_loss=None, zero_jump_loss=None,
                    difference=None)
    target = truth[selected]
    predicted = float(np.mean(1-np.cos(prediction[selected]-target)))
    zero = float(np.mean(1-np.cos(target)))
    return dict(opportunities=denominator, predicted=count,
                prediction_loss=predicted, zero_jump_loss=zero,
                difference=predicted-zero)


def analyze_path(folder, expected, metadata_path, metadata_sha):
    inputs = folder/'input'
    gate = folder/'gate3'
    input_sha = {name: digest(inputs/name) for name in
                 ('samples.npy', 'observations.npz', 'metadata.json')}
    assert input_sha == expected['input_sha256']
    assert input_sha['metadata.json'] == metadata_sha
    output_sha = {name: digest(gate/name) for name in ('fit.npz', 'tracking.npz')}
    for name, sha in output_sha.items():
        assert sha == expected['paths']['gate3']['output_sha256'][name]
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    with np.load(inputs/'observations.npz') as saved:
        record = dict(saved)
    record['samples'] = np.load(inputs/'samples.npy', mmap_mode='r')
    with np.load(gate/'fit.npz') as saved:
        fit = dict(saved)
    with np.load(gate/'tracking.npz') as saved:
        tracking = dict(saved)
    prns, templates, calibration, rate = extract_startup(record)
    np.testing.assert_array_equal(prns, tracking['startup_prns'])
    np.testing.assert_allclose(calibration, tracking['calibration'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rate, tracking['calibration_rate'], rtol=0, atol=1e-12)
    prompts = reference_prompts(record, prns, templates, calibration, rate)
    direction, acquired, degenerate, eigenvalues = startup_directions(
        prompts, len(record['codes']), prns)
    boundary = np.flatnonzero(
        np.arange(len(fit['weight']))*BLOCK_S >= FIRST_BOUNDARY_S)
    predicted = predict_phase(fit['weight'], direction, rate,
                              acquired, degenerate, boundary)

    # No generator parameter reaches startup_directions or predict_phase.
    physical_gain, _ = generator_truth(metadata, record['nominal'])
    truth, true_before, true_after = truth_after_prediction(
        fit['weight'], record['nominal'], physical_gain, calibration, rate,
        boundary)
    truth_zero = ((true_before <= RESPONSE_ZERO)
                  | (true_after <= RESPONSE_ZERO))
    truth[truth_zero] = np.nan
    scorable = predicted['available'] & ~truth_zero
    full = np.ones_like(predicted['available'], bool)
    stable_gain = ((fit['gain'][boundary-1] >= .05)
                   & (fit['gain'][boundary] >= .05))
    all_score = loss_summary(predicted['prediction'], truth,
                             scorable, full)
    stable_score = loss_summary(predicted['prediction'], truth,
                                scorable, stable_gain)
    summary = dict(input_sha256=input_sha, output_sha256=output_sha,
                   startup_captured_prns=int(acquired.sum()),
                   boundary_count=len(boundary),
                   opportunity_count=int(full.sum()),
                   missing=int(predicted['missing'].sum()),
                   ambiguous=int(predicted['ambiguous'].sum()),
                   zero_response=int(predicted['zero'].sum()),
                   predictable=int(predicted['available'].sum()),
                   truth_zero_response=int(truth_zero.sum()),
                   all_prn=all_score, original_gain_ge_005=stable_score,
                   truth_after_attack_counterfactual=(metadata['scenario']
                                                       == 'genuine_absent'))
    arrays = dict(boundary_s=predicted['boundary_s'],
                  startup_prns=prns, startup_eigenvalues=eigenvalues,
                  startup_direction=direction,
                  prediction=predicted['prediction'], truth=truth,
                  missing=predicted['missing'], ambiguous=predicted['ambiguous'],
                  zero_response=predicted['zero'],
                  predictable=predicted['available'],
                  truth_zero_response=truth_zero, scorable=scorable,
                  original_gain_ge_005=stable_gain,
                  predicted_before_amplitude=predicted['before_amplitude'],
                  predicted_after_amplitude=predicted['after_amplitude'],
                  true_before_amplitude=true_before,
                  true_after_amplitude=true_after)
    return summary, arrays


def analyze():
    assert digest(study.OUTPUT/'summary.json') == STAGE47_SUMMARY_SHA256
    frozen = json.loads((study.OUTPUT/'summary.json').read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    for source in SOURCES:
        assert digest(root/source) == frozen['source_sha256'][source], source
    cases, arrays = {}, {}
    for seed in study.SEEDS:
        cases[str(seed)] = {}
        for scene in SCENES:
            folder = study.OUTPUT/str(seed)/scene
            expected = frozen['cases'][str(seed)][scene]
            result, detail = analyze_path(
                folder, expected, folder/'input'/'metadata.json',
                expected['input_sha256']['metadata.json'])
            cases[str(seed)][scene] = result
            arrays[f'{seed}_{scene}'] = detail
    def valid(scene, relation):
        values = [cases[str(seed)][scene][relation] for seed in study.SEEDS]
        if scene == 'weak':
            return all(item['predicted'] > 0 and item['difference'] < 0
                       for item in values)
        return all(item['predicted'] > 0 and item['difference'] <= 0
                   for item in values)
    supported = all(valid(scene, relation)
                    for scene in ('weak', 'h0_drop')
                    for relation in ('all_prn', 'original_gain_ge_005'))
    return dict(scope=__doc__, source_sha256=digest(Path(__file__)),
                stage47_summary_sha256=STAGE47_SUMMARY_SHA256,
                frozen_source_sha256={name: frozen['source_sha256'][name]
                                      for name in SOURCES},
                fixed_support_rule_passed=bool(supported), cases=cases), arrays


def main():
    summary, arrays = analyze()
    output = study.OUTPUT
    summary_path = output/'startup_complex_phase_transfer_summary.json'
    manifest_path = output/'startup_complex_phase_transfer_manifest.json'
    assert not summary_path.exists() and not manifest_path.exists()
    for key in arrays:
        assert not (output/f'startup_complex_phase_transfer_{key}.npz').exists()
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False),
                            encoding='utf-8')
    hashes = {summary_path.name: digest(summary_path)}
    for key, detail in arrays.items():
        path = output/f'startup_complex_phase_transfer_{key}.npz'
        np.savez_compressed(path, **detail)
        hashes[path.name] = digest(path)
    manifest_path.write_text(json.dumps(hashes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
