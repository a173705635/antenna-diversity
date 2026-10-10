"""One shared startup residual phase estimate for saved beam weights.

This is an offline phase-transfer diagnosis, not another suppression or
navigation mechanism. Truth is constructed only after all predictions.
"""
import json
from pathlib import Path

import numpy as np

from experiments import single_source_projection_long as study
from experiments.single_source_complex_response_audit import generator_truth
from experiments.single_source_suppression_study import ANTENNAS, digest
from experiments.single_source_template_coherence import extract_startup
from tools import startup_complex_phase_transfer as frozen_phase


STAGE47_SUMMARY_SHA256 = frozen_phase.STAGE47_SUMMARY_SHA256
FROZEN_PHASE_SUMMARY_SHA256 = 'f30cb1f5032aef51786d522d751df774ef43c4b22249bc46865474b250f468a3'
SCENES = frozen_phase.SCENES
SOURCES = frozen_phase.SOURCES


def estimate_common_gain(prompts, nominal, prns, amplitude):
    """Equal-PRN covariance after nominal removal; one phase per antenna."""
    q = prompts/nominal[:, prns, None]
    covariance = np.einsum('ism,jsm->ij', q, q.conj()) / (len(prns)*q.shape[2])
    crosses = np.array([covariance[1, 2], covariance[0, 2],
                        covariance[2, 1], covariance[0, 1]])
    threshold = np.finfo(float).eps*np.max(np.abs(covariance))
    identifiable = bool(np.all(np.abs(crosses) > threshold))
    phase = np.full(3, np.nan)
    gain = np.full(nominal.shape, np.nan+0j, dtype=complex)
    ratios = np.full(3, np.nan+0j)
    if identifiable:
        ratios = np.array([1., covariance[1, 2]/covariance[0, 2],
                           covariance[2, 1]/covariance[0, 1]])
        phase = np.angle(ratios)
        gain = amplitude[:, None]*np.exp(1j*phase[:, None])*nominal
    return dict(covariance=covariance, crosses=crosses,
                threshold=float(threshold), identifiable=identifiable,
                ratios=ratios, phase=phase, gain=gain)


def predict_common(weights, estimate, rate, boundary):
    """Frozen common direction, actual weights, and receiver rate only."""
    stars = weights.shape[2]
    predicted = frozen_phase.predict_phase(
        weights, estimate['gain'], rate, np.ones(stars, bool),
        np.full(stars, not estimate['identifiable']), boundary)
    predicted['common_unidentified'] = predicted.pop('ambiguous')
    return predicted


def analyze_path(folder, expected):
    inputs = folder/'input'
    gate = folder/'gate3'
    input_sha = {name: digest(inputs/name) for name in
                 ('samples.npy', 'observations.npz', 'metadata.json')}
    assert input_sha == expected['input_sha256']
    output_sha = {name: digest(gate/name) for name in
                  ('fit.npz', 'tracking.npz', 'trace.npz')}
    for name, sha in output_sha.items():
        assert sha == expected['paths']['gate3']['output_sha256'][name]
    metadata = json.loads((inputs/'metadata.json').read_text(encoding='utf-8'))
    with np.load(inputs/'observations.npz') as saved:
        record = dict(saved)
    record['samples'] = np.load(inputs/'samples.npy', mmap_mode='r')
    with np.load(gate/'fit.npz') as saved:
        fit = dict(saved)
    with np.load(gate/'tracking.npz') as saved:
        tracking = dict(saved)
    with np.load(gate/'trace.npz') as saved:
        trace = dict(saved)
    prns, templates, calibration, rate = extract_startup(record)
    np.testing.assert_array_equal(prns, tracking['startup_prns'])
    np.testing.assert_allclose(calibration, tracking['calibration'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rate, tracking['calibration_rate'], rtol=0, atol=1e-12)
    amplitude = fit['relative_amplitude']
    np.testing.assert_array_equal(amplitude, tracking['startup_relative_amplitude'])
    nominal = record['nominal'][np.array(ANTENNAS)]
    np.testing.assert_array_equal(trace['trace_nominal'], amplitude[:, None]*nominal)
    prompts = frozen_phase.reference_prompts(record, prns, templates,
                                              calibration, rate)
    estimate = estimate_common_gain(prompts, nominal, prns, amplitude)
    boundary = np.flatnonzero(np.arange(len(fit['weight']))
                              *frozen_phase.BLOCK_S >= frozen_phase.FIRST_BOUNDARY_S)
    predicted = predict_common(fit['weight'], estimate, rate, boundary)

    # The generator is deliberately unavailable to the estimator above.
    physical_gain, _ = generator_truth(metadata, record['nominal'])
    truth, before, after = frozen_phase.truth_after_prediction(
        fit['weight'], record['nominal'], physical_gain, calibration, rate,
        boundary)
    truth_zero = (before <= frozen_phase.RESPONSE_ZERO) | (after <= frozen_phase.RESPONSE_ZERO)
    truth[truth_zero] = np.nan
    scorable = predicted['available'] & ~truth_zero
    full = np.ones_like(scorable, bool)
    stable_gain = ((fit['gain'][boundary-1] >= .05)
                   & (fit['gain'][boundary] >= .05))
    all_score = frozen_phase.loss_summary(predicted['prediction'], truth,
                                          scorable, full)
    stable_score = frozen_phase.loss_summary(predicted['prediction'], truth,
                                             scorable, stable_gain)
    summary = dict(input_sha256=input_sha, output_sha256=output_sha,
                   startup_captured_prns=len(prns),
                   common_identifiable=estimate['identifiable'],
                   cross_covariance_threshold=estimate['threshold'],
                   cross_covariance_magnitude=np.abs(estimate['crosses']).tolist(),
                   boundary_count=len(boundary), opportunity_count=int(full.sum()),
                   common_unidentified=int(predicted['common_unidentified'].sum()),
                   zero_response=int(predicted['zero'].sum()),
                   predictable=int(predicted['available'].sum()),
                   truth_zero_response=int(truth_zero.sum()),
                   all_prn=all_score, original_gain_ge_005=stable_score,
                   truth_after_attack_counterfactual=(metadata['scenario']
                                                       == 'genuine_absent'))
    arrays = dict(boundary_s=predicted['boundary_s'], startup_prns=prns,
                  covariance=estimate['covariance'], phase_ratio=estimate['ratios'],
                  common_phase=estimate['phase'], common_gain=estimate['gain'],
                  saved_relative_amplitude=amplitude,
                  prediction=predicted['prediction'], truth=truth,
                  common_unidentified=predicted['common_unidentified'],
                  zero_response=predicted['zero'],
                  predictable=predicted['available'],
                  truth_zero_response=truth_zero, scorable=scorable,
                  original_gain_ge_005=stable_gain,
                  predicted_before_amplitude=predicted['before_amplitude'],
                  predicted_after_amplitude=predicted['after_amplitude'],
                  true_before_amplitude=before, true_after_amplitude=after)
    return summary, arrays


def analyze():
    assert digest(study.OUTPUT/'summary.json') == STAGE47_SUMMARY_SHA256
    previous = study.OUTPUT/'startup_complex_phase_transfer_summary.json'
    assert digest(previous) == FROZEN_PHASE_SUMMARY_SHA256
    previous_summary = json.loads(previous.read_text(encoding='utf-8'))
    assert digest(Path(frozen_phase.__file__)) == previous_summary['source_sha256']
    frozen = json.loads((study.OUTPUT/'summary.json').read_text(encoding='utf-8'))
    root = Path(__file__).resolve().parents[1]
    for source in SOURCES:
        assert digest(root/source) == frozen['source_sha256'][source], source
    cases, arrays = {}, {}
    for seed in study.SEEDS:
        cases[str(seed)] = {}
        for scene in SCENES:
            folder = study.OUTPUT/str(seed)/scene
            result, detail = analyze_path(folder, frozen['cases'][str(seed)][scene])
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
                frozen_phase_summary_sha256=FROZEN_PHASE_SUMMARY_SHA256,
                frozen_phase_source_sha256=previous_summary['source_sha256'],
                frozen_source_sha256={name: frozen['source_sha256'][name]
                                      for name in SOURCES},
                fixed_support_rule_passed=bool(supported), cases=cases), arrays


def main():
    summary, arrays = analyze()
    output = study.OUTPUT
    summary_path = output/'common_residual_phase_transfer_summary.json'
    manifest_path = output/'common_residual_phase_transfer_manifest.json'
    assert not summary_path.exists() and not manifest_path.exists()
    for key in arrays:
        assert not (output/f'common_residual_phase_transfer_{key}.npz').exists()
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False),
                            encoding='utf-8')
    hashes = {summary_path.name: digest(summary_path)}
    for key, detail in arrays.items():
        path = output/f'common_residual_phase_transfer_{key}.npz'
        np.savez_compressed(path, **detail)
        hashes[path.name] = digest(path)
    manifest_path.write_text(json.dumps(hashes, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
