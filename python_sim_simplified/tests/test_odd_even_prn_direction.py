import json

import numpy as np

from core.array_processing import prompt_projector
from tools.spoofing import odd_even_prn_direction as study


def test_parity_split_precedes_grouping_and_preserves_column_order():
    prns = np.arange(24)[::-1]
    prompts = np.zeros((3, 24, 20), complex)
    for column, star in enumerate(prns):
        prompts[:, column] = (star + 1) * np.arange(1, 4)[:, None]
    odd, even = study.fold_indices(prns)
    assert np.array_equal(prns[odd] + 1, np.arange(23, 0, -2))
    assert np.array_equal(prns[even] + 1, np.arange(24, 0, -2))
    grouped = study.group_five_ms(prompts[:, odd])
    assert grouped.shape == (3, 48)
    np.testing.assert_allclose(grouped[0, ::4],
                               np.sqrt(5) * (prns[odd] + 1))


def test_six_step_direction_matches_original_projector_and_phase():
    rng = np.random.default_rng(312)
    desired = rng.normal(size=(3, 48)) + 1j * rng.normal(size=(3, 48))
    prompts = (desired * (rng.normal(size=48) + 1j*rng.normal(size=48)) +
               np.array([[1.], [.4j], [-.3]]) @
               (rng.normal(size=(1, 48)) + 1j*rng.normal(size=(1, 48))))
    vector, eigen, failure = study.fit_direction(prompts, desired)
    assert failure == -1
    assert eigen.shape == (7, 2)
    projector, _ = prompt_projector(prompts, desired, 6)
    np.testing.assert_allclose(np.eye(3) - np.outer(vector, vector.conj()),
                               projector, rtol=0, atol=1e-12)
    shifted, _, shifted_failure = study.fit_direction(prompts * np.exp(.7j),
                                                      desired * np.exp(.7j))
    assert shifted_failure == -1
    np.testing.assert_allclose(np.outer(shifted, shifted.conj()),
                               np.outer(vector, vector.conj()), atol=1e-12)


def test_degenerate_spectrum_and_missing_fold_are_not_scored():
    desired = np.ones((3, 48), complex)
    vector, eigen, failure = study.fit_direction(np.zeros((3, 48)), desired)
    assert failure == 0 and np.isnan(vector).all()
    assert eigen[0, 0] == 0
    vector, _, failure = study.fit_direction(
        np.vstack((np.eye(3, dtype=complex), np.zeros((45, 3)))).T,
        desired)
    assert failure == 0 and np.isnan(vector).all()
    prns = np.arange(23)
    directions, _, failures, sizes = study.fit_block(
        np.ones((3, 23, 20)), np.ones((3, 24)), prns)
    assert sorted(sizes) == [11, 12]
    assert -2 in failures
    assert np.isnan(directions[failures == -2]).all()


def test_truth_rate_and_pair_coverage(monkeypatch):
    physical = np.array([1., 1j, -1.])
    fake = np.array([1., 1., 1.])
    monkeypatch.setattr(study, 'generator_truth', lambda metadata, nominal:
                        (physical, fake))
    labels = study.truth_labels({}, np.ones((3, 24)),
                                np.ones(3), np.array([0., 2., -1.]))
    time = (study.FIRST_BLOCK + .5) * study.WIDTH / study.FS
    expected = physical * np.exp(-1j * np.array([0., 2., -1.]) * time)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(labels[0], expected)
    assert labels.shape == (985, 3)
    weak = dict(valid=np.ones(985, bool), alignment=np.ones(985),
                truth_alignment=np.ones((985, 2)))
    h0 = dict(valid=np.ones(985, bool), alignment=np.zeros(985),
              truth_alignment=np.zeros((985, 2)))
    assert study.score_pair(weak, h0)['pass_rule']
    weak['valid'][0] = False
    assert not study.score_pair(weak, h0)['pass_rule']


def test_fit_scene_cannot_read_generator_truth():
    assert 'generator_truth' not in study.fit_scene.__code__.co_names
    assert 'generator_truth' not in study.fit_block.__code__.co_names


def test_no_common_valid_blocks_is_strict_json_failure_result():
    for weak_valid, h0_valid in ((np.zeros(985, bool), np.zeros(985, bool)),
                                 (np.arange(985) % 2 == 0,
                                  np.arange(985) % 2 == 1)):
        weak = dict(valid=weak_valid, alignment=np.full(985, np.nan),
                    truth_alignment=np.full((985, 2), np.nan))
        h0 = dict(valid=h0_valid, alignment=np.full(985, np.nan),
                  truth_alignment=np.full((985, 2), np.nan))
        result = study.score_pair(weak, h0)
        assert result['paired_valid'] == 0
        assert result['median_delta'] == [None, None, None]
        assert result['pass_rule'] is False
        assert 'NaN' not in json.dumps(result, allow_nan=False)


def test_all_scenes_fit_before_any_truth_label(monkeypatch):
    calls = []
    monkeypatch.setattr(study, 'SEEDS', (12,))
    monkeypatch.setattr(study, 'load_scene', lambda seed, scene, frozen:
                        (dict(scene=scene, nominal=np.ones((3, 24))),
                         dict(scene=scene), None, dict(samples='hash')))

    def fitted(record, saved):
        calls.append(('fit', record['scene']))
        return dict(calibration=np.ones(3), calibration_rate=np.zeros(3))

    def label(metadata, nominal, calibration, rate):
        calls.append(('truth', metadata['scene']))
        assert calls[:2] == [('fit', 'weak'), ('fit', 'h0_drop')]
        return np.ones((985, 3))

    monkeypatch.setattr(study, 'fit_scene', fitted)
    monkeypatch.setattr(study, 'truth_labels', label)
    monkeypatch.setattr(study, 'score_scene', lambda fit, labels: (
        np.ones(985), np.ones((985, 2)), np.ones(985, bool)))
    frozen = {'cases': {'12': {scene: {'paths': {'gate3': {
        'output_sha256': {}}}} for scene in study.SCENES}}}
    fits, inputs, _ = study.fit_all_scenes(frozen)
    study.score_all_scenes(fits, inputs)
    assert calls == [('fit', 'weak'), ('fit', 'h0_drop'),
                     ('truth', 'weak'), ('truth', 'h0_drop')]
