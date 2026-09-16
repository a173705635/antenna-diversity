import unittest
import numpy as np
import recovery_third as r


class RecoveryThirdTests(unittest.TestCase):
    def test_observation_boundary_and_geometry(self):
        obs, truth = r.generate(94101, 'weak')
        self.assertNotIn('truth', obs)
        self.assertNotIn('scenario', obs)
        self.assertEqual(obs['z'].shape, (2, 6, 4, 41))
        self.assertAlmostEqual(np.ptp(r.POSITIONS[:, 0]), .0759)
        self.assertAlmostEqual(np.ptp(r.POSITIONS[:, 1]), .1609)

    def test_same_input_determinism(self):
        obs, truth = r.generate(94101, 'weak')
        a = r.estimate(obs)
        b = r.estimate(obs)
        np.testing.assert_allclose(a['wave'], b['wave'])

    def test_truth_only_evaluation(self):
        obs, truth = r.generate(94102, 'absent')
        self.assertEqual(float(np.linalg.norm(truth['wave'])), 0)
        result = r.estimate(obs)
        self.assertTrue(np.isfinite(result['wave']).all())

    def test_frozen_noise_and_absent_are_not_authentic_support(self):
        import recovery_third_selected as selected
        for scenario in ['noise', 'absent']:
            obs, truth = selected.generate(94102, scenario)
            before = obs['z'].copy()
            result = selected.estimate(obs)
            self.assertEqual(int(result['accepted'].sum()), 0)
            np.testing.assert_array_equal(before, obs['z'])

    def test_frozen_source_matches_committed_freeze(self):
        import hashlib, json, pathlib
        freeze=json.loads(pathlib.Path('results/research-third-hundred/recovery_freeze.json').read_text())
        actual=hashlib.sha256(pathlib.Path('recovery_third_selected.py').read_bytes()).hexdigest()
        self.assertEqual(actual,freeze['source_sha'])

if __name__ == '__main__':
    unittest.main()
