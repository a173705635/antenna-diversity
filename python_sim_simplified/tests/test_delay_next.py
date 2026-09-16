"""Behavioral regressions for the independent-noise space-delay estimator."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from delay_next import estimate
from delay_next_campaign import evaluate,observations
from lag_calibration import shape


class DelayNextTests(unittest.TestCase):
    def test_old_normal_failure_and_delay_gauge(self):
        row=evaluate(estimate,32051,'normal')
        self.assertTrue(row['usable'])
        self.assertLess(row['phase_rmse_deg'],1.)
        self.assertEqual(row['channel_delay_chips'],[0.,0.,0.,0.])
        self.assertLess(max(abs(np.array(row['code_error_m']))),10.)

    def test_group_delay_is_not_phase(self):
        row=evaluate(estimate,32051,'skew')
        self.assertTrue(row['usable'])
        self.assertLess(row['group_delay_rmse_chips'],.01)
        self.assertGreater(row['channel_delay_chips'][-1],.04)

    def test_noise_model_has_independent_heldout_evidence(self):
        row=evaluate(estimate,32051,'normal')
        self.assertEqual(row['noise_training_snapshots'],256)
        self.assertEqual(row['noise_validation_snapshots'],256)
        self.assertGreater(row['noise_model_p'],.001)

    def test_common_phase_gauge_invariance(self):
        z,a,lag,g,t,shift=observations(32051,'normal')
        rng=np.random.default_rng(32051+902000)
        covariance=.99*shape(lag[:,None],lag[None,:],.9)+.01*np.eye(len(lag))
        noise=.02*(rng.normal(size=(512,len(lag)))+1j*rng.normal(size=(512,len(lag))))@np.linalg.cholesky(covariance).T
        g1,i1=estimate(z,a,lag,noise)
        g2,i2=estimate(z*np.exp(.73j),a,lag,noise)
        np.testing.assert_allclose(np.angle(g1/g2),0.,atol=.002)
        np.testing.assert_allclose(i1['selected_delay_chips'],i2['selected_delay_chips'],atol=.0001)


if __name__=='__main__':unittest.main()
