"""Small analytic checks for the standalone source-evidence toy model."""

import numpy as np
from scipy.stats import f

from experiments.source_evidence_model import binomial_interval, gates, ideal_probabilities, statistics


def test_group_statistics_use_coherent_sum_and_disjoint_off_residual():
    prompts = np.empty((1, 20, 2), complex)
    prompts[0, :, 0] = 1
    prompts[0, :, 1] = np.tile([1, -1, 1, -1, 1], 4)

    old, on_stat, off_stat = statistics(prompts)

    np.testing.assert_allclose(old, [1.])
    np.testing.assert_allclose(on_stat, [5 / 1.2])
    np.testing.assert_allclose(off_stat, [0.2 / 1.2])


def test_ton_gate_has_same_fixed_basis_null_tail_as_old_gate():
    for dimension in (2, 3):
        thresholds = gates(dimension)
        old_tail = f.sf(3, 40, 40 * (dimension - 1))
        np.testing.assert_allclose(thresholds["old_pfa"], old_tail, rtol=1e-13)
        np.testing.assert_allclose(
            f.sf(thresholds["ton"], 8, 32 * (dimension - 1)),
            old_tail,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            f.cdf(thresholds["toff"], 8 * (dimension - 1), 32 * (dimension - 1)),
            .99,
            rtol=1e-12,
        )


def test_exact_interval_does_not_call_zero_events_zero_risk():
    low, high = binomial_interval(0, 20_000)
    assert low == 0
    assert high > 0


def test_zero_noncentrality_uses_central_f_survival():
    threshold = gates(2)
    prediction = ideal_probabilities(2, {"kind": "noise"}, threshold)
    np.testing.assert_allclose(prediction["old"], threshold["old_pfa"])
    np.testing.assert_allclose(prediction["ton"], threshold["old_pfa"])
