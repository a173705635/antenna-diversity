import numpy as np


def test_error_diagnostics_preserve_fixed_bias_and_missing_values():
    from audit_pseudorange_bias import error_stats
    result = error_stats(np.array([[1., 3.], [3., np.nan], [5., 3.]]))
    np.testing.assert_allclose(result['mean_m'], [3., 3.])
    np.testing.assert_allclose(result['std_m'], [np.sqrt(8/3), 0.])
    np.testing.assert_allclose(result['rms_m'], [np.sqrt(35/3), 3.])
    assert result['epochs'] == [3, 2]
