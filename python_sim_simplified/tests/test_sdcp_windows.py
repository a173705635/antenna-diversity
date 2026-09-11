import numpy as np
from audit_sdcp_windows import window_mean, WAVELENGTH


def test_causal_complete_windows_preserve_bias_and_handle_wrap():
    x = np.array([0.,0.,.01,.01,.01])
    np.testing.assert_allclose(window_mean(x,2), [0,.005,.01,.01], atol=1e-12)
    np.testing.assert_allclose(window_mean(x[:3],2), window_mean(x,2)[:2])
    assert abs(abs(window_mean(np.array([WAVELENGTH/2-.001,-WAVELENGTH/2+.001]),2)[0])-WAVELENGTH/2)<1e-12
