import numpy as np
import pytest
from core.sdcp import calibrated_sdcp


def test_integer_invariance_static_echo_and_external_bias():
    wavelength = .19
    geometry, bias, echo = -.03, .012, .02
    observed = geometry+bias+echo+np.arange(-20, 21)*wavelength
    np.testing.assert_allclose(calibrated_sdcp(observed, geometry, wavelength, bias), echo, atol=1e-14)
    np.testing.assert_allclose(calibrated_sdcp(observed, geometry, wavelength, 0), echo+bias, atol=1e-14)
    assert abs(calibrated_sdcp(wavelength/2, 0, wavelength, 0)) == pytest.approx(wavelength/2)
    assert calibrated_sdcp(wavelength, 0, wavelength, 0) == pytest.approx(0)
