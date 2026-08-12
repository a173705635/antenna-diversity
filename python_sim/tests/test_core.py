import numpy as np
import pytest

from antenna_diversity.array.steering import steering_vector
from antenna_diversity.config import default_config
from antenna_diversity.gnss.ca_code import ca_code


def test_prn1_has_expected_gold_code_autocorrelation():
    """Catch wrong LFSR feedback or one-based-to-zero-based tap conversion."""
    code = ca_code(1)

    assert code.shape == (1023,)
    assert set(np.unique(code)) == {-1.0, 1.0}
    correlation = np.fft.ifft(np.abs(np.fft.fft(code)) ** 2).real
    assert correlation[0] == pytest.approx(1023.0, abs=1e-9)
    assert set(np.rint(correlation[1:]).astype(int)) == {-65, -1, 63}


@pytest.mark.parametrize("invalid_prn", [0, 33, 1.5])
def test_ca_code_rejects_prn_outside_integer_range(invalid_prn):
    """Catch silent generation of unsupported or fractional PRNs."""
    with pytest.raises(ValueError, match="1 through 32"):
        ca_code(invalid_prn)


def test_broadside_referenced_steering_phase():
    """Catch a flipped angle convention or missing wavelength normalization."""
    cfg = default_config()
    a = steering_vector(
        [0.0, cfg.array.spacing_m],
        30.0,
        cfg.signal.wavelength_m,
    )
    expected = np.exp(
        -1j
        * 2
        * np.pi
        * cfg.array.spacing_m
        * np.sin(np.deg2rad(30.0))
        / cfg.signal.wavelength_m
    )

    np.testing.assert_allclose(a, [1.0, expected], atol=1e-12)
