import numpy as np
import pytest

from antenna_diversity.config import default_config
from antenna_diversity.numeric import matlab_round


def test_default_config_satisfies_matlab_sampling_constraints():
    """Catch drift in the sample-rate relationships required by the receiver."""
    cfg = default_config()

    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.dfe.fs_out_hz / cfg.signal.code_rate_hz == 4
    assert cfg.dfe.num_taps == 129
    assert cfg.diversity.modes == ("single", "egc", "mvdr")


def test_matlab_round_moves_half_ties_away_from_zero():
    """Catch use of NumPy's ties-to-even rounding in hardware quantizers."""
    values = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])

    actual = matlab_round(values)

    np.testing.assert_array_equal(actual, [-3, -2, -1, 0, 1, 2, 3])


def test_invalid_even_fir_length_is_rejected():
    """Catch configurations whose FIR group delay is not an integer sample."""
    cfg = default_config()
    cfg.dfe.num_taps = 128

    with pytest.raises(ValueError, match="odd"):
        cfg.validate()
