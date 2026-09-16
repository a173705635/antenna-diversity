"""Continuous local-code reference checks, independent of interference."""

import numpy as np
import pytest

from core.config import TrackingConfig
from core.gnss import ca_code
from core.tracking import _sample_code, bandlimited_code_replica


def test_bandlimited_replica_preserves_dc_and_periodic_boundary():
    code = ca_code(1)
    replica = bandlimited_code_replica(code, 1.023e6, 1.8e6)
    assert np.mean(replica) == pytest.approx(np.mean(code), abs=1e-12)
    phases = np.array([-0.001, 0.001, 1022.999, 1023.001])
    values = _sample_code(replica, phases)
    np.testing.assert_allclose(values[:2], values[2:], atol=1e-11)
    assert abs(values[0] - values[1]) < 0.05


def test_bandlimited_replica_resolves_fractional_phase_and_chip_center():
    # Alternating chips provide known half-wave antisymmetry.
    code = np.tile([1.0, -1.0], 512)
    replica = bandlimited_code_replica(code, 1.0, 1.5)
    # The reference must preserve the rectangular pulse origin [n, n+1),
    # rather than shifting the code by half a chip when filtering.
    assert abs(_sample_code(replica, np.array([0.0]), code_length=1024)[0]) < 1e-12
    values = _sample_code(replica, np.array([0.01, 0.24, 0.5, 1.5]), code_length=1024)
    assert abs(values[0] - values[1]) > 0.1
    assert values[2] == pytest.approx(-values[3], abs=1e-12)
    assert values[2] > 0


def test_reference_is_opt_in_and_rejects_invalid_bandwidth():
    assert TrackingConfig().code_replica_bandwidth_hz is None
    with pytest.raises(ValueError):
        bandlimited_code_replica(ca_code(1), 1.023e6, -1.0)
