"""Continuous local-code reference checks, independent of interference."""

import numpy as np
import pytest

from config import TrackingConfig
from gnss import ca_code
from tracking import _sample_code, bandlimited_code_replica


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


@pytest.mark.parametrize("doppler", [-1500.0, 1500.0])
def test_linear_tracking_is_consistent_across_output_sample_rates(doppler):
    from acquisition import acquire
    from audit_multipath_baseline import baseline_config
    from dfe import process_dfe
    from gnss import _desired_waveform
    from tracking import track

    cfg = baseline_config()
    cfg.signal.doppler_hz = doppler
    cfg.tracking.code_replica_bandwidth_hz = cfg.dfe.cutoff_hz
    cfg.dfe.output_bits = 0
    n = np.arange(round(cfg.signal.fs_rf_hz * 0.1))
    waveform, _, _ = _desired_waveform(cfg, n, n / cfg.signal.fs_rf_hz)
    phases = []
    for decimation in (1, 4):
        cfg.dfe.decimation = decimation
        channels, info = process_dfe(
            waveform[None, :], cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz
        )
        fs_hz = info["fs_out_hz"]
        acquisition = acquire(channels[0], fs_hz, 1, cfg.signal.code_rate_hz, cfg.acquisition)
        assert acquisition["success"]
        # Isolate the correlator: coarse acquisition may select different
        # physical epochs on different grids. Give both the same start time.
        # This known-delay initialization is confined to this reference test.
        acquisition["code_phase_samples"] = round(
            cfg.signal.code_delay_chips * fs_hz / cfg.signal.code_rate_hz
        )
        result = track(channels[0], fs_hz, 1, acquisition, cfg.tracking,
                       cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
        phases.append(result["code_phase_chips"])
    # Same physical signal and filter should give the same phase trajectory;
    # 0.0001 chip is about 3 cm, far below the previous metre-scale discrepancy.
    difference = (phases[0] - phases[1] + 511.5) % 1023 - 511.5
    assert np.max(np.abs(difference)) < 1e-4
