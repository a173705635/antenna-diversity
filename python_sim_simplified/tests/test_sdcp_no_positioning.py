import numpy as np

from audit_sdcp_no_positioning import raw_sdcp


def test_common_time_reconstruction_preserves_whole_and_half_cycles():
    fs = 10000.
    wavelength = .19
    times = np.array([.0011, .0021, .0031])
    first_starts = np.array([2, 12, 22, 32])
    second_starts = np.array([5, 15, 25, 35])
    first = {'epoch_start_samples': first_starts,
             'carrier_nco_phase_rad': 2*np.pi*1000*first_starts/fs,
             'carrier_nco_frequency_hz': np.full(4, 1000.)}
    second = {'epoch_start_samples': second_starts,
              'carrier_nco_phase_rad': 2*np.pi*1002*second_starts/fs + 7*np.pi,
              'carrier_nco_frequency_hz': np.full(4, 1002.)}
    np.testing.assert_allclose(raw_sdcp(first, second, fs, times, wavelength),
                               -wavelength*(3.5+2*times), atol=1e-14)
