"""Raw carrier observations from independent RF/DFE/tracking chains."""
from dataclasses import replace

import numpy as np

from ..acquisition import acquire
from ..dfe import process_dfe
from ..rf import process_rf
from ..tracking import track


def raw_sdcp(first, second, fs, times, wavelength):
    phases = []
    for h in (first, second):
        starts = h['epoch_start_samples']/fs
        ix = np.searchsorted(starts, times, side='right')-1
        phases.append(h['carrier_nco_phase_rad'][ix]
                      + 2*np.pi*h['carrier_nco_frequency_hz'][ix]*(times-starts[ix]))
    return -wavelength*(phases[1]-phases[0])/(2*np.pi)


def receive(signal, cfg, seed, hardware=False):
    rf_cfg = replace(cfg.rf, amplitude_error_db=1. if hardware else 0.,
                     phase_error_deg=3. if hardware else 0.)
    adc, rf_info = process_rf(signal[None, :], rf_cfg, cfg.reference_impedance_ohm, seed)
    baseband, dfe_info = process_dfe(adc, cfg.dfe, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    del adc
    fs = dfe_info['fs_out_hz']
    acq = acquire(baseband[0], fs, cfg.signal.prn, cfg.signal.code_rate_hz, cfg.acquisition)
    h = track(baseband[0], fs, cfg.signal.prn, acq, cfg.tracking,
              cfg.signal.carrier_hz, cfg.signal.code_rate_hz)
    return h, {'noise_seed': seed, 'acquisition': acq, 'tracking_status': h['status'],
               'tracking_success': h['success'], 'rf': rf_info, 'dfe': dfe_info}
