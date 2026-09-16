"""Deterministic narrowband timing experiment, not a receiver or fault detector.

Axes are epoch/channel/PRN. Positive RF delay samples an earlier wave; positive
ADC skew samples later; positive NCO offset advances local phase. Rate error
is fractional sample-clock frequency error: actual time is t/(1+epsilon).
Channel timing parameters are explicit arrays. Carrier ambiguities remain.
"""
import numpy as np

from config import SPEED_OF_LIGHT_MPS


def replay(time_s, doppler_hz, *, carrier_hz, if_hz, rf_delay_s,
           adc_skew_s, sample_rate_error, nco_offset_s, label_offset_s):
    t = np.asarray(time_s)[:, None, None]
    d = np.asarray(doppler_hz)[None, None, :]
    tau = np.asarray(rf_delay_s)[None, :, None]
    skew = np.asarray(adc_skew_s)[None, :, None]
    epsilon = np.asarray(sample_rate_error)[None, :, None]
    nco = np.asarray(nco_offset_s)[None, :, None]
    label = np.asarray(label_offset_s)[None, :, None]
    actual = t/(1+epsilon)+skew
    # Downconverted wave retains the RF phase of a physical path delay.
    received_phase = 2*np.pi*((if_hz+d)*actual-(carrier_hz+d)*tau)
    local_phase = 2*np.pi*if_hz*(t+nco)
    tracked_phase = received_phase-local_phase
    clean_phase = 2*np.pi*d*t
    wavelength = SPEED_OF_LIGHT_MPS/carrier_hz
    return dict(
        time_s=np.asarray(time_s), actual_sample_time_s=actual[..., 0],
        label_time_s=(t+label)[..., 0],
        phasor=np.exp(1j*tracked_phase),
        clean_phasor=np.broadcast_to(np.exp(1j*clean_phase), tracked_phase.shape),
        # Simulator evaluation only: a real receiver does not know this branch.
        unwrapped_error_m=-(tracked_phase-clean_phase)*wavelength/(2*np.pi),
        label_aligned_error_m=-(tracked_phase-2*np.pi*d*(t+label))*wavelength/(2*np.pi))


def difference_errors(phasor, clean_phasor, wavelength_m):
    """Wrapped error in channel single differences and PRN-0 double differences.

PRN 0 is an algebraic reference, not an authenticated satellite.
"""
    error = phasor*np.conj(clean_phasor)
    sd = error[:, 1:, :]*np.conj(error[:, :1, :])
    dd = sd[..., 1:]*np.conj(sd[..., :1])
    return (-np.angle(sd)*wavelength_m/(2*np.pi),
            -np.angle(dd)*wavelength_m/(2*np.pi))
