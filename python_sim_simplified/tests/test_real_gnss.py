import numpy as np
import pytest


def test_recorded_iq_preserves_signed_components(tmp_path):
    from core.recorded_receiver import read_iq8
    path = tmp_path/'iq.bin'
    path.write_bytes(bytes([127, 128, 255, 1, 0, 17]))
    np.testing.assert_array_equal(read_iq8(path, 1, 2), [-1+1j, 17j])


def test_costas_tracking_does_not_integrate_navigation_bit_flips():
    # Same carrier/code with and without 50-bps data. A full-angle PLL
    # incorrectly puts the data sign changes into its accumulated NCO phase.
    from core.config import TrackingConfig
    from core.gnss import ca_code
    from core.tracking import track
    fs = 1.023e6
    n = np.arange(307*1023)
    x = ca_code(7)[n % 1023]*np.exp(1j*(.2+2*np.pi*40*n/fs))
    bits = np.where((n//(20*1023)) % 2, -1., 1.)
    acq = {'code_phase_samples':0, 'doppler_hz':0.}
    cfg = TrackingConfig(max_epochs=300)
    cfg.carrier_discriminator = 'costas'
    with np.errstate(divide='ignore', invalid='ignore'):
        pilot = track(x,fs,7,acq,cfg,1575.42e6,1.023e6)
        data = track(x*bits,fs,7,acq,cfg,1575.42e6,1.023e6)
    np.testing.assert_allclose(data['carrier_nco_phase_rad'],pilot['carrier_nco_phase_rad'],atol=1e-9)
    assert abs(np.median(data['carrier_nco_frequency_hz'][100:])-40) < .1
    assert np.median(data['phase_lock_metric'][100:]) > .99


def test_lnav_parity_recovers_payload_and_rejects_every_single_bit_error():
    from core.gps_lnav import decode_lnav_word
    # RTKLIB decode_word reference vectors; previous D29/D30 occupy bits 31/30.
    for word, payload in [(0,0), (0x2000002a,0x800000),
                          (0x22c00012,0x8b0000), (0x5d3fffc4,0x8b0000)]:
        assert decode_lnav_word(word) == payload
        for bit in range(32):
            assert decode_lnav_word(word ^ (1<<bit)) is None


def test_constant_prompt_is_not_navigation_evidence_even_if_zero_words_pass_parity():
    from core.gps_lnav import navigation_check
    result=navigation_check({'prompt':np.ones(9000,dtype=complex)})
    assert result['valid_words'] > 0  # all-zero GPS words pass parity
    assert result.get('demodulation_evidence') is False
