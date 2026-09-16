import numpy as np

from channel_delay import apply_channel_delays
from sync_replay import replay, difference_errors

C = 299792458.
FC = 1575.42e6
IF = 1e6
W = C/FC


def run(**changes):
    parameters = dict(rf_delay_s=np.zeros(4), adc_skew_s=np.zeros(4),
                      sample_rate_error=np.zeros(4), nco_offset_s=np.zeros(4),
                      label_offset_s=np.zeros(4))
    parameters.update(changes)
    return replay(np.arange(100)/1e4, [-3000., 0., 4000.],
                  carrier_hz=FC, if_hz=IF, **parameters)


def test_rf_delay_matches_existing_time_domain_source():
    fs = 8e6
    t = np.arange(512)/fs
    delay = 1/fs
    tone = np.exp(2j*np.pi*(IF+4000)*t)
    delayed = apply_channel_delays(tone[None, :], [delay], fs, FC, IF)[0]
    result = replay(t, [4000.], carrier_hz=FC, if_hz=IF,
                    rf_delay_s=[delay], adc_skew_s=[0.], sample_rate_error=[0.],
                    nco_offset_s=[0.], label_offset_s=[0.])
    np.testing.assert_allclose(result['phasor'][1:, 0, 0],
                               delayed[1:]*np.exp(-2j*np.pi*IF*t[1:]), atol=2e-12)


def test_dd_cancels_common_carrier_term_but_retains_doppler_times_skew():
    skew = 1e-8
    result = run(adc_skew_s=[0., skew, skew, skew])
    sd, dd = difference_errors(result['phasor'], result['clean_phasor'], W)
    np.testing.assert_allclose(sd[:, 0], np.broadcast_to(-W*(IF+np.array([-3000., 0., 4000.]))*skew, (100, 3)), atol=2e-12)
    np.testing.assert_allclose(dd[:, 0], np.broadcast_to(-W*np.array([3000., 7000.])*skew, (100, 2)), atol=2e-12)


def test_label_error_changes_alignment_without_changing_samples():
    clean = run()
    shifted = run(label_offset_s=[0., .001, .001, .001])
    np.testing.assert_array_equal(shifted['phasor'], clean['phasor'])
    np.testing.assert_allclose(shifted['label_aligned_error_m'][:, 1]-clean['label_aligned_error_m'][:, 1],
                               np.broadcast_to(W*np.array([-3000., 0., 4000.])*.001, (100, 3)), atol=2e-12)


def test_clock_rate_error_accumulates_and_common_channel_error_cancels():
    result = run(sample_rate_error=[0., 1e-6, 1e-6, 1e-6])
    expected = W*IF*result['time_s']*1e-6/(1+1e-6)
    np.testing.assert_allclose(result['unwrapped_error_m'][:, 1, 1], expected, atol=2e-12)
    common = run(adc_skew_s=[1e-8]*4)
    sd, dd = difference_errors(common['phasor'], common['clean_phasor'], W)
    np.testing.assert_allclose(sd, 0., atol=2e-12)
    np.testing.assert_allclose(dd, 0., atol=2e-12)


def test_rf_delay_is_not_adc_skew_and_integer_cycles_remain_ambiguous():
    rf = run(rf_delay_s=[0., 1e-9, 1e-9, 1e-9])
    adc = run(adc_skew_s=[0., 1e-9, 1e-9, 1e-9])
    np.testing.assert_allclose(rf['unwrapped_error_m'][:, 1, 1], C*1e-9, atol=2e-12)
    np.testing.assert_allclose(adc['unwrapped_error_m'][:, 1, 1], -W*IF*1e-9, atol=2e-12)
    sd, _ = difference_errors(rf['phasor'], rf['clean_phasor'], W)
    np.testing.assert_allclose(rf['unwrapped_error_m'][:, 1, 1]-sd[:, 0, 1], 2*W, atol=2e-12)


def test_common_frequency_nco_offset_cancels_in_dd():
    result = run(nco_offset_s=[0., 1e-8, 1e-8, 1e-8])
    sd, dd = difference_errors(result['phasor'], result['clean_phasor'], W)
    np.testing.assert_allclose(sd, W*IF*1e-8, atol=2e-12)
    np.testing.assert_allclose(dd, 0., atol=2e-12)
