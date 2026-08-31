import numpy as np

from channel_delay import apply_channel_delays, delay_phase, fractional_delay


def test_zero_delay_is_an_exact_copy():
    signal = np.array([1.0 + 2.0j, -3.0 + 4.0j, 5.0 - 6.0j])

    delayed = fractional_delay(signal, 0.0)

    assert delayed is not signal
    np.testing.assert_array_equal(delayed, signal)


def test_integer_delay_zero_fills_front_and_truncates_tail():
    signal = np.arange(6, dtype=np.float64).astype(np.complex128)

    delayed = fractional_delay(signal, 2.0)

    np.testing.assert_array_equal(
        delayed,
        np.array([0, 0, 0, 1, 2, 3], dtype=np.complex128),
    )


def test_complex_if_delay_has_rf_equivalent_phase():
    fs_hz = 8.0e6
    if_hz = 1.0e6
    offset_hz = 100.0e3
    carrier_hz = 1_575.42e6
    delay_s = 0.375 / fs_hz
    n = np.arange(512)
    tone = np.exp(1j * 2.0 * np.pi * (if_hz + offset_hz) * n / fs_hz)

    actual = apply_channel_delays(
        tone[None, :],
        np.array([delay_s]),
        fs_hz,
        carrier_hz,
        if_hz,
    )[0]
    expected = tone * np.exp(
        -1j * 2.0 * np.pi * (carrier_hz + offset_hz) * delay_s
    )

    np.testing.assert_allclose(actual[32:-32], expected[32:-32], atol=2e-3)
    np.testing.assert_allclose(
        delay_phase(np.array([delay_s]), carrier_hz + offset_hz),
        [np.exp(-1j * 2.0 * np.pi * (carrier_hz + offset_hz) * delay_s)],
    )
