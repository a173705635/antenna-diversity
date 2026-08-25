import numpy as np
import pytest

from antenna_diversity.receiver.carrier_nco import FixedCarrierNCO


def make_nco(*, bits: int = 30, initial_frequency_hz: float = 0.0):
    return FixedCarrierNCO(
        clock_hz=6.6e6,
        word_bits=bits,
        coherent_interval_s=1e-3,
        initial_frequency_hz=initial_frequency_hz,
    )


def test_frequency_lsb_matches_word_length():
    nco_30 = make_nco(bits=30)
    nco_43 = make_nco(bits=43)

    assert nco_30.frequency_lsb_hz == pytest.approx(0.006146728992462158)
    assert nco_43.frequency_lsb_hz == pytest.approx(7.503331289626658e-07)
    assert nco_30.frequency_lsb_hz / nco_43.frequency_lsb_hz == pytest.approx(
        8192.0
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_half_lsb_rounds_away_from_zero(sign):
    nco = make_nco()

    update = nco.update(
        command_frequency_hz=sign * 0.5 * nco.frequency_lsb_hz,
        phase_correction_rad=0.0,
    )

    assert update.fcw == sign
    assert update.frequency_hz == pytest.approx(
        sign * 0.006146728992462158
    )
    assert update.quantization_error_hz == pytest.approx(
        update.frequency_hz - update.command_frequency_hz
    )


def test_initial_frequency_is_quantized_and_exposed():
    nco = make_nco(initial_frequency_hz=1500.123)

    assert nco.fcw == 244_052
    assert nco.frequency_hz == pytest.approx(
        244_052 * 0.006146728992462158
    )
    assert nco.command_frequency_hz == 1500.123
    assert nco.quantization_error_hz == pytest.approx(
        nco.frequency_hz - 1500.123
    )


def test_update_advances_phase_with_new_fcw_and_phase_correction():
    nco = make_nco()
    command_hz = 25.0 * nco.frequency_lsb_hz
    correction_words = 7
    correction_rad = 2.0 * np.pi * correction_words / nco.modulus

    update = nco.update(command_hz, correction_rad)

    assert update.fcw == 25
    assert update.phase_word == (25 * 6600 + correction_words) % 2**30
    assert nco.phase_word == update.phase_word
    assert nco.phase_rad == pytest.approx(
        2.0 * np.pi * update.phase_word / 2**30
    )


def test_negative_fcw_wraps_phase_word_into_unsigned_range():
    nco = make_nco()

    update = nco.update(-nco.frequency_lsb_hz, 0.0)

    assert update.fcw == -1
    assert update.phase_word == (-6600) % 2**30
    assert 0 <= update.phase_word < 2**30


def test_signed_frequency_range_includes_negative_nyquist_only():
    nco = make_nco()

    update = nco.update(-3.3e6, 0.0)
    assert update.fcw == -(2**29)
    assert update.frequency_hz == -3.3e6

    with pytest.raises(ValueError, match="FCW"):
        nco.update(3.3e6, 0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"clock_hz": 0.0}, "clock_hz"),
        ({"clock_hz": np.nan}, "clock_hz"),
        ({"word_bits": True}, "word_bits"),
        ({"word_bits": 1}, "word_bits"),
        ({"word_bits": 63}, "word_bits"),
        ({"coherent_interval_s": 1.5 / 6.6e6}, "integer number"),
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs, message):
    arguments = dict(
        clock_hz=6.6e6,
        word_bits=30,
        coherent_interval_s=1e-3,
        initial_frequency_hz=0.0,
    )
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        FixedCarrierNCO(**arguments)


@pytest.mark.parametrize("command_hz", [np.nan, np.inf, -np.inf, 3.3e6])
def test_update_rejects_nonfinite_or_overflowing_frequency(command_hz):
    nco = make_nco()

    with pytest.raises(ValueError, match="frequency|FCW"):
        nco.update(command_hz, 0.0)


@pytest.mark.parametrize("correction_rad", [np.nan, np.inf, -np.inf])
def test_update_rejects_nonfinite_phase_correction(correction_rad):
    nco = make_nco()

    with pytest.raises(ValueError, match="phase_correction_rad"):
        nco.update(0.0, correction_rad)
