import numpy as np
import pytest

from config import Direction, WidebandInterferer
from interference import generate_wideband_waveform


FS_HZ = 16.368e6
IF_HZ = 4.092e6
IMPEDANCE_OHM = 50.0
NUM_SAMPLES = 131_072


def _generate(interferer):
    return generate_wideband_waveform(
        interferer,
        num_samples=NUM_SAMPLES,
        fs_hz=FS_HZ,
        if_hz=IF_HZ,
        impedance_ohm=IMPEDANCE_OHM,
    )


def test_wideband_waveform_is_repeatable_and_has_configured_total_power():
    interferer = WidebandInterferer(
        Direction(80.0, 10.0),
        bandwidth_hz=2.0e6,
        random_seed=1001,
    )

    first = _generate(interferer)
    second = _generate(interferer)
    different_seed = _generate(
        WidebandInterferer(
            Direction(80.0, 10.0),
            bandwidth_hz=2.0e6,
            random_seed=1002,
        )
    )

    assert first.shape == (NUM_SAMPLES,)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different_seed)
    expected_watts = 10.0 ** ((interferer.power_dbm - 30.0) / 10.0)
    measured_watts = np.mean(np.abs(first) ** 2) / IMPEDANCE_OHM
    assert measured_watts == pytest.approx(expected_watts, rel=1e-12)


def test_wideband_waveform_energy_is_concentrated_in_configured_band():
    interferer = WidebandInterferer(
        Direction(80.0, 10.0),
        bandwidth_hz=2.0e6,
        random_seed=1001,
    )

    waveform = _generate(interferer)
    baseband = waveform * np.exp(
        -1j * 2.0 * np.pi * IF_HZ * np.arange(NUM_SAMPLES) / FS_HZ
    )
    spectrum_power = np.abs(np.fft.fft(baseband)) ** 2
    frequencies_hz = np.fft.fftfreq(NUM_SAMPLES, 1.0 / FS_HZ)
    in_band = np.abs(frequencies_hz) <= 0.55 * interferer.bandwidth_hz

    assert spectrum_power[in_band].sum() / spectrum_power.sum() >= 0.95


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
        ({"bandwidth_hz": FS_HZ}, "bandwidth_hz"),
        ({"filter_taps": 128}, "filter_taps"),
        ({"filter_taps": 1}, "filter_taps"),
        ({"random_seed": -1}, "random_seed"),
        ({"center_offset_hz": 4.5e6}, "Nyquist"),
    ],
)
def test_wideband_waveform_rejects_invalid_configuration(overrides, message):
    parameters = {
        "direction": Direction(80.0, 10.0),
        "bandwidth_hz": 2.0e6,
        "random_seed": 1001,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        _generate(WidebandInterferer(**parameters))
