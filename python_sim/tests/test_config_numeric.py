import numpy as np
import pytest

from antenna_diversity.config import default_config, mate70_default_config
from antenna_diversity.numeric import matlab_round


def test_default_config_satisfies_matlab_sampling_constraints():
    """Catch drift in the sample-rate relationships required by the receiver."""
    cfg = default_config()

    assert cfg.signal.fs_rf_hz == 16.368e6
    assert cfg.dfe.fs_out_hz == 4.092e6
    assert cfg.dfe.fs_out_hz / cfg.signal.code_rate_hz == 4
    assert cfg.dfe.num_taps == 129
    assert cfg.diversity.modes == ("single", "egc", "mvdr", "blind_null")


def test_default_config_is_mate70_four_corner_array():
    cfg = default_config()

    assert cfg.array.active_antennas == ("TL", "TR", "BL", "BR")
    assert cfg.array.num_elements == 4


def test_blind_null_defaults_are_explicit():
    cfg = mate70_default_config()

    assert cfg.diversity.blind_null.detection_threshold_db == 6.0
    assert cfg.diversity.blind_null.reference_channel == 0
    assert cfg.diversity.blind_null.min_reference_projection_power == 0.01


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("detection_threshold_db", np.nan, "detection threshold"),
        ("reference_channel", True, "reference channel"),
        ("reference_channel", -1, "reference channel"),
        ("min_reference_projection_power", 0.0, "projection power"),
        ("min_reference_projection_power", 1.1, "projection power"),
    ],
)
def test_blind_null_config_rejects_invalid_values(field, value, message):
    cfg = mate70_default_config()
    setattr(cfg.diversity.blind_null, field, value)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_unknown_diversity_mode_is_rejected_before_pipeline_dispatch():
    cfg = mate70_default_config()
    cfg.diversity.modes = ("single", "music")

    with pytest.raises(ValueError, match="diversity.modes"):
        cfg.validate()


def test_diversity_modes_reject_duplicates_after_normalization():
    cfg = mate70_default_config()
    cfg.diversity.modes = ("single", "SINGLE")

    with pytest.raises(ValueError, match="diversity.modes"):
        cfg.validate()


def test_diversity_modes_reject_nonstring_members():
    cfg = mate70_default_config()
    cfg.diversity.modes = ("single", ["mvdr"])

    with pytest.raises(ValueError, match="diversity.modes"):
        cfg.validate()


def test_matlab_round_moves_half_ties_away_from_zero():
    """Catch use of NumPy's ties-to-even rounding in hardware quantizers."""
    values = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])

    actual = matlab_round(values)

    np.testing.assert_array_equal(actual, [-3, -2, -1, 0, 1, 2, 3])


def test_invalid_even_fir_length_is_rejected():
    """Catch configurations whose FIR group delay is not an integer sample."""
    cfg = default_config()
    cfg.dfe.num_taps = 128

    with pytest.raises(ValueError, match="odd"):
        cfg.validate()


@pytest.mark.parametrize("mode", ["integer", "FIXED", "", None, ["fixed"]])
def test_tracking_rejects_unknown_carrier_nco_mode(mode):
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = mode

    with pytest.raises(ValueError, match="carrier_nco_mode"):
        cfg.validate()


@pytest.mark.parametrize(
    "clock_hz",
    [
        0.0,
        -1.0,
        np.nan,
        np.inf,
        True,
        1.0 + 0.0j,
        np.array(6.6e6),
        np.array([6.6e6]),
    ],
)
def test_tracking_rejects_invalid_carrier_nco_clock(clock_hz):
    cfg = default_config()
    cfg.tracking.carrier_nco_clock_hz = clock_hz

    with pytest.raises(ValueError, match="carrier_nco_clock_hz"):
        cfg.validate()


@pytest.mark.parametrize("word_bits", [True, 1, 63, 30.5])
def test_tracking_rejects_invalid_carrier_nco_word_bits(word_bits):
    cfg = default_config()
    cfg.tracking.carrier_nco_word_bits = word_bits

    with pytest.raises(ValueError, match="carrier_nco_word_bits"):
        cfg.validate()


def test_fixed_tracking_rejects_fractional_nco_ticks_per_epoch():
    cfg = default_config()
    cfg.tracking.carrier_nco_mode = "fixed"
    cfg.tracking.carrier_nco_clock_hz = 6_600_000.5

    with pytest.raises(ValueError, match="integer number of NCO ticks"):
        cfg.validate()
