import numpy as np
import pytest

from antenna_diversity.config import RFStage, default_config
from antenna_diversity.gnss.scenario import generate_scenario
from antenna_diversity.rf.cascade import cascade_metrics
from antenna_diversity.rf.frontend import process_frontend


def test_single_stage_cascade_preserves_gain_and_noise_figure():
    """Catch an incorrect Friis initialization for a one-stage receiver."""
    result = cascade_metrics((RFStage("OnlyStage", 20.0, 2.0),))

    assert result["total_gain_db"] == pytest.approx(20.0, abs=1e-12)
    assert result["noise_figure_db"] == pytest.approx(2.0, abs=1e-12)


def test_frontend_is_repeatable_and_produces_signed_integer_iq_codes():
    """Catch global RNG use, wrong ADC limits, or noninteger quantizer output."""
    cfg = default_config()
    cfg.jammer.enable = False
    x, _ = generate_scenario(cfg)

    adc1, info1 = process_frontend(x, cfg.rf, cfg.random_seed)
    adc2, info2 = process_frontend(x, cfg.rf, cfg.random_seed)

    np.testing.assert_array_equal(adc1, adc2)
    np.testing.assert_array_equal(
        info1["saturation_fraction"], info2["saturation_fraction"]
    )
    assert adc1.shape == x.shape
    assert np.all((-512 <= adc1.real) & (adc1.real <= 511))
    assert np.all((-512 <= adc1.imag) & (adc1.imag <= 511))
    np.testing.assert_array_equal(adc1.real, np.trunc(adc1.real))
    np.testing.assert_array_equal(adc1.imag, np.trunc(adc1.imag))


def test_hard_limiter_clips_i_and_q_independently():
    """Catch magnitude clipping where the model requires component ADC rails."""
    cfg = default_config()
    cfg.rf.enable_soft_limiting = False
    cfg.rf.temperature_k = 0.0
    cfg.rf.stages = (RFStage("Unity", 0.0, 0.0),)
    x = np.array([[1.0 + 0.1j, -1.0 - 0.1j]], dtype=np.complex128)

    adc, info = process_frontend(x, cfg.rf, cfg.random_seed)

    assert info["saturation_fraction"][0] == pytest.approx(1.0)
    np.testing.assert_array_equal(adc.real, [[511, -512]])
    np.testing.assert_array_equal(adc.imag, [[102, -102]])
