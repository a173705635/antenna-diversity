"""Conditional CV predictions must expire and cannot become measurements."""

import numpy as np

from experiments.spoofing.single_source_cv_bridge import cv_bridge


def test_prediction_expires_from_last_measurement_not_last_prediction():
    time = np.array([0., .03, .06, .10, .20, .25, .26, .260001])
    position = np.full((len(time), 3), np.nan)
    position[:3] = np.column_stack((time[:3] * 2., np.zeros((3, 2))))
    formal = np.full(len(time), np.nan)
    formal[:3] = 1.

    result = cv_bridge(time, position, formal)

    assert result["status"].tolist() == ["measured"] * 3 + ["predicted"] * 4 + ["unavailable"]
    np.testing.assert_allclose(result["position_m"][6, 0], .52)
    assert result["last_measured_time_s"][6] == .06
    assert result["age_s"][6] == .20
    assert result["age_s"][7] > .2
    assert result["velocity_fit_points"][6] == 3
    np.testing.assert_allclose(result["velocity_fit_span_s"][6], .06)


def test_insufficient_new_measurements_clear_previous_velocity():
    time = np.array([0., .03, .06, .27, .28])
    position = np.full((len(time), 3), np.nan)
    position[:3] = np.column_stack((time[:3] * 2., np.zeros((3, 2))))
    position[3] = [1., 0., 0.]
    formal = np.full(len(time), np.nan)
    formal[:4] = 1.

    result = cv_bridge(time, position, formal)

    assert result["status"].tolist() == ["measured", "measured", "measured",
                                          "measured", "unavailable"]
    assert np.isnan(result["velocity_m_s"][3]).all()


def test_no_initialization_and_future_does_not_change_prefix():
    time = np.array([0., .03, .06, .12])
    position = np.full((4, 3), np.nan)
    position[0] = [0., 0., 0.]
    formal = np.array([1., np.nan, np.nan, np.nan])
    early = cv_bridge(time[:3], position[:3], formal[:3])
    position[3] = [50., 0., 0.]
    formal[3] = 1.
    full = cv_bridge(time, position, formal)

    np.testing.assert_array_equal(full["status"][:3], early["status"])
    np.testing.assert_array_equal(full["position_m"][:3], early["position_m"])
    assert full["status"][1] == "unavailable"
