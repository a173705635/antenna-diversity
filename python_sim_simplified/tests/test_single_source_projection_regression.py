"""Run-before-data absolute safety check for the fixed Stage 26 IQ replay."""
import numpy as np

from experiments import single_source_projection_regression as regression


def positions(east):
    return np.column_stack((east, np.zeros(len(east))))


def test_same_late_error_passes_relative_but_fails_absolute():
    time = np.array([.29, .30, .31, .49, .50, .51])
    wrong = positions(np.array([np.nan, 0., 0., 0., 100., 100.]))
    result = regression.historical_screen(time, wrong, wrong, velocity_east=0.)
    assert result['relative']['pass']
    assert result['candidate_after_half_published'] == 2
    assert result['candidate_ge20'] == 2
    assert not result['pass']


def test_early_serious_error_fails_even_with_no_late_publications():
    time = np.array([.29, .30, .31, .49, .50, .51])
    wrong = positions(np.array([np.nan, 20., 0., np.nan, np.nan, np.nan]))
    result = regression.historical_screen(time, wrong, wrong, velocity_east=0.)
    assert result['relative']['pass']
    assert result['candidate_after_half_published'] == 0
    assert result['candidate_ge20'] == 1
    assert not result['pass']


def test_good_early_publications_then_stop_passes():
    time = np.array([.29, .30, .31, .49, .50, .51])
    baseline = positions(np.array([np.nan, 0., 1., np.nan, np.nan, np.nan]))
    candidate = positions(np.array([np.nan, 0., 1., 2., np.nan, np.nan]))
    result = regression.historical_screen(time, baseline, candidate,
                                           velocity_east=0.)
    assert result['relative']['pass']
    assert result['candidate_after_half_published'] == 0
    assert result['candidate_ge20'] == 0
    assert result['pass']


def test_fixed_reference_and_gate_pair():
    assert regression.GATES == (3., 2.)
    assert regression.SCENE == 'genuine_absent'
    assert regression.STAGE26.name == 'spoof_power_zero'
    assert regression.OUTPUT.name == 'projection_threshold_regression'
