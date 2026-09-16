import numpy as np
from continuous_improvement import Case,generate as original
from satellite_count_generator import generate


def test_existing_eight_satellites_and_rf_are_unchanged():
    case=Case(duration=.04,stars=8,phase_deg=10.,amplitude_db=1.,velocity_east=2.,attack_start=.03)
    for a,b in zip(original(case,114001,True),generate(case,114001,True)):
        np.testing.assert_array_equal(a,b)


def test_twenty_four_keeps_original_eight_geometry_and_motion():
    eight=generate(Case(duration=.2,stars=8),114001,False)
    full=generate(Case(duration=.2,stars=24),114001,False)
    np.testing.assert_array_equal(full[1][:,:8],eight[1])
    np.testing.assert_array_equal(full[4][:,:8],eight[4])
    assert full[1].shape==(4,24)
    assert len(full[2])==24
