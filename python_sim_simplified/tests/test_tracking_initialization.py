import numpy as np
import pytest

from audit_multipath_oracle import run_case


@pytest.mark.parametrize("doppler,delay", [
    (1500., 137.25), (-1500., 137.25),
    (1500., 137.37), (-1500., 137.37),
])
def test_data_only_initialization_removes_coarse_grid_startup_error(doppler, delay):
    result = run_case(doppler=doppler, delay=delay, refine=True)
    row = result["branches"][0]
    assert row["acquired"]
    assert row["all_epochs_rms_m"] < 0.1
    assert np.max(np.abs(row["code_phase_error_m"])) < 0.2
    assert row["initialization"]["uses_truth"] is False
    assert not row["initialization"]["at_search_boundary"]
