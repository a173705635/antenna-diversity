import numpy as np

from audit_array_reacquisition import available_at


def test_fixed_evaluation_clock_counts_missing_stale_and_future_outputs_unavailable():
    trace=[dict(earliest_evidence_time_s=.2005,residual_m=0.),
           dict(earliest_evidence_time_s=.203,residual_m=None)]
    np.testing.assert_array_equal(available_at(trace,np.array([.200,.2006,.2026,.2031])),
                                  [False,True,False,False])
