import numpy as np
from audit_phase_origin import make_streams, run_variant
from acquisition import acquire


def test_locked_phase_retains_fraction_with_changed_start_and_nco():
    x,fs,g,cfg=make_streams('ideal',2.4)
    acq=[acquire(v,fs,1,cfg.signal.code_rate_hz,cfg.acquisition) for v in x]
    row=run_variant(x,fs,g,cfg,acq,'test',(4,17),(1.7,-2.1))
    assert row['tail_fractional_max_abs_m'] < 1e-5
    biased=run_variant(x,fs,g,cfg,acq,'bias',(4,17),(1.7,-2.1),.4)
    assert biased['tail_fractional_max_abs_m'] > .01
    assert biased['tail_known_bias_corrected_max_abs_m'] < 1e-5
