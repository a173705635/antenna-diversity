import numpy as np
import pytest

from antenna import antenna_response, build_antenna_model
from config import Direction
from code_replica import bandlimited_code_replica, _sample_code
from gnss import ca_code
from presets import default_config
from multipath import refine_single_path_delays


@pytest.mark.parametrize("delay", [.25, .5, .7])
def test_refinement_recovers_excess_delay_despite_shifted_pilot(delay):
    cfg=default_config();fs=4.092e6
    model=build_antenna_model(cfg.phone,cfg.signal.carrier_hz)
    los,_=antenna_response(model,cfg.signal.assumed_direction,cfg.signal.assumed_polarization,estimated=True)
    echo,_=antenna_response(model,Direction(110.,15.),cfg.signal.assumed_polarization,estimated=True)
    replica=bandlimited_code_replica(ca_code(1),1.023e6,1.8e6)
    n=np.arange(4*4092);phase=n/4-137.25
    channels=los[:,None]*_sample_code(replica,phase)+.5*echo[:,None]*_sample_code(replica,phase-delay)*np.exp(1j*2.1)
    acq={"code_phase_samples":550,"doppler_hz":0.}  # 137.5 chips: biased by 0.25.
    result=refine_single_path_delays(channels,fs,1,1.023e6,cfg.signal.carrier_hz,
                                   acq,los,echo,1.8e6,3.)
    assert result["status"]=="refined"
    assert abs(result["excess_delay_chips"]-delay)<.002
    assert result["uses_truth"] is False


def test_same_direction_is_rejected_before_delay_fitting():
    steering=np.ones(4,dtype=complex)
    result=refine_single_path_delays(np.ones((4,16368)),4.092e6,1,1.023e6,1.57542e9,
        {"code_phase_samples":0,"doppler_hz":0.},steering,steering,1.8e6,3.)
    assert result["status"]=="spatially_unresolvable"
