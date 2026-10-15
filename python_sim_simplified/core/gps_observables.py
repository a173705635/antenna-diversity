"""Common-time observations from tracked code and decoded GPS LNAV time."""
import numpy as np
from .config import SPEED_OF_LIGHT_MPS as C
from .gps_broadcast import satellite_state, reception_frame, week_difference
from .positioning import solve_position, carrier_range


def transmit_time(history, anchor, times_s, fs_hz):
    """Satellite code-clock TOW at common sample times; no coarse range aid.

    The decoded TLM start identifies an integer code epoch. Its fractional
    position inside the integration block comes from the tracked code NCO.
    Only the inherently periodic code phase is unwrapped; carrier is untouched.
    """
    t=np.asarray(times_s,dtype=float)
    result=np.full(t.shape,np.nan)
    if not history['success_flag']:return result
    starts=np.asarray(history['epoch_start_samples'],dtype=float)/fs_hz
    phase=np.asarray(history['code_nco_phase_chips'],dtype=float)
    rate=np.asarray(history['code_nco_rate_hz'],dtype=float)
    if (len(starts)<2 or phase.shape!=starts.shape or rate.shape!=starts.shape
            or not np.isfinite([starts,phase,rate]).all()
            or not np.allclose(np.diff(starts),.001,atol=1e-10,rtol=0)):
        raise ValueError('transmit time requires continuous 1-ms tracking')
    a=anchor['epoch_index']
    residual=np.unwrap(phase*2*np.pi/1023)*1023/(2*np.pi)
    residual-=round(residual[a]/1023)*1023
    i=np.searchsorted(starts,t,side='right')-1;j=np.clip(i,0,len(starts)-1)
    valid_flag=(i>=0)&(t<starts[j]+.001)
    result[valid_flag]=(anchor['tow_s']+(j[valid_flag]-a)*.001+
        (residual[j[valid_flag]]+rate[j[valid_flag]]*(t[valid_flag]-starts[j[valid_flag]]))/1.023e6)
    return result


def solve_gps_epoch(ephemerides, transmit_tow_s, receiver_tow_s, *, atmosphere=None):
    """Code SPP with broadcast clocks/TGD and iterated Sagnac correction.

    Receiver TOW is an arbitrary common clock origin near GNSS time; its offset
    is solved, not supplied from truth. Atmosphere=None means uncorrected.
    """
    tx=np.asarray(transmit_tow_s,float)
    raw=C*np.array([week_difference(receiver_tow_s-v) for v in tx])
    positions=[];clocks=[]
    for eph,t in zip(ephemerides,tx):
        if eph['health']!=0 or abs(week_difference(t-eph['toe']))>7200:
            raise ValueError('unhealthy or stale ephemeris')
        gps_t=t
        for _ in range(3):
            xyz,clock=satellite_state(eph,gps_t);gps_t=t-clock
        xyz,clock=satellite_state(eph,gps_t)
        positions.append(xyz);clocks.append(clock-eph['tgd'])
    corrected=raw+C*np.array(clocks)
    travel=np.full(len(tx),.075)
    delay=np.zeros(len(tx))
    for _ in range(5):
        states=np.array([reception_frame(p,tau) for p,tau in zip(positions,travel)])
        sol=solve_position(states,corrected-delay)
        travel=np.linalg.norm(states-sol['position_m'],axis=1)/C
        if atmosphere is not None:
            delay=np.array([atmosphere(sol['position_m'],p) for p in states])
    return {**sol,'raw_pseudorange_m':raw,'corrected_pseudorange_m':corrected-delay,
        'satellite_positions_m':states,'satellite_clock_correction_s':np.array(clocks),
        'atmosphere_delay_m':delay,'receiver_tow_s':receiver_tow_s,
        'gps_receive_tow_s':receiver_tow_s-sol['clock_bias_m']/C}
