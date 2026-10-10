"""The PLL candidate must affect only receive-stage tracking configuration."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from core import single_source
from core.config import TrackingConfig
from core.recovery_geometry import FS
from core.tracking import _TrackingState, _update_tracking_loops


def test_candidate_pll_is_not_used_for_startup_tracking(monkeypatch):
    startup=[];received=[]
    monkeypatch.setattr(single_source,'acquire_array',lambda *args,**kwargs:dict(success=True))
    def startup_track(signal,fs,prn,acquisition,cfg,*args):
        startup.append((cfg.pll_kp,cfg.pll_ki_hz))
        return dict(success=True,epoch_start_samples=np.array([0]),
                    code_nco_phase_chips=np.array([0.]),code_nco_rate_hz=np.array([1.023e6]),
                    carrier_nco_frequency_hz=np.array([0.]))
    monkeypatch.setattr(single_source,'track',startup_track)
    monkeypatch.setattr(single_source,'local_signal',
                        lambda code,template,t:np.ones(len(t),complex))
    monkeypatch.setattr(single_source,'calibrate',
                        lambda block,local,desired:(np.ones(len(block),complex),np.zeros(len(block))))
    monkeypatch.setattr(single_source,'beam',
                        lambda block,a,local,cov,prns:(a/len(a),np.ones(a.shape[1]),0.))
    def receive(signal,prn,acquisition_cfg,tracking_cfg):
        received.append((tracking_cfg.pll_kp,tracking_cfg.pll_ki_hz))
        return dict(segments=[])
    monkeypatch.setattr(single_source,'receive_measured',receive)
    samples=np.zeros((3,round(.12*FS)),complex)
    record=dict(samples=samples,nominal=np.ones((3,4),complex),
                codes=np.zeros((4,1)),times=np.array([.105]),
                satellites=np.zeros((1,4,3)),coarse=np.zeros(4))
    single_source.frontend(record,[0,1,2],receiver_pll=(.15,1.))
    assert startup==[(.30,4.)]*4
    assert received==[(.15,1.)]*4


PLL25 = (.06315066862415342,.32776091874805263)


def closed_loop(offset_hz,rate_hz_s,fll_gain=.5,pll=(.15,1.),
                epochs=1000,step_hz=0.):
    cfg=TrackingConfig(pll_kp=pll[0],pll_ki_hz=pll[1],fll_gain=fll_gain)
    state=_TrackingState(0.,0.,0.,1.023e6)
    truth_phase=0.;previous=None
    frequencies=[];errors=[]
    for epoch in range(epochs):
        true_frequency=offset_hz+rate_hz_s*epoch*.001+(step_hz if epoch>=500 else 0.)
        truth_phase+=2*np.pi*true_frequency*.001
        prompt=np.exp(1j*(truth_phase-state.carrier_phase_rad))
        phase_error=.5*np.angle(prompt**2)
        fll_error=(0. if previous is None else
                   .5*np.angle((prompt*np.conj(previous))**2)/(2*np.pi*.001))
        _update_tracking_loops(state,prompt,0.,phase_error,fll_error,epoch,.001,
                               2046,FS,cfg,1575.42e6,1.023e6)
        frequencies.append(state.carrier_frequency_hz)
        errors.append(phase_error)
        previous=prompt
    return np.array(frequencies),np.array(errors),state


def test_ideal_prompt_loop_converges_with_fll_start_and_fixed_offset():
    assisted,errors,state=closed_loop(20.,0.)
    unassisted,_,_=closed_loop(20.,0.,fll_gain=0.)
    assert assisted[1]>unassisted[1]
    assert abs(assisted[-1]-20.)<.01
    assert max(abs(errors[-100:]))<.01
    assert state.code_rate_hz>1.023e6


def test_ideal_prompt_loop_tracks_both_100_hz_per_second_slopes():
    for rate in (-100.,100.):
        frequencies,errors,_=closed_loop(0.,rate)
        assert abs(frequencies[-1]-rate*.999)<3.
        assert max(abs(errors[-100:]))<.11


def test_pll25_discrete_noise_bandwidth_is_25_hz():
    kp,ki=PLL25;dt=.001
    transition=np.array([[1-kp-2*np.pi*dt*ki,2*np.pi*dt],[-ki,1.]])
    noise=np.array([[-kp-2*np.pi*dt*ki],[-ki]])
    covariance=solve_discrete_lyapunov(transition,noise@noise.T)
    np.testing.assert_allclose(covariance[0,0]/(2*dt),25.,rtol=1e-11)


def test_pll25_actual_loop_handles_grid_edge_slopes_and_step():
    for offset in (-125.,125.):
        frequency,error,_=closed_loop(offset,0.,pll=PLL25,epochs=1500)
        assert abs(frequency[-1]-offset)<.01
        assert max(abs(error[-100:]))<.01
        assert np.sign(frequency[1])==np.sign(offset)
    for rate in (-100.,100.):
        frequency,error,_=closed_loop(0.,rate,pll=PLL25,epochs=1500)
        assert abs(frequency[-1]-rate*1.499)<3.
        assert max(abs(error[-100:]))<.31
    for step in (-40.,40.):
        frequency,error,_=closed_loop(0.,0.,pll=PLL25,epochs=1500,step_hz=step)
        assert abs(frequency[-1]-step)<.01
        assert max(abs(error[500:650]))<np.pi/2
        assert max(abs(error[-100:]))<.01
