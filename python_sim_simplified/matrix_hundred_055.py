from matrix_hundred_core import *
PROPOSAL=dict(hypothesis='A separately measured coherent pilot reference removes the source-path/common-hardware delay ambiguity.',
 change='Add explicit reference-monitor output and deembed it before estimating the hardware matrix.',
 input_model='Same 120 ns reference delay; matrix observations noise 0.003; independently observed reference transfer noise 0.002; monitor path delay assumed known, no bench support.')
def experiment(seed=41060):
    d=scene(seed);f=np.linspace(-1e6,1e6,13);r=np.exp(-2j*np.pi*f*120e-9)
    truth=np.broadcast_to(d['h'],(13,4,4));y=r[:,None,None]*truth+.003*noise(d['rng'],truth.shape)
    monitor=r+.002*noise(d['rng'],r.shape)
    corrected=y/monitor[:,None,None]
    measured_phase=np.unwrap(np.angle(monitor));estimated=-np.polyfit(f,measured_phase,1)[0]/(2*np.pi)
    return dict(unmonitored_error=relative_error(y,truth),monitored_error=relative_error(corrected,truth),
       measured_reference_delay_ns=float(estimated*1e9),delay_estimation_error_ns=float((estimated-120e-9)*1e9),
       external_requirement='Independent coherent monitor with characterized path delay; unverified hardware design.')
