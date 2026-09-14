"""Deterministic SDCP semantics/state audit; no new RF or detector-rate claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sdcp import modified_sdcp, calibrated_sdcp
from sdcp_self_check import detect, SelfCheckConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    wave=299792458/1575.42e6;t=np.arange(120)*.01;g=-.03+.001*t
    baseline=g+3*wave
    cases={'clean':baseline,'startup_bias_8mm':baseline+.008,
           'late_bias_8mm':baseline+.008*(t>=.6),
           'integer_jump':baseline+wave*(t>=.6),
           'half_jump':baseline+wave/2*(t>=.6)}
    rows=[];raw={}
    for name,d in cases.items():
        ref=modified_sdcp(t,d*2*np.pi/wave,wave,g)['residual_m']
        wrap=calibrated_sdcp(d,g,wave,0)
        raw[name]={'reference_m':ref.tolist(),'calibrated_m':wrap.tolist()}
        rows.append({'case':name,'reference_late_mean_mm':float(ref[t>=.8].mean()*1000),
                     'calibrated_late_abs_mean_mm':float(abs(wrap[t>=.8]).mean()*1000)})
    assert max(abs(np.array(raw['startup_bias_8mm']['reference_m'])))<1e-12
    assert max(abs(np.array(raw['integer_jump']['calibrated_m'])))<1e-12
    # Same residual can be explained by multipath, geometry error, or hardware.
    assert np.allclose(calibrated_sdcp(baseline+.008,g,wave,0),
                       calibrated_sdcp(baseline,g-.008,wave,0))
    config=SelfCheckConfig(window=20,block=10,confirmation=3)
    states=[]
    for kind in ['integer_jump','half_jump','validity_gap','untrusted_after_gap']:
        r=np.zeros((120,8));valid=np.ones_like(r,bool)
        if kind=='integer_jump':r[60:,0]=wave
        if kind=='half_jump':r[60:,0]=wave/2
        if 'gap' in kind:valid[60:65,0]=False
        if kind=='untrusted_after_gap':valid[65:,0]=False
        result=detect(t,r,wave,.004,valid=valid,geometry_uncertainty_m=0.,channel_skew_s=0.,config=config)
        later=np.flatnonzero((t>=.65)&(result['state'][:,0]==0))
        states.append({'case':kind,'late_states':np.unique(result['state'][90:,0]).tolist(),
                       'first_clear_at_or_after_650ms_s':float(t[later[0]]) if len(later) else None})
    assert states[0]['late_states']==[0]
    assert states[1]['late_states']==[-1]
    assert states[2]['late_states']==[0]
    assert states[3]['late_states']==[-1]
    result={'scope':__doc__,'wavelength_m':wave,'residual_cases':rows,'state_cases':states,
        'sensitivity_not_measurements':{
            'geometry_vector_error_bound_mm':{str(deg):float(2*.0759*np.sin(np.deg2rad(deg)/2)*1000) for deg in [1,3,5]},
            'time_skew_2000hz_mm':{str(us):float(wave*2000*us*1e-6*1000) for us in [1,10,100]},
            'hardware_phase_5deg_mm':float(wave*5/360*1000)},
        'notes':['valid must include caller phase-arc validation; gap recovery is not authentication',
                 'geometry bound assumes baseline or LOS angular perturbation of given size, not false alarm rate',
                 'time-skew scale uses accumulated carrier at 2000Hz; not a complete oscillator error budget'],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('sdcp.py'),Path(__file__).with_name('sdcp_self_check.py')]}}
    for name,data in [('summary.json',result),('residuals.json',raw)]:
        (out/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
