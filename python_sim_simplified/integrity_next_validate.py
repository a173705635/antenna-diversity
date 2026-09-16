"""Frozen-seed validation and explicit violations of proposed integrity claims."""
import hashlib
import json
from pathlib import Path
import numpy as np
from integrity_next import receive
from integrity_next_model import make_case, score, NAMES

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/research-next-hundred'
SEEDS=[72121,72127,72131,72139]

def counterexamples(seed):
    out={}
    o,y=make_case('bounded',seed)
    o['acceleration']+=.2
    out['unreported_runtime_bias']=(o,y)
    o,y=make_case('false_stop',seed)
    o['stationary_origin']='fixture'
    out['falsely_certified_stationarity']=(o,y)
    o,y=make_case('anchor_recovery',seed)
    for anchor in o['anchors'].values(): anchor['z']+=200
    out['falsely_certified_absolute_source']=(o,y)
    o,y=make_case('bounded',seed)
    # Regression: first freeze failed because it substituted current acceleration
    # for the causal historical integral. Corrected source must enclose this.
    a=np.zeros_like(o['acceleration']); a[18,0]=100
    p=np.zeros_like(a); v=np.zeros_like(a)
    for k in range(1,len(a)):
        p[k]=p[k-1]+v[k-1]+a[k]/2; v[k]=v[k-1]+a[k]
    o['acceleration']=a+np.array([.018,-.012,.009])
    o['initial_position']=np.zeros(3); o['initial_velocity']=np.zeros(3)
    o['initial_radius']=np.full(3,1000.)
    o['velocities']={20:(v[20],np.full(3,.01))}
    o['anchors']={20:dict(z=p[17],radius=np.ones(3),lag=3,origin='independent_synthetic')}
    out['delayed_anchor_acceleration_history']=(o,p)
    o2={**o}; a2=np.zeros_like(a)
    a2[17:21]=np.array([[20.,-5.,8.],[-30.,12.,-2.],[60.,5.,-15.],[-5.,-1.,3.]])
    p2=np.zeros_like(a); v2=np.zeros_like(a)
    for k in range(1,len(a)):
        p2[k]=p2[k-1]+v2[k-1]+a2[k]/2; v2[k]=v2[k-1]+a2[k]
    o2['acceleration']=a2+np.array([.018,-.012,.009])
    o2['velocities']={20:(v2[20],np.full(3,.01))}
    o2['anchors']={20:dict(z=p2[17],radius=np.ones(3),lag=3,origin='independent_synthetic')}
    out['delayed_anchor_time_varying_history']=(o2,p2)
    return out

def main():
    source=(ROOT/'integrity_next.py').read_bytes()
    result=dict(source_sha256=hashlib.sha256(source).hexdigest(), seeds=SEEDS,
                domain='Synthetic bounded declarations, 600 seconds, 3-axis leakage model. No phone certificate.',
                selection='Iteration175 implementation correction uses causal historical acceleration transport; first failed freeze preserved separately. Not hardware validated.',
                metrics={},counterexamples={})
    traces={}
    for seed in SEEDS:
        for name in NAMES:
            o,y=make_case(name,seed); r=receive(o); key=f'{name}_{seed}'
            result['metrics'][key]=score(r,y)
            traces[key+'_position']=r['position']; traces[key+'_bound']=r['bound']; traces[key+'_normal']=r['normal']; traces[key+'_truth']=y
    for name,(o,y) in counterexamples(SEEDS[0]).items():
        r=receive(o); result['counterexamples'][name]=score(r,y)
        traces[name+'_position']=r['position']; traces[name+'_bound']=r['bound']; traces[name+'_normal']=r['normal']; traces[name+'_truth']=y
    ms=list(result['metrics'].values())
    result['aggregate']=dict(cases=len(ms), epochs=sum(x['epochs'] for x in ms),
                            normal_output_rate=float(np.mean([x['normal_output_rate'] for x in ms])),
                            full_success=sum(x['full_success'] for x in ms),
                            normal_ge30=sum(x['normal_ge30'] for x in ms),normal_ge50=sum(x['normal_ge50'] for x in ms),
                            bound_violations=sum(x['bound_violations'] for x in ms))
    (OUT/'integrity_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (OUT/'integrity_selected_source.py').write_bytes(source)
    np.savez_compressed(OUT/'integrity_validation_traces.npz',**traces)
    print(json.dumps(dict(aggregate=result['aggregate'],counterexamples=result['counterexamples']),indent=2))

if __name__=='__main__': main()
