"""Same-record weight ablation. Antenna-specific modes use scenario knowledge: diagnostics only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import BASELINE
from positioning import solve_position
from run_gps_spp import receiver_reference, serializable

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/reports/data/2026-09-12-gf-weight-ablation'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(name):
        p=ROOT/'docs/reports/data'/name
        inputs[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text(encoding='utf-8'))
    a=read('2026-09-12-geometry-free-rf/L1CA_0.json')
    b=read('2026-09-12-gf-startup-rf/L1CA_1_static.json')
    clean=read('2026-09-12-geometry-free-rf/L1CA_1.json')
    previous=read('2026-09-12-gf-pvt-lifecycle/summary.json')
    sats=read('2026-09-12-geometry-free-rf/manifest.json')['satellites']
    gf=read('2026-09-12-gf-startup-rf/summary.json')
    t=np.asarray(a['time_s']); reference=receiver_reference()[0]
    for r in [b,clean,gf]:
        if not np.array_equal(t,r['time_s']): raise ValueError('unaligned times')
    weights=np.asarray(previous['rows'][1]['modes']['conditional_confirmed']['multipliers'])
    pa,pb,pc=[np.asarray(r['pseudoranges_m']) for r in [a,b,clean]]
    n=len(sats); ones=np.ones_like(weights)
    cases={'base':np.concatenate([ones,ones],axis=1),
           'GF_pair':np.concatenate([weights,weights],axis=1),
           'GF_B_only_oracle':np.concatenate([ones,weights],axis=1),
           'GF_A_only_oracle':np.concatenate([weights,ones],axis=1)}
    always=np.ones((len(t),2*n)); always[:,n]=.25
    cases['always_B_PRN1_oracle']=always
    results={}
    for name,w in cases.items():
        errors=[]; conditions=[]
        for k,time in enumerate(t):
            states=np.array([np.asarray(s['position_m'])+np.asarray(s['velocity_mps'])*time for s in sats])
            geom=np.concatenate([states,states-BASELINE]); codes=np.r_[pa[k],pb[k]]
            solution=solve_position(geom,codes,sigma_m=30/np.sqrt(w[k]))
            errors.append(float(np.linalg.norm(solution['position_m']-reference)))
            conditions.append(solution['geometry_condition'])
        results[name]={'rms_m':float(np.sqrt(np.mean(np.square(errors)))),
                       'p95_m':float(np.quantile(errors,.95)),'error_m':errors,
                       'condition_mean':float(np.mean(conditions))}
    np.testing.assert_allclose(results['GF_pair']['error_m'],previous['rows'][1]['modes']['conditional_confirmed']['error_m'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(results['base']['error_m'],previous['rows'][1]['modes']['base']['error_m'],atol=1e-8,rtol=0)
    active=weights[:,0]<1
    delta=pb[:,0]-pc[:,0]
    # Pair difference isolates combined injection/receiver response, not pure analytical path bias.
    diagnostics={'PRN1_gf_active_epochs':int(active.sum()),
        'B_PRN1_paired_code_change_m':delta,
        'paired_code_change_rms_active_m':float(np.sqrt(np.mean(delta[active]**2))),
        'paired_code_change_rms_inactive_m':float(np.sqrt(np.mean(delta[~active]**2))),
        'abs_GF_vs_abs_paired_code_change_correlation':float(np.corrcoef(abs(np.asarray(gf['static_gf_m'])[:,0]),abs(delta))[0,1])}
    # Audit only: source position and clock are introduced here, after all solvers/weights.
    for label,codes,shift in [('A',pa,0),('B_static',pb,1),('B_los',pc,1)]:
        states=np.array([np.asarray(sats[0]['position_m'])+np.asarray(sats[0]['velocity_mps'])*time for time in t])
        error=codes[:,0]-np.linalg.norm(states-reference-shift*BASELINE,axis=1)-75000
        diagnostics[label+'_PRN1_true_code_rms_m']=float(np.sqrt(np.mean(error**2)))
    data={'scope':__doc__,'modes':results,'diagnostics':diagnostics,'inputs':inputs,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'warning':'20 correlated epochs, one noise realization; oracle modes are not deployable policies.'}
    (OUT/'summary.json').write_text(json.dumps(serializable(data),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(serializable({'modes':{k:{x:y for x,y in v.items() if x!='error_m'} for k,v in results.items()},
        'diagnostics':diagnostics}),indent=2))


if __name__=='__main__':main()
