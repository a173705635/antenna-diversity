"""Conditional GF lifecycle -> dual-antenna L1 code PVT, archived RF replay.

GF uses four streams but applies the same multiplier to both L1 observations
of each satellite: the combination cannot identify which antenna is wrong.
Calibration events and parameters are illustrative, not commercially validated.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import BASELINE
from gf_runtime import GFRuntime
from positioning import solve_position
from run_gps_spp import receiver_reference, serializable

ROOT=Path(__file__).resolve().parents[1]
LOS=ROOT/'docs/reports/data/2026-09-12-geometry-free-rf'
STATIC=ROOT/'docs/reports/data/2026-09-12-gf-startup-rf'
OUT=ROOT/'docs/reports/data/2026-09-12-gf-pvt-lifecycle'


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    inputs={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding='utf-8'))
    for directory in [LOS,STATIC]:
        for name,digest in read(directory/'sha256.json').items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest: raise ValueError('hash mismatch')
    a=read(LOS/'L1CA_0.json'); clean=read(LOS/'L1CA_1.json'); dirty=read(STATIC/'L1CA_1_static.json')
    gf=read(STATIC/'summary.json'); sats=read(LOS/'manifest.json')['satellites']
    t=np.asarray(a['time_s']); reference=receiver_reference()[0]
    rows=[]
    for scene,b,key in [('los',clean,'los_gf_m'),('startup',dirty,'static_gf_m')]:
        if not np.array_equal(t,b['time_s']) or not np.array_equal(t,gf['time_s']): raise ValueError('unaligned times')
        codes=np.concatenate([a['pseudoranges_m'],b['pseudoranges_m']],axis=1)
        residual=np.asarray(gf[key]); validity=np.asarray(gf['valid'],bool)
        results={}
        for mode in ['base','conditional_confirmed','calibration_outage','unconfirmed']:
            runtimes=[GFRuntime(.025,.25) for _ in sats]
            weights=np.ones(residual.shape); labels=np.full(residual.shape,-1)
            errors=[]
            for k,time in enumerate(t):
                for j,runtime in enumerate(runtimes):
                    if mode=='base': continue
                    outage=mode=='calibration_outage' and 10<=k<15
                    confirm=(k==0 and mode!='unconfirmed') or (mode=='calibration_outage' and k==15)
                    result=runtime.step(float(time),float(residual[k,j]),.010,
                        valid=bool(validity[k,j]),arc_id='paired_arc',calibration_id='illustrative',
                        calibration_valid=not outage,nuisance_bound_m=.003,
                        branch_confirmation=('paired_arc','illustrative') if confirm else None)
                    weights[k,j]=result['base_weight_multiplier']; labels[k,j]=result['state']
                states=np.array([np.asarray(s['position_m'])+np.asarray(s['velocity_mps'])*time for s in sats])
                geometry=np.concatenate([states,states-BASELINE])
                sigma=30/np.sqrt(np.tile(weights[k],2))
                valid=np.isfinite(codes[k])
                try:
                    solution=solve_position(geometry[valid],codes[k,valid],sigma_m=sigma[valid])
                    errors.append(float(np.linalg.norm(solution['position_m']-reference)))
                except ValueError: errors.append(float('nan'))
            e=np.asarray(errors); good=np.isfinite(e)
            item={'solved':int(good.sum()),'epochs':len(t),'error_m':e,
                  'multipliers':weights,'state':labels}
            if good.any(): item.update(rms_m=float(np.sqrt(np.mean(e[good]**2))),
                p95_m=float(np.quantile(e[good],.95)),max_m=float(e[good].max()))
            results[mode]=item
        base=np.asarray(results['base']['error_m'])
        for mode,item in results.items():
            delta=np.asarray(item['error_m'])-base
            good=np.isfinite(delta)
            item['worse_than_base_epochs']=int(np.count_nonzero(delta[good]>1e-8))
            item['max_error_increase_m']=float(max(0.,delta[good].max())) if good.any() else None
        rows.append({'scene':scene,'modes':results})
    output={'scope':__doc__,'threshold_m':.01,'bound_m':.003,'multiplier':.25,
        'outage_epochs':[10,14],'reconfirm_epoch':15,'rows':rows,'inputs':inputs,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('gf_runtime.py'),Path(__file__).with_name('positioning.py')]}}
    (OUT/'summary.json').write_text(json.dumps(serializable(output),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps([{ 'scene':r['scene'],'modes':{m:{k:v for k,v in d.items() if k not in
        ['error_m','multipliers','state']} for m,d in r['modes'].items()}} for r in rows],indent=2))


if __name__=='__main__':main()
