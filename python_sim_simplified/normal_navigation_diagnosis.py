"""Read-only navigation instrumentation; assert outputs equal frozen results."""
import numpy as np
from satellite_count_study import ROOT,OLD
from single_source_twenty import read,save,load


source=(ROOT/'receiver.py').read_text(encoding='utf-8')
source=source.replace('out=np.full((len(z[\'time\']),3),np.nan);','trace=[];out=np.full((len(z[\'time\']),3),np.nan);')
source=source.replace('separation=np.inf','separation=np.inf;measured_p=np.full(3,np.nan);common_count=0;phase_residual_max=None')
source=source.replace('fresh=p.copy()','fresh=p.copy();measured_p=p.copy()')
source=source.replace("common &= z['arc'][k]==z['arc'][k-1]","common &= z['arc'][k]==z['arc'][k-1];common_count=int(common.sum())")
source=source.replace('variance=max(.02**2','phase_residual_max=float(np.max(abs(phase_residual)))\n                        variance=max(.02**2')
needle='            if not isolated and (separation<=20. or phase_ok and k-anchor<=20):out[k]=p'
source=source.replace(needle,needle+"\n        trace.append(dict(k=k,available_time=float(z['available_time'][k]),valid=int(valid[k].sum()),common=common_count,separation=float(separation) if np.isfinite(separation) else None,phase_ok=bool(phase_ok),phase_residual_max=phase_residual_max,isolated=bool(isolated),anchor_age=k-anchor,code_position=measured_p.tolist(),published=bool(np.isfinite(out[k]).all()),candidate_finite=bool(np.isfinite(p).all()),spatial_confirmed=int((valid[k]&(z['support'][k]>=.85)).sum())))")
source=source.replace('return dict(position=out,','return dict(trace=trace,position=out,')
path=ROOT/'instrumented_navigation.py';path.write_text(source,encoding='utf-8');model=load(path)
rows=[]
for subset in ('three','four'):
    row=next(r for r in read(OLD/f'corrective_{subset}.json') if r['case']==dict(seed=114001,kind='normal',duration=2.))
    with np.load(OLD/'evaluations'/row['identity']/'tracking.npz') as raw:z={k:raw[k] for k in raw.files}
    result=model.navigate(z)
    with np.load(OLD/'evaluations'/row['identity']/'navigation.npz') as n:
        np.testing.assert_allclose(result['position'],n['position'],rtol=0,atol=1e-9,equal_nan=True)
    for d in result['trace']:
        q=d['code_position'];d['code_error_m']=float(np.linalg.norm(np.array(q[:2])-[2*d['available_time'],0.])) if np.isfinite(q).all() else None
    post=[t for t in result['trace'] if t['available_time']>=.3]
    missing=[t for t in post if not t['published']]
    summary=dict(subset=subset,missing=len(missing),projection_fraction=float(np.mean(z['ratio']>=3)),
                 median_gain=float(np.median(z['gain'])),median_valid=float(np.median([t['valid'] for t in post])),
                 direct_rejection=sum(not t['candidate_finite'] for t in missing),
                 held_while_candidate_finite=sum(t['candidate_finite'] and t['isolated'] for t in missing))
    rows.append(dict(summary=summary,trace=result['trace']))
    print(summary,flush=True)
    print('first missing',missing[:4],flush=True)
save(ROOT/'normal_navigation_diagnosis.json',rows)
