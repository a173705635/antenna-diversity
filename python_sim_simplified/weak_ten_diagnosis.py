"""Trace unchanged navigation decisions and check against archived positions."""
import sys
from pathlib import Path
import numpy as np
from weak_ten_study import ROOT,read,save,load


def diagnose(path,label,model=None):
    row=read(path);base=Path(row[label+'_root']);cache=base/'evaluations'/row[label+'_id']
    with np.load(cache/'tracking.npz') as data:z={k:data[k] for k in data.files}
    source=(Path(model) if model else ROOT/'baseline.py').read_text(encoding='utf-8')
    threshold=3. if model else .85
    source=source.replace("out=np.full((len(z['time']),3),np.nan);","trace=[];out=np.full((len(z['time']),3),np.nan);")
    source=source.replace('separation=np.inf','separation=np.inf;common_count=0;phase_residual_max=None')
    source=source.replace("common &= z['arc'][k]==z['arc'][k-1]","common &= z['arc'][k]==z['arc'][k-1];common_count=int(common.sum())")
    source=source.replace('variance=max(.02**2','phase_residual_max=float(np.max(abs(phase_residual)))\n                        variance=max(.02**2')
    needle='            if not isolated and (separation<=20. or phase_ok and k-anchor<=20):out[k]=p'
    source=source.replace(needle,needle+"\n        trace.append(dict(time=float(z['available_time'][k]),valid=int(valid[k].sum()),common=common_count,separation=float(separation) if np.isfinite(separation) else None,phase_ok=bool(phase_ok),phase_residual_max=phase_residual_max,isolated=bool(isolated),anchor_age=k-anchor,published=bool(np.isfinite(out[k]).all()),candidate=bool(np.isfinite(p).all()),spatial_confirmed=int((valid[k]&(z['support'][k]>=.85)).sum())))")
    source=source.replace('return dict(position=out,','return dict(trace=trace,position=out,')
    source=source.replace("z['support'][k]>=.85",f"z['support'][k]>={threshold}")
    instrumented=ROOT/'instrumented_baseline.py';instrumented.write_text(source,encoding='utf-8')
    result=load(instrumented).navigate(z)
    with np.load(cache/'navigation.npz') as n:np.testing.assert_allclose(result['position'],n['position'],rtol=0,atol=1e-9,equal_nan=True)
    trace=[r for r in result['trace'] if r['time']>=.34]
    summary=dict(input=row['input'],subset=row['subset'],epochs=len(trace),
                 published=sum(r['published'] for r in trace),candidate=sum(r['candidate'] for r in trace),
                 six_stars=sum(r['valid']>=6 for r in trace),code_consistent=sum(r['separation'] is not None and r['separation']<=20 for r in trace),
                 phase_ok=sum(r['phase_ok'] for r in trace),spatial_confirmed=sum(r['spatial_confirmed']>=6 for r in trace),
                 recovery_held=sum(r['isolated'] and r['candidate'] for r in trace))
    output=path.with_name(path.stem+'_diagnosis.json');save(output,dict(summary=summary,trace=result['trace']));print(summary,flush=True)


if __name__=='__main__':diagnose(Path(sys.argv[1]),sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None)
