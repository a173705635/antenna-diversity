"""Replay saved RF quality through SDCP arc lifecycle, with explicit test approvals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sdcp_arc import SDCPArc


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'docs/reports/data/2026-09-13-sdcp-tracking-gap'
    a=np.load(source/'A.npz');ta=a['epoch_start_samples']/6.6e6
    qa=(a['phase_lock_metric']>=.8)&(abs(a['pll_error_rad'])<np.pi/2)
    rows=[];traces={}
    for name in ['B_clean','B_gap','B_gap_phase','B_gap_reacquired','B_gap_phase_reacquired']:
        b=np.load(source/(name+'.npz'));tb=b['epoch_start_samples']/6.6e6
        qb=(b['phase_lock_metric']>=.8)&(abs(b['pll_error_rad'])<np.pi/2)
        # Quality pairing only: closest preceding A epoch within 1ms, not phase interpolation.
        idx=np.searchsorted(ta,tb,side='right')-1;safe=np.clip(idx,0,len(ta)-1)
        quality=qb&qa[safe]&(idx>=0)&(tb-ta[safe]<.001000001)
        manager=SDCPArc(name,.0015);trace=[];issued=False
        for i,t in enumerate(tb):
            r=manager.step(float(t),quality_valid=bool(quality[i]),phase_state_valid=True,
                tracking_ids=('A-running',name),calibration_id='simulation-fixed-zero-bias')
            # Reference proof is only an explicit pre-outage trusted-start assumption.
            if not issued and .2<=t<.29 and quality[i]:
                manager.accept_reference(r['arc_id'],'pre-outage-reference','test-initial-proof');issued=True
            trace.append({'time_s':float(t),**r})
        tail=[r for r in trace if .65<=r['time_s']<.79]
        rows.append({'record':name,'tail_epochs':len(tail),'quality_pass_tail':int(quality[(tb>=.65)&(tb<.79)].sum()),
            'reference_usable_tail':sum(r['state']=='reference_usable' for r in tail),
            'await_reference_tail':sum(r['state']=='await_reference' for r in tail),
            'arcs_opened':manager.arc_number,'initial_reference_assumption_applied':issued})
        traces[name]=trace
    assert rows[0]['reference_usable_tail']==rows[0]['tail_epochs']
    assert all(r['reference_usable_tail']==0 for r in rows[1:])
    result={'scope':__doc__,'rows':rows,
        'limitations':['phase_state_valid=True is a simulation interface assumption, not receiver-proven half-cycle validity',
                      'Only initial continuous record has explicit trusted-start test approval; no approval after gap/reacquisition',
                      'No source truth is used to restore any post-outage reference; no new RF generated'],
        'input_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['A.npz']+[r['record']+'.npz' for r in rows]},
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('sdcp_arc.py')]}}
    for name,data in [('summary.json',result),('trace.json',traces)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
