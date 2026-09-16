"""Replay saved RF quality through SDCP arc lifecycle, with explicit test approvals."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sdcp_arc import SDCPArc


def consume_tracking(a, b, tracking_ids, *, session_id, reference_tracking_ids=None):
    """Replay actual NCO histories; initial trusted reference is an explicit assumption.

    A phase is propagated only within its current 1 ms epoch. Tracking start
    indices are observation times, not raw ADC continuity evidence.
    """
    fs=6.6e6;wavelength=299792458/1575.42e6
    sa=a['epoch_start_samples'];sb=b['epoch_start_samples'];tb=sb/fs
    idx=np.searchsorted(sa,sb,side='right')-1;safe=np.clip(idx,0,len(sa)-1)
    delta_samples=sb-sa[safe]
    aligned=(idx>=0)&(delta_samples<6600)
    qa=(a['phase_lock_metric']>=.8)&(abs(a['pll_error_rad'])<np.pi/2)
    qb=(b['phase_lock_metric']>=.8)&(abs(b['pll_error_rad'])<np.pi/2)
    quality=aligned&qa[safe]&qb
    phase_a=a['carrier_nco_phase_rad'][safe]+2*np.pi*a['carrier_nco_frequency_hz'][safe]*delta_samples/fs
    sd=-(b['carrier_nco_phase_rad']-phase_a)*wavelength/(2*np.pi)
    a_ids=['A-running']*len(sa) if reference_tracking_ids is None else reference_tracking_ids
    manager=SDCPArc(session_id,.0015);trace=[];issued=False;reference=None
    for i,t in enumerate(tb):
        r=manager.step(float(t),quality_valid=bool(quality[i]),phase_state_valid=True,
                       tracking_ids=(a_ids[safe[i]],tracking_ids[i]),calibration_id='simulation-fixed-zero-bias')
        if not issued and .2<=t<.29 and quality[i]:
            manager.accept_reference(r['arc_id'],'pre-outage-reference','test-initial-proof')
            issued=True;reference=float(sd[i])
            r.update(state='reference_usable',reference_id=manager.reference_id,evidence_id=manager.evidence_id)
        trace.append(dict(time_s=float(t),earliest_evidence_time_s=float((sb[i]+6600)/fs),
                          **r,tracking_id=tracking_ids[i],
                          quality_valid=bool(quality[i]),
                          raw_sd_m=float(sd[i]) if aligned[i] else None,
                          residual_m=float(sd[i]-reference) if r['state']=='reference_usable' else None))
    return trace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[1]/'docs/reports/data/2026-09-13-sdcp-tracking-gap'
    a=np.load(source/'A.npz')
    rows=[];traces={}
    names=['B_clean','B_gap','B_gap_phase','B_gap_reacquired','B_gap_phase_reacquired']
    records={name:dict(np.load(source/(name+'.npz'))) for name in names}
    for name in names+['B_gap_joined','B_gap_phase_joined']:
        if name.endswith('_joined'):
            base=name.removesuffix('_joined');old=records[base];new=records[base+'_reacquired']
            before=old['epoch_start_samples']<new['epoch_start_samples'][0]
            b={key:np.concatenate([old[key][before],new[key]]) for key in old}
            ids=[base]*int(before.sum())+[base+'_reacquired']*len(new['epoch_start_samples'])
        else:
            b=records[name];ids=[name]*len(b['epoch_start_samples'])
        trace=consume_tracking(a,b,ids,session_id=name)
        tail=[r for r in trace if .65<=r['time_s']<.79]
        rows.append({'record':name,'tail_epochs':len(tail),'quality_pass_tail':sum(r['quality_valid'] for r in tail),
            'reference_usable_tail':sum(r['state']=='reference_usable' for r in tail),
            'await_reference_tail':sum(r['state']=='await_reference' for r in tail),
            'residual_outputs_tail':sum(r['residual_m'] is not None for r in tail),
            'output_withdrawals':[{'time_s':current['time_s'],
                                   'earliest_evidence_time_s':current['earliest_evidence_time_s'],
                                   'reason':current['last_break_reason']}
                                  for previous,current in zip(trace,trace[1:])
                                  if previous['residual_m'] is not None and current['residual_m'] is None],
            'arcs_opened':len({r['arc_id'] for r in trace if r['arc_id'] is not None}),
            'initial_reference_assumption_applied':any(r['reference_id'] is not None for r in trace)})
        traces[name]=trace
    assert rows[0]['reference_usable_tail']==rows[0]['tail_epochs']
    assert all(r['reference_usable_tail']==0 for r in rows[1:])
    result={'scope':__doc__,'rows':rows,
        'limitations':['phase_state_valid=True is a simulation interface assumption, not receiver-proven half-cycle validity',
                      'earliest_evidence_time_s includes 1 ms integration only; processing and publication latency are unknown',
                      'Only initial continuous record has explicit trusted-start test approval; no approval after gap/reacquisition',
                      'No source truth is used to restore any post-outage reference; no new RF generated'],
        'input_sha256':{n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['A.npz']+[name+'.npz' for name in names]},
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('sdcp_arc.py')]}}
    for name,data in [('summary.json',result),('trace.json',traces)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    (out/'sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()},indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
