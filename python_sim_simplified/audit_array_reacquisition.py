"""Four RF channels, three PRNs: autonomous reacquisition and surviving pairs.

Fixed 2/3/4-channel subsets, all pair references established before the outage.
No post-fault admission, automatic reference switch, or navigation solution.
"""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from audit_spatial_diversity_rf import channel_parameters
from audit_sdcp_arc import consume_tracking
from autonomous_tracking import receive
from dfe import process_dfe
from presets import mate70_phase_centers
from rf import process_rf
from run_gps_spp import satellite_setup, build_scene, serializable


def available_at(trace, times):
    """Fixed evaluation clock: missing, stale and not-yet-ready outputs count false."""
    ready=np.array([r['earliest_evidence_time_s'] for r in trace])
    usable=np.array([r['residual_m'] is not None for r in trace])
    index=np.searchsorted(ready,times,side='right')-1
    selected=np.clip(index,0,len(ready)-1)
    return (index>=0)&(times-ready[selected]<.001)&usable[selected]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (out/name).write_text(json.dumps(serializable(data),indent=2,allow_nan=False),encoding='utf-8')
    sats=satellite_setup()[:3];names=('TL','TR','BL','BR');runs={};receivers=[]
    save('manifest.json',dict(scope=__doc__,prns=[s['prn'] for s in sats],
                             positions_m=mate70_phase_centers(),duration_ms=800,outage_s=[.30,.45],
                             seeds=[10800+i for i in range(4)],loss_confirm_epochs=5,loss_settle_epochs=100,
                             source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                            for n in ('audit_array_reacquisition.py','autonomous_tracking.py','tracking.py','audit_sdcp_arc.py')}))
    # Unchanged channels reuse their exact clean samples/history in paired comparisons.
    for channel,gap in [(i,False) for i in range(4)]+[(0,True),(1,True)]:
        key=f'{names[channel]}_'+('gap' if gap else 'clean')
        genuine,_=channel_parameters(sats,channel)
        scene,cfg=build_scene(genuine,800)
        fs0=cfg.signal.fs_rf_hz
        if gap:scene[:,round(.30*fs0):round(.45*fs0)]=0
        adc,_=process_rf(scene,cfg.rf,cfg.reference_impedance_ohm,10800+channel);del scene
        x,info=process_dfe(adc,cfg.dfe,fs0,cfg.signal.if_hz);del adc
        fs=info['fs_out_hz'];x=x[0];runs[key]={}
        for sat in sats:
            prn=sat['prn']
            result=receive(x,fs,prn,cfg.signal.code_rate_hz,cfg.signal.carrier_hz,cfg.acquisition,cfg.tracking)
            segments=result['segments']
            arrays=[{k:v for k,v in s['history'].items() if isinstance(v,np.ndarray)} for s in segments]
            history={k:np.concatenate([h[k] for h in arrays]) for k in arrays[0]}
            ids=[s['tracking_id'] for s in segments for _ in s['history']['epoch_start_samples']]
            runs[key][prn]=(history,ids)
            np.savez_compressed(out/f'{key}_prn{prn}.npz',**history,tracking_ids=ids)
            save(f'{key}_prn{prn}_events.json',result['events'])
            receivers.append(dict(record=key,prn=prn,tracking_instances=len(segments),
                                  failed_acquisitions=sum(e['kind']=='acquisition' and not e['success'] for e in result['events']),
                                  tracking_start_s=[s['history']['epoch_start_samples'][0]/fs for s in segments]))
        del x
        print(key+' done',flush=True)
    times=np.arange(651,791)/1000.;summaries=[];pair_metrics=[];masks={}
    for case,fault in [('clean',None),('TL_gap',0),('TR_gap',1)]:
        records=[runs[f'{name}_'+('gap' if i==fault else 'clean')] for i,name in enumerate(names)]
        pair_available={}
        for a,b in combinations(range(4),2):
            columns=[]
            for prn in (1,2,3):
                ha,ia=records[a][prn];hb,ib=records[b][prn]
                trace=consume_tracking(ha,hb,ib,session_id=f'{case}-{a}-{b}-{prn}',reference_tracking_ids=ia)
                mask=available_at(trace,times);columns.append(mask)
                pair_metrics.append(dict(case=case,pair=[names[a],names[b]],prn=prn,
                                         available=int(mask.sum()),opportunities=len(times)))
            pair_available[a,b]=np.column_stack(columns)
            masks[f'{case}_{a}_{b}']=pair_available[a,b]
        for count in (2,3,4):
            cube=np.stack([pair_available[p] for p in combinations(range(count),2)],axis=1)
            summaries.append(dict(case=case,channels=count,available_pair_prn_epochs=int(cube.sum()),
                                  opportunities=int(cube.size),fraction=float(cube.mean()),
                                  epochs_with_any_pair_all_3_prns=int(cube.all(axis=2).any(axis=1).sum()),
                                  evaluation_epochs=len(times)))
    np.savez_compressed(out/'availability.npz',time_s=times,**masks)
    save('summary.json',dict(rows=summaries,pairs=pair_metrics,receivers=receivers,
                             limitations=['Pair availability is not spoof detection rate or navigation accuracy',
                                          'Three genuine PRNs, four synthetic channels, one seed per channel',
                                          'All surviving pair references were established before the outage',
                                          'No real-time compute budget or post-fault reference authentication']))
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
