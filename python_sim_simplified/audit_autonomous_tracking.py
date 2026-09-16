"""New digital RF experiment: quality-driven reacquisition and phase-arc consumption."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acquisition import acquire
from autonomous_tracking import receive
from audit_sdcp_arc import consume_tracking
from audit_sdcp_positioning import channel_satellites
from dfe import process_dfe
from rf import process_rf
from run_gps_spp import satellite_setup, build_scene, serializable
from tracking import track


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (out/name).write_text(json.dumps(serializable(data),indent=2,allow_nan=False),encoding='utf-8')
    save('manifest.json',dict(scope=__doc__,duration_ms=800,outage_s=[.30,.45],
                             seeds=[9800,9801],loss_confirm_epochs=5,loss_settle_epochs=100,
                             source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                            for name in ('audit_autonomous_tracking.py','autonomous_tracking.py','tracking.py','audit_sdcp_arc.py')}))
    sats=satellite_setup()[:1];summary=[]
    for label,ch,gap,phase in [('A',0,False,0.),('B_clean',1,False,0.),
                                ('B_gap',1,True,0.),('B_gap_phase',1,True,.6)]:
        scene,cfg=build_scene(channel_satellites(sats,ch),800)
        fs0=cfg.signal.fs_rf_hz
        if gap:
            scene[:,round(.30*fs0):round(.45*fs0)]=0
        scene[:,round(.45*fs0):]*=np.exp(1j*phase)
        adc,_=process_rf(scene,cfg.rf,cfg.reference_impedance_ohm,9800+ch);del scene
        x,info=process_dfe(adc,cfg.dfe,fs0,cfg.signal.if_hz);del adc
        fs=info['fs_out_hz'];x=x[0]
        if label=='A':
            acq=acquire(x,fs,1,cfg.signal.code_rate_hz,cfg.acquisition)
            a=track(x,fs,1,acq,cfg.tracking,cfg.signal.carrier_hz,cfg.signal.code_rate_hz)
            np.savez_compressed(out/'A.npz',**{k:v for k,v in a.items() if isinstance(v,np.ndarray)})
        else:
            result=receive(x,fs,1,cfg.signal.code_rate_hz,cfg.signal.carrier_hz,cfg.acquisition,cfg.tracking)
            segments=result['segments']
            arrays=[{k:v for k,v in s['history'].items() if isinstance(v,np.ndarray)} for s in segments]
            b={k:np.concatenate([h[k] for h in arrays]) for k in arrays[0]}
            ids=[s['tracking_id'] for s in segments for _ in s['history']['epoch_start_samples']]
            trace=consume_tracking(a,b,ids,session_id=label)
            tail=[r for r in trace if .65<=r['time_s']<.79]
            for i,h in enumerate(arrays):
                np.savez_compressed(out/f'{label}_{i}.npz',**h)
            save(label+'_trace.json',trace);save(label+'_events.json',result['events'])
            summary.append(dict(record=label,tracking_instances=len(segments),
                                acquisition_attempts=sum(e['kind']=='acquisition' for e in result['events']),
                                failed_acquisitions=sum(e['kind']=='acquisition' and not e['success'] for e in result['events']),
                                tracking_start_s=[float(s['history']['epoch_start_samples'][0]/fs) for s in segments],
                                loss_evidence_s=[e['evidence_end_sample']/fs for e in result['events']
                                                 if e['kind']=='tracking_end' and e['status']=='tracking_quality_lost'],
                                tail_epochs=len(tail),tail_quality_pass=sum(r['quality_valid'] for r in tail),
                                tail_residual_outputs=sum(r['residual_m'] is not None for r in tail)))
        del x
        print(label+' done',flush=True)
    save('summary.json',dict(rows=summary,scope=__doc__,
                             limitations=['Single PRN, two channels, synthetic RF, one noise pair',
                                          '5 bad epochs and 100 epoch settling are uncalibrated research parameters',
                                          'No post-loss reference admission; no navigation or authentication claim',
                                          'Offline sample-time schedule; compute latency not measured']))
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
