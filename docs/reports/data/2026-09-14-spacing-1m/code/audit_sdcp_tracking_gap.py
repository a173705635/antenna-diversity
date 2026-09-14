"""Single-PRN dual RF outage/reacquisition audit, no spoof authentication."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from acquisition import acquire
from dfe import process_dfe
from rf import process_rf
from tracking import track
from positioning import carrier_range
from audit_sdcp_positioning import channel_satellites, WAVELENGTH
from run_gps_spp import satellite_setup, build_scene, serializable


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    out=p.parse_args().output;out.mkdir(parents=True,exist_ok=False)
    def save(name,obj):
        (out/name).write_text(json.dumps(serializable(obj),indent=2,allow_nan=False),encoding='utf-8')
    sats=satellite_setup()[:1];times=np.arange(20,79)*.01
    histories={};rows=[]
    save('manifest.json',{'scope':__doc__,'duration_ms':800,'gap_s':[.30,.45],
        'scheduled_reacquisition_cut_s':.46,'post_return_phase_rad':.6,
        'seeds':{'A':9800,'B':9801},'prn':1,'rf_hz':33e6,'dfe_decimation':5,
        'reacquisition':'Externally scheduled suffix acquisition for audit; not autonomous loss detector.',
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}})
    for label,ch,gap,phase in [('A',0,False,0),('B_clean',1,False,0),('B_gap',1,True,0),('B_gap_phase',1,True,.6)]:
        scene,cfg=build_scene(channel_satellites(sats,ch),800)
        fs0=cfg.signal.fs_rf_hz
        if gap:
            scene[:,round(.3*fs0):round(.45*fs0)]=0
            assert not np.any(scene[:,round(.31*fs0):round(.44*fs0)])
        if phase:scene[:,round(.45*fs0):]*=np.exp(1j*phase)
        adc,rfinfo=process_rf(scene,cfg.rf,cfg.reference_impedance_ohm,9800+ch);del scene
        x,di=process_dfe(adc,cfg.dfe,fs0,cfg.signal.if_hz);del adc
        fs=di['fs_out_hz'];x=x[0]
        acq=acquire(x,fs,1,cfg.signal.code_rate_hz,cfg.acquisition)
        if not acq['success']:raise RuntimeError(f'initial acquisition failed: {label}')
        h=track(x,fs,1,acq,cfg.tracking,cfg.signal.carrier_hz,cfg.signal.code_rate_hz)
        if not h['success']:raise RuntimeError(f'tracking processing failed: {label}')
        def record(name,history,acquisition):
            np.savez_compressed(out/(name+'.npz'),**{k:v for k,v in history.items() if isinstance(v,np.ndarray)})
            t=history['epoch_start_samples']/fs
            quality=(history['phase_lock_metric']>=.8)&(abs(history['pll_error_rad'])<np.pi/2)
            regions={}
            for key,lo,hi in [('before',.2,.29),('outage_interior',.32,.43),('tail',.65,.79)]:
                m=(t>=lo)&(t<hi)
                regions[key]={'epochs':int(m.sum()),'quality_pass':int(quality[m].sum()),
                    'prompt_median':float(np.median(abs(history['prompt'][m]))) if m.any() else None}
            histories[name]=carrier_range(history,times,fs,.001,WAVELENGTH)
            rows.append({'record':name,'global_success':history['success'],'global_status':history['status'],
                'acquisition':acquisition,'regions':regions,'sample_rate_hz':fs})
        record(label,h,acq)
        if gap:
            cut=round(.46*fs);suffix=x[cut:]
            qa=acquire(suffix,fs,1,cfg.signal.code_rate_hz,cfg.acquisition)
            if not qa['success']:raise RuntimeError(f'reacquisition failed: {label}')
            tc=copy.deepcopy(cfg.tracking);tc.max_epochs=340
            nh=track(suffix,fs,1,qa,tc,cfg.signal.carrier_hz,cfg.signal.code_rate_hz)
            if not nh['success']:raise RuntimeError('new tracking failed')
            nh['epoch_start_samples']+=cut
            record(label+'_reacquired',nh,qa)
        del x
        print(label+' done',flush=True)
    clean=histories['B_clean']-histories['A'];pairs=[]
    for name in histories:
        if name=='A':continue
        d=histories[name]-histories['A'];delta=d-clean
        wrap=(delta+WAVELENGTH/2)%WAVELENGTH-WAVELENGTH/2
        m=(times>=.65)&np.isfinite(wrap)
        pairs.append({'record':name,'tail_finite_epochs':int(m.sum()),
            'paired_raw_change_mean_m':float(np.mean(delta[m])),
            'paired_modulo_change_mean_mm':float(np.mean(wrap[m])*1000),
            'paired_modulo_change_std_mm':float(np.std(wrap[m])*1000)})
    save('summary.json',{'scope':__doc__,'records':rows,'paired_tail':pairs,
        'expected_phase_step_range_mm':-.6*WAVELENGTH/(2*np.pi)*1000,
        'interpretation':'Tail differences compare matched noise/source records; finite values are NOT trusted phase arcs.'})
    np.savez_compressed(out/'paired_ranges.npz',time_s=times,**histories)
    save('sha256.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()})
    print(json.dumps(serializable(pairs),indent=2))


if __name__=='__main__':main()
