"""Startup PRN1 echo in B L1 RF; reuse three paired LOS streams. No detector rates."""
import hashlib
import json
from pathlib import Path
import numpy as np
from config import MultipathPath, Direction
from audit_sdcp_positioning import BASELINE, channel_satellites
from positioning import carrier_range, solve_position
from run_gps_spp import build_scene, run_record, serializable

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/reports/data/2026-09-12-geometry-free-rf'
OUTPUT=ROOT/'docs/reports/data/2026-09-12-gf-startup-rf'


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    read=lambda n:json.loads((SOURCE/n).read_text(encoding='utf-8'))
    def save(n,v): (OUTPUT/n).write_text(json.dumps(serializable(v),indent=2,allow_nan=False),encoding='utf-8')
    hashes=read('sha256.json')
    for name,digest in hashes.items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest: raise ValueError('hash mismatch')
    manifest=read('manifest.json'); sats=manifest['satellites']
    for s in sats:
        s['position_m']=np.asarray(s['position_m']); s['velocity_mps']=np.asarray(s['velocity_mps'])
    shifted=channel_satellites(sats,1)
    scene,cfg=build_scene(shifted,manifest['duration_ms'],multipath_by_prn={1:[
        MultipathPath(Direction(110.,15.),relative_power_db=-6.,excess_delay_chips=.25,reflection_phase_rad=np.pi/2)]})
    save('manifest.json',{'scope':__doc__,'source_hashes':hashes,'seed':7101,
        'echo':'B L1 PRN1 only, -6 dB, .25 L1 chips, pi/2 phase, present from start',
        'hardware_bias':'zero RF bias; later offset sweep is observation-domain only',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}})
    dirty=run_record(7101,shifted,scene,cfg); del scene
    save('L1CA_1_static.json',dirty)
    records={k:read(k+'.json') for k in ['L1CA_0','L1CA_1','L5Q_0','L5Q_1']}
    t=np.asarray(records['L1CA_0']['time_s'])
    states=np.array([[s['position_m']+s['velocity_mps']*v for s in sats] for v in t])
    approx=np.array([solve_position(s,p)['position_m'] for s,p in zip(states,records['L1CA_0']['pseudoranges_m'])])
    g=np.linalg.norm(states-approx[:,None,:]-BASELINE,axis=2)-np.linalg.norm(states-approx[:,None,:],axis=2)
    w=np.array([299792458/1575.42e6,299792458/1176.45e6])
    def extract(r,wave):
        if not np.array_equal(t,r['time_s']): raise ValueError('time mismatch')
        return np.column_stack([carrier_range(dict(h,success=h.get('tracking_status')=='tracking'),t,
            r['fs_dfe_hz'],.001,wave) for h in r['satellite_tracking']])
    a1,b1,a5,b5=[extract(records[k],wave) for k,wave in zip(records,[w[0],w[0],w[1],w[1]])]
    bd=extract(dirty,w[0]); valid=np.isfinite(a1+b1+a5+b5+bd)
    for r in list(records.values())+[dirty]: valid &= np.isfinite(r['pseudoranges_m'])
    sd=np.array([b1-a1,b5-a5]); ds=np.array([bd-a1,b5-a5])
    def corrected(x):
        # Only evaluate finite observations, avoid invalid float-to-integer conversion.
        n=np.rint((x-g)/w[:,None,None])
        y=x-n*w[:,None,None]
        return y[0]-y[1],n
    clean,nc=corrected(sd); echo,nd=corrected(ds)
    rows=[]
    for j,s in enumerate(sats):
        m=valid[:,j]; row={'prn':s['prn'],'valid_epochs':int(m.sum()),'total_epochs':len(t)}
        if m.any(): row.update(los_mean_mm=float(clean[m,j].mean()*1000),static_mean_mm=float(echo[m,j].mean()*1000),
            paired_change_mean_mm=float((echo-clean)[m,j].mean()*1000),static_std_mm=float(echo[m,j].std()*1000),
            branch_changes=int(np.count_nonzero(nc[:,m,j]!=nd[:,m,j])))
        rows.append(row)
    sweep=[]
    for bias in [0,.003,.008,.020,.050,.110]:
        x=ds.copy(); x[0]+=bias # common hardware offset across all PRNs at B L1
        z,n=corrected(x)
        sweep.append({'B_L1_common_bias_m':bias,'prn1_mean_mm':float(z[valid[:,0],0].mean()*1000),
            'other_mean_mm':float(z[:,1:][valid[:,1:]].mean()*1000),
            'branch_changes':int(np.count_nonzero((n!=nd)&valid[None,:,:]))})
    save('summary.json',{'scope':__doc__,'rows':rows,'observation_bias_sweep_not_rf':sweep,
        'policy':'Unknown bias bound => unknown absolute multipath classification. No threshold tuned on this record.',
        'time_s':t,'los_gf_m':clean,'static_gf_m':echo,'valid':valid})
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUT.glob('*.json')})
    print(json.dumps(serializable({'rows':rows,'bias_sweep':sweep}),indent=2))


if __name__=='__main__':main()
