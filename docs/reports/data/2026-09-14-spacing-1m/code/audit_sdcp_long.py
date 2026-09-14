"""Three-second RF records for 200/500/1000 ms causal SDCP windows."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import receiver, satellite_setup, serializable
from audit_calibrated_sdcp import residuals, ROOT
from audit_sdcp_windows import window_mean
from sdcp import calibrated_sdcp
from audit_sdcp_positioning import WAVELENGTH

OUT = ROOT/'docs/reports/data/2026-09-11-sdcp-long'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (OUT/name).write_text(json.dumps(serializable(data),indent=2,allow_nan=False),encoding='utf-8')
    satellites = satellite_setup()
    save('manifest.json', {'duration_ms':3000, 'calibration_seeds':[6200,6202,6204],
         'evaluation_seeds':[6300,6302], 'window_ms':[200,500,1000], 'onset_s':.5,
         'evaluation_start_s':1.5, 'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
             for p in Path(__file__).parent.glob('*.py')},
         'scope':__doc__, 'geometry':'known baseline, per epoch TL SPP, calibrated zero hardware bias',
         'threshold':'independent LOS abs circular mean 99th percentile higher at t>=1.5',
         'satellites':satellites})
    def record(seed,ch,kind):
        name=f'{kind}_{seed}_{ch}.json'
        if (OUT/name).exists():
            return json.loads((OUT/name).read_text())
        print(f'RF {name}',flush=True)
        r=receiver(seed,ch,kind,satellites,3000)
        save(name,r)
        return r
    def pair(seed,kind):
        records=[record(seed,0,'common' if kind=='common' else 'los'),record(seed,1,kind)]
        t,s,g,_=residuals(records,satellites)
        r=calibrated_sdcp(s,g,WAVELENGTH,0)
        save(f'residual_{kind}_{seed}.json',{'time_s':t,'residual_m':r})
        return t,r
    cal=[pair(seed,'los') for seed in (6200,6202,6204)]
    thresholds={n:float(np.quantile(np.concatenate([abs(window_mean(r,n)[t[n-1:]>=1.5]).ravel()
                           for t,r in cal]),.99,method='higher')) for n in (20,50,100)}
    save('thresholds.json',thresholds)
    rows=[]
    for seed in (6300,6302):
        for kind in ('los','strong','weak','static','common'):
            times,raw=pair(seed,kind)
            for n,threshold in thresholds.items():
                t=times[n-1:]
                r=window_mean(raw,n)
                alarm=abs(r)>threshold
                late=t>=1.5
                post=np.flatnonzero((t>=.5)&alarm[:,0])
                row={'seed':seed,'kind':kind,'window_ms':n*10,'threshold_mm':threshold*1000,
                     'late_epochs':int(late.sum()), 'prn1_fraction':float(alarm[late,0].mean()),
                     'other_fraction':float(alarm[late,1:].mean()),'all_fraction':float(alarm[late].mean()),
                     'temporal_std_mm':float(np.std(r[late],axis=0,ddof=1).mean()*1000),
                     'first_available_s':float(t[0]),
                     'first_post_onset_crossing_s':float(t[post[0]]) if len(post) else None}
                rows.append(row)
                save(f'w{n}_{kind}_{seed}.json',{'time_s':t,'residual_m':r,'alarm':alarm})
                print(row,flush=True)
            save('summary.json',rows)
    save('sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.json') if p.name!='sha256.json'})


if __name__=='__main__':
    main()
