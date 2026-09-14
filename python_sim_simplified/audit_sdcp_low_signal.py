"""Evaluate one real RF signal-level reduction with explicit observation masks."""
from pathlib import Path
import json
import argparse
import numpy as np
from positioning import carrier_range
from audit_calibrated_sdcp import residuals
from audit_sdcp_positioning import WAVELENGTH
from sdcp_self_check import detect, SelfCheckConfig


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'docs/reports/data/2026-09-12-sdcp-fresh-rf'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--output',type=Path,default=root/'docs/reports/data/2026-09-12-sdcp-low-signal')
    args=parser.parse_args();out=args.output
    if args.generate:
        from audit_sdcp_positioning import channel_satellites, serializable
        from run_gps_spp import satellite_setup, build_scene, run_record
        out.mkdir(parents=True,exist_ok=False)
        sats=channel_satellites(satellite_setup(),1)
        scene,cfg=build_scene(sats,3000);scene*=10**(-6/20)
        d=run_record(6401,sats,scene,cfg)
        (out/'low_6400_1.json').write_text(json.dumps(serializable(d),indent=2,allow_nan=False))
        (out/'manifest.json').write_text(json.dumps({'scope':'TR source amplitude -6 dB before RF; paired noise seed6401; no multipath/interference','duration_ms':3000,'amplitude_scale':10**(-6/20)},indent=2))
    def read(p):return json.loads(p.read_text())
    pair=[read(source/f'los_6400_{ch}.json') for ch in [0,1]]
    low=read(out/'low_6400_1.json')
    t,_,g,_=residuals(pair,read(source/'manifest.json')['satellites'])
    phases=[]
    for record in [pair[0],low]:
        columns=[]
        for h in record['satellite_tracking']:
            if (not h['acquired'] or h['tracking_status'] != 'tracking'):columns.append(np.full(len(t),np.nan))
            else:columns.append(carrier_range(dict(h,success=True),t,record['fs_dfe_hz'],.001,WAVELENGTH))
        phases.append(np.column_stack(columns))
    r=(-(phases[1]-phases[0])-g+WAVELENGTH/2)%WAVELENGTH-WAVELENGTH/2
    valid=np.isfinite(np.asarray(low['pseudoranges_m'],float))
    floors=read(root/'docs/reports/data/2026-09-12-sdcp-self-check-v2/manifest.json')['floors_m']
    rows=[]
    for n in [50,100]:
        o=detect(t,r,WAVELENGTH,floors[str(n)],valid=valid,
                 geometry_uncertainty_m=0.,channel_skew_s=0.,config=SelfCheckConfig(window=n))
        a=o['state'][t>=1.5];known=a>=0
        rows.append({'window_ms':n*10,'coverage':float(known.mean()),
            'false_alarm_all':float((a==1).mean()),
            'false_alarm_judged':float((a[known]==1).mean()) if known.any() else None,
            'fixed_confirmed_false_alarm':float(o['fixed_confirmed'][t>=1.5].mean()),
            'raw_valid_fraction':float(valid.mean()),
            'per_prn_coverage':known.mean(axis=0).tolist()})
        (out/f'state_w{n}.json').write_text(json.dumps({'time_s':t.tolist(),
                       'state':o['state'].tolist(),'reason':o['reason'].tolist()},indent=2))
    (out/'summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
