"""Replay a changed assumed baseline; source RF geometry remains unchanged."""
from pathlib import Path
import json
import numpy as np
from audit_sdcp_positioning import BASELINE, WAVELENGTH
from audit_calibrated_sdcp import residuals
from run_gps_spp import receiver_reference
from positioning import solve_position
from sdcp_self_check import detect, SelfCheckConfig


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'docs/reports/data/2026-09-12-sdcp-fresh-rf'
    out = root/'docs/reports/data/2026-09-12-sdcp-geometry-stress'
    out.mkdir(exist_ok=False)
    def read(name): return json.loads((source/name).read_text())
    floors = json.loads((root/'docs/reports/data/2026-09-12-sdcp-self-check-v2/manifest.json').read_text())['floors_m']
    satellites = read('manifest.json')['satellites']
    rows=[]
    for seed in [6400,6402]:
        pair=[read(f'los_{seed}_{ch}.json') for ch in [0,1]]
        t,s,g,_=residuals(pair,satellites)
        states=np.array([[np.array(x['position_m'])+np.array(x['velocity_mps'])*tt
                          for x in satellites] for tt in t])
        approximate=np.array([solve_position(x,p)['position_m']
                              for x,p in zip(states,pair[0]['pseudoranges_m'])])
        for degrees in [0,1,3,5,10]:
            a=np.radians(degrees)
            assumed=receiver_reference()[1].T @ (.0759*np.array([np.cos(a),np.sin(a),0]))
            gg=-(np.linalg.norm(states-approximate[:,None,:]-assumed,axis=2)
                 -np.linalg.norm(states-approximate[:,None,:],axis=2))
            r=(s-gg+WAVELENGTH/2)%WAVELENGTH-WAVELENGTH/2
            bound=float(np.linalg.norm(assumed-BASELINE))
            for reported in [False,True]:
                for n in [50,100]:
                    result=detect(t,r,WAVELENGTH,floors[str(n)],valid=True,
                         geometry_uncertainty_m=bound if reported else 0.,channel_skew_s=0.,
                         config=SelfCheckConfig(window=n))
                    state=result['state'][t>=1.5];known=state>=0
                    rows.append({'seed':seed,'yaw_error_deg':degrees,'window_ms':n*10,
                        'bound_reported':reported,'baseline_error_bound_mm':bound*1000,
                        'max_geometry_error_mm':float(abs(gg-g).max()*1000),
                        'coverage':float(known.mean()),'false_alarm_all':float((state==1).mean()),
                        'false_alarm_judged':float((state[known]==1).mean()) if known.any() else None})
    (out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'README.md').write_text('# Assumed-baseline stress\n\nYaw rotation in local ENU, fixed 75.9 mm baseline length. RF records unchanged; per-epoch TL SPP is used for geometry. This is compensation-error replay, not antenna-motion RF. No parameters retuned.\n',encoding='utf-8')
    print('Geometry cases:',len(rows))


if __name__=='__main__':main()
