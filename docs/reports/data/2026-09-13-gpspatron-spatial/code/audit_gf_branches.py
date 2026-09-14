"""Archived LOS RF branch diagnostic plus labelled observation-domain injections.

Geometry comes from A L1 code positioning and configured baseline. Zero hardware
bias is an external simulation assumption, not estimated from a clean interval.
Nearest branches are hypotheses, not ambiguity-fixed navigation solutions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_sdcp_positioning import BASELINE
from positioning import carrier_range, solve_position
from run_gps_spp import serializable

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'docs/reports/data/2026-09-12-geometry-free-rf'
OUTPUT = ROOT/'docs/reports/data/2026-09-12-gf-branches'


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False)
    read = lambda name: json.loads((SOURCE/name).read_text(encoding='utf-8'))
    for name,digest in read('sha256.json').items():
        if hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('input hash mismatch')
    records = {k:read(k+'.json') for k in ['L1CA_0','L1CA_1','L5Q_0','L5Q_1']}
    t = np.asarray(records['L1CA_0']['time_s'])
    sats = read('manifest.json')['satellites']
    states = np.array([[np.asarray(s['position_m'])+np.asarray(s['velocity_mps'])*v for s in sats] for v in t])
    approx = np.array([solve_position(s,p)['position_m'] for s,p in zip(states,records['L1CA_0']['pseudoranges_m'])])
    wavelengths = np.array([299792458/1575.42e6,299792458/1176.45e6])
    sd = []
    for kind,w in zip(['L1CA','L5Q'],wavelengths):
        phase = []
        for c in (0,1):
            r=records[f'{kind}_{c}']
            if not np.array_equal(t,r['time_s']): raise ValueError('unaligned records')
            phase.append(np.column_stack([carrier_range(dict(h,success=True),t,r['fs_dfe_hz'],.001,w)
                                          for h in r['satellite_tracking']]))
        sd.append(phase[1]-phase[0])
    sd=np.asarray(sd)
    def geometry(baseline):
        return np.linalg.norm(states-approx[:,None,:]-baseline,axis=2)-np.linalg.norm(states-approx[:,None,:],axis=2)
    g=geometry(BASELINE)
    def branch(values,geo):
        n=np.rint((values-geo)/wavelengths[:,None,None]).astype(int)
        residual=values-n*wavelengths[:,None,None]
        return n,residual[0]-residual[1]
    n,gf=branch(sd,g)
    rows=[]
    for j,s in enumerate(sats):
        rows.append({'prn':s['prn'],'l1_branches':np.unique(n[0,:,j]),'l5_branches':np.unique(n[1,:,j]),
            'raw_mean_mm':float(np.mean(sd[0,:,j]-sd[1,:,j])*1000),
            'branch_corrected_mean_mm':float(gf[:,j].mean()*1000),
            'std_mm':float(gf[:,j].std()*1000)})
    # Analytic bound: no nearest-integer error if |geometry error + bias + multipath + noise| < lambda/2.
    stress=[]
    for degrees in [0,3,5,10,30,60,90,180]:
        # Rotate baseline around ECEF radial axis, preserving length; configured geometry stress only.
        axis=approx[0]/np.linalg.norm(approx[0]); angle=np.radians(degrees)
        b=BASELINE*np.cos(angle)+np.cross(axis,BASELINE)*np.sin(angle)+axis*(axis@BASELINE)*(1-np.cos(angle))
        ns,vs=branch(sd,geometry(b))
        stress.append({'baseline_rotation_deg':degrees,'changed_branch_count':int(np.count_nonzero(ns!=n)),
                       'max_gf_change_mm':float(abs(vs-gf).max()*1000)})
    injections=[]
    for name,offset in [('static_multipath',.008),('hardware_bias',.008),('large_offset',.110)]:
        injected=sd.copy(); injected[0,:,0]+=offset
        ni,vi=branch(injected,g)
        injections.append({'label':name,'injected_l1_prn1_m':offset,
                           'output_change_mm':float(np.mean(vi[:,0]-gf[:,0])*1000),
                           'branch_changes':int(np.count_nonzero(ni!=n))})
    result={'scope':__doc__,'epochs':len(t),'rows':rows,'geometry_stress':stress,
            'observation_injections_not_rf':injections,
            'branch_sufficient_bound_m':(wavelengths/2).tolist(),
            'input_hashes':read('sha256.json'),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUTPUT/'summary.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(serializable(result),indent=2))


if __name__=='__main__':main()
