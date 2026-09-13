"""Independent short-block acquisition to inspect recording continuity.

Usage from repository root: python docs/reports/data/.../check_continuity.py IQ_PATH
Uses a fixed 10 MHz / +420 kHz ION HackRF format, PRNs 7 and 30.
"""
from pathlib import Path
import sys
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]/'python_sim_simplified'))
from audit_real_gnss import read_iq8
from acquisition import acquire
from config import DFEConfig, AcquisitionConfig
from dfe import process_dfe


def main():
    # Only short blocks are read. No tracking loop, no phase unwrapping.
    rows=[]
    for k in range(1000):
        sample=k*100000
        raw=read_iq8(sys.argv[1],sample,20000)
        x,di=process_dfe(raw[None,:],DFEConfig(decimation=2,output_bits=0),1e7,420e3)
        row=[k*.01]
        for prn,fd in ((7,850.),(30,2900.)):
            a=acquire(x[0],5e6,prn,1.023e6,AcquisitionConfig(doppler_bins_hz=np.array([fd]),noncoherent_ms=1))
            row += [a['metric'],a['code_phase_samples']]
        rows.append(row)
    r=np.array(rows);np.save(ROOT/'hackrf_code_continuity_10s.npy',r)
    jumps=[]
    for i in range(1,len(r)):
        # Circular difference is appropriate ONLY for the primary-code lag,
        # which is inherently measured modulo 1 ms. This is not carrier SDCP.
        d=(r[i,[2,4]]-r[i-1,[2,4]]+2500)%5000-2500
        if max(abs(d))>10:
            jumps.append({'previous_s':r[i-1,0],'current_s':r[i,0],
                'lag_step_samples':d.tolist(),'metrics_before':r[i-1,[1,3]].tolist(),
                'metrics_after':r[i,[1,3]].tolist()})
    (ROOT/'continuity_summary.json').write_text(json.dumps({'columns':['time_s','prn7_metric','prn7_lag_samples','prn30_metric','prn30_lag_samples'],
        'fs_after_dfe':5e6,'lag_step_flag_samples':10,'jumps':jumps,
        'interpretation':'Flagged lag discontinuities, not proof of hardware cause; no data repair.'},indent=2))
    print(jumps)


if __name__=='__main__': main()
