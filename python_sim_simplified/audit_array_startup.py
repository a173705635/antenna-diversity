"""Acquisition-domain experiments, not PVT or authentication success."""
import sys,json,shutil
from pathlib import Path
import numpy as np
from continuous_improvement import Case,generate
from signal_recovery import FS,C
from presets import default_config
from acquisition import acquire
from array_acquisition import acquire_array
from audit_recovery_fifty import save


def run(number):
    root=Path(f'results/research-hundred/iteration{number:03d}');rows=[]
    for name in ('array_acquisition.py','audit_array_startup.py'):shutil.copy2(name,root/name)
    changes=[dict(cn0=35),dict(cn0=30),dict(cn0=45),dict(cn0=35,attack_start=0),dict(cn0=35,phase_deg=20),dict(cn0=35,doppler_scale=1.037),dict(cn0=35,doppler_rate=500.)]
    for i,change in enumerate(changes):
        case=Case(duration=.1,**change);seed=41001+i
        x,a,codes,time,sats,radii=generate(case,seed,i==3)
        cfg=default_config().acquisition;cfg.doppler_bins_hz=np.arange(-4000,4001,250.)
        for prn in (1,3,6,8,9,10):
            old=acquire(x[0],FS,prn,1.023e6,cfg);new=acquire_array(x,FS,prn,1.023e6,cfg)
            auxiliary={}
            if number==26:
                from dataclasses import replace
                from research_hundred import load
                long=load(Path('results/research-hundred/iteration025/array_acquisition.py')).acquire_array(x,FS,prn,1.023e6,replace(cfg,noncoherent_ms=100))
                auxiliary=dict(long_noncoherent_success=long['success'],long_noncoherent_metric=long['metric'])
            rows.append(dict(seed=seed,change=change,prn=prn,present=prn<=8,
                             old_success=old['success'],new_success=new['success'],
                             old_metric=old['metric'],new_metric=new['metric'],
                             old_phase=old['code_phase_samples'],new_phase=new['code_phase_samples'],
                             old_doppler=old['doppler_hz'],new_doppler=new['doppler_hz'],**auxiliary))
            if prn<=8:
                rows[-1]['true_initial_doppler']=case.doppler_scale*(-1500+(prn-1)*500.)
                rows[-1]['true_rate']=case.doppler_rate*(-1+2*(prn-1)/7)
                rows[-1]['estimated_rate']=new['doppler_rate_hz_s']
                rows[-1]['constant_doppler']=new['constant_doppler_hz']
                rows[-1]['dynamic_model']=new['dynamic_model']
        print('finished seed',seed,flush=True)
    summary={str(seed):{key:sum(r[key] for r in rows if r['seed']==seed and r['present']==present)
        for key in ('old_success','new_success')} for seed in range(41001,41001+len(changes)) for present in (True,)}
    false={key:sum(r[key] for r in rows if not r['present']) for key in ('old_success','new_success')}
    save(root/'results.json',dict(domain='synthetic RF acquisition only',rows=rows,summary=summary,absent_prn_detections=false))
    print(json.dumps(dict(summary=summary,false=false)),flush=True)


if __name__=='__main__':run(int(sys.argv[1]))
