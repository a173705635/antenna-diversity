"""Same seeded RF with measured-sample injection and explicit ideal aids.

The engine retains nominal steering, ephemeris and coarse code-period aiding.
Position truth is used only for scoring; cold attack is not trusted startup.
"""
import sys,shutil,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import continuous_improvement as engine
from array_acquisition import acquire_array
from research_hundred import load,ROOT
from audit_recovery_two_hundred import score
from audit_recovery_fifty import save


def receiver(case,seed,attacked):
    captured={};generate=engine.generate;diagnostic={'usable':False}
    from calibration_candidate import calibrate
    def measured_startup(block,local,desired):
        m,n=block.shape;s=len(local);chunks=n//2046
        z=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),local.conj().reshape(s,chunks,2046))/2046
        gain,info=calibrate(z,desired);diagnostic.update(info)
        return np.exp(1j*np.angle(gain))
    def capture(*args):
        result=generate(*args);captured['samples']=result[0]
        return result
    def acquisition(signal,fs,prn,rate,cfg):
        raw=captured['samples']
        samples=raw[:,:len(signal)] if np.shares_memory(signal,raw) else signal[None,:]
        return acquire_array(samples,fs,prn,rate,cfg)
    with patch.object(engine,'generate',capture),patch.object(engine,'acquire',acquisition),patch.object(engine,'startup_phase',measured_startup):
        traces,metadata=engine.experiment(case,seed,attacked,['raw','calibrated'],extended=True)
    traces['joint_usable']=np.array(diagnostic['usable'])
    metadata['joint_calibration']=diagnostic
    return traces,metadata


def run(number):
    root=ROOT/f'iteration{number:03d}';rows=[]
    for name in ('array_acquisition.py','audit_array_navigation.py'):
        shutil.copy2(name,root/name)
    candidate=load(ROOT/'iteration001/navigation_hundred.py').candidate
    for seed,kind,changes in ((42001,'normal',dict(cn0=35.)),
                              (42002,'cold',dict(cn0=35.,attack_start=0.)),
                              (42003,'phase',dict(phase_deg=20.,attack_start=0.))):
        case=engine.Case(duration=.6,**changes);attacked=kind!='normal'
        with np.load(ROOT/'iteration036'/f'{seed}_current.npz') as data:baseline={k:data[k] for k in data.files}
        current,metadata=receiver(case,seed,attacked)
        before=candidate(baseline);after=candidate(current)
        if not current['joint_usable']:after[:]=np.nan
        np.savez_compressed(root/f'{seed}_baseline.npz',**baseline)
        np.savez_compressed(root/f'{seed}_current.npz',**current)
        np.savez_compressed(root/f'{seed}_positions.npz',baseline=before,current=after)
        rows.append(dict(seed=seed,kind=kind,case=metadata,
            baseline=score(before,baseline['truth_position']),
            current=score(after,current['truth_position']),
            startup_before=baseline['startup_prns'].tolist(),startup_after=current['startup_prns'].tolist()))
        print(json.dumps(rows[-1]),flush=True)
    save(root/'results.json',dict(domain='continuous synthetic RF / tracking / PVT',rows=rows))


if __name__=='__main__':run(int(sys.argv[1]))
