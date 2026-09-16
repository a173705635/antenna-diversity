"""Iteration 20: actual tracked startup prompts, same seeded IQ for both receivers."""
import argparse,hashlib,json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from continuous_improvement import Case,experiment
from calibration_candidate import calibrate
from navigation_candidate import candidate
from carrier_admission import Rule,evaluate
from audit_recovery_two_hundred import score
from audit_recovery_fifty import save

ROOT=Path('results/research-twenty/iteration20')


def run(seed,kind):
    changes={'normal':{},'cold':dict(attack_start=0.),'phase':dict(attack_start=0.,phase_deg=20.),
             'moving':dict(velocity_east=2.),'absent':dict(attack_start=0.,genuine=False),
             'two':dict(antennas=2),'three':dict(antennas=3),
             'drag':dict(fake_east=0.,fake_rate=20.),'noauth':dict(genuine=False),
             'weak':dict(spoof_db=0.)}
    case=Case(duration=.6,**changes[kind]);attacked=kind!='normal';diagnostic={}
    def measured_startup(block,local,desired):
        m,n=block.shape;s=len(local);chunks=n//2046
        z=np.einsum('mkn,skn->msk',block.reshape(m,chunks,2046),
                    local.conj().reshape(s,chunks,2046))/2046
        g,info=calibrate(z,desired);diagnostic.update(info)
        diagnostic['gain_real']=g.real.tolist();diagnostic['gain_imag']=g.imag.tolist()
        return np.exp(1j*np.angle(g))
    base,metadata=experiment(case,seed,attacked,['raw','calibrated'],extended=True)
    # Research dependency injection only; the production/default function is restored by patch.
    with patch('continuous_improvement.startup_phase',measured_startup):
        joint,joint_metadata=experiment(case,seed,attacked,['raw','calibrated'],extended=True)
    np.testing.assert_array_equal(base['raw_ranges'],joint['raw_ranges'])
    old=evaluate(base,'calibrated',Rule(window=5,mature=1,divergence=np.inf,spatial=1.))['position']
    nav=candidate(base);new=candidate(joint)
    if not diagnostic['usable']:new[:]=np.nan
    folder=ROOT/f'{seed}_{kind}';folder.mkdir()
    np.savez_compressed(folder/'baseline.npz',**base)
    np.savez_compressed(folder/'joint.npz',**joint)
    np.savez_compressed(folder/'positions.npz',old=old,navigation=nav,joint=new)
    result=dict(seed=seed,kind=kind,metadata=metadata,calibration=diagnostic,
                old=score(old,base['truth_position']),navigation=score(nav,base['truth_position']),
                joint=score(new,base['truth_position']))
    save(folder/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind');parser.add_argument('seed',type=int)
    args=parser.parse_args();run(args.seed,args.kind)
