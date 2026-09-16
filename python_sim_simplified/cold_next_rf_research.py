"""Sequential same-seeded RF proposals, retaining complete tracked evidence."""
import hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np
from cold_next_rf import Case,experiment
from cold_next_research import ROOT,save
from navigation_hundred import candidate
from audit_recovery_two_hundred import score


def run(number,parent,hypothesis,seed=43002,kind='low',duration=.4):
    folder=ROOT/f'iteration{number:03d}';folder.mkdir(exist_ok=True)
    changes={'low':dict(cn0=35.),'normal':dict(cn0=35.),'phase':dict(phase_deg=20.),
             'absent':dict(cn0=35.,genuine=False),'two':dict(antennas=2),'three':dict(antennas=3),
             'amplitude':dict(cn0=35.,phase_deg=20.,amplitude_db=3.)}[kind]
    case=Case(duration=duration,attack_start=0.,**changes)
    save(folder/'proposal.json',dict(number=number,parent=parent,hypothesis=hypothesis,
        layer='sampled synthetic RF, acquisition, tracking, PVT',source='continuous_improvement.py and cold_next.py',
        aids='nominal sky/ephemeris, attitude, coarse code-period range; 100 ms buffered startup; unknown bits'))
    for name in ('cold_next.py','cold_next_rf.py','cold_next_rf_research.py'):shutil.copy2(name,folder/name)
    key=f'{seed}_{kind}'
    oldpath=ROOT/(f'iteration{parent:03d}' if parent>=141 else 'cold_rf_baseline')
    oldpath.mkdir(exist_ok=True)
    if not (oldpath/f'{key}.npz').exists():
        before,metadata=experiment(case,seed,kind!='normal',['raw','calibrated'],extended=True,revision=parent)
        np.savez_compressed(oldpath/f'{key}.npz',**before)
        save(oldpath/f'{key}.json',metadata)
    with np.load(oldpath/f'{key}.npz') as data:before={k:data[k] for k in data.files}
    after,metadata=experiment(case,seed,kind!='normal',['raw','calibrated'],extended=True,revision=number)
    np.testing.assert_array_equal(before['raw_ranges'],after['raw_ranges'])
    np.savez_compressed(folder/f'{key}.npz',**after);save(folder/f'{key}.json',metadata)
    positions=[];scores=[]
    for data in (before,after):
        p=candidate(data)
        if not data['joint_usable']:p[:]=np.nan
        positions.append(p);scores.append(dict(navigation=score(p,data['truth_position']),
            calibrated_spp=score(data['calibrated_position'],data['truth_position']),
            usable=bool(data['joint_usable']),startup_prns=data['startup_prns'].tolist()))
    np.savez_compressed(folder/f'{key}_positions.npz',parent=positions[0],proposal=positions[1])
    row=dict(seed=seed,kind=kind,duration=duration,parent=scores[0],proposal=scores[1],
        parent_input_sha256=hashlib.sha256((oldpath/f'{key}.npz').read_bytes()).hexdigest())
    rows=json.loads((folder/'results.json').read_text(encoding='utf-8'))['rows'] if (folder/'results.json').exists() else []
    rows=[r for r in rows if (r['seed'],r['kind'])!=(seed,kind)]+[row]
    save(folder/'results.json',dict(rows=rows))
    decision='retain for comparison' if scores[1]['navigation']['good20']>scores[0]['navigation']['good20'] else 'no demonstrated navigation gain'
    save(folder/'conclusion.json',dict(decision=decision,evidence=row,
        limitations='Synthetic RF, short duration, no authenticity or long-term guarantee.'))
    print(number,json.dumps(row),flush=True)


if __name__=='__main__':run(int(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
