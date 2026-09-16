"""Frozen unseen-seed receiver checks; validation does not add research iterations."""
import hashlib,json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from cold_next_rf import Case,experiment
from cold_next_research import ROOT,save
from navigation_hundred import candidate
from audit_recovery_two_hundred import score

CASES=[(62101,'low',dict(cn0=35.)),(62102,'phase',dict(phase_deg=20.)),
       (62103,'absent',dict(cn0=35.,genuine=False)),(62104,'normal',dict(cn0=35.)),
       (62105,'two',dict(antennas=2)),(62106,'three',dict(antennas=3)),
       (62107,'amplitude',dict(cn0=35.,phase_deg=20.,amplitude_db=3.))]
SOURCES=('cold_next.py','cold_next_rf.py','cold_next_validate.py','array_acquisition.py',
         'calibration_candidate.py','continuous_improvement.py','tracking.py','navigation_hundred.py',
         'carrier_admission.py','continuous_admission.py','signal_recovery.py','code_replica.py','presets.py')


def freeze(revision):
    save(ROOT/'cold_freeze.json',dict(revision=revision,cases=CASES,duration=.4,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCES},
        model='single common emitter, nominal sky/ephemeris/attitude and coarse code-period aid; unknown bits',
        geometry_m=[.0759,.1609],selection='Frozen before these seeds were executed.'))


def validate(index):
    frozen=json.loads((ROOT/'cold_freeze.json').read_text(encoding='utf-8'))
    for name,digest in frozen['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    revision=frozen['revision'];seed,kind,changes=CASES[index]
    case=Case(duration=frozen['duration'],attack_start=0.,**changes)
    folder=ROOT/'cold_validation';folder.mkdir(exist_ok=True);row=dict(seed=seed,kind=kind,case=asdict(case))
    originals=[]
    for label,version in (('baseline',125),('proposal',revision)):
        data,metadata=experiment(case,seed,kind!='normal',['raw','calibrated'],extended=True,revision=version)
        originals.append(data['raw_ranges'])
        np.savez_compressed(folder/f'{seed}_{label}.npz',**data)
        save(folder/f'{seed}_{label}.json',metadata)
        nav=candidate(data)
        if not data['joint_usable']:nav[:]=np.nan
        np.savez_compressed(folder/f'{seed}_{label}_position.npz',position=nav)
        row[label]=dict(navigation=score(nav,data['truth_position']),
            calibrated_spp=score(data['calibrated_position'],data['truth_position']),
            calibration=metadata['joint_calibration'])
    np.testing.assert_array_equal(*originals)
    save(folder/f'{seed}_result.json',row);print(json.dumps(row),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze(int(sys.argv[2]))
    else:validate(int(sys.argv[1]))
