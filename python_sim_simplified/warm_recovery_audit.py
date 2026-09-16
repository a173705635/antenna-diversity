"""Replay exact source snapshots and report short correlator continuity."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import warm_recovery_campaign as campaign

OUT=campaign.OUT

def module(path):
    spec=importlib.util.spec_from_file_location('frozen_recovery',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def replay():
    ledger=json.loads((OUT/'ledger.json').read_text());checks=[]
    for entry in ledger:
        folder=OUT/f"round-{entry['round']:03d}"
        for name,target in [('parent','before'),('candidate','after')]:
            mod=module(folder/f'{name}.py');campaign.core=mod
            actual=campaign.evaluate(mod.SELECTED_METHODS,[111001,111013]);expected=json.loads((folder/f'{target}.json').read_text())
            assert actual==expected,(entry['round'],name)
            checks.append(dict(round=entry['round'],version=name,exact=True))
    freeze=json.loads((OUT/'freeze.json').read_text());path=OUT/'warm_recovery_frozen.py';assert campaign.digest(path)==freeze['sha256']
    mod=module(path);campaign.core=mod
    assert campaign.evaluate(mod.SELECTED_METHODS,freeze['holdout_seeds'])==json.loads((OUT/'holdout.json').read_text())
    campaign.save(OUT/'replay.json',dict(snapshot_checks=checks,holdout_exact=True))
    print('50 parent/candidate snapshots and frozen holdout match exactly.')

def diagnostics():
    mod=module(OUT/'warm_recovery_frozen.py');campaign.core=mod;methods=mod.SELECTED_METHODS
    ablations={}
    for removed in ('init_average','spatial_whiten','projected_fit','continuity_gate'):
        ablations[removed]=campaign.evaluate([v for v in methods if v!=removed],[111001,111013])['summary']
    campaign.save(OUT/'causal_ablations.json',ablations)
    geometry=[]
    for pair in ((0,1),(0,2),(0,3),(1,2)):
        for seed in (111001,111013):
            for scenario in mod.SCENARIOS:
                obs,truth=mod.generate(seed,scenario)
                sub={**obs,**{k:obs[k][:,:,list(pair),:].copy() for k in ('init','z','noise')}}
                r=mod.estimate(sub,methods);error=abs(r['delay']-truth['delay'][-1])*mod.CHIP_M
                geometry.append(dict(pair=pair,seed=seed,scenario=scenario,accepted=r['accepted'].tolist(),error_m=error.tolist(),exists=truth['exists'],init_completed_s=r['init_completed_s']))
    campaign.save(OUT/'geometry_pairs.json',geometry)
    records=[]
    for seed in (112001,112013,112037):
        for scenario in mod.SCENARIOS:
            obs,truth=mod.generate(seed,scenario)
            for m in (2,3,4):
                sub=mod.subset(obs,m);decisions=[];false=[];longest=0;run=np.zeros(6,int)
                for t in range(1,len(sub['z'])+1):
                    result=mod.estimate({**sub,'z':sub['z'][:t]},methods)
                    accepted=result['accepted']&result['init_ok']
                    error=abs(result['delay']-truth['delay'][min(t-1,len(truth['delay'])-1)])*mod.CHIP_M
                    decisions.append(accepted.tolist());false.append((accepted&((error>=30)|(~np.full(6,truth['exists'])))).tolist())
                    run=np.where(accepted,0,run+1);longest=max(longest,int(run.max()))
                records.append(dict(seed=seed,scenario=scenario,antennas=m,init_completed_s=result['init_completed_s'],accepted_by_prefix=decisions,false_by_prefix=false,max_per_satellite_outage_s=longest*obs['dt']))
    campaign.save(OUT/'continuity.json',dict(domain='Eight cumulative correlator windows, 160 ms total. Not navigation continuity.',records=records))
    print('Ablation and cumulative-window continuity diagnostics saved; no model changes.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');args=parser.parse_args()
    if args.replay:replay()
    else:diagnostics()
