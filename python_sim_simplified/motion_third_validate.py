"""New-seed freeze, all snapshot replay, and measurement-equivalence audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from motion_third import estimate
import motion_third_research as research

SEEDS = [97601,97607,97609,97613]


def counterexamples(seed):
    o,tr=research.simulate('common',seed)
    # A narrow but genuinely nonidentical set of directions can pass finite resolution.
    center=tr['direction']/np.linalg.norm(tr['direction'])
    dirs=center[None,:]+np.linspace(-.007,.007,7)[:,None]*np.array([1.,0,0])
    dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
    x=np.einsum('tij,bj->tbi',tr['rotation'],o['baselines'])*o['wave_number']
    rng=np.random.default_rng(seed+30000)
    clustered=dict(o);clustered['phase']=np.angle(np.exp(1j*(np.einsum('tbi,pi->tbp',x,dirs)+rng.uniform(-.025,.025,o['phase'].shape))))
    result=estimate(clustered)
    # Arbitrary shared channel drift exactly trades with common source direction.
    other=np.array([-.7,.2,np.sqrt(1-.7**2-.2**2)])
    compensating_drift=np.einsum('tbi,i->tb',x,center-other)
    ideal_one=np.einsum('tbi,i->tb',x,center)
    ideal_two=np.einsum('tbi,i->tb',x,other)+compensating_drift
    identical=float(np.max(np.abs(ideal_one-ideal_two)))
    return dict(seed=seed,clustered_diverse_status=result['status'],clustered_false_alarm=result['status']=='common',
                shared_drift_gauge_phase_difference=identical,
                indistinguishable_direction_separation_deg=float(np.degrees(np.arccos(center@other))))


def main():
    source=research.ROOT/'motion_third.py'
    rows=[dict(seed=s,**r) for s in SEEDS for r in research.evaluate(source,s)]
    replay=[]
    for number in range(276,301):
        folder=research.OUT/f'iteration{number}'
        expected=json.loads((folder/'results.json').read_text())
        for which in ('before','after'):
            actual=research.evaluate(folder/f'{which}.py',2760)
            assert actual==expected[which],(number,which)
        replay.append(number)
    common=[r for r in rows if r['common_truth']];diverse=[r for r in rows if not r['common_truth']]
    available=[r for r in rows if r['direction_available']]
    hard=[counterexamples(s) for s in SEEDS]
    summary=dict(iterations=list(range(276,301)),seeds=SEEDS,windows=len(rows),duration_seconds=12,samples_per_window=121,
        false_alarms=sum(r['false_alarm'] for r in diverse),diverse_windows=len(diverse),
        missed_common=sum(r['missed_common'] for r in common),common_windows=len(common),
        common_abstentions=sum(r['status']=='indeterminate' for r in common),
        common_not_detected=sum(r['status']!='common' for r in common),
        direction_available_windows=len(available),
        available_direction_median_error_deg=float(np.median([r['direction_error_deg'] for r in available])),
        available_direction_max_error_deg=max(r['direction_error_deg'] for r in available),
        support_diagnostic_violations=sum(r['direction_error_deg']>r['direction_radius_deg'] for r in available),
        absolute_position_available=0,replayed_iterations=replay,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        clustered_diverse_false_alarms=sum(x['clustered_false_alarm'] for x in hard),clustered_diverse_windows=len(hard),
        common_drift_gauge_max_difference=max(x['shared_drift_gauge_phase_difference'] for x in hard),
        support_is_certificate=False,real_rf_navigation=False)
    for case in research.CASES:
        selected=[r for r in rows if r['case']==case]
        np.savez_compressed(research.OUT/f'motion_frozen_{case}.npz',
            **{f'seed{s}_{k}':v for s in SEEDS for k,v in research.simulate(case,s)[0].items()})
    (research.OUT/'motion_validation.json').write_text(json.dumps(dict(summary=summary,rows=rows,counterexamples=hard),indent=2))
    (research.OUT/'motion_summary.json').write_text(json.dumps(summary,indent=2))
    (research.ROOT/'motion_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
