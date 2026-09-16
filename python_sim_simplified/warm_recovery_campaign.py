"""One evidence-driven research round per invocation; no seed/parameter rounds."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import warm_recovery as core

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/research-warm-hundred/recovery'
OPTIONS={
'init_average':'Average twelve clean initialization frames to reduce measured steering variance.',
'peak_baseline':'Replace absolute peak lag with amplitude-profile matched-pulse likelihood.',
'subgrid':'Use local quadratic likelihood curvature for continuous delay interpolation.',
'rank1_init':'Estimate clean spatial response by rank-one SVD over all correlator lags.',
'spatial_whiten':'Whiten channels using independent observed noise covariance before spatial nulling.',
'residual_direction':'Estimate common source after removing each measured authentic sky subspace.',
'projected_fit':'Fit a common direction through satellite-specific orthogonal projections by variable projection.',
'phase_align':'Align frame carrier phase from measured complex recovered correlation before integration.',
'lag_whiten':'Whiten lag correlation using independent noise-only measurements in profile fitting.',
'init_gate':'Withhold satellites whose twelve initialization code estimates scatter by at least 8 m.',
'presence_gate':'Require conditional matched signal energy above a multiple-comparison detection threshold.',
'precision_gate':'Require three times local code uncertainty below 20 m.',
'shape_gate':'Reject single-pulse model residual inconsistent with measured thermal noise.',
'temporal_gate':'Require independent frame code agreement to reject unstable leakage peaks.',
'motion_fit':'Regress recovered frame delays against time and report the last-epoch delay.',
'calibration_uncertainty':'Propagate measured initialization scatter into local delay uncertainty.',
'block_direction':'Re-estimate the common spatial source separately in each attack frame.',
'robust_average':'Replace coherent sample mean by componentwise temporal median.',
'ambiguity_gate':'Compare best profile likelihood with distant competing delay solutions.',
'continuity_gate':'Reject delay innovations above 30 m relative to measured trusted initialization.',
'crossfit_direction':'Learn nuisance direction on first half only; evaluate remaining observations independently.',
'holdout_gate':'Require second-half matched energy and code agreement.',
'projection_gate':'Reject spatial configurations retaining under eight percent authentic steering energy.',
'joint_fit':'Jointly optimize authentic and counterfeit delays with complex amplitudes eliminated by least squares.',
'persistence_gate':'Require six of eight individual attack frames to confirm delay and conditional energy.'}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def evaluate(methods,seeds):
    rows=[]
    for seed in seeds:
        for scenario in core.SCENARIOS:
            obs,truth=core.generate(seed,scenario)
            input_hash=hashlib.sha256(obs['z'].tobytes()+obs['init'].tobytes()).hexdigest()
            for m in (2,3,4):
                r=core.estimate(core.subset(obs,m),methods)
                err=abs(r['delay']-truth['delay'][-1])*core.CHIP_M
                accepted=r['accepted'];false=accepted&((err>=30)|(~np.full(6,truth['exists'])))
                rows.append(dict(seed=seed,scenario=scenario,antennas=m,input_hash=input_hash,error_m=err.tolist(),accepted=accepted.tolist(),false_release=false.tolist(),exists=truth['exists'],init_ok=r['init_ok'].tolist(),init_completed_s=r['init_completed_s'],attack_start_s=r['attack_start_s'],gap_s=r['gap_s']))
    summaries={}
    for m in (2,3,4):
        rr=[r for r in rows if r['antennas']==m];accepted=np.concatenate([r['accepted'] for r in rr]);errors=np.concatenate([r['error_m'] for r in rr]);false=np.concatenate([r['false_release'] for r in rr]);existing=np.concatenate([np.full(6,r['exists']) for r in rr])
        valid=accepted&existing
        summaries[str(m)]=dict(total=len(errors),accepted=int(accepted.sum()),availability=float(accepted.mean()),rejection_rate=float(1-accepted.mean()),false_release=int(false.sum()),accepted_p95_code_error_m=float(np.quantile(errors[valid],.95)) if valid.any() else None,existing_unconditional_p95_m=float(np.quantile(errors[existing],.95)),init_failed_records=sum(r['init_completed_s'] is None for r in rr),accepted_error_ge50=int(np.sum(valid&(errors>=50))))
    return dict(summary=summaries,rows=rows)

def objective(result):
    s=[result['summary'][m] for m in ('3','4')]
    # Preregistered engineering utility: each unsafe output costs ten valid outputs.
    return sum(x['accepted']-11*x['false_release'] for x in s)

def round_run(number,method):
    OUT.mkdir(parents=True,exist_ok=True)
    state_path=OUT/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(methods=[],ledger=[])
    assert number==51+len(state['ledger'])
    assert method not in [r['method'] for r in state['ledger']]
    folder=OUT/f'round-{number:03d}';folder.mkdir()
    previous=state['ledger'][-1]['conclusion'] if state['ledger'] else 'Initial uncalibrated single-frame steering/peak baseline; no validation.'
    proposal=dict(round=number,method=method,hypothesis=OPTIONS[method],prior_evidence=previous,parent_methods=state['methods'],development_seeds=[111001,111013],criterion='maximize accepted - 11*unsafe over 3/4 antennas, same inputs; ties reject',created_ns=time.time_ns())
    save(folder/'proposal.json',proposal)
    parent=state['methods'];child=parent+[method]
    source=Path(core.__file__).read_text(encoding='utf-8')
    (folder/'parent.py').write_text(source+'\nSELECTED_METHODS = '+repr(parent)+'\n',encoding='utf-8')
    (folder/'candidate.py').write_text(source+'\nSELECTED_METHODS = '+repr(child)+'\n',encoding='utf-8')
    before=evaluate(parent,[111001,111013]);save(folder/'before.json',before)
    after=evaluate(child,[111001,111013]);save(folder/'after.json',after)
    assert [(r['seed'],r['scenario'],r['antennas'],r['input_hash']) for r in before['rows']]==[(r['seed'],r['scenario'],r['antennas'],r['input_hash']) for r in after['rows']]
    keep=objective(after)>objective(before)
    conclusion=dict(retained=keep,before_utility=objective(before),after_utility=objective(after),before=before['summary'],after=after['summary'],reason='Improved preregistered safe-output utility.' if keep else 'Rejected: no strict improvement in safe-output utility.',next_step='Inspect residual false release, then test a structurally different estimator or evidence criterion; no parameter sweep.',candidate_sha256=digest(folder/'candidate.py'),parent_sha256=digest(folder/'parent.py'))
    save(folder/'conclusion.json',conclusion)
    state['ledger'].append(dict(round=number,method=method,conclusion=conclusion));state['methods']=child if keep else parent
    save(state_path,state);save(OUT/'ledger.json',state['ledger'])
    print(json.dumps(dict(round=number,method=method,retained=keep,before=objective(before),after=objective(after),metrics=after['summary'])))

def finalize():
    state=json.loads((OUT/'state.json').read_text());assert len(state['ledger'])==25
    frozen=OUT/'warm_recovery_frozen.py';frozen.write_text(Path(core.__file__).read_text(encoding='utf-8')+'\nSELECTED_METHODS = '+repr(state['methods'])+'\n',encoding='utf-8')
    manifest=dict(sha256=digest(frozen),frozen_ns=time.time_ns(),methods=state['methods'],holdout_seeds=[112001,112013,112037]);save(OUT/'freeze.json',manifest)
    result=evaluate(state['methods'],manifest['holdout_seeds']);save(OUT/'holdout.json',result)
    primary={}
    for m in (2,3,4):
        rows=[r for r in result['rows'] if r['antennas']==m and r['init_completed_s'] is not None]
        accepted=np.concatenate([r['accepted'] for r in rows]);error=np.concatenate([r['error_m'] for r in rows]);exists=np.concatenate([np.full(6,r['exists']) for r in rows]);false=np.concatenate([r['false_release'] for r in rows]);valid=accepted&exists
        primary[str(m)]=dict(total=len(error),accepted=int(accepted.sum()),availability=float(accepted.mean()),false_release=int(false.sum()),accepted_p95_code_error_m=float(np.quantile(error[valid],.95)),accepted_error_ge50=int(np.sum(valid&(error>=50))),absent_false_release=sum(sum(r['accepted']) for r in rows if not r['exists']),initialization_failed_records=sum(r['antennas']==m and r['init_completed_s'] is None for r in result['rows']))
    save(OUT/'summary.json',dict(completed_rounds=25,retained=sum(r['conclusion']['retained'] for r in state['ledger']),freeze=manifest,holdout_all_including_initialization_failures=result['summary'],primary_initialized_holdout=primary,scope='correlator-domain, not RF acquisition/PVT; independent observed noise calibration supplied',pvt_metrics=None))
    print(json.dumps(primary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int);p.add_argument('--method',choices=OPTIONS);p.add_argument('--finalize',action='store_true');args=p.parse_args()
    if args.finalize:finalize()
    else:round_run(args.round,args.method)
