"""Freeze, genuinely new-seed validation, and selected snapshot replay."""
import hashlib,json
from pathlib import Path
import numpy as np
from delay_next_campaign import ROOT,CASES,load,evaluate,write

HERE=Path(__file__).parent
SEEDS=(52101,52107,52119)
KINDS=tuple(dict.fromkeys(kind for _,kind in CASES))


def freeze():
    names=('delay_next.py','delay_next_campaign.py','delay_next_verify.py','tests/test_delay_next.py','lag_calibration.py','lag_source_support.py','audit_lag_calibration.py','audit_lag_support.py','signal_recovery.py','presets.py')
    files={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
    write(ROOT/'delay_freeze.json',dict(seeds=SEEDS,kinds=KINDS,files=files,
        claim='Frozen before evaluating new seeds. All prior seeds are development. No model tuning after this file.',
        assumptions=['known sky and Mate70 75.9x160.9mm corners','diagonal complex gains and per-channel delay','single common interferer; up to two correlation templates per PRN','independent 512 noise calibration snapshots, 256 training/256 validation','synthetic real stationary lag covariance family, tested mismatch case','local nonlinear optimizer; confidence tests are approximate and selected-model conditional']))


def validate():
    frozen=json.loads((ROOT/'delay_freeze.json').read_text())
    for name,digest in frozen['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    candidate=load(HERE/'delay_next.py').estimate
    old=load(HERE/'lag_source_support.py').estimate
    rows=[]
    for seed in SEEDS:
        for kind in KINDS:
            before=evaluate(old,seed,kind);after=evaluate(candidate,seed,kind)
            rows.append(dict(before=before,after=after))
            write(ROOT/'delay_validation.json',dict(frozen_seeds=SEEDS,rows=rows,completed=len(rows)==len(SEEDS)*len(KINDS)))
            print('VALIDATE',seed,kind,after['usable'],after['phase_rmse_deg'],after['admitted_code_rmse_m'],flush=True)


def replay():
    rows=[]
    for number in (101,106,115,120,125):
        folder=ROOT/f'iteration{number:03d}'
        original=json.loads((folder/'results.json').read_text(encoding='utf-8'))['rows']
        method=load(folder/'delay_next.py').estimate
        for kind in ('normal','overlap','skew','noise'):
            expected=next(row['after'] for row in original if row['after']['seed']==32051 and row['after']['kind']==kind)
            actual=evaluate(method,32051,kind)
            assert actual['input_sha256']==expected['input_sha256']
            assert actual['supported_prns']==expected['supported_prns']
            assert actual['usable']==expected['usable']
            np.testing.assert_allclose(actual['selected_delay_chips'],expected['selected_delay_chips'],rtol=0,atol=1e-7)
            np.testing.assert_allclose(actual['channel_delay_chips'],expected['channel_delay_chips'],rtol=0,atol=1e-7)
            rows.append(dict(iteration=number,kind=kind,passed=True));print('REPLAY',number,kind,flush=True)
    write(ROOT/'delay_replay.json',dict(rows=rows,passed=True))


def audit():
    rows=[]
    for number in range(101,126):
        folder=ROOT/f'iteration{number:03d}'
        proposal=json.loads((folder/'proposal.json').read_text(encoding='utf-8'))
        conclusion=json.loads((folder/'conclusion.json').read_text(encoding='utf-8'))
        result=json.loads((folder/'results.json').read_text(encoding='utf-8'))
        assert all(row['before']['input_sha256']==row['after']['input_sha256'] for row in result['rows'])
        rows.append(dict(iteration=number,hypothesis=proposal['hypothesis'],decision=conclusion['decision'],paired_cases=len(result['rows']),model_sha256=hashlib.sha256((folder/'delay_next.py').read_bytes()).hexdigest()))
    assert len({row['model_sha256'] for row in rows})==25
    write(ROOT/'delay_iteration_audit.json',dict(completed=25,unique_model_snapshots=25,rows=rows))


if __name__=='__main__':
    import sys
    {'freeze':freeze,'validate':validate,'replay':replay,'audit':audit}[sys.argv[1]]()
