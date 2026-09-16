"""Sequential, pre-registered source discriminator experiments and frozen audit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import warm_source as ws

ROOT = Path(__file__).parent
OUT = ROOT/'results/research-warm-hundred/source'
SOURCE = ROOT/'warm_source.py'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def evaluate(score, seeds, thresholds=None):
    rows = []
    for seed in seeds:
        for scenario in ws.SCENARIOS:
            for attack in (False, True):
                case = ws.make_case(seed, scenario, attack)
                for m in (2,3,4):
                    warm, current = case['warm'][:, :m], case['current'][:, :m]
                    state = ws.initialize(warm)
                    rows.append(dict(seed=seed, scenario=scenario, attack=attack, antennas=m,
                                     score=score(warm,current), initialized=state['complete'],
                                     initialization_time=state['time'], attack_time=case['attack_time'],
                                     initialization_gap=.4 if state['complete'] else None,
                                     initialization_mismatch=state['mismatch']))
    if thresholds is None:
        thresholds = {m:float(np.quantile([r['score'] for r in rows if r['antennas']==m
                     and not r['attack'] and r['scenario']!='reflection' and r['initialized']],.95))
                      for m in (2,3,4)}
    for r in rows:
        r['detected'] = bool(r['score'] > thresholds[r['antennas']])
    summary = {}
    for m in (2,3,4):
        selected = [r for r in rows if r['antennas']==m]
        valid = [r for r in selected if r['initialized']]
        attacks = [r for r in valid if r['attack']]
        nulls = [r for r in valid if not r['attack'] and r['scenario']!='reflection']
        reflection = [r for r in valid if not r['attack'] and r['scenario']=='reflection']
        summary[m] = dict(detection=float(np.mean([r['detected'] for r in attacks])),
                          false_alarm=float(np.mean([r['detected'] for r in nulls])),
                          reflection_false_alarm=float(np.mean([r['detected'] for r in reflection])),
                          initialization_failure=sum(not r['initialized'] for r in selected),
                          total=len(selected), attack_count=len(attacks),null_count=len(nulls))
    return dict(summary=summary, thresholds=thresholds, rows=rows)


def run_round(number, hypothesis, body, next_step):
    OUT.mkdir(parents=True, exist_ok=True)
    folder = OUT/f'{number:03d}'
    folder.mkdir()
    ledger_path = OUT/'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    parent = SOURCE.read_text(encoding='utf-8')
    write(folder/'proposal.json', dict(round=number,hypothesis=hypothesis,
          parent_round=ledger[-1]['selected_round'] if ledger else 25,
          registered_at=time.time(),domain='spatial observation; not RF/PVT',
          selection='mean detection(3,4) - 2*mean false alarm(3,4); strict improvement',
          calibration_seeds=[111001,111020],evaluation_seeds=[111021,111045]))
    (folder/'before.py').write_text(parent,encoding='utf-8')
    parent_score = ws.score
    # The hypothesis exists before executable candidate source is written.
    candidate = parent[:parent.index('def score(warm, current):')]+'def score(warm, current):\n'+body+'\n'
    (folder/'source.py').write_text(candidate,encoding='utf-8')
    namespace = {}
    exec(compile(candidate,str(folder/'source.py'),'exec'),namespace)
    before_cal = evaluate(parent_score,range(111001,111021))
    after_cal = evaluate(namespace['score'],range(111001,111021))
    before = evaluate(parent_score,range(111021,111046),before_cal['thresholds'])
    after = evaluate(namespace['score'],range(111021,111046),after_cal['thresholds'])
    write(folder/'before.json',before)
    write(folder/'after.json',after)
    utility = lambda result: float(np.mean([result['summary'][m]['detection']-
                         2*result['summary'][m]['false_alarm'] for m in (3,4)]))
    accepted = utility(after)>utility(before)+1e-12
    if accepted:
        SOURCE.write_text(candidate,encoding='utf-8')
    record = dict(round=number,hypothesis=hypothesis,accepted=accepted,
                  before_utility=utility(before),after_utility=utility(after),
                  source_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                  selected_round=number if accepted else (ledger[-1]['selected_round'] if ledger else 25),
                  conclusion='retain: improved primary detection/false-alarm utility' if accepted
                    else 'reject: no primary utility improvement; counterexample rates retained',
                  next_step=next_step,completed_at=time.time())
    write(folder/'conclusion.json',record)
    ledger.append(record)
    write(ledger_path,ledger)
    print(json.dumps(dict(record=record,before=before['summary'],after=after['summary'])))


def freeze():
    source = SOURCE.read_bytes()
    frozen = OUT/'frozen.py'
    frozen.write_bytes(source)
    calibration = evaluate(ws.score,range(111001,111021))
    write(OUT/'freeze.json',dict(sha256=hashlib.sha256(source).hexdigest(),frozen_at=time.time(),
                               thresholds=calibration['thresholds'],holdout_seeds=[112001,112013,112037]))
    result = evaluate(ws.score,[112001,112013,112037],calibration['thresholds'])
    write(OUT/'holdout.json',result)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--proposal')
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    else:
        run_round(**json.loads(Path(args.proposal).read_text(encoding='utf-8')))
