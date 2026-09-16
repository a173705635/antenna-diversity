"""Uncounted requirement correction; rerun reserved seeds without retuning."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import warm_recovery_release as release

ROOT=Path(__file__).parent
OUT=ROOT/'results/research-warm-hundred/recovery/release-correction'

def save(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(seeds):
    rows=[]
    for seed in seeds:
        for scenario in release.RESEARCH.SCENARIOS:
            obs,truth=release.RESEARCH.generate(seed,scenario)
            for m in (2,3,4):
                result=release.estimate(release.RESEARCH.subset(obs,m))
                error=abs(result['delay']-truth['delay'][-1])*release.RESEARCH.CHIP_M
                rows.append(dict(seed=seed,scenario=scenario,antennas=m,init_completed_s=result['init_completed_s'],init_ok=result['init_ok'].tolist(),research_accepted=result['research_accepted'].tolist(),publish_accepted=result['publish_accepted'].tolist(),error_m=error.tolist(),exists=truth['exists']))
    metrics={}
    for m in (2,3,4):
        rr=[r for r in rows if r['antennas']==m];error=np.concatenate([r['error_m'] for r in rr]);exists=np.concatenate([np.full(6,r['exists']) for r in rr]);per={}
        for key in ('research_accepted','publish_accepted'):
            accepted=np.concatenate([r[key] for r in rr]);valid=accepted&exists
            per[key]=dict(total=len(error),accepted=int(accepted.sum()),availability=float(accepted.mean()),false_release=int(np.sum(accepted&((error>=30)|~exists))),accepted_p95_code_error_m=float(np.quantile(error[valid],.95)),accepted_error_ge50=int(np.sum(valid&(error>=50))),accepted_in_failed_init=sum(sum(r[key]) for r in rr if r['init_completed_s'] is None))
        per['initialization_failed_records']=sum(r['init_completed_s'] is None for r in rr);metrics[str(m)]=per
    return dict(rows=rows,summary=metrics)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    if args.verify:
        manifest=json.loads((OUT/'freeze.json').read_text())
        assert all(sha(ROOT/name)==digest for name,digest in manifest['sha256'].items())
        assert run(manifest['seeds'])==json.loads((OUT/'paired_holdout.json').read_text())
        print('Publication correction replay matches; frozen hashes unchanged.')
    else:
        OUT.mkdir(exist_ok=True)
        save(OUT/'proposal.json',dict(reason='Scope correction REC-INIT: publication requires complete observed record initialization.',selection='No optimization or tuning; unconditional user requirement.',counted_rounds_added=0,created_ns=time.time_ns(),holdout_note='Original reserved seeds already seen in research; independent policy rule and frozen re-evaluation, not new unseen data.'))
        files=['warm_recovery_release.py','warm_recovery_release_audit.py','tests/test_warm_recovery.py','results/research-warm-hundred/recovery/warm_recovery_frozen.py']
        manifest=dict(frozen_ns=time.time_ns(),sha256={f:sha(ROOT/f) for f in files},seeds=[112001,112013,112037]);save(OUT/'freeze.json',manifest)
        (OUT/'publication_source.py').write_bytes((ROOT/'warm_recovery_release.py').read_bytes())
        result=run(manifest['seeds']);save(OUT/'paired_holdout.json',result)
        save(OUT/'before.json',{m:value['research_accepted'] for m,value in result['summary'].items()})
        save(OUT/'after.json',{m:value['publish_accepted'] for m,value in result['summary'].items()})
        save(OUT/'summary.json',dict(counted_rounds_added=0,summary=result['summary'],initialization_failures_in_overall_denominator=True,unchanged_research_archive=True))
        print(json.dumps(result['summary']))
