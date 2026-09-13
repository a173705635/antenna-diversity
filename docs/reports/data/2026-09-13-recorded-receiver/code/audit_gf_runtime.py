"""Deterministic lifecycle event replay; no RF performance or runtime benchmark."""
import hashlib
import json
from pathlib import Path
from gf_runtime import GFRuntime


def main():
    target=Path(__file__).resolve().parents[1]/'docs/reports/data/2026-09-12-gf-runtime'
    target.mkdir(parents=True,exist_ok=False)
    runtime=GFRuntime(.025,.25)
    events=[('boot',0,{}),('confirm',.01,{'branch_confirmation':('arc1','cal1')}),
            ('loss',.02,{'valid':False}),('reacquired',.03,{}),
            ('reconfirm',.04,{'branch_confirmation':('arc1','cal1')}),
            ('temperature_outside_calibration',.05,{'calibration_valid':False}),
            ('temperature_back',.06,{}),('confirm_again',.07,{'branch_confirmation':('arc1','cal1')}),
            ('gap',.2,{}),('gap_reconfirm',.21,{'branch_confirmation':('arc1','cal1')}),
            ('new_arc',.22,{'arc_id':'arc2'}),
            ('new_arc_confirm',.23,{'arc_id':'arc2','branch_confirmation':('arc2','cal1')})]
    rows=[]
    for event,t,changes in events:
        args=dict(valid=True,arc_id='arc1',calibration_id='cal1',calibration_valid=True,nuisance_bound_m=.001)
        args.update(changes)
        result=runtime.step(t,.02,.01,**args)
        rows.append({'event':event,'time_s':t,**result})
    data={'scope':__doc__,'illustrative_settings':{'threshold_m':.01,'residual_m':.02,
          'bound_m':.001,'max_gap_s':.025,'anomaly_multiplier':.25},'events':rows,
          'external_confirmation':'Injected test events, not receiver-derived integrity evidence.',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('gf_runtime.py'),Path(__file__).with_name('gf_quality_gate.py')]}}
    (target/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
