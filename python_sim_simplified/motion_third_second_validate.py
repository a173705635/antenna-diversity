"""Explicit pre-evaluation freeze; preserves all first-validation artifacts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import scipy
import motion_third_research as research
from motion_third_validate import counterexamples

ROOT=research.ROOT
OUT=research.OUT
FREEZE=OUT/'motion_second_freeze.json'
INDEPENDENT=[97621,97627,97639,97643]
REUSED=97631


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path,value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2)


def dependencies():
    records={}
    for package in (np,scipy):
        base=Path(package.__file__).parent
        records[package.__name__]={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*')) if p.is_file() and p.suffix in ('.py','.pyd','.dll')}
    return records


def freeze():
    # This command hashes inputs only. It does not call simulate/evaluate/estimate.
    started=now()
    package_hashes=dependencies()
    write_new(OUT/'motion_second_dependencies.json',package_hashes)
    local=['motion_third.py','motion_third_research.py','motion_third_validate.py','motion_third_second_validate.py','tests/test_motion_third.py']
    legacy=[OUT/'motion_validation.json',OUT/'motion_summary.json',ROOT/'motion_summary.json',*OUT.glob('motion_frozen_*.npz')]
    legacy += [p for n in range(276,301) for p in (OUT/f'iteration{n}').iterdir() if p.is_file()]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(legacy)}
    write_new(OUT/'motion_second_legacy_manifest.json',manifest)
    (OUT/'motion_second_previous_report.md').write_bytes((ROOT/'motion_third_report.md').read_bytes())
    declaration=dict(created_at_utc=now(),hashing_started_at_utc=started,independent_seeds=INDEPENDENT,
        originally_requested_seeds=[97621,97627,97631,97639],reused_diagnostic_seed=REUSED,
        seed_exclusion_reason='97631 already used by all first-batch motion tests and counterexample test; exclude from independent metrics and replace with 97643.',
        local_source_sha256={name:sha(ROOT/name) for name in local},
        dependencies_file='motion_second_dependencies.json',dependencies_sha256=sha(OUT/'motion_second_dependencies.json'),
        legacy_manifest_sha256=sha(OUT/'motion_second_legacy_manifest.json'),
        python_executable=str(Path(sys.executable)),python_executable_sha256=sha(Path(sys.executable)),
        python_version=sys.version,numpy_version=np.__version__,scipy_version=scipy.__version__,
        environment={key:os.environ[key] for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        cases=research.CASES,windows=100,window_seconds=12,samples_per_window=121,
        additional_cases=['clustered_diverse','shared_drift_direction_gauge'],
        fixed_decision_thresholds=dict(common_phase_spread_rad=.12,minimum_singular=.5,direction_rms_rad=.08,direction_support_degrees=30),
        protocol='Hash verification before evaluating fixed source on 4 unseen seeds; no post-freeze model/threshold selection. Reused seed reported separately. No new research iteration.',
        first_validation_status='Retained evidence; no independent pre-evaluation freeze manifest existed. Its output hash was computed after evaluation, so artifacts alone cannot prove pre-freeze. Do not count as independently preregistered validation.',
        first_validation_usage='Results inspected for reporting and counterexample review. No algorithm/threshold selection based on first holdout is recorded; removal of a numerically irrelevant 1e-30 norm floor was checked against existing development results and first validation rerun. No claim of an independently preregistered first holdout.',
        timestamp_limit='UTC wall-clock and filesystem timestamp plus tool execution order are audit evidence; no external trusted timestamp service or immutable storage is claimed.')
    write_new(FREEZE,declaration)
    print(json.dumps(dict(freeze_file=str(FREEZE),created_at_utc=declaration['created_at_utc'],freeze_sha256=sha(FREEZE),independent_seeds=INDEPENDENT),indent=2))


def validate():
    verified_at=now();m=json.loads(FREEZE.read_text());freeze_hash=sha(FREEZE)
    assert {name:sha(ROOT/name) for name in m['local_source_sha256']}==m['local_source_sha256']
    assert sha(Path(sys.executable))==m['python_executable_sha256']
    assert sha(OUT/m['dependencies_file'])==m['dependencies_sha256']
    assert dependencies()==json.loads((OUT/m['dependencies_file']).read_text())
    assert {key:os.environ[key] for key in m['environment']}==m['environment']
    assert m['created_at_utc']<verified_at
    started=now()
    # Durable start receipt is written after verification but before first evaluation.
    write_new(OUT/'motion_second_started.json',dict(freeze_sha256=freeze_hash,freeze_created_at_utc=m['created_at_utc'],freeze_mtime_ns=FREEZE.stat().st_mtime_ns,
        dependency_verification_started_at_utc=verified_at,evaluation_started_at_utc=started,all_hashes_match=True))
    source=ROOT/'motion_third.py'
    rows=[dict(seed=s,**r) for s in m['independent_seeds'] for r in research.evaluate(source,s)]
    hard=[counterexamples(s) for s in m['independent_seeds']]
    reused=dict(seed=m['reused_diagnostic_seed'],independent=False,
        rows=research.evaluate(source,m['reused_diagnostic_seed']),counterexamples=counterexamples(m['reused_diagnostic_seed']))
    common=[r for r in rows if r['common_truth']];diverse=[r for r in rows if not r['common_truth']]
    available=[r for r in rows if r['direction_available']]
    summary=dict(validation_kind='second_explicit_pre_freeze',research_iterations_added=0,
        seeds=m['independent_seeds'],excluded_reused_seed=m['reused_diagnostic_seed'],windows=len(rows),
        duration_seconds=12,samples_per_window=121,source_sha256=sha(source),freeze_file=FREEZE.name,freeze_sha256=freeze_hash,
        freeze_created_at_utc=m['created_at_utc'],evaluation_started_at_utc=started,evaluation_finished_at_utc=now(),
        pre_and_post_hashes_verified=True,false_alarms=sum(r['false_alarm'] for r in diverse),diverse_windows=len(diverse),
        missed_common=sum(r['missed_common'] for r in common),common_windows=len(common),
        common_abstentions=sum(r['status']=='indeterminate' for r in common),common_not_detected=sum(r['status']!='common' for r in common),
        direction_available_windows=len(available),available_direction_median_error_deg=float(np.median([r['direction_error_deg'] for r in available])),
        available_direction_max_error_deg=max(r['direction_error_deg'] for r in available),
        clustered_diverse_false_alarms=sum(x['clustered_false_alarm'] for x in hard),clustered_diverse_windows=len(hard),
        combined_diverse_false_alarms=sum(r['false_alarm'] for r in diverse)+sum(x['clustered_false_alarm'] for x in hard),combined_diverse_windows=len(diverse)+len(hard),
        common_drift_gauge_max_difference=max(x['shared_drift_gauge_phase_difference'] for x in hard),
        support_diagnostic_violations=sum(r['direction_error_deg']>r['direction_radius_deg'] for r in available),
        support_is_certificate=False,absolute_position_available=0,real_rf_navigation=False)
    for case in m['cases']:
        np.savez_compressed(OUT/f'motion_second_frozen_{case}.npz',
            **{f'seed{s}_{k}':v for s in m['independent_seeds'] for k,v in research.simulate(case,s)[0].items()})
    assert sha(FREEZE)==freeze_hash
    assert {name:sha(ROOT/name) for name in m['local_source_sha256']}==m['local_source_sha256']
    assert sha(OUT/'motion_second_legacy_manifest.json')==m['legacy_manifest_sha256']
    old=json.loads((OUT/'motion_second_legacy_manifest.json').read_text())
    assert all(sha(ROOT/name)==expected for name,expected in old.items())
    summary['legacy_files_verified_unchanged']=len(old)
    write_new(OUT/'motion_second_validation.json',dict(summary=summary,rows=rows,counterexamples=hard))
    write_new(OUT/'motion_second_reused_diagnostic.json',reused)
    write_new(OUT/'motion_second_summary.json',summary)
    write_new(OUT/'motion_second_output_manifest.json',{p.name:sha(p) for p in sorted(OUT.glob('motion_second_*')) if p.is_file()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','validate'))
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    else:validate()
