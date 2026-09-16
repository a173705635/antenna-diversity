"""Final evidence packaging after validation, without changing research models."""
import hashlib,json,platform,re,shutil
from pathlib import Path
import numpy,scipy
from audit_recovery_fifty import save
from research_next_navigation import ROOT
from summarize_research_next import ledger,report

def run():
    ledger();report()
    text=Path('next_all_tests_corrected.log').read_text(encoding='utf-8-sig')
    assert '321 passed, 1 xfailed' in text and ' failed' not in text
    frozen=json.loads((ROOT/'navigation_corrected_freeze.json').read_text())
    for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    rows=json.loads((ROOT/'iteration_structure_audit.json').read_text())['rows']
    assert [r['iteration'] for r in rows]==list(range(101,201))
    save(ROOT/'verification.json',dict(actual_iterations=100,pytest='321 passed, 1 strict xfailed, 1 existing warning',
         strict_xfail='150 pure-noise support false acceptance remains unfixed; rejected research model',
         navigation_behavior_tests=7,navigation_historical_replays=49,delay_replays=20,cold_replays=60,integrity_replays=25,
         prior_archive_hashes=2881,final_navigation_frozen_sources_verified=len(frozen['sources']),
         runtime=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__),
         commercial_goal_met=False,hardware_validation=False))
    save(ROOT/'status.json',dict(requested_iterations=100,completed_iterations=100,phase='complete',commercial_goal_met=False))
    folder=ROOT/'final_sources'
    names=set(frozen['sources'])|set(['delay_next.py','cold_next.py','cold_next_rf.py','integrity_next.py','integrity_next_model.py',
         'audit_research_next.py','summarize_research_next.py','finalize_research_next.py','README.md',
         'RESEARCH_NEXT_HUNDRED_PLAN.md','RESEARCH_NEXT_HUNDRED_REPORT.md','RESEARCH_NEXT_HUNDRED_METHODS.md','RESEARCH_NEXT_HUNDRED_LEDGER.md',
         'delay_next_REPORT.md','cold_next_report.md','integrity_next_report.md'])
    names.update(str(p) for p in Path('tests').glob('test_*next.py'))
    for name in sorted(names):
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    for path in Path('.').glob('next_*.log'):shutil.copy2(path,ROOT/path.name)
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    save(ROOT/'manifest.json',dict(actual_iterations=100,sha256=hashes))
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps(dict(iterations=100,verified_files=len(hashes),pytest='321 passed, 1 xfailed',commercial_goal_met=False)))

if __name__=='__main__':run()
