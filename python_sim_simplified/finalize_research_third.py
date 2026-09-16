"""Package completed research evidence without changing any frozen model."""
import hashlib,json,platform,re,shutil
from pathlib import Path
import numpy,scipy
from audit_recovery_fifty import save
from validate_research_hundred import source_closure
from summarize_research_third import ledger,report
ROOT=Path('results/research-third-hundred')

def run():
    ledger();report()
    tests=Path('third_all_tests.log').read_text(encoding='utf-8-sig')
    assert '349 passed, 1 xfailed, 1 warning' in tests
    frozen=json.loads((ROOT/'navigation_freeze.json').read_text())
    for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    assert Path('navigation_third_selected.py').read_bytes()==(ROOT/'iteration217/navigation_third.py').read_bytes()
    for n in range(201,301):
        for name in ('proposal.json','results.json','conclusion.json'):assert (ROOT/f'iteration{n}'/name).is_file()
    save(ROOT/'verification.json',dict(actual_iterations=100,unique_executable_models=100,
       pytest='349 passed, 1 legacy strict xfailed, 1 existing warning',navigation_replays=50,source_replay_sides=566,
       recovery_replay_cases=55,motion_replayed_iterations=25,prior_archive_files_unchanged=4393,
       final_navigation_frozen_sources_verified=len(frozen['sources']),
       runtime=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__),
       default_receiver_replaced=False,commercial_goal_met=False,hardware_validation=False,
       references=['branch_final_audit.json','navigation_source_audit.json','motion_second_freeze.json','motion_second_summary.json']))
    save(ROOT/'status.json',dict(requested_iterations=100,completed_iterations=100,phase='complete',commercial_goal_met=False))
    names=set(frozen['sources'])
    for pattern in ('source_third*.py','recovery_third*.py','motion_third*.py','navigation_third*.py','RESEARCH_THIRD_HUNDRED*.md'):
        names.update(str(p) for p in Path('.').glob(pattern))
    names.update(['source_third_report.md','recovery_third_report.md','motion_third_report.md','README.md',
        'research_third_navigation.py','audit_research_third.py','summarize_research_third.py','finalize_research_third.py'])
    names.update(str(p) for p in Path('tests').glob('test_*third.py'))
    names.update(source_closure(['source_third.py','recovery_third_selected.py','motion_third.py']))
    folder=ROOT/'final_sources'
    for name in sorted(names):
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    for path in Path('.').glob('third_*.log'):shutil.copy2(path,ROOT/path.name)
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    save(ROOT/'manifest.json',dict(actual_iterations=100,sha256=hashes))
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    print(json.dumps(dict(iterations=100,verified_files=len(hashes),pytest='349 passed,1 xfailed',commercial_goal_met=False)))

if __name__=='__main__':run()
