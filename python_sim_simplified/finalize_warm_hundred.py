"""Assemble the 100 mechanism ledger, verify old archive, seal new artifacts."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('results/research-warm-hundred')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    rows=[]
    for number in range(1,26):
        folder=ROOT/'array'/f'iteration{number:03d}'
        p=read(folder/'proposal.json');c=read(folder/'conclusion.json');r=read(folder/'results.json')
        rows.append(dict(number=number,line='array',hypothesis=p['hypothesis'],decision=c['decision'],
                         result=c['reason'],evidence=str(folder/'results.json'),before=r['totals']['before'],after=r['totals']['after']))
    for row in read(ROOT/'source/ledger.json'):
        rows.append(dict(number=row['round'],line='source',hypothesis=row['hypothesis'],
                         decision='retain' if row['accepted'] else 'reject',result=row['conclusion'],
                         evidence=str(ROOT/'source'/f"{row['round']:03d}"/'after.json'),
                         before=row['before_utility'],after=row['after_utility']))
    for row in read(ROOT/'recovery/ledger.json'):
        c=row['conclusion']
        rows.append(dict(number=row['round'],line='recovery',hypothesis=row['method'],
                         decision='retain' if c['retained'] else 'reject',
                         result=f"utility {c['before_utility']} -> {c['after_utility']}",
                         evidence=str(ROOT/'recovery/ledger.json'),before=c['before'],after=c['after']))
    for text in (ROOT/'navigation/ledger.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(text)
        rows.append(dict(number=row['round'],line='navigation',hypothesis=row['hypothesis'],
                         decision='retain' if row['accepted'] else 'reject',result=row['conclusion'],
                         evidence=str(ROOT/'navigation/ledger.jsonl'),before=row['before'],after=row['after']))
    assert [r['number'] for r in rows]==list(range(1,101))
    for row in rows:assert Path(row['evidence']).exists(),row
    save(ROOT/'ledger.json',rows)
    lines=['# 更新需求后100轮逐项台账','',
           '每行是一个具体机制候选实验，四线独立比较，指标不能相加为整机收益。恢复线为预先实现机制库的顺序选型；纠错和补验不计轮。详细数字见链接。','',
           '|轮|研究线|假设/机制|决定|证据|','|---:|---|---|---|---|']
    for r in rows:
        hypothesis=r['hypothesis'].replace('|','/')
        lines.append(f"|{r['number']:03d}|{r['line']}|{hypothesis}|{r['decision']}|[配对结果]({r['evidence'].replace(chr(92),'/')})|")
    Path('RESEARCH_WARM_HUNDRED_LEDGER.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    oldroot=Path('results/research-third-hundred')
    old=read(oldroot/'manifest.json')['sha256']
    for name,expected in old.items():assert digest(oldroot/name)==expected,name
    save(ROOT/'old_archive_verification.json',dict(archive=str(oldroot),verified_files=len(old),unchanged=True))
    assert read(ROOT/'array/replay.json')['passed']
    assert all(row['before_exact'] and row['after_exact'] and row['hash_matches'] for row in read(ROOT/'source/round_replay.json'))
    test_log=(ROOT/'tests_final.log').read_text(encoding='utf-8')
    assert '370 passed, 1 xfailed, 1 warning' in test_log
    save(ROOT/'status.json',dict(status='100 mechanism experiments complete',completed_rounds=100,
         independent_research_lines=4,iterations_per_line=25,
         continuous_rf_integration_complete=False,three_antenna_goal_met=False,commercial_goal_met=False,
         default_receiver_changed=False,cold_attack_excluded=True,
         test_result='370 passed, 1 xfailed, 1 existing warning',
         audit_corrections=['exact source geometry validation','recovery initialization publication gate','071 partial crossfit clarification'],
         finished_utc=datetime.now(timezone.utc).isoformat()))
    source_paths=set(Path('.').glob('warm_*.py'))|set(Path('.').glob('*warm_array.py'))|{Path('finalize_warm_hundred.py')}
    source_paths |= set(Path('tests').glob('test_warm_*.py'))
    source_paths |= set(Path('.').glob('RESEARCH_WARM_HUNDRED_*.md'))
    source_paths |= {Path('RESEARCH_SCOPE.md')}
    for source in source_paths:
        dest=ROOT/'final_sources'/source
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=ROOT/'manifest.json']
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in sorted(paths)}
    save(ROOT/'manifest.json',dict(completed_rounds=100,sha256=manifest))
    for name,expected in manifest.items():assert digest(ROOT/name)==expected,name
    print(json.dumps(dict(rounds=len(rows),old_files_verified=len(old),new_files_verified=len(manifest))),flush=True)


if __name__=='__main__':main()
