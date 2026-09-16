"""Uncounted geometry correction: frozen score/thresholds, corrected input only."""
import hashlib
import json
from pathlib import Path
import time
import warm_source as ws
from warm_source_research import OUT, evaluate, write
from warm_source_audit import load_score


def run():
    freeze = json.loads((OUT/'freeze.json').read_text())
    frozen = OUT/'frozen.py'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == freeze['sha256']
    original = frozen.read_text(encoding='utf-8')
    old = 'positions = np.array([[0, 0], [.07, 0], [0, .14], [.07, .14]])'
    new = 'positions = np.array([[0, 0], [.0759, 0], [0, .1609], [.0759, .1609]])'
    assert original.count(old) == 1
    corrected = original.replace(old,new)
    manifest = dict(registered_at=time.time(),frozen_sha256=freeze['sha256'],
                    selected_round=49,thresholds=freeze['thresholds'],seeds=freeze['holdout_seeds'],
                    original_geometry_m=[.07,.14],corrected_geometry_m=[.0759,.1609],
                    change='Only generated antenna coordinates changed; frozen scoring function and thresholds unchanged.',
                    extra_round=False,selection_permitted=False)
    write(OUT/'exact_geometry_manifest.json',manifest)
    (OUT/'exact_geometry_model.py').write_text(corrected,encoding='utf-8')
    namespace = {}
    exec(compile(corrected,'exact_geometry_model.py','exec'),namespace)
    original_generator = ws.make_case
    ws.make_case = namespace['make_case']
    thresholds = {int(k):v for k,v in freeze['thresholds'].items()}
    result = evaluate(load_score(frozen),freeze['holdout_seeds'],thresholds)
    ws.make_case = original_generator
    for m in (2,3,4):
        rows = [r for r in result['rows'] if r['antennas']==m and r['initialized']]
        benign = [r for r in rows if not r['attack']]
        result['summary'][m]['all_benign_false_alarm'] = sum(r['detected'] for r in benign)/len(benign)
        result['summary'][m]['all_benign_count'] = len(benign)
        result['summary'][m]['incorrect_release_count'] = sum(not r['detected'] for r in rows if r['attack'])
    write(OUT/'exact_geometry_holdout.json',result)
    write(OUT/'exact_geometry_summary.json',dict(manifest=manifest,summary=result['summary'],
          domain='Spatial observations only; not RF/PVT',completed_at=time.time()))
    report = Path(__file__).parent/'warm_source_REPORT.md'
    marker = '\n## 项目精确几何补验（不计轮）\n'
    lines = [marker,
    '原026–050轮及原留出使用70×140mm，偏离项目75.9×160.9mm固定安装尺寸；这是原探索输入的几何偏差。'
    '原始证据未改写。以下仅替换生成器四个坐标，沿用冻结049源码评分及原门限，使用112001/112013/112037，'
    '不重新校准、不选型、不新增轮数。总报告应采用本节精确几何结果。', '',
    '|天线|检测率|普通误报率|含反射全部良性误报率|初始化失败/总事件|','|---:|---:|---:|---:|---:|']
    for m,s in result['summary'].items():
        lines.append(f"|{m}|{s['detection']:.2%}|{s['false_alarm']:.2%}|{s['all_benign_false_alarm']:.2%}|{s['initialization_failure']}/{s['total']}|")
    lines.extend(['','上述检测与误报以成功空间初始化为条件；全部逐事件得分、初始化状态见exact_geometry_holdout.json。'
                  '仍只有三个独立种子，不能把重复场景当独立样本。空间初始化不是RF捕获/跟踪/PVT完成证据。', '',
                  '复验：`D:/Python/Python311/python.exe warm_source_exact_geometry.py`。'])
    report.write_text(report.read_text(encoding='utf-8').split(marker)[0]+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    run()
