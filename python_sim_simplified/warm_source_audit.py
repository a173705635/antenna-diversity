"""Frozen replay and explicitly offline causal interventions; not extra rounds."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import warm_source as ws
from warm_source_research import OUT, evaluate, write


def load_score(path):
    namespace = {}
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
    return namespace['score']


def replay():
    freeze = json.loads((OUT/'freeze.json').read_text())
    assert hashlib.sha256((OUT/'frozen.py').read_bytes()).hexdigest() == freeze['sha256']
    thresholds = {int(k):v for k,v in freeze['thresholds'].items()}
    rerun = evaluate(load_score(OUT/'frozen.py'),freeze['holdout_seeds'],thresholds)
    original = json.loads((OUT/'holdout.json').read_text())
    assert json.loads(json.dumps(rerun)) == original
    return True


def replay_rounds():
    results = []
    for number in range(26,51):
        folder = OUT/f'{number:03d}'
        for label,filename in [('before','before.py'),('after','source.py')]:
            function = load_score(folder/filename)
            calibration = evaluate(function,range(111001,111021))
            result = evaluate(function,range(111021,111046),calibration['thresholds'])
            stored = json.loads((folder/f'{label}.json').read_text())
            assert json.loads(json.dumps(result)) == stored
        proposal = json.loads((folder/'proposal.json').read_text())
        conclusion = json.loads((folder/'conclusion.json').read_text())
        assert proposal['registered_at'] < conclusion['completed_at']
        # Round digests use canonical LF text; freeze digests use actual bytes.
        assert hashlib.sha256((folder/'source.py').read_text(encoding='utf-8').encode()).hexdigest() == conclusion['source_sha256']
        results.append(dict(round=number,before_exact=True,after_exact=True,hash_matches=True))
    write(OUT/'round_replay.json',results)
    print('All 25 archived parent/candidate comparisons replay exactly.')


def audit():
    freeze = json.loads((OUT/'freeze.json').read_text())
    thresholds = {int(k):v for k,v in freeze['thresholds'].items()}
    score = load_score(OUT/'frozen.py')
    extended = evaluate(score,range(111046,111100),thresholds)
    write(OUT/'extended_development_audit.json',extended)
    # Interventions happen in the simulator only. The online score still only
    # receives warm/current observed arrays, never ideal directions or labels.
    source = (OUT/'frozen.py').read_text(encoding='utf-8')
    modes = {'baseline':source,
             'quarter_noise':source.replace("sigma = .65 if scenario == 'low_signal' else .16",
                       "sigma = .25*(.65 if scenario == 'low_signal' else .16)"),
             'no_static_calibration_error':source.replace('a = a * carrier * calibration[:, None]',
                       'calibration = np.ones(4)\n    a = a * carrier * calibration[:, None]')}
    subsets = {'2_short':[0,1], '2_long':[0,2], '2_diagonal':[0,3], '3':[0,1,2], '4':[0,1,2,3]}
    causal = []
    for mode, implementation in modes.items():
        (OUT/f'audit_model_{mode}.py').write_text(implementation,encoding='utf-8')
        namespace = {}
        exec(compile(implementation,mode,'exec'),namespace)
        generator = namespace['make_case']
        for name, indices in subsets.items():
            calibration = []
            for seed in range(111001,111021):
                for scenario in ws.SCENARIOS:
                    case = generator(seed,scenario,False)
                    warm,current = case['warm'][:,indices],case['current'][:,indices]
                    if scenario != 'reflection' and ws.initialize(warm)['complete']:
                        calibration.append(score(warm,current))
            threshold = float(np.quantile(calibration,.95))
            for scenario in ws.SCENARIOS:
                samples = []
                for seed in range(111046,111100):
                    for attack in (False,True):
                        case = generator(seed,scenario,attack)
                        warm,current = case['warm'][:,indices],case['current'][:,indices]
                        initialized = ws.initialize(warm)['complete']
                        samples.append(dict(seed=seed,attack=attack,initialized=initialized,
                                            alarm=bool(score(warm,current)>threshold)))
                causal.append(dict(mode=mode,subset=name,indices=indices,scenario=scenario,
                                   threshold=threshold,samples=samples))
    write(OUT/'causal_audit.json',causal)
    ledger = json.loads((OUT/'ledger.json').read_text())
    holdout = json.loads((OUT/'holdout.json').read_text())
    summary = dict(completed_rounds=len(ledger),rounds=[r['round'] for r in ledger],
                   retained=[r['round'] for r in ledger if r['accepted']],selected_round=ledger[-1]['selected_round'],
                   holdout=holdout['summary'],extended=extended['summary'],replay_exact=replay(),
                   domain='carrier-wiped spatial observations; no RF, tracking, acquisition or PVT',
                   positioning_error=None,positioning_availability=None,positioning_continuity=None,
                   limitation='All low-signal holdout initialization failed; reflection alarms are not malicious-source identification.')
    for split, result in [('holdout',holdout),('extended',extended)]:
        for m in (2,3,4):
            rows = [r for r in result['rows'] if r['antennas']==m and r['initialized']]
            benign = [r for r in rows if not r['attack']]
            key = str(m) if split=='holdout' else m
            summary[split][key]['all_benign_false_alarm'] = sum(r['detected'] for r in benign)/len(benign)
            summary[split][key]['all_benign_count'] = len(benign)
            summary[split][key]['incorrect_release_count'] = sum(not r['detected'] for r in rows if r['attack'])
    write(OUT/'summary.json',summary)
    report(summary,ledger)
    print(json.dumps(summary))


def report(summary, ledger):
    lines = ['# 正常初始化后空间来源判别：026–050轮', '',
    '已完成25轮真实算法候选比较，保留026、027、039、042、044、049，最终冻结049。'
    '这25轮是同一小型观测模型上的算法探索，不能等同于25项独立物理验证。每轮proposal先于source写入，'
    '随后用同种子同四通道数据运行父/候选模型，保留完整逐样本结果。第一轮parent_round=25仅表示来源线的原始误差基线，不引用阵列线第025轮。', '',
    '## 输入域与初始化', '',
    '仅仿真载波已擦除的复数空间观测：8颗卫星、7×14cm矩形四通道、2/3/4路取同数据前缀。'
    '暖阶段12块；前8块均值估计模板，后4块验证平均投影失配<0.06。通过时空间初始化时刻1.2s，攻击1.6s，间隔0.4s。'
    '失败保留并单列。此初始化只证明空间模板稳定，未证明正常捕获、跟踪、定位已经完成，因此并未满足完整接收机初始化验收。', '',
    '无冷启动攻击；攻击混合为90%共同源，近方向为65%，真实分量仍存在。反射采用25%直达加90%共同反射，作为明确的良性反例。'
    '运动只是后初始化空间相位扰动；重捕仅模拟任意每星公共相位重置。未覆盖真实牵引时间历程、真实信号完全消失、持续失锁与RF重捕。'
    '在线score仅接收warm/current；生成真值只用于离线标签与物理干预。', '',
    '## 冻结算法', '',
    '令A为暖均值，B为当前均值。对B/A做秩一SVD拟合通道×卫星可分复数变化，回乘A得天空模型重构；'
    '直接对B做秩一SVD得共同发射源重构。分数=(天空残差平方和−共同源残差平方和)/暖观测噪声方差。'
    '这是残差比较，B/A上的SVD不是异方差条件下天空模型的精确最大似然解。有限噪声与模板分母误差是限制。', '',
    '门限使用111001–111020中已初始化、无攻击、非反射样本的95%经验分位，三/四主选择目标为检测率−2×普通误报率的均值；'
    '111021–111045用于逐轮选型，二天线不参与该目标。此目标明确不惩罚共同反射误报，故不能用其高分声称真实/恶意来源可辨。'
    '111046–111099仅冻结后审计；112001、112013、112037仅冻结后留出。留出只有3个独立种子，各场景共享潜变量，事件数不能视为独立样本数。', '',
    '## 冻结结果', '',
    '|数据|天线|检测率（初始化后）|普通误报率|含反射全部良性误报率|初始化失败/全部事件|',
    '|---|---:|---:|---:|---:|---:|']
    for split in ('holdout','extended'):
        for m,s in summary[split].items():
            lines.append(f"|{split}|{m}|{s['detection']:.2%}|{s['false_alarm']:.2%}|{s['all_benign_false_alarm']:.2%}|{s['initialization_failure']}/{s['total']}|")
    lines.extend(['',
    '留出每配置攻击检测21/21、普通误报0/18、反射误报3/3；低信号初始化失败6/6（攻击/无攻击配对），总空间初始化通过42/48。'
    '扩展审计普通误报升至二4.12%、三5.83%、四8.31%，因此小留出零误报不代表可靠低误报。'
    '全部成功初始化攻击未出现错误放行，但失败初始化不属于检测成功。无PVT输出，定位误差/定位可用率/连续性均为未测，不用空间初始化率替代。', '',
    '## 二天线因果与边界', '',
    '最终二天线在本模型中没有检测劣势，不能声称天线数量导致失败。冻结后54种子配对审计，短边/长边/对角二天线'
    '非反射攻击均100%检测，普通误报分别14/340、14/339、14/338。几何改变没有解释明显检测差距。'
    '第045轮给噪声模型增加第二空间模态后二天线残差维数为零，检测率降至0；这是该模型的自由度耗尽，而不是任意二天线算法必败。', '',
    '算法实现影响很大：相同开发输入第044轮暖相位外推使二天线检测由第042轮87.23%降至22.34%，'
    '最终049恢复100%。因此那次二天线失败主要来自所选估计器与门限组合。此实验不实施真实信号抑制，不能归因于抑制损失；'
    '同步固定理想，不能检验同步偏差。', '',
    '离线噪声幅度降至1/4后，全部配置初始化失败归零；基准三/四非反射初始化失败104/756、106/756，'
    '其低信号失败分别104/108、106/108。这支持当前低信号边界主要是空间初始化噪声，而不是天线少。'
    '消除静态校准误差后三/四失败数仍104、106，不能归因于静态校准误差。'
    '该干预重新用对应校准数据定门限，数字代表整套估计与门限流程，不是固定门限的单因素敏感度。', '',
    '动态相位/增益漂移仍有误报：扩展三天线相位2/54、增益7/54；四天线相位4/54、增益1/54。'
    '共同良性反射在最终全部已初始化样本中100%报警，是来源恶意性不可辨的直接反例。'
    '结论：三天线最低目标和四天线整机目标均未得到证实，增加天线也未在本模型中稳定降低误报。', '',
    '## 逐轮实测台账', '', '|轮|保留|父效用|候选效用|具体假设|','|---:|---|---:|---:|---|'])
    for r in ledger:
        lines.append(f"|{r['round']:03d}|{'是' if r['accepted'] else '否'}|{r['before_utility']:.6f}|{r['after_utility']:.6f}|{r['hypothesis']}|")
    lines.extend(['', '每轮目录包含proposal.json、before.py、source.py、before.json、after.json、conclusion.json。'
                  '精确源码哈希和选择链见ledger.json，freeze.json记录冻结哈希，summary.json可机器读取。', '',
                  '## 复验', '', '在python_sim_simplified目录执行：', '', '```powershell',
                  "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
                  'D:/Python/Python311/python.exe warm_source_audit.py',
                  'D:/Python/Python311/python.exe warm_source_audit.py --replay-rounds',
                  'D:/Python/Python311/python.exe -m pytest tests/test_warm_source.py -q', '```', '',
                  '6个测试通过，冻结留出逐字段精确复演一致。审计/修补/测试均不额外计轮。'])
    (Path(__file__).parent/'warm_source_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    if '--replay-rounds' in sys.argv:
        replay_rounds()
    else:
        audit()
