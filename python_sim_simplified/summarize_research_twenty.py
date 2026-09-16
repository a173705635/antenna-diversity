"""Generate a factual report and verify frozen research artifacts."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_recovery_fifty import save

ROOT=Path('results/research-twenty')
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
frozen=read(ROOT/'final_frozen.json')
for name,digest in frozen['sha256'].items():
    assert hashlib.sha256((ROOT/'final_source'/name).read_bytes()).hexdigest()==digest
for name in ('navigation_candidate.py','calibration_candidate.py','audit_joint_startup.py'):
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==frozen['sha256'][name]
ledger=[]
for n in range(1,21):
    folder=ROOT/f'iteration{n:02d}'
    ledger.append(dict(proposal=read(folder/'proposal.json'),conclusion=read(folder/'conclusion.json')))
    if n<20:assert (folder/'results.json').is_file()
rf=sorted([read(p) for p in (ROOT/'iteration20').glob('*/result.json')],key=lambda r:r['seed'])
assert len(rf)==15
holdout=[r for r in rf if r['seed']>=23001];assert len(holdout)==10
assert {r['kind']:r['seed'] for r in holdout}==frozen['seeds']
save(ROOT/'iteration20/results.json',dict(domain='RF integration',integration=rf[:5],final_validation=holdout))
old=read(ROOT/'old_replay.json');assert len(old['rows'])==240
prompts=read(ROOT/'prompt_holdout.json');assert len(prompts['rows'])==200
tests=(ROOT/'tests.txt').read_text();assert '273 passed' in tests
save(ROOT/'ledger.json',ledger)

lines=['# 20 次研究闭环：结果与逐次决策','',
'本批完成 20 次有实际模型/算法改动的研究闭环，失败尝试也逐项保留。内部求解循环、参数/种子测试、修复后的回归均不另计次数。没有宣称 20 次都改善，更没有宣称达到商用。','',
'最终保留两部分研究代码：双波束绝对锚点＋载波位移递推，以及可观测性约束下的启动联合校准。旧默认接收机未替换；研究入口为 audit_joint_startup.py。','',
'## 已看到的收益及仍然存在的失败','',
'旧批次 321–400 的 160 条记录此次全部作为开发回归：攻击全历元 ≤20m 从 10487/15120（69.36%）增至 12781/15120（84.53%）；>=30m 从 295 降至 6，>=50m 从 24 降至 5。这里仅改变导航后处理，没有给这些旧记录重新生成联合校准观测。','',
'正常输出从 14761/15120（97.63%）降至 14568/15120（96.35%），未满足 99% 目标。上述比例是预设集合的历元加权结果，不是现实成功概率，相邻历元高度相关。','',
'剩余 6 个 >=30m 错误都在旧 round365（C/N0=35dB-Hz），最大 71.205m。旧 candidate 在该记录没有 >=30m 错误，因此这也是新方案引入的具体退步。该记录启动未成功捕获任何 PRN，raw/calibrated 的整条伪距数组完全相同；这时所谓两条链实际不提供两份不同证据。重新锚定会接纳共同错误位置，解间距离为零也不能验证真实性。详见 remaining_failure.json。','',
'新 RF 留出中，受骗启动 seed23002：旧方案 0/44 个历元 ≤20m、有 6 个 >=30m；只加新导航后处理为 0 输出；联合校准后 42/44 个历元 ≤20m，最大 5.848m。该对照支持校准改动的局部收益。带通道相位偏差的 seed23003，联合校准的最大误差 10.17m，比仅新导航的 8.64m 略差，不能说每种场景都改善。','',
'## 逐次改动与决策','',
'| 次数 | 提出的具体问题 / 实际改动 | 结果与判断 |','|---|---|---|']
for n,row in enumerate(ledger,1):
    proposal=row['proposal'];conclusion=row['conclusion']
    lines.append(f"| {n:02d} | {proposal['hypothesis']} | **{conclusion['decision']}**：{conclusion['evidence']} |")
lines+=['','第 20 次集成验证中另发现：第 10 次只修正载波位移，未重置码平滑弧，会留下约 0.543m 污染。已补齐弧重置，并通过精确运动/周跳测试；原版与修复版源码分别保存，不把该回归测试算第 21 次。','',
'## 最终新种子 RF 验证','',
'每条 0.6s，计分 44 个计划历元；此表的“联合”同时使用新校准和最终导航。“导航”保持旧校准，只替换最终导航。>=30m 旧/导航/联合见最后一列。','',
'| 工况 / 种子 | 旧 ≤20m/44 | 导航 ≤20m/44 | 联合 ≤20m/44 | 联合最大误差 m | >=30m 旧/导航/联合 |',
'|---|---:|---:|---:|---:|---:|']
names=dict(normal='正常',cold='受骗启动',phase='受骗启动＋20°通道偏差',moving='接收机2m/s',absent='启动即无真实信号',two='两天线',three='三天线',drag='一致码载波拖曳20m/s',noauth='攻击后真实信号消失',weak='伪信号0dB')
for r in holdout:
    maximum=r['joint']['max_output_m'];display='无输出' if maximum is None else f'{maximum:.3f}'
    lines.append(f"| {names[r['kind']]} / {r['seed']} | {r['old']['good20']} | {r['navigation']['good20']} | {r['joint']['good20']} | {display} | {r['old']['error30']}/{r['navigation']['error30']}/{r['joint']['error30']} |")
lines+=['','两天线的双分量盲校准在数学上不可辨，因此联合模式整条不可用；没有悄悄回退到另一算法冒充联合成功。攻击后无真实源仅最初两历元有输出，窗口仍含攻击前样本，不能解释为没有真实信号仍可持续导航。','',
'## 相关量模型的独立种子复核','',
'20 个新天空种子 × 10 条件，共 200 条相关量记录；每个条件的分母为 20。这是相位校准试验，不是 200 条 RF 定位记录。误差均值包含被判为不可用的数值诊断；不可用估计不用于最终定位。','',
'| 条件 | 校准可用数 /20 | 平均通道相位 RMSE ° | 最大 RMSE ° |','|---|---:|---:|---:|']
for kind,r in prompts['summary'].items():
    lines.append(f"| {kind} | {r['usable']} | {r['mean_error_deg']:.3f} | {r['max_error_deg']:.3f} |")
lines+=['','无真实分量、两天线都被拒绝；相干和多径仍存在拒绝/精度退化。当前奇异值阈值只是研究可观测性判据，不是认证或商用误警指标。','',
'## 验证、计数与局限','',
'- 01–10：12 条唯一旧 RF 记录的机制开发对照，重复回放不增独立样本数；之后另做全部 240 条旧记录回归。',
'- 11–19：40 条唯一开发相关量输入，比较九种模型修订；另有 200 条最终相关量条件记录。',
'- 20：5 条新 RF 集成诊断＋10 条最终新种子验证，共 15 条唯一 IQ 输入，每条分别运行旧、新接收机。',
'- 273 项测试通过，1 项既有 tracking.py 数值警告。最初无目录范围的 pytest 误收集归档内同名测试，发生 126 项导入冲突；已改为显式 pytest tests，保留原失败日志，没有删除测试或掩盖失败。',
'- 每次方案源码均已独立重载：120 次导航方案回放、360 次校准方案复算、15 条 RF 观测定位重放全部匹配保存结果。上批归档 1343 项内容哈希复核通过。',
'- 载波运动、前缀因果性、单星周跳修复、双周跳不可解、校准相位不变性均有测试；一致 200m 伪位置及一致拖曳仍可骗过算法，作为负对照保留。',
'- 最长 8s 的收益只来自旧记录的导航后处理。最终联合校准新 RF 测试仅 0.6s，尚无长期导航、手机姿态变化、真实多径/温漂/同步 RF 台架证据。',
'','结论：本批有实质算法进展和新的可复现收益，但还存在超 50m 错误和正常可用性损失；不能放行商用。下一步应处理共同受骗后的重新锚定与误差增长约束，不能把两条同源观测的一致性当作独立认证。','',
'数学模型、前提和复现说明见 RESEARCH_TWENTY_METHODS.md。证据位于 results/research-twenty/，每次包含 proposal / 源码快照 / results / conclusion。','',
'![旧数据回归比较](results/research-twenty/comparison.png)','',
'![仍然存在的失败](results/research-twenty/failure365.png)']
Path('RESEARCH_TWENTY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(10,3.6))
axes[0].bar(['Old','Navigation'],[10487/15120*100,12781/15120*100],color=['#64748b','#0f766e'])
axes[0].set_ylabel('All-epoch <=20 m (%)');axes[0].set_ylim(0,100)
axes[1].bar(['Old','Navigation'],[295,6],color=['#64748b','#0f766e'])
axes[1].set_ylabel('Epochs with error >=30 m')
fig.suptitle('Old-data regression only: 15,120 attacked epochs')
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
path=Path('results/recovery-four-hundred/round365/1100365_1.npz')
with np.load(path) as z:time=z['time_s'];truth=z['truth_position']
with np.load(ROOT/'old_replay/1100365_1.npz') as r:
    fig,ax=plt.subplots(figsize=(9,3.5))
    for name,label in [('old','Old'),('position','Navigation')]:
        ax.plot(time,np.linalg.norm((r[name]-truth)[:,:2],axis=1),'.-',label=label,markersize=5)
    ax.axhline(30,color='orange',ls='--');ax.axhline(50,color='red',ls='--')
    ax.set(xlabel='Time (s)',ylabel='Horizontal error (m)',title='Failure retained: low C/N0, agreeing false anchors')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'failure365.png',dpi=160);plt.close(fig)

for name in ('RESEARCH_TWENTY_REPORT.md','RESEARCH_TWENTY_METHODS.md','RESEARCH_TWENTY_PLAN.md',
             'summarize_research_twenty.py','validate_research_twenty.py','replay_research_twenty.py'):
    shutil.copy2(name,ROOT/name)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts}
save(ROOT/'manifest.json',dict(iterations=20,old_rf_records=240,new_rf_records=15,
                              final_rf_holdout_records=10,sha256=manifest))
print(json.dumps(dict(iterations=20,artifact_files=len(manifest),tests='273 passed',new_rf=15)))
