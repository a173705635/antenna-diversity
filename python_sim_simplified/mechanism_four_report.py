"""Audit frozen observation replays, compare mechanisms, and render the report."""
from pathlib import Path
import numpy as np
from mechanism_four_study import ROOT,read,save,digest,load


def build():
    freeze=read(ROOT/'freeze.json');selected=freeze['source_sha256']
    models={digest(p):p for p in [ROOT/'baseline.py',ROOT/'selected.py',*ROOT.glob('round*/*.py')]}
    checked={};records={};rows=[];rounds=[]
    for number in range(1,5):
        folder=ROOT/f'round{number:02d}'
        rounds.append(dict(**read(folder/'proposal.json'),**read(folder/'conclusion.json')))
        for path in sorted(folder.glob('*.json')):
            row=read(path)
            if not isinstance(row,dict) or 'after_id' not in row:continue
            meta=read(Path(row['input'])/'metadata.json');records[row['input']]=meta
            for label in ['before','after']:
                key=(row[label+'_root'],row[label+'_id']);cache=Path(key[0])/'evaluations'/key[1]
                if key not in checked:
                    with np.load(cache/'tracking.npz') as d:z=dict(d)
                    with np.load(cache/'navigation.npz') as d:position=d['position'].copy()
                    actual=load(models[row[label]['source_sha256']]).navigate(z)
                    np.testing.assert_allclose(actual['position'],position,atol=1e-9,rtol=0,equal_nan=True)
                    post=z['available_time']>=meta['case']['attack_start']
                    truth=np.zeros_like(position);truth[:,0]=meta['case']['velocity_east']*z['available_time']
                    error=np.linalg.norm((position-truth)[:,:2],axis=1)[post];finite=np.isfinite(error)
                    for name,value in [('opportunities',len(error)),('solved',finite.sum()),('good20',(error<=20).sum()),('error30',(error>=30).sum()),('error50',(error>=50).sum())]:
                        assert row[label][name]==int(value),(key,name)
                    gap=float((np.diff(np.flatnonzero(np.r_[True,finite,True]))-1).max()*.01)
                    assert gap==row[label]['longest_unavailable_s']
                    pre=int(np.isfinite(position[~post]).all(1).sum())
                    checked[key]=dict(ready=bool(pre and row[label]['initialized_before_attack']),pre_outputs=pre,
                        max_error=float(np.nanmax(error)) if finite.any() else None,valid_median=float(np.median(actual['selected'][post].sum(1))),
                        tail_good20=int((error[-100:]<=20).sum()),tail_opportunities=min(100,len(error)))
                row[label+'_diagnostics']=checked[key]
            row['round']=number;rows.append(row)
    for folder,meta in records.items():
        assert digest(Path(folder)/'samples.npy')==meta['samples_sha256']
        assert digest(Path(folder)/'observations.npz')==meta['observations_sha256']
    for name,sha in freeze['dependencies'].items():assert digest(Path(name))==sha,name
    for name,sha in read(ROOT/'mechanism_dependencies.json').items():assert digest(Path(name))==sha,name
    assert digest(Path('single_source_receiver_history.py'))==selected
    for front in (ROOT/'frontends').glob('*/reuse.json'):
        source=read(front)
        assert digest(front.parent/'tracking.npz')==source['sha256']==digest(Path(source['source']))
    for cache in (ROOT/'evaluations').iterdir():
        assert all((cache/name).is_file() for name in ['score.json','tracking.npz','navigation.npz']),cache
    latest={(r['input'],r['subset']):r for r in rows if r['after']['source_sha256']==selected}
    tests=Path('mechanism_four_tests.log').read_text(encoding='utf-8')
    assert 'passed' in tests and 'failed' not in tests.replace('xfailed','')
    fronts=list((ROOT/'frontends').glob('*/tracking.npz'))
    summary=dict(rounds=rounds,paired_rows=len(rows),unique_navigation_replays=len(checked),unique_rf_records=len(records),
        computed_rf_frontends=sum(not p.with_name('reuse.json').exists() for p in fronts),copied_identical_frontends=sum(p.with_name('reuse.json').exists() for p in fronts),
        selected_rows=list(latest.values()),source_sha256=selected,test_summary=tests.strip().splitlines()[-1],commercial_goal_met=False)
    save(ROOT/'audited_rows.json',rows);save(ROOT/'summary.json',summary)
    lines=['# 四轮机制研究：短筛选间隙与位置接续','',
        '**结论：已知四天线切入错误得到机制性修复，新种子4秒连续通过；三天线仍不能保底。** 旧困难样本四天线165→171/171、354→371/371，最大误差25.76→12.59m、33.87→16.68m，没有减少输出。冻结新种子四天线基线/候选均371/371，最大7.91→5.30m，属于未退化和精度改善验证；三天线仅4→15/371。真实消失负例均停发，未发生错误恢复。强信号近源及撤除仍有短暂中断，尚未证明商用或长期导航。','',
        '本批实际研究两个机制，保留单平滑、单接续和组合三个候选；4轮依次为根因/单机制、消融、统一版本回归和冻结新种子验证。没有把种子数当迭代数。基线是上一批 single_source_receiver_weak.py，三天线不能保底的失败继续计入。','',
        '## 逐轮记录','', '|轮|问题|结论|证据|','|---|---|---|---|']
    for r in rounds:lines.append(f"|{r['number']}|{r['problem']}|{r['decision']}|{r['evidence']}|")
    lines+=['','## 组合候选：同RF的完整场景对照','',
        '每格为水平误差≤20m的输出/全部机会；停发计入分母。第3轮及此前所有旧种子均为开发/回归，第4轮才是本批独立留出。真实消失的过渡输出不算持续导航成功。','',
        '|轮|输入|天线|基线合格|候选合格|基线最大m|候选最大m|候选≥30/≥50m|最长缺失s|启动前输出|末1秒合格|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in latest.values():
        b,a=r['before'],r['after'];bd,d=r['before_diagnostics'],r['after_diagnostics']
        bm='无输出' if bd['max_error'] is None else f"{bd['max_error']:.2f}";am='无输出' if d['max_error'] is None else f"{d['max_error']:.2f}"
        lines.append(f"|{r['round']}|{Path(r['input']).name}|{r['subset']}|{b['good20']}/{b['opportunities']}|{a['good20']}/{a['opportunities']}|{bm}|{am}|{a['error30']}/{a['error50']}|{a['longest_unavailable_s']:.2f}|{d['pre_outputs']}|{d['tail_good20']}/{d['tail_opportunities']}|")
    lines+=['','## 消融','', '|输入|天线|候选|≤20m/机会|最大误差m|≥30m|','|---|---|---|---:|---:|---:|']
    for r in rows:
        if r['round']>2:continue
        a,d=r['after'],r['after_diagnostics'];m='无输出' if d['max_error'] is None else f"{d['max_error']:.2f}"
        lines.append(f"|{Path(r['input']).name}|{r['subset']}|{Path(r['model']).stem}|{a['good20']}/{a['opportunities']}|{m}|{a['error30']}|")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    group=[r for r in latest.values() if r['subset']=='four' and records[r['input']]['kind']=='weak']
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained');y=np.arange(len(group))
    for shift,label,color in [(-.18,'before','#9aa4b2'),(.18,'after','#287e80')]:
        ax.barh(y+shift,[100*r[label]['good20']/r[label]['opportunities'] for r in group],height=.34,label=label,color=color)
    ax.set_yticks(y,[Path(r['input']).name for r in group],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,105)
    ax.set_xlabel('Within 20 m / all opportunities (%)');ax.set_title('Four antennas: development and frozen holdout');ax.legend()
    fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    lines+=['','![四天线开发与冻结留出](results/mechanism-four/comparison.png)','','## 方法、边界和后续','',
        '- 核心修改是保留同跟踪弧、最多跨一个被拒历元的测量历史，以及从最近已发布位置计算实际载波位移；没有冻结旧位置充数、放宽20m目标或读取仿真真值。被拒测量不更新状态，重捕和超时重置。',
        '- 已知0.34s的12颗星与0.33s的有效集合不相交，但与0.32s有12颗同弧共同星。跨20ms的实际载波拟合残差仅0.00821m。原方法同时重置平滑并放弃载波接续，解释了单历元码误差如何进入位置。',
        '- 单平滑消除了已知4秒记录的30m错误，但另一旧种子仍有20m超限；单接续在4秒仍3次20m超限。组合在这两份旧记录全部20m内，并未减少输出；它们现在是开发样本，不能当独立泛化证明。',
        '- 三天线困难记录仍因可靠观测不足和恢复证据不连续停发。两天线也没有被补齐；正常弱信号及零陷后的可辨识问题分别保留。',
        '- 本批强信号近源/撤除是使用当前统一前端重跑。更早空间触发版本曾经通过的结果不能自动继承；当前基线已有的中断也列在表中。撤除回归为历史8星，其余主案例24星，不能说已经测过24星撤除。',
        '- 同跟踪弧不排除未检测周跳；低残差也不能证明历史绝对位置无偏。阈值未经现场完整性标定；仍依赖同步、标称固定姿态/导向、星历与粗码周期辅助。只有2至4秒仿真，无台架、分钟级导航或商用保证。','',
        '## 审计与复现','',
        f"{len(rows)}组配对记录，{len(records)}份RF输入，{len(checked)}份唯一导航重演一致；本批从RF计算{summary['computed_rf_frontends']}个前端，另按相同AST/RF身份复制{summary['copied_identical_frontends']}个冻结前端。输入、运行依赖和复制观测哈希均核对。",'',summary['test_summary'],'',
        '[研究入口](single_source_receiver_history.py)、[数学方法](MECHANISM_FOUR_METHODS.md)、[计划](MECHANISM_FOUR_PLAN.md)、[审计明细](results/mechanism-four/audited_rows.json)、[冻结配置](results/mechanism-four/freeze.json)。','',
        '`python mechanism_four_report.py` 重演并审计；`python -m pytest tests -q` 运行测试；`python mechanism_four_study.py 轮号 模型路径 输入目录 four` 执行配对，已有身份的缓存复用，不覆盖历史结果。']
    Path('MECHANISM_FOUR_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({k:summary[k] for k in ['paired_rows','unique_navigation_replays','unique_rf_records','computed_rf_frontends','copied_identical_frontends']},flush=True)


if __name__=='__main__':build()
