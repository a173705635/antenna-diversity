"""Replay all ten-round navigation records and verify RF/source identities."""
from pathlib import Path
import numpy as np
from weak_ten_study import ROOT,read,save,digest,load


def build():
    freeze=read(ROOT/'freeze.json');selected=freeze['source_sha256']
    models={digest(p):p for p in [ROOT/'baseline.py',ROOT/'selected.py',*ROOT.glob('round*/receiver.py')]}
    rows=[];rounds=[];checked={};records={}
    for number in range(1,11):
        folder=ROOT/f'round{number:02d}'
        rounds.append(dict(**read(folder/'proposal.json'),**read(folder/'conclusion.json')))
        for path in sorted(folder.glob('*.json')):
            row=read(path)
            if not isinstance(row,dict) or 'after_id' not in row:continue
            meta=read(Path(row['input'])/'metadata.json');records[row['input']]=meta
            for label in ('before','after','parent'):
                if label not in row:continue
                key=(row[label+'_root'],row[label+'_id']);cache=Path(key[0])/'evaluations'/key[1]
                if key not in checked:
                    with np.load(cache/'tracking.npz') as data:z=dict(data)
                    with np.load(cache/'navigation.npz') as data:position=data['position'].copy()
                    model=load(models[row[label]['source_sha256']]);actual=model.navigate(z)
                    np.testing.assert_allclose(actual['position'],position,rtol=0,atol=1e-9,equal_nan=True)
                    post=z['available_time']>=meta['case']['attack_start']
                    truth=np.zeros_like(position);truth[:,0]=meta['case']['velocity_east']*z['available_time']
                    errors=np.linalg.norm((position-truth)[:,:2],axis=1)[post]
                    for name,value in [('opportunities',len(errors)),('solved',np.isfinite(errors).sum()),('good20',(errors<=20).sum()),('error30',(errors>=30).sum()),('error50',(errors>=50).sum())]:
                        assert row[label][name]==int(value),(key,name)
                    published=np.isfinite(errors)
                    gap=float((np.diff(np.flatnonzero(np.r_[True,published,True]))-1).max()*.01)
                    assert abs(gap-row[label]['longest_unavailable_s'])<1e-10
                    pre=int(np.isfinite(position[~post]).all(1).sum())
                    checked[key]=dict(ready=bool(pre and row[label]['initialized_before_attack']),pre_attack_outputs=pre,
                        max_error=float(np.nanmax(errors)) if published.any() else None,
                        valid_median=float(np.median(actual['selected'][post].sum(1))),
                        late_good20=int((errors[-100:]<=20).sum()),late_opportunities=min(100,len(errors)),
                        first_missing_time=float(z['available_time'][post][~published][0]) if (~published).any() else None)
                row[label+'_diagnostics']=checked[key]
            row['round']=number;rows.append(row)
    for folder,meta in records.items():
        for name,key in [('samples.npy','samples_sha256'),('observations.npz','observations_sha256')]:
            assert digest(Path(folder)/name)==meta[key]
    for name,sha in freeze['dependencies'].items():assert digest(Path(name))==sha,name
    assert digest(Path('single_source_receiver_weak.py'))==selected
    for cache in (ROOT/'evaluations').iterdir():
        assert all((cache/name).is_file() for name in ('score.json','tracking.npz','navigation.npz')),cache
    latest={(r['input'],r['subset']):r for r in rows if r['after']['source_sha256']==selected}
    tests=Path('weak_ten_tests.log').read_text(encoding='utf-8')
    assert 'failed' not in tests.replace('xfailed','') and 'passed' in tests
    summary=dict(rounds=rounds,paired_rows=len(rows),unique_navigation_replays=len(checked),unique_rf_records=len(records),
        new_rf_frontend_runs=len(list((ROOT/'frontends').glob('*/tracking.npz'))),selected_rows=list(latest.values()),
        selected_source_sha256=selected,commercial_goal_met=False,reliable_navigation_accepted=False,
        test_summary=tests.strip().splitlines()[-1])
    save(ROOT/'audited_rows.json',rows);save(ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,subset in zip(axes,('three','four')):
        group=[r for r in latest.values() if r['subset']==subset and records[r['input']]['kind']!='absent']
        y=np.arange(len(group))
        for shift,label,color in ((-.18,'before','#a0aab8'),(.18,'after','#287e80')):
            ax.barh(y+shift,[100*r[label]['good20']/r[label]['opportunities'] for r in group],height=.34,color=color,label=label)
        ax.set_yticks(y,[Path(r['input']).name for r in group],fontsize=8);ax.invert_yaxis()
        ax.set_xlim(0,105);ax.set_title(subset+' antennas');ax.set_xlabel('Within 20 m / all opportunities (%)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend()
    fig.savefig(ROOT/'navigation_comparison.png',dpi=160);plt.close(fig)
    lines=['# 弱信号抗欺骗：10轮研究结果','',
        '本批接续空间触发修复，基线为 single_source_receiver_spatial.py。10轮包含问题、假设、实验与结论；实际评估5个代码候选（第3、5、6、7、8轮，其中第8轮为初样本自由度修正），其余为诊断、数值可行性及冻结验证。不把种子或天线运行数当轮数。单源、24星、正常真实导航启动后受攻击；暂停计入分母。','',
        '**结论：部分弱信号场景的可用率明显提高，但候选未通过可靠导航要求。** 开发样本三/四天线30dB-Hz均171/171；冻结新种子三天线仅4/171，四天线165/171；4秒另一种子三天线4/371、四天线354/371，四天线出现2次≥30m错误、最大33.87m。正常弱信号控制改善不能代替抗欺骗放行，三天线尚不保底。','',
        '## 逐轮证据','', '|轮|问题|结论|证据|','|---|---|---|---|']
    for r in rounds:lines.append(f"|{r['number']}|{r['problem']}|{r['decision']}|{r['evidence']}|")
    lines+=['','## 最终研究候选的同RF对照','',
        '数值均为攻击开始（控制组为噪声变化）后，≤20m输出数/全部机会。真实信号消失行的少数输出处于接入缓冲过渡，随后应停发，不能当作持续导航能力。','',
        '|轮|输入|天线|基线≤20m|候选≤20m|最大误差m|≥30/≥50m|最长缺失s|攻击前实际输出数|末1秒≤20m|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in latest.values():
        b,a,d=r['before'],r['after'],r['after_diagnostics'];m='无输出' if d['max_error'] is None else f"{d['max_error']:.2f}"
        lines.append(f"|{r['round']}|{Path(r['input']).name}|{r['subset']}|{b['good20']}/{b['opportunities']}|{a['good20']}/{a['opportunities']}|{m}|{a['error30']}/{a['error50']}|{a['longest_unavailable_s']:.2f}|{d['pre_attack_outputs']}|{d['late_good20']}/{d['late_opportunities']}|")
    lines+=['','![有真实信号场景的全机会合格率](results/weak-ten/navigation_comparison.png)','','## 失败与限制','',
        '- 第3轮只延长Hatch窗口没有收益；24星35dB-Hz的改善首先来自观测冗余，不能归功于新增算法。',
        '- 第6轮只改善跟踪质量时，三天线真实信号消失后出现一次88.85m错误；第7轮补测同版本四天线另有两次≥30m错误。三天线失效历元的码解分离19.11m仍通过，但载波不通过、方向证据全部失效。该版本明确否决；第7轮把方向证据纳入平滑和状态更新前的观测准入。失败RF、源码、结果和回归测试均保留。',
        '- 100ms I/Q功率估计依赖已锁定Costas相位；它与相位锁定指标并非独立证据。当前方向证据用20ms数据，减少质量估计滞后的风险，也可能丢弃弱星、延长恢复中断。',
        '- 开发弱信号最大误差仍为三天线14.16m、四天线17.21m。三天线峰值在1.63s，11颗有效星、载波检查通过，但绝对位置仍有误差，码锚点与载波系统偏差的贡献尚需消融区分；四天线峰值在0.34s，仅1颗同弧共同星、无法载波递推，虽码解分离7.07m通过，真实水平误差仍17.21m。优先研究观测集合切换与绝对码锚点，不能把一致性门限当真实误差上界。',
        '- 4秒四天线的两次≥30m错误发生于0.34/0.35s，即攻击切入后的转换阶段，并非晚段累积：0.34s共同星为0，码解分离18.76m通过，却建立了33.87m错误锚点；0.35s载波残差只有0.0097m，但继承了错误锚点，位置误差仍30.87m。载波增量一致不能证明绝对位置正确；尚未分离码噪声、抑制残留及切换瞬态各自贡献。',
        '- 4秒三天线并非所有时刻都少于6星：367个稳定阶段历元中218个达到6星，但仅8个码解一致；恢复确认最长连续2历元，达不到原10历元条件。不能仅通过削减确认时长就宣称问题解决。',
        '- 阈值3没有现场虚警标定，B和方向证据来自相关RF，不能按独立统计量计算商用完整性保证。残余子空间至少二维才可比较方向能量与正交能量；两天线零陷后仅一维，无法提供该项证据。',
        '- 标称姿态/阵列导向、同步、星历和粗码周期辅助仍是明确前提。没有测试手持姿态误差、实际天线互耦、多径全分布或台架数据；不研究冷启动、多物理源或IMU。',
        '- 本批延长录波仍只有几秒，独立种子也只是仿真留出；未证明分钟级导航、商用可信边界或手机实时处理。误差统计以水平位置为准。','',
        '## 审计与复现','',
        f"{len(rows)}组同RF配对记录，{len(records)}份RF输入，{summary['new_rf_frontend_runs']}次本批RF前端计算，{len(checked)}份导航结果重演一致。输入哈希、冻结依赖、机会数、误差计数、最长缺失和实际启动输出已核对。",'',tests.strip().splitlines()[-1],'',
        '入口：[single_source_receiver_weak.py](single_source_receiver_weak.py)。[数学方法](WEAK_TEN_METHODS.md)、[原计划及调整](WEAK_TEN_PLAN.md)、[逐案例审计](results/weak-ten/audited_rows.json)、[冻结配置](results/weak-ten/freeze.json)。','',
        '复现：`python weak_ten_report.py` 重演导航并核对；`python -m pytest tests -q` 测试。`python weak_ten_study.py run 轮号 模型路径 输入路径 three` 按源码/RF哈希复用或计算；首次生成新输入用 `create 种子 场景 CNO 秒数`，不覆盖已有档案。']
    Path('WEAK_TEN_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(ROOT/'manifest.json',{str(p):digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts})
    print({k:summary[k] for k in ('paired_rows','unique_navigation_replays','unique_rf_records','new_rf_frontend_runs')},flush=True)


if __name__=='__main__':build()
