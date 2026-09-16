"""Re-score saved positions and contrast old/new gates on identical holdout data."""
import argparse,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score,SOURCES
from continuous_admission import Gate,admit
from recovery_admission import Policy,admit as old_admit
from solution_separation import separation_scores

KEYS=('opportunities','solved','good20','error30','error50')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;selected=json.loads((out/'selected.json').read_text());method=selected['method']
    table=[];groups={};trials=0;comparisons=0;worst=[];reference=[]
    for number in range(101,201):
        directory=out/f'round{number:03d}';summary=json.loads((directory/'summary.json').read_text())
        if 141<=number<=160:
            item=next(v for v in summary if v['method']==method)
            table.append(dict(round=number,label='admission '+str(number-140),normal=item['normal'],attack=item['attack'],baseline=None))
            continue
        frozen=json.loads((directory/'frozen.json').read_text());totals={}
        for path in sorted(directory.glob('*.npz')):
            z=np.load(path);attack=path.stem.endswith('_1');suffix='_attack' if attack else '_normal'
            methods=frozen['methods']+(['candidate'] if number>=161 else ['calibrated'])
            for m in methods:
                value=score(z[m+'_position'],z['truth_position']);name=m+suffix
                totals.setdefault(name,{k:0 for k in KEYS})
                for k in KEYS:totals[name][k]+=value[k]
                comparisons+=1
            trials+=1
            if number>=161:
                new=admit(z['satellite_positions'],z[method+'_ranges'],z[method+'_lock'],z[method+'_cn0'],
                          z[method+'_gain'],Gate(**selected['gate']))
                np.testing.assert_allclose(new['position'],z['candidate_position'],rtol=1e-9,atol=1e-7,equal_nan=True)
                old=old_admit(z['time_s'],z['satellite_positions'],z[method+'_ranges'],np.isfinite(z[method+'_ranges']),
                    Policy(residual_m=10.,exclusions=0,minimum_stars=4,geometry_limit=1000.,confirmations=1,use_continuity=False))
                d=separation_scores(z['satellite_positions'],z[method+'_ranges'],old['selected'],old['position'])
                old['position'][d>5.]=np.nan
                old_score=score(old['position'],z['truth_position'])
                reference.append(dict(round=number,input=path.name,attack=attack,old_gate=old_score,
                                      new_gate=score(new['position'],z['truth_position'])))
                error=np.linalg.norm((z['candidate_position']-z['truth_position'])[:,:2],axis=1)
                for k in np.flatnonzero(error>=30):
                    worst.append(dict(round=number,input=path.name,time_s=float(z['time_s'][k]),error_m=float(error[k]),
                         scores=z['candidate_scores'][k].tolist(),stars=int(z['candidate_selected'][k].sum())))
        for name,values in totals.items():
            for k in KEYS:assert values[k]==summary[name][k],(number,name,k)
        group='development' if number<=140 else 'independent' if number<=180 else 'boundary'
        for name,values in totals.items():
            groups.setdefault(group,{}).setdefault(name,{k:0 for k in KEYS})
            for k in KEYS:groups[group][name][k]+=values[k]
        m=method if number<=140 else 'candidate'
        table.append(dict(round=number,label=json.dumps({k:v for k,v in summary['case'].items()
            if k in ('duration','stars','antennas','cn0','angle','phase_deg','fake_east','short_ms')},ensure_ascii=False),
            normal=totals[m+'_normal'],attack=totals[m+'_attack'],baseline=totals['raw_attack']))
    save(out/'rescored.json',dict(groups=groups,rounds=100,unique_rf_trials=trials,position_method_comparisons=comparisons,selected=selected))
    save(out/'same_input_old_gate.json',reference);save(out/'failures.json',sorted(worst,key=lambda r:-r['error_m']))
    lines=['|轮次|配置摘录/准入序号|候选正常≤20m|候选攻击≤20m|攻击≥30m|攻击≥50m|','|---:|---|---:|---:|---:|---:|']
    for row in table:
        a=row['attack'];n=row['normal']
        lines.append(f"|{row['round']}|{row['label']}|{n['good20']}/{n['opportunities']}|{a['good20']}/{a['opportunities']}|{a['error30']}|{a['error50']}|")
    (out/'round_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    for label,group in [('Development signal',table[:40]),('Development admission',table[40:60]),('Frozen holdout',table[60:])]:
        axes[0].plot([r['round'] for r in group],[100*r['attack']['good20']/r['attack']['opportunities'] for r in group],'o-',ms=3,label=label)
        axes[1].bar([r['round'] for r in group],[r['attack']['error30'] for r in group])
    axes[0].set(ylabel='Scheduled attack epochs within 20 m (%)',ylim=(0,102));axes[0].legend()
    axes[1].set(xlabel='Round (different development stages)',ylabel='Candidate output errors >=30 m')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'rounds.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,group in zip(axes,('independent','boundary')):
        data=groups[group];names=list(dict.fromkeys(['raw',method,'candidate']))
        ax.bar(np.arange(len(names))-.18,[100*data[m+'_attack']['good20']/data[m+'_attack']['opportunities'] for m in names],.36,label='Attack good20')
        ax.bar(np.arange(len(names))+.18,[100*data[m+'_normal']['solved']/data[m+'_normal']['opportunities'] for m in names],.36,label='Normal availability')
        ax.set(xticks=np.arange(len(names)),xticklabels=names,ylim=(0,103),title=group,ylabel='All scheduled epochs (%)');ax.legend(fontsize=8)
    fig.savefig(out/'holdout.png',dpi=170);plt.close(fig)
    old_new={}
    for group,low,high in [('independent',161,180),('boundary',181,200)]:
        old_new[group]={}
        for attack in (False,True):
            for policy in ('old_gate','new_gate'):
                values=[r[policy] for r in reference if low<=r['round']<=high and r['attack']==attack]
                old_new[group][policy+('_attack' if attack else '_normal')]={k:sum(v[k] for v in values) for k in KEYS}
    save(out/'same_input_gate_totals.json',old_new)
    trade_root=out/'tradeoff-holdout';trade_frozen=json.loads((trade_root/'frozen.json').read_text())
    trade_summary=json.loads((trade_root/'summary.json').read_text());trade_totals={k:{field:0 for field in KEYS} for k in trade_summary}
    for path in sorted(trade_root.glob('*.npz')):
        z=np.load(path);suffix='_attack' if path.stem.endswith('_1') else '_normal'
        for name,policy in (('primary',trade_frozen['primary']),('alternate',trade_frozen['alternate'])):
            m=policy['method'];result=admit(z['satellite_positions'],z[m+'_ranges'],z[m+'_lock'],z[m+'_cn0'],z[m+'_gain'],Gate(**policy['gate']))
            np.testing.assert_allclose(result['position'],z[name+'_position'],rtol=1e-9,atol=1e-7,equal_nan=True)
        for key in trade_totals:
            if key.endswith(suffix):
                m=key.removesuffix(suffix);v=score(z[m+'_position'],z['truth_position'])
                for field in KEYS:trade_totals[key][field]+=v[field]
    assert trade_totals==trade_summary
    save(out/'tradeoff_rescored.json',trade_totals)
    trade_rows=json.loads((trade_root/'rows.json').read_text())
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for label,offset in (('primary',-.18),('alternate',.18)):
        values=[];failures=[]
        for case in range(10):
            samples=[r[label] for r in trade_rows if r['case']==case and r['attack']]
            values.append(100*sum(v['good20'] for v in samples)/sum(v['opportunities'] for v in samples))
            failures.append(sum(v['error30'] for v in samples))
        axes[0].bar(np.arange(10)+offset,values,.36,label=label)
        axes[1].bar(np.arange(10)+offset,failures,.36,label=label)
    axes[0].set(ylabel='Scheduled epochs within 20 m (%)',ylim=(0,103));axes[0].legend()
    axes[1].set(ylabel='Output errors >=30 m',xlabel='Supplemental case (two shared new sky seeds)',xticks=np.arange(10),
        xticklabels=['Nominal','CN40','Phase5','Phase10','Angle15','Angle0','No genuine','3 antennas','2 sec','Poisoned start'])
    axes[1].tick_params(axis='x',rotation=25)
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'tradeoff.png',dpi=170);plt.close(fig)
    top=max(worst,key=lambda r:r['error_m']);z=np.load(out/f"round{top['round']:03d}"/top['input'])
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for m in ('raw','candidate'):
        ax.plot(z['time_s'],np.linalg.norm((z[m+'_position']-z['truth_position'])[:,:2],axis=1),label=m)
    ax.axhline(20,color='green',ls=':',label='20 m');ax.axhline(30,color='orange',ls=':',label='30 m')
    ax.set(xlabel='Time (s)',ylabel='Horizontal error (m)',title=f"Largest admitted error: round {top['round']}, {top['input']}")
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'worst_trace.png',dpi=170);plt.close(fig)
    report=['# 第 101–200 轮连续抗欺骗研究结果','',
        '已完成 100 个编号实验。当前结果仍属于合成连续基带研究，不是商用放行或长期可靠导航证明。',
        '',f"新增 {trials} 条不同场景/种子/正常攻击组合的 RF 记录，{comparisons} 次保存位置序列比较；另将启动校准方法在 80 条同输入开发记录上重算，20 个准入规则重复使用这些开发数据，不增加独立攻击试验数。",
        '',f"开发选中方法：`{method}`；规则来自第 {selected['round']} 轮。符合预先设定的开发可用性折中条件的候选数：{selected['feasible_development_rules']}。",'',
        '另追加10个工况、两个新种子正常/攻击配对的40条选型取舍留出，单独列出，不改变100轮主验证。','',
        '## 配对结果','',
        '|阶段|方法|正常输出率|正常≤20m/总历元|攻击输出率|攻击≤20m/总历元|攻击≥30m|攻击≥50m|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for group,data in groups.items():
        names=['raw','short','mvdr','sic','calibrated'] if group=='development' else list(dict.fromkeys(['raw',method,'candidate']))
        for m in names:
            n=data[m+'_normal'];a=data[m+'_attack']
            report.append(f"|{group}|{m}|{100*n['solved']/n['opportunities']:.2f}%|{n['good20']}/{n['opportunities']}|{100*a['solved']/a['opportunities']:.2f}%|{a['good20']}/{a['opportunities']}|{a['error30']}|{a['error50']}|")
    report+=['','`development` 是调参数据，`independent` 为 161–180 轮新种子，`boundary` 为 181–200 轮边界新种子。不同阶段含不同难度与时长，不能把百分比差异解释为相同输入上的提升。',
        '', '![留出可用性与合格位置](results/recovery-two-hundred/holdout.png)','',
        '## 同一观测上的旧/新准入比较','',
        '这里固定本轮选中的信号恢复方法，只改变准入。旧规则为残差≤10m、删星子集位置差≤5m；新规则使用观测质量和形式协方差归一化。',
        '', '|阶段|规则|正常输出/机会|攻击输出/机会|攻击≤20m|攻击≥30m|攻击≥50m|','|---|---|---:|---:|---:|---:|---:|']
    for group,data in old_new.items():
        for policy in ('old_gate','new_gate'):
            n=data[policy+'_normal'];a=data[policy+'_attack']
            report.append(f"|{group}|{policy}|{n['solved']}/{n['opportunities']}|{a['solved']}/{a['opportunities']}|{a['good20']}|{a['error30']}|{a['error50']}|")
    report+=['','## 新种子的选型取舍补充','',
        '主规则优先减少≥50m错误；备选优先减少全部≥30m错误，两者都从开发集选出。补充数据生成前已冻结规则，主验证未改动。这里固定同一输入，直接比较两种完整策略。',
        '', '|方法/策略|正常输出/机会|正常≥30m|攻击输出/机会|攻击≤20m|攻击≥30m|攻击≥50m|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in list(dict.fromkeys([trade_frozen['primary']['method'],trade_frozen['alternate']['method'],'primary','alternate'])):
        n=trade_totals[m+'_normal'];a=trade_totals[m+'_attack']
        report.append(f"|{m}|{n['solved']}/{n['opportunities']}|{n['error30']}|{a['solved']}/{a['opportunities']}|{a['good20']}|{a['error30']}|{a['error50']}|")
    report+=['',f"备选为 `{trade_frozen['alternate']['method']}` + 第{trade_frozen['alternate']['round']}轮准入。只有两套新基础天空几何，跨十工况复用；这是局部证据，不是广泛实测。",'',
        '![两种选型取舍的逐工况结果](results/recovery-two-hundred/tradeoff.png)']
    report+=['','## 冻结规则','', '```json',json.dumps(selected,ensure_ascii=False,indent=2),'```','',
        '开发的正常≥95%、攻击≥50%可用率仅为避免选择退化的折中，不取代正常99%及错误位置不放行的产品要求。通过规则不等于完成真实性认证。新增恢复方法若未被选中，其收益只得到开发集支持，不能据本轮最终留出宣称已经验证。',
        '', '## 可核查的失败','',
        '|轮次|记录|时间/s|候选水平错误/m|使用星数|','|---:|---|---:|---:|---:|']
    for r in sorted(worst,key=lambda r:-r['error_m'])[:10]:
        report.append(f"|{r['round']}|{r['input']}|{r['time_s']:.2f}|{r['error_m']:.2f}|{r['stars']}|")
    report+=['','完整失败历元含归一化残差、分离分数及水平形式标准差，见 `results/recovery-two-hundred/failures.json`。没有因留出失败而事后修改准入规则。','',
        '严重反例：第185轮 seed940185 的 t=0.17s，原始水平误差12.13m，选星并加权重估后变成108.03m。此时仅使用PRN 2/3/5/6/7/8，归一化残差0.642、分离分数0.871、水平形式标准差18.35m均通过门限。新增准入包含重新估计位置，所以不仅可能漏拒，还可能把原本较好的解变坏。形式协方差未覆盖真实偏差，不能用它作为认证保护级。','',
        '![最大已准入错误所在轨迹](results/recovery-two-hundred/worst_trace.png)',
        '', '## 方法、论文和验证','',
        '用户提供的论文已核对公开摘要，并查阅作者公开博士论文；本项目的空间辅助局部副本抵消属于受其启发的实现，尚非完整矢量接收机复现。论文细节、数学公式、假设与对应差异见 [方法说明](RECOVERY_TWO_HUNDRED_METHODS.md)。',
        '', '全部逐轮数据见 [100轮表](results/recovery-two-hundred/round_table.md)。准入轮表展示最终选中恢复方法，各轮完整五方法数据另存对应 `summary.json`。',
        '', '![逐轮结果](results/recovery-two-hundred/rounds.png)','',
        '测试覆盖空间分离副本抵消、未知比特符号、相位校准、正常/攻击同输入、攻击撤离、污染启动以及一致假位置仍能通过准入的反例。实际测试输出：','', '```',
        (out/'tests.txt').read_text(encoding='utf-8').strip(),'```','',
        '汇总程序重新评价全部保存位置，并重算冻结候选位置核对一致；旧准入在同一留出观测上另行计算。`final_source/` 保存源码，`manifest.json` 保存结果哈希。',
        '', '## 局限与下一步','',
        '本轮最长连续记录仅4秒，接收机静止、真实方向和姿态已知、绝大多数组假设前100ms干净。第190轮故意破坏可信启动前提。没有真实台架、在线矢量导航反馈、独立欺骗PLL、长期模板更新或GNSS/IMU闭环。',
        '', '短时解扩能减少未知比特相消，固定相位偏差可由干净启动部分估计，但误差仍可能来自空间估计污染、近同方向不可分、模板漂移及码跟踪偏差。加权残差和分离只检验自洽性，完全一致的假位置仍可能通过。下一步先针对错误选星/加权放大与启动信任建立研究，结合恢复后逐星空间真实性、可信动态模板和独立运动约束；使用新的留出验证可用率与错误率同时改善。',
        '', '## 复现','',
        '在独立工作副本中运行，输出必须为新目录。环境为本机 Python 3.11 与现有 NumPy/SciPy/Matplotlib/pytest；将工作目录设为该副本的 `python_sim_simplified`。下面 PowerShell 命令每条单独执行：','',
        '```powershell',"$env:OPENBLAS_NUM_THREADS='1'","$env:OMP_NUM_THREADS='1'",
        'python audit_recovery_two_hundred.py --output results/recovery-two-hundred --rounds (101..140)',
        'python augment_startup_calibration.py --output results/recovery-two-hundred --rounds (101..140)',
        'python audit_recovery_two_hundred.py --output results/recovery-two-hundred --calibrate',
        'python audit_recovery_two_hundred.py --output results/recovery-two-hundred --select',
        'python audit_recovery_two_hundred.py --output results/recovery-two-hundred --rounds (161..200)',
        'python audit_recovery_tradeoff.py',
        'python -m pytest tests -q > results/recovery-two-hundred/tests.txt',
        'python summarize_recovery_two_hundred.py --output results/recovery-two-hundred','```','',
        '离线运行时间明显长于波形时长；10/20/40ms块处理还需要对应缓冲延迟，未验证实时处理能力。']
    Path('RECOVERY_TWO_HUNDRED_RESEARCH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    source=out/'final_source';source.mkdir(exist_ok=True)
    for p in Path('.').glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
    for name in ('RECOVERY_TWO_HUNDRED_PLAN.md','RECOVERY_TWO_HUNDRED_METHODS.md','RECOVERY_TWO_HUNDRED_RESEARCH.md'):
        (source/name).write_bytes(Path(name).read_bytes())
    (source/'test_continuous_improvement.py').write_bytes(Path('tests/test_continuous_improvement.py').read_bytes())
    manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*')
              if p.is_file() and p.name!='manifest.json'}
    save(out/'manifest.json',dict(rounds=100,sha256=manifest))
    print(json.dumps(dict(groups=groups,old_new=old_new,trials=trials,comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
