"""Recompute archived results, ablations and paired comparisons for rounds 201-400."""
import hashlib,json,platform,shutil
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_recovery_four_hundred import ROOT,KEYS,totals
from audit_recovery_fifty import save
from audit_recovery_two_hundred import score
from carrier_admission import rule_from_record,evaluate


def archive():
    snapshot=ROOT/'final_source';snapshot.mkdir(exist_ok=True)
    for pattern in ('*.py','*.md','dual_frequency_weights.json','requirements.txt'):
        for path in Path('.').glob(pattern):shutil.copyfile(path,snapshot/path.name)
    shutil.copytree('tests',snapshot/'tests',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile('four_hundred_tests_final.log',ROOT/'tests.txt')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(ROOT/'manifest.json',dict(rounds=200,rf_records=240,archived_failure_replays=3,
        same_input_calibration_ablation=1,sha256=manifest))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
    return len(manifest)


def main():
    selected=json.loads((ROOT/'selected.json').read_text());rule=rule_from_record(selected['rule'])
    for n in range(201,401):
        assert (ROOT/f'round{n:03d}'/'summary.json').exists(),n
        frozen=json.loads((ROOT/f'round{n:03d}'/'frozen.json').read_text())
        source=ROOT/'source-v1' if n<=320 else Path('.')
        for p,h in frozen['sha256'].items():assert hashlib.sha256((source/p).read_bytes()).hexdigest()==h,(n,p)
    rows=[];failures=[];paired=[];all_errors={};stats={};replayed=0
    for stage,numbers in (('development',range(201,241)),('independent',range(321,361)),('boundary',range(361,401))):
        stage_rows=[]
        for n in numbers:
            root=ROOT/f'round{n:03d}';archived=json.loads((root/'rows.json').read_text())
            case=json.loads((root/'frozen.json').read_text())['case']
            for path,old in zip(sorted(root.glob('*.npz')),archived):
                with np.load(path) as stored_arrays:z={k:stored_arrays[k] for k in stored_arrays.files}
                methods=['raw','calibrated','previous','alternate'];attack=path.stem.endswith('_1')
                row=dict(round=n,stage=stage,attack=attack,input=str(path))
                for name,r in (('candidate',rule),('no_smoothing',replace(rule,window=1,mature=1)),
                               ('no_spatial',replace(rule,spatial=-np.inf)),
                               ('no_consensus',replace(rule,consensus=np.inf))):
                    result=evaluate(z,selected['method'],r)
                    if name=='candidate' and stage!='development':
                        for key in result:np.testing.assert_allclose(result[key],z['candidate_'+key],atol=1e-8,rtol=0,equal_nan=True)
                        replayed+=1
                    z[name+'_position']=result['position'];methods.append(name)
                for m in methods:
                    result=score(z[m+'_position'],z['truth_position']);row[m]=result
                    squared=np.sum((z[m+'_position'][:,:2]-z['truth_position'][:,:2])**2,axis=1)
                    direct=dict(opportunities=len(squared),solved=int(np.isfinite(z[m+'_position']).all(axis=1).sum()),
                                good20=int((squared<=400).sum()),error30=int((squared>=900).sum()),error50=int((squared>=2500).sum()))
                    assert all(direct[key]==result[key] for key in KEYS),(path,m)
                    if m in old:
                        for key in KEYS:assert result[key]==old[m][key],(path,m,key)
                    err=np.linalg.norm((z[m+'_position']-z['truth_position'])[:,:2],axis=1)
                    all_errors.setdefault((stage,attack,m),[]).append(err)
                active=(z['time_s']>=case['attack_start'])&(z['time_s']<case['attack_end'])
                row['active_window']={m:score(z[m+'_position'][active],z['truth_position'][active]) for m in methods}
                error=np.linalg.norm((z['candidate_position']-z['truth_position'])[:,:2],axis=1)
                for k in np.flatnonzero(error>=30):
                    failures.append(dict(round=n,stage=stage,attack=attack,input=str(path),index=int(k),
                        time_s=float(z['time_s'][k]),error_m=float(error[k]),
                        alternate_error_m=float(np.linalg.norm((z['alternate_position'][k]-z['truth_position'][k])[:2])),
                        raw_error_m=float(np.linalg.norm((z['raw_position'][k]-z['truth_position'][k])[:2]))))
                stage_rows.append(row);rows.append(row)
            print('verified '+str(n),flush=True)
        stats[stage]=totals(stage_rows,methods)
        if stage=='development':
            for flag in ('normal','attack'):
                for key in KEYS:assert stats[stage]['candidate_'+flag][key]==selected[flag][key],(flag,key)
        for attack in (False,True):
            for m in methods:
                errors=np.concatenate(all_errors[stage,attack,m]);finite=np.isfinite(errors)
                result=stats[stage][m+('_attack' if attack else '_normal')]
                result['p95_m']=float(np.percentile(errors[finite],95)) if finite.any() else None
                result['maximum_m']=float(errors[finite].max()) if finite.any() else None
                group=[r for r in stage_rows if r['attack']==attack]
                result['records_with_error30']=sum(r[m]['error30']>0 for r in group)
                result['records']=len(group)
                result['records_all_good20']=sum(r[m]['good20']==r[m]['opportunities'] for r in group)
                result['longest_unavailable_s']=max(r[m]['longest_unavailable_s'] for r in group)
        for r in stage_rows:
            paired.append(dict(round=r['round'],attack=r['attack'],stage=stage,
                good20_delta=r['candidate']['good20']-r['alternate']['good20'],
                error30_delta=r['candidate']['error30']-r['alternate']['error30'],
                solved_delta=r['candidate']['solved']-r['alternate']['solved']))
    failures.sort(key=lambda r:r['error_m'],reverse=True)
    active_stats={stage:totals([dict(attack=r['attack'],**r['active_window']) for r in rows if r['stage']==stage],methods)
                  for stage in stats}
    save(ROOT/'active_window_totals.json',active_stats)
    save(ROOT/'rescored.json',dict(selected=selected,totals=stats,rows=rows))
    save(ROOT/'failures.json',failures);save(ROOT/'paired_deltas.json',paired)
    save(ROOT/'verification.json',dict(rounds=200,rf_records=len(rows),replayed_holdout_records=replayed,
                                     opportunities_are_not_independent_trials=True))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    labels=['Previous','Alternate','Candidate'];colors=['#64748b','#d97706','#087f8c']
    for ax,stage in zip(axes,['independent','boundary']):
        for i,(m,color) in enumerate(zip(['previous','alternate','candidate'],colors)):
            v=stats[stage][m+'_attack'];x=np.array([0,1,2])+i*.22
            ax.bar(x,[100*v['solved']/v['opportunities'],100*v['good20']/v['opportunities'],
                100*v['error30']/v['opportunities']],width=.21,label=labels[i],color=color)
        ax.set_xticks(np.arange(3)+.22,['Available','All-epoch <=20 m','All-epoch >=30 m'])
        ax.set_title(stage.capitalize());ax.set_ylabel('Attack-record epochs (%)');ax.set_ylim(0,105);ax.legend()
    fig.savefig(ROOT/'holdout.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
    group=[r for r in rows if r['attack'] and r['stage']!='development']
    for m,color in zip(['alternate','candidate'],colors[1:]):
        axes[0].plot([r['round'] for r in group],[100*r[m]['good20']/r[m]['opportunities'] for r in group],'.-',label=m,color=color)
        axes[1].plot([r['round'] for r in group],[r[m]['error30'] for r in group],'.-',label=m,color=color)
    axes[0].set_ylabel('All-epoch <=20 m (%)');axes[1].set_ylabel('>=30 m epochs');axes[1].set_xlabel('Frozen holdout round')
    axes[0].legend();fig.savefig(ROOT/'rounds.png',dpi=160);plt.close(fig)
    # Include a failure trace only when a failure was observed; this is a reporting condition.
    if failures:
        worst=next(r for r in failures if r['stage']!='development')
        z=np.load(worst['input']);fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
        for m in ('raw','alternate','candidate'):
            position=evaluate(z,selected['method'],rule)['position'] if m=='candidate' else z[m+'_position']
            e=np.linalg.norm((position-z['truth_position'])[:,:2],axis=1);ax.plot(z['time_s'],e,'.-',markersize=3,label=m)
        ax.axhline(20,color='#087f8c',linestyle=':');ax.axhline(30,color='#c0392b',linestyle=':')
        ax.set(xlabel='Time (s)',ylabel='Horizontal error (m)',title=f'Worst held-out candidate error: round {worst["round"]}')
        ax.legend();fig.savefig(ROOT/'worst_trace.png',dpi=160);plt.close(fig)
    lines=['# 第 201–400 轮研究结果','',
        '结论：有局部精度和可用性收益，但按用户≥30m即失败的要求，新候选未在整体留出中胜出。独立加边界的攻击记录中，≥30m输出合计由284增到295，≥50m由56降到24；不能只看严重错误减少。正常输出率为95.0%和98.4%，也未达到99%目标。','',
        '已完成200个编号实验：40个开发RF工况/种子组合、60个同输入规则比较、20个观测域反例、40个独立RF留出及40个边界RF留出。共240条正常/攻击RF记录、120套不同种子天空几何；规则重复、配对记录和相邻历元不算独立攻击样本。另重放3条旧失败记录，不计入新种子成绩。','',
        '所选方法：`'+selected['method']+'`，开发规则来自第'+str(selected['round'])+'轮。','',
        'RF记录指合成基带试验的观测归档，不是台架实测；全量IQ由种子与源码重生成。冻结接收方法和后处理Rule，部分预注册灵敏度工况改变前端积累窗，因此合并比例不是单一默认硬件配置的现场成功率。','',
        '```json',json.dumps(selected['rule'],indent=2),'```','',
        '只在开发集选型，留出阶段参数冻结。95%正常可用性是开发折中，产品目标99%保持不变。120个开发候选中没有任何一个达到99%正常输出率。最长RF记录8秒，观测域反例10秒；均不代表分钟级、长期或实机导航验证。','',
        '## 同一输入配对结果','',
        '|阶段|方法|正常输出率|正常≤20m/全历元|攻击输出率|攻击≤20m/全历元|攻击≥30m|攻击≥50m|攻击输出P95(m)|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage in stats:
        for m in ('raw','calibrated','previous','alternate','candidate'):
            a=stats[stage][m+'_attack'];n=stats[stage][m+'_normal'];p95=a['p95_m']
            lines.append(f'|{stage}|{m}|{100*n["solved"]/n["opportunities"]:.2f}%|{n["good20"]}/{n["opportunities"]}|{100*a["solved"]/a["opportunities"]:.2f}%|{a["good20"]}/{a["opportunities"]}|{a["error30"]}|{a["error50"]}|{p95:.2f}|' if p95 is not None else f'|{stage}|{m}|无输出|')
    lines+=['','previous为上一轮主策略raw+rule143；alternate为上一轮备选calibrated+rule146。candidate才是本次新增平滑/空间观测链。未输出不能计为成功导航；P95只针对已输出，所以须与全历元比例一起阅读。','',
        '![配对结果](results/recovery-four-hundred/holdout.png)','',
        '## 仅在发射攻击的时间窗统计','',
        '上表按整条攻击记录计分，含撤销攻击后的恢复段。下表严格使用已配置的攻击起止时间，避免把恢复段计入持续受攻击能力。','',
        '|阶段|策略|有解/攻击窗历元|≤20m|≥30m|≥50m|','|---|---|---:|---:|---:|---:|']
    for stage in ('independent','boundary'):
        for m in ('alternate','candidate'):
            a=active_stats[stage][m+'_attack']
            lines.append(f'|{stage}|{m}|{a["solved"]}/{a["opportunities"]}|{a["good20"]}|{a["error30"]}|{a["error50"]}|')
    lines+=['',
        '## 留出消融','',
        '消融均使用冻结候选的其余参数，不按留出表现重新选择。','',
        '|阶段|方法|攻击有解/全部|攻击≤20m|攻击≥30m|攻击≥50m|',
        '|---|---|---:|---:|---:|---:|']
    for stage in ('independent','boundary'):
        for m in ('candidate','no_smoothing','no_spatial','no_consensus'):
            a=stats[stage][m+'_attack'];lines.append(f'|{stage}|{m}|{a["solved"]}/{a["opportunities"]}|{a["good20"]}|{a["error30"]}|{a["error50"]}|')
    lines+=['','no_smoothing同时取消平滑和弧成熟等待，是时间处理整体消融；no_spatial取消空间分数限制；no_consensus取消加权/无权解差限制。若冻结候选本来没启用该项，其消融自然相同。','',
        '空间门限没有稳定收益：边界留出中，取消它时≥30m输出为168，开启后为211，全历元≤20m数量均为7889。363轮t=1.79s，同一输入在无空间门限时误差19.67m，开启后52.92m；PRN8的空间分数约0.323，被排除后其余7星形成了更偏的位置。该记录有7个历元从≤20m变为≥30m。见spatial_gate_counterexample.json；不能把新增质量门限默认解释成可靠性提升。','',
        '## 按记录而非历元观察失败','',
        '|阶段|方法|攻击中出现≥30m的记录|最长连续无输出(s)|最大输出误差(m)|',
        '|---|---|---:|---:|---:|']
    for stage in ('independent','boundary'):
        for m in ('alternate','candidate'):
            a=stats[stage][m+'_attack'];lines.append(f'|{stage}|{m}|{a["records_with_error30"]}/{a["records"]}|{a["longest_unavailable_s"]:.2f}|{a["maximum_m"]:.2f}|')
    lines+=['','![逐轮收益及错误](results/recovery-four-hundred/rounds.png)','',
        '## 旧失败重放：改善与归因','',
        '三个旧种子只用于归因，未加入新种子通过率。原始raw观测和位置逐数组重现。','',
        '|旧编号|策略|有解/全部|≤20m|≥30m|输出P95(m)|','|---|---|---:|---:|---:|---:|']
    replay=json.loads((ROOT/'archived-failure-replay'/'comparison.json').read_text())
    for v in replay['rows']:
        for m in ('previous','alternate','candidate'):
            a=v[m];lines.append(f'|{v["round"]}|{m}|{a["solved"]}/{a["opportunities"]}|{a["good20"]}|{a["error30"]}|{a["p95_m"]:.2f}|')
    lines+=['','旧185轮t=0.17s：previous水平误差108.03m，alternate约8.48m，candidate约2.54m。大部分修复已来自上一轮备选接收链，本轮平滑带来进一步改进，不能把全部改善都归功于新增算法。','',
        '## 冷启动和弱欺骗的失败机理','',
        '新350轮真实通道相位失配为0°，受骗启动估计的相对相位却约为0°、−108.8°、171.5°、35.3°。校准前提被破坏后，空间分数仍能通过：83/84个历元输出，0个≤20m，67个≥30m、8个≥50m，最大约54.55m。旧190轮改善不能推广为冷启动已解决。见cold_start_diagnostic.json及方法文档中的混合信号公式。','',
        '新338轮弱欺骗功率为+3dB：原始协方差特征值比始终小于3，空间证据计算退化为预期方向匹配相关；该方向仍可能含欺骗投影，因此高空间分数不能证明锁定真码。短时解扩已在进行零陷，说明恢复链与这个空间证据的可观测前提不同。见weak_source_diagnostic.json。','',
        '350轮追加一次同IQ消融：保持短时解扩，取消启动相位校准，原位置84/84个≤20m；再使用冻结后处理规则后，仍为84/84、P95约2.78m、最大约3.28m。它支持“这条记录的错误校准造成失败”的归因；此工况真实相位失配为0，不能推出取消校准能处理真实通道失配。本次候选保持原样，未按这个事后样例重新选型；见cold-calibration-ablation/comparison.json。','',
        '## 8秒与运动边界','',
        '|编号|策略|有解/全部|≤20m|≥30m|≥50m|最长无输出(s)|','|---|---|---:|---:|---:|---:|---:|']
    for v in rows:
        if v['attack'] and v['round'] in (369,370,379,380,381,382,383,384,399,400):
            for m in ('alternate','candidate'):
                a=v[m];lines.append(f'|{v["round"]}|{m}|{a["solved"]}/{a["opportunities"]}|{a["good20"]}|{a["error30"]}|{a["error50"]}|{a["longest_unavailable_s"]:.2f}|')
    lines+=['','369/379分别为2/4秒15m/s运动；370/380为2/4秒20m/s渐变拖曳。381–384依次为8秒名义、2m/s运动、10m/s渐变拖曳、无真实信号；399为2秒30m/s运动，400为2秒受骗启动且无真实信号。无真信号场景初期短暂好坐标可能来自启动跟踪状态与20ms观测窗口，不能当作真实信号已恢复。','',
        '381轮仅307/784个历元输出，全部在20m内。所有历元均有7或8星，形式水平标准差没有超限；残差和删星一致性门限分别触发476和466次（重叠计数），总共拒绝477次，其中475次未经准入的校准位置当时已≤20m。这是准入保守性/模型不匹配的明确代价，不能事后知道真值达标就放松在线门限。见long_record_admission_diagnostic.json。','',
        '382轮为2m/s运动：未经准入的校准解在前4秒没有≥30m，后段逐渐失稳；PRN4在5秒后的相对公共钟差码偏差中位数约118m，C/N0约35.4dB-Hz、锁定指标约0.637，仍有约67%的后段历元通过初步选星。原始协方差方法该记录783/784个≤20m。需要检查恢复/跟踪链的偏差来源，不能把它都当成可由平滑消除的白噪声。见moving_long_record_diagnostic.json和moving_long_channel_diagnostic.json。','',
        '## 数值反例','']
    if failures:
        v=worst;lines+=[f'留出最大已输出候选水平误差：第{v["round"]}轮，t={v["time_s"]:.2f}s，{v["error_m"]:.2f}m。同历元raw误差{v["raw_error_m"]:.2f}m，上一轮备选误差{v["alternate_error_m"]:.2f}m。开发集最大为{failures[0]["error_m"]:.2f}m；见failures.json完整逐历元记录。','',
            '![失败轨迹](results/recovery-four-hundred/worst_trace.png)','']
    lines+=['|编号|观测域工况|伪距RMS(m)|最大伪距误差(m)|≥30m观测数|','|---|---|---:|---:|---:|']
    for n in range(301,321):
        v=json.loads((ROOT/f'round{n:03d}'/'summary.json').read_text())
        lines.append(f'|{n}|{v["name"]}|{v["rms_m"]:.2f}|{v["max_m"]:.2f}|{v["error30"]}|')
    lines+=['','观测域统计单位是每星伪距，不是位置误差，不与RF水平定位表合并。所有20条反例通过前缀因果性检查。coherent_false_step/slow_pull在这组观测域输入中给所有星相同公共距离变化，能被位置解中的钟差吸收，不能用其伪距误差声称水平定位失败。真正一致的200m假位置由单元负例验证；RF里的渐变假位置由各星不同几何伪距驱动。码载波一致的伪变不会被Hatch递推本身识别。','',
        '## 可复现性与范围','',
        '在python_sim_simplified目录运行，设置OPENBLAS_NUM_THREADS=1和OMP_NUM_THREADS=1。依次执行：','',
        '```powershell',
        '# 在另一个结果目录复现时，先修改audit_recovery_four_hundred.py中的ROOT。',
        'python -c "from audit_recovery_four_hundred import ROOT,FILES; import shutil; p=ROOT/\'source-v1\'; p.mkdir(parents=True); [shutil.copyfile(f,p/f) for f in FILES]"',
        'python audit_recovery_four_hundred.py --rounds (201..240)',
        'python audit_recovery_four_hundred.py --rounds (241..300)',
        'python audit_recovery_four_hundred.py --select',
        'python audit_recovery_four_hundred.py --rounds (301..320)',
        'python audit_recovery_four_hundred.py --rounds (321..400)',
        'python diagnose_recovery_four_hundred.py --prepare',
        'python diagnose_recovery_four_hundred.py',
        'python diagnose_recovery_four_hundred.py --calibration-ablation',
        'python -m pytest tests -q > four_hundred_tests_final.log',
        'python summarize_recovery_four_hundred.py','```','',
        '复现须使用一个尚不存在的结果目录（修改脚本ROOT或使用干净检出），不要覆盖已归档实验。逐轮frozen.json、NPZ、rows和summary共同记录输入与输出。最终汇总重算全部保存位置指标、重放160条冻结留出准入并逐数组核对，生成消融与逐历元失败清单。','',
        '旧失败重放依赖results/recovery-two-hundred中相应历史归档；主试验生成不依赖旧数据。汇总保存源码、测试输出和SHA256清单。当前全量266项测试通过，1条既有无噪声相位测试的除零警告保持原状。','',
        '完整数学模型和假设见RECOVERY_FOUR_HUNDRED_METHODS.md；最初编号/选择约定见RECOVERY_FOUR_HUNDRED_PLAN.md。仍缺同步RF台架数据、时变姿态/多径/温漂/ADC和可信冷启动认证；没有进行商业放行或推送。']
    lines+=['','下一步优先研究显式真/伪双分量的启动校准、持续更新的解扩/跟踪状态，以及符合残余偏差的准入模型；方法文档给出两天线自由双分量拟合的不可辨识反例。这些是后续待验证方向，本轮没有将其包装成已完成商用能力。']
    lines+=['','冻结重放前修复了一个实验配置读取问题：既有save函数把无限阈值写成JSON null，新增明确的阈值反序列化来恢复“禁用该门限”的数学含义。原始开发计算未改变，开发与反例源码保存在source-v1/，后续源码在final_source/；汇总分别核对相应SHA256并重新验证所选开发总数。null不是缺字段默认回退。']
    Path('RECOVERY_FOUR_HUNDRED_RESEARCH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(ROOT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform()))
    print('archived hashes '+str(archive()),flush=True)
    print(json.dumps(dict(selected=selected,totals=stats)),flush=True)


if __name__=='__main__':main()
