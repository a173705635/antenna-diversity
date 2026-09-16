"""Build the report from frozen evidence; no experiments or model selection."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from single_source_twenty import ROOT,read,save,digest

LABELS=dict(normal='无攻击',drift='线性通道漂移',weak='启动即低信号',absent='真实信号消失',
            return_='真实信号恢复',near='一星与源重合',cluster='三星与源近方向',drag='连续拖偏',release='攻击撤除',warm_weak='初始化后转弱',nominal='常规持续单源攻击')
LABELS['return']=LABELS.pop('return_')
METRICS=('opportunities','solved','good20','error30','error50')


def totals(rows):
    total={key:sum(r['score'][key] for r in rows) for key in METRICS}
    errors=[e for r in rows for e in r['score']['errors_m'] if e is not None]
    total.update(records=len(rows),not_ready=sum(not r['operationally_ready'] for r in rows),
                 max_error_m=max(errors) if errors else None,
                 longest_unavailable_s=max(r['score']['longest_unavailable_s'] for r in rows) if rows else None)
    return total


def table(rows):
    lines=['|配置|录波数|0.3s前未完成初始化并发布位置|全部机会|输出|≤20m|≥30m|≥50m|最长中断s|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for subset in ('two','three','four'):
        t=totals([r for r in rows if r['subset']==subset])
        lines.append(f"|{subset}|{t['records']}|{t['not_ready']}|{t['opportunities']}|{t['solved']}|{t['good20']}|{t['error30']}|{t['error50']}|{t['longest_unavailable_s']:.2f}|")
    return '\n'.join(lines)


def finalize():
    assert '375 passed, 1 xfailed, 1 warning' in Path('single_source_all_tests.log').read_text(encoding='utf-8')
    assert '376 passed, 1 xfailed, 1 warning' in Path('single_source_corrected_tests.log').read_text(encoding='utf-8')
    assert read(ROOT/'audit.json')['completed_rounds']==20
    assert read(ROOT/'rf_replay.json')['full_replay_matches']
    plan=read(ROOT/'freeze.json')
    rows=[r for subset in plan['subsets'] for r in read(ROOT/f'holdout_{subset}.json')]
    assert len(rows)==len(plan['cases'])*3
    extra=[r for subset in plan['subsets'] for r in read(ROOT/f'warm_weak_{subset}.json')]
    assert len(extra)==9
    nominal=[r for subset in plan['subsets'] for r in read(ROOT/f'nominal_{subset}.json')]
    assert len(nominal)==9
    corrected=[r for subset in plan['subsets'] for r in read(ROOT/f'corrective_{subset}.json')]
    assert len(corrected)==108
    corrected_sha=read(ROOT/'post_freeze_correction.json')['corrected_sha256']
    assert digest(ROOT/'audit_corrected.py')==corrected_sha==digest(Path('single_source_receiver_fixed.py'))
    baseline=[r for subset in plan['subsets'] for r in read(ROOT/f'holdout_baseline_{subset}.json')]
    assert len(baseline)==9
    for r in baseline:r['case']=dict(seed=114001,kind=r['kind'],duration=2.)
    assert digest(ROOT/'selected.py')==plan['selected_sha256']
    for name,sha in plan['dependencies'].items():assert digest(Path(name))==sha
    diagnostics=[];seen=set()
    for r in rows+extra+nominal+baseline+corrected:
        case=r['case'];folder=ROOT/'inputs'/f"{case['seed']}_{case['kind']}_{case['duration']}"
        meta=read(folder/'metadata.json')
        if folder not in seen:
            assert digest(folder/'samples.npy')==meta['samples_sha256']
            assert digest(folder/'observations.npz')==meta['observations_sha256'];seen.add(folder)
        assert r['score']['input_sha256']==meta['samples_sha256']
        assert r['score']['source_sha256'] in (plan['selected_sha256'],digest(ROOT/'baseline.py'),corrected_sha)
        cache=ROOT/'evaluations'/r['identity']
        with np.load(cache/'tracking.npz') as z,np.load(cache/'navigation.npz') as n:
            elapsed=z['available_time'];mask=z['time']>=meta['case']['attack_start']+.02
            error=np.linalg.norm(n['position'][:,:2]-np.column_stack((meta['case']['velocity_east']*elapsed,np.zeros(len(elapsed)))),axis=1)
            candidate_error=np.linalg.norm(n['candidate_position'][:,:2]-np.column_stack((meta['case']['velocity_east']*elapsed,np.zeros(len(elapsed)))),axis=1)
            pre_published=int((np.isfinite(error)&(elapsed<meta['case']['attack_start'])).sum())
            r['operationally_ready']=r['score']['initialized_before_attack'] and pre_published>0
            t=dict(case=case,subset=r['subset'],initialized_before_attack=r['score']['initialized_before_attack'],
                   source_sha256=r['score']['source_sha256'],pre_attack_published=pre_published,operationally_ready=r['operationally_ready'],
                   initial_prns=int(len(z['startup_prns'])),median_selected_prns=float(np.median(n['selected'].sum(axis=1))),
                   median_retained_gain=float(np.median(z['gain'])),fully_post_attack_opportunities=int(mask.sum()),
                   projected_fraction=float(np.mean(z['ratio']>=3.)),
                   candidate_finite_post=int((np.isfinite(candidate_error)&mask).sum()),
                   candidate_error30_post=int(((candidate_error>=30)&mask).sum()),
                   fully_post_attack_solved=int((np.isfinite(error)&mask).sum()),fully_post_attack_good20=int(((error<=20)&mask).sum()),
                   fully_post_attack_error30=int(((error>=30)&mask).sum()),fully_post_attack_error50=int(((error>=50)&mask).sum()))
            if 'calibration_rate' in z.files:t['estimated_phase_rate_rad_s']=z['calibration_rate'].tolist()
            if case['kind']=='return':
                recovered=elapsed[(elapsed>=.8)&np.isfinite(error)]
                t['first_recovery_delay_s']=float(recovered[0]-.8) if len(recovered) else None
            diagnostics.append(t)
    save(ROOT/'holdout_diagnostics.json',diagnostics)
    by_subset={s:totals([r for r in rows if r['subset']==s]) for s in plan['subsets']}
    warm={s:totals([r for r in rows if r['subset']==s and r['operationally_ready']]) for s in plan['subsets']}
    summary=dict(selected_round=20,holdout=by_subset,initialized_before_attack_only=warm,
                 supplement={s:totals([r for r in extra if r['subset']==s]) for s in plan['subsets']},
                 nominal_control={s:totals([r for r in nominal if r['subset']==s]) for s in plan['subsets']},
                 corrective_replay={s:totals([r for r in corrected if r['subset']==s]) for s in plan['subsets']},
                 corrective_ready_only={s:totals([r for r in corrected if r['subset']==s and r['operationally_ready']]) for s in plan['subsets']},
                 frozen_candidate_withdrawn=True,current_research_model='single_source_receiver_fixed.py',
                 unique_final_rf_records_verified=len(seen),commercial_goal_met=False)
    save(ROOT/'summary.json',summary)

    ledger=['# 20轮问题、改动、前后结果与取舍','',
            '计数为20轮完整问题闭环，不是20次成功；种子、同数据天线子集、内部修错、补充验证均不另计轮。', '',
            '|轮次/父版本|问题与假设|输出 / ≤20m / ≥30m / ≥50m：前→后|决定|',
            '|---|---|---|---|']
    for number in range(1,21):
        f=ROOT/f'round{number:03d}';p=read(f/'proposal.json');c=read(f/'conclusion.json');result=read(f/'results.json')
        compact=lambda side:'/'.join(str(result['totals'][side][k]) for k in ('solved','good20','error30','error50'))
        ledger.append(f"|[{number}](results/single-source-twenty/round{number:03d}/proposal.json) / {p['parent']}|{p['problem']}；{p['hypothesis']}|{compact('before')} → {compact('after')}，分母{result['totals']['after']['opportunities']}|{c['decision']}：{c['evidence']}|")
    Path('SINGLE_SOURCE_TWENTY_LEDGER.md').write_text('\n'.join(ledger)+'\n',encoding='utf-8')

    report=['# 单源正常初始化后：20轮研究结果','',
        '**20轮已完成；末次审计另发现并修复了状态交接缺陷。三天线保底、四天线商用可用和长期持续导航仍未证实。** 第20轮冻结候选保留作原始证据，已撤销其使用建议；当前后续研究入口为 `single_source_receiver_fixed.py`（归档 `audit_corrected.py`）。', '',
        '本批是同一连续RF接收机的顺序改进：标定相对相位漂移、实际捕获跟踪和重捕、码载波融合、观测一致性、来源恢复、短时载波桥接。20轮中的否定结果见[逐轮台账](SINGLE_SOURCE_TWENTY_LEDGER.md)，数学公式、模型边界及复现见[方法](SINGLE_SOURCE_TWENTY_METHODS.md)。','',
        '## 当前审计修正版：同录波回放','',
        '卫星集合从6星切至另一组5星、前后仅3颗公共星时，冻结版跳过了载波分支内部的状态检查，未经一致性检验的码解进入last，下一历元被当成载波传播起点。观测域构造反例在0.53–0.55秒放行约1公里偏移；这是状态交接反例，不能冒充已生成RF的攻击结果。', '',
        '修正版将判据放到所有位置状态更新之前，并要求来源恢复时的码重新锚定也通过解分离。回归测试先在旧实现失败、修后通过。下表为同一36份RF观测的108次**修后回放**，包括主验证及两组补充；它不再是未参与修正的独立留出。原RF前端完全相同，复用跟踪缓存，没有重新生成录波或按真值修改阈值。','',
        table(corrected),'',
        '修正版常规持续单源攻击补充对照：','',table([r for r in corrected if r['case']['kind']=='nominal']),'',
        '修正版正常初始化后转弱：','',table([r for r in corrected if r['case']['kind']=='warm_weak']),'',
        '该组有1份三天线记录在攻击前未就绪；只统计攻击前完成初始化并发布过位置的记录如下。三天线为18/342机会≤20m，四天线167/513。未就绪记录仍保留在上表，但不能把它攻击后的7次输出算作正常初始化后的成功。','',
        table([r for r in corrected if r['case']['kind']=='warm_weak' and r['operationally_ready']]),'',
        '修正版是供下一步研究的代码，尚需新的独立验证和真实台架数据；不能用修后回放无大误差替代商用完整性证明。','',
        '## 冻结后的主验证','',
        '以下各表对应发现状态缺陷之前冻结的第20轮原版，不与上面的审计修正版混算。开发八场景原版2139/2736个机会有输出，均≤20m；独立结果如下。','',
        '3个独立种子，9类2秒录波及3份4秒延长记录，共30份四通道录波、90次二/三/四子集评估。以下保留所有尝试；初始化未完成不能解释成已完成正常初始化的抗欺骗成绩。时间历元相互相关，不是独立统计样本。','',table(rows),'',
        '主表保留全部尝试。条件统计进一步要求：初始化标记早于0.3秒，且此前实际发布过有限位置；仅有稳定候选PVT标记但从未发布的不算操作就绪。原始score仍保留原标记，修正口径只用于此处审计汇总，不修改模型。无攻击对照也按0.3秒参考时刻列状态。','',
        '|配置|符合初始化条件的记录|全部机会|输出|≤20m|≥30m|≥50m|','|---|---:|---:|---:|---:|---:|---:|']
    for s,t in warm.items():report.append(f"|{s}|{t['records']}|{t['opportunities']}|{t['solved']}|{t['good20']}|{t['error30']}|{t['error50']}|")
    report += ['', '2秒场景分别汇总如下。每个单元格为 **输出 / ≤20m / ≥30m / ≥50m；未按时初始化记录数**，每配置每场景总分母513。','',
               '|场景|二天线|三天线|四天线|','|---|---|---|---|']
    for kind in LABELS:
        group=[r for r in rows if r['case']['kind']==kind and r['case']['duration']==2.]
        if not group:continue
        fields=[]
        for s in plan['subsets']:
            t=totals([r for r in group if r['subset']==s]);fields.append('/'.join(str(t[k]) for k in ('solved','good20','error30','error50'))+f"；{t['not_ready']}")
        report.append('|'+LABELS[kind]+'|'+'|'.join(fields)+'|')
    report += ['', '## 正常初始化后转为低信号：补充验证','',
               '原weak场景从启动即35 dB-Hz，会产生启动失败，不能单独代表用户要求。补充协议在查看其输出前冻结，采用114101/114113/114137，先45 dB-Hz，0.3秒攻击接入时降到35 dB-Hz。模型仍为同一第20轮，无调参。以下9次评估与主验证分开。','',table(extra),'',
               '## 常规持续单源攻击：补充对照','',
               '为将基本持续攻击与漂移、近方向、撤除等压力条件分开，另预注册114201/114213/114237：45 dB-Hz、源夹角60°、100m东偏、真实信号持续、2秒录波。模型保持冻结；该补充与原主协议分列，不作为调参数据。','',table(nominal),'',
               '## 4秒连续记录','',table([r for r in rows if r['case']['duration']==4.]),'',
               '## 与本批初始基线的同录波对比','',
               '只比较预注册114001的无攻击、漂移、低信号三类。不能将这9次基线结果外推成全部90次留出对照。','',
               '|配置|版本|机会|输出|≤20m|≥30m|≥50m|','|---|---|---:|---:|---:|---:|---:|']
    for s in plan['subsets']:
        for label,rr in [('初始基线',[r for r in baseline if r['subset']==s]),('第20轮',[r for r in rows if r['subset']==s and r['case']['seed']==114001 and r['case']['kind'] in ('normal','drift','weak')])]:
            t=totals(rr);report.append('|'+s+'|'+label+'|'+'|'.join(str(t[k]) for k in METRICS)+'|')
    report += ['', '## 结论边界与下一步','',
        '- **消除某些偏差有收益，但不能等同于解决单源攻击。** 开发漂移案例的初始基线有203个≥30m、135个≥50m历元，第1轮线性相位估计后该组消失。独立114001漂移同录波，三/四天线≥30m从82/41降到0/0；但四天线≤20m从106降到101。大误差减少不等于20m目标达标。',
        '- **三天线最低目标尚未证实。** 弱信号、近方向、观测冗余和恢复条件会限制输出；四天线也必须按上表的错误及缺失评判，不能只看已输出的成功率。114001三星近源案例中，三/四天线完全攻击后仍有111/104个有限内部候选位置，其中90/74个≥30m。暂停输出确实阻止了这些错误，但并没有恢复导航。',
        '- **物理损失与策略约束须区分。** 三星近源时通常只剩4–5颗有效星，而当前一致码校正要求6颗，算法会停止发布。4–5颗星并不在数学上一概无解；上述内部候选错误又说明不能简单放宽星数就宣布解决。',
        '- **二天线问题不全是捕获不到。** 114001无攻击案例初始捕获8颗PRN，但全记录有效观测中位数仅3、保留增益中位数约0.103，未在参考时刻完成初始化；近方向中位有效观测2、增益约0.056。公共源投影、标定和准入共同影响结果，不能仅凭数量下结论。',
        '- **真实信号消失不等于仍能导航。** 开头少量发布含缓冲/过渡观测；完全攻击后20ms窗口统计见诊断文件。没有独立惯性/外部约束时，本批不会用停住的位置伪造持续导航。',
        '- **恢复条件和误差界仍是研究假设。** 6星空间支持、解分离、载波方差均不构成经验证的来源认证或完整性上界；即使某组没有≥30m，也不能宣称客户位置不会被带偏。',
        '- 下一步应先针对冻结留出中实际失败逐条复现，区分启动、空间损失、漂移外推、码/载波误锁和恢复拒绝；同时加入真实同步RF台架数据。当前最长4秒，尚未验证分钟级导航、手机手持转动、车内反射或商用实时性。','',
        '## 验证与归档','',
        '原冻结版本全套测试375 passed、1 xfailed；审计修正后376 passed、1 xfailed，均保留原有1项C/N₀除零警告。20轮196份去重导航结果重演一致；178组前后配对输入一致，源码均有实际AST变化；之前1296个归档文件哈希未变。额外原始RF回放及0.9/1.3秒前缀检查覆盖中断和重捕。','',
        '文件：[冻结清单](results/single-source-twenty/freeze.json)、[主汇总](results/single-source-twenty/summary.json)、[逐录波诊断](results/single-source-twenty/holdout_diagnostics.json)、[审计](results/single-source-twenty/audit.json)、[RF因果重演](results/single-source-twenty/rf_replay.json)、[原测试日志](single_source_all_tests.log)、[修后测试](single_source_corrected_tests.log)、[修正声明](results/single-source-twenty/post_freeze_correction.json)、[反例](results/single-source-twenty/state_handoff_probe.json)。','',
        '![同录波子集的合格输出比例](results/single-source-twenty/holdout.png)']
    Path('SINGLE_SOURCE_TWENTY_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    kinds=['normal','drift','weak','absent','return','near','cluster','drag','release']
    fig,ax=plt.subplots(figsize=(11,4.5))
    for j,s in enumerate(plan['subsets']):
        values=[totals([r for r in rows if r['subset']==s and r['case']['kind']==k and r['case']['duration']==2.]) for k in kinds]
        ax.bar(np.arange(len(kinds))+(j-1)*.25,[100*t['good20']/t['opportunities'] for t in values],width=.25,label=s)
    ax.set_xticks(np.arange(len(kinds)),kinds);ax.set_ylabel('Outputs within 20 m / all opportunities (%)')
    ax.set_ylim(0,105);ax.legend();ax.set_title('Frozen holdout: missing and failed initialization remain in denominator')
    fig.tight_layout();fig.savefig(ROOT/'holdout.png',dpi=160);plt.close(fig)
    status=dict(requested_rounds=20,completed_rounds=list(range(1,21)),selected_round=20,
                validation_complete=True,holdout_rows=len(rows),supplement_rows=len(extra)+len(nominal),
                tests_passed=376,tests_xfailed=1,corrective_replay_rows=len(corrected),
                current_research_model='single_source_receiver_fixed.py',frozen_candidate_withdrawn=True,commercial_goal_met=False)
    save(ROOT/'status.json',status);save(ROOT/'progress.json',status)
    archive=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json']
    artifacts=list(Path('.').glob('single_source*.py'))+list(Path('.').glob('SINGLE_SOURCE_TWENTY_*.md'))
    artifacts += [Path('tests/test_single_source_twenty.py'),Path('tests/test_single_source_state_handoff.py'),Path('single_source_all_tests.log'),Path('single_source_corrected_tests.log'),Path('README.md')]
    save(ROOT/'manifest.json',dict(sha256={str(p.relative_to(ROOT)):digest(p) for p in archive},
          artifacts_sha256={str(p):digest(p) for p in artifacts}))
    print(summary,flush=True)


if __name__=='__main__':finalize()
