"""Audit paired results and explain normal-output loss; no algorithm edits."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from satellite_count_study import ROOT,OLD
from single_source_twenty import read,save,digest,load


def build():
    assert '378 passed, 1 xfailed, 1 warning' in Path('satellite_count_tests.log').read_text(encoding='utf-8')
    plan=read(ROOT/'plan.json');rows=[];records=set();replays=0
    for seed in plan['seeds']:
        for kind in plan['kinds']:
            for subset in plan['subsets']:
                row=read(ROOT/f'{seed}_{kind}_{subset}.json');metrics={}
                for label,base,identity in (('before',OLD,row['before_id']),('after',ROOT,row['after_id'])):
                    cache=base/'evaluations'/identity
                    with np.load(cache/'tracking.npz') as raw:z={k:raw[k] for k in raw.files}
                    with np.load(cache/'navigation.npz') as n:
                        pos=n['position'].copy();valid=n['selected'].copy()
                    post=z['available_time']>=.3
                    metrics[label]=dict(median_valid=float(np.median(valid[post].sum(1))),
                         min_valid=int(valid[post].sum(1).min()),
                         max_valid=int(valid[post].sum(1).max()),
                         projection_fraction=float(np.mean(z['ratio'][post]>=3)),
                         gain_median=float(np.median(z['gain'][post])),
                         first_three_gain_median=float(np.median(z['gain'][post,:3])),
                         ready=row[label]['initialized_before_attack'] and bool(np.isfinite(pos[~post]).all(1).any()))
                    if label=='after':
                        actual=load(ROOT/'receiver.py').navigate(z)
                        np.testing.assert_allclose(actual['position'],pos,rtol=0,atol=1e-9,equal_nan=True);replays+=1
                        assert row[label]['source_sha256']==plan['source_sha256']
                row['diagnostics']=metrics;rows.append(row)
            folder=ROOT/'inputs'/f'{seed}_{kind}_24';meta=read(folder/'metadata.json')
            assert digest(folder/'samples.npy')==meta['samples_sha256']
            assert digest(folder/'observations.npz')==meta['observations_sha256'];records.add(str(folder))
    save(ROOT/'comparison.json',rows)
    lines=['# 24星扩展及无攻击中断排查','',
        '24星实验沿用审计修正版，未修改空间抑制、观测准入、定位和恢复门限。3个已有种子×3类场景×三/四天线，共18次24星连续RF运行；与同源8星归档比较。属于受控扩展，不是独立留出或商用验证。','',
        '## 8星与24星结果','',
        '每行由3份2秒录波合计，评估分母513。≤20m计数的分母包含暂停输出；攻击0.3秒接入，沿用20ms可用时标，包含少量缓冲过渡。', '',
        '|场景|天线|8星：输出 / ≤20m / ≥30m|24星：输出 / ≤20m / ≥30m|24星最大输出误差m|24星未就绪录波|',
        '|---|---|---:|---:|---:|---:|']
    labels=dict(normal='无攻击',near='一星与源重合',cluster='三星近源')
    totals=[]
    for kind in plan['kinds']:
        for subset in plan['subsets']:
            group=[r for r in rows if r['kind']==kind and r['subset']==subset]
            sides={s:{k:sum(r[s][k] for r in group) for k in ('opportunities','solved','good20','error30','error50')} for s in ('before','after')}
            errors=[v for r in group for v in r['after']['errors_m'] if v is not None]
            maximum=max(errors) if errors else None
            maximum_text='无输出' if maximum is None else f'{maximum:.2f}'
            compact=lambda s:'/'.join(str(sides[s][k]) for k in ('solved','good20','error30'))
            not_ready=sum(not r['diagnostics']['after']['ready'] for r in group)
            lines.append(f"|{labels[kind]}|{subset}|{compact('before')}|{compact('after')}|{maximum_text}|{not_ready}|")
            totals.append(dict(kind=kind,subset=subset,**sides,max_error=maximum,not_ready=not_ready))
    lines += ['', '剩余有效观测数（每份录波攻击后中位数，按114001/114013/114037顺序）：','',
              '|场景|天线|8星|24星|','|---|---|---|---|']
    for kind in plan['kinds']:
        for subset in plan['subsets']:
            group=[r for r in rows if r['kind']==kind and r['subset']==subset]
            fields=[' / '.join(str(r['diagnostics'][s]['median_valid']) for r in group) for s in ('before','after')]
            lines.append('|'+labels[kind]+'|'+subset+'|'+'|'.join(fields)+'|')
    lines += ['', '## 为什么无攻击仍会中断','',
        '这里区分“没有发布位置”和“位置偏差大”。上一批修正版三天线无攻击共484/513合格，29个缺失全部来自114001；四天线513/513。进一步检查发现，三天线首次中断时内部加权码解误差已达54.04m，因此不能简单放宽最终准入门限。','',
        '同录波逐历元诊断与原位置输出完全一致：29个缺失中14个是观测/状态条件不满足直接拒绝，15个是候选位置已有解但仍等待来源恢复确认。首次直接拒绝只有4颗有效星；码一致性要求6星，载波检查要求至少5颗公共星，该历元两条条件都不满足。','',
        '更上游的问题是无攻击时仍持续实施空间投影。该三天线记录评估时段的投影启用比例100%，投影保留增益中位数约0.187。这个指标描述保留的空间能量和理想白噪声下相对信噪比，不是最终归一化后幅度只剩18.7%；归一化补偿幅度的同时会放大噪声，对应约7.3dB信噪比损失。跟踪/有效观测减少后，再触发定位拒绝与恢复等待。','',
        '当前触发比值计算自尚未扣除真实卫星的相关矩阵：', '',
        r'\[q=\lambda_{\max}(ZZ^H)/\operatorname{mean}(\lambda_{1:M-1}),\quad q\ge3\Rightarrow\text{投影抑制}.\]', '',
        r'即使没有欺骗且忽略噪声，\(R_0=\sum_i P_i a_i a_i^H\)也不必各向同性。用该案例的理想真实卫星导向和等功率计算，三/四天线的比值已分别为8.94/11.67，超过3。大特征值本身不是欺骗存在的充分证据。', '',
        '这是空间抑制触发机制的缺陷，叠加观测损失与恢复延迟；不能将其概括成“正常GNSS本来就无法定位”，也不能全归咎于最终门限过严。','',
        '## 同录波关闭投影消融','',
        '只在已知无攻击114001中关闭投影，保留标定、捕获跟踪和导航门限，三/四天线均恢复171/171输出且全部≤20m。三天线与原142/171相比消除了29次缺失。这支持空间投影是该案例中断的上游原因。该消融不能直接用于攻击场景，正式修复应改善无攻击/单源假设区分及空间处理触发条件。','',
        '以114001三星近源为例，扩星后前三颗星保留增益中位数仍约1%，因此它们仍基本被舍弃；但三/四天线剩余有效观测中位数由4/4增加到17/18，支持“舍弃近源星、靠其他方向卫星继续导航”的解释。收益并非把近源真实星强行恢复。','',
        '## 比较的边界','',
        '- 原8颗PRN的方向、距离、多普勒、比特、噪声及通道误差保持不变，只追加16颗；3个近源方向也不变。回归测试专门避免按24星重新分配多普勒而混淆变量。',
        '- 24星时一个物理欺骗源同样伪造全部24个PRN，每PRN真假功率比仍10dB；额外星不是未受攻击的真值辅助。',
        '- 接收机按实际保留增益、锁定和C/N₀排除不合格观测，没有按生成器真值手动删除近源星。',
        '- 24个同频C/A PRN是等效研究输入，并非手机多星座、多频段、人体遮挡或车内实测。24颗可见星不自动等于24颗合格观测。',
        '- 增加卫星不等于修复上述投影触发机制。即使24星正常案例通过，仍应修复无攻击误抑制，不能靠更多卫星掩盖该问题。',
        '- 本批没有分钟级验证、IMU桥接或硬件实时性结论，也没有证明所有几何都可达20m。','',
        '## 文件及复现','',
        '全部18份24星导航结果从存档跟踪观测重演一致，9份四路输入哈希已核对；扩展配对测试及原回归套件共378项通过，另1项预期失败及原有1项无噪声C/N₀除零警告。接收机源码哈希与实验开始时一致。','',
        '参数与清单：[实验计划](SATELLITE_COUNT_24_PLAN.md)、[冻结配置](results/satellite-count-24/plan.json)。',
        '证据：[逐记录比较](results/satellite-count-24/comparison.json)、[无攻击逐历元诊断](results/satellite-count-24/normal_navigation_diagnosis.json)、[关闭投影消融](results/satellite-count-24/normal_ablation.json)、[测试日志](satellite_count_tests.log)。', '',
        '已有输入时运行 `python satellite_count_study.py 0` 至 `3` 可复用缓存复核；`python satellite_count_report.py`重建报告。`freeze`、`inputs`是首次建档步骤，不要用来覆盖已冻结输入。','',
        '![8星与24星合格输出](results/satellite-count-24/comparison.png)']
    Path('SATELLITE_COUNT_24_REPORT.md').write_text(('\n'.join(lines)+'\n').replace(chr(92)*2,chr(92)),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,4))
    for j,side in enumerate(('before','after')):
        ax.bar(np.arange(6)+(j-.5)*.35,[100*t[side]['good20']/t[side]['opportunities'] for t in totals],.35,label='8 PRNs' if j==0 else '24 PRNs')
    ax.set_xticks(np.arange(6),[f"{t['kind']}\n{t['subset']}" for t in totals]);ax.set_ylim(0,105)
    ax.set_ylabel('Outputs within 20 m / all opportunities (%)');ax.legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    assert digest(ROOT/'receiver.py')==plan['source_sha256']==digest(Path('single_source_receiver_fixed.py'))
    assert digest(Path('satellite_count_generator.py'))==plan['generator_sha256']
    save(ROOT/'summary.json',dict(totals=totals,rf_records=len(records),navigation_replays=replays,commercial_goal_met=False))
    artifacts=[Path(p) for p in ('satellite_count_generator.py','satellite_count_study.py','satellite_count_report.py',
               'normal_navigation_diagnosis.py','tests/test_satellite_count.py','SATELLITE_COUNT_24_PLAN.md',
               'SATELLITE_COUNT_24_REPORT.md','satellite_count_tests.log')]
    save(ROOT/'manifest.json',dict(sha256={str(p.relative_to(ROOT)):digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'},
          artifacts_sha256={str(p):digest(p) for p in artifacts}))
    print(totals,flush=True)


if __name__=='__main__':build()
