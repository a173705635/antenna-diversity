"""Recompute navigation and metrics before publishing the ten-round report."""
import ast
from pathlib import Path
import numpy as np
from spatial_ten_study import ROOT,read,save,digest,load


def build():
    selected=read(ROOT/'freeze.json')['source_sha256']
    models={digest(p):p for p in [ROOT/'baseline.py',ROOT/'round05/receiver.py',ROOT/'round06/receiver.py',ROOT/'selected.py']}
    records={};checked={};rows=[];rounds=[]
    for number in range(1,11):
        folder=ROOT/f'round{number:02d}'
        rounds.append(dict(**read(folder/'proposal.json'),**read(folder/'conclusion.json')))
        for path in folder.glob('*.json'):
            row=read(path)
            if not isinstance(row,dict) or 'after_id' not in row:continue
            records[row['input']]=read(Path(row['input'])/'metadata.json')
            for label in ('before','after','parent'):
                if label not in row:continue
                base=Path(row['before_root']) if label=='before' else ROOT
                identity=row[label+'_id'];key=(str(base),identity)
                if key not in checked:
                    cache=base/'evaluations'/identity
                    with np.load(cache/'tracking.npz') as data:z={k:data[k] for k in data.files}
                    with np.load(cache/'navigation.npz') as data:position=data['position'].copy()
                    model=load(models[row[label]['source_sha256']]);actual=model.navigate(z)
                    np.testing.assert_allclose(actual['position'],position,atol=1e-9,rtol=0,equal_nan=True)
                    meta=records[row['input']];post=z['available_time']>=meta['case']['attack_start']
                    truth=np.zeros_like(position);truth[:,0]=meta['case']['velocity_east']*z['available_time']
                    error=np.linalg.norm((position-truth)[:,:2],axis=1)[post]
                    for metric,expected in [('solved',np.isfinite(error).sum()),('good20',(error<=20).sum()),('error30',(error>=30).sum()),('error50',(error>=50).sum())]:
                        assert row[label][metric]==int(expected),(key,metric)
                    checked[key]=dict(ready=row[label]['initialized_before_attack'] and bool(np.isfinite(position[~post]).all(1).any()),
                        max_error=float(np.nanmax(error)) if np.isfinite(error).any() else None,
                        valid_median=float(np.median(actual['selected'][post].sum(1))),
                        projection_fraction=float(np.mean(z['ratio'][post]>=3)),
                        source_sha256=row[label]['source_sha256'])
                row[label+'_diagnostics']=checked[key]
            row['round']=number;rows.append(row)
    for folder,meta in records.items():
        for name,key in [('samples.npy','samples_sha256'),('observations.npz','observations_sha256')]:
            assert digest(Path(folder)/name)==meta[key]
    freeze=read(ROOT/'freeze.json')
    for name,sha in freeze['dependencies'].items():assert digest(Path(name))==sha,name
    assert digest(Path('single_source_receiver_spatial.py'))==selected
    navigation=[]
    for p in (ROOT/'baseline.py',ROOT/'selected.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        navigation.append(ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='navigate')))
    assert navigation[0]==navigation[1], 'navigation admission changed'
    latest={(r['input'],r['subset']):r for r in rows if r['after']['source_sha256']==selected}
    tests=Path('spatial_ten_tests.log').read_text(encoding='utf-8')
    assert 'failed' not in tests.replace('xfailed','') and 'passed' in tests
    for cache in (ROOT/'evaluations').iterdir():
        assert all((cache/name).is_file() for name in ('score.json','tracking.npz','navigation.npz')),cache
    frontend_count=len(list((ROOT/'frontends').glob('*/tracking.npz')))
    save(ROOT/'audited_rows.json',rows)
    save(ROOT/'summary.json',dict(rounds=rounds,paired_rows=len(rows),unique_navigation_replays=len(checked),
         unique_rf_records=len(records),new_rf_frontend_runs=frontend_count,selected_cases=len(latest),selected_rows=list(latest.values()),
         source_sha256=selected,commercial_goal_met=False,test_summary=tests.strip().splitlines()[-1]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,8),layout='constrained')
    for ax,subset in zip(axes,('three','four')):
        group=[r for r in latest.values() if r['subset']==subset];y=np.arange(len(group))
        for shift,label,color in ((-.18,'before','#9aa4b2'),(.18,'after','#26876b')):
            value=[100*r[label]['good20']/r[label]['opportunities'] for r in group]
            ax.barh(y+shift,value,height=.34,color=color,label='Baseline' if label=='before' else 'Candidate')
        labels=[]
        for r in group:
            meta=records[r['input']];c=meta['case']
            labels.append(f"{c['stars']} stars / {meta['kind']} / {meta['seed']} / {c['duration']:g}s")
        ax.set_yticks(y,labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,105)
        ax.set_xlabel('Outputs within 20 m / all opportunities (%)');ax.set_title(subset+' antennas')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper right')
    fig.savefig(ROOT/'navigation_comparison.png',dpi=160);plt.close(fig)
    lines=['# 空间触发修复：10轮研究结果','',
        '本批按10个研究问题推进，包含假设否决、算法修改、完整链路对照和冻结验证；不是10个全新算法，也没有把不同种子或天线配置单独计轮。数值/相关域筛查与完整RF验证分别列出。', '',
        '## 逐轮记录','', '|轮|研究问题|结论|证据|','|---|---|---|---|']
    for r in rounds:lines.append(f"|{r['number']}|{r['problem']}|{r['decision']}|{r['evidence']}|")
    lines += ['', '## 最终候选的完整RF对照','',
        '每个单元格为“≤20m / 总机会”；暂停计入分母。所有行均与同一输入RF的旧审计修正版比较。8星保留为边界回归，24星为主配置。开发样本、新种子和延长时长分别由轮次标识。', '',
        '|轮|录波|天线|基线≤20m/机会|候选≤20m/机会|候选最大误差m|候选≥30m/≥50m|最长缺失s|攻击前就绪|',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    for r in latest.values():
        b,a=r['before'],r['after'];d=r['after_diagnostics'];maximum='无输出' if d['max_error'] is None else f"{d['max_error']:.2f}"
        lines.append(f"|{r['round']}|{Path(r['input']).name}|{r['subset']}|{b['good20']}/{b['opportunities']}|{a['good20']}/{a['opportunities']}|{maximum}|{a['error30']}/{a['error50']}|{a['longest_unavailable_s']:.2f}|{d['ready']}|")
    lines += ['', '![同RF导航合格率](results/spatial-ten/navigation_comparison.png)', '', '## 方法与边界','',
        '- 最终修改：按真实卫星模型与单公共源模型的噪声归一化残差减少量触发；公共源交替拟合增加到6次。导航准入、保护输出、载波桥接和恢复门限与基线相同。',
        '- 第2轮固定残差比例会把噪声误当欺骗，已否决；第5轮两次拟合候选四天线近源退化，第6轮六次拟合改善，但8星近源仍会因有效观测不足中断。',
        '- 误差低的已发布位置不等于全程导航。低信噪比、近源簇、长期通道误差等必须同时查看缺失和攻击前就绪。',
        '- 噪声尺度含信号能量，相关样本不完全独立，阈值3没有标定现场虚警概率；标称姿态/导向、同步、星历及粗码周期辅助仍是前提。',
        '- 两天线的真实导向和公共源导向可张成整个二维复观测空间，单纯低残差不能证明分离可靠；其结果仅作比较，不强求补齐三/四天线性能。',
        '- 新种子只是本批冻结后首次评价的仿真留出，不是台架实测。延长时长使用另一随机录波，不能解释为原2秒录波的同噪声续接。',
        '- 只研究已完成真实导航启动后的单物理欺骗源，源可以伪造全部24个PRN。没有引入多源、冷启动或IMU。没有证明分钟级导航、手机实时运行能力或商用完整性。','',
        '## 审计和复现','',
        f"共{len(rows)}组完整RF配对记录（部分另含父候选），{len(records)}份不同RF输入；本批实际计算{frontend_count}次RF前端，其余基线使用同输入同源码归档。{len(checked)}份唯一导航结果重演一致；输入哈希和冻结依赖已核对。",'',
        tests.strip().splitlines()[-1], '',
        '研究入口：[single_source_receiver_spatial.py](single_source_receiver_spatial.py)。原入口保留用于基线；当前候选的适用边界以本报告为准。', '',
        '文件：[计划](SPATIAL_TEN_PLAN.md)、[数学方法](SPATIAL_TEN_METHODS.md)、[完整结果](results/spatial-ten/summary.json)、[逐案例审计](results/spatial-ten/audited_rows.json)、[冻结配置](results/spatial-ten/freeze.json)。', '',
        '运行 `python spatial_ten_report.py` 可重新检查已存档导航及输入；`spatial_ten_diagnostics.py 2`（或3、4）复现相关域对照；`spatial_ten_study.py pair 轮号 模型路径 输入目录 天线子集` 从RF重跑/复用有哈希身份的缓存。输入生成脚本只用于首次建立新档案，不覆盖现有目录。', '',
        '测试命令为 `python -m pytest tests -q`。初次未指定tests目录的命令误收集历史归档中的同名测试而报collection mismatch；原日志保留在 spatial_ten_collection_error.log，不据此删除或修改归档。']
    Path('SPATIAL_TEN_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest={str(p):digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    save(ROOT/'manifest.json',manifest)
    print(dict(paired_rows=len(rows),records=len(records),replays=len(checked),archive_files=len(manifest)),flush=True)


if __name__=='__main__':build()
