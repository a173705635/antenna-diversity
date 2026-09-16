"""Current-version scenario report; recompute metrics from archived navigation."""
from pathlib import Path
import html,json,hashlib
import numpy as np

BASE=Path('results/mechanism-four')
OUT=Path('results/current-performance-2026-09-16')
TITLE='当前各场景性能报告'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    source=read(BASE/'summary.json');freeze=read(BASE/'freeze.json')
    for name,value in freeze['dependencies'].items():assert sha(Path(name))==value,name
    rows=[];records={}
    for r in source['selected_rows']:
        folder=Path(r['input']);meta=read(folder/'metadata.json');records[str(folder)]=meta
        cache=Path(r['after_root'])/'evaluations'/r['after_id']
        with np.load(cache/'tracking.npz') as d:t=d['available_time'].copy()
        with np.load(cache/'navigation.npz') as d:p=d['position'].copy()
        mask=t>=meta['case']['attack_start'];times=t[mask]
        truth=np.zeros_like(p);truth[:,0]=meta['case']['velocity_east']*t
        error=np.linalg.norm((p-truth)[:,:2],axis=1)[mask];valid=np.isfinite(error);finite=error[valid]
        score=r['after']
        for key,value in [('opportunities',len(error)),('solved',valid.sum()),('good20',(error<=20).sum()),('error30',(error>=30).sum()),('error50',(error>=50).sum())]:assert score[key]==int(value)
        changes=np.flatnonzero(np.diff(np.r_[False,~valid,False]));gaps=[]
        for left,right in zip(changes[::2],changes[1::2]):
            gaps.append(dict(first_missing_s=float(times[left]),last_missing_s=float(times[right-1]),recovered_s=float(times[right]) if right<len(times) else None,duration_s=float((right-left)*.01)))
        assert np.isfinite(p[~mask]).all(1).any() and score['initialized_before_attack']
        record=dict(name=folder.name,kind=meta['kind'],seed=meta['seed'],stars=meta['case']['stars'],duration_s=meta['case']['duration'],
            antennas={'two':2,'three':3,'four':4}[r['subset']],evidence='本批冻结留出' if r['round']==4 else '开发/回归',
            opportunities=len(error),published=int(valid.sum()),good20=int((error<=20).sum()),rate_pct=float(100*np.mean(error<=20)),
            p50_m=float(np.median(finite)),p95_m=float(np.quantile(finite,.95)),maximum_m=float(finite.max()),
            error30=int((error>=30).sum()),error50=int((error>=50).sum()),longest_gap_s=score['longest_unavailable_s'],gaps=gaps,
            last_published_s=float(times[valid][-1]),tail_good20=int((error[-100:]<=20).sum()),pre_outputs=int(np.isfinite(p[~mask]).all(1).sum()),
            baseline_good20=r['before']['good20'],baseline_maximum_m=r['before_diagnostics']['max_error'],
            navigation_file=str(cache/'navigation.npz'),navigation_sha256=sha(cache/'navigation.npz'),tracking_sha256=sha(cache/'tracking.npz'))
        rows.append(record)
    for folder,m in records.items():
        assert sha(Path(folder)/'samples.npy')==m['samples_sha256']
        assert sha(Path(folder)/'observations.npz')==m['observations_sha256']
    by={(r['name'],r['antennas']):r for r in rows}
    sections=[]
    def paragraph(text):sections.append(('p',text))
    def heading(text):sections.append(('h',text))
    def table(headers,data):sections.append(('t',(headers,data)))
    paragraph('截至2026年9月16日，最新研究版本在已测四天线弱信号案例中均实现连续20m内输出，修复了此前已知的切入错误；三天线仍不能保底。近源、噪声冲击及攻击撤除测试存在不同阶段的中断，真实信号消失后可以停止输出。现有证据不足以支持稳定1m、全工况持续导航或商用可靠性。')
    paragraph('本文只汇总 single_source_receiver_history.py 的26个最终案例，不混入中间候选或更早版本的最好成绩。相同四路RF上的不同天线子集不算独立录波；本报告覆盖12份RF、3/4天线各12个案例，以及2天线2个比较案例。')
    heading('1. 测试范围与统计口径')
    table(['项目','本报告口径'],[
        ['系统版本',f"single_source_receiver_history.py；SHA-256：{freeze['source_sha256']}"],
        ['环境与阵列','手机原固定布局，约75.9×160.9mm；固定标称姿态，开阔环境模型，四路同步假设。'],
        ['启动','先以45dB-Hz真实信号启动；0.3秒开始攻击或噪声变化。所有案例都核对了攻击前实际位置输出。'],
        ['攻击','一个物理发射源，可同时伪造全部PRN；每PRN欺骗功率名义上比真实信号高10dB，伪造约100m东向偏移。'],
        ['卫星与时长','主线24星；撤除回归为历史8星。每份录波2或4秒，0.3秒后的评分机会分别171或371次。'],
        ['合格率','水平误差≤20m的发布次数÷全部评分机会；停发计入分母。当前最终案例全部已发布位置均≤20m，因此本批合格率等于发布率。'],
        ['精度','P50、P95、最大值只统计已发布位置；停发不是零误差。不能用仅少量过渡输出的低误差证明持续导航精度。'],
        ['失败阈值','≥30m、≥50m发布次数单列；没有观察到错误不等于已标定现场错误概率。'],
        ['时间与留出','时间指输出可用时间，包含模型20ms处理延迟；仅117901为本批冻结后的留出，其余是开发/回归。'],
        ['研究范围','不纳入冷启动攻击、多物理欺骗源；当前未接入IMU，无实机/台架数据。']])
    heading('2. 三／四天线主要场景总览')
    paragraph('表内格式为“20m内次数/机会（百分比）；最大水平误差”。这里只评价所列样本，不把单个种子的通过外推为该类场景普遍通过。')
    cases=[
        ('115901_normal_2.0','无攻击，45dB-Hz，24星2秒','该回归样本连续通过。'),
        ('114101_control_30_24_2','无攻击，转30dB-Hz，24星2秒','该开发样本连续通过；尚无该工况的新种子留出。'),
        ('114101_weak_30_24_2','30dB-Hz单源攻击，开发A，2秒','三/四均通过；三天线最大误差较旧版14.16m有所增加。'),
        ('116901_weak_30_24_2','30dB-Hz单源攻击，开发B，2秒','四天线连续通过；三天线过渡后停发。'),
        ('116921_weak_30_24_4','30dB-Hz单源攻击，开发C，4秒','四天线已知30m错误修复；三天线失败。'),
        ('117901_weak_30_24_4','30dB-Hz单源攻击，冻结留出，4秒','四天线通过且基线本已通过；三天线仍失败。'),
        ('115901_cluster_2.0','强信号，3星近源簇，24星2秒','3颗星在源方向及约±0.5°；切入中断0.10秒后恢复。'),
        ('114001_release_2.0','强信号攻击后撤除，8星2秒','切入有0.10秒中断；0.45秒撤除后未再中断。'),
        ('114101_burst_35_24_2','35dB-Hz攻击＋短时20dB-Hz冲击','冲击结束后仍需恢复时间；非全程导航。')]
    overview=[]
    for name,label,verdict in cases:
        cells=[]
        for ant in [3,4]:
            r=by[name,ant];cells.append(f"{r['good20']}/{r['opportunities']}（{r['rate_pct']:.1f}%）；{r['maximum_m']:.2f}m")
        overview.append([label,*cells,verdict])
    table(['场景','三天线','四天线','判读'],overview)
    heading('3. 真实信号消失：保护输出专项')
    paragraph('0.3秒真实信号消失后，系统仍有3～4次过渡正确输出，随后保持停发，直到录波结束未恢复。这些输出处于观测/处理过渡，不计为“无真实信号仍能导航”。三个种子中117901为冻结留出。')
    table(['种子','天线','过渡输出数','最后发布时刻s','最大误差m','≥30m/≥50m','末1秒发布'],[
        [r['seed'],r['antennas'],r['published'],f"{r['last_published_s']:.2f}",f"{r['maximum_m']:.2f}",f"{r['error30']}/{r['error50']}",f"{r['tail_good20']}/100"] for r in rows if r['kind']=='absent'])
    paragraph('结论仅为已测负例没有错误恢复。没有验证多分钟的无真实信号保持，也没有获得统计完整性上界；此时持续导航目标没有实现。')
    heading('4. 中断具体发生在哪个阶段')
    interval_rows=[]
    for name,label in [('115901_cluster_2.0','近源簇'),('114001_release_2.0','攻击后撤除'),('114101_burst_35_24_2','短时强噪声')]:
        for ant in [3,4]:
            r=by[name,ant]
            for g in r['gaps']:
                interval_rows.append([label,ant,f"{g['first_missing_s']:.2f}～{g['last_missing_s']:.2f}",f"{g['recovered_s']:.2f}",f"{g['duration_s']:.2f}"])
    table(['场景','天线','无输出历元范围s','首个恢复输出s','缺失时长s'],interval_rows)
    paragraph('撤除测试的攻击区间为0.30～0.45秒：0.43秒已经恢复输出，所以记录中的0.10秒空窗属于切入，不能说成撤除引起的中断。噪声冲击区间为0.60～0.80秒，三/四天线分别到1.07/1.08秒才恢复，按输出可用时间比冲击结束晚约0.27/0.28秒；这包含处理延迟及观测恢复，不是独立测得的捕获器延迟。')
    heading('5. 两天线比较与三天线失败原因')
    table(['两天线案例','合格次数/机会','合格率','最大误差m','最长缺失s'],[
        ['无攻击30dB-Hz' if r['kind']=='control' else '30dB-Hz单源攻击',f"{r['good20']}/{r['opportunities']}",f"{r['rate_pct']:.1f}%",f"{r['maximum_m']:.2f}",f"{r['longest_gap_s']:.2f}"] for r in rows if r['antennas']==2])
    paragraph('两天线施加一个零陷后只剩一维，当前方向能量比无法提供正交方向对照；无攻击弱信号下也会因准入后的观测及恢复连续性不足而停发。这是两类不同限制，不要求本批补齐两天线性能。')
    paragraph('三天线的主要限制是空间抑制后的真实信号增益和可靠观测冗余不足。旧困难种子116901在质量/增益筛选后有效星中位数6，方向证据加入后为2；116921虽有不少≥6星历元，码一致性和连续恢复证据仍不足。保留短间隙历史只能修复“信息存在但没有利用”的情形，不能创造真实缺失的观测。新种子15/371且末1秒没有输出，仍属于持续导航失败。')
    heading('6. 四天线改善来自什么，证据有多强')
    table(['案例','合格次数：旧→新','最大误差m：旧→新','证据性质'],[
        [r['name'],f"{r['baseline_good20']}→{r['good20']}/{r['opportunities']}",f"{r['baseline_maximum_m']:.2f}→{r['maximum_m']:.2f}",r['evidence']] for r in rows if r['kind']=='weak' and r['antennas']==4])
    paragraph('已知0.34秒错误来自相邻有效星集合切换：跟踪弧未断，但平滑重置、相邻历元共同星为零，新码位置进入导航。新版本保留最多一个被拒历元前的同弧平滑历史，并使用最近已发布位置与实际载波位移接续。没有降低方向阈值、放宽20m目标或减少这些四天线案例的输出次数。')
    paragraph('117901是一个新几何/噪声种子的留出，四天线基线本来已371/371；新版本保持通过、最大误差降至5.30m。这支持本次修改在该种子没有退化，不足以证明所有几何都可靠。三天线开发A的最大误差从14.16m增到16.66m，保留历史也可能延续偏差；不能把局部收益描述成全面精度提升。')
    heading('7. 当前没有证明的能力')
    table(['项目','当前证据边界'],[
        ['稳定1m / 稳定数米级','弱信号四天线已测最大误差仍可达16.68m，三天线16.66m；不能承诺稳定1m或所有时刻数米级。'],
        ['三天线持续导航保底','不满足；多份困难几何记录长期停发。'],
        ['全部攻击工况','仅有限几何、名义+10dB、全PRN同源攻击；功率扫描、缓慢拖曳、部分PRN攻击等未在最新候选统一覆盖。'],
        ['时长','最新统一候选最长连续录波4秒。更早版本6秒近源结果不能继承，更不能外推分钟级。'],
        ['24星撤除 / 35dB-Hz稳态攻击','最新候选没有对应的独立专门记录；撤除只有8星，35dB-Hz只有叠加噪声冲击案例。'],
        ['手机实际使用','未验证手持转动、车内遮挡与多径、互耦与实测天线方向图、完整同步/温漂误差分布、实时算力和功耗。'],
        ['商用完整性','阈值未作现场虚警/漏警标定，低残差并非真实误差上界；当前零错误发布只适用于有限样本。'],
        ['明确排除的研究方向','冷启动攻击、多物理欺骗源不在本次需求范围，不列为待解决的当前场景。']])
    heading('8. 全部26个最终案例的精度与可用性明细')
    paragraph('P50/P95仅基于已发布位置；尤其对大量停发的三/两天线，低分位误差不能代表全时段定位能力。所有行≥30m和≥50m发布均为0，且当前行的已发布数与20m内数一致。')
    table(['输入ID','天线','证据','发布/机会','≤20m率','P50 m','P95 m','最大m','最长缺失s','末1秒≤20m'],[
        [r['name'],r['antennas'],r['evidence'],f"{r['published']}/{r['opportunities']}",f"{r['rate_pct']:.1f}%",f"{r['p50_m']:.2f}",f"{r['p95_m']:.2f}",f"{r['maximum_m']:.2f}",f"{r['longest_gap_s']:.2f}",f"{r['tail_good20']}/100"] for r in sorted(rows,key=lambda r:(r['name'],r['antennas']))])
    heading('9. 数据来源与本次核对')
    paragraph('来源为 mechanism-four 的冻结候选及审计结果：26个最终案例来自12份RF；研究批次整体包含44组配对、64份唯一导航重演，10次新RF前端计算和16份相同前端缓存复用。后两组数字包含消融/基线，不能当作26个最终案例之外的新场景。测试结果为397通过、1预期失败、1条原有零噪声警告。')
    paragraph('本报告重新读取26份导航结果，核对20m、30m、50m计数与实际启动，重算P50/P95、停发区间和末段输出；验证12份输入的样本/观测哈希及33项冻结运行依赖。没有新增实验，也没有修改接收机算法。当前性能JSON保存逐例指标、时段及导航/跟踪哈希。')
    paragraph('后续优先级：先扩大四天线未参与调参的几何与时长验证，并继续检查绝对位置准入；三天线重点研究低保留增益下的测量质量及恢复连续性。仿真达到稳定结果后仍需要同步RF台架和实际天线验证。')
    md=[f'# {TITLE}',''];body=[f'<h1>{TITLE}</h1><p class="subtitle">2026-09-16 · 最新冻结候选 · 单源、正常启动后 · 仿真证据</p>']
    for kind,data in sections:
        if kind=='p':md += [data,''];body.append('<p>'+html.escape(data)+'</p>')
        elif kind=='h':md += ['## '+data,''];body.append('<h2>'+html.escape(data)+'</h2>')
        else:
            headers,items=data
            md += ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']
            md += ['|'+'|'.join(str(x).replace('|','／') for x in row)+'|' for row in items];md.append('')
            body.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in items)+'</tbody></table></div>')
    css='body{font-family:"Microsoft YaHei",system-ui,sans-serif;margin:0;color:#203040;background:#eef2f5;line-height:1.7}main{max-width:1280px;margin:32px auto;padding:40px;background:white}h1{font-size:32px;color:#153c48}h2{margin-top:36px;border-bottom:2px solid #287e80;padding-bottom:8px;font-size:21px}.subtitle{color:#667788}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;margin:14px 0 24px}th{text-align:left;background:#153c48;color:white}td,th{padding:9px 11px;border:1px solid #d7e1e6;vertical-align:top}tr:nth-child(even){background:#f5f8fa}td:first-child{overflow-wrap:anywhere}p{font-size:15px}@media print{body{background:white}main{margin:0;padding:0;max-width:none}h2{break-after:avoid}tr{break-inside:avoid}thead{display:table-header-group}table{font-size:9px}td,th{padding:5px}}'
    Path('CURRENT_SCENARIO_PERFORMANCE_2026-09-16.md').write_text('\n'.join(md),encoding='utf-8')
    Path('CURRENT_SCENARIO_PERFORMANCE_2026-09-16.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+TITLE+'</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>',encoding='utf-8')
    (OUT/'performance.json').write_text(json.dumps(dict(source_sha256=freeze['source_sha256'],cases=rows,records=records),ensure_ascii=False,indent=2),encoding='utf-8')
    names=['CURRENT_SCENARIO_PERFORMANCE_2026-09-16.md','CURRENT_SCENARIO_PERFORMANCE_2026-09-16.html','build_current_performance_report.py',str(OUT/'performance.json')]
    (OUT/'manifest.json').write_text(json.dumps({p:sha(Path(p)) for p in names},indent=2),encoding='utf-8')
    print(dict(final_cases=len(rows),rf_records=len(records),published_over30=sum(r['error30'] for r in rows),published_over50=sum(r['error50'] for r in rows)),flush=True)


if __name__=='__main__':build()
