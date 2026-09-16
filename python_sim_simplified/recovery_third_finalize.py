"""Freeze once, evaluate untouched held-out seeds, then audit snapshots."""
import hashlib, json, pathlib, shutil, datetime
import numpy as np
import recovery_third_campaign as c

ROOT=c.ROOT

def main():
    candidates=[]
    for number in range(251,276):
        folder=ROOT/f'iteration{number}'
        results=json.loads((folder/'results.json').read_text())
        after=results['after']
        candidates.append((number,after))
        conclusion=json.loads((folder/'conclusion.json').read_text())
        if conclusion['delta_false_accepts']<0:
            conclusion['interpretation']='Authentic-presence gate reduced false acceptance; reconstruction error unchanged'
        elif number==265:
            conclusion['interpretation']='Unsupported absence/noise waveforms removed; genuine reconstruction unchanged'
        elif conclusion['delta_accepted']<0 and conclusion['delta_nmse']==0:
            conclusion['interpretation']='Additional uncertainty rejection costs availability; no waveform improvement claimed'
        c.write(folder/'conclusion.json',conclusion)
    eligible=[v for v in candidates if v[1]['false_accepts']==0 and v[1]['accepted']>0]
    chosen=min(eligible,key=lambda v:(v[1]['mean_nmse'],v[0]))[0]
    source=ROOT/f'iteration{chosen}'/'source.py'
    selected=pathlib.Path('recovery_third_selected.py');shutil.copyfile(source,selected)
    freeze=dict(selected_iteration=chosen,source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),seeds=[95101,95102,95103],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='Development seed94101 only: minimum all-present waveform NMSE among zero false acceptance and nonempty genuine acceptance; exact tie earlier revision',scope='Synthetic correlator signal recovery; no RF samples/PVT/hardware',extra_receiver_inputs=['nominal sky steering','two constant-state coherent blocks','160 independent same-condition noise-only banks','64 independent common-injection complex-gain samples'])
    c.write(ROOT/'recovery_freeze.json',freeze)
    baseline=c.evaluate(c.load(ROOT/'recovery_baseline.py'),freeze['seeds'])
    frozen=c.evaluate(c.load(selected),freeze['seeds'])
    c.write(ROOT/'recovery_holdout.json',dict(baseline=baseline,selected=frozen))
    audit=[]
    for number in range(251,276):
        folder=ROOT/f'iteration{number}';proposal=json.loads((folder/'proposal.json').read_text());result=json.loads((folder/'results.json').read_text())
        audit.append(dict(iteration=number,sha_matches=hashlib.sha256((folder/'source.py').read_bytes()).hexdigest()==proposal['source_sha'],same_inputs=[r['input_sha'] for r in result['before']['rows']]==[r['input_sha'] for r in result['after']['rows']]))
    assert all(r['sha_matches'] and r['same_inputs'] for r in audit)
    replay=[]
    for number in [251,260,264,271,275]:
        model=c.load(ROOT/f'iteration{number}'/'source.py');actual=c.evaluate(model,[94101]);expected=json.loads((ROOT/f'iteration{number}'/'results.json').read_text())['after']
        equal=all(np.isclose(actual[k],expected[k],rtol=1e-9,atol=1e-10) for k in ['mean_nmse','mean_code_rmse_m','accepted','false_accepts'])
        assert equal
        replay.append(dict(iteration=number,matches=bool(equal)))
    c.write(ROOT/'recovery_verification.json',dict(audit=audit,replay=replay,distinct_sha=len(set(json.loads((ROOT/f'iteration{n}'/'proposal.json').read_text())['source_sha'] for n in range(251,276)))))
    errors=[e for r in frozen['rows'] for e in r['accepted_code_squared_m']]
    samebaseline=[e for old,new in zip(baseline['rows'],frozen['rows']) for e in []]
    summary=dict(freeze=freeze,closed_iterations=25,baseline={k:v for k,v in baseline.items() if k!='rows'},selected={k:v for k,v in frozen.items() if k!='rows'},accepted_code_rmse_m=float(np.sqrt(np.mean(errors))),accepted_code_max_m=float(np.sqrt(max(errors))),verification='25 source hashes/input comparisons and5 snapshots replayed on11 development scenarios',limitations=['not full RF nor PVT','known sky and independent gain/noise calibration required','two fixed-state blocks, no navigation bit/Doppler tracking','two/three antennas can lose code observability','noise and BIC diagnostics selected from data are not calibrated integrity bounds','three held-out seeds cannot establish small false-alarm probabilities'])
    c.write(ROOT/'recovery_summary.json',summary)
    lines=['# 第251–275轮信号恢复研究','',f'25个真实机制变更已逐轮闭环，保留失败；最终按开发数据选择第{chosen}轮。没有新增RF/PVT或台架结论。','',
    '相关器域模型使用手机四角75.9×160.9mm，2/3天线为前2/3角，6PRN、41个lag、两个固定状态复相关块、单个公共欺骗源。真实分量波形和真码只进入生成/评价。接收机另需标称天空、64个独立共同注入通道标定样本、160个同态无信号噪声bank；这些不是手机自然可得的已认证输入。','',
    '论文只查阅了[ION公开摘要](https://m-private.ion.org/publications/abstract.cfm?articleID=19852)，没有取得全文。摘要介绍空间抑制结合SIC；本支检验这一思路的相关器域误差，不声称复现其完整接收机。','',
    '冻结文件在留出评估前写入；种子95101–95103共33案例。冻结后未修改模型。真实源27案例共162PRN，缺真/纯噪6案例共36PRN。','',
    '| 指标 | 基线 | 冻结方案 |','|---|---:|---:|',
    f"| 全部真实源案例平均波形NMSE | {baseline['mean_nmse']:.6f} | {frozen['mean_nmse']:.6f} |",
    f"| 真实PRN接受数/162 | {baseline['accepted']} | {frozen['accepted']} |",
    f"| 缺真或纯噪误接受数/36 | {baseline['false_accepts']} | {frozen['false_accepts']} |",
    f"| 所有案例码RMSE均值/m（包含无真源，不作为导航指标） | {baseline['mean_code_rmse_m']:.3f} | {frozen['mean_code_rmse_m']:.3f} |",'',
    f"冻结方案所接受真实PRN码RMSE {summary['accepted_code_rmse_m']:.3f}m、最大绝对码误差 {summary['accepted_code_max_m']:.3f}m。接受子集是经过筛选的，不代表全部场景恢复。波形NMSE对全部存在真实信号的案例计算，并非仅对接受案例美化。",'',
    '| 场景 | 基线波形NMSE均值 | 冻结波形NMSE均值 | 冻结接受PRN |','|---|---:|---:|---:|']
    for scenario in c.SCENARIOS:
        old=[r for r in baseline['rows'] if r['scenario']==scenario];new=[r for r in frozen['rows'] if r['scenario']==scenario]
        nm=lambda rows: f"{np.mean([r['nmse'] for r in rows]):.6f}" if rows[0]['genuine'] else '无真实源'
        lines.append(f"| {scenario} | {nm(old)} | {nm(new)} | {sum(r['accepted'] for r in new)} |")
    lines+=['','## 逐轮机制与真实对照','', '| 轮次 | 机制假设 | 开发NMSE前→后 | 接受真实PRN前→后 |','|---|---|---:|---:|']
    for number,_ in candidates:
        folder=ROOT/f'iteration{number}';p=json.loads((folder/'proposal.json').read_text());r=json.loads((folder/'results.json').read_text())
        lines.append(f"| {number} | {p['hypothesis']} | {r['before']['mean_nmse']:.6f}→{r['after']['mean_nmse']:.6f} | {r['before']['accepted']}→{r['after']['accepted']} |")
    lines+=['','## 失败与限制','',
    '252把各PRN真实方向投影后的残差直接PCA造成公共方向畸变。259平稳Toeplitz噪声约束没有改善开发恢复。266局部码Fisher门限损失半数开发可用PRN；267当前开发GOF判决不变。268只用训练块重构有明显方差代价。272固定0.18chip间隔的额外路径是额外模型先验，释放它导致真/反射分量难分、全部拒绝；未作为最终方案。274两侧边界审计只有有限局部覆盖，接受18→1；不采用为默认。275明确做欺骗波形重构抵消再真实方向合成，开发误差0.042286→0.107375，残余噪声仍然显著。', '',
    '271开发中有一次字符串替换误触reshape的实现错误，在任何结果产生前修正，未增加轮数，记录在其development_correction.txt。25轮均有真实算法/模型差异，不把种子、场景、参数扫、优化器内部步骤、测试或纠错计为轮数。', '',
    '模型未估未知天空/绝对位置/钟差，未做真实C/A RF采样、导航比特、动态载波/Doppler或连续导航；独立标定有效性未由硬件证明。第二块门控的方向/码来自第一块，但共享同一标定输入；门控之后的两块联合点估计不获得独立覆盖率保证。非线性局部极值、两天线几何退化、模板失配仍可能恶化时延。最终残差门限自由度为两源近似，独立噪声样本有限且选择后p值未校准，不是完整性保护级。', '',
    '验证：25个独特源SHA、25轮同输入SHA配对；251/260/264/271/275共5快照×11开发场景完整指标重放，容差1e-9；单元测试命令 `D:/Python/Python311/python.exe -m unittest tests.test_recovery_third`。脚本与逐轮输入哈希均保留。']
    pathlib.Path('recovery_third_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
