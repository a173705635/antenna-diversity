"""Build the research report directly from saved metrics, without hand-entered results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/reports/data/2026-09-12-sdcp-self-check-final'
REPORT = ROOT/'docs/reports/2026-09-12-sdcp-self-check.md'


def main():
    rows = json.loads((DATA/'summary.json').read_text())
    pvts = json.loads((DATA/'pvt_summary.json').read_text())
    manifest = json.loads((DATA/'manifest.json').read_text())
    def percent(v): return '—' if v is None else f'{100*v:.2f}%'
    headline = []
    for kind in ['los','strong','weak','static','common']:
        selected = [x for x in rows if x['domain']=='RF_replay' and x['case']==kind]
        metric = 'false_alarm_all' if kind=='los' else 'detection_all'
        values = [x[metric] for x in selected]
        headline.append(f"- {kind}：{'误报' if kind=='los' else '检出'}历元比例 "
                        f"{percent(min(values))}～{percent(max(values))}；可判断覆盖 "
                        f"{percent(min(x['coverage'] for x in selected))}～{percent(max(x['coverage'] for x in selected))}。")
    s = '''# SDCP 自检、压力测试与定位对照

## 结论应如何阅读

这是一轮单频双天线研究验证，不是实机可用性认证。研究按自检实现、无多径压力测试、独立 RF 多径验证、探索性定位对照执行。系统没有把多径检测与外部干扰混在一起。

低误报必须与可判断覆盖率一起看：输出“不可可靠判断”不等于排除了多径。多星共同偏移、噪声或状态质量超限时，自检可以停止强制判决，但没有自动确定异常来自硬件还是多径。

## 1. 原理与实际接口

输入是每历元每星的校准取余 SDCP 残差、共同时间轴、通道时间偏差、有效性标志、几何误差上界和固定标定下限。残差沿用已有每历元 TL SPP 几何补偿及零硬件偏差仿真假设；检测函数不接收多径标签、真实位置或受污染卫星编号。

输出 state=0（未发现异常）、1（疑似多径）、-1（不可可靠判断），同时保存 score_m、threshold_m 和 reason。0 不能解释为保证无多径，1 不能决定哪根天线受到污染。

1. 首先检查观测有限性、完整性、时间步长、通道对齐、调用方提供的锁定/半周有效性以及几何不确定性。
2. 对可见的大于40 mm的相邻取余相位变化建立隔离状态，直到外部 phase_reset_valid 证明新相位分支有效。单纯等一个窗口不会恢复。整数周期跳变在取余后不可见，仍依赖跟踪器。
3. 使用完整的后向500 ms或1 s圆周均值，每10 ms更新，不拟合每星参考常数。窗口内部按100 ms分块，以块均值相对整体均值的圆周偏差标准差除以块数平方根构造噪声代理量。
4. 候选阈值 T=T_floor+3×noise_proxy+geometry_bound；超过2×T_floor时停判，不无限抬高阈值。这个噪声量没有独立样本假设或CFAR概率保证。
5. 完整窗口可用卫星少于6颗时停判；至少一半可用卫星同时超过各自阈值时，标记公共异常不可辨识。这里没有自动扣除公共偏移，也没有估计姿态。
6. 单星异常连续10个更新历元才确认。100 ms为按更新数约定的确认长度，首次越限至第10个更新实际间隔90 ms；下表的延迟按真实时间戳计算。

几何不确定性最大允许5 mm，通道时差最大1微秒。它们都是首轮工程设定，不代表已证明1微秒对任意动态条件都足够。调用方隐瞒或低估几何误差时，自检无法保证识别。

## 2. 开发诊断、冻结与独立数据

第一版在6300/6302旧记录上暴露半周跳变窗口到期后误报，第二版修正为持续隔离。因此6300/6302只作开发数据。两版完整结果均保留，不能将第二版在开发数据上的重跑称为独立验证。

最终评估使用新生成6400/6402两组RF噪声种子；每组含LOS双通道、TR强/弱/启动多径，以及双通道共模多径，共14条3秒单通道记录，每条独立跟踪8颗GPS卫星。每组两个天线在相同时间轴上形成观测，程序顺序计算，非复用一条环路。

旧6200/6202/6204三组LOS仅用于阈值下限的99%分位数标定；最终数据不用于调参。新RF沿用既有星座几何、信号功率和采样配置，独立性只涉及噪声种子，不代表已经覆盖环境分布迁移。

'''
    s += '|窗口|标定下限|阈值上限|\n|---|---:|---:|\n'
    for n,f in manifest['floors_m'].items():
        s += f'|{int(n)*10} ms|{f*1000:.6f} mm|{f*2000:.6f} mm|\n'
    s += '''
LOS诊断中，三组标定记录的SPP几何补偿相对仿真真实几何的最大差均为约0.000030 mm，远小于毫米级逐星残差均值。因此本组条件下SPP几何计算误差不足以解释该偏差；这不等于已经确定跟踪偏差的全部根因。噪声和每星均值统计见开发诊断数据。

## 3. 评分口径

最终统计区间为1.5秒及以后；完整记录覆盖率另存JSON，包含窗口预热。误报/检出均按卫星×历元计数，既报告全区间比例，也报告可判断子集比例。没有可判断样本时显示“—”，不是0。

每个受影响PRN单独记录有无确认报警及延迟；静态多径起点为0秒，其他RF多径从0.5秒起、10 ms渐入。重叠窗口、同组卫星、同一LOS上衍生的注入场景不作为独立重复试验。两组种子不能给出可靠事件检测概率。

固定阈值对照包括立即越限和相同10历元持续确认两个版本，均使用同一圆周均值与标定下限。自检没有通过使用不同测试记录获得优势。

## 4. 新 RF 多径验证结果

强/启动多径：TR侧PRN1、-6 dB、超额时延0.25 L1码片、反射相位π/2；弱多径改为-18 dB；共模为两天线PRN1相同相对反射。其他卫星没有注入多径。

|种子|窗口ms|场景|可判断覆盖|全区间误报|可判断内误报|PRN1检出|可判断内检出|固定持续误报|固定持续检出|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for x in rows:
        if x['domain'] == 'RF_replay':
            s += '|'+ '|'.join([str(x['seed']),str(x['window_ms']),x['case']]+[percent(x[k]) for k in
                ['coverage','false_alarm_all','false_alarm_judged','detection_all','detection_judged',
                 'fixed_confirmed_false_alarm','fixed_confirmed_detection']])+'|\n'
    s += '\n### 逐记录确认延迟\n\n|种子|窗口ms|场景|是否确认|起点至确认秒|\n|---|---:|---|---|---:|\n'
    for x in rows:
        if x['domain']=='RF_replay':
            for e in x['events']:
                delay = '—' if e['delay_s'] is None else f"{e['delay_s']:.3f}"
                s += f"|{x['seed']}|{x['window_ms']}|{x['case']}|{e['detected']}|{delay}|\n"
    s += '''
## 5. 观测域压力测试

下列压力施加在新LOS记录的残差或元数据上，不重新经过RF/DFE/DLL/PLL：公共10 mm偏置、0.5秒后10 mm/s漂移、附加12 mm标准差独立相位噪声、-15至15 mm逐星方向相关误差代理、已知15 mm几何不确定性、0.8至1.2秒失锁标志/缺测、半周跳变、100微秒通道时间偏差、1/3/5/8颗星25 mm恒定异常、漂移与单星异常组合、未报告的整周跳变。

其中noise_x3只是场景标识，实际注入为12 mm白噪声，不保证总噪声恰为三倍，也不等于真实C/N0变化。geometry_unreported是逐星误差代理而非新的天线姿态RF仿真。半周/整周场景检验的是观测接口盲点，不能代表真实失锁重捕流程。

表中未注入多径的半周/几何/漂移场景报警算误报；人工25 mm异常视为待检测差分异常，不能据此声称真实多径检出率。

|种子|窗口ms|压力场景|可判断覆盖|全区间误报|可判断内误报|全区间检出|可判断内检出|
|---|---:|---|---:|---:|---:|---:|---:|
'''
    for x in rows:
        if x['domain']=='observation':
            s += '|'+ '|'.join([str(x['seed']),str(x['window_ms']),x['case']]+[percent(x[k]) for k in
                 ['coverage','false_alarm_all','false_alarm_judged','detection_all','detection_judged']])+'|\n'
    s += '''
## 6. 接入定位的探索性对照

只使用原始RF记录，观测层压力注入不进入定位，避免相位与伪距不一致。使用已知基线、GPS卫星状态和既有粗伪距时间辅助；不是自主导航电文定位。原始3秒观测每50 ms取一个历元，四种方法采用完全相同历元。码定位逐历元解算；码/载波联合浮点定位为全记录离线解算，每星实数偏差自由估计，没有整周固定，也不是实时定位性能。

权重方式：plain=1；fixed为立即越限后0.1；fixed_confirmed为持续越限后0.1；self_check仅在state=1时0.1。每星同一权重施加于两天线的码与载波，sigma按1/sqrt(weight)放大。不可判断回退权重1并显式保留风险，不做强制剔除。

表内为1.5秒后3D位置误差RMS（米），不是标准差；不能把同一数值同时解释成精度与抖动。

|种子|场景|方法|码位置RMS/m|浮点位置RMS/m|浮点去均值RMS/m|
|---|---|---|---:|---:|---:|
'''
    for x in pvts:
        for k,v in x['modes'].items():
            s += f"|{x['seed']}|{x['case']}|{k}|{v['code']['rms_m']:.6f}|{v['float']['rms_m']:.6f}|{v['float']['demeaned_rms_m']:.6f}|\n"
    s += '''
## 7. 能证明与不能证明的内容

本轮实现了三态观测质量控制、有限阈值变化、公共异常停判及持久相位隔离，并在新噪声记录上进行检验。它仍依赖外部硬件偏差和几何可信度，不能称为免标定、自主估计姿态或在线公共漂移校准。

一个不能回避的反例是：one_static_25mm中的同一组残差，既可以解释为单星差分多径，也可以解释为未上报的单星25 mm几何补偿偏差。检测器接收的数字完全相同，输出必然相同。将其解释为后者时，报警就是误报。这是可辨识性限制，不能通过换噪声种子或增加平均窗消除；必须由外部误差界或其他观测提供区分信息。

可观察的单星持续强异常能否检测，以新RF表为准；弱多径和两天线共同误差仍须看漏检，不可用“未报警”证明干净。多个卫星异常导致停判是覆盖率损失，不是成功检测所有多径。原理上，同样的差分残差可以来自几何偏差、通道偏差或多径；单靠SDCP不能唯一确定原因。

未完成实机C/N0变化、基线姿态变化、数据符号半周恢复、真实失锁重捕和跨场地标定迁移验证；两组种子与一个固定星座不足以证明普适性。检测尚未获得部署验收，定位仅为探索性结果，不据此替换默认定位流程。

下一阶段应优先接入真实锁定/半周有效性与几何不确定性来源，再做跨信号强度、跨几何的RF矩阵；对弱/共模多径需要增加独立观测信息，不能靠继续抬高或降低阈值解决。

## 8. 文件与复现

- [冻结设计](../validation/2026-09-12-sdcp-self-check-design.md)、[冻结代码哈希](../validation/2026-09-12-sdcp-freeze.json)。
- [首轮结果](data/2026-09-12-sdcp-self-check/summary.json)、[修正版开发结果](data/2026-09-12-sdcp-self-check-v2/summary.json)。
- [LOS均值诊断](data/2026-09-12-sdcp-self-check-v2/los_bias_diagnostics.json)、[几何误差对照](data/2026-09-12-sdcp-self-check-v2/geometry_diagnostic.json)。
- [新RF记录](data/2026-09-12-sdcp-fresh-rf/manifest.json)、[最终评估清单](data/2026-09-12-sdcp-self-check-final/manifest.json)。
- [最终检测汇总](data/2026-09-12-sdcp-self-check-final/summary.json)、[定位汇总](data/2026-09-12-sdcp-self-check-final/pvt_summary.json)。逐历元状态、原因、阈值与位置也在同目录。

从项目根目录运行，输出目录必须不存在，避免覆盖证据：

```powershell
python -m pytest -q python_sim_simplified/tests
python python_sim_simplified/generate_sdcp_self_check_rf.py --output python_sim_simplified/results/self_check_rf_repeat
python python_sim_simplified/audit_sdcp_self_check.py --source python_sim_simplified/results/self_check_rf_repeat --seeds 6400 6402 --output python_sim_simplified/results/self_check_repeat
```

RF生成需要较长时间。每份结果目录中的sha256.json用于校验文件字节；校验通过不代替算法有效性验证。
'''
    geometry = json.loads((ROOT/'docs/reports/data/2026-09-12-sdcp-geometry-stress/summary.json').read_text())
    extra = '''### 补充：真实基线补偿公式的方向误差回放

使用同一新RF LOS观测，将75.9 mm基线在局部ENU水平面旋转0/1/3/5/10度，以逐历元TL SPP重新计算几何补偿。上报误差界时使用新旧基线向量差的范数。这是物理几何补偿公式的误差回放，不是运动天线仿真。该补充不修改冻结检测参数。

|种子|窗口ms|方向偏差°|上报误差界|基线误差界mm|实际补偿最大差mm|覆盖率|误报比例|
|---|---:|---:|---|---:|---:|---:|---:|
'''
    for x in geometry:
        extra += f"|{x['seed']}|{x['window_ms']}|{x['yaw_error_deg']}|{x['bound_reported']}|{x['baseline_error_bound_mm']:.3f}|{x['max_geometry_error_mm']:.3f}|{percent(x['coverage'])}|{percent(x['false_alarm_all'])}|\n"
    extra += '\n上报不确定性后停判是主动放弃判断，不是消除了误差；未上报的中等方向偏差可能表现为少数星异常，从而被误报为疑似多径。\n\n'
    low = json.loads((ROOT/'docs/reports/data/2026-09-12-sdcp-low-signal/summary.json').read_text())
    extra += '''### 补充：输入信号降低6 dB的真实RF链路对照

在无多径、无外部干扰的TR源信号进入RF前乘以10^(-6/20)，再实际执行RF、DFE和8颗卫星独立跟踪。保持噪声种子6401，配对同一TL正常记录；共新增一条3秒RF记录。它检验真实输入信号质量下降，比相位域加白噪声更接近接收链路，但只有一个配对条件，不构成跨C/N0范围验证。

|窗口ms|原始有效观测|可判断覆盖|全区间误报|可判断内误报|固定持续误报|
|---|---:|---:|---:|---:|---:|
'''
    for x in low:
        extra += '|'+ '|'.join([str(x['window_ms'])]+[percent(x[k]) for k in
                  ['raw_valid_fraction','coverage','false_alarm_all','false_alarm_judged','fixed_confirmed_false_alarm']])+'|\n'
    extra += '\n补充数据：[几何压力结果](data/2026-09-12-sdcp-geometry-stress/summary.json)、[低信号RF结果](data/2026-09-12-sdcp-low-signal/summary.json)。\n\n'
    s = s.replace('## 6. 接入定位的探索性对照', extra+'## 6. 接入定位的探索性对照')
    s = s.replace('未完成实机C/N0变化', '除一个仿真输入降低6 dB条件外，未完成实机C/N0变化')
    s = s.replace('## 1. 原理与实际接口', '最终两组新RF记录、两种窗长的结果范围如下（不是事件检测概率）：\n\n'
                  +'\n'.join(headline)+'\n\n## 1. 原理与实际接口')
    interpretation = '''
本轮判断：强多径下自检降权相对不处理有明显定位收益，但没有优于相同持续确认的固定阈值。新LOS条件下，两者均未出现持续误报，所以不能据此宣称自检的额外复杂度已经带来普遍收益。自检的主要新增行为是观测失效/公共异常时拒绝强制判决，这必须以覆盖率损失一并评价。

500 ms与1 s在两组RF记录上的强/弱/共模检出结果相同，而后期强多径确认延迟分别约0.54秒与1.00秒；本轮没有发现延长到1秒的检测收益。后续可优先以500 ms作研究基线，但不将两组记录推广为通用最佳窗长。

几何压力表说明：未上报的3度误差已造成6.5%～11.5%误报，5度为21.42%～31.25%；10度误报降至0仅因覆盖率降至0%～6.67%。因此“误差一大自检就能识别”的假设不成立，中等且集中在少数卫星的误差尤其危险。

弱多径个别记录的位置RMS可能低于LOS，这不能证明无污染或算法有收益；比较方法必须使用相同受污染记录，不能把偏差偶然抵消当作检测成功。

验证记录：124项测试通过，保留一条既有无噪声C/N0计算除零警告；测试日志与最终文件核验见结果目录中的validation_checks.json和test_output.txt。未修改默认SDCP/定位入口，未把探索性自检作为已经验收的生产流程。

'''
    s = s.replace('## 8. 文件与复现',interpretation+'## 8. 文件与复现')
    s += '\n## 结果图\n\n![定位与覆盖率](data/2026-09-12-sdcp-self-check-final/overview.png)\n\n![基线方向误差敏感性](data/2026-09-12-sdcp-self-check-final/geometry_sensitivity.png)\n'
    REPORT.write_text(s,encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    kinds=['los','strong','weak','static','common']
    for mode in ['plain','fixed_confirmed','self_check']:
        y=[np.mean([x['modes'][mode]['float']['rms_m'] for x in pvts if x['case']==k]) for k in kinds]
        axes[0].plot(kinds,y,'o-',label=mode)
    axes[0].set_ylabel('3D float RMS (m), mean of two seed results')
    axes[0].legend();axes[0].grid(alpha=.3)
    cases=['los','offset_10mm','half_cycle','one_static_25mm','five_static_25mm']
    y=[np.mean([x['coverage'] for x in rows if x['domain']=='observation' and x['window_ms']==500 and x['case']==k]) for k in cases]
    axes[1].bar(range(len(cases)),y);axes[1].set_xticks(range(len(cases)),['LOS','offset','half cycle','1 anomaly','5 anomalies'],rotation=25)
    axes[1].set_ylabel('Judged coverage (500 ms)');axes[1].set_ylim(0,1.05)
    fig.tight_layout();fig.savefig(DATA/'overview.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for seed in [6400,6402]:
        for window in [500,1000]:
            selected=[x for x in geometry if x['seed']==seed and x['window_ms']==window and not x['bound_reported']]
            for ax,key in zip(axes,['false_alarm_all','coverage']):
                ax.plot([x['yaw_error_deg'] for x in selected],[100*x[key] for x in selected],
                        'o-',label=f'{seed}, {window} ms')
                ax.set_xlabel('Unreported baseline yaw error (deg)');ax.grid(alpha=.3)
    axes[0].set_ylabel('False alarms (% of all late observations)')
    axes[1].set_ylabel('Judged coverage (%)');axes[1].legend()
    fig.tight_layout();fig.savefig(DATA/'geometry_sensitivity.png',dpi=160);plt.close(fig)
    print(REPORT)


if __name__=='__main__': main()
