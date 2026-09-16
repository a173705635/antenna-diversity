"""Render closed-record numerical review and Chinese report, no extra iteration."""
import json
from pathlib import Path
from integrity_next_research import replay, execute
from integrity_next_validate import counterexamples
import hashlib

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results/research-next-hundred'

def main():
    table=[]
    for n in range(151,176):
        folder=OUT/f'iteration{n}'; record=json.loads((folder/'record.json').read_text(encoding='utf-8'))
        before=record['before'][record['target']]; after=record['after'][record['target']]
        review=(f"同输入目标场景 {record['target']}：正常率 {before['normal_output_rate']:.4%}→{after['normal_output_rate']:.4%}；"
                f"边界漏包 {before['bound_violations']}→{after['bound_violations']}；"
                f"改后正常≥30/50m={after['normal_ge30']}/{after['normal_ge50']}；全历元成功={after['full_success']}。")
        if before==after: review+=' 指标完全不变，不能宣称该机制在当前输入有收益。'
        review+=' '+record['conclusion']
        record['observed_review']=review
        if n==164: record['frozen_validation_review']='第一次冻结的延迟窗口加速度突变反例失败；175轮后实施历史积分公式纠错，原始失败快照保留。'
        (folder/'record.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8')
        (folder/'conclusion.md').write_text(review+'\n'+record.get('frozen_validation_review',''),encoding='utf-8')
        replay(n)
        table.append(f"| {n} | {record['hypothesis']} | {record['target']} | {before['normal_output_rate']:.1%}→{after['normal_output_rate']:.1%} | {before['bound_violations']}→{after['bound_violations']} | {after['normal_ge30']}/{after['normal_ge50']} |")
    validation=json.loads((OUT/'integrity_validation.json').read_text(encoding='utf-8'))
    original=(OUT/'integrity_first_freeze_source.py').read_text(encoding='utf-8')
    corrected=(ROOT/'integrity_next.py').read_text(encoding='utf-8')
    regressions={k:v for k,v in counterexamples(72101).items() if k.startswith('delayed_anchor')}
    correction=dict(iteration=175, extra_iteration=False, type='implementation formula correction after failed first freeze',
                    mechanism='Causal historical discrete acceleration integral and accumulated interval support; unsupported history gives no delayed constraint.',
                    comparison_seed=72101, before=execute(original,regressions),after=execute(corrected,regressions),
                    first_freeze_sha256=hashlib.sha256((OUT/'integrity_first_freeze_source.py').read_bytes()).hexdigest(),
                    corrected_source_sha256=hashlib.sha256((ROOT/'integrity_next.py').read_bytes()).hexdigest(),new_validation_seeds=validation['seeds'],
                    tests='8 focused tests passed, including pulse, varying 3-axis history, unavailable prefix and uncertified support.')
    (OUT/'integrity_implementation_correction.json').write_text(json.dumps(correction,indent=2),encoding='utf-8')
    report='''# 惯性完整性下一批 151–175：25 轮研究及失败边界

没有达到“手机长期数米、全历元≤20m”的目标。完成25次实际接收机源码机制改变、每轮相同输入前后对照及结论；种子、场景和验证不计迭代。第一次冻结发现延迟观测运输公式错误，已作为175轮实现纠错修正，未增加研究轮次。最终版仍**不是经过真实手机完整性验证的可部署方案**。

纠错后重新冻结使用72121、72127、72131、72139四个新种子，26场景×4=104个600秒案例，共62504历元。常规声明域正常率18.0596%，全历元成功8/104，均来自外部固定台架零速或独立位置+速度场景；正常≥30/50m均为0、边界漏包0。保护、缺失输出都算连续性失败。上一批120秒二维验证31.24%、8/84与本批场景和时长不同，不能直接比较算法升降。第一次冻结72101、72103、72109、72119及其失败均保存在integrity_first_freeze_*。

## 必须保留的反例

- 延迟锚点初次冻结失败：真实独立锚点落后3秒，窗口中有100m/s²的一秒脉冲而当前加速度为零。原代码以当前加速度界运输整个窗口，遗漏历史运动，出现80个正常≥30m及80个正常≥50m历元，最大正常误差50.074m、边界漏包133次。纠错改为 `p_k-p_(k-L)=L*v_k-Σ(j+0.5)*a_(k-L+1+j)`，支持为 `L*r_v+Σ(j+0.5)*r_a`，保留每个历史样本当时的校正和支持。历史不足或没有硬支持时，该延迟约束不可用。纠错后脉冲及三轴时变历史两个回归均0漏包、0正常≥30/50m，最大正常误差0.135m。1秒等间隔模型下公式是精确离散积分，不是当前加速度近似。
- 运行时漏报0.2m/s²偏置：53个正常≥30m、50个≥50m，最大正常误差756.195m。独立留出标定可发现标定时已存在的失效，发现不了其后未声明的跳变。
- 匀速移动却谎称固定台架零速：585个正常≥30m、575个≥50m，最大1198.974m。加速度为零不能证明速度为零。
- 假冒独立位置源：交集矛盾使正常率为零，有限候选保护界仍有30次漏包；消费者必须检查normal掩码，不能读取内部候选轨迹作为定位结果。

## 输入证据与模型边界

本批没有真实手机台架数据。training和holdout是独立随机流生成的“模拟仪器读数”，不是在线真值；接收机不读取truth、场景名称、攻击时刻或GNSS真伪标签。误差包络来自固定、显式的模拟域声明，绝不取某次模拟误差的绝对值。它仍然不是硬件证书。real_phone_unavailable场景没有绝对初值及硬包络，正常率严格为零。

经验最大残差只用于估计，不能证明未来硬界。高斯分布无有限硬支撑，本模型不把置信区间冒充硬保护级。声明有界噪声±0.002m/s²、台架参考±0.001m/s²时，使用各条测量区间交集识别恒定偏置区间；参考误差、噪声和稳定性都是外部假设。留出数据只能证伪，不能证明所有未见环境。温度系数界、漂移率界、姿态界、杆臂界、来源和到期时间也均是显式输入。

速度每10秒±0.01m/s、位置每20秒±1m是模拟外部仪器能力，不是手机现有传感器的承诺。151未知绝对初值保留无限位置集；175排除来源为GNSS的伪独立位置。来源字符串是测试场景元数据，未实现密码认证、硬件信任根或实际来源验证。双波束同源一致和IMU加速度仍不能消除绝对平移不可辨识性。

模型为三轴离散平动，逐轴sin姿态泄漏及Lipschitz界，长度600秒、步长1秒；它**不是**完整SO(3)旋转、地球转动、圆锥/划桨、真实手机动态IMU模型。杆臂是逐历元已测导航系向量，不包含陀螺到加计安装杆臂的角加速度项。初始时间运输假定初始延迟区间匀速。位置和速度盒忽略相关性，保守膨胀；165独立速度增量识别没有比台架交集更紧，结果零收益，未宣传改善。

## 逐轮同输入结果

每轮开发seed固定1510，对全部26场景测试。表格只显示该轮目标场景；不是不同场景间横向排名。每个iteration目录保存proposal、before.py、after.py、record及conclusion，保留失败。原始175源码快照不修改；当前选定实现的历史积分纠错、哈希和新冻结单独记录在integrity_implementation_correction.json。

| 轮次 | 实际模型/估计机制 | 目标场景 | 正常率前→后 | 边界漏包前→后 | 后正常≥30/50m |
|---|---|---|---|---|---|
'''+ '\n'.join(table)+'''

## 复现与验收

使用D:/Python/Python311/python.exe，工作目录python_sim_simplified；OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

    python -m pytest tests/test_integrity_next.py -q
    python integrity_next_validate.py
    python integrity_next_research.py --replay 151
    python integrity_next_research.py --replay 175
    python integrity_next_report.py

8项针对性测试通过；25轮全部逐一重放一致。延迟历史积分测试覆盖脉冲、三轴时变输入、历史不足和历史无硬支持。integrity_validation.json保存纠错后冻结哈希、新种子逐场景指标及反例，NPZ保存位置、真值、保护界、正常掩码；验证/回放不计研究轮次。不要再次运行integrity_next_steps.py覆盖历史，它只用于单次创建实验。

最终结论仍是：有独立且真实有效的信息时，部分连续性可以恢复；没有证据时只可报告不可用，不能默认存在证书。下一步需要真实独立台架、时间同步、来源数据建立适用域，并扩展完整旋转运动模型。没有这些证据，不能宣称解决绝对欺骗或满足长期全历元20米目标。
'''
    (ROOT/'integrity_next_report.md').write_text(report,encoding='utf-8')
    (OUT/'integrity_summary.json').write_text(json.dumps(dict(iterations=list(range(151,176)),closed=25,source_sha256=validation['source_sha256'],aggregate=validation['aggregate'],counterexamples=validation['counterexamples'],goal_met=False,hardware_validation=False,known_delayed_transport_defect=False,first_freeze_failed_delayed_transport=True),indent=2),encoding='utf-8')

if __name__=='__main__': main()
