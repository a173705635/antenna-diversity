# 惯性完整性研究 081–100：结果与条件

结论：完成20次独立源码演进及同输入前后对照，但没有解决“手机长期数米、全历元≤20m”的目标。最终原型只证明了**在显式、有界且真实有效的输入前提下**可以计算保护界，超界的历元不再宣称正常定位。模型未连接RF、双波束或真实手机IMU；不能把本批数值当作抗欺骗定位性能。

开发输入固定seed=810，19场景，每场景120秒、241历元。每次先写proposal、实际改变receiver源码，再对同一观测比较旧/新版本、记录结果并决定下一步；下一轮的`--after-review`写入上一轮的conclusion。源码快照和逐历元traces均保留。种子、重放和测试不计研究次数。

## 逐轮结果

表中正常率、边界漏包、≥30/50米均是该轮指定目标场景，不是跨轮不同输入的横向排名。≤20米连续成功要求全部历元均正常且误差≤20米，保护/缺失历元按不成功计。

| 轮次 | 实际机制改动 | 同输入目标 | 正常输出率前→后 | 边界漏包前→后 | 改后正常≥30 / ≥50m |
|---|---|---|---|---|---|
| 081 | Local inertial innovation admission | step | 100.0 → 100.0% | 221 → 200 | 135 / 105 |
| 082 | Independent inertial backbone | creep | 100.0 → 100.0% | 235 → 235 | 153 / 110 |
| 083 | Bounded acceleration reachable set | normal | 100.0 → 100.0% | 161 → 0 | 141 / 110 |
| 084 | Initial velocity support propagation | velocity | 100.0 → 100.0% | 114 → 0 | 169 / 139 |
| 085 | Gravity leakage support | tilt | 100.0 → 100.0% | 201 → 0 | 190 / 175 |
| 086 | Gyro driven attitude growth | gyro | 100.0 → 100.0% | 184 → 0 | 172 / 156 |
| 087 | Preserve anchor across a GNSS outage | outage | 100.0 → 100.0% | 214 → 0 | 141 / 110 |
| 088 | Unobservable absolute initialization | unknown | 100.0 → 100.0% | 0 → 0 | 141 / 110 |
| 089 | Output centered protection level | tilt | 100.0 → 10.8% | 0 → 0 | 0 / 0 |
| 090 | Independent position certificate intersection | auth | 19.5 → 46.1% | 0 → 0 | 0 / 0 |
| 091 | Independent velocity certificate | zero_velocity | 19.5 → 86.7% | 0 → 0 | 0 / 0 |
| 092 | Explicit stationary bias calibration | bias_calibration | 19.5 → 63.9% | 0 → 0 | 0 / 0 |
| 093 | Initial certificate time transport | timing | 21.6 → 19.5% | 0 → 0 | 0 / 0 |
| 094 | Antenna/body lever arm frame | lever | 20.7 → 19.5% | 0 → 0 | 0 / 0 |
| 095 | Certified vehicle nonholonomic velocity | mounted | 19.5 → 22.8% | 0 → 0 | 0 / 0 |
| 096 | Independent attitude calibration | calibrated | 10.8 → 19.1% | 0 → 0 | 0 / 0 |
| 097 | Union of ambiguous absolute hypotheses | ambiguous | 19.5 → 0.0% | 0 → 0 | 0 / 0 |
| 098 | Independent hypothesis elimination | resolved | 0.0 → 21.6% | 0 → 0 | 0 / 0 |
| 099 | Bias drift support | drift | 18.3 → 17.4% | 154 → 0 | 0 / 0 |
| 100 | Minimax center for protected output | normal | 19.5 → 26.6% | 0 → 0 | 0 / 0 |

081门限只挡突跳，仍有长期漂移；082隔离GNSS对惯性骨架的反馈后，全历元成功从15/19降至0/19，是明显退步，不能称为解决方案。083–086建立加计、初速、重力泄漏、陀螺误差传播，它们主要改进可解释性，完全没有把长期位置变准。087取消断弧重锚，outage最大潜在误差342.6→161.5m，仍失败。088名义误差未变，但补充同输入200m平移世界反例从181次漏包变为0；无界支持并不产生有限精度。

089正常≥30/50m错误变为0，但平均正常率仅17.65%，因此不能算定位成功。090独立位置交集和091独立速度观察恢复一部分可用性；091首次产生1/19全程成功。092偏置标定和096姿态标定均有收益，但证书不能凭GNSS一致性自行产生。093同步与094杆臂修正稍降名义正常率，因为之前偏差相消；093无GNSS补充对照消除了15次错误支持，不能为了指标好看保留错误坐标/时标。

095仅在显式“固定支架、直行、横向速度近零”证书有效时作用；手持case不启用。097保留两种相隔200m假设，输出率下降到0是不可辨识结果；098独立位置证书排除不兼容假设后恢复21.6%。099偏置漂移支持消除剩余154次漏包，正常率略降。100采用集合中心而非GNSS候选位置，整体正常率27.23→33.13%，全程成功1→2/19，但正常场景最大正常误差1.32→13.49m：可用率上升不等于达到数米精度。

## 冻结验证与反例

最终源码冻结后，用12081、14083、16087、18089四个新种子验证。19原场景加无GNSS、未知初值且真实轨迹平移200m，共84个场景、20244个历元。记录见`results/research-hundred/integrity_validation.json`。

- 正常率31.2389%；全历元≤20m仅8/84，全部来自独立位置+速度及独立零速场景。
- 正常输出≥30m/≥50m均为0，声明域内边界漏包为0；不能据有限样本估计航空级风险概率。
- 未知绝对初值、无真实GNSS时仍不可辨识，不允许由一致GNSS创造信任。
- 故意漏报0.2m/s²加速度误差：正常输出最大110.50m，31个正常历元≥30m、22个≥50m；边界233次漏包。
- 将手持错误声明为车载横向约束：正常最大46.28m，28个正常历元≥30m，237次边界漏包。

这些反例说明真实有效的边界与证书是必要条件。没有台架数据，当前取值不是经过标定的手机完整性参数。

## 数学条件与模型域

每轴使用显式位置盒和速度盒：p∈[p̂−r_p,p̂+r_p]，v∈[v̂−r_v,v̂+r_v]。假定每个区间内真实加速度误差满足逐轴上界a_r，递推为

    p̂_next = p̂ + v̂*h + a_meas*h²/2
    v̂_next = v̂ + a_meas*h
    r_p_next = r_p + r_v*h + a_r*h²/2
    r_v_next = r_v + a_r*h

本实验加速度支持为加计支持+g×初始倾角支持+g×陀螺支持×t+偏置变化率支持×t；使用离散右端点传播。连续形式对应r_p(t)≈r_p0+r_v0*t+(b_a+g*theta0)*t²/2+(g*b_g+b_dot)*t³/6。该式来自积分误差，并非协方差倍数或统计置信度。小角度重力泄漏在此被**定义为二维线性模型**；尚未验证完整三维姿态、手机旋转、地球自转、振动整流或动态杆臂。

对给定输出q，盒的最远欧氏距离为PL(q)=||r_p+|q−p̂|||₂；添加杆臂每轴不确定度。100令q=p̂，得到盒的最小最大误差中心。PL≤20m才标normal。`position`保存潜在内部轨迹供分析，`normal`为对外定位有效掩码，消费者不得忽略它。没有实现用户界面或设备输出链。

绝对证书采用盒交集；空交集使假设不可行，不能用GNSS重置可信边界。速度证书直接替换速度及其支持。多假设保留并集，使用覆盖各分支球的外包盒，保守但未优化相关性；只有独立绝对证书可淘汰分支。未知绝对初值用无界位置集。时间运输只覆盖初始证书固定延迟和已知速度支持；不是一般IMU/GNSS异步融合。杆臂为固定导航系二维偏移；车载横向约束仅针对无转弯的明确坐标系。

所有额外证书都是仿真观察输入，真实轨迹仅用于生成带已知误差的观察与评分，receiver不访问truth、运动名称或攻击时刻。初始位置±2m每轴、初始速度支持、加计±0.022m/s²每轴、采样噪声±0.002m/s²、各姿态/陀螺/偏置支持均是假设输入，不是手机规格。尤其倾角、陀螺和偏置漂移支持直接取模拟误差项的绝对值；这验证了条件数学模型，完全没有独立建立硬件误差包络。独立位置每20s±1m、速度/零速每5s±0.01m/s也不是已实现能力。它们用于回答“还需要什么信息”，不能默认为手机现有条件。

不可辨识反例：若只知道加速度，p(t)与p(t)+c具有同一IMU观测；攻击者可在两世界呈现同一GNSS序列。未知位置时，任何有限绝对误差声明都不能同时覆盖任意c。选择c=(200m,0)已超出20m目标。双波束同源一致无法解除这一平移自由度。

## 可复现

在`python_sim_simplified`目录，用`D:/Python/Python311/python.exe`，环境变量`OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。

    python -m pytest tests/test_integrity_hundred_inertial.py -q
    python integrity_hundred_inertial_validate.py
    python integrity_hundred_inertial_research.py 81 --replay
    python integrity_hundred_inertial_research.py 100 --replay

所有81–100轮均可用`--replay`只读重放旧/新源码对照。不要对现存结果再次运行无`--replay`的生成命令。最终7个针对性测试通过；测试不计研究轮次。100轮之后的选定源码清理只删除过时注释/无用状态，并将多假设全不可行情形显式标为无有效位置、非normal；历史源码快照未重写，清理后重新冻结验证。历史算法和核心导航文件未修改。
