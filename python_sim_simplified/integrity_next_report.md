# 惯性完整性下一批 151–175：25 轮研究及失败边界

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
| 151 | Unknown initial origin is unobservable; propagate an unbounded position set. | unknown_origin | 4.7%→0.0% | 85→0 | 0/0 |
| 152 | Estimate bias from a separate stationary bench record, not runtime truth. | bounded | 4.7%→4.7% | 0→0 | 0/0 |
| 153 | Try empirical calibration residual maxima as an error envelope. | empirical_only | 4.7%→10.0% | 0→0 | 0/0 |
| 154 | Separate empirical or unbounded Gaussian support from hard bounded support. | gaussian | 10.0%→0.0% | 0→0 | 0/0 |
| 155 | Identify a bias interval by intersecting separately measured bench intervals with declared fixture error. | bounded | 10.0%→11.3% | 0→0 | 0/0 |
| 156 | Allow time-correlated bias drift rather than treating it as independent white noise. | drift | 11.3%→8.7% | 457→0 | 0/0 |
| 157 | Use a held-out bench sequence as a falsification check for calibration validity. | holdout_fault | 11.3%→0.0% | 596→0 | 0/0 |
| 158 | Compensate measured temperature with an externally calibrated coefficient interval. | temperature | 11.3%→11.1% | 444→0 | 0/0 |
| 159 | Use independent bounded velocity observations to limit integrated drift. | velocity | 11.3%→61.4% | 0→0 | 0/0 |
| 160 | Permit zero-velocity updates only from an independent stationary fixture, not quiet accelerometers. | stationary | 11.3%→100.0% | 0→0 | 0/0 |
| 161 | Transport an old initial position measurement to the integration epoch. | initial_delay | 11.3%→11.3% | 52→0 | 0/0 |
| 162 | Propagate initial timestamp uncertainty and velocity support into position support. | jitter | 11.3%→10.0% | 0→0 | 0/0 |
| 163 | Stop using a calibration after its declared validity interval expires. | expired | 11.3%→6.8% | 0→0 | 0/0 |
| 164 | Transport delayed independent position measurements with velocity and acceleration support. | delayed_anchor | 61.4%→100.0% | 0→0 | 0/0 |
| 165 | Intersect constant-bias support with independent velocity increments using time-correlated noise support. | bias_identify | 61.4%→61.4% | 0→0 | 0/0 |
| 166 | Account for three-axis gravity leakage using the sine Lipschitz bound. | tilt3d | 11.3%→1.3% | 597→0 | 0/0 |
| 167 | Use an independent attitude observation and its interval to remove nominal gravity leakage. | attitude | 1.3%→7.5% | 0→0 | 0/0 |
| 168 | Propagate time-correlated gyro error into attitude and gravity support. | gyro | 7.5%→5.2% | 544→0 | 0/0 |
| 169 | Transform the body-center position to the time-varying antenna phase center. | lever | 11.3%→11.3% | 0→0 | 0/0 |
| 170 | Include uncertain antenna/body lever geometry in the output-centered error set. | lever_uncertain | 11.3%→11.0% | 0→0 | 0/0 |
| 171 | Bound multiplicative scale error from measured acceleration rather than true acceleration. | scale | 11.3%→6.3% | 0→0 | 0/0 |
| 172 | Invalidate integrated inertial support after a flagged clipped sample with unknown missing acceleration. | clipping | 11.3%→11.3% | 496→0 | 0/0 |
| 173 | Treat contradictory anchor intersections as an integrity inconsistency, not an absolute-valued radius. | conflicting_anchor | 62.4%→6.7% | 20→0 | 0/0 |
| 174 | Allow a fresh absolute observation to provide a single-epoch position when inertial support is unavailable. | anchor_recovery | 0.0%→5.0% | 0→0 | 0/0 |
| 175 | Reject position certificates whose provenance depends on the attacked GNSS stream. | circular_trust | 3.3%→11.3% | 0→0 | 0/0 |

## 复现与验收

使用D:/Python/Python311/python.exe，工作目录python_sim_simplified；OPENBLAS_NUM_THREADS=1、OMP_NUM_THREADS=1。

    python -m pytest tests/test_integrity_next.py -q
    python integrity_next_validate.py
    python integrity_next_research.py --replay 151
    python integrity_next_research.py --replay 175
    python integrity_next_report.py

8项针对性测试通过；25轮全部逐一重放一致。延迟历史积分测试覆盖脉冲、三轴时变输入、历史不足和历史无硬支持。integrity_validation.json保存纠错后冻结哈希、新种子逐场景指标及反例，NPZ保存位置、真值、保护界、正常掩码；验证/回放不计研究轮次。不要再次运行integrity_next_steps.py覆盖历史，它只用于单次创建实验。

最终结论仍是：有独立且真实有效的信息时，部分连续性可以恢复；没有证据时只可报告不可用，不能默认存在证书。下一步需要真实独立台架、时间同步、来源数据建立适用域，并扩展完整旋转运动模型。没有这些证据，不能宣称解决绝对欺骗或满足长期全历元20米目标。
