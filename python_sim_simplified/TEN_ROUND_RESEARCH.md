# 十轮抗欺骗导航研究记录

十轮主实验已完成：**832 组主结果，加 384 组终轮配对基线，共 1216 组策略运行**。第 1–8 轮用于开发和消融，第 9 轮压力测试，第 10 轮冻结候选后的独立留出验证。不是十次重复运行同一组参数。所有失败轮次、原始分数、策略参数和逐轨迹结果均保留。

主要进展是把候选证据期间的更新管理、更新前创新检验和有效加速度误差估计接到一起。最终留出中，3 PRN 快速接管的前 10 s 最大水平误差由基线 **40.64 m 降到 1.78 m**。但差异偏差叠加平滑伪加速度时 **0/48 隔离、48/48 在仍允许更新时超过 30 m**。因此结论是找到了一种有短时收益的研究候选，**没有完成可商用系统，也没有替换默认接收机算法**。

## 数据、对照与计数

| 用途 | 种子 | 基础轨迹数 | 用法 |
|---|---|---:|---|
| 正常创新标定 | 750001–750008 | 16 | 每种子静止/转弯；仅正常输入 |
| 第 1–8 轮开发 | 740001、740010、740024、720004 | 8 | 每轨迹六场景；48 组/轮；含此前已知坏例 |
| 第 9 轮压力测试 | 760001–760004 | 8 | 每轨迹八场景，共 64 组 |
| 第 10 轮最终留出 | 770001–770024 | 48 | 每轨迹八场景，共 384 组；另配对运行基线 384 组 |

同种子的两个运动共享天空，每个基础轨迹的场景共享 IMU、初值和正常 PVT 噪声。因此不能把 1216 组结果当成 1216 个独立样本。最终留出在第 8 轮候选冻结、第 9 轮确认弱点后生成，未根据最终结果再次调参。

复用 `navigation_inputs`、`carrier_stream`、`spatial_scores`。原 50 Hz IMU 传播不变，在每个 10 Hz PVT 时刻汇总惯性增量；无门控对照中，与原 50 Hz 滤波的观测时刻位置相差小于 1e−10 m。本轮错误边界计数在 **10 Hz 采样点**计算，不声称捕获两点之间的全部峰值。

共同条件：四角 75.9×160.9 mm 手机阵列，四路同步载波，8 颗分散卫星，100 Hz 载波、100 ms 窗口，80 s 轨迹，20 s 开始攻击，撤离组 30 s 恢复正常 PVT。IMU 的初始倾角、零偏、噪声、杆臂及 PVT 噪声沿用 `NAVIGATION_FUSION.md`。可信初始化是外部前提，不能把由真值加随机初始误差生成的初值解释为已解决实际初始化。

**这是合成观测与人为 PVT 拉偏实验。** 尚未经过 RF 捕获、PLL/DLL、伪距和 Doppler 解算形成攻击后的 PVT；三颗被接管卫星是否能制造这里指定的 PVT，未在本轮验证。GNSS PVT 假设在其时间标签处即可用，未加实际接收机输出延迟。

## 政策和数学模型

空间确认始终采用前轮冻结的阈值 $\tau=0.012710679080318886$ 和连续三窗确认。候选暂停参数 $q$ 控制恢复前需要的连续安静窗数：

- $q=0$：确认前继续更新。
- $q>0$：第一窗低分证据出现就暂停更新，连续 $q$ 窗安静后解除尚未确认的暂停。
- 三窗确认后进入导航隔离锁存，安静窗口不能解除已确认隔离。

这是核心研究策略，不是为异常文件、缺字段或空输入增加的防御代码。输入假定为规则、有效、同频同步观测；无宽泛异常捕获或静默回退。

六状态预测沿用原 $[p,v]$ 模型。新增九状态 $x=[p,v,b_e]$，其中 $b_e$ 是**导航系有效加速度误差**，可以吸收部分倾角、零偏和建模残差，但不是分别估出了真实加速度计零偏、陀螺零偏与姿态：

$$
F=\begin{bmatrix}I&\Delta t I&-\frac12\Delta t^2I\\0&I&-\Delta t I\\0&0&I\end{bmatrix},\quad
x_k^-=Fx_{k-1}^++\begin{bmatrix}\Delta p_k^I-v_{k-1}^I\Delta t\\\Delta v_k^I\\0\end{bmatrix},\quad
z_k=Hx_k+\epsilon_k,\quad H=[I_6\;0].
$$

初始 $b_e=0$、每轴标准差 0.05 m/s²；连续随机游走强度为 $(0.0005)^2$，过程协方差保留积分产生的交叉项：

$$Q_b=q_b\begin{bmatrix}
\Delta t^5I/20&\Delta t^4I/8&-\Delta t^3I/6\\
\Delta t^4I/8&\Delta t^3I/3&-\Delta t^2I/2\\
-\Delta t^3I/6&-\Delta t^2I/2&\Delta t I
\end{bmatrix}.$$

原白加速度过程噪声和 Joseph 协方差更新保留。这里仍是条件滤波协方差，**不是完整性保护界限**。没有据此自动认证 20 m、30 m 或 50 m 输出边界。

更新前计算创新 $r_k=z_k-Hx_k^-$。三个可选监测量为水平位置创新范数、水平速度创新范数，以及完整十个历元的平均水平速度创新范数：

$$V_k=\|r_{v,xy,k}\|,\quad P_k=\|r_{p,xy,k}\|,\quad
M_k=\left\|\frac1{10}\sum_{i=k-9}^{k}r_{v,xy,i}\right\|.$$

超过对应门限就在**当前更新前**锁存隔离，当前异常观测不会先进入滤波。启动不足十个历元时不使用均值监测。标定用正常轨迹、保留空间暂停但关闭创新隔离，避免用门限剔除异常样本后再自我标定。

第 5 轮六状态正常标定得到 $V_{max}=1.125528758$ m/s、$P_{max}=12.020187909$ m、$M_{max}=0.475565856$ m/s。第 7 轮九状态的速度最大值降到 0.961936456 m/s，但开发组出现一次误隔离。第 8 轮取六/九状态正常包络中较大的速度与均值门限，并启用均值监测；没有根据攻击样本设定门限。候选最终使用 $q=1$、九状态、速度和均值监测，不使用位置创新隔离。

这种“有限正常样本最大值”门限不是统计安全保证。开发选择也存在对开发样本的适应，必须看最后留出结果。

## 十轮记录

表中误差都是攻击起点至第 10 s 的最大水平误差；每列取该组最坏轨迹，包含误报提前隔离造成的坏结果。正常拒绝更新时间以完整 80 s 为分母折算每小时，**不是实际进行了一小时连续实测，也不是导航输出停机时间**。

| 轮次 | 假设与变化 | 正常确认误隔离 | 正常拒绝更新折算 s/h | 3 PRN 快速接管误差 m | 判断 |
|---|---|---:|---:|---:|---|
| 1 | 复现三窗、确认前仍更新的六状态基线 | 0/8 | 0 | 46.806 | 复现污染失败，保留对照 |
| 2 | 第一窗证据即暂停，一窗安静即恢复 | 0/8 | 3.375 | 42.079 | 有收益，但断续恢复仍污染 |
| 3 | 三窗安静才恢复 | 0/8 | 9.000 | 26.164 | 污染减少，仍超过 20 m |
| 4 | 十窗安静才恢复 | 0/8 | 24.750 | 10.128 | 正常中断代价过高，不保留长暂停 |
| 5 | 回到短暂停，增加更新前速度创新隔离 | 0/8 | 3.375 | 4.912 | 保留；只证明速度突变有证据 |
| 6 | 额外加入位置创新隔离 | 2/8 | 902.250 | 25.397 | 正常误隔离导致退步，拒绝此组合 |
| 7 | 去掉位置门限，加入有效加速度估计并重新标定 | 1/8 | 160.875 | 1.766 | 估计器有益，速度门限过紧 |
| 8 | 保留估计器，取较大正常创新包络，加一秒均值检验 | 0/8 | 3.375 | 1.766 | 冻结为后续候选；无商用放行 |
| 9 | 新种子；加入差异偏差+慢拖/平滑伪加速度 | 0/8 | 1.125 | 1.228 | 快速攻击收益保留，平滑攻击严重失败 |
| 10 | 冻结第 8 轮策略，全新留出并配对基线 | 0/48 | 1.781 | 1.781 | 重复确认收益和边界，不能商用放行 |

第 9、10 轮使用不同数据，不能把其数值变化当成又一次参数改进。第 9 轮平滑伪加速度组 0/8 隔离，8/8 在允许更新期间超过 30 m；慢拖组仅 1/8 隔离。这促使第 10 轮原样保留该压力场景，没有把它移出测试集。

![开发轮次的误差与正常更新代价](results/ten-rounds/rounds.png)

## 最终独立留出：同数据配对

本节基线为第 1 轮三窗、确认前继续更新的六状态模型，不是最初单窗模型。两种策略输入逐样本相同。

| 场景，各 48 条 | 基线前 10 s 最大误差 m | 候选前 10 s 最大误差 m | 候选确认隔离/48 | 候选允许更新时超过 30 m 的轨迹 |
|---|---:|---:|---:|---:|
| 正常，误差区间同取 20–30 s | 1.274 | 0.951 | 0 | 0 |
| 全部快速接管 | 10.799 | 1.781 | 48 | 0 |
| 3 PRN 快速接管 | 40.636 | 1.781 | 48 | 0 |
| 空间证据完整的 0.5 m/s 慢拖 | 5.798 | 1.828 | 48 | 0 |
| 快速接管后撤离 | 10.799 | 1.781 | 48 | 0 |
| 差异偏差 + 5 m/s 快速拉偏 | 49.353 | 1.781 | 48 | 0 |
| 差异偏差 + 0.5 m/s 慢拖 | 5.875 | 5.516 | 18 | 26 |
| 差异偏差 + 平滑伪加速度 | 3.402 | 3.146 | 0 | 48 |

快速拉偏组在 20 s 的 PVT 速度中直接加入 5 m/s 突变。候选能在该观测更新前拒绝，表中隔离延迟因此为 **0 个额外 10 Hz 历元**；这不是物理接收机零延迟，也不是证明多天线解决了差异偏差。此时主要证据来自导航速度创新，空间检测仍可能失效。空间证据完整的慢拖则最迟到 20.3 s 的 PVT 时刻锁存隔离。

导航创新异常只能证明观测与预测不一致，尚不能区分欺骗、接收机异常或惯性故障。这里“隔离成功”指测试中拒绝了攻击观测，不代表已完成攻击来源归因。

候选正常 GNSS 更新比例为 **99.9505%**，确认误隔离 0/48，但有 **18 次候选暂停中断**，总拒绝更新时间 1.9 s。按 1.067 h 总暴露时间折算约 16.875 次/h、1.781 s/h。若把候选暂停视作一次用户可见降级事件，频次就不满足每小时一次的目标；不能只引用总时长较短。更新比例也不等于经过完整性认证的导航可用率。

只有 24 个新天空种子、每条 80 s，且环境模型较简单，零确认误报不能建立商用误报概率界限。没有通过统计置信度或长时间工况验收。

### 长时间边界

全部/3 PRN/偏差快速接管组，候选前 30 s 最坏误差 **11.87 m**，前 60 s 最坏 **60.88 m**；空间证据完整的慢拖组前 60 s 最坏 **65.03 m**。因此不能把十秒收益外推为分钟级导航能力。

差异偏差+平滑伪加速度组前 30 s 最坏 **22.94 m**，前 60 s 最坏 **90.55 m**，且始终没有确认隔离。差异偏差+慢拖组部分检测后仍有污染和惯性漂移，前 60 s 最坏 **86.62 m**。研究曲线继续积分以暴露失败，不表示产品应继续输出有效位置。

## 为什么平滑攻击仍会失败

第 9/10 轮新增两种压力条件，空间载波都沿用每通道×PRN 固定标准差 20 mm 的偏差组。令 $u=\max(t-20,0)$，PVT 施加：

$$\text{慢拖：}\ d_x=0.5u,\ \dot d_x=0.5;\qquad
\text{平滑伪加速度：}\ d_x=0.025u^2,\ \dot d_x=0.05u.$$

后者在攻击起点的位置和速度都连续，伪加速度为 0.05 m/s²。若 GNSS 位置变成 $p_G=p+d$，而惯性侧有效加速度为 $a_I=\ddot p+b_e$，则滤波用被污染 GNSS 对齐时可能估出：

$$\hat b_e\approx a_I-\ddot p_G=b_e-\ddot d.$$

因此在没有独立约束时，部分平滑拉偏可以与有效惯性误差混淆。估计器改善了正常和短时惯性传播，却也可能使被污染 GNSS 与预测“更一致”。

对预先选取的首个压力种子 760001、静止轨迹做正常/受攻配对：最终有效 x 加速度误差估计由 **0.07738** 变为 **0.03445 m/s²**，相差 **−0.04294 m/s²**；实际位置误差达 **89.88 m**，没有确认隔离。差值接近施加的 0.05 m/s² 量级，但不等于精确估出了伪加速度，其余差值受时变惯性误差和滤波动态影响。

![平滑攻击下的误差吸收反例](results/ten-rounds/absorption.png)

该反例不能证明所有融合算法都无解，也不能证明此输入一定来自可实现的 RF 攻击；它证明**本轮观测模型与策略尚不足以辨别这类污染**。仅靠创新较小、残差较小或偏差估计收敛，不能认证位置可信。

## 代码与复现

- `navigation_research.py`：候选暂停、六/九状态融合、更新前创新监测。消费者不接收真值、攻击起止或场景标签。
- `audit_navigation_iterations.py`：显式生成不同数据集、正常标定、逐轮冻结与评估。
- `summarize_navigation_iterations.py`：终轮基线对照、失败诊断、图片与数据哈希。
- `audit_navigation_fusion.py`：仅抽出可复用的合成输入函数，原实验入口保留。

在本目录执行，输出目录用 `results/ten-rounds`；重跑会更新该实验目录：

```powershell
python audit_navigation_iterations.py --output results/ten-rounds --split calibration --prepare
python audit_navigation_iterations.py --output results/ten-rounds --split development --prepare
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 1
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 2 --quiet-windows 1
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 3 --quiet-windows 3
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 4 --quiet-windows 10
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 5 --quiet-windows 1 --velocity-gate
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 6 --quiet-windows 1 --velocity-gate --position-gate
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 7 --quiet-windows 1 --velocity-gate --estimate-acceleration
```

第 8 轮提案的确定性生成代码（在 Python 中执行）：

```python
import json
from pathlib import Path
root = Path('results/ten-rounds')
a = json.loads((root/'round05/frozen.json').read_text())['policy']
b = json.loads((root/'round07/frozen.json').read_text())['policy']
b['velocity_limit'] = max(a['velocity_limit'], b['velocity_limit'])
b['mean_velocity_limit'] = max(a['mean_velocity_limit'], b['mean_velocity_limit'])
b['mean_velocity_gate'] = True
(root/'round08-proposal.json').write_text(json.dumps({'policy': b}, indent=2), encoding='utf-8')
```

继续执行：

```powershell
python audit_navigation_iterations.py --output results/ten-rounds --split development --round 8 --policy results/ten-rounds/round08-proposal.json
python audit_navigation_iterations.py --output results/ten-rounds --split stress --prepare
python audit_navigation_iterations.py --output results/ten-rounds --split stress --round 9 --policy results/ten-rounds/round08/frozen.json
python audit_navigation_iterations.py --output results/ten-rounds --split holdout --prepare
python audit_navigation_iterations.py --output results/ten-rounds --split holdout --round 10 --policy results/ten-rounds/round08/frozen.json
python summarize_navigation_iterations.py --output results/ten-rounds
python -m pytest -q
```

每轮目录包含 `frozen.json`、`rows.json` 和 `summary.json`。冻结文件记录种子、策略、源码 SHA-256；总分析记录 80 份数据文件哈希。第 10 轮另有 `baseline_rows.json`/`baseline_summary.json`，诊断保留 JSON 和完整状态 NPZ；不是仅保存汇总数字。

新增五项测试覆盖候选暂停与确认区分、异常更新前隔离、因果前缀一致性、未准入与隔离不自动恢复、有效加速度误差估计及完整均值窗口。验证结果见本报告末尾记录。

最终全量回归：**236 passed，95.22 s**；保留已有零噪声 C/N0 除零警告 1 项。新增入口通过编译检查，10 轮源码清单、分析脚本哈希、80 份数据哈希和 1216 行结果计数核对通过，两个生成图已做视觉检查。

## 保留什么，下一步优先做什么

保留第 8 轮组合为显式研究候选，保留第 1 轮和失败轮次作为对照。默认接收机与原研究入口没有切换到这个候选；本轮也没有把条件协方差包装成可信输出边界。

下一步优先研究**平滑拉偏与惯性误差的可区分性**，以及能独立验证的输出失效边界。需要检验真实伪距/Doppler、同步阵列时变相位与姿态约束是否提供额外信息，而不是继续靠压低创新门限把漏检换成误报。候选暂停的频次与再准入也需单独验收。全姿态/传感器零偏误差状态、手持/车内遮挡、多径、冷启动攻击、真实 RF 接管、台架和实测标定仍未完成。
