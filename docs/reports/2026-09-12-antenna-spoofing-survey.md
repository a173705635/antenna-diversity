# 天线分集抗欺骗调研与首轮选型

调研日期：2026-09-12。采用原始论文、研究机构和公司公开资料；专利证明技术公开，不单独证明产品效果。本轮不沿极化分集推进。

## 1. 推荐什么，为什么

当前代码最合适的首轮路线是**双天线、多 PRN 载波相位差的空间一致性检测**。这里“最合适”指公开实测依据与当前接口匹配，不能理解为已经证明它是所有威胁下最强的方案。

真实卫星来自不同方向，其天线间传播距离差通常不同；同一个发射天线发送多颗伪造 PRN 时，这些信号具有共同空间响应。先测每颗信号的天线间相位差，再比较不同 PRN，比检查伪距是否自洽多一个空间证据。它检测共同来源，不能仅凭空间相近认定恶意，也不能认证未报警信号。

| 路线 | 原始公开证据 | 当前适用性与代价 |
|---|---|---|
| 双天线载波差/共同来源假设 | Cornell、UT Austin、Sea ID 的2014年研究有实时实船试验 | 两路载波输出可复用；首轮选择，但部分欺骗和混合阶段是边界 |
| 多阵元测向、姿态一致性及空间抑制 | DLR 的2014年小型阵列研究有现场转发信号试验 | 需要相关器空间观测和阵列响应；抑制前还须辨别可信信号，实施成本较高 |
| 双天线逐星认证与定位残差结合 | NovAtel US11906633B2 给出双天线实例及多指标流程 | 可作为部分欺骗后续候选，不能简化为“相位相同就删除” |
| 多层商业接收机防护 | Septentrio AIM+ 官方介绍含一致性检查、认证与多频多星座 | 证明商用品通常采用多层证据，未公开完整算法，不能直接复现其性能 |

## 2. 核心来源及可核实结论

**Psiaki 等，GNSS Spoofing Detection using Two-Antenna Differential Carrier Phase，ION GNSS+ 2014。** 原文有真实与欺骗两种模型的最大似然代价比较，处理共同线路偏差和整数模糊度；实船原型报告快速检测，但系统处理延迟可使响应达到6秒。原文也明确初始真假信号混合及少数信号受骗困难。不能把“亚秒算法响应”当作本项目端到端结果。[作者机构全文](https://radionavlab.ae.utexas.edu/images/stories/files/papers/Psiaki_WROD_2014.pdf)、[ION书目信息](https://www.ion.org/publications/abstract.cfm?articleID=12530)。

**Montgomery、Humphreys、Ledvina，Receiver-Autonomous Spoofing Detection，2009。** 公开摘要报告共振荡器双天线的实测，讨论载波多径、线路偏差与误报问题。本轮只用摘要支持该路线已有早期实验，不引用未读全文的性能数字。[ION原始摘要](https://www.ion.org/publications/abstract.cfm?articleID=8295)。

**NovAtel US11906633B2，System and method for navigation system spoofing detection using a plurality of antennas。** 授权文本公开日2024-02-20，发明人 Ali Broumandan、Thomas Taylor。说明书给出 OEM7720、15cm 基线例子。权利要求1组合载波双差、C/N0双差、阈值标签、定位及残差判别；本轮未实现这一整套流程。其“authentic”是专利算法标签，不可在我们的输出中照搬为安全认证。文本技术比较不构成自由实施结论。[专利全文与权利要求](https://patents.google.com/patent/US11906633B2/en)。

**DLR，Autonomous Spoofing Detection and Mitigation with a Miniaturized Adaptive Antenna Array，2014。** 公开现场相关器记录分析采用65mm阵元间距的小型阵列；强转发信号容易检测，弱转发信号有跟踪受扰、DOA不可用区间。更小孔径、互耦和方向图差异会降低测向质量，不能把理想阵列仿真当硬件效果。[DLR全文](https://elib.dlr.de/95121/1/GNSS14-0271.pdf)。

**D3 理论研究，Performance Analysis of the Dispersion of Double Differences Algorithm to Detect Single-Source GNSS Spoofing，2021。** 原始出版摘要说明门限、误检/漏检和C/N0关系是需要统计建模的问题；本轮未获取可完整核对的正文，因此不使用其公式或宣称复现D3。[出版DOI](https://doi.org/10.1109/TAES.2021.3061822)。

**Septentrio AIM+ 官方资料。** 描述启发式检查、多频多星座交叉检查、OSNMA及系统级辅助。未公开我们可复现的全部判决过程；其商业宣传不作为本项目定量验证。[官方技术介绍](https://www.septentrio.com/en/learn-more/advanced-positioning-technology/aim-anti-spoofing-protection)。

## 3. 首轮实现与文献的差别

实现采用圆统计分散度，不是2014年完整双假设最大似然判决。使用4π倍米制相位差/波长，消除整数与半周差异；不拟合真实天空几何，因此输入更少，也失去利用正常星座几何拒绝误报的能力。模型标定门限不具有D3理论门限的保证。

首轮固定8颗PRN的完整集合，100ms非重叠观察块，任何缺测块输出未知。不能通过挑选相似的两颗PRN就直接报警，否则正常卫星投影偶合会造成多重比较误报。后续若研究部分欺骗，必须明确子集搜索与整体误报控制。

双天线共同通道偏差可以被本统计量消除；方向/PRN相关天线误差、前端频率响应和多径不能都归入一个共同常数。因此相比早期SDCP减少的是对绝对共同偏差的依赖，不是消除所有标定需求。

## 4. 商用判断

可落地优势是复用双天线接收机的载波观测，在观测层工作，无需把全部原始RF送到主机。但必须能读到匹配PRN的同历元相位、明确失锁/半周状态，并验证实际噪声、安装方向图和输出延迟。

75.9mm基线只验证该配置，不能套用到任意终端。单基线测得的是方向投影，相同投影不等于同一三维方向；短基线、少星、集中天空和分布式欺骗会削弱区分能力。实际门限需要按支持的卫星集合、噪声与环境进行独立评估。

首轮“抗”体现为检测和定位可信度标记，不是恢复导航。全部真实信号缺失时无法靠双天线生成真实位置。没有硬件功耗、BOM、实机测试或许可清查结果，本阶段不作商用验收结论。

实施及验证定义见[设计计划](../validation/2026-09-12-antenna-spoofing-plan.md)，本项目数据与结论见[首轮验证报告](2026-09-12-antenna-spoofing-validation.md)。
