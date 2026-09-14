# GPSPATRON 算法公开程度调研

调研日期：2026-09-14。范围：公开网页、厂商技术说明与手册、专利线索及公开论文；本轮没有运行新实验，也没有获得或测试 GP-Cloud 商用软件。

## 结论

目前找到的是**公开空间检测原理、产品架构、部署与校准要求，但未找到足以复现 GP-Probe TGE2 / GP-Cloud 完整检测器的公式、参数和源码**。因此，“依据 GPSPATRON 公开原理的自主实现”是准确描述；“复现 GPSPATRON 专有算法”证据不足。

“专有”与“完全不公开”不是同一概念：本次确实找到厂商的专利保护声明，但尚未识别出能明确归属于该公司、并对应其多天线检测算法的具体专利公开号。未检出不等于不存在，也不能据此认定其全部算法均为商业秘密。

## 1. 找到了什么、还缺什么

| 资料 | 已公开的内容 | 对可复现性的限制 |
|---|---|---|
| [空间检测技术说明](https://gpspatron.com/coherent-gnss-spoofing-detection-multi-antenna/) | 跨天线载波相位/时序关系、空间特征、信号方向与卫星几何的一致性 | 未给出完整统计量、阈值、模糊度处理或分类流程 |
| [TGE2 产品规格与选型](https://gpspatron.com/gp-probe-time-guard-edition/) | CH1：异步欺骗；CH2：同步欺骗；CH3：同步多发射源欺骗 | 这是厂商能力分类，不是覆盖所有攻击条件的数学保证 |
| [原始数据异常检测说明](https://gpspatron.com/anomaly-detection-in-raw-gnss-data-for-time-server-protection-against-spoofing/) | 将多个 GNSS 观测视为相关时序；讨论相位、锁定时间、伪距、残差、Doppler、功率等信息 | 不能从输入字段推断其特征融合公式，也没有依据认定采用某一种具体机器学习模型 |
| [校准说明](https://support.gpspatron.com/support/solutions/articles/101000492698-gp-probe-calibration) | 搬迁、更换天线或电缆后重校准；完整校准最长 24 小时；异常期间校准会污染正常基准 | 没有说明哪些检测分支依赖该基准、如何更新模型及如何防止污染 |
| [Lua SYS 接口](https://support.gpspatron.com/support/solutions/articles/101000467309-sys-lib-description) | 读取设备状态和各星座欺骗/干扰状态 | 状态接口不是检测算法源码 |

所以此前“阈值和校准不公开”应收窄为：**校准操作和部分工程条件是公开的，空间检测器内部的统计标定方法未见完整公开。** 单通道 DIN L1 文档中的异常百分比或频谱阈值，也不能直接移作 TGE2 多天线空间判据的阈值。

## 2. 专利线索及检索边界

[官方 TGE2 手册 v1.1](https://gpspatron.com/wp-content/uploads/2024/05/GP-Probe-TGE2-User-Manual-v11.pdf) 第 3 页声明产品在欧盟内外受到专利法保护，但该声明没有给出相应专利号或技术对应关系。PDF 封面标 v1.1，内页仍有 v1.0 页眉，引用时按下载文件及封面辨识版本。

[GP-Cloud 本地部署许可协议](https://gpspatron.com/gp-cloud-on-premise-end-user-license-agreement/) 将软件列为厂商专有知识产权。其列出的 **88894699 和 018123323 是商标标识，不是抗欺骗算法专利号**。本地部署授权也不能据此理解为提供源代码或公开算法。

本次实际采用公开网页检索，包括：

- `"GPSPATRON" patent`、`"GPSPATRON" "carrier phase"`、`"GPSPATRON" paper algorithm phase detection`。
- `"GPSPATRON" site:patents.google.com`，以及对 WIPO / Espacenet 域名的索引限定搜索。
- 公司名 `GPSPATRON Sp. z o. o.`；人名 `Barodzka`、`Borodko` 与 patent / GNSS / spoofing 组合。Maksim Barodzka 可由[官方合作公告](https://gpspatron.com/gpspatron-and-flying-eye-announce-strategic-partnership-to-enhance-resilient-gnss-in-france/)核实为公司负责人；另一拼写仅作为扩展检索词，不认定为同一人。

检索未获得可确认归属的算法专利号。Google Patents 的申请人检索页面本次访问失败；WIPO / Espacenet 仅进行了搜索引擎索引检索，没有完成数据库内全面检索或波兰专利档案核查。因此，本结论是**公开技术资料的可复现性判断，不是“无专利”的证明或完整专利族检索结果**。若要继续锁定对应专利，最有价值的新增信息是厂商提供的公开号、申请人或专利族编号。

## 3. 对现有 2–4 天线实验的重要影响

### 天线孔径不能忽略

产品页推荐水平间距 **0.25–5 m**；[安装说明](https://support.gpspatron.com/support/solutions/articles/101000466605-gnss-antenna-installation-tips)写 **0.20–5 m**，同时称几何布置无硬性要求。两份材料下限略有差异，应保留这一差异，不能把推荐值说成绝对工作边界，也不能把“无硬性要求”解读为任意孔径性能相同。

本地阵列尺寸为 75.9 × 160.9 mm，最大对角线约 177.9 mm，小于上述两个推荐下限。因而已有结果只能说明当前小孔径、当前模型下的收益，不能直接预测商用设备的性能。

### 环境基准不同

本地实现不需要运行时干净启动参考，但冻结阈值来自合成无欺骗校准集；这不等于完全不需要正常数据。厂商则公开了现场环境校准流程。两者的正常分布、校准时间和漂移处理不同，尚不能视为同一算法。

[虚警排查说明](https://support.gpspatron.com/support/solutions/articles/101000466612-high-false-positive-rate-of-spoofing-jamming-detection)给出良好安装环境下“日区间内 0.2%”的厂商指标，并指出建筑反射和工业宽带噪声可增加虚警。页面没有提供足以统一统计口径的分母和事件合并规则，不能与本地每窗口名义 1% 虚警率直接比较。

### 检测统计量是本地设计

[本地源码](../../python_sim_simplified/spatial_spoofing.py)中的以下选择没有找到 GPSPATRON 原算法依据：

- `exp(4jπΔL/λ)` 倍相圆统计及由此引入的空间折叠。
- `global` 全星离散度与 `cluster3` 三 PRN 子集扫描。
- 无欺骗分布的 1% 下分位阈值，以及 100 ms 判决窗口。
- 无效观测置为未知、因果输出门控等具体规则。

它们是可检验的自主研究方案。`cluster3` 中的“3”指三颗卫星/PRN，**不是**厂商 CH3 的三条天线接收通道。厂商三通道“多发射源检测”的描述，也不能反推出采用本地三 PRN 聚类。

已有[实验报告](2026-09-14-gpspatron-spatial-diversity.md)保留价值：它比较了明确实现下的天线数、部分同源欺骗收益和混合信号失败条件。但这些检测率、漏检及未知比例均不能冠以 GPSPATRON 产品性能。

## 4. 更适合逐公式复现的公开研究

| 研究 | 与本项目的联系 | 已核实范围 |
|---|---|---|
| Psiaki 等，2014，[GNSS Spoofing Detection Using Two-Antenna Differential Carrier Phase](https://www.ion.org/publications/abstract.cfm?articleID=12530)；[作者预印本](https://gps.mae.cornell.edu/Paper_E6b_1_ION_GNSS_2014b.pdf) | 双天线单差载波相位、单发射源共同到达方向、假设检验 | ION 摘要确认实船试验，也明确真假信号功率接近时存在拍频与跟踪/分类困难；适合对照本地混合信号失败 |
| Borio、Gioia，ION GNSS+ 2015，[A Dual-antenna Spoofing Detection System Using GNSS Commercial Receivers](https://publications.jrc.ec.europa.eu/repository/handle/JRC94730) | 两个商用接收机独立振荡器、无需假定阵列几何 | JRC 官方记录确认架构与实时试验；本轮未逐式审计全文，不能声称已完成公式复现准备 |
| 2018，[GNSS Spoofing Detection Technique Using Fraction Parts of Double-difference Carrier Phases](https://www.cambridge.org/core/journals/journal-of-navigation/article/abs/gnss-spoofing-detection-technique-using-fraction-parts-of-doubledifference-carrier-phases/43726DFB962005448F3C2A412D723151) | 双差相位小数部分、无需求整周与基线向量、M-of-N 检验 | 出版方摘要限定单发射天线全信号情形；本轮仅核实摘要，不将其视为已解决部分/多源欺骗 |

这些是独立公开研究，未发现证据表明它们就是 GPSPATRON 实际采用的算法。更扎实的后续路线是选其中一篇开展公式与实现逐项对应，再在相同 2–4 天线观测、统一虚警口径下比较；同时单独比较手机孔径与厂商推荐孔径。这样能验证具体方法的收益，而无需猜测厂商内部实现。

## 本轮交付边界

本轮新增公开程度调研和实验解释补充，未修改算法、冻结阈值或既有数值结果。仍缺：可确认对应的厂商专利号、完整商业检测器说明、同条件商业设备实测对照。
