# 抗干扰接收链实验

## 无干扰相位中心偏移验证（2026-10-16）

本轮只评估固定相位中心偏移PCO，不评估方向相关PCV、动态移动、码包络时延或定位精度。真实位置用 `phone.phase_centers_m`，接收机名义位置用 `phone.assumed_phase_centers_m`；各自独立，不能修改真实位置后自动同步名义位置。

公共条件：TL/BR、L1、目标方位30°/仰角60°、输入−128.5 dBm，理想等增益天线，无互耦、干扰、遮挡及附加通道误差。100 ms、首尾各10 ms分量统计去边；方法为SINGLE（固定TL）、严格等幅EGC和MVDR。u为目标来波单位向量，v=(-sin30°, cos30°, 0)与u垂直；表中偏移单位mm，相对预设角点，未列出的通道为0。

| 条件 | 真实偏移 | 名义偏移 | 目的 |
|---|---|---|---|
| PC-001 | 全部0 | 全部0 | 基准 |
| PC-002 | TL/BR均10u | 全部0 | 公共相位，不改变相对相位 |
| PC-003 | BR=10v | 全部0 | 垂直来波，投影为0 |
| PC-004 | BR=1u | 全部0 | 小误差 |
| PC-005 | BR=5u | 全部0 | 5 mm误差 |
| PC-006 | BR=10u | 全部0 | 10 mm误差 |
| PC-007 | BR=20u | 全部0 | 边界压力 |
| PC-008 | BR=−10u | 全部0 | 与006反号 |
| PC-009 | BR=10u | BR=10u | 准确标定 |
| PC-010 | BR=10u | BR=5u | 欠补偿，残余5 mm |
| PC-011 | BR=10u | BR=15u | 过补偿，残余−5 mm |
| PC-012 | BR=10u | BR=−10u | 补偿反向，残余20 mm |
| PC-013 | TL=−5u、BR=5u | 全部0 | 与006同相对误差、不同公共相位 |
| PC-014 | BR=(10,0,0) | 全部0 | x轴误差 |
| PC-015 | BR=(0,10,0) | 全部0 | y轴误差 |
| PC-016 | BR=(0,0,10) | 全部0 | z轴误差 |

每个条件按single/egc_strict/mvdr展开为三条独立用例，共48条/seed。沿来波方向的残余位置误差产生 −2πΔr·u/λ 相位；等幅两路、理想独立等强噪声下，严格EGC相对准确对齐损失为 −10log10(cos²(Δφ/2))。1/5/10/20 mm的理论损失约0.001/0.030/0.119/0.482 dB；它是理想参考，不将有限噪声实现偏差误判为算法错误。MVDR不套用此严格等幅公式作验收。

### 实施与验证步骤

复用validation入口，不改核心接收算法；条件放validation_config.py，独立投影/功率复算与配对统计在实验支持层，CSV保存仍由validation.py负责。先以本地测试验证48条映射、公共/垂直偏移、补偿残余符号和真实信号一致性，再执行seed 42全部链路；正式统计配置保留seed 42～71，本轮先不执行1440次正式批次。结果按同算法PC-001、同条件SINGLE及同真实位置准确标定分别比较，不混用对照。PC-006、009～012的实际DFE输入必须相同。捕获峰值比保留无量纲，跟踪C/N₀缺失不填0；完成跟踪流程不等于证明持续锁定。

### 运行与结果表

在python_sim_simplified目录执行 `python -m experiments.interference.validation`。默认 `BATCH="phase_center_noise_only"`；种子、时长、并行数和结果目录分别由validation_config.py顶部的 `PHASE_CENTER_SEEDS`、`PHASE_CENTER_DURATION_MS`、`PHASE_CENTER_WORKERS`、`PHASE_CENTER_OUTPUT_DIR` 控制。正式默认30个seed；下面命令只复现本轮单seed验证，输出到不同目录，不覆盖互耦或增益结果：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MPLBACKEND='Agg'
python -u -B -c 'from experiments.interference import validation_config as config; from experiments.interference import validation; config.PHASE_CENTER_SEEDS=(42,); config.PHASE_CENTER_OUTPUT_DIR=config.INTERFERENCE_RESULTS_DIR/"phase_center_noise_only_smoke"; validation.main()'
```

优先看 `case_summary.csv`：每行一个PC条件及方法，包含各天线真实/名义xyz偏移mm、残余视线投影mm、相对相位deg、目标响应幅度dB及相位deg、捕获峰值比和跟踪C/N₀。`snr_delta_vs_pc001_mean_db` 是相对同算法无偏移基准的变化；`snr_delta_vs_accurate_calibration_mean_db` 是PC006/010/011/012相对同真实位置PC009的变化；`snr_gain_vs_same_condition_single_mean_db` 是相对同工况SINGLE的分集收益。负差值表示变差；`ideal_strict_egc_loss_db` 则用正值表示理论损失。参考用例名称单独列出，未设置的对照留空。

CI为逐seed配对差值均值的Student-t 95%区间，只有一个seed时留空，不伪装成零不确定度。`summary.csv`/`summary.json`保留逐seed指标，`verification.json`保存数值核验，`manifest.json`保存实际参数及源码哈希。SNR使用线性目标/热噪声分量独立复算，TRK C/N₀为真实接收链估计，两者不混用；默认不输出图片。

### 本轮实际结果

seed 42、100 ms共48次链路已运行，结果在 `results/interference/phase_center_noise_only_smoke/`。156项核验全部通过；48/48在0.5 chip、125 Hz容差内正确捕获，48条完成跟踪并有后段C/N₀估计，不代表已经验证持续锁定概率。case_summary有48行，无嵌套JSON单元格，无PNG；单seed置信区间留空。运行前源码哈希与manifest一致。

下表为相对同算法PC-001的线性目标/热噪声分量SNR变化，单位dB；PC-001按0列示。输出相位列为严格EGC的真实目标复响应相位，不是BR与TL之间的残余相位。

| 条件 | 严格EGC SNR变化/dB | MVDR SNR变化/dB | 严格EGC输出相位/° |
|---|---:|---:|---:|
| PC-001 | 0.000000 | 0.000000 | 0.000 |
| PC-002 | 0.000000 | 0.000000 | -18.918 |
| PC-003 | 0.000000 | 0.000000 | -0.000 |
| PC-004 | -0.001184 | -0.001063 | -0.946 |
| PC-005 | -0.029626 | -0.029657 | -4.730 |
| PC-006 | -0.118910 | -0.120920 | -9.459 |
| PC-007 | -0.482335 | -0.492459 | -18.918 |
| PC-008 | -0.118910 | -0.123938 | 9.459 |
| PC-009 | 0.001548 | 0.001554 | 0.000 |
| PC-010 | -0.028966 | -0.029043 | -4.730 |
| PC-011 | -0.026985 | -0.028713 | 4.730 |
| PC-012 | -0.482906 | -0.494203 | -18.918 |
| PC-013 | -0.118910 | -0.120541 | 0.000 |
| PC-014 | -0.022213 | -0.022153 | -4.096 |
| PC-015 | -0.007400 | -0.007229 | -2.365 |
| PC-016 | -0.089080 | -0.090318 | -8.192 |

结论：共同偏移只改变公共相位；垂直来波的偏移无目标相位投影。BR单独沿来波偏移1/5/10/20 mm时，严格EGC损失与独立理论一致。PC006与PC013相对位置误差相同、SNR损失相同，但输出相位不同。PC014/015/016的轴向10 mm分别投影为4.330/2.500/8.660 mm，不能按偏移向量长度直接计算损失。

准确标定PC009的严格EGC相对同工况SINGLE收益为3.001518 dB，无偏移PC001为2.999970 dB；两者相差0.001548 dB是改变权值方向后有限噪声样本的功率变化，不是位置偏移创造了理论收益。对同一真实输入，PC006未补偿相对PC009损失0.120458 dB，欠补偿PC010损失0.030514 dB，过补偿PC011损失0.028533 dB，反向补偿PC012损失0.484454 dB。正负残余的理想损失对称，有限噪声实现不必完全对称。

TRK C/N₀不是上述分量SNR：例如严格EGC的PC001/PC002/PC009后段估计分别为47.346/46.943/46.949 dB-Hz，尽管理论相干损失均为0。100 ms单次估计、信号相对噪声相位变化及实际量化接收链不能用微小分量SNR差值概括；本轮不能据这些估计排序标定优劣，也未验证其跨seed偏差。正式30seed批次未运行。

最大阵列响应复数误差5.56e−17、权值相对误差3.15e−16、独立理论/分量测量SNR误差7.11e−15 dB。专项测试6项通过；完整回归见CHANGELOG。本轮未改变核心接收算法，测试仅本地保留。上述档位是敏感性测试，不代表手机实测PCO分布。

## 互耦验证用例与实现约束（2026-10-16）

目标：先验证窄带互耦实现，再区分物理互耦与名义响应失配。共55条独立用例：7条确定性模型检查，以及12个链路条件×4种算法=48条链路用例。模型检查每批一次；标准链路每条30 seed，共1440次，不把模型检查虚报为接收仿真。

物理模型：`C=[[1,c],[c,1]]`，`c=10**(L/20)*exp(j*phi)`。两个交叉项是同一复数而非共轭，表示本轮互易交叉电压假设；它不是能量归一化网络、实测S参数或效率模型。无互耦用交叉系数0表示，不能用0 dB代替。当前代码在天线响应处作用C，RF接收机热噪声在后级独立加入，不经过C；本轮不引入天空噪声、外部相关噪声或互耦损耗模型。

公共条件：TL/BR理想天线，阵元附加增益均0 dB；手机角点位置沿用预设，方向30°/60°，真实/名义方向一致；信号−128.5 dBm；无干扰、额外通道误差、时延、遮挡。RF/DFE与捕获跟踪沿用默认值。链路100 ms、统计去边各10 ms、seed 42～71；四算法single、egc（幅度匹配）、egc_strict（严格等幅）、mvdr。SINGLE固定TL，不随条件选择强通道。

### 模型级用例：逐条定义

| 用例 | 条件 | 独立预期 |
|---|---|---|
| MC-001 | 任意双通道复序列，无互耦 | 输出等于输入 |
| MC-002 | 仅TL输入，双向−20 dB/0° | TL原样，BR为输入的0.1倍 |
| MC-003 | 仅BR输入，双向−20 dB/0° | BR原样，TL为输入的0.1倍 |
| MC-004 | 仅TL输入，双向−20 dB/90° | BR为输入的0.1j倍 |
| MC-005 | 仅TL输入，双向−20 dB/180° | BR为输入的−0.1倍 |
| MC-006 | 两路独立复序列，双向−10 dB/45° | 逐通道显式交叉相加，与实际矩阵输出一致 |
| MC-007 | 同步交换物理通道顺序及矩阵行列 | 按天线名还原后输出一致 |

### 链路级用例：逐条定义

每行只运行名称后缀对应的一种算法，编号不是数组位置依赖；对照按条件和算法关联。无互耦对照为同算法MC-008；准确已知对照为真实物理条件相同、名义参数准确的用例。MC-015～019只改名义响应，必须以同seed的DFE输入指纹证明与MC-012实际信号完全相同。

| 独立用例 | 真实交叉耦合 | 接收机名义交叉耦合 | 唯一变化／用途 |
|---|---|---|---|
| MC-008-single | 无互耦 | 无互耦 | 零耦合基准 |
| MC-008-egc | 无互耦 | 无互耦 | 零耦合基准 |
| MC-008-egc_strict | 无互耦 | 无互耦 | 零耦合基准 |
| MC-008-mvdr | 无互耦 | 无互耦 | 零耦合基准 |
| MC-009-single | −30 dB / 0° | 同真实 | 弱耦合，比较008 |
| MC-009-egc | −30 dB / 0° | 同真实 | 弱耦合，比较008 |
| MC-009-egc_strict | −30 dB / 0° | 同真实 | 弱耦合，比较008 |
| MC-009-mvdr | −30 dB / 0° | 同真实 | 弱耦合，比较008 |
| MC-010-single | −20 dB / 0° | 同真实 | 中等耦合，比较008 |
| MC-010-egc | −20 dB / 0° | 同真实 | 中等耦合，比较008 |
| MC-010-egc_strict | −20 dB / 0° | 同真实 | 中等耦合，比较008 |
| MC-010-mvdr | −20 dB / 0° | 同真实 | 中等耦合，比较008 |
| MC-011-single | −10 dB / 0° | 同真实 | 较强耦合，比较008 |
| MC-011-egc | −10 dB / 0° | 同真实 | 较强耦合，比较008 |
| MC-011-egc_strict | −10 dB / 0° | 同真实 | 较强耦合，比较008 |
| MC-011-mvdr | −10 dB / 0° | 同真实 | 较强耦合，比较008 |
| MC-012-single | −10 dB / 90° | 同真实 | 相位扫描及失配组基准 |
| MC-012-egc | −10 dB / 90° | 同真实 | 相位扫描及失配组基准 |
| MC-012-egc_strict | −10 dB / 90° | 同真实 | 相位扫描及失配组基准 |
| MC-012-mvdr | −10 dB / 90° | 同真实 | 相位扫描及失配组基准 |
| MC-013-single | −10 dB / 180° | 同真实 | 与011比较相位影响 |
| MC-013-egc | −10 dB / 180° | 同真实 | 与011比较相位影响 |
| MC-013-egc_strict | −10 dB / 180° | 同真实 | 与011比较相位影响 |
| MC-013-mvdr | −10 dB / 180° | 同真实 | 与011比较相位影响 |
| MC-014-single | −10 dB / 270° | 同真实 | 与012比较相位影响 |
| MC-014-egc | −10 dB / 270° | 同真实 | 与012比较相位影响 |
| MC-014-egc_strict | −10 dB / 270° | 同真实 | 与012比较相位影响 |
| MC-014-mvdr | −10 dB / 270° | 同真实 | 与012比较相位影响 |
| MC-015-single | −10 dB / 90° | 无互耦 | 完全忽略互耦，比较012 |
| MC-015-egc | −10 dB / 90° | 无互耦 | 完全忽略互耦，比较012 |
| MC-015-egc_strict | −10 dB / 90° | 无互耦 | 完全忽略互耦，比较012 |
| MC-015-mvdr | −10 dB / 90° | 无互耦 | 完全忽略互耦，比较012 |
| MC-016-single | −10 dB / 90° | −13 dB / 90° | 低估耦合幅度3 dB，比较012 |
| MC-016-egc | −10 dB / 90° | −13 dB / 90° | 低估耦合幅度3 dB，比较012 |
| MC-016-egc_strict | −10 dB / 90° | −13 dB / 90° | 低估耦合幅度3 dB，比较012 |
| MC-016-mvdr | −10 dB / 90° | −13 dB / 90° | 低估耦合幅度3 dB，比较012 |
| MC-017-single | −10 dB / 90° | −7 dB / 90° | 高估耦合幅度3 dB，比较012 |
| MC-017-egc | −10 dB / 90° | −7 dB / 90° | 高估耦合幅度3 dB，比较012 |
| MC-017-egc_strict | −10 dB / 90° | −7 dB / 90° | 高估耦合幅度3 dB，比较012 |
| MC-017-mvdr | −10 dB / 90° | −7 dB / 90° | 高估耦合幅度3 dB，比较012 |
| MC-018-single | −10 dB / 90° | −10 dB / 60° | 相位偏差−30°，比较012 |
| MC-018-egc | −10 dB / 90° | −10 dB / 60° | 相位偏差−30°，比较012 |
| MC-018-egc_strict | −10 dB / 90° | −10 dB / 60° | 相位偏差−30°，比较012 |
| MC-018-mvdr | −10 dB / 90° | −10 dB / 60° | 相位偏差−30°，比较012 |
| MC-019-single | −10 dB / 90° | −10 dB / 120° | 相位偏差+30°，比较012 |
| MC-019-egc | −10 dB / 90° | −10 dB / 120° | 相位偏差+30°，比较012 |
| MC-019-egc_strict | −10 dB / 90° | −10 dB / 120° | 相位偏差+30°，比较012 |
| MC-019-mvdr | −10 dB / 90° | −10 dB / 120° | 相位偏差+30°，比较012 |

### 核验与输出要求

- 从位置和角度独立计算几何相位，再显式交叉相加得到真实/名义响应，与生产模型核对；不能再次调用同一响应函数冒充独立复算。
- 固定实际接收权值，按独立响应、源标量功率和热噪声样本协方差计算 `P*abs(wᴴa)**2/(wᴴRn*w)`，对照线性分量实测SNR。量化/限幅残差独立保留，不混入热噪声。
- 独立核对算法权值与名义响应约束；失配时真实目标响应可偏离1，作为性能结果记录，不能据此直接判实现失败。
- 报告相对无耦合及准确已知条件的配对SNR变化；不预设耦合一定有害，也不要求MVDR等于幅度匹配。
- 每条保存真实/名义耦合强度和相位、各阵元耦合前后增益和相位、目标/热噪声功率、SNR及复算偏差、捕获峰值比、过门限次数与正确捕获次数、TRK C/N₀及有效seed数。正确捕获采用过门限且码误差≤0.5 chip、多普勒误差≤半个搜索格；跟踪流程完成不等于保持锁定。
- 默认只保存数据和表，不批量绘图。测试仅本地保存；不恢复历史多径接口，不修改物理核心模型。本轮先做模型、短链及单seed完整时长核验，标准30seed配置与实际已跑范围分开记录。

实施顺序：先补模型及用例构造测试；在现有validation入口接入互耦评分和统计；检查失配物理输入相同、串并行一致、表格完整及旧批次回归；完成后由独立审查核对科学判据，再写入实际验证结果。

### 运行互耦批次

仍使用 `python -m experiments.interference.validation`。在validation_config.py中选择 `BATCH="mutual_coupling"`；互耦专用设置为 `COUPLING_SEEDS`、`COUPLING_DURATION_MS`、`COUPLING_WORKERS`、`COUPLING_OUTPUT_DIR`，默认30 seed、100 ms、4进程，输出 `results/interference/mutual_coupling/`。它不使用旧批次的SEEDS/OUTPUT_DIR，避免覆盖增益结果；切回legacy或pattern_gain_first44仍使用原有设置。互耦批次默认不生成或弹出图片。

优先阅读 `model_checks.csv`（7行模型检查）和 `case_summary.csv`（48行接收条件汇总）；`summary.csv`记录逐seed指标，`summary.json`、`verification.json`保留独立复算与失败原因，`manifest.json`记录实际配置。接收机权值仍只来自实际DFE数据和名义响应，真值与分量仅用于接收完成后的评分。


### 本轮实际验证（2026-10-16）

已运行7条模型检查，以及seed 42、100 ms的全部48条链路用例；结果保存在 `results/interference/mutual_coupling_smoke/`，与正式30 seed输出目录分开。171项核验全部通过，48条均通过捕获门限且码相位/多普勒误差在本轮判据内；不等于已经验证持续锁定概率。未运行正式1440次链路批次。

本轮最大响应复数绝对误差为5.56e−17，权重相对误差3.52e−16，独立理论与分量测量的SNR差7.11e−15 dB。MC-015～019与MC-012同seed、同算法的DFE输入指纹一致；运行前源码哈希与结果manifest一致。输出含7行model_checks、48行case_summary，未生成PNG。

在此双天线几何与噪声假设下，−10 dB交叉互耦相位为0°时，EGC输出热噪声SNR相对无互耦增加约2.340 dB；180°时降低约3.133 dB。90°时TL/BR有效电压响应增益分别为−0.233/+0.977 dB，270°时交换。它们是交叉电压叠加的模型结果，不代表被动天线凭空产生功率，也不能代替效率或相关噪声研究。

本轮名义响应失配的EGC/MVDR损失仅约0.002～0.024 dB（相对同物理条件的准确已知响应）。这只说明当前方向、对称互耦和失配幅度下影响较小，不能推断其他方向、非对称互耦或实际硬件也如此。单seed结果不作统计显著性结论。DFE分量功率采用数字码值平方，不标为瓦特；捕获峰值比仍为无量纲。

复现本轮单seed验证（在python_sim_simplified目录，不改配置文件）：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MPLBACKEND='Agg'
python -u -B -c 'from experiments.interference import validation_config as config; from experiments.interference import validation; config.COUPLING_SEEDS=(42,); config.COUPLING_OUTPUT_DIR=config.INTERFERENCE_RESULTS_DIR/"mutual_coupling_smoke"; validation.main()'
```

专项验证：`python -m pytest tests/test_coupling_validation.py -q`，7 passed。完整回归结果见CHANGELOG本轮记录。测试只保留本地，不强制加入版本库。

## 用例汇总表字段

case_summary.csv每行一个条件及算法，n为seed数；新增字段如下：

- `TL_gain_db`、`BR_gain_db`及对应`*_attenuation_db`：本条各阵元注入增益/衰减，按实际天线名称生成列，衰减=−增益。例如增益−6 dB即衰减6 dB；不是包含几何、极化等因素后的总有效增益。
- `acquisition_peak_ratio_mean`：所有seed捕获峰值比的算术均值，包含未过门限样本，无量纲，不转dB。
- `tracking_cn0_mean_db_hz`：每seed已有“跟踪后段C/N₀中位数”的跨有效seed算术均值，单位dB-Hz；不是全历元混合平均或线性功率平均。无有效估计时留空。
- `tracking_cn0_valid_seed_count`：有后段C/N₀估计的seed数，配合原missing_count阅读，不能把仅成功样本均值当作整体接收性能。

保存流程专项8项通过，包含短链实际输出校验。此次未重跑正式批次；当前CSV被占用无法覆盖，旧文件未改动，关闭表格后可从已有summary.json导出，无须重新生成信号。

后续更新：用户关闭占用程序后，已从现存summary.json重新导出case_summary.csv，50行及新增列检查通过。只更新汇总CSV，原始指标、核验结论及图片保持不变，没有重跑仿真。

## 批量验证默认只保存数据

`validation_config.py` 默认 `GENERATE_PLOTS_FLAG=False`、`SHOW_PLOTS_FLAG=False`：运行validation只保存原始指标、case_summary.csv汇总、配置和核验文件，不生成PNG也不弹窗。终端显示输出目录、失败项数量及名称/原因；核验失败仍先保存证据再报错。

需要图片时设 `GENERATE_PLOTS_FLAG=True`；只保存图片保持 `SHOW_PLOTS_FLAG=False`，需要弹窗再设为True。此设置只影响通用validation入口，不改变full_chain等其他入口。原有图片本次未删除；下次运行仍按既有规则刷新该批次输出目录。下文生成50张图的记录是历史重绘，不再是默认行为。

验证：入口专项测试覆盖默认不调用绘图、开启绘图、失败先保存并打印原因；另在临时目录用seed 42、12 ms、1 ms去边执行TC-001/002，核验通过、case_summary.csv存在、无PNG。临时产物已自动清理，未覆盖研究结果；短跑只确认输出流程，不产生性能结论。

全套 `python -B -m pytest -p no:cacheprovider -q` 为193 passed、2 failed、2 deselected；失败是原绘图测试未开启新开关。调整 `test_short_saved_run_is_isolated_and_has_readable_plots` 与 `test_saved_phase_pairs_include_csv_columns_and_failed_verification`，并补齐后者的失败样本reason字段；失败列表只取status=failed，排除not_applicable。随后 `-k 'skips_plots or short_saved or saved_phase_pairs'` 三项通过，未再次运行全套。

## 增益验证图的横轴（2026-10-15）

增益批次的五类指标（热噪声SNR收益、目标功率收益、SINR、捕获峰值比、跟踪C/N₀）均按物理条件分图，不再把所有TC混在同一横轴：

- factor_1：SINGLE的TL增益，单位dB。
- factor_2：两路共同增益，单位dB。
- factor_3：固定TL=0 dB时的BR增益，单位dB。
- factor_4：强弱交换，横轴为TL/BR增益对，例如0/−6与−6/0，不连线。

当前50条件生成50张图（单天线5张，其余三类各3种算法×5张）。零增益点用于共同增益及BR增益曲线的基准，交换图补上交换前的条件；同一物理条件、算法和seed去重，不重复计入均值。横轴编号仅用于原始数据追溯，绘图转换不改原始记录。其他相位/方向扫描保留原物理横轴，但范围按当前图的方法计算。

图题n为该图seed数量，曲线为跨seed算术均值；C/N₀只平均有估计的样本，缺测不补零，不代表接收成功率。收益图固定相对SINGLE-TL：交换后TL变弱时，相对收益大不等于绝对输出增强。捕获峰值比保持无量纲及2.5门限，SNR收益为dB、C/N₀为dB-Hz。图例在图外，不遮住数据。

本次读取现存summary.json（seed 42）直接重绘到 `results/interference/pattern_gain_first44/physical_axes/`，保留旧图及原始指标，未重跑RF/DFE/接收链；下次正常运行自动使用新图格式并按原SHOW_PLOTS_FLAG设置弹窗。45项增益与通用验证专项测试通过，已检查单天线捕获及强弱交换收益图。

完整回归 `python -B -m pytest -p no:cacheprovider -q`：194 passed、2 deselected（201.39 s）。末次交换排序及标签微调后，`-k 'plot or plots'` 绘图专项3 passed；单天线C/N₀图也已检查。

## 增益扫描对照修复（2026-10-15）

`validation_config.py::_pattern_gain_cases()` 现在按(TL增益, BR增益, 算法)查找对照，不再写死TC编号。用户将SINGLE扫描改为0至−10 dB后，共50个条件；共同衰减基准自动指向TC-012～014，强弱交换自动指向对应的交换前条件。非零增益SINGLE均与零增益SINGLE配对。扫描仍需保留0 dB基准，以及交换实验所引用的增益点。

验证：配置对照断言在修复前失败、修复后通过；`python -B -m pytest -p no:cacheprovider -q tests/test_pattern_gain_validation.py` 为7 passed。读取用户现存500行summary.json，用新配置的对照映射调用 `summarize_pattern_gain(rows, references)`，50条核验全部通过。此操作只在内存复算，未覆盖原summary/verification/manifest及图片，也未重新生成IQ。

当前参数仍为100 ms、seed 42，运行入口不变。原500次文件仍保留错误对照下的判定，不能当作已经修复的报告；重新运行会使用新关联并覆盖同名输出目录。TC编号代表本批次顺序，扩展后的50条不能直接套用历史44条文档编号；下文历史结果保留当时条件。

全套回归命令 `python -B -m pytest -p no:cacheprovider -q`：190 passed、2 failed、2 deselected（209.81 s）。失败为 `test_first44_builds_one_method_and_one_physical_condition_per_tc` 和 `test_short_parallel_pattern_batch_keeps_order_and_scalar_progress`，断言仍使用原44条及−3 dB第二点；同步为用户50条及−1 dB后，增益文件7项复验通过。未再次运行全套，测试仅本地保留。

当前代码的布尔名称统一带 `_flag`（常量为 `_FLAG`），如 `SHOW_PLOTS_FLAG`、`success_flag`、`passed_flag`；旧结果及下文历史运行记录保留原名称。

无干扰方向图研究见 [天线方向图增益验证用例](antenna_pattern_validation_cases.md)：20 个分类下共 **425 条独立用例 TC-001～TC-425**，每条固定参数、一个被测方法、明确对照和判据。TC-001～044已执行，余下381条未执行，不将扫描组记作一条用例。

首批执行入口仍为 `python -m experiments.interference.validation`。`validation_config.py` 的 `BATCH="pattern_gain_first44"` 固定运行 TC-001～044，默认100 ms、seed 42～71、4个进程；输出 `results/interference/pattern_gain_first44/`。`progress.jsonl` 是逐次完成记录，`summary.csv` 是逐seed数据，`case_summary.csv` 是44条汇总结论，完整统计和判据在 `summary.json` / `verification.json`。切回旧实验用 `BATCH="legacy"`，并自行设置 `SEEDS`、`OUTPUT_DIR` 等公共参数。运行中不要再次启动同输出目录的批次。

`egc` 保持旧幅度匹配实现，新增 `egc_strict` 才是等幅相位对齐；首批每条仅运行文档指定的一种方法。SINGLE的输出有单位响应归一，天线增益传递通过合并前的逐通道线性分量功率单独验证。TC编号是离散类别，因此本批图只画散点、不连成连续扫描曲线。

接收统计口径：`acquisition_success_count` 是捕获峰值比通过门限的次数，需要结合保存的码相位/多普勒误差判断是否捕获到正确位置；`tracking_success_count` 是当前跟踪函数成功返回（流程完成）的次数，不能当作保持锁定成功率。当前默认跟踪入口没有启用独立的失锁确认判据，因此本批不作真实锁定概率结论。上述接收统计不决定增益模型的A/B/D科学判定。

在 `python_sim_simplified/` 目录运行：

```powershell
python -m experiments.interference.full_chain
```

入口为 [full_chain.py](../../experiments/interference/full_chain.py)，输出为 `results/interference/full_chain/`。时长和绘图选项位于入口顶部，信号与阵列条件在 [presets.py](../../core/presets.py) 中。

当前默认配置包含 CW 和带限噪声干扰，比较 SINGLE、EGC、MVDR、Blind Null。2026-10-14 已移除抗欺骗与多径专用代码，以及公共链路中的回波配置、生成、检测、评估、LCMV 和专用绘图接口；没有通过开关重新启用的路径。历史联合场景结果仍按当时配置解读，复现需使用对应冻结源码。

专用算法在 [interference/](../../core/interference/)：`interference.py` 生成带限噪声，`scene.py` 合成干扰到各天线的响应，`combining.py` 计算盲零陷。共享的天线、RF/DFE、捕获、跟踪、协方差合并及链路编排位于 `core/`。

重跑入口只重建它自己的输出目录。已冻结的历史数据和来源清单保留原内容。

## 2026-10-15 首批44条增益用例结果

条件：TC-001～044，TL/BR双天线，无干扰，入射信号−128.5 dBm，固定RF噪声系数；100 ms、seed 42～71、统计10～90 ms，详见用例公共条件C0。每个用例独立执行完整链路，没有提高信号强度。它们验证理想阵元的定值增益传递与合并，尚未验证参数化方向图的角度响应。

在精简版目录执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLBACKEND='Agg'
python -u -B -m experiments.interference.validation
```

1320次运行正常结束；44条各30次、无重复或缺项，1364项核验通过（1320项输出恒等式、44项增益/权值/预算判据）。逐通道目标功率变化误差最大4.44e-16 dB，热噪声变化为0；44条收益均值与解析参考最大偏差0.001710 dB。独立按用例参数重算非MVDR公式，单seed最大偏差0.018233 dB；MVDR独立求解权值相对误差最大5.09e-16，响应约束误差小于1e-8。以上不等于所有接收都成功。

| 条件（TL/BR增益） | 幅度匹配 egc 收益 | 严格等增益 egc_strict 收益 | MVDR 收益 |
|---|---:|---:|---:|
| 0 / 0 dB | +3.008927 dB | +3.008927 dB | +3.008936 dB |
| 0 / −6 dB | +0.972433 dB | +0.517024 dB | +0.972444 dB |
| 0 / −10 dB | +0.413438 dB | −0.625052 dB | +0.413449 dB |
| 0 / −20 dB | +0.043078 dB | −2.183819 dB | +0.043091 dB |
| 0 / −40 dB | +0.000422 dB | −2.925245 dB | +0.000435 dB |

表中均为30次线性热噪声SNR收益均值，相对固定SINGLE-TL，不是C/N₀估计。等强时约3 dB收益；一根明显变弱时严格等增益会引入过多噪声，幅度匹配与本条件下MVDR则接近强单天线。共同衰减不会消除相对合并收益，但绝对SNR仍随衰减下降。强弱交换后，幅度匹配的绝对输出SNR均值差最大约0.00150 dB；相对固定TL的收益会随参考通道变弱而增大，不能误读为额外获得20 dB阵列收益。

接收方面，SINGLE在TL增益0/−3/−6/−10/−20 dB时，捕获门限通过分别30/24/4/0/0次（各30次）。全批1035次过门限，其中TC-002、seed 57的多普勒误差为+250 Hz（一个搜索格）；其余过门限记录多普勒误差0，码相位误差绝对值均不超过0.08 chip。因此不能把门限通过直接称为正确捕获或保持锁定。缺测C/N₀原样保留，成功样本的C/N₀统计不代表所有运行。

文件：每条汇总在 [case_summary.csv](../../results/interference/pattern_gain_first44/case_summary.csv)，逐次数据在同目录summary.csv/progress.jsonl，置信区间和检查详情在summary.json/verification.json，共20张PNG。本次使用Agg保存图，不弹GUI；图为跨seed汇总，离散TC不连线。结果仅本地保留。

来源限制：运行期间其他窗口进行了布尔字段重命名等修改。进程使用启动时加载的模块和配置；manifest的source_sha256保留启动前哈希，source_sha256_at_save保留原保存器读取的结束时磁盘哈希，并附说明。未保存启动源码完整副本，哈希本身不能恢复源码。本批及此前191 passed、2 deselected回归结果不能作为后续最新工作区的验证；未覆盖其他窗口修改，也未运行TC-045～425。

## 统一实际链与按需评估（2026-10-15）

参考 FULLCHAIN 的阶段组织：配置 → 源波形 → 天线响应/互耦/通道时延 → 天线端合成 → RF 滤波与热噪声 → RF 增益/幅相 → 限幅/ADC → DDC/FIR/抽取/DFE 量化 → 合并、捕获跟踪 → 评估。保留原有物理运算顺序。

- `core/pipeline.py` 负责实际前端与接收；当前通用验证逐用例独立执行，不再按配置判断能否共享前端。GPS 多卫星接收继续共用 RF/DFE 底层函数，保留自身多星处理流程。
- `retain_analysis_flag=True` 仅额外保留本次实际 RF 热噪声；`core/evaluation.py::linear_components` 按需传播源真值与噪声，复用实际滤波器、NCO、抽取和码值尺度。FULLCHAIN 默认不调用此分析；精确分量功率仍需要线性滤波，不能直接从非线性输出拆分。
- 线性分量绕过限幅及量化，固定真实接收权值后才评分；不参与求权。其相加结果与实际输出的差是联合非线性/量化残差，分量交叉项另列，不能称为独立热噪声。消融实验保留独立 ADC 重建核验。
- 各实验的配置、统计窗口、输出字段和绘图保持原口径；共用分量传播，旧相位入口也调用公共前端/接收函数。调用 `evaluate_budget` 前需显式保留分析输入，不再事后重抽噪声。

### 本次重构数值对照

基准为 Git `d53e639`。`python -B tests/frontend_refactor_probe.py before` / `after`（本地测试脚本）使用 seed 42、20 ms：十组实际链覆盖理想/参数化天线、无/CW/宽带/同时干扰、RF 幅相、非负非零通道时延、软硬限幅及浮点 DFE；比较场景、ADC、DFE、各分支权值/合并/捕获/跟踪和指标，指纹完全一致。

同一对照还包括 12 行天线消融、24 行直接响应、54 行旧相位和 48 行通用验证。相位点为 0/90/180°，统计去边 2 ms；通用验证包含反序 BR/TL、四种干扰集合及相位/方向/增益条件。138 行结果与核验状态全部一致；旧相位入口恢复原分量求和顺序后，交叉项与残差也严格一致。前后数据和比较结果在 `results/interference/frontend_refactor/{before,after}.pickle`、`comparison.json`。

另从 Git 基准在内存加载旧分量函数和旧 DFE/RF 实现，以 4 ms、BR/TL 反序、RF 增益 3/−2 dB、相位 25/−40°、NCO 偏移 0/123 Hz、DFE 位宽 0/8、四种干扰集合核对 16 组条件，共 48 个分量数组完全一致。这是重构等价验证，不产生新的抗干扰性能结论；历史批次与来源哈希未改写。

完整回归：`python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_frontend_refactor_final`，**164 passed、2 deselected**；覆盖实际噪声仅生成一次、非零 RF 幅相、错误噪声拒绝、残差及各入口输出。两项依赖本地冻结证据的测试按默认配置排除；未重跑 100 ms 完整研究批次或 GUI 交互。

## 天线模型与参数

`full_chain.py` 顶部的 `ANTENNA_MODEL = "parametric"` 默认启用 `synthetic_phone_corner_v2`。改成 `"ideal"` 可运行原各向同性基线；公共 `default_config()` 仍保持理想模型。两种模型均支持 2～4 根天线的任意组合，在配置生成后设置 `cfg.phone.active_antennas = ("TL", "BR")` 等即可。

参数化预设位于 `core/presets.py` 的 `parametric_config()`。Mate70 的 75.9 × 160.9 mm 外形只用于定义四角布局；相位中心、增益、极化、互耦及遮挡参数都是合成研究假设，不是手机实测或产品规格。

接收响应按以下顺序计算：

```text
相位中心几何相位 × 局部方向图/极化响应 × 方向相关相位变化
× 阵元固定幅相误差 × 复数遮挡响应 → 耦合电压矩阵 → 通道幅相响应
```

通道真实时延仍由信号序列生成阶段单独施加，不与天线方向相位重复计算。新增项进入原有 RF→DFE→合并→捕获→跟踪链，并由二维、三维曲面和球面图使用相同天线响应接口显示。

| 参数 | 单位与作用 |
|---|---|
| `phone.phase_centers_m` / `assumed_phase_centers_m` | 米；真实/估计相位中心位置，支持修改几何投影，不能把壳体角点视为实测电气相位中心 |
| `ParametricPattern.peak_gain_db`、`cosine_exponent`、`back_lobe_gain_db` | 原有仰角增益模型；峰值增益与背瓣增益为 dB |
| `azimuth_peak_deg`、`azimuth_attenuation_db` | 局部坐标下的优选方位角及方位衰减深度；天顶方向不受此项影响 |
| `phase_variation_mm=(qx,qy,qxy)` | 毫米；方向相关的额外接收路径残差，转为载频处的相位旋转 |
| `nominal_polarization`、`cross_polar_discrimination_db`、`cross_phase_deg` | 标称极化、交叉极化鉴别率及交叉分量相位 |
| `AntennaState.orientations` | 度；各阵元相对手机坐标的 yaw/pitch/roll，方向图和极化同时变换 |
| `AttenuationRegion.attenuation_db`、`phase_shift_deg` | dB / 度；遮挡幅度损失和附加接收电压相位；使用相同角域和过渡权重 |
| `CouplingPath.magnitude_db`、`phase_deg` | dB / 度；固定窄带交叉电压传递，不是直接输入的 RF S 参数 |

令 `az,el` 为阵元局部方向，新增方位衰减为：

```text
ΔG_dB = -azimuth_attenuation_db × cos(el)² × (1-cos(az-azimuth_peak_deg))/2
```

方向相位采用一个可修改的平滑研究模型：

```text
ux = cos(el) cos(az), uy = cos(el) sin(az)
d_mm = qx ux² + qy uy² + qxy ux uy
PCV_response = exp(-j 2π carrier_hz / c × d_mm × 1e-3)
```

正 `d_mm` 表示额外接收路径，产生负相位；正 `phase_shift_deg` 表示正电压相位旋转，两者定义不同。该多项式用于扫描敏感性，不能当作真实 PCV 标定表，也不自动代表完整宽带群延迟。PCV 是随入射方向及频率变化的相位校准部分，实际数据需要以安装态和参考点为基准取得。[NGS 的 PCO/PCV 定义](https://www.ngs.noaa.gov/ANTCAL/FAQ.xhtml)

在 `full_chain.py` 中生成 `cfg` 后，可以分别修改真实值与接收机假设值，例如：

```python
truth = cfg.phone.antenna_model.true_state
assumed = cfg.phone.antenna_model.assumed_state
truth.parametric_patterns["TL"].azimuth_attenuation_db = 6.0
truth.parametric_patterns["TL"].phase_variation_mm = (3.0, -1.0, 0.5)
assumed.parametric_patterns["TL"].phase_variation_mm = (0.0, 0.0, 0.0)
```

以上示例表示接收机没有校准 TL 的方向相位残差。真实与假设的方向图、姿态及耦合参数已独立保存，修改真实参数不会同时改变估计参数。预设中方向图/PCV/耦合初始值相同，遮挡仅进入真实状态；这种假设不等于已实现在线标定。

`results["truth"]["antenna_model_diagnostics"]["desired_true"]` 中可查看 `pcv_path_length_mm`、`pcv_response`、`blockage_voltage` 等逐级诊断，`desired_assumed` 保存接收机假设响应。信号功率仍为原设置，不为非理想天线增加发射功率。

### 实测数据与当前边界

- `measured_config()` 继续接入 CSV：`element_name, frequency_hz, azimuth_deg, elevation_deg, h_az_real, h_az_imag, h_el_real, h_el_imag`。这些 `h` 是相对模型输入电压归一化的无量纲接收 Jones 系数，不是米制有效高度；响应为 `hᴴ e`，导入其他软件数据前须核对复数共轭及坐标约定。
- 频率按绝对 1 Hz 容差选取，避免把邻近频率的角度数据混在一起；当前未做频率插值，实测模式不额外叠加参数化 PCV。角度插值仍是复数分量插值，仰角表外沿用端点，不能把表外值称作实测。
- 导入整阵嵌入方向图时，须确认其中是否已包含互耦、安装和方向相位，避免再叠加相同作用。嵌入方向图本身可包含互耦影响。[嵌入方向图与互耦说明](https://www.mathworks.com/help/phased/ug/modeling-mutual-coupling-in-large-arrays-using-embedded-element-pattern.html)
- 当前互耦是固定窄带矩阵，选择天线时使用活动阵元子矩阵；不包含未选端口负载变化、完整 S 参数网络或互耦噪声相关。新增遮挡相位也不是人体散射的电磁求解。
- 动态握持/姿态、温漂、宽带天线频响和自动通道时延估计/序列对齐仍未在本轮实现。RF 噪声谱密度已按下节修正；天线参数仍是合成假设，结果用于模型和算法敏感性分析，不是实测手机性能。

## RF 噪声标定与通道随机流

`core/rf.py` 的 RF 接口显式传入采样率、IF 和物理通道名。输入等效热噪声密度为 `N0 = k × T × F`，`F` 按各级增益与噪声系数独立使用 Friis 公式计算。复数 IF 的 I、Q 白噪声各占一半功率，未滤波电压方差合计为 `N0 × R × fs`。

`rf.noise_bandwidth_hz` 表示以 IF 为中心的**总/双边等效噪声带宽**，不是单边截止频率，也不是 3 dB 带宽。默认 129 抽头、奇数长度的单位 DC 增益 FIR 满足 `fs × sum(h²) = 8.184 MHz`，滤波器截止频率由这个条件求解。输入信号与热噪声使用相同等效 RF 滤波器；默认 RF 增益仍为 80 dB，级联 NF 仍为 1.235029 dB。

滤波前下变频到基带，滤波后回到原 IF；补偿已知 64 点 RF 群延迟，保留输入长度与时间原点。有限信号记录两端使用零延拓，噪声使用预历史采样保持开头稳态。该非因果对齐是离线等效处理，不代表真实硬件没有传播时延，也不是不同通道未知时延的自动校准。功率预算去掉前后各 10 ms；捕获跟踪仍使用完整记录。

噪声随机流以 `TL/TR/BL/BR` 物理名称及 seed 生成，I/Q 分别独立。因此同一天线的噪声不随通道数、行顺序或记录长度前缀改变。在没有改变该天线实际接收响应的情况下，选择 2/3/4 根天线不再改变相同参考天线的 SINGLE 输入；已有端口耦合子矩阵改变仍可能真实改变信号，不能要求所有耦合场景的 SINGLE 都相同。

旧实现将 `kTFB` 总功率的独立白噪声铺在整个采样带宽上，33 MHz 下带内密度比 `kTF` 低约 6.06 dB。此次不是给 C/N₀ 结果统一减去一个常数，而是修正信号与噪声的 RF 处理并重新仿真。历史结果与冻结源码保留原样，不把旧噪声模型的数值当作新版本性能证据。

本次三 seed、100 ms 理想 TL 无干扰标定：理论噪声密度为 −172.740158 dBm/Hz，理论输入 C/N₀ 为 44.240158 dB-Hz，滤波 ENBW 为 8.184 MHz。

| seed | Welch 密度相对理论误差 / dB | 总噪声功率相对误差 / % | 后半段跟踪 C/N₀ 中位数 / dB-Hz |
|---|---:|---:|---:|
| 42 | +0.0054 | +0.1007 | 43.981 |
| 43 | +0.0103 | −0.1209 | 43.847 |
| 44 | −0.0030 | +0.0288 | 43.314 |

跟踪估计使用原有 10 历元短窗口，有限样本和码匹配损失仍在，因此不强行校正为理论输入值。功率/密度核验使用零绝对容差和明确相对容差；测试还把再生噪声电压放大两倍，确认错误的四倍功率会使核验失败。

## 天线单因素配对消融

```powershell
python -m experiments.interference.antenna_ablation
```

入口 [antenna_ablation.py](../../experiments/interference/antenna_ablation.py) 一次完成数据生成、独立复算、统计和绘图。顶部设置 `SEEDS`、`DURATION_MS`、`ANTENNAS`、`EVALUATION_EDGE_TRIM_MS` 和 `SHOW_PLOTS_FLAG`。默认四天线、100 ms、噪声种子 42/43/44，不提高 GNSS 功率（−128.5 dBm），不改变捕获门限 2.5、盲干扰检测门限 6 dB，关闭多径。

有两个独立比较场景：默认 LOS 为方位 30°/仰角 60°；遮挡生效场景为 225°/20°。CW 和 2 MHz 宽带干扰均为 −90 dBm、方向 80°/10°。每个场景内部与同 seed 理想模型配对，不能把两个 LOS 之间的差异全部归因于遮挡。默认目标和干扰都不在遮挡区域内，因此不能用默认场景判断遮挡影响。

| 配置 | 开启内容 |
|---|---|
| `ideal` | 理想基线 |
| `elevation_gain_only` | 峰值/仰角增益/背瓣 |
| `orientation_polarization_only` | 姿态与极化响应作为一个因子组 |
| `azimuth_dependence_only` | 方位增益依赖 |
| `pcv_only` | 方向相关相位残差 |
| `coupling_only` | 复数耦合矩阵 |
| `blockage_amplitude_only` | 仅遮挡幅度 |
| `blockage_phase_only` | 仅遮挡相位 |
| `full_known` | 完整模型，真实/导向假设完全匹配 |
| `full_parametric` | 完整预设，接收机导向假设不包含遮挡 |

单因子配置的其他因素保持中性，真实/假设参数匹配；`full_known` 与 `full_parametric` 的差别用来单独检验遮挡失配。总共 60 条接收记录、240 个分支评估，另有 3 条理想单天线无干扰噪声标定。这是固定参数的小样本配对试验，不是所有角度、所有握持状态或全部 11 种天线组合的验证。

输出位于 `results/interference/antenna_ablation/`：`manifest.json` 保存配置、种子、源码哈希与指标定义；`summary.json/csv` 保存逐分支预算和同 seed 的理想模型差值；`noise_calibration.json` 保存 Welch 噪声密度、总功率与无干扰跟踪估计；五张图分别展示热噪声 SNR 增益、实际干扰抑制、目标/噪声输出增益、捕获峰比和正确捕获计数。不同指标使用不同子图，图默认保存并弹窗；Agg 无窗口运行只保存。

### 指标与独立复算的边界

- 接收机先正常生成权值、捕获和跟踪；真实目标/CW/宽带分量及再生热噪声只用于事后评分，不用于盲算法或接收机导向估计。
- 在实际冻结权值下，分别经过线性 RF 和浮点 DFE 重算分量功率，保持实际 DFE 的码单位与缩放。自行复算限幅和 ADC 量化后逐点核对 ADC；另用样本输出功率与 `wᴴRw` 核对合并。联合非线性/量化残差及交叉项明确保存，残差闭合本身是恒等式，不单独当作独立验证证据。
- 干扰抑制度是相对于同配置/seed 的 SINGLE **实际分量功率比**，不是方向图中相对扫描峰值的 dB 值。热噪声 SNR 增益同时包含目标功率与输出噪声变化；目标方向增益小不等于损失大小已经完整确定。
- `thermal_oracle_prompt_cn0_db_hz` 是事后用真码/真多普勒做 1 ms 相关的目标/热噪声功率比，包含 RF/DFE 与码匹配损失，不含残留干扰、限幅和量化残差，不是接收机直接可用的 C/N₀。实际跟踪的短窗口 C/N₀ 另列，不能要求二者逐点相等。
- 正确捕获要求原门限通过、码误差不超过 0.5 chip、多普勒误差不超过 125 Hz。`tracking_success/tracked_count` 仅表示跟踪函数执行成功，尚不是经过锁定质量判决的稳定锁定率。
- 当前 CW/宽带干扰同方向，互耦后的外部噪声相关、宽带天线频响、动态权值和真实硬件标定仍未覆盖；本试验不证明多方向干扰或商用性能。

### 本次实际结果（60 条记录，seed 42/43/44）

下表均为三个 seed 的均值；SNR 增益和干扰抑制度相对同配置的 SINGLE，捕获为满足码/多普勒误差定义的正确捕获计数。表中的干扰量是线性分量实际功率比，不包含联合非线性/量化残差，不能把 75 dB 分量抑制说成完整接收输出干扰降低 75 dB。此次 CW 与宽带同方向、零通道时延、固定窄带天线响应，所以两种分量的功率抑制比一致；这不是任意宽带场景的一般规律。

| LOS 场景 | 模型 | 算法 | 正确捕获 | 热噪声 SNR 增益 / dB | CW/宽带分量抑制 / dB | 真码热噪声相关 C/N₀ / dB-Hz |
|---|---|---|---:|---:|---:|---:|
| 默认 30°/60° | 理想 | MVDR | 3/3 | +5.56 | 55.51 | 49.63 |
| 默认 30°/60° | 理想 | Blind Null | 3/3 | −0.26 | 74.68 | 43.80 |
| 默认 30°/60° | 完整，已知/未知遮挡相同 | MVDR | 3/3 | +4.63 | 34.31 | 47.97 |
| 默认 30°/60° | 完整，已知/未知遮挡相同 | Blind Null | 0/3 | −9.96 | 36.17 | 33.00 |
| 遮挡 225°/20° | 完整，导向完全匹配 | MVDR | 0/3 | +0.34 | 32.46 | 39.09 |
| 遮挡 225°/20° | 完整，未建模遮挡 | MVDR | 1/3 | −0.74 | 38.45 | 38.24 |
| 遮挡 225°/20° | 完整，导向完全匹配 | Blind Null | 0/3 | −1.09 | 40.25 | 37.80 |
| 遮挡 225°/20° | 完整，未建模遮挡 | Blind Null | 0/3 | −1.09 | 40.41 | 37.80 |

由这组数据可以确定：

1. 默认方向下，完整模型的 Blind Null 比 MVDR 多抑制约 1.86 dB 的干扰分量，但其目标输出相对 SINGLE 损失约 8.6 dB，输出热噪声还增加约 1.4 dB，因此失捕。仅看零陷深度会得到错误的算法判断。
2. 默认方向没有遮挡作用，`full_known/full_parametric` 的目标、噪声、干扰功率和捕获峰比逐 seed 完全一致。不能把该场景下的差异归因于未知遮挡相位。
3. 遮挡方向下，仰角增益单因子使 MVDR 真码热噪声相关 C/N₀ 相对理想低约 6.63 dB，方位增益单因子低约 1.81 dB，遮挡幅度单因子低约 2.66 dB。完整模型平均输入目标增益约 −9.70 dB；即使导向完全匹配，当前固定参数下仍三次失捕。导向匹配（包括幅相标定）不保证补回已丢失的目标功率或捕获余量。
4. 未建模遮挡的 MVDR 只有 seed 43 的峰比 2.5058 略过 2.5 门限，不能用 1/3 对 0/3 宣称失配更好。完全匹配模型的真码热噪声相关 C/N₀ 平均高约 0.85 dB，但固定权值的剩余干扰、非线性残差和有限样本会共同影响捕获。
5. 已正确纳入导向的本次小幅 PCV 单因子，对 MVDR 的相关 C/N₀ 变化很小（默认约 −0.04 dB、遮挡方向约 +0.01 dB）。这只检验已标定的合成 PCV，不代表未知通道相位/时延可以忽略。不同因子的相位、幅度与耦合交互改变空间响应，完整模型结果不能由单因子差值简单相加。

此外，EGC 在默认方向的仰角增益单因子、以及遮挡方向的姿态/极化单因子下各正确捕获 3/3，其余配置均为 0/3；SINGLE 全部为 0/3。没有通过调整功率或接收门限制造这些结果。

遮挡场景的两种完整模型使用同一真实信号和噪声，Blind Null 的权值和输出相同；表中 40.25/40.41 dB 的微小差别来自 SINGLE 按不同假设导向做电压归一化后，比较基准略有不同，不是 Blind Null 利用了目标导向信息。

全部 240 个分支的 ADC 独立重建最大误差为 0，合并逐点误差最大 `9.06e-15`，样本功率与 `wᴴRw` 的最大相对误差 `2.10e-14`。全套默认代码测试为 587 passed、25 local-evidence deselected；测试临时目录使用项目内可写位置，避免沙箱临时目录的文件替换权限限制。五张图已逐张检查，源码哈希与运行清单一致。以上是模型内小样本证据，不是硬件或商用验证。

## 源级响应直接赋值与边界扫描

```powershell
python -B -m experiments.interference.direct_response_boundary
```

[direct_response_boundary.py](../../experiments/interference/direct_response_boundary.py) 是独立实验入口，不改变 `full_chain` 的参数化方向图或默认运行方式。它使用理想天线基底，保留手机四角几何相位；目标、CW 和宽带干扰分别指定每个天线的非几何电压响应：

```text
c_i(source) = 10^(gain_db_i/20) × exp(j × phase_deg_i × π/180)
a_i(source) = a_ideal_geometry_i(source) × c_i(source) × channel_response_i
```

`gain_db` 对应功率增益的 dB 数值，作用到电压时除以 20；`phase_deg` 是额外响应相位，不包含已经由几何、硬件通道幅相和通道时延计算的相位。直接指定的系数实际乘到各源信号序列及其真实/假设响应上，而不只用于绘图或修改标签。理想基底没有 cosine 增益、方位衰减、遮挡、PCV 或互耦，因此可以隔离所赋参数的作用；这些因素仍由原参数化入口研究。

入口顶部集中编辑：

- `ANTENNAS`：任意 2～4 个角点组合，行顺序按该元组；`TL/TR/BL/BR` 为左上/右上/左下/右下。
- `DESIRED_GAIN_DB`、`DESIRED_PHASE_DEG`：目标各角点的基础幅相；CW 与宽带各有独立的 `*_GAIN_DB`、`*_PHASE_DEG` 字典。
- `TRUE_CHANNEL_DELAY_NS`、`ESTIMATED_CHANNEL_DELAY_NS`：实际/估计硬件通道时延，单位 ns。真实时延仍延迟整个 IF 序列并补上被下变频移除的 RF 相位，且作用于所有源；估计时延只进入目标导向的窄带相位，**不会自动对齐 DFE 输出的码包络**。
- `PERTURBED_ANTENNA`、`PHASE_ERRORS_DEG`：给指定天线的真实目标响应追加相位偏差；`SEEDS`、`DURATION_MS`、`EVALUATION_EDGE_TRIM_MS`、`SHOW_PLOTS_FLAG` 控制实验和显示。

每个扫描点有两组：`known_response` 的目标假设系数包含真实追加相位；`unmodeled_phase` 只知道基础系数，未建模该追加相位。两组使用相同的实际信号、噪声、RF 和 DFE 数据，只改变接收机导向假设；Blind Null 不获得目标真值。这里的“已知”仅指源级指定系数，不意味着估计硬件时延必然准确，也不意味着已完成序列对齐。

默认四天线、扰动 TR、seed 42、100 ms，扫描 0/20/40/60/90/120/180°，共 14 条接收记录、56 个分支评估。保留目标 −128.5 dBm、CW/宽带各 −90 dBm、原 NF 和捕获门限；多径关闭。复用消融实验的独立 RF/ADC 重建和冻结权值功率预算，一次运行完成接收、核验、统计与三张分量分开的图，输出在 `results/interference/direct_response_boundary/`。`manifest.json` 保存条件、响应定义与源码哈希，`summary.json/csv` 保存逐分支结果。真码热噪声相关 C/N₀ 与实际跟踪 C/N₀ 分开显示；未捕获分支没有跟踪估计，不填成 0 dB-Hz。

范围限制：本轮默认只扫描单个天线的静态目标相位，增益和时延参数可编辑但未做完整边界扫描；一个 seed、100 ms 的结果不是统计性能边界。源级常数响应只定义指定来波，在宽带内保持不变，不代表实测宽带频响。各源幅相可以独立赋值，不保证对应同一副真实天线的全角度方向图；因此该入口不绘制未定义的二维/三维全角度响应。

大时延评分另有前提：目前正确捕获的码误差仍相对配置中的原始 `code_delay_chips`，真码热噪声相关也沿用未作通道序列校准的时间参考。若所有通道共同延迟超过 0.5 chip，接收机可能正确捕获延迟后的信号，却被这个原始时间参考判为“码误差超限”；相干 oracle 也可能出现参考失配损失。后续做大时延边界实验时应先定义接收/校准后的参考时刻，再区分共同时间偏移与通道间未对齐，不能直接把现有标签解释为算法失捕。本轮零硬件时延的相位实验不受该问题影响。

### 2026-10-13 直接响应相位扫描验证

实验条件按上述默认值执行：TL/TR/BL/BR 四路，TR 真实目标追加相位，seed 42、100 ms，目标 −128.5 dBm，CW 与 2 MHz 宽带各 −90 dBm；所有基础源增益/相位与真实/估计硬件时延为 0，多径关闭。每个扫描点分别运行已知与未建模追加相位两组。功率预算去掉两端各 10 ms，接收机仍处理完整记录。没有改变信号功率、噪声系数或接收门限。

下表是实际捕获峰比，不是方向增益；门限仍为 2.5。Blind Null 在同一扫描点的两种导向假设下结果完全相同，因此只列一列。

| TR 追加相位 / ° | 已知相位 MVDR | 未建模相位 MVDR | Blind Null（两组相同） | 未建模 MVDR 目标功率相对 SINGLE / dB |
|---:|---:|---:|---:|---:|
| 0 | 20.898 | 20.898 | 5.521 | 0.00 |
| 20 | 20.815 | 19.838 | 5.353 | −0.38 |
| 40 | 17.549 | 17.624 | 5.683 | −0.99 |
| 60 | 15.328 | 14.779 | 6.440 | −1.85 |
| 90 | 12.452 | 9.813 | 8.395 | −3.59 |
| 120 | 11.782 | 6.359 | 10.555 | −5.71 |
| 180 | 16.178 | 4.910 | 13.556 | −6.43 |

本次可得出的结论：

1. 已知/未建模两组的 MVDR 和 Blind Null 在七个扫描点都正确捕获；SINGLE 与 EGC 都未捕获。这里的 7/7 是七个不同相位点、同一个噪声 seed 的计数，不是七次随机重复，也不是捕获成功率估计。**尚未找到失捕边界**。
2. 已知相位的 MVDR 在零时延条件下保持目标线性输出功率相对 SINGLE 约 0 dB；未建模相位在 180°时损失 6.43 dB。对应真码热噪声相关 C/N₀ 为 48.31 / 43.45 dB-Hz，实际跟踪后半段有限值中位数为 47.97 / 43.38 dB-Hz。相位失配的影响已在实际信号与接收结果中出现，不只是导向标签变化。
3. 在 120°与 180°两个采样点，未建模 MVDR 的峰比低于 Blind Null；已知相位 MVDR 的峰比仍高于 Blind Null。不能据此推出“Blind Null 总比 MVDR 好”或“相位准确就一定成功”。峰比在 40°附近还有小幅交叉，不能要求有限样本峰比随失配严格单调。
4. Blind Null 随真实目标相位变化而变化，但同一真实输入下，已知/未知导向假设的分量预算、捕获及跟踪结果完全一致。它没有使用目标导向真值；相位扫描改变了目标与干扰的相对空间响应，因而也会改变盲投影保留的目标功率。
5. 本次 CW 与宽带线性分量的抑制度一致：两者同方向、源级系数相同、硬件时延为零，模型在带内采用同一空间响应。这不是任意宽带干扰的结论，也不意味着估计时延已经对齐码包络。约 74～77 dB 的 Blind Null 线性分量抑制不代表含非线性残差的完整输出也改善同样多。

实际执行与核验：

- `MPLBACKEND=Agg` 下执行 `python -u -B -m experiments.interference.direct_response_boundary`，最终完成 14 条记录、56 个分支。输出为 `manifest.json`、`summary.json/csv`、`acquisition_metric.png`、`target_and_interference.png`、`cn0_oracle_receiver.png`；三图逐张检查，指标各自分行，已知/未知分列，算法颜色一致。默认交互运行仍按 `SHOW_PLOTS=True` 保存后弹窗；本次无窗口验证只保存。
- 所有分支独立 ADC 重建最大误差 0，逐样本合并误差最大 `8.88e-15`，样本功率与 `wᴴRw` 最大相对误差 `2.03e-14`。清单的 128 个来源文件 SHA-256 与实际运行源码逐项一致。
- `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_direct_response`：**599 passed、25 deselected**；25 项历史本地证据测试按默认配置排除，未重新执行历史 60 条消融或 RF 噪声标定。
- 新测试核对源级赋值、真实/假设分离、2/3/4 路及行重排、非零时延下系数只作用一次、33 点实际序列延迟、零系数注入与原接收链逐点一致、同输入的 Blind Null 一致性，以及真实绘图横坐标与清单一致。
- 迭代中修正了新增模块尚未实现时的导入失败、时延手推值的浮点单位表达、Path 清单序列化，以及函数默认参数提前绑定导致的配置/绘图不同步。最后一项先用“替换为 3 dB 后仍得到旧 0 dB”测试复现，修复后新增目标文件 **9 passed**；只读复审无未解决的实质性问题。首轮默认批次在五个 case 后主动中断，没有当作完成结果；最终从头重跑，保留的清单与结果均来自最终代码。

## 阵列响应绘图

二维图下方的图标图例同样显示各方向的方位角、仰角和具体 dB 响应值，与三维曲面图的数值一致。数值按真实方向直接计算，不取最近的网格点，也不受色图显示下限截断。该值是相对扫描峰值的方向响应，不是干扰输入/输出总功率的抑制比。

`full_chain` 保留各合并分支的二维响应图，并同时输出独立的 `detailed_array_response_3d_{mode}.png` 三维曲面图。方位角为 0～360°，仰角为 0～90°，间隔为 2°；高度与颜色均表示 `20 log10(|wᴴa| / max|wᴴa|)`，每个分支按自己的扫描最大值归一化。目标和干扰方向始终标出；开启多径研究后才标出回波方向。

显示下限为 −60 dB，低于该值的响应只在绘图时截到 −60 dB；二维和三维使用相同的扫描数据。该图展示当前天线模型、目标极化及载频下的固定权值空间响应，不代表绝对增益或宽带内所有频率的抑制度。入口默认 `SHOW_PLOTS_FLAG = True`，保存全部图片后同时弹窗显示，三维图可拖动视角。设置 `SHOW_PLOTS_FLAG = False` 可只保存不显示；无窗口的 Agg 后端只输出文件。

另输出 `detailed_array_response_spherical_{mode}.png`。扫描仰角 −90～90°，仍计算全空间响应；为看清深凹陷，两个相反视角分别画各自面向观察者的半球，避免背面曲面在投影中把前方零陷填实。这是固定半球剖视，不是完整不透明球壳；拖动视角只旋转该剖面，不会自动重建被隐藏的背面。两视图互补，未删除或改变背面响应数据。

当前球形图采用 **dB 显示半径**，不再用线性幅度作为半径：`G_dB=20 log10(|wᴴa|/max|wᴴa|)`，`r=clip(1-G_dB/G_floor,0,1)`，默认 `G_floor=-40 dB`。因此 0/−10/−20/−30/≤−40 dB 对应 1/0.75/0.5/0.25/0，增益越低越朝球心凹入；不能用负 dB 的绝对值作为半径，否则抑制越深反而越向外。`SPHERICAL_RESPONSE_FLOOR_DB` 在 `core/plotting.py` 集中设置显示范围。

此半径只为便于看图，不是归一化电压或功率、绝对增益，也不改变仿真数据。低于显示下限的零陷在几何上都缩到原点，不能靠半径区分 −40 与 −60 dB；图例仍直接计算真实方向的 dB 数值，不截断、不取最近网格值。颜色使用低增益冷色、高增益暖色，叠加光照明暗和细网格帮助辨认曲率；阴影不是额外增益损失。理想各向同性单天线仍为单位球，明暗变化只来自光照，不人为制造凹凸。

“向内凹”是相对 0 dB 参考球半径收缩，不保证每个低增益方向都为局部极小：弱旁瓣若仍比邻近方向强，也可以形成位于参考球内部的小凸瓣。不能为了画成单个球加几个坑而抹掉真实旁瓣或改变增益排序。

虚线圆是该视角下 0 dB 参考球的投影轮廓，中心黑点表示球心；所有实际显示半径不超过 1。目标用绿色星形、CW 用红色叉号、宽带干扰用品红菱形、多径用蓝色圆形，不标数字；图标移到参考球外侧半径 1.06，仅表示方向，不表示增益，避免大图标被误看成低增益凸起。连接虚线的内端落在精确方向的显示半径上；图例保留方位角、仰角和 dB。图标/引线是叠加的方向辅助，不宣称严格曲面遮挡。

首视角方位比第一个启用干扰偏转 25°，仰角 25°，便于斜看凹陷；另一视角取其反向，无干扰时以目标方位为基准。移除占空间的坐标框并放大图体。二维和原有三维曲面图不变，仍按上半空间峰值归一化；本图按全空间峰值归一化。下半空间响应由模型实际计算，不镜像上半球；实测模型须确认覆盖下半空间，表外插值不能视为实测。

### 2026-10-13 凹凸三维方向图验证

本轮只修改球面方向图的几何半径、方向标记和说明，不修改天线响应、RF/DFE、合并算法或接收门限。新增测试先在旧实现上得到 2 failed、1 passed：非均匀响应仍被画成单位球，离网格方向标记仍固定在 1.03 半径；修改后这三项全部通过。测试验证零陷半径不受颜色下限影响、标记按真实离网格响应定位，以及各向同性响应仍为球形。

实际绘图使用四天线 `synthetic_phone_corner_v2`、seed 42、100 ms、目标方向 30°/60°、默认 CW/宽带干扰、多径关闭。新版四张图保存到 `results/interference/gain_shaped_response/`，不覆盖既有 `full_chain` 或消融证据；正常重跑 `full_chain` 会在原文件名下生成新版图，并按现有 `SHOW_PLOTS=True` 保存后弹窗。本轮不重新执行 60 条天线消融，历史统计结论保持原样。

验证结果：默认回归命令 `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_gain_shaped` 为 **590 passed、25 deselected**。四张实际图片检查后发现反向视角的长坐标标签靠近标题，缩短坐标标签并增加标题间距；此排版微调后，`python -B -m pytest -p no:cacheprovider -q tests/test_plotting_runner.py --basetemp=results/_pytest_gain_shaped_final_plot` 为 **11 passed**，并重新生成四张图。运行中逐项核对绘图前后的指标字典及各分支合并权值，完全一致；默认当前噪声模型下 MVDR 峰比仍为 12.864、其余三个分支失捕，不把旧 RF 模型的结果当作此次绘图改变造成的变化。`git diff --check` 无空白错误。该验证只证明绘图映射和接收流程未被绘图修改，不新增抗干扰性能结论。

### 2026-10-14 球形图内凹表达与阅读性修正

排查发现原线性幅度半径没有方向反转：−20 dB 对应 0.1、低增益确实靠近球心。问题是低响应压成小团、图标相对曲面过大、背面曲面在投影中遮住凹陷，加上大坐标框占空间，容易被看成低增益向外凸。本轮仅替换显示映射与渲染，不修改天线、RF/DFE、权值、捕获跟踪或实际响应数值；2026-10-13 的线性半径图片和历史核验记录保留原样，当前绘图按本轮 dB 半径规则执行。

实际改动为：0 dB 参考轮廓与球心标记、dB 显示半径、精确响应内端和外置方向图标、冷低暖高的曲面光照/网格、扩大图体与移除坐标框，以及两个互补的固定半球剖视。首次新图检查后，背面仍可能填住前方凹陷，因此继续加入显示遮罩；剖视/光照是可视化方法，不是新物理模型，也不是算法抑制度变化。

验证条件：四天线 `synthetic_phone_corner_v2`，seed 42、100 ms，目标 30°/60°、默认 CW/2 MHz 宽带干扰、多径关闭。用同一接收结果及相同全空间 2°网格生成四个分支图片；四图逐张检查，最终图写入 `results/interference/spherical_readability/detailed_array_response_spherical_{mode}.png`。同目录 `manifest.json` 保存配置、源码 SHA-256、权值实部/虚部及映射说明，`summary.json` 保存接收指标。原 `gain_shaped_response/`、`full_chain/`、消融和直接响应实验结果不覆盖；用户正常重跑 `full_chain` 后，原文件名会生成新版图，保存/弹窗规则不变。本轮用 Agg 后端无窗口生成，未核验 GUI 拖动。

实际执行结果：

- 半径/外置图标/参考轮廓新期望在旧实现上为 **3 failed**；第一版显示修改后球形相关测试 **5 passed**。随后背面遮罩测试在未遮罩版本上 **1 failed**，加入固定半球剖视后 `python -B -m pytest -p no:cacheprovider -q tests/test_plotting_runner.py -k spherical` 为 **6 passed、7 deselected**。
- `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_spherical_readability` 为 **601 passed、25 deselected**；历史本地证据测试按默认配置排除。只读审查未发现实质性或阻塞问题；`git diff --check` 无空白错误。
- 实际绘图前后逐项核对接收指标和全部分支权值，完全一致；128 个来源文件哈希与最终源码一致。MVDR 峰比仍为 12.863909，其余三分支仍失捕。本轮不重新执行历史消融或相位扫描，也不新增抗干扰性能结论。

限制：半径并非线性幅度，≤−40 dB 的几何深度不再区分；深度看图例真实值。图标和引线是方向辅助、不是严格遮挡判定。固定半球遮罩不会随 GUI 相机旋转自动更新；两视图互补展示原始方向集合，但半球切边可能出现网格锯齿，不能把它视为真实波瓣边界。

### 2026-10-14 无干扰下 EGC/MVDR 的小差异核对

目的：区分“从含干扰数据求权、事后只评价目标/热噪声”与“实际关闭全部干扰”的实验，并检查无干扰时 EGC 是否明显优于 MVDR。本轮读取到磁盘 `full_chain/manifest.json` 与 `direct_response_boundary/manifest.json` 仍各有两个启用干扰，未取得用户所指无干扰运行的具体图片/配置。因此下面是独立控制组，不声称已复现用户那次运行。

控制条件为 `default_config()` 理想四角、真实/假设响应一致、零通道时延、seed 42、100 ms、目标 −128.5 dBm；进程内设置 `cfg.interferers=[]`、关闭并清空多径，只运行 SINGLE/EGC/MVDR。接收处理完整记录，独立线性目标/热噪声评分去掉首尾各 10 ms；实际跟踪 C/N₀ 取后半段有限值中位数。未修改源码或持久配置，没有写结果文件或绘图，只将 stdout 数值记在此处；默认入口条件不变。

| 指标 | EGC | MVDR |
|---|---:|---:|
| 权值范数平方 | 0.2500000000 | 0.2500005234 |
| 线性分量热噪声 SNR / dB | −15.324913 | −15.324910 |
| 真码热噪声相关 C/N₀ / dB-Hz | 50.326788 | 50.326124 |
| 后半段实际跟踪 C/N₀ 中位数 / dB-Hz | 51.332753 | 51.339575 |
| 捕获峰比 | 21.996093 | 21.982893 |

两者目标真实/假设导向的最大差为 0。MVDR 相对 EGC 的权值向量差约 0.1447%；按理想独立同功率热噪声与相同目标单位响应计算，额外白噪声损失仅 `9.09e-6 dB`。相关 oracle 上 EGC 高 `0.000665 dB`，跟踪短窗估计则 MVDR 高 `0.006822 dB`，并不存在一致、明显的 EGC 优势。

解释与边界：独立同功率空间白噪声且响应准确时，`w=a/(aᴴa)` 是保持目标单位响应的最小范数权值。当前四路等幅，所以它也是相位对齐的等权平均；代码名为 EGC 的分支在幅度不等时实为 MRC 型权重，而非严格等幅 EGC。精确总体协方差下，MVDR 会得到相同权值；实际代码使用有限记录的 `XXᴴ/N`，与理想总体协方差不同，oracle 和跟踪的短窗评分又不是该全记录协方差优化目标，因此微小偏差或指标次序变化不能据此认定算法有问题。[MathWorks 的 MVDR 协方差估计说明](https://www.mathworks.com/help/phased/ref/phased.mvdrbeamformer-system-object.html)

本次结果只覆盖一个 seed、零额外相位的理想无干扰控制；不证明其他增益、时延、方向失配或任意配置下的等价，也未诊断用户未提供的具体运行。没有改算法，未重跑 pytest、消融、相位扫描或标定。

实际执行为在 `python_sim_simplified/` 下用 `python -u -B -c '<Python 代码>'` 运行；以下是该内存诊断代码的排版展开，不另建脚本：

```python
from dataclasses import replace
import numpy as np
from core.presets import default_config
from core.pipeline import run_end_to_end
from core.dfe import process_dfe
from core.rf import filter_rf_input, generate_rf_noise
from experiments.interference.antenna_ablation import thermal_oracle_prompt_cn0

cfg = default_config()
cfg.interferers = []
cfg.enable_multipath_research = False
cfg.signal.multipath_paths = []
cfg.signal.duration_ms = cfg.tracking.max_epochs = 100
cfg.diversity.modes = ('single', 'egc', 'mvdr')
r = run_end_to_end(cfg)
info = r['rf_info']
a = r['truth']['desired_steering_true']
fs = r['dfe_info']['fs_out_hz']
scale = r['dfe_info']['output_scale']
fcfg = replace(cfg.dfe, output_bits=0)
gain = (10 ** (info['channel_total_gain_db'] / 20)
        * np.exp(1j * np.deg2rad(info['phase_error_deg'])))[:, None] / info['adc_lsb_v']

def linear_dfe(v):
    output, _ = process_dfe(v * gain, fcfg, cfg.signal.fs_rf_hz, cfg.signal.if_hz)
    return output / scale

signal = linear_dfe(filter_rf_input(
    r['truth']['desired_component'], info['filter_coefficients'],
    cfg.signal.fs_rf_hz, cfg.signal.if_hz))
noise = linear_dfe(generate_rf_noise(
    r['scene_input_v'].shape[1], info['noise_psd_w_per_hz'],
    cfg.reference_impedance_ohm, cfg.signal.fs_rf_hz, cfg.signal.if_hz,
    cfg.phone.active_antennas, cfg.random_seed, info['filter_coefficients']))
trim = round(0.01 * fs)
interval = slice(trim, -trim)
for mode, branch in r['branches'].items():
    w = branch['combining']['weights']
    target, thermal = w.conj() @ signal, w.conj() @ noise
    snr = 10 * np.log10(np.mean(abs(target[interval]) ** 2)
                        / np.mean(abs(thermal[interval]) ** 2))
    oracle = thermal_oracle_prompt_cn0(target, thermal, r['truth'], fs, trim)
    cn0 = branch['tracking']['cn0_db_hz']
    print(mode, np.vdot(w, w).real, snr, oracle,
          np.nanmedian(cn0[len(cn0) // 2:]), branch['acquisition']['metric'])
w0 = r['branches']['egc']['combining']['weights']
w1 = r['branches']['mvdr']['combining']['weights']
print('relative_weight_difference', np.linalg.norm(w1 - w0) / np.linalg.norm(w0))
print('white_noise_penalty_dB', 10 * np.log10(np.vdot(w1, w1).real / np.vdot(w0, w0).real))
print('matched_steering_error', np.max(abs(a - r['truth']['desired_steering_assumed'])))
```

## 2026-10-14 双天线整通道相位补偿闭环

### 本轮范围与实施约定

本轮只验证恒定、频率无关的复数通道相位误差及其准确补偿。入口为 `experiments/interference/channel_phase_validation.py`；历史 `direct_response_boundary` 是目标源级响应变化实验，不作为本轮通道校准证据，也不覆盖其数据。

误差放在 **DFE 输出之后、天线合并之前**，对整行复数序列施加同一个相位旋转。目标、CW、热噪声以及已有前端量化残差共同旋转。此位置用于隔离相位补偿的线性数学闭环，不等同于在 RF 限幅/ADC 前注入误差，不验证硬件校准能力或真实手机天线。

设基准数据为 `X0`，通道相位矩阵为 `D = diag(exp(jφ))`：

| 组别 | 合并器输入 | 接收机使用的导向 |
|---|---|---|
| `baseline` 无误差基准 | `X0` | 基准目标导向 `a0` |
| `uncalibrated` 有误差、未补偿 | `D @ X0` | 仍使用 `a0`，故存在失配 |
| `calibrated` 同样误差、准确补偿 | `D_est^-1 @ D @ X0`，本轮 `D_est=D` | 使用 `a0`，不重复补偿导向 |

补偿作用于整个接收序列，不是只调整本地码或目标分量。准确相位值是本轮人为提供的已知校准量；接收机没有自动估计相位。

- 理想双天线默认 `TL/BR`，参考为第一根，误差施加到 `BR`；顶部可改为任意组合。默认对角组合用于减少本轮目标/CW 在短横基线上的空间响应接近问题，保留位置导致的几何相位，真实/假设几何匹配。
- 固定 `noise_only` 和 `single_cw` 两个场景，后者只保留原 CW；不增加宽带干扰、方向图、互耦、遮挡或多径。
- seeds 为 `42/43/44`，每条记录 100 ms；扫描 `0/30/60/90/120/180°`；目标功率、CW 功率、NF、采样率和捕获门限沿用原配置。
- 每个 seed/场景只生成一次 RF/DFE 基准。全部相位点、组别、SINGLE/EGC/MVDR 使用配对数据和同一个参考通道，避免重新抽噪声造成伪差异。
- 独立重建目标、CW 和热噪声的线性 RF/DFE 分量；实际 DFE 与线性总和的差单列为非线性/量化残差，不当作热噪声。
- 分开显示热噪声合并收益、CW 抑制、带内 `C/(N+I)`、实际捕获与跟踪 C/N₀；带内 SINR 不是 C/N₀，纯热噪声指标不是抗干扰综合成绩。

实施顺序：先写相位整行旋转/逆旋转、独立白噪声收益和短入口的失败测试，再实现实验；运行默认批次及回归、独立审查与绘图检查后追加实际结果。严格检查补偿序列/输出恢复及独立功率重建；不要求未补偿组通过捕获，不为获得成功而调整信号强度或门限。

线性独立等功率噪声下，双天线等幅同相合并的理论热噪声收益为 `10 log10(2)=3.0103 dB`。有限数据的实测功率和短窗跟踪估计允许统计差异；先判断与理论和基准的一致性，不用某次 C/N₀ 排名代替校准验证。RF/ADC 前相位误差及量化、限幅的非交换性留作下一项独立复验。

对于双路等幅目标，第一路无额外相位、第二路有未补偿的 `φ` 时，EGC 的目标电压响应为 `(1+exp(jφ))/2`，功率响应为 `cos²(φ/2)`。因此 90° 时目标功率比基准低 3.0103 dB，180° 时理论抵消；准确逆旋转后恢复。该手推关系核验的是目标失配损失，与 CW 抑制不能混为一个指标。

### 实际验证与结论

本轮代码阅读顺序如下，参数集中在入口顶部：

1. `build_config()`：固定理想双天线、无干扰/单 CW 两个场景，不改功率、NF 和门限。
2. `run_sweep()`：每个 seed/场景只生成一次 RF/DFE 输入和基准接收结果，再遍历相位点。
3. `apply_channel_phase()` / `_case_channels()`：按物理天线名旋转整行数据，对同一份受扰数据逆旋转；参考通道不动。更换双天线组合时同步设置 `PERTURBED_ANTENNA` 为非参考通道。
4. `_receive()`：调用既有合并、捕获和跟踪算法；不读取评分用的真值分量，也不将精确相位交给相位估计算法。
5. `_linear_components()` / `_score_branch()`：独立重放线性目标、CW、噪声，固定已求出的权重核算功率；联合非线性/量化残差另列。
6. `verify_rows()`：对照每条补偿组与同 seed/场景/相位/方法的基准；核验数据、权值、功率、捕获和跟踪状态及数值。
7. `plot_summary()` / `main()`：一次运行完成数据生成、核验、保存和三张图，核验失败会写明失败项并以非零退出结束。

运行：在 `python_sim_simplified/` 内执行 `python -m experiments.interference.channel_phase_validation`。只刷新 `results/interference/channel_phase_validation/`，不覆盖旧目标响应扫描。`summary.json/csv` 为逐条件结果，`verification.json` 为核验判据与配对差，`manifest.json` 保存实际条件、指标定义和源码哈希。

三图按方法分行、场景分列，不把所有方法挤在同一坐标轴：

- `thermal_gain.png`：显示线性热噪声 SNR 收益倍率，约 2 对应约 3.01 dB；表格仍保存完整 dB 值。180° 理论抵消在浮点中产生约 −316 dB 的残值，若直接画 dB 会压扁其他相位点，因此仅改变显示单位，不截断或修改数据。
- `target_and_cw.png`：目标功率比与 CW 抑制分开；“无 CW”明确标出。未补偿 MVDR 的目标功率比可能超过 1，因为它此时仅约束错误的目标导向；不把放大目标误认为正确校准。
- `receiver.png`：实际捕获峰比含门限线，跟踪 C/N₀ 缺测不记为 0，缺口/无曲线有说明。曲线按每点可用 seed 值求均值，不是成功概率；补偿组与基准重合是闭环预期。

`SHOW_PLOTS=True` 默认保存并弹出图；本轮自动验证使用 `MPLBACKEND=Agg`，只核验保存图，尚未实际测试桌面弹窗交互。

实现中的首次模块缺失、评分更新顺序、Path/numpy bool JSON 序列化问题已修正。独立审查又复现了“基准有有限 C/N₀、补偿组无值仍被核验为通过”的缺口；已先负测试再增加跟踪成功、可用性和有限样本数的配对检查，只允许双方都不可评价时跳过数值差。图 QA 中的 dB 压缩和空曲线横轴问题也先复现再修正。

实际执行：

- 修改前 `python -B -m pytest -p no:cacheprovider -q tests/test_rf_dfe.py tests/test_pipeline.py tests/test_antenna.py --basetemp=results/_pytest_phase_baseline`：31 passed。
- 新测试初次因模块缺失而收集失败；首次实现 4 passed，修正核验缺口与图显示后 `python -m pytest tests/test_channel_phase_validation.py -q`：5 passed，包含篡改目标功率、单侧 C/N₀ 缺失、跟踪成功状态改变的负测试，以及线性显示/缺测坐标轴检查。
- 首次默认扫描完成 324 行，初次完整回归 605 passed、25 deselected；其后修正核验与图显示，没有用初轮结果冒充最终版本。
- 最终 `MPLBACKEND=Agg` 下执行 `python -u -B -m experiments.interference.channel_phase_validation`，退出码 0；6 份基准 RF/DFE、36 个 seed/场景/相位配对条件、324 行方法结果完成，21 项核验通过，108 个补偿/基准方法配对一致。
- 最终 `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_channel_phase_final`：**606 passed、25 deselected**，252.35 s；本地归档证据测试按既有配置排除。
- Main 用 `python -B -` 内存脚本从角点坐标与方位/仰角独立计算 `exp(-j2π f r·u/c)`，不调用天线响应或相位助手：目标功率比最大绝对差 `1.78e-15`，CW 抑制最大差 `4.97e-14 dB`；EGC 对 `cos²(φ/2)` 的最大绝对差 `3.33e-16`。最终清单全部 **129** 个来源哈希匹配，三张保存图逐张检查，`git diff --check` 无空白错误。

结果：

| 条件 | 实际结果 | 判断 |
|---|---|---|
| 无干扰、无误差/准确补偿 | EGC 热噪声收益 2.9986～3.0073 dB；与理论最大差 0.01167 dB。MVDR 为 2.9987～3.0073 dB | 两者接近，同样满足双天线基准 |
| 无干扰、90° 未补偿、EGC | 目标功率保留 0.5 | 符合手推 −3.0103 dB 目标损失 |
| 无干扰、180° 未补偿、EGC | 目标功率保留约 `1.09e-32`；三个 seed 均失捕 | 浮点残值对应理论抵消，不是正常目标保留 |
| 单 CW、无误差/准确补偿、MVDR | CW 抑制 36.94～37.10 dB；三个 seed 的基准均正确捕获。六个相位点全部恢复各自基准 | 校准恢复空间抑制及实际接收；18 个配对成功不等于独立捕获概率 |
| 单 CW、无误差/准确补偿、EGC | CW 抑制 4.14 dB；三个 seed 均失捕，跟踪 C/N₀ 无值 | 相位校准正确不等于 EGC 具备足够抗干扰能力 |
| seed 42、单 CW、90° 未补偿/准确补偿、MVDR | CW 抑制 5.74→36.94 dB；捕获峰比 1.10→7.01，门限 2.5 | 未补失捕，补偿后恢复基准；不能只看热噪声指标 |

准确补偿后的 DFE 数据、合并输出相对基准最大误差分别为 `1.06e-16`、`8.70e-16`；权值逐值一致。捕获峰比最大绝对差 `1.07e-14`，可用跟踪 C/N₀ 最大差 `7.11e-14 dB-Hz`；CW 抑制最大差 `7.11e-15 dB`。这说明本轮闭环恢复到了浮点舍入精度。

结论只适用于本轮模型：**整通道恒定相位误差在准确逆补偿后恢复基准；补偿正确性与不同合并算法的抗干扰能力是两回事。** 没有实现自动相位估计、相位漂移、频率相关误差或整序列时延对齐；没有验证 RF/ADC 前误差或真实手机方向图/校准能力。三种 seed 与固定姿态/功率/方向不足以推断一般可靠性。主链路默认配置未修改，历史研究批次未重跑；未提交、推送或合并 Git。

## 2026-10-14 通用验证框架

日常入口为 `python -m experiments.interference.validation`，直接执行 `experiments/interference/validation.py` 也使用相同配置。只需编辑 [validation_config.py](../../experiments/interference/validation_config.py)，原有各实验入口和历史结果继续保留其独立身份。

常用修改位置：

| 目的 | 配置位置 |
|---|---|
| 选择两至四根天线、改变顺序 | `ANTENNAS`；第一根是 SINGLE 参考 |
| 选择合并方法 | `MODES`；默认 `single`、`egc`、`mvdr`、`blind_null` |
| 修改噪声重复次数、时长和统计去边 | `SEEDS`、`DURATION_MS`、`EVALUATION_EDGE_TRIM_MS`；统计窗口应在记录内部 |
| 扫描整通道相位 | `SELECTED_SERIES = ("phase",)`、`PHASE_ANTENNA`、`PHASE_SCAN_DEG` |
| 选择相位用例 | `BATCH = "legacy"` 的相位系列包含基准、未补偿、准确补偿三组；单独运行用 `SELECTED_CASES` 指定完整名称，准确补偿自动带基准 |
| 扫描目标真实方位角 | `SELECTED_SERIES = ("direction",)`、`TARGET_AZIMUTH_SCAN_DEG` |
| 扫描阵元真实增益 | `SELECTED_SERIES = ("gain",)`、`ELEMENT_GAIN_ANTENNA`、`ELEMENT_GAIN_SCAN_DB` |
| 设置卫星真实/名义方向及干扰方向 | `TARGET_TRUE_DIRECTION`、`TARGET_ASSUMED_DIRECTION`、`JAMMER_DIRECTION`；角度单位为度 |
| 只跑指定条件 | `SELECTED_CASES`；例如 `("single_cw/phase_90/calibrated",)`，构造用例时自动加入其参考条件 |
| 显示/保存图 | `SHOW_PLOTS_FLAG`、`OUTPUT_DIR` |
| 更改其他模型参数 | 同一文件内 `_base_config()` 直接赋值现有 `SimulationConfig` 字段 |

例如，把 `SELECTED_SERIES` 改成 `("direction",)`、`TARGET_AZIMUTH_SCAN_DEG` 改成 `(20, 30, 40)`，就运行这三个真实方位角。名义方向保持配置值，表示研究导向失配；若要研究已知方向，应在方向条件构造处同时明确赋值 `cfg.signal.assumed_direction`。多个系列分别扫描，不交叉展开为全部参数组合。每个条件独立复制基础配置，改一个条件不会污染其他条件。

需要自定义单点时，在 `build_cases()` 中增加具名条件并明确选择规则，最后统一展开种子；相位补偿条件在同处带入基准，基准先于补偿执行。`dfe_phase_deg` 和 `dfe_estimated_phase_deg` 按物理天线名配置；`compensate_dfe_phase_flag=True` 才执行估计相位的逆旋转。相位误差和补偿都作用于整行复数样本。真实阵元增益则修改 `cfg.phone.antenna_model.true_state.amplitude_error_db`，它改变目标和各干扰在天线端的响应，不改变独立 RF 热噪声的生成强度。

处理阶段固定在 [validation_support.py](../../experiments/interference/validation_support.py)：每个用例独立生成前端并执行接收和独立预算，同一用例的各算法共用本次采样。独立评分逐个重建启用的干扰源，不要求场景同时包含 CW 与宽带；公共接收机仍仅使用完整 IQ 和配置的名义导向。输出的线性 SINR 不含联合非线性/量化残差，分量交叉项单列，跟踪 C/N₀ 缺测不画成零。

本轮验证首先记录当前维护版的 20 ms、seed 42、默认四天线、CW＋宽带、12 跟踪历元数值，提取前端/接收函数后重算。场景、ADC、DFE 以及四种方法的权值、合并序列、捕获/跟踪序列化结果哈希全部一致。

### 2026-10-15 用例构造更新

`validation_config.py` 的 `build_cases()` 一次完成系列/具名选择、相位基准配对和 seed 展开；每个最终用例的 `config` 已包含实际种子、时长及跟踪上限。运行器按每个用例的完整配置独立执行，保存入口不再重建配置。加入 Blind Null 后默认是 36 个条件 × 3 个 seed × 4 种方法（432 行）；两种选择都为空时生成空批次，不再隐式运行全部系列。

`ValidationSettings` 只保留统计去边、输出目录和绘图开关；直接调用运行器的自定义脚本需先准备完整用例。输出行与指标字段不变；清单的 `actual_receiver_configs[条件名][seed]` 路径保留，`cases` 中的 `base_config` 改为逐 seed 的 `receiver_configs`，便于追溯实际执行参数。

本轮改前/改后对照：TL/BR、SINGLE/EGC/MVDR、seed 42/43、12 ms、两端各去除 1 ms，无干扰/单 CW，扫描相位 0/90/180°、目标方位 30/40/50°和 BR 增益 0/6.020599913279624 dB；168 行全部字段及完整核验报告严格一致。对照命令为 `$env:PYTHONPATH="."; python -B tests/case_refactor_probe.py`，本地证据在 `results/interference/case_refactor/`；未重跑完整历史批次。完整回归 `python -B -m pytest -p no:cacheprovider -q --basetemp=results/_pytest_case_refactor_final`：167 passed、2 deselected。

后续简化将选择、基准配对和种子展开全部放入 `build_cases()`，删除 `prepare_cases()` 和递归遍历。专项测试 25 passed；对照提交 `efd8a1d`，全部 46 个具名条件的单选、8 种系列组合及混合/重复选择共 57 组的完整配置和顺序一致，以上 168 行对照再次严格通过。本次未重跑全套核心测试。

### 2026-10-15 Blind Null 与入口整理

默认 `MODES` 新增 `blind_null`。`validation.py` 的 `main()` 依次读取设置、构造用例、仿真、保存、绘图和验收；`save_results()` 集中刷新输出目录和保存清单/JSON/CSV，直接复用结果中的实际配置，不改变输出格式，验收失败仍先保存证据再报错。

专项 `python -m pytest tests/test_validation.py -q`：26 passed。按上节 12 ms、seed 42/43、相位/方向/增益配置运行四方法得到 224 行，含 56 行 Blind Null；全部核验通过，原三方法的 168 行及检查与原基准严格一致。本地数据在 `results/interference/validation_blind_null/short_comparison.pickle`；未重跑完整默认批次或全套核心测试。

### 2026-10-15 验证运行器简化

此前版本的 `run_validation()` 负责物理分组、调用组内处理和汇总核验（后续独立执行方案已取消物理分组）；`_run_physical_group()` 只返回结果行，避免上一组波形和基准分支被局部变量带入下一组。相位注入/逆补偿系数供完整 IQ 与分量共用，仍执行两步乘法；同一条件的 DFE 恢复误差不再逐方法重复计算。

专项 `python -m pytest tests/test_validation.py -q`：29 passed，新增跨组数组弱引用释放、DFE 范数调用次数和 35° 两步相位逐位一致性检查。沿用上节四方法、两 seed、12 ms 的 224 行条件，完整摘要、配置及核验报告与改前严格一致；本地对照文件为 `results/interference/validation_blind_null/short_comparison.pickle` 和 `group_refactor.pickle`。未测量峰值内存或耗时降幅，未重跑完整默认批次或全套核心测试。

### 2026-10-15 逐用例独立执行

当前 `run_validation()` 按输入顺序调用 `run_case()` 执行信号生成、天线、RF/ADC/DFE、相位处理及接收，再调用 `evaluate_case()` 做分量分析、指标与配对比较；已删除 `_physical_key()` 和跨用例前端复用。同一用例的四种算法共用本次采样，基准与补偿用例用相同种子分别生成数据。默认 108 个用例对应 108 次前端、432 行方法结果；不再按物理分组调整行序。

基准缓存只供评分后比较，按 `(条件名, seed)` 隔离，仅保存 DFE 样本、各方法合并序列和权值，最后消费者结束后释放；不保存 RF 波形，也不把参考数据作为其他用例的接收输入。

专项 `python -m pytest tests/test_validation.py -q`：30 passed，覆盖独立前端及完整配置、名义参数和非零估计时延、不同 seed 配对、多个消费者与数组释放。沿用四方法、seed 42/43、12 ms、相位/方向/增益条件，56 个用例分别运行 56 次前端；按条件名/seed/方法对照，224 行、完整配置和核验报告与复用版严格一致。本地证据为 `results/interference/validation_blind_null/group_refactor.pickle` 与 `independent_cases.pickle`；未重跑完整默认批次或全套核心测试，未测量耗时增幅。

### 2026-10-15 接收与评估分离

同一 `validation_support.py` 内，`run_case(case)` 只执行实际前端、两步相位处理和接收，保留本次噪声但不调用分量分析；`evaluate_case()` 读取接收结果计算分量、指标和基准配对，合并原 `_score()`，不重跑实际链。批次循环评估后立即释放 `result`，仅配对必需的数据按既有规则保留。

专项 `python -m pytest tests/test_validation.py -q`：32 passed，覆盖接收不调用评估、评估不重跑或修改接收结果，以及原数组释放检查。沿用上节四方法、两 seed、12 ms 的 224 行条件，完整配置、数值、输出顺序和核验报告均严格一致；本地对照为 `results/interference/validation_blind_null/independent_cases.pickle` 与 `separate_evaluation.pickle`。未重跑完整默认批次或全套核心测试。

### 2026-10-15 绘图使用更新

现在每指标、每方法独立窗口，只横排比较场景。原三方法默认 18 张图，加入 Blind Null 后为 24 张：thermal_gain、target_gain、sinr、acquisition、cn0、suppression_0 分别加方法名后缀；多个扫描因素另加 factor 后缀。热噪声收益、目标功率相对 SINGLE 的差异、SINR 用 dB；C/N₀ 用 dB-Hz。捕获峰值比保持无量纲比值，门限仍为原始门限（默认 2.5），不转 dB。跨 seed 在各图所示单位下取算术平均。

不同条件使用不同线型和空心标记；曲线整段在所示单位下绝对差不超过 1e-8 时注明 Overlapping，不偏移真实数据。局部交点保持原始坐标。极低功率保留实际 dB，完全抵消为负无穷，不设置虚构下限。本轮只复用已有 summary 重绘，保留原仿真 manifest。下节四张/八张是旧布局记录，三方法更新布局后分别为 18 张/36 张；当前四方法分别为 24 张/48 张。

### 实际验证与结论（2026-10-15 完成）

- 默认批次：运行 `python -u -B -m experiments.interference.validation`，TL/BR、seed 42/43/44、100 ms、两端各去除 10 ms，6 个相位点、无干扰/单 CW、基准/未补/准确补偿。324 行方法结果、450 项逐行/理论/配对检查全部通过；这些检查不是 450 次独立实验。输出在 `results/interference/validation/`，包括 summary JSON/CSV、verification、manifest，以及 thermal_gain、target_interference、receiver、suppression_0 四张 PNG。
- 准确整通道补偿后，DFE 和合并输出相对参考的最大误差分别为 `1.0552e-16`、`8.7018e-16`，权值误差为 0。无干扰基准 EGC 热噪声收益 2.99863～3.00730 dB，符合双天线约 3 dB 的预期。基准和补偿曲线重叠是恢复成功，并非漏画。该结论以外部提供准确相位为前提，不代表已实现自动相位估计。
- 方向/增益示例：选择 `("direction", "gain")`，seed 42、20 ms、trim 2 ms，真实方位 30/40/50°，BR 增益 0/6.020599913279624 dB，名义参数保持不变。30 行、30 项检查全部通过，输出 `results/interference/validation_examples/`，两种因素分别绘图，共 8 张 PNG。独立用阵元坐标复算 EGC 目标功率比：方向扫描约为 1、0.937691、0.775836；增益扫描为 1、2.25，与输出最大绝对差 `2.22e-16`。真实增益变化不是估计参数自动跟随，更不是自动校准。
- 读图注意：目标输出功率变大不等于 SINR 改善，权值同时可能放大噪声；应结合热噪声收益、干扰抑制和捕获结果判断。跟踪 C/N₀ 缺测显示缺测，不以零代替，也不能仅凭剩余成功跟踪点评价捕获可靠性。预算中的线性 SINR 不含非线性/量化残差。
- 回归：完整测试 159 passed、2 deselected；最终绘图/注释调整后针对测试 17 passed。新旧相位入口 54 行数值对照通过。最终两批 manifest 的 47 个源码哈希均与当前源码一致。自动验证使用 `MPLBACKEND=Agg`；保存图已检查，桌面弹图交互未实测。信号强度、NF 及接收算法未为获得结论而修改。
