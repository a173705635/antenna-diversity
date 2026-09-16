# 源存在性分支226–250：25个机制闭环

冻结250在16类、每类100个新种子的提示模型上执行了1600次检测。11类无持续四源条件均0/100误放行；真实源条件存在明显漏检。此分支没有接入RF、码跟踪或导航，不能据此认定手机可持续定位或通过射频/完整性认证。

## 数据和接收机接口

`detect(z, a, noise)`只接收4天线×8PRN×240提示观测、名义天空导向矢量和独立离码噪声观测，不接收场景名、发射符号、真假源标记或硬件真值。合成器和独立评分器持有标签；合成器只用于建立可重复反例。每条提示按1 ms模型解释，前80提示用于训练，后160用于检测，两检测窗各80提示。固定阵列使用项目手机相位中心；天空几何固定，不代表几何普遍性。

开发种子82600+场景索引。226–240使用9类；241–244增加两公共源、检测噪声漂移、脉冲噪声、仅3真实源；245–250再增加移动公共源、与第一天空方向重合的公共源及AR2噪声，共16类。新增场景本身不计轮数，每轮仍有单独算法修改及同输入before/after。新增方向重合例中公共源与第一PRN重合，并非与全部天空方向重合。

每轮目录保存proposal.json、before.py、after.py、results.json、conclusion.json。源码逐轮实际修改后运行，并按输出决定下一修改；没有预写25个开关批扫。结果每条有观测+导向+离码噪声字节哈希。失败版本保留，返回父方案的谱系记录在proposal。

## 逐轮结果

|轮次|机制|结论|
|---|---|---|
|226|Independent unaligned holdout mean|同数据父150的IID与时间相关纯噪声放行被拒；公共源仍误放行。原始相干均值会受未知BPSK消去。|
|227|Noncoherent block energy likelihood|非相干能量移除未知比特抵消；判定没有改变，空间有色噪声最小p近零，不能宣称统计已校准。|
|228|Finite-reference F law|把有限参考噪声的不确定性纳入F分布，IID最小p由0.000380变0.001064；判定不变。|
|229|Familywise per-PRN hypothesis accounting|从单个PRN显著性改为多假设家族检验预算，数学零假设改变；判定不变。不是用开发结果选阈值。|
|230|Directional receiver noise normalization|同方向独立噪声归一化使空间无源最小p恢复0.101646；公共源仍误放行。|
|231|Independent full covariance whitening|独立训练噪声全协方差白化改变匹配滤波几何，判定不变；时间相关无源的p仍不可信。|
|232|Train-only common spatial nuisance elimination|公共源由放行变拒绝，但弱真实源也被拒；不能无条件删除最大特征向量。|
|233|MDL-selected single-source nuisance model|MDL仅秩一投影恢复弱源，但混合源总秩不止一，可能绕开公共消元。|
|234|Training sky-versus-common spatial competition|训练天空/公共方向竞争避免仅按总秩选择；开发判定不变。|
|235|Receiver-noise AR1 innovation filter|AR1创新把时间相关无源最小p从0.003458恢复0.091484；是拟合模型结果，不是精确独立性证明。|
|236|Bit-insensitive local coherent integration|固定10提示相干块再能量求和，未用检测相位自对齐；开发判定不变。|
|237|Train-only BPSK-squared frequency prediction|平方训练信号频率预测冻结后用于检测；判定不变，不能把频率扫描次数当研究轮数。|
|238|Partial-conjunction null for at least four sources|使用至多三有效源为零假设的partial conjunction p=min(1,5p(4))；依赖有效边际p，不能修复错误噪声模型。|
|239|Temporal intersection-union persistence test|双80提示窗口均须支持同一PRN，停发瞬态由放行变拒绝，持续弱源仍检出。|
|240|Shared stationary noise reference across persistence windows|在平稳独立参考模型下合并两个噪声参考窗，保持检测窗分离；判定不变。|
|241|Finite-training common-axis leakage budget|分半训练公共轴不确定性增加泄漏预算；发现两公共源及检测噪声增大仍误放行，预算非严格上界。|
|242|MDL-dimensional nuisance subspace elimination|全部MDL维度投影拒绝两公共源，但也拒绝gain和mixed真实源；后续返回241。|
|243|Orthogonal residual noise consistency model|空间正交残差一致性拒绝两公共源和噪声漂移；幅度失配真实源也被拒。|
|244|Training-only complex diagonal hardware model|训练复增益没有恢复gain，且mixed检出变拒绝；后续返回243。|
|245|AR innovation adequacy test over residual lags|检验AR创新剩余滞后1–5；新增AR2/移动公共源/近公共源挑战，开发分类不变。|
|246|Exact rank-based block-power source test|精确秩块功率检验具有分布鲁棒性前提，但近公共方向真实源检出丢失；后续返回245，不能宣传最终版本具有此鲁棒性。|
|247|Phase-only hardware nuisance calibration|单位模训练增益仍破坏mixed与near_common，后续返回245。|
|248|Per-source residual model validation before conjunction|逐PRN空间残差检验替代全局一票否决，gain检出恢复；仍要四个同时间支持源。|
|249|Train-extrapolated moving common-source null|训练公共方向切线外推无分类收益，common最小p下降，发现训练噪声外推问题。|
|250|Penalized constant-versus-linear nuisance motion model|常方向/线性运动模型用训练BIC竞争，静态公共源p恢复；冻结后不再调整算法。|

## 冻结留出

冻结清单source_freeze.json在运行留出前写入，记录核心、合成器、验证器及共享依赖哈希和NumPy/SciPy版本。种子92601+1000×场景索引+重复索引，每类100；冻结后没有改算法或根据留出重选版本。下表区间为逐类Clopper–Pearson双侧95%区间，不作跨16类同时覆盖保证。

|场景|放行/100|含义|95%区间|
|---|---:|---|---|
|iid|0|误放行|0.00%–3.62%|
|spatial|0|误放行|0.00%–3.62%|
|temporal|0|误放行|0.00%–3.62%|
|common|0|误放行|0.00%–3.62%|
|weak|95|真实源检出|88.72%–98.36%|
|strong|97|真实源检出|91.48%–99.38%|
|gain|64|真实源检出|53.79%–73.36%|
|mixed|70|真实源检出|60.02%–78.76%|
|transient|0|误放行|0.00%–3.62%|
|two_common|0|误放行|0.00%–3.62%|
|noise_drift|0|误放行|0.00%–3.62%|
|impulsive|0|误放行|0.00%–3.62%|
|three_sources|0|误放行|0.00%–3.62%|
|moving_common|0|误放行|0.00%–3.62%|
|near_common|86|真实源检出|77.63%–92.13%|
|ar2|0|误放行|0.00%–3.62%|

## 数学前提及尚未解决的问题

- 没有检测数据任意相位对齐后再按普通均值方差计分。频率、空间滤波及运动模型只使用训练块；独立噪声经过相同滤波。全协方差白化作用于观测与导向，不是仅逆增益后继续假定IID噪声。
- 单PRN能量F分布依赖圆对称复高斯、独立平稳创新和匹配噪声参考。rho由训练噪声估计，空间公共轴也估计；泄漏预算是启发式误差传播，不是覆盖概率证明。AR滤波首个检测提示使用训练末提示，存在一个样本的训练/检测边界耦合，冻结版没有消除。这些都限制严格p值解释。
- 双窗p取max是要求同一PRN持续存在的交并检验；p(4)为8个p中第四小值，min(1,5p(4))在有效边际p前提下检验“至多3源”，不要求PRN之间独立。它不能让失配的边际p自动有效。逐PRN模型拒绝将p设为1只降低放行机会，但失配模型通过本身不是认证。
- 最终没有采用246的秩检验；不能声称最终F版本具有分布无关保证。运动BIC和创新适配检验只是模型筛选。未建模非平稳共同干扰、多源协同、非高斯实际前端及捕获后的选择偏差。
- 11个无持续四源模型类各0/100不是零风险；每类区间上界3.62%。不能把异质无源类合并后较窄区间当成所有工况的虚警界限。固定几何、长度与参数使区间仅描述这些随机合成记录。
- 幅度失配漏检36/100、mixed漏检30/100、近公共方向漏检14/100；停止输出属于失败，不能忽略这些分母。纯强源也漏检3/100，弱源漏检5/100。该模块未提供码距或定位精度。
- 旧150原始4条100提示纯噪声反例测试仍4/4放行。250接口需要240提示；回归测试逐字保留原100提示并补140新噪声、提供离码参考后为0/4。这个补长回归不是旧4条相同完整输入的概率修复证明。226的父150/候选对照则确实共享同一240提示输入。

## 检查和复现

5项测试全部通过，涵盖旧噪声反例、公共复标量不变性、训练运动拟合不使用检测段、三源/停发等反例、冻结哈希和25套工件。所有25轮共566个父/候选场景侧从快照重新执行，输入哈希、判定、p及诊断数值全部复现。复放与额外种子没有计入25轮。

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
D:/Python/Python311/python.exe -m pytest tests/test_source_third.py -q
D:/Python/Python311/python.exe source_third_validate.py replay
D:/Python/Python311/python.exe source_third_validate.py validate
```

工作目录为python_sim_simplified。最后两条会重写本分支对应验证文件。不要重新执行freeze来覆盖原冻结证据。没有改动历史实现或README。
