# 正常初始化后空间来源判别：026–050轮

已完成25轮真实算法候选比较，保留026、027、039、042、044、049，最终冻结049。这25轮是同一小型观测模型上的算法探索，不能等同于25项独立物理验证。每轮proposal先于source写入，随后用同种子同四通道数据运行父/候选模型，保留完整逐样本结果。第一轮parent_round=25仅表示来源线的原始误差基线，不引用阵列线第025轮。

## 输入域与初始化

仅仿真载波已擦除的复数空间观测：8颗卫星、7×14cm矩形四通道、2/3/4路取同数据前缀。暖阶段12块；前8块均值估计模板，后4块验证平均投影失配<0.06。通过时空间初始化时刻1.2s，攻击1.6s，间隔0.4s。失败保留并单列。此初始化只证明空间模板稳定，未证明正常捕获、跟踪、定位已经完成，因此并未满足完整接收机初始化验收。

无冷启动攻击；攻击混合为90%共同源，近方向为65%，真实分量仍存在。反射采用25%直达加90%共同反射，作为明确的良性反例。运动只是后初始化空间相位扰动；重捕仅模拟任意每星公共相位重置。未覆盖真实牵引时间历程、真实信号完全消失、持续失锁与RF重捕。在线score仅接收warm/current；生成真值只用于离线标签与物理干预。

## 冻结算法

令A为暖均值，B为当前均值。对B/A做秩一SVD拟合通道×卫星可分复数变化，回乘A得天空模型重构；直接对B做秩一SVD得共同发射源重构。分数=(天空残差平方和−共同源残差平方和)/暖观测噪声方差。这是残差比较，B/A上的SVD不是异方差条件下天空模型的精确最大似然解。有限噪声与模板分母误差是限制。

门限使用111001–111020中已初始化、无攻击、非反射样本的95%经验分位，三/四主选择目标为检测率−2×普通误报率的均值；111021–111045用于逐轮选型，二天线不参与该目标。此目标明确不惩罚共同反射误报，故不能用其高分声称真实/恶意来源可辨。111046–111099仅冻结后审计；112001、112013、112037仅冻结后留出。留出只有3个独立种子，各场景共享潜变量，事件数不能视为独立样本数。

## 冻结结果

|数据|天线|检测率（初始化后）|普通误报率|含反射全部良性误报率|初始化失败/全部事件|
|---|---:|---:|---:|---:|---:|
|holdout|2|100.00%|0.00%|14.29%|6/48|
|holdout|3|100.00%|0.00%|14.29%|6/48|
|holdout|4|100.00%|0.00%|14.29%|6/48|
|extended|2|100.00%|4.12%|17.26%|76/864|
|extended|3|100.00%|5.83%|19.21%|104/864|
|extended|4|100.00%|8.31%|21.37%|106/864|

留出每配置攻击检测21/21、普通误报0/18、反射误报3/3；低信号初始化失败6/6（攻击/无攻击配对），总空间初始化通过42/48。扩展审计普通误报升至二4.12%、三5.83%、四8.31%，因此小留出零误报不代表可靠低误报。全部成功初始化攻击未出现错误放行，但失败初始化不属于检测成功。无PVT输出，定位误差/定位可用率/连续性均为未测，不用空间初始化率替代。

## 二天线因果与边界

最终二天线在本模型中没有检测劣势，不能声称天线数量导致失败。冻结后54种子配对审计，短边/长边/对角二天线非反射攻击均100%检测，普通误报分别14/340、14/339、14/338。几何改变没有解释明显检测差距。第045轮给噪声模型增加第二空间模态后二天线残差维数为零，检测率降至0；这是该模型的自由度耗尽，而不是任意二天线算法必败。

算法实现影响很大：相同开发输入第044轮暖相位外推使二天线检测由第042轮87.23%降至22.34%，最终049恢复100%。因此那次二天线失败主要来自所选估计器与门限组合。此实验不实施真实信号抑制，不能归因于抑制损失；同步固定理想，不能检验同步偏差。

离线噪声幅度降至1/4后，全部配置初始化失败归零；基准三/四非反射初始化失败104/756、106/756，其低信号失败分别104/108、106/108。这支持当前低信号边界主要是空间初始化噪声，而不是天线少。消除静态校准误差后三/四失败数仍104、106，不能归因于静态校准误差。该干预重新用对应校准数据定门限，数字代表整套估计与门限流程，不是固定门限的单因素敏感度。

动态相位/增益漂移仍有误报：扩展三天线相位2/54、增益7/54；四天线相位4/54、增益1/54。共同良性反射在最终全部已初始化样本中100%报警，是来源恶意性不可辨的直接反例。结论：三天线最低目标和四天线整机目标均未得到证实，增加天线也未在本模型中稳定降低误报。

## 逐轮实测台账

|轮|保留|父效用|候选效用|具体假设|
|---:|---|---:|---:|---|
|026|是|-0.115835|0.843607|Per-satellite carrier phase after reacquisition is arbitrary; projective direction residual removes this nuisance whereas raw complex error does not.|
|027|是|0.843607|0.953466|Residual projective change still responds to phase drift; center antenna phase change across satellites to remove a satellite-independent channel phase nuisance.|
|028|否|0.953466|0.886976|Phase-change alarm cannot distinguish reflection. Test a competing source-convergence statistic: increase in pairwise satellite spatial coherence, invariant to shared unitary phase drift.|
|029|否|0.953466|0.860530|Normalize each antenna/satellite to unit phasors before inter-satellite coherence so gain drift cannot collapse the apparent array rank.|
|030|否|0.953466|0.890801|A dominant covariance eigenvalue explicitly tests one-emitter concentration instead of mean pairwise geometry change.|
|031|否|0.953466|0.873731|Spectral entropy loss can detect source collapse distributed across all secondary spatial modes, missed by the maximum eigenvalue alone.|
|032|否|0.953466|0.840687|Estimate the full channel noise covariance from repeated warm observations and whiten before comparing directions; this tests whether ignoring covariance explains residual false alarms.|
|033|否|0.953466|0.946799|Use all antenna-pair phase changes instead of reference-antenna ratios so a noisy reference does not contaminate every residual.|
|034|否|0.953466|0.906932|Whiten centered phase innovations using warm estimated covariance, explicitly accounting for reference-channel correlation and unequal phase uncertainty.|
|035|否|0.953466|0.926799|Weight circular drift fitting and residuals by observed inverse phase variance, downweighting weak satellite-channel observations without truth SNR.|
|036|否|0.953466|0.913687|Replace mean nuisance phase by a circular medoid minimizing total wrapped angular distance, so one displaced satellite cannot pull the shared drift estimate.|
|037|否|0.953466|0.953466|Leave-one-satellite-out drift estimation removes self-fit bias and tests spatial change on data excluded from nuisance fitting.|
|038|否|0.953466|0.460976|Double-center log amplitude ratios to cancel channel gains and per-satellite amplitude changes, testing nonseparable spatial fading independent of phase.|
|039|是|0.953466|0.960132|Under only channel and satellite complex gain changes, B/A is rank one; its normalized rank-one reconstruction error tests both phase and amplitude interaction.|
|040|否|0.960132|0.946799|Constrain the separable nuisance fit to phase-only channel/satellite factors so large gain channels do not dominate the rank-one error.|
|041|否|0.960132|0.926932|Require rank-one violation in both independent current time halves; a transient noisy block should not suffice for a sustained source-change alarm.|
|042|是|0.960132|0.966755|Accumulate per-block spatial ratio covariance before eigendecomposition, eliminating coherent temporal carrier cancellation while retaining a shared rank-one nuisance model.|
|043|否|0.966755|0.953554|A one-sided cumulative-sum detector relative to observed warm residual energy can separate sustained spatial change from the estimator's nonzero noise floor.|
|044|是|0.966755|0.973377|Extrapolate a per-channel/per-satellite phase slope estimated during normal initialization, then test rank-one residual against that moving template.|
|045|否|0.973377|0.966755|The selected trend predictor severely hurts two-antenna detection; test a rank-two nuisance subspace for one additional deformation mode and explicitly expose lost spatial degrees of freedom.|
|046|否|0.973377|0.860486|Require at least four satellites to violate the shared phase-drift model; this tests a multi-source consistency requirement instead of a mean residual that can be driven by one satellite.|
|047|否|0.973377|0.880442|Algebraic connectivity of the satellite coherence graph may detect convergence of separate direction clusters that an average coherence statistic hides.|
|048|否|0.973377|0.960132|Estimate the shared channel factor from the ratio SVD, remove it, then project each satellite onto its warm direction; this uses geometric reconstruction error rather than ratio singular values.|
|049|是|0.973377|0.986755|Compare residual likelihoods of a common spatial emitter versus the initialized sky with shared calibration nuisance; source convergence should favor the former explicitly.|
|050|否|0.986755|0.927020|Remove estimated receiver-noise covariance from the noncoherent ratio covariance to test whether noise-floor bias, rather than source structure, caused earlier rank-score false alarms.|

每轮目录包含proposal.json、before.py、source.py、before.json、after.json、conclusion.json。精确源码哈希和选择链见ledger.json，freeze.json记录冻结哈希，summary.json可机器读取。

## 复验

在python_sim_simplified目录执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
D:/Python/Python311/python.exe warm_source_audit.py
D:/Python/Python311/python.exe warm_source_audit.py --replay-rounds
D:/Python/Python311/python.exe -m pytest tests/test_warm_source.py -q
```

6个测试通过，冻结留出逐字段精确复演一致。审计/修补/测试均不额外计轮。

## 项目精确几何补验（不计轮）

原026–050轮及原留出使用70×140mm，偏离项目75.9×160.9mm固定安装尺寸；这是原探索输入的几何偏差。原始证据未改写。以下仅替换生成器四个坐标，沿用冻结049源码评分及原门限，使用112001/112013/112037，不重新校准、不选型、不新增轮数。总报告应采用本节精确几何结果。

|天线|检测率|普通误报率|含反射全部良性误报率|初始化失败/总事件|
|---:|---:|---:|---:|---:|
|2|100.00%|0.00%|14.29%|6/48|
|3|100.00%|0.00%|14.29%|6/48|
|4|100.00%|0.00%|14.29%|6/48|

上述检测与误报以成功空间初始化为条件；全部逐事件得分、初始化状态见exact_geometry_holdout.json。仍只有三个独立种子，不能把重复场景当独立样本。空间初始化不是RF捕获/跟踪/PVT完成证据。

复验：`D:/Python/Python311/python.exe warm_source_exact_geometry.py`。
