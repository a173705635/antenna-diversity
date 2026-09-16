# 更新需求后100轮逐项台账

每行是一个具体机制候选实验，四线独立比较，指标不能相加为整机收益。恢复线为预先实现机制库的顺序选型；纠错和补验不计轮。详细数字见链接。

|轮|研究线|假设/机制|决定|证据|
|---:|---|---|---|---|
|001|array|无攻击初始化相关器相位交叉标定替代名义单位增益，检验固定通道相位误差可否消除|conditional|[配对结果](results/research-warm-hundred/array/iteration001/results.json)|
|002|array|相对复增益估计加入幅度，纠正相位标定未覆盖的通道幅度失配|conditional|[配对结果](results/research-warm-hundred/array/iteration002/results.json)|
|003|array|联合最小二乘分解各PRN波形，替代互相关近似以减少码间泄漏|conditional|[配对结果](results/research-warm-hundred/array/iteration003/results.json)|
|004|array|跨所有通道PRN消旋后秩一SVD估计相对增益，避免只依赖第零通道交叉乘积|conditional|[配对结果](results/research-warm-hundred/array/iteration004/results.json)|
|005|array|攻击时段以样本最大特征向量建立单源零陷，归一化保持每星名义响应|retain|[配对结果](results/research-warm-hundred/array/iteration005/results.json)|
|006|array|以最大特征值与剩余谱均值比检测共同空间分量，仅比值大于3时置零陷|retain|[配对结果](results/research-warm-hundred/array/iteration006/results.json)|
|007|array|源方向改由当前减启动协方差估计，消除稳定天空与噪声背景对特征方向的牵引|reject|[配对结果](results/research-warm-hundred/array/iteration007/results.json)|
|008|array|以启动协方差为度量求广义最大特征向量，再映射回物理源方向，兼顾有色背景|retain|[配对结果](results/research-warm-hundred/array/iteration008/results.json)|
|009|array|用启动背景协方差构造LCMV真星无失真与源零陷双约束，降低简单投影噪声代价|reject|[配对结果](results/research-warm-hundred/array/iteration009/results.json)|
|010|array|初始化拟合观测码波形并从残差估计噪声协方差，LCMV在噪声度量下最小方差|reject|[配对结果](results/research-warm-hundred/array/iteration010/results.json)|
|011|array|当前样本协方差加载MVDR替代硬零陷，以降低近方向真实星噪声放大|reject|[配对结果](results/research-warm-hundred/array/iteration011/results.json)|
|012|array|零陷后每星保留响应低于阵列响应5%则不参与PVT，将不可观测方向代价显式计入|retain|[配对结果](results/research-warm-hundred/array/iteration012/results.json)|
|013|array|先估计各PRN码延迟，联合拟合复系数后逐星消去其他PRN并重新估计码延迟|tradeoff|[配对结果](results/research-warm-hundred/array/iteration013/results.json)|
|014|array|将逐星重捕改为投影后多PRN联合实数Gauss-Newton延迟修正，复振幅为消去的干扰参数|reject|[配对结果](results/research-warm-hundred/array/iteration014/results.json)|
|015|array|对连续被接纳的每星伪距做因果指数平滑，降低采样码噪声而完整统计拖曳风险|conditional|[配对结果](results/research-warm-hundred/array/iteration015/results.json)|
|016|array|以星间中心化创新的MAD检验去除突变卫星，共同钟差被消去而全星协调拖曳作为反例|retain|[配对结果](results/research-warm-hundred/array/iteration016/results.json)|
|017|array|在完整初始化窗口内融合相对复增益估计，冻结后进入攻击阶段；初始化窗口不输出实时可信导航|reject|[配对结果](results/research-warm-hundred/array/iteration017/results.json)|
|018|array|从无攻击初始化窗口相对通道相位拟合线性漂移，因果外推校正攻击期固定频偏|reject|[配对结果](results/research-warm-hundred/array/iteration018/results.json)|
|019|array|用BIC比较常相位与线性漂移模型，只有初始化数据支持时才外推频偏|reject|[配对结果](results/research-warm-hundred/array/iteration019/results.json)|
|020|array|用相对启动背景的广义最大特征值检测攻击新增能量，替代当前谱内部比值|conditional|[配对结果](results/research-warm-hundred/array/iteration020/results.json)|
|021|array|按每星启动相关幅度建立相对功率基准，抑制后功率不足启动10%则不接纳残余码峰|retain|[配对结果](results/research-warm-hundred/array/iteration021/results.json)|
|022|array|使用相关峰质量倒数作为每星相对方差进行加权PVT，测试三天线和二长基线弱信号误差|retain|[配对结果](results/research-warm-hundred/array/iteration022/results.json)|
|023|array|以相关质量作为相对信息量累积并遗忘，替代固定比例码平滑，失去接纳时重置观测弧|retain|[配对结果](results/research-warm-hundred/array/iteration023/results.json)|
|024|array|加权PVT加入Huber残差IRLS，减少单星码偏差影响，不把小残差当真实性证明|reject|[配对结果](results/research-warm-hundred/array/iteration024/results.json)|
|025|array|初始化必须产生连续有限PVT且水平离散不超过20m，否则后续输出保持不可用并计初始化失败|retain|[配对结果](results/research-warm-hundred/array/iteration025/results.json)|
|026|source|Per-satellite carrier phase after reacquisition is arbitrary; projective direction residual removes this nuisance whereas raw complex error does not.|retain|[配对结果](results/research-warm-hundred/source/026/after.json)|
|027|source|Residual projective change still responds to phase drift; center antenna phase change across satellites to remove a satellite-independent channel phase nuisance.|retain|[配对结果](results/research-warm-hundred/source/027/after.json)|
|028|source|Phase-change alarm cannot distinguish reflection. Test a competing source-convergence statistic: increase in pairwise satellite spatial coherence, invariant to shared unitary phase drift.|reject|[配对结果](results/research-warm-hundred/source/028/after.json)|
|029|source|Normalize each antenna/satellite to unit phasors before inter-satellite coherence so gain drift cannot collapse the apparent array rank.|reject|[配对结果](results/research-warm-hundred/source/029/after.json)|
|030|source|A dominant covariance eigenvalue explicitly tests one-emitter concentration instead of mean pairwise geometry change.|reject|[配对结果](results/research-warm-hundred/source/030/after.json)|
|031|source|Spectral entropy loss can detect source collapse distributed across all secondary spatial modes, missed by the maximum eigenvalue alone.|reject|[配对结果](results/research-warm-hundred/source/031/after.json)|
|032|source|Estimate the full channel noise covariance from repeated warm observations and whiten before comparing directions; this tests whether ignoring covariance explains residual false alarms.|reject|[配对结果](results/research-warm-hundred/source/032/after.json)|
|033|source|Use all antenna-pair phase changes instead of reference-antenna ratios so a noisy reference does not contaminate every residual.|reject|[配对结果](results/research-warm-hundred/source/033/after.json)|
|034|source|Whiten centered phase innovations using warm estimated covariance, explicitly accounting for reference-channel correlation and unequal phase uncertainty.|reject|[配对结果](results/research-warm-hundred/source/034/after.json)|
|035|source|Weight circular drift fitting and residuals by observed inverse phase variance, downweighting weak satellite-channel observations without truth SNR.|reject|[配对结果](results/research-warm-hundred/source/035/after.json)|
|036|source|Replace mean nuisance phase by a circular medoid minimizing total wrapped angular distance, so one displaced satellite cannot pull the shared drift estimate.|reject|[配对结果](results/research-warm-hundred/source/036/after.json)|
|037|source|Leave-one-satellite-out drift estimation removes self-fit bias and tests spatial change on data excluded from nuisance fitting.|reject|[配对结果](results/research-warm-hundred/source/037/after.json)|
|038|source|Double-center log amplitude ratios to cancel channel gains and per-satellite amplitude changes, testing nonseparable spatial fading independent of phase.|reject|[配对结果](results/research-warm-hundred/source/038/after.json)|
|039|source|Under only channel and satellite complex gain changes, B/A is rank one; its normalized rank-one reconstruction error tests both phase and amplitude interaction.|retain|[配对结果](results/research-warm-hundred/source/039/after.json)|
|040|source|Constrain the separable nuisance fit to phase-only channel/satellite factors so large gain channels do not dominate the rank-one error.|reject|[配对结果](results/research-warm-hundred/source/040/after.json)|
|041|source|Require rank-one violation in both independent current time halves; a transient noisy block should not suffice for a sustained source-change alarm.|reject|[配对结果](results/research-warm-hundred/source/041/after.json)|
|042|source|Accumulate per-block spatial ratio covariance before eigendecomposition, eliminating coherent temporal carrier cancellation while retaining a shared rank-one nuisance model.|retain|[配对结果](results/research-warm-hundred/source/042/after.json)|
|043|source|A one-sided cumulative-sum detector relative to observed warm residual energy can separate sustained spatial change from the estimator's nonzero noise floor.|reject|[配对结果](results/research-warm-hundred/source/043/after.json)|
|044|source|Extrapolate a per-channel/per-satellite phase slope estimated during normal initialization, then test rank-one residual against that moving template.|retain|[配对结果](results/research-warm-hundred/source/044/after.json)|
|045|source|The selected trend predictor severely hurts two-antenna detection; test a rank-two nuisance subspace for one additional deformation mode and explicitly expose lost spatial degrees of freedom.|reject|[配对结果](results/research-warm-hundred/source/045/after.json)|
|046|source|Require at least four satellites to violate the shared phase-drift model; this tests a multi-source consistency requirement instead of a mean residual that can be driven by one satellite.|reject|[配对结果](results/research-warm-hundred/source/046/after.json)|
|047|source|Algebraic connectivity of the satellite coherence graph may detect convergence of separate direction clusters that an average coherence statistic hides.|reject|[配对结果](results/research-warm-hundred/source/047/after.json)|
|048|source|Estimate the shared channel factor from the ratio SVD, remove it, then project each satellite onto its warm direction; this uses geometric reconstruction error rather than ratio singular values.|reject|[配对结果](results/research-warm-hundred/source/048/after.json)|
|049|source|Compare residual likelihoods of a common spatial emitter versus the initialized sky with shared calibration nuisance; source convergence should favor the former explicitly.|retain|[配对结果](results/research-warm-hundred/source/049/after.json)|
|050|source|Remove estimated receiver-noise covariance from the noncoherent ratio covariance to test whether noise-floor bias, rather than source structure, caused earlier rank-score false alarms.|reject|[配对结果](results/research-warm-hundred/source/050/after.json)|
|051|recovery|init_average|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|052|recovery|peak_baseline|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|053|recovery|subgrid|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|054|recovery|rank1_init|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|055|recovery|spatial_whiten|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|056|recovery|residual_direction|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|057|recovery|projected_fit|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|058|recovery|phase_align|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|059|recovery|lag_whiten|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|060|recovery|init_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|061|recovery|presence_gate|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|062|recovery|precision_gate|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|063|recovery|shape_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|064|recovery|temporal_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|065|recovery|motion_fit|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|066|recovery|calibration_uncertainty|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|067|recovery|block_direction|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|068|recovery|robust_average|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|069|recovery|ambiguity_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|070|recovery|continuity_gate|retain|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|071|recovery|crossfit_direction|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|072|recovery|holdout_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|073|recovery|projection_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|074|recovery|joint_fit|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|075|recovery|persistence_gate|reject|[配对结果](results/research-warm-hundred/recovery/ledger.json)|
|076|navigation|Use measured per-satellite C/N0 weighting to reduce weak-channel contribution.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|077|navigation|Equal per-epoch C/N0 produced no change; robust satellite residual weights may reduce isolated code faults.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|078|navigation|Residual robustness leaves coherent attacks; first remove persistent single-channel faults with leave-one-out consistency.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|079|navigation|Code-only faults remain large; carrier propagation with code blending should reduce code drag and noise.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|080|navigation|Warm mean position lags motion; transport initialization code estimates to current epoch using observed carrier differences.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|081|navigation|Carrier smoothing still follows code drag; monitor code-minus-carrier change in position subspace, allowing receiver clock change.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|082|navigation|CMC cannot see common code/carrier steps; a large measured carrier position jump should latch output protection.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|083|navigation|Common slow pull is invisible to CMC; compare TDCP velocity against velocity estimated during benign acquisition.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|084|navigation|Zero >=30m errors now costs over half of output epochs; make warm-velocity gate a three-epoch sequential detector to avoid isolated noisy triggers.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|085|navigation|Continuity remains poor; isolated satellite cycle slips should be excluded in TDCP residual space before velocity screening.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|086|navigation|Cycle-slip screening improves availability; robust CMC should stop one carrier ambiguity jump poisoning the shared source monitor.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|087|navigation|Robust CMC still reacts to isolated weak-code noise; require persistent CMC disagreement before latching.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|088|navigation|After a tracking gap, prior authenticity is insufficient: require five consecutive code/phase-consistent epochs before output.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|089|navigation|Fixed quarantine lost outputs without safety benefit. Test conditional ambiguity rebasing only when reacquired code agrees with benign observed trajectory; agreement is not authentication.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|090|navigation|Weak-signal availability is 19/180; normalize CMC boundary using declared code-noise covariance instead of an absolute noise-independent threshold.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|091|navigation|Adaptive CMC releases weak outputs, but some exceed20m; use C/N0-dependent code injection into carrier smoother.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|092|navigation|Weak carrier rejection repeatedly resets smoother to noisy code; use observed warm velocity propagation plus code correction when TDCP is unavailable.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|093|navigation|Prediction improves <=20m count; TDCP screening should scale with measured phase uncertainty rather than reject most weak observations.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|094|navigation|Fixed benign velocity falsely stops a genuine turn; permit gradual velocity adaptation only below the existing innovation boundary.|retain|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|095|navigation|Adaptive motion restores turn outputs but slow drag remains; test accumulated TDCP displacement against initial observed constant-velocity trajectory.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|096|navigation|Trajectory envelope suppresses legitimate turns; instead test initialization evidence after detrending motion and require horizontal mean uncertainty <=5m.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|097|navigation|Initialization variant changes no tested outputs; limit horizontal code innovation influence to prevent large code faults injecting into carrier trajectory.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|098|navigation|Bounded code influence gave no threshold-count benefit; test divergence-free carrier increment using half the measured code-minus-carrier increment to compensate opposite-sign ionospheric drift.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|099|navigation|Divergence correction admitted severe coherent errors and was rejected; add a geometry/noise horizontal protection bound to withhold unreliable absolute fixes during weak tracking.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
|100|navigation|Absolute protection bound only removes usable weak epochs; test alarm release after ten consistent epochs agreeing with observed warm trajectory and CMC, explicitly without source-authentication claim.|reject|[配对结果](results/research-warm-hundred/navigation/ledger.jsonl)|
