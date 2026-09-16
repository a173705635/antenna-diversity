# 第 101–200 轮逐轮账本

每个编号是一项算法或模型改动；留出种子、测试、重放和实现纠错不计轮次。151–175 的结论格式为 Markdown，其余为 JSON。

| 轮次 | 假设 / 实际机制 | 对照与结论 |
|---|---|---|
| 101 | 移除有限窗FFT通道时延回归，检验其是否制造正常场景的虚假群时延；真实skew作为明确代价。 | [证据与结论](results/research-next-hundred/iteration101/conclusion.json) |
| 102 | 以相关域变投影直接估计共同通道群时延，暂固定上轮每PRN时延，检验消除FFT窗泄漏后仍有何偏差。 | [证据与结论](results/research-next-hundred/iteration102/conclusion.json) |
| 103 | 联合优化每PRN源时延和参考通道规范下的相对群时延，避免固定码时延吸收硬件skew。 | [证据与结论](results/research-next-hundred/iteration103/conclusion.json) |
| 104 | 空间一致性之外加入已知相关噪声方差的信号对纯噪声似然比和复杂度惩罚，防止噪声误支持。 | [证据与结论](results/research-next-hundred/iteration104/conclusion.json) |
| 105 | 加入0.03chip零均值通道群时延高斯先验，以偏差换取重合源弱可辨识场景的方差；该先验是额外假设。 | [证据与结论](results/research-next-hundred/iteration105/conclusion.json) |
| 106 | 以联合变投影的零群时延嵌套模型BIC选择是否释放硬件时延，不再依赖高斯硬件先验。 | [证据与结论](results/research-next-hundred/iteration106/conclusion.json) |
| 107 | 将共享信号带宽加入联合时延变投影的nuisance参数；噪声白化冻结，避免残差通过改变权重获利。 | [证据与结论](results/research-next-hundred/iteration107/conclusion.json) |
| 108 | 引入独立512快拍无信号噪声校准，其实部样本协方差白化联合profile；声明需要额外噪声校准输入。 | [证据与结论](results/research-next-hundred/iteration108/conclusion.json) |
| 109 | 从独立噪声校准快拍估计每实维噪声方差，移除已知0.02噪声oracle并用于信号/群时延证据。 | [证据与结论](results/research-next-hundred/iteration109/conclusion.json) |
| 110 | 对独立噪声数据进行结构协方差最大似然：估计相关带宽、白噪声比例和profile方差，减少自由样本协方差的有限样本方差。 | [证据与结论](results/research-next-hundred/iteration110/conclusion.json) |
| 111 | 校准协方差贯穿每PRN初始化、源数选择及最终分量拟合，不再在局部函数隐式切回信号带宽协方差。 | [证据与结论](results/research-next-hundred/iteration111/conclusion.json) |
| 112 | 源计数使用独立标定噪声的绝对高斯似然及实际参数维数，避免各模型各自估残差尺度奖励过拟合。 | [证据与结论](results/research-next-hundred/iteration112/conclusion.json) |
| 113 | 以消去同PRN另一模板后的分量Fisher信息加权天空增益，代替各PRN单位范数等权校准。 | [证据与结论](results/research-next-hundred/iteration113/conclusion.json) |
| 114 | 每PRN天空与公共源支持均用其余PRN估计空间方向，再评估留出PRN，减少空间校准自我支持。 | [证据与结论](results/research-next-hundred/iteration114/conclusion.json) |
| 115 | 固定已估天空增益和公共源空间向量，将每通道独立复幅压缩为每源每PRN单复幅，相关域相干refine码时延；空间支持继续检查未约束测量分量。 | [证据与结论](results/research-next-hundred/iteration115/conclusion.json) |
| 116 | 相干refine联合估计跨PRN共享的对角通道相位和每PRN码时延，解除固定初始相位造成的偏差。 | [证据与结论](results/research-next-hundred/iteration116/conclusion.json) |
| 117 | 公共源复空间向量以参考通道为1的规范参与相干联合profile，解除顺序PCA方向估计的泄漏偏差。 | [证据与结论](results/research-next-hundred/iteration117/conclusion.json) |
| 118 | 相干模型同时profile信号带宽、天空增益、公共源向量与PRN时延，独立噪声白化继续冻结。 | [证据与结论](results/research-next-hundred/iteration118/conclusion.json) |
| 119 | 在相干联合模型内部估计群时延并与其零群时延嵌套模型比较BIC，消除前级自由分量时延与后级空间约束不一致。 | [证据与结论](results/research-next-hundred/iteration119/conclusion.json) |
| 120 | 新增第三径模型失配同输入挑战，用独立噪声方差下的全局相干残差卡方拟合优度拒绝错误双源模型；p值为渐近诊断而非认证。 | [证据与结论](results/research-next-hundred/iteration120/conclusion.json) |
| 121 | 对每PRN码时延导数消去所有nuisance列，获得局部Fisher标准差；采用3sigma小于10m的精度预算筛选并加入低幅正常信号同输入挑战。 | [证据与结论](results/research-next-hundred/iteration121/conclusion.json) |
| 122 | 扩展对角校准为共享幅度+相位，联合profile幅度nuisance；新增非单位通道幅度同输入挑战，并在码Fisher中消去幅度自由度。 | [证据与结论](results/research-next-hundred/iteration122/conclusion.json) |
| 123 | 每PRN额外比较相干天空+公共源与仅公共源的条件似然证据，区分任意信号存在与所选天空源支持。 | [证据与结论](results/research-next-hundred/iteration123/conclusion.json) |
| 124 | 对Fisher信息最弱PRN固定码时延±10m，非线性重新profile其余参数；若任一边界在delta卡方9内则拒绝该PRN，其他PRN仍仅局部Fisher。 | [证据与结论](results/research-next-hundred/iteration124/conclusion.json) |
| 125 | 独立噪声快拍分半：256拟合结构协方差，256检验白化协方差；新增随lag改变方差的噪声失配挑战，不能默认白化模型正确。 | [证据与结论](results/research-next-hundred/iteration125/conclusion.json) |
| 126 | 逐PRN主奇异向量相位对齐后相干均值，检验弱空间分量噪声降低 | [证据与结论](results/research-next-hundred/iteration126/conclusion.json) |
| 127 | 仅估计BPSK符号而保留幅相信息，分10ms块避免完全秩压缩 | [证据与结论](results/research-next-hundred/iteration127/conclusion.json) |
| 128 | 平方提示估计残余频率后再估计未知符号，避免时变相位抵消 | [证据与结论](results/research-next-hundred/iteration128/conclusion.json) |
| 129 | 显式单位模通道增益模型，移除幅度自由度对RF校正的影响 | [证据与结论](results/research-next-hundred/iteration129/conclusion.json) |
| 130 | 名义零相位收缩先验，检验低噪声收益与未知硬件偏差代价 | [证据与结论](results/research-next-hundred/iteration130/conclusion.json) |
| 131 | 单位模硬件增益与公共源联合非线性可变投影，保留每星两个未知复系数 | [证据与结论](results/research-next-hundred/iteration131/conclusion.json) |
| 132 | 按符号对齐后的观测残差估计逐星噪声，白化联合标定 | [证据与结论](results/research-next-hundred/iteration132/conclusion.json) |
| 133 | 加入名义相位独立初始化，通过同一观测目标选择局部解 | [证据与结论](results/research-next-hundred/iteration133/conclusion.json) |
| 134 | 联合拟合采用softL1残差损失，抑制PRN串扰异常 | [证据与结论](results/research-next-hundred/iteration134/conclusion.json) |
| 135 | 用联合目标雅可比最小奇异值代替旧投影奇异值作局部可辨识性检查 | [证据与结论](results/research-next-hundred/iteration135/conclusion.json) |
| 136 | 明确检验天空分量幅度相对观测噪声，拒绝仅有公共源的可拟合模型 | [证据与结论](results/research-next-hundred/iteration136/conclusion.json) |
| 137 | 分开前后50ms拟合和支持检验，减少同一噪声被拟合后自证 | [证据与结论](results/research-next-hundred/iteration137/conclusion.json) |
| 138 | 独立半段相位一致性，拒绝随噪声改变的局部标定 | [证据与结论](results/research-next-hundred/iteration138/conclusion.json) |
| 139 | 加入仅天空信号模型与两分量BIC竞争，避免正常工况过拟合公共源 | [证据与结论](results/research-next-hundred/iteration139/conclusion.json) |
| 140 | 显式公共源单分量零假设竞争，不能把拟合天空标签当作真实信号存在 | [证据与结论](results/research-next-hundred/iteration140/conclusion.json) |
| 141 | 将140观测分半与模型竞争标定接回完整RF接收链，验证相位改进是否带来实际定位收益 | [证据与结论](results/research-next-hundred/iteration141/conclusion.json) |
| 142 | 公共源估计累计过去提示而非每20ms独立重置，检验弱信号持续子空间稳定性 | [证据与结论](results/research-next-hundred/iteration142/conclusion.json) |
| 143 | 100ms缓冲得到公共源后回放启动段抑制，再对抑制流捕获，避免先锁定攻击峰 | [证据与结论](results/research-next-hundred/iteration143/conclusion.json) |
| 144 | 持续历史先做观测BPSK符号对齐与10ms相干压缩，再估计公共源，抑制1ms噪声 | [证据与结论](results/research-next-hundred/iteration144/conclusion.json) |
| 145 | 移除天空分量后按逐PRN残差能量限权，减少单颗强攻击或串扰控制公共源方向 | [证据与结论](results/research-next-hundred/iteration145/conclusion.json) |
| 146 | 利用已抑制的过去100msRF重新捕获跟踪并更新模板，替代永久外推启动模板 | [证据与结论](results/research-next-hundred/iteration146/conclusion.json) |
| 147 | 模板重启后按观测主空间向量及固定参考通道对齐任意载波相位，避免历史相干和被新模板相位抵消 | [证据与结论](results/research-next-hundred/iteration147/conclusion.json) |
| 148 | 仅天空模型胜出时不设置公共源零陷，避免相干积分把正常天空结构误当强攻击源 | [证据与结论](results/research-next-hundred/iteration148/conclusion.json) |
| 149 | 零陷后使用残余阵列全通道捕获，而非仅目标波束标量捕获，保留弱信号空间能量 | [证据与结论](results/research-next-hundred/iteration149/conclusion.json) |
| 150 | 完整复增益联合标定及逆增益校正，在幅度失配冷攻击检验相位限定模型的遗漏 | [证据与结论](results/research-next-hundred/iteration150/conclusion.json) |
| 151 | Unknown initial origin is unobservable; propagate an unbounded position set. | [证据与结论](results/research-next-hundred/iteration151/conclusion.md) |
| 152 | Estimate bias from a separate stationary bench record, not runtime truth. | [证据与结论](results/research-next-hundred/iteration152/conclusion.md) |
| 153 | Try empirical calibration residual maxima as an error envelope. | [证据与结论](results/research-next-hundred/iteration153/conclusion.md) |
| 154 | Separate empirical or unbounded Gaussian support from hard bounded support. | [证据与结论](results/research-next-hundred/iteration154/conclusion.md) |
| 155 | Identify a bias interval by intersecting separately measured bench intervals with declared fixture error. | [证据与结论](results/research-next-hundred/iteration155/conclusion.md) |
| 156 | Allow time-correlated bias drift rather than treating it as independent white noise. | [证据与结论](results/research-next-hundred/iteration156/conclusion.md) |
| 157 | Use a held-out bench sequence as a falsification check for calibration validity. | [证据与结论](results/research-next-hundred/iteration157/conclusion.md) |
| 158 | Compensate measured temperature with an externally calibrated coefficient interval. | [证据与结论](results/research-next-hundred/iteration158/conclusion.md) |
| 159 | Use independent bounded velocity observations to limit integrated drift. | [证据与结论](results/research-next-hundred/iteration159/conclusion.md) |
| 160 | Permit zero-velocity updates only from an independent stationary fixture, not quiet accelerometers. | [证据与结论](results/research-next-hundred/iteration160/conclusion.md) |
| 161 | Transport an old initial position measurement to the integration epoch. | [证据与结论](results/research-next-hundred/iteration161/conclusion.md) |
| 162 | Propagate initial timestamp uncertainty and velocity support into position support. | [证据与结论](results/research-next-hundred/iteration162/conclusion.md) |
| 163 | Stop using a calibration after its declared validity interval expires. | [证据与结论](results/research-next-hundred/iteration163/conclusion.md) |
| 164 | Transport delayed independent position measurements with velocity and acceleration support. | [证据与结论](results/research-next-hundred/iteration164/conclusion.md) |
| 165 | Intersect constant-bias support with independent velocity increments using time-correlated noise support. | [证据与结论](results/research-next-hundred/iteration165/conclusion.md) |
| 166 | Account for three-axis gravity leakage using the sine Lipschitz bound. | [证据与结论](results/research-next-hundred/iteration166/conclusion.md) |
| 167 | Use an independent attitude observation and its interval to remove nominal gravity leakage. | [证据与结论](results/research-next-hundred/iteration167/conclusion.md) |
| 168 | Propagate time-correlated gyro error into attitude and gravity support. | [证据与结论](results/research-next-hundred/iteration168/conclusion.md) |
| 169 | Transform the body-center position to the time-varying antenna phase center. | [证据与结论](results/research-next-hundred/iteration169/conclusion.md) |
| 170 | Include uncertain antenna/body lever geometry in the output-centered error set. | [证据与结论](results/research-next-hundred/iteration170/conclusion.md) |
| 171 | Bound multiplicative scale error from measured acceleration rather than true acceleration. | [证据与结论](results/research-next-hundred/iteration171/conclusion.md) |
| 172 | Invalidate integrated inertial support after a flagged clipped sample with unknown missing acceleration. | [证据与结论](results/research-next-hundred/iteration172/conclusion.md) |
| 173 | Treat contradictory anchor intersections as an integrity inconsistency, not an absolute-valued radius. | [证据与结论](results/research-next-hundred/iteration173/conclusion.md) |
| 174 | Allow a fresh absolute observation to provide a single-epoch position when inertial support is unavailable. | [证据与结论](results/research-next-hundred/iteration174/conclusion.md) |
| 175 | Reject position certificates whose provenance depends on the attacked GNSS stream. | [证据与结论](results/research-next-hundred/iteration175/conclusion.md) |
| 176 | Use the separately processed capture route when joint calibration is unobservable instead of suppressing all navigation; retain trusted-startup limitation. | [证据与结论](results/research-next-hundred/iteration176/conclusion.json) |
| 177 | Select the capture navigation at individual missing joint epochs rather than locking one route for the entire record. | [证据与结论](results/research-next-hundred/iteration177/conclusion.json) |
| 178 | Reject mutually inconsistent finite route positions; a supported calibration cannot override contradictory spatial processing. | [证据与结论](results/research-next-hundred/iteration178/conclusion.json) |
| 179 | Average mutually compatible route positions; avoid privileging the joint route when both carry valid but noisy estimates. | [证据与结论](results/research-next-hundred/iteration179/conclusion.json) |
| 180 | Replace equal averaging with nominal independent-variance fusion derived from route CN0 and geometry; test its availability and error assumptions. | [证据与结论](results/research-next-hundred/iteration180/conclusion.json) |
| 181 | Treat cross-route covariance as unknown: scalar covariance-intersection chooses the lower nominal variance rather than claiming independent averaging gain. | [证据与结论](results/research-next-hundred/iteration181/conclusion.json) |
| 182 | Fuse corresponding finite code ranges before navigation, preserving joint carrier arcs and route-conflict rejection. | [证据与结论](results/research-next-hundred/iteration182/conclusion.json) |
| 183 | Causally align per-star carrier offsets at first simultaneous observation, then bridge joint carrier gaps with capture carrier and capture quality. | [证据与结论](results/research-next-hundred/iteration183/conclusion.json) |
| 184 | Terminate cross-route carrier offset at capture loss, preventing stale ambiguity transport into a later joint outage. | [证据与结论](results/research-next-hundred/iteration184/conclusion.json) |
| 185 | When joint calibration has observed support, navigate the calibrated range/carrier directly rather than require agreement with an attacked raw stream; retain cross-processed route conflicts. | [证据与结论](results/research-next-hundred/iteration185/conclusion.json) |
| 186 | Cap code weighting by equivalent CN0 inferred from observed spatial prompt SNR, accounting for weak desired energy after source projection. | [证据与结论](results/research-next-hundred/iteration186/conclusion.json) |
| 187 | Estimate displacement from reference-star differences; assess the effect of neglecting common-reference covariance against clock-nuisance least squares. | [证据与结论](results/research-next-hundred/iteration187/conclusion.json) |
| 188 | Whiten reference-star differences with I+11T covariance instead of treating them independent. | [证据与结论](results/research-next-hundred/iteration188/conclusion.json) |
| 189 | Use heteroscedastic carrier endpoint variance in full reference-difference covariance rather than equal phase variance. | [证据与结论](results/research-next-hundred/iteration189/conclusion.json) |
| 190 | Use a common carrier arc endpoint and phase-noise weights, preserving temporal differencing covariance implicitly instead of accumulating independent step estimates. | [证据与结论](results/research-next-hundred/iteration190/conclusion.json) |
| 191 | Estimate per-PRN code-minus-carrier slope causally and apply opposite half-changes to code/carrier under an explicit dispersive-delay hypothesis. | [证据与结论](results/research-next-hundred/iteration191/conclusion.json) |
| 192 | Use fresh raw-code position corrections rather than Hatch-smoothed positions, avoiding reuse of carrier noise in both propagation and absolute correction. | [证据与结论](results/research-next-hundred/iteration192/conclusion.json) |
| 193 | Replace fixed0.5s complementary correction with a vector Kalman update using nominal code and carrier geometry covariance. | [证据与结论](results/research-next-hundred/iteration193/conclusion.json) |
| 194 | Use an arc-origin information filter with endpoint carrier-noise variance in each code-minus-displacement observation, avoiding repeated independent increment noise accumulation. | [证据与结论](results/research-next-hundred/iteration194/conclusion.json) |
| 195 | Inflate fresh-code innovation covariance using a Huber likelihood instead of applying Gaussian vector updates to arbitrarily large innovations. | [证据与结论](results/research-next-hundred/iteration195/conclusion.json) |
| 196 | A nominal 2 m persistent code bias floor prevents repeated fresh-code updates from collapsing arc-origin covariance; this is assumed uncertainty, not an integrity certificate. | [证据与结论](results/research-next-hundred/iteration196/conclusion.json) |
| 197 | Use measured spatial SNR cap for carrier endpoint regression and covariance as well as code, so nulling loss enters phase weighting. | [证据与结论](results/research-next-hundred/iteration197/conclusion.json) |
| 198 | When both routes agree, retain calibrated arc estimate instead of averaging with capture PVT; capture remains a missing-support route. | [证据与结论](results/research-next-hundred/iteration198/conclusion.json) |
| 199 | Apply calibrated arc estimator to causal phase-aligned merged code/carrier observations; retain original separate-route conflict evidence. | [证据与结论](results/research-next-hundred/iteration199/conclusion.json) |
| 200 | A detected cross-route position contradiction remains unresolved when routes later agree; latch ambiguity for this recording without an independent re-anchor. | [证据与结论](results/research-next-hundred/iteration200/conclusion.json) |
