# 第251–275轮信号恢复研究

**弱星恢复目标尚未达到：冻结验证weak/overlap/two/three场景全部被拒绝。** 25个真实机制变更已逐轮闭环，保留失败；最终按开发数据选择第271轮。没有新增RF/PVT或台架结论。

相关器域模型使用手机四角75.9×160.9mm，2/3天线为前2/3角，6PRN、41个lag、两个固定状态复相关块、单个公共欺骗源。真实分量波形和真码只进入生成/评价。接收机另需标称天空、64个独立共同注入通道标定样本、160个同态无信号噪声bank；这些不是手机自然可得的已认证输入。

论文只查阅了[ION公开摘要](https://m-private.ion.org/publications/abstract.cfm?articleID=19852)，没有取得全文。摘要介绍空间抑制结合SIC；本支检验这一思路的相关器域误差，不声称复现其完整接收机。

冻结文件在留出评估前写入；种子95101–95103共33案例。冻结后未修改模型。真实源27案例共162PRN，缺真/纯噪6案例共36PRN。

| 指标 | 基线 | 冻结方案 |
|---|---:|---:|
| 全部真实源案例平均波形NMSE | 22.457086 | 0.239233 |
| 真实PRN接受数/162 | 162 | 14 |
| 缺真或纯噪误接受数/36 | 36 | 0 |
| 真实源案例码RMSE均值/m（全部估计，非导航指标） | 112.527 | 38.910 |

冻结方案所接受真实PRN码RMSE 8.853m、最大绝对码误差 18.717m。接受子集是经过筛选的，不代表全部场景恢复。波形NMSE对全部存在真实信号的案例计算，并非仅对接受案例美化。

| 场景 | 基线波形NMSE均值 | 冻结波形NMSE均值 | 冻结接受PRN |
|---|---:|---:|---:|
| weak | 27.865842 | 0.454006 | 0 |
| overlap | 27.612894 | 0.535340 | 0 |
| two | 82.430070 | 0.672780 | 0 |
| three | 49.604793 | 0.347872 | 0 |
| colored | 2.923765 | 0.004419 | 9 |
| bandwidth | 2.948139 | 0.007198 | 0 |
| multipath | 2.938830 | 0.116350 | 5 |
| drift | 2.841079 | 0.007642 | 0 |
| cal_outlier | 2.948366 | 0.007492 | 0 |
| absent | 无真实源 | 无真实源 | 0 |
| noise | 无真实源 | 无真实源 | 0 |

## 逐轮机制与真实对照

| 轮次 | 机制假设 | 开发NMSE前→后 | 接受真实PRN前→后 |
|---|---|---:|---:|
| 251 | Replace sky beam with distortionless common-source spatial null to suppress spoof waveform | 30.876387→0.421736 | 54→54 |
| 252 | Estimate common direction after per-PRN authentic spatial null; test sky leakage against projection distortion | 0.421736→9.585332 | 54→54 |
| 253 | Undo distinct sky-null projection distortion by joint common-direction variable projection | 9.585332→0.477261 | 54→54 |
| 254 | Introduce independent measured complex injection calibration; test amplitude and phase correction | 0.477261→0.236905 | 54→54 |
| 255 | Propagate inverse gain to measured noise and use minimum-variance distortionless null weights | 0.236905→0.229683 | 54→54 |
| 256 | Replace noisy output bank by fitted autocorrelation waveform reconstruction | 0.229683→0.158452 | 54→54 |
| 257 | Continuous variable-projection delay replaces discrete maximum bank bin | 0.158452→0.118159 | 54→54 |
| 258 | Independent measured lag covariance whitens continuous template fit rather than treating bank lags independent | 0.118159→0.101640 | 54→54 |
| 259 | Constrain independent lag-noise covariance to stationary Toeplitz family to reduce estimation variance | 0.101640→0.106738 | 54→54 |
| 260 | Joint spatial-delay two-source profile estimates and reconstructs authentic coefficient conditional on counterfeit template | 0.106738→0.060079 | 54→54 |
| 261 | Propagate measured channel cross-covariance into joint two-source likelihood | 0.060079→0.058925 | 54→54 |
| 262 | Robust independent injection location estimator limits calibration impulse contamination | 0.058925→0.049667 | 54→54 |
| 263 | Estimate shared interblock carrier phase before coherent combination; test drift and noise selection bias | 0.049667→0.044462 | 54→54 |
| 264 | Conditional authentic-source likelihood evidence against counterfeit-only null using independent noise scale | 0.044462→0.044462 | 54→54 |
| 265 | Nested source-presence decision removes unsupported reconstructed waveforms instead of reporting noise as recovered signal | 0.044462→0.044462 | 54→54 |
| 266 | Project delay derivative away from complex amplitude and counterfeit delay nuisance; require local 3sigma below20m | 0.044462→0.044462 | 54→27 |
| 267 | Add whitened two-source residual goodness-of-fit rejection for multipath and model mismatch | 0.044462→0.044462 | 27→27 |
| 268 | Split training and held block; freeze direction and delay before phase-invariant held-block authentic support test | 0.044462→0.237099 | 27→18 |
| 269 | After frozen held-block gate combine both blocks for point reconstruction; retain independent admission mask | 0.237099→0.044462 | 18→18 |
| 270 | Conditional information noise-power subtraction shrinks low-evidence authentic coefficient for reconstruction MSE | 0.044462→0.042535 | 18→18 |
| 271 | Estimate shared front-end signal bandwidth from training common-source profiles while freezing independent noise whitening | 0.042535→0.042286 | 18→18 |
| 272 | Add freely complex same-direction delayed path nuisance at declared0.18chip spacing to separate direct waveform from reflection | 0.042286→0.575634 | 18→0 |
| 273 | Use independent-noise complexity penalty to select two-source versus delayed-path nuisance rather than always freeing near-collinear path | 0.575634→0.042286 | 0→18 |
| 274 | Nonlinear plus-minus20m beam-domain profile boundary audit supplements local delay Fisher gate | 0.042286→0.042286 | 18→1 |
| 275 | Explicit counterfeit waveform reconstruction subtraction followed by authentic spatial beam; compare true clean-bank error with template synthesis | 0.042286→0.107375 | 1→1 |

## 失败与限制

252把各PRN真实方向投影后的残差直接PCA造成公共方向畸变。259平稳Toeplitz噪声约束没有改善开发恢复。266局部码Fisher门限损失半数开发可用PRN；267当前开发GOF判决不变。268只用训练块重构有明显方差代价。272固定0.18chip间隔的额外路径是额外模型先验，释放它导致真/反射分量难分、全部拒绝；未作为最终方案。274两侧边界审计只有有限局部覆盖，接受18→1；不采用为默认。275明确做欺骗波形重构抵消再真实方向合成，开发误差0.042286→0.107375，残余噪声仍然显著。

271开发中有一次字符串替换误触reshape的实现错误，在任何结果产生前修正，未增加轮数，记录在其development_correction.txt。25轮均有真实算法/模型差异，不把种子、场景、参数扫、优化器内部步骤、测试或纠错计为轮数。

模型未估未知天空/绝对位置/钟差，未做真实C/A RF采样、导航比特、动态载波/Doppler或连续导航；独立标定有效性未由硬件证明。第二块门控的方向/码来自第一块，但共享同一标定输入；门控之后的两块联合点估计不获得独立覆盖率保证。非线性局部极值、两天线几何退化、模板失配仍可能恶化时延。最终残差门限自由度为两源近似，独立噪声样本有限且选择后p值未校准，不是完整性保护级。

验证：25个独特源SHA、25轮同输入SHA配对；251/260/264/271/275共5快照×11开发场景完整指标重放，容差1e-9；单元测试命令 `D:/Python/Python311/python.exe -m unittest tests.test_recovery_third`。脚本与逐轮输入哈希均保留。

同一组14个冻结接受PRN上，简单基线码RMSE 110.085m→8.853m。基线是同观测上的未校准标称天空波束，不是delay125或上一批RF导航实现，不可跨模型比较为既有系统收益。5项单测已通过，配对子集明细在recovery_paired_support.json。
