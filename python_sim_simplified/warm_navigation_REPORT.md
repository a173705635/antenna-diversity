# 正常初始化后连续导航：076–100 轮

## 结论

25轮已执行，12轮保留、13轮拒绝。这些是观测域算法实验，不能作为RF抗欺骗或三/四天线整机通过证据。冻结留出未出现水平误差≥30m/≥50m的有效输出，但可用率只有69.07%，最长中断90秒，因此持续定位目标未通过。

留出共48个种子×场景组合（3种子×16场景），来自3次独立种子抽样；同一种子的16个场景复用天空几何和共同噪声，是配对反事实组合，不是48个统计独立样本。共8640个攻击后/对照同期历元；二、三、四天线标签共用这些记录，不将其重复为三倍样本。

|集合|同期历元|输出|水平≤20m|水平≥30m|水平≥50m|无输出|可用率|有效输出水平条件P95|最长中断|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|开发|5760|4007|3917|0|0|1753|69.57%|12.18m|90.0s|
|冻结留出|8640|5968|5839|0|0|2672|69.07%|14.42m|90.0s|

上述误差阈值及P95均按水平位置误差统计，P95仅对有效输出取分位数。独立复核另外计算的留出三维位置条件P95为17.58m（17.581001700822377m），三维≥30m和≥50m有效输出也均为0；此补充不替换逐轮冻结的水平统计。

全部初始化在epoch29、14.5s完成；攻击注入epoch60、30s，间隔15.5s。初始化不是等待固定时长：需要30个有效码定位以及窗口水平散布≤12m。零初始化失败仅是本次48个留出组合结果；失败通过返回None和独立统计表达，测试覆盖。时长0.5s/epoch，240epoch；攻击后180epoch。

## 输入与可识别性边界

- 使用10星固定名义LOS、局部线性化伪距及载波米值，时钟由四维最小二乘单独估计；没有真实采样RF、相关器、阵列流形、消干扰或星历迭代。单源攻击用所有星的共同位置/时钟牵引代理，不证明单RF源可实现该输入。
- 2/3/4仅是同一导航输入的配置标签。没有用天线数缩小噪声，也没有声称两天线失败源于自由度；阵列几何、同步、标定和真实信号损失均未在这条线建模，因此无法归因或证明三/四天线收益。三天线最低目标与四天线主配置均尚未证实。
- normal、clock_drag、turn、colored、weak、ionosphere等包含正常/误差对照；汇总分母并非全是攻击序列，逐场景见下表。攻击接入、真信号消失以观测牵引或lock位代理；authentic_absent在140epoch后全部失锁，并未模拟攻击源仍可跟踪而真源独自消失。
- 初始化位置与速度来自观测，不读truth、场景名称、种子或attack时刻。truth与attack只用于评分。固定名义LOS仍是仿真假设。
- 初始位置窗口均值有运动滞后；080的当前时刻对齐减小滞后却增加攻击阈值计数，因此按预定排序被拒绝。这暴露指标选型偏差，不能把该拒绝解释为更准确的初始化。
- CMC差异能看到码载波不一致，无法认证码和载波共同牵引。速度门依赖初始化后运动受限；094的温和速度适应改善转弯，也会吸收慢速攻击。慢牵引仍有20–30m水平误差，水平≥30m零错误不能外推更长攻击时长。
- 失锁期间不给输出。正常dropout重新接上码观测，攻击重捕示例因之前报警继续拒绝；没有独立来源认证。088/089/100的隔离、重基准、恢复策略未改善预定分数而拒绝。不能宣称一般攻击重捕获已解决。
- 076/093/096/097/100等虽有实质算法改动，但本次阈值统计无改进，按规则拒绝，未当成成功迭代。两次执行错误（076缩进、098替换定位）在对应轮修正，未另计轮。

## 留出逐场景（3个种子合计）

|场景|输出/540|水平≤20m|水平≥30m|水平≥50m|最大中断s|
|---|---:|---:|---:|---:|---:|
|normal|540|540|0|0|0.0|
|code_drag|97|97|0|0|76.0|
|joint_drag|9|9|0|0|88.5|
|joint_step|0|0|0|0|90.0|
|clock_drag|540|540|0|0|0.0|
|code_fault|540|540|0|0|0.0|
|slip|540|540|0|0|0.0|
|dropout|480|480|0|0|10.0|
|weak|504|504|0|0|18.0|
|ionosphere|540|540|0|0|0.0|
|turn|540|540|0|0|0.0|
|reacquire_attack|9|9|0|0|88.5|
|slow_drag|540|411|0|0|0.0|
|phase_fault|540|540|0|0|0.0|
|colored|540|540|0|0|0.0|
|authentic_absent|9|9|0|0|88.5|

## 逐轮实测决策

排序按水平误差固定为最少≥50m、最少≥30m、最多≤20m有效输出；每轮在相同开发输入下运行父版及候选。没有将重复种子、扫描、修bug或注释计轮。源哈希只是必要证据，每轮机制写在proposal.json。

|轮|机制/假设|决策|水平≥30m前→后|水平≤20m前→后|
|---|---|---|---:|---:|
|76|Use measured per-satellite C/N0 weighting to reduce weak-channel contribution.|拒绝|1279→1279|3916→3916|
|77|Equal per-epoch C/N0 produced no change; robust satellite residual weights may reduce isolated code faults.|保留|1279→1278|3916→3978|
|78|Residual robustness leaves coherent attacks; first remove persistent single-channel faults with leave-one-out consistency.|拒绝|1278→1278|3978→3977|
|79|Code-only faults remain large; carrier propagation with code blending should reduce code drag and noise.|保留|1278→1188|3978→4099|
|80|Warm mean position lags motion; transport initialization code estimates to current epoch using observed carrier differences.|拒绝|1188→1197|4099→4083|
|81|Carrier smoothing still follows code drag; monitor code-minus-carrier change in position subspace, allowing receiver clock change.|保留|1188→990|4099→3181|
|82|CMC cannot see common code/carrier steps; a large measured carrier position jump should latch output protection.|保留|990→630|3181→3181|
|83|Common slow pull is invisible to CMC; compare TDCP velocity against velocity estimated during benign acquisition.|保留|630→0|3181→2683|
|84|Zero >=30m errors now costs over half of output epochs; make warm-velocity gate a three-epoch sequential detector to avoid isolated noisy triggers.|保留|0→0|2683→2696|
|85|Continuity remains poor; isolated satellite cycle slips should be excluded in TDCP residual space before velocity screening.|保留|0→0|2696→2976|
|86|Cycle-slip screening improves availability; robust CMC should stop one carrier ambiguity jump poisoning the shared source monitor.|保留|0→0|2976→3401|
|87|Robust CMC still reacts to isolated weak-code noise; require persistent CMC disagreement before latching.|保留|0→0|3401→3591|
|88|After a tracking gap, prior authenticity is insufficient: require five consecutive code/phase-consistent epochs before output.|拒绝|0→0|3591→3581|
|89|Fixed quarantine lost outputs without safety benefit. Test conditional ambiguity rebasing only when reacquired code agrees with benign observed trajectory; agreement is not authentication.|拒绝|0→0|3591→3591|
|90|Weak-signal availability is 19/180; normalize CMC boundary using declared code-noise covariance instead of an absolute noise-independent threshold.|保留|0→0|3591→3752|
|91|Adaptive CMC releases weak outputs, but some exceed20m; use C/N0-dependent code injection into carrier smoother.|拒绝|0→0|3752→3725|
|92|Weak carrier rejection repeatedly resets smoother to noisy code; use observed warm velocity propagation plus code correction when TDCP is unavailable.|保留|0→0|3752→3835|
|93|Prediction improves <=20m count; TDCP screening should scale with measured phase uncertainty rather than reject most weak observations.|拒绝|0→0|3835→3835|
|94|Fixed benign velocity falsely stops a genuine turn; permit gradual velocity adaptation only below the existing innovation boundary.|保留|0→0|3835→3917|
|95|Adaptive motion restores turn outputs but slow drag remains; test accumulated TDCP displacement against initial observed constant-velocity trajectory.|拒绝|0→0|3917→3771|
|96|Trajectory envelope suppresses legitimate turns; instead test initialization evidence after detrending motion and require horizontal mean uncertainty <=5m.|拒绝|0→0|3917→3917|
|97|Initialization variant changes no tested outputs; limit horizontal code innovation influence to prevent large code faults injecting into carrier trajectory.|拒绝|0→0|3917→3917|
|98|Bounded code influence gave no threshold-count benefit; test divergence-free carrier increment using half the measured code-minus-carrier increment to compensate opposite-sign ionospheric drift.|拒绝|0→913|3917→4111|
|99|Divergence correction admitted severe coherent errors and was rejected; add a geometry/noise horizontal protection bound to withhold unreliable absolute fixes during weak tracking.|拒绝|0→0|3917→3557|
|100|Absolute protection bound only removes usable weak epochs; test alarm release after ten consistent epochs agreeing with observed warm trajectory and CMC, explicitly without source-authentication claim.|拒绝|0→0|3917→3917|

## 冻结与复验

`results/research-warm-hundred/navigation/`下保存每轮proposal、before.py/after.py、before.json/after.json、conclusion，ledger.jsonl、development.json、holdout.json、freeze.json及开发输入NPZ/manifest。冻结文件包括估计器、生成器、运行器和测试源码；冻结后只运行新种子112001/112013/112037，未据留出回调。最终选择为094累计保留源码。

在`F:/antenna_diversity/python_sim_simplified`运行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
D:/Python/Python311/python.exe -m pytest tests/test_warm_navigation.py -q
D:/Python/Python311/python.exe warm_navigation_research.py --verify
D:/Python/Python311/python.exe warm_navigation_research.py --holdout
```

本线测试6项通过；--verify完成076–100全部25轮档案复演，父版及候选的已归档汇总逐字段一致。验证状态另存verification.json。
