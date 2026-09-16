# 第 101–200 轮连续抗欺骗研究结果

已完成 100 个编号实验。当前结果仍属于合成连续基带研究，不是商用放行或长期可靠导航证明。

新增 240 条不同场景/种子/正常攻击组合的 RF 记录，720 次保存位置序列比较；另将启动校准方法在 80 条同输入开发记录上重算，20 个准入规则重复使用这些开发数据，不增加独立攻击试验数。

开发选中方法：`raw`；规则来自第 143 轮。符合预先设定的开发可用性折中条件的候选数：35。

另追加10个工况、两个新种子正常/攻击配对的40条选型取舍留出，单独列出，不改变100轮主验证。

## 配对结果

|阶段|方法|正常输出率|正常≤20m/总历元|攻击输出率|攻击≤20m/总历元|攻击≥30m|攻击≥50m|
|---|---|---:|---:|---:|---:|---:|---:|
|development|raw|100.00%|2495/2560|100.00%|1588/2560|683|387|
|development|short|100.00%|2456/2560|100.00%|1978/2560|442|278|
|development|mvdr|100.00%|2498/2560|100.00%|696/2560|1480|738|
|development|sic|100.00%|2495/2560|100.00%|1451/2560|780|440|
|development|calibrated|100.00%|2449/2560|100.00%|1977/2560|437|249|
|independent|raw|100.00%|3286/3360|100.00%|2358/3360|686|481|
|independent|candidate|97.14%|3261/3360|63.90%|1910/3360|56|3|
|boundary|raw|94.09%|5606/5960|94.09%|3493/5960|1295|884|
|boundary|candidate|93.26%|5555/5960|57.37%|2989/5960|174|15|

`development` 是调参数据，`independent` 为 161–180 轮新种子，`boundary` 为 181–200 轮边界新种子。不同阶段含不同难度与时长，不能把百分比差异解释为相同输入上的提升。

![留出可用性与合格位置](results/recovery-two-hundred/holdout.png)

## 同一观测上的旧/新准入比较

这里固定本轮选中的信号恢复方法，只改变准入。旧规则为残差≤10m、删星子集位置差≤5m；新规则使用观测质量和形式协方差归一化。

|阶段|规则|正常输出/机会|攻击输出/机会|攻击≤20m|攻击≥30m|攻击≥50m|
|---|---|---:|---:|---:|---:|---:|
|independent|old_gate|2419/3360|504/3360|489|1|0|
|independent|new_gate|3264/3360|2147/3360|1910|56|3|
|boundary|old_gate|4039/5960|736/5960|725|1|0|
|boundary|new_gate|5558/5960|3419/5960|2989|174|15|

## 新种子的选型取舍补充

主规则优先减少≥50m错误；备选优先减少全部≥30m错误，两者都从开发集选出。补充数据生成前已冻结规则，主验证未改动。这里固定同一输入，直接比较两种完整策略。

|方法/策略|正常输出/机会|正常≥30m|攻击输出/机会|攻击≤20m|攻击≥30m|攻击≥50m|
|---|---:|---:|---:|---:|---:|---:|
|raw|1880/1880|0|1880/1880|1226|368|244|
|calibrated|1880/1880|0|1880/1880|1363|321|248|
|primary|1870/1880|0|1428/1880|1346|4|0|
|alternate|1876/1880|0|1633/1880|1594|1|0|

备选为 `calibrated` + 第146轮准入。只有两套新基础天空几何，跨十工况复用；这是局部证据，不是广泛实测。

![两种选型取舍的逐工况结果](results/recovery-two-hundred/tradeoff.png)

## 冻结规则

```json
{
  "round": 143,
  "method": "raw",
  "gate": {
    "min_gain": 0.05,
    "min_lock": 0.8,
    "min_cn0": 25.0,
    "sigma45": 5.0,
    "residual_z": 3.0,
    "separation_z": 5.0,
    "horizontal_sigma": 20.0
  },
  "normal": {
    "opportunities": 2560,
    "solved": 2477,
    "good20": 2453,
    "error30": 6,
    "error50": 0
  },
  "attack": {
    "opportunities": 2560,
    "solved": 1629,
    "good20": 1341,
    "error30": 115,
    "error50": 6
  },
  "feasible_development_rules": 35,
  "selection_rule": "Normal availability >=95%, attack >=50% if feasible; then minimize >=50, >=30 errors, maximize good20. No claim of integrity."
}
```

开发的正常≥95%、攻击≥50%可用率仅为避免选择退化的折中，不取代正常99%及错误位置不放行的产品要求。通过规则不等于完成真实性认证。新增恢复方法若未被选中，其收益只得到开发集支持，不能据本轮最终留出宣称已经验证。

## 可核查的失败

|轮次|记录|时间/s|候选水平错误/m|使用星数|
|---:|---|---:|---:|---:|
|185|940185_1.npz|0.17|108.03|6|
|171|920171_1.npz|0.39|96.82|8|
|172|920172_1.npz|0.35|95.97|6|
|191|950191_1.npz|0.97|95.50|7|
|191|950191_1.npz|0.30|95.21|7|
|191|950191_1.npz|0.92|92.20|7|
|191|950191_1.npz|0.63|90.68|7|
|191|950191_1.npz|0.62|88.64|7|
|191|950191_1.npz|0.57|85.39|6|
|186|950186_1.npz|1.27|73.22|6|

完整失败历元含归一化残差、分离分数及水平形式标准差，见 `results/recovery-two-hundred/failures.json`。没有因留出失败而事后修改准入规则。

严重反例：第185轮 seed940185 的 t=0.17s，原始水平误差12.13m，选星并加权重估后变成108.03m。此时仅使用PRN 2/3/5/6/7/8，归一化残差0.642、分离分数0.871、水平形式标准差18.35m均通过门限。新增准入包含重新估计位置，所以不仅可能漏拒，还可能把原本较好的解变坏。形式协方差未覆盖真实偏差，不能用它作为认证保护级。

![最大已准入错误所在轨迹](results/recovery-two-hundred/worst_trace.png)

## 方法、论文和验证

用户提供的论文已核对公开摘要，并查阅作者公开博士论文；本项目的空间辅助局部副本抵消属于受其启发的实现，尚非完整矢量接收机复现。论文细节、数学公式、假设与对应差异见 [方法说明](RECOVERY_TWO_HUNDRED_METHODS.md)。

全部逐轮数据见 [100轮表](results/recovery-two-hundred/round_table.md)。准入轮表展示最终选中恢复方法，各轮完整五方法数据另存对应 `summary.json`。

![逐轮结果](results/recovery-two-hundred/rounds.png)

测试覆盖空间分离副本抵消、未知比特符号、相位校准、正常/攻击同输入、攻击撤离、污染启动以及一致假位置仍能通过准入的反例。实际测试输出：

```
........................................................................ [ 27%]
........................................................................ [ 55%]
........................................................................ [ 83%]
............................................                             [100%]
============================== warnings summary ===============================
tests/test_phase_origin.py::test_locked_phase_retains_fraction_with_changed_start_and_nco
  F:\antenna_diversity\python_sim_simplified\tracking.py:82: RuntimeWarning: divide by zero encountered in scalar divide
    coherent_power / (noise_power * coherent_interval_s)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
260 passed, 1 warning in 159.92s (0:02:39)
```

汇总程序重新评价全部保存位置，并重算冻结候选位置核对一致；旧准入在同一留出观测上另行计算。`final_source/` 保存源码，`manifest.json` 保存结果哈希。

## 局限与下一步

本轮最长连续记录仅4秒，接收机静止、真实方向和姿态已知、绝大多数组假设前100ms干净。第190轮故意破坏可信启动前提。没有真实台架、在线矢量导航反馈、独立欺骗PLL、长期模板更新或GNSS/IMU闭环。

短时解扩能减少未知比特相消，固定相位偏差可由干净启动部分估计，但误差仍可能来自空间估计污染、近同方向不可分、模板漂移及码跟踪偏差。加权残差和分离只检验自洽性，完全一致的假位置仍可能通过。下一步先针对错误选星/加权放大与启动信任建立研究，结合恢复后逐星空间真实性、可信动态模板和独立运动约束；使用新的留出验证可用率与错误率同时改善。

## 复现

在独立工作副本中运行，输出必须为新目录。环境为本机 Python 3.11 与现有 NumPy/SciPy/Matplotlib/pytest；将工作目录设为该副本的 `python_sim_simplified`。下面 PowerShell 命令每条单独执行：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python audit_recovery_two_hundred.py --output results/recovery-two-hundred --rounds (101..140)
python augment_startup_calibration.py --output results/recovery-two-hundred --rounds (101..140)
python audit_recovery_two_hundred.py --output results/recovery-two-hundred --calibrate
python audit_recovery_two_hundred.py --output results/recovery-two-hundred --select
python audit_recovery_two_hundred.py --output results/recovery-two-hundred --rounds (161..200)
python audit_recovery_tradeoff.py
python -m pytest tests -q > results/recovery-two-hundred/tests.txt
python summarize_recovery_two_hundred.py --output results/recovery-two-hundred
```

离线运行时间明显长于波形时长；10/20/40ms块处理还需要对应缓冲延迟，未验证实时处理能力。
