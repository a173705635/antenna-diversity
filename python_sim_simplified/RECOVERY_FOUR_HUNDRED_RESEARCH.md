# 第 201–400 轮研究结果

结论：有局部精度和可用性收益，但按用户≥30m即失败的要求，新候选未在整体留出中胜出。独立加边界的攻击记录中，≥30m输出合计由284增到295，≥50m由56降到24；不能只看严重错误减少。正常输出率为95.0%和98.4%，也未达到99%目标。

已完成200个编号实验：40个开发RF工况/种子组合、60个同输入规则比较、20个观测域反例、40个独立RF留出及40个边界RF留出。共240条正常/攻击RF记录、120套不同种子天空几何；规则重复、配对记录和相邻历元不算独立攻击样本。另重放3条旧失败记录，不计入新种子成绩。

所选方法：`calibrated`，开发规则来自第257轮。

RF记录指合成基带试验的观测归档，不是台架实测；全量IQ由种子与源码重生成。冻结接收方法和后处理Rule，部分预注册灵敏度工况改变前端积累窗，因此合并比例不是单一默认硬件配置的现场成功率。

```json
{
  "window": 5,
  "mature": 1,
  "divergence": null,
  "spatial": 1.0,
  "consensus": null,
  "weighted": true
}
```

只在开发集选型，留出阶段参数冻结。95%正常可用性是开发折中，产品目标99%保持不变。120个开发候选中没有任何一个达到99%正常输出率。最长RF记录8秒，观测域反例10秒；均不代表分钟级、长期或实机导航验证。

## 同一输入配对结果

|阶段|方法|正常输出率|正常≤20m/全历元|攻击输出率|攻击≤20m/全历元|攻击≥30m|攻击≥50m|攻击输出P95(m)|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|development|raw|100.00%|2526/2560|100.00%|1665/2560|451|256|100.74|
|development|calibrated|100.00%|2446/2560|100.00%|2080/2560|374|271|140.70|
|development|previous|96.84%|2449/2560|64.96%|1293/2560|112|17|31.11|
|development|alternate|96.60%|2399/2560|84.57%|2085/2560|36|16|16.78|
|development|candidate|96.60%|2458/2560|86.09%|2155/2560|20|10|12.45|
|independent|raw|100.00%|3334/3360|100.00%|2047/3360|878|319|67.73|
|independent|calibrated|100.00%|3276/3360|100.00%|2525/3360|575|350|107.21|
|independent|previous|94.85%|3186/3360|59.20%|1659/3360|92|0|29.64|
|independent|alternate|94.88%|3175/3360|80.18%|2512/3360|88|19|23.54|
|independent|candidate|95.00%|3192/3360|82.17%|2598/3360|84|8|21.96|
|boundary|raw|98.44%|11548/11760|96.87%|7956/11760|2314|1948|3919.67|
|boundary|calibrated|98.44%|11091/11760|98.44%|7955/11760|3208|2725|2527.75|
|boundary|previous|96.83%|11365/11760|65.82%|7174/11760|52|30|22.00|
|boundary|alternate|97.89%|11469/11760|70.72%|7721/11760|196|37|23.00|
|boundary|candidate|98.38%|11569/11760|72.19%|7889/11760|211|16|22.74|

previous为上一轮主策略raw+rule143；alternate为上一轮备选calibrated+rule146。candidate才是本次新增平滑/空间观测链。未输出不能计为成功导航；P95只针对已输出，所以须与全历元比例一起阅读。

![配对结果](results/recovery-four-hundred/holdout.png)

## 仅在发射攻击的时间窗统计

上表按整条攻击记录计分，含撤销攻击后的恢复段。下表严格使用已配置的攻击起止时间，避免把恢复段计入持续受攻击能力。

|阶段|策略|有解/攻击窗历元|≤20m|≥30m|≥50m|
|---|---|---:|---:|---:|---:|
|independent|alternate|2635/3301|2453|88|19|
|independent|candidate|2702/3301|2539|84|8|
|boundary|alternate|8218/11661|7622|196|37|
|boundary|candidate|8391/11661|7791|211|16|

## 留出消融

消融均使用冻结候选的其余参数，不按留出表现重新选择。

|阶段|方法|攻击有解/全部|攻击≤20m|攻击≥30m|攻击≥50m|
|---|---|---:|---:|---:|---:|
|independent|candidate|2761/3360|2598|84|8|
|independent|no_smoothing|2695/3360|2512|88|19|
|independent|no_spatial|2760/3360|2598|84|8|
|independent|no_consensus|2761/3360|2598|84|8|
|boundary|candidate|8489/11760|7889|211|16|
|boundary|no_smoothing|8354/11760|7725|223|42|
|boundary|no_spatial|8461/11760|7889|168|20|
|boundary|no_consensus|8489/11760|7889|211|16|

no_smoothing同时取消平滑和弧成熟等待，是时间处理整体消融；no_spatial取消空间分数限制；no_consensus取消加权/无权解差限制。若冻结候选本来没启用该项，其消融自然相同。

空间门限没有稳定收益：边界留出中，取消它时≥30m输出为168，开启后为211，全历元≤20m数量均为7889。363轮t=1.79s，同一输入在无空间门限时误差19.67m，开启后52.92m；PRN8的空间分数约0.323，被排除后其余7星形成了更偏的位置。该记录有7个历元从≤20m变为≥30m。见spatial_gate_counterexample.json；不能把新增质量门限默认解释成可靠性提升。

## 按记录而非历元观察失败

|阶段|方法|攻击中出现≥30m的记录|最长连续无输出(s)|最大输出误差(m)|
|---|---|---:|---:|---:|
|independent|alternate|5/40|0.84|77.64|
|independent|candidate|3/40|0.84|54.55|
|boundary|alternate|6/40|7.83|76.76|
|boundary|candidate|4/40|7.83|71.78|

![逐轮收益及错误](results/recovery-four-hundred/rounds.png)

## 旧失败重放：改善与归因

三个旧种子只用于归因，未加入新种子通过率。原始raw观测和位置逐数组重现。

|旧编号|策略|有解/全部|≤20m|≥30m|输出P95(m)|
|---|---|---:|---:|---:|---:|
|171|previous|2/84|1|1|92.22|
|171|alternate|84/84|81|0|18.40|
|171|candidate|84/84|84|0|12.17|
|185|previous|100/184|99|1|9.89|
|185|alternate|135/184|135|0|12.52|
|185|candidate|140/184|140|0|8.11|
|190|previous|128/184|1|104|41.61|
|190|alternate|184/184|149|4|25.21|
|190|candidate|184/184|177|0|18.99|

旧185轮t=0.17s：previous水平误差108.03m，alternate约8.48m，candidate约2.54m。大部分修复已来自上一轮备选接收链，本轮平滑带来进一步改进，不能把全部改善都归功于新增算法。

## 冷启动和弱欺骗的失败机理

新350轮真实通道相位失配为0°，受骗启动估计的相对相位却约为0°、−108.8°、171.5°、35.3°。校准前提被破坏后，空间分数仍能通过：83/84个历元输出，0个≤20m，67个≥30m、8个≥50m，最大约54.55m。旧190轮改善不能推广为冷启动已解决。见cold_start_diagnostic.json及方法文档中的混合信号公式。

新338轮弱欺骗功率为+3dB：原始协方差特征值比始终小于3，空间证据计算退化为预期方向匹配相关；该方向仍可能含欺骗投影，因此高空间分数不能证明锁定真码。短时解扩已在进行零陷，说明恢复链与这个空间证据的可观测前提不同。见weak_source_diagnostic.json。

350轮追加一次同IQ消融：保持短时解扩，取消启动相位校准，原位置84/84个≤20m；再使用冻结后处理规则后，仍为84/84、P95约2.78m、最大约3.28m。它支持“这条记录的错误校准造成失败”的归因；此工况真实相位失配为0，不能推出取消校准能处理真实通道失配。本次候选保持原样，未按这个事后样例重新选型；见cold-calibration-ablation/comparison.json。

## 8秒与运动边界

|编号|策略|有解/全部|≤20m|≥30m|≥50m|最长无输出(s)|
|---|---|---:|---:|---:|---:|---:|
|369|alternate|184/184|184|0|0|0.00|
|369|candidate|184/184|184|0|0|0.00|
|370|alternate|184/184|184|0|0|0.00|
|370|candidate|184/184|184|0|0|0.00|
|379|alternate|384/384|384|0|0|0.00|
|379|candidate|384/384|384|0|0|0.00|
|380|alternate|384/384|384|0|0|0.00|
|380|candidate|384/384|384|0|0|0.00|
|381|alternate|291/784|291|0|0|3.32|
|381|candidate|307/784|307|0|0|4.22|
|382|alternate|451/784|444|0|0|1.48|
|382|candidate|454/784|452|0|0|3.02|
|383|alternate|784/784|784|0|0|0.00|
|383|candidate|784/784|784|0|0|0.00|
|384|alternate|1/784|1|0|0|7.83|
|384|candidate|1/784|1|0|0|7.83|
|399|alternate|184/184|184|0|0|0.00|
|399|candidate|184/184|184|0|0|0.00|
|400|alternate|0/184|0|0|0|1.84|
|400|candidate|0/184|0|0|0|1.84|

369/379分别为2/4秒15m/s运动；370/380为2/4秒20m/s渐变拖曳。381–384依次为8秒名义、2m/s运动、10m/s渐变拖曳、无真实信号；399为2秒30m/s运动，400为2秒受骗启动且无真实信号。无真信号场景初期短暂好坐标可能来自启动跟踪状态与20ms观测窗口，不能当作真实信号已恢复。

381轮仅307/784个历元输出，全部在20m内。所有历元均有7或8星，形式水平标准差没有超限；残差和删星一致性门限分别触发476和466次（重叠计数），总共拒绝477次，其中475次未经准入的校准位置当时已≤20m。这是准入保守性/模型不匹配的明确代价，不能事后知道真值达标就放松在线门限。见long_record_admission_diagnostic.json。

382轮为2m/s运动：未经准入的校准解在前4秒没有≥30m，后段逐渐失稳；PRN4在5秒后的相对公共钟差码偏差中位数约118m，C/N0约35.4dB-Hz、锁定指标约0.637，仍有约67%的后段历元通过初步选星。原始协方差方法该记录783/784个≤20m。需要检查恢复/跟踪链的偏差来源，不能把它都当成可由平滑消除的白噪声。见moving_long_record_diagnostic.json和moving_long_channel_diagnostic.json。

## 数值反例

留出最大已输出候选水平误差：第367轮，t=1.53s，71.78m。同历元raw误差56.82m，上一轮备选误差69.23m。开发集最大为118.92m；见failures.json完整逐历元记录。

![失败轨迹](results/recovery-four-hundred/worst_trace.png)

|编号|观测域工况|伪距RMS(m)|最大伪距误差(m)|≥30m观测数|
|---|---|---:|---:|---:|
|301|nominal|0.52|3.63|0|
|302|common_clock|0.51|1.90|0|
|303|receiver_motion|0.46|3.14|0|
|304|coherent_false_step|83.74|101.47|5600|
|305|coherent_slow_pull|33.85|70.71|3209|
|306|code_only_step|29.51|101.82|698|
|307|carrier_cycle_slip|0.59|8.90|0|
|308|outage|0.56|5.61|0|
|309|missing_time|0.55|3.63|0|
|310|constant_code_bias|100.03|102.59|7984|
|311|nominal|0.28|4.19|0|
|312|common_clock|0.27|4.89|0|
|313|receiver_motion|0.25|3.00|0|
|314|coherent_false_step|83.76|100.58|5600|
|315|coherent_slow_pull|33.87|70.31|3198|
|316|code_only_step|29.57|100.43|698|
|317|carrier_cycle_slip|0.80|9.40|0|
|318|outage|0.39|3.77|0|
|319|missing_time|0.37|3.08|0|
|320|constant_code_bias|100.00|101.84|7984|

观测域统计单位是每星伪距，不是位置误差，不与RF水平定位表合并。所有20条反例通过前缀因果性检查。coherent_false_step/slow_pull在这组观测域输入中给所有星相同公共距离变化，能被位置解中的钟差吸收，不能用其伪距误差声称水平定位失败。真正一致的200m假位置由单元负例验证；RF里的渐变假位置由各星不同几何伪距驱动。码载波一致的伪变不会被Hatch递推本身识别。

## 可复现性与范围

在python_sim_simplified目录运行，设置OPENBLAS_NUM_THREADS=1和OMP_NUM_THREADS=1。依次执行：

```powershell
# 在另一个结果目录复现时，先修改audit_recovery_four_hundred.py中的ROOT。
python -c "from audit_recovery_four_hundred import ROOT,FILES; import shutil; p=ROOT/'source-v1'; p.mkdir(parents=True); [shutil.copyfile(f,p/f) for f in FILES]"
python audit_recovery_four_hundred.py --rounds (201..240)
python audit_recovery_four_hundred.py --rounds (241..300)
python audit_recovery_four_hundred.py --select
python audit_recovery_four_hundred.py --rounds (301..320)
python audit_recovery_four_hundred.py --rounds (321..400)
python diagnose_recovery_four_hundred.py --prepare
python diagnose_recovery_four_hundred.py
python diagnose_recovery_four_hundred.py --calibration-ablation
python -m pytest tests -q > four_hundred_tests_final.log
python summarize_recovery_four_hundred.py
```

复现须使用一个尚不存在的结果目录（修改脚本ROOT或使用干净检出），不要覆盖已归档实验。逐轮frozen.json、NPZ、rows和summary共同记录输入与输出。最终汇总重算全部保存位置指标、重放160条冻结留出准入并逐数组核对，生成消融与逐历元失败清单。

旧失败重放依赖results/recovery-two-hundred中相应历史归档；主试验生成不依赖旧数据。汇总保存源码、测试输出和SHA256清单。当前全量266项测试通过，1条既有无噪声相位测试的除零警告保持原状。

完整数学模型和假设见RECOVERY_FOUR_HUNDRED_METHODS.md；最初编号/选择约定见RECOVERY_FOUR_HUNDRED_PLAN.md。仍缺同步RF台架数据、时变姿态/多径/温漂/ADC和可信冷启动认证；没有进行商业放行或推送。

下一步优先研究显式真/伪双分量的启动校准、持续更新的解扩/跟踪状态，以及符合残余偏差的准入模型；方法文档给出两天线自由双分量拟合的不可辨识反例。这些是后续待验证方向，本轮没有将其包装成已完成商用能力。

冻结重放前修复了一个实验配置读取问题：既有save函数把无限阈值写成JSON null，新增明确的阈值反序列化来恢复“禁用该门限”的数学含义。原始开发计算未改变，开发与反例源码保存在source-v1/，后续源码在final_source/；汇总分别核对相应SHA256并重新验证所选开发总数。null不是缺字段默认回退。
