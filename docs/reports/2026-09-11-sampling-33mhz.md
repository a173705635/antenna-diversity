# 33 MHz RF / 5倍抽取 / 6.6 MHz跟踪

2026-09-11，按用户指定采样率更新默认配置。

## 当前参数

- `SignalConfig.fs_rf_hz = 33e6`。
- `DFEConfig.decimation = 5`。
- 捕获及DLL/FLL/PLL从DFE返回值获取采样率，实际均使用6.6 MHz；每1 ms处理6600点，约6.4516 samples/chip。
- 中频4.092 MHz、GPS L1载频1575.42 MHz、码率1.023 MHz。
- 仍用矩形信号源及矩形本地码，不启用连续副本或初始相位细化。
- FIR仍为129 taps、截止1.8 MHz，RF增益、噪声带宽、量化、环路增益及定位权重未另行调整。

变化集中在 `python_sim_simplified/config.py`。旧固定数值回归测试明确保留16.368 MHz/4倍抽取快照；历史伪距诊断从归档读取原采样率，避免默认值改变导致旧对照悄悄换条件。新定位入口使用新的默认值。

## 验证

重新生成单天线8个GPS PRN，3组RF种子3000、3001、3002，各1秒；每组200～990 ms的80个位置全部解出，共240/240。每组8颗卫星均捕获成功并满足现有观测筛选。联合浮点解读取本次6.6 MHz NCO历史，不复用旧采样率的相位记录。

| 种子 | 伪距定位三维RMS（m） | 联合浮点三维RMS（m） | 联合解去均值三维RMS（mm） |
|---|---:|---:|---:|
|3000|6.388|3.647|2.267|
|3001|5.416|2.786|2.391|
|3002|6.502|3.557|2.255|

仍是时间辅助合成星座、无多径/外部干扰；联合解是离线浮点解，未固定整周数，去均值毫米级波动不是绝对定位精度。原来周跳保护不完整的边界仍适用。

同样的FIR taps/截止参数在新RF采样率下对应的离散滤波响应有所变化；现有RF噪声是独立离散样点模型，采样率变化也影响其经DFE后的噪声表现。因此表中收益不能全部归因于DLL网格变细，也不是仅改变采样率的真实硬件精度保证。

首次全套测试107通过、2个跨采样率测试失败。定位发现测试分别把同一理想延迟四舍五入到不同采样网格，实际起点不相同；修正为共同RF网格起点后，跨采样率检查5项通过，数值精度门限未放宽。没有修改跟踪算法来迎合测试。

最终完整测试为109项通过（85.02秒）。两份新结果归档各7个文件均通过独立校验：伪距归档核对240次定位，联合浮点归档核对输入哈希及240次联合定位。历史16.368 MHz/4倍抽取覆盖参数也已检查，输出仍为4.092 MHz。

## 运行与归档

```powershell
python python_sim_simplified/run_gps_spp.py --output python_sim_simplified/results/gps_spp_33mhz_repeat
python python_sim_simplified/audit_code_carrier_float.py --input python_sim_simplified/results/gps_spp_33mhz_repeat --output python_sim_simplified/results/code_carrier_float_33mhz_repeat
python -m pytest python_sim_simplified/tests -q
```

结果分别归档至 `docs/reports/data/2026-09-11-gps-spp-33mhz/` 和 `docs/reports/data/2026-09-11-code-carrier-float-33mhz/`。默认研究入口均拒绝覆盖已有输出目录。
