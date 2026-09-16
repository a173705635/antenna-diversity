# SDCP multipath without positioning — frozen experiment evidence

完整中文方法、结果、图示与限制见 [验证报告](../../SDCP_MULTIPATH_NO_POSITIONING_REPORT.md)，预定主方案和标定后补充方案见 [实验计划](EXPERIMENT_PLAN.md)。

- `plan.json` / `config.json` / `code/`：主实验配置、随机种子、实际运行源码与哈希。
- `cal_*` / `threshold.json`：独立正常标定及测试前冻结的绝对门限。
- `*_A.npz` / `*_B.npz`：66 次独立接收链调用的跟踪历史，包含累积 NCO、Prompt 等；同名前缀 JSON 是捕获、RF、DFE 元数据。
- `*_decision.npz` / `results.json`：48 条测试的原始 SDCP 残差、200 ms 分数、逐次判定和统计；不取余、不去本记录均值。
- `window_change/`：探索性相邻双窗口变化判据，保留门限、48 条分数/报警轨迹及逐场景表。
- `calibration_diagnostics.json`：正常数据中的半周分支与 Prompt 交叉相位诊断。
- `verified_results.json`：独立复算核对的统计及事后同噪声配对解释量；配对真值对照不进入检测器。
- `verification.json` / `completion.json`：核验范围、源码哈希与运行完成记录。
- `analysis_code/`：本轮后处理、复算代码和相关新增测试快照。
- `SHA256SUMS.json`：本目录其他全部文件的 SHA-256 清单；自身不在清单中。

本批是本地仿真，不是实测录波。绝对门限因未解决的 Costas 半周分支达到 119.835 mm，本批 39 条多径记录均未报警；变化判据门限 0.951 mm，可触发部分中途差分变化，但启动静态污染漏检、共同污染不稳定、非多径慢漂移误报。不能将这批小样本计数当作商用检测概率。

实际执行相关测试：

```powershell
D:/Python/Python311/python.exe -m pytest tests/test_multipath.py tests/test_channel_delay.py tests/test_rf_dfe.py tests/test_tracking_initialization.py tests/test_sdcp_multipath_no_positioning.py tests/test_sdcp_no_positioning.py -q
```

结果：22 passed in 12.37s。主仿真 864.78 s；48 条测试、3 条标定以及 48 条补充判定全部独立复算通过。
