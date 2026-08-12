# 捕获搜索结果可视化设计

## 目标

在不改变 FFT 捕获数值流程的前提下，直观显示码相位与 Doppler 二维搜索结果，并保留最佳 Doppler 下的一维相关切片，方便判断主峰是否清晰、错误峰来自哪个维度，以及 single、EGC、MVDR 的差异。

## 接口和职责

- `receiver.acquire` 继续只计算捕获结果，额外返回第二峰码相位和主峰排除宽度；
- 新增 `analysis.plot_acquisition(results)`，只读取结果并返回 figure 句柄；
- `analysis.plot_results(results)` 调用新函数，不在捕获计算循环中创建图窗；
- `cfg.plot.enable=false` 时端到端流程仍完全不创建图窗。

## 图形布局

每个分支占一行、两列：

1. 左列为码相位 × Doppler 二维搜索热力图；
2. 右列为估计 Doppler 所在行的一维码相位切片；
3. 功率统一除以各分支主峰并转换为 dB，色轴固定为 -40～0 dB；
4. 主峰使用红色标记，第二峰使用橙色标记；
5. 标题显示模式、捕获成功状态、估计 Doppler、码相位和峰值比；
6. 码相位横轴使用 chip，Doppler 纵轴使用 Hz。

## 边界与验证

- 搜索矩阵、Doppler 栅格或采样信息缺失时给出明确错误；
- 非有限或零功率通过 `eps` 保护，避免 `log10(0)`；
- 测试在 figure 不可见模式下检查图窗、二维图像、曲线和峰值字段；
- 不要求额外 MATLAB 工具箱。
