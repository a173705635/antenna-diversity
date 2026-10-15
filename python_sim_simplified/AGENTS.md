# 精简版协作规则

## 命名

- 自有布尔变量、参数、配置和结果字段统一以 `_flag` 结尾，布尔常量以 `_FLAG` 结尾（如 `show_plots_flag`、`SHOW_PLOTS_FLAG`）；同步修改调用方，第三方 API 的既定名称保持不变。

## 维护范围

- `archive/` 只在本地保存，不提交、不推送，不使用 `git add -f` 绕过忽略规则。
- 当前 `tests/`、`test/` 目录、`test_*.py`、`*_test.py` 和 `conftest.py` 仅在本地保留，不提交、不推送；取消跟踪时保留工作区文件。历史报告中的冻结测试副本保持原样。
- 日常开发维护 `core/` 与 `experiments/`；不继续维护冻结版源码。
- 本地具备测试文件时，默认 `python -m pytest -q` 必须不依赖本地 archive 或 results 数据；新克隆的仓库不包含当前测试文件。
- 依赖本地冻结源码或研究数据的测试标记为 `local_evidence`，在本机显式执行 `python -m pytest -m local_evidence -q`。
- 当前只维护抗干扰与通用接收/定位代码，抗干扰入口在 `experiments/interference/`；不恢复已删除的抗欺骗、多径接口来兼容历史脚本。
- 2026-10-14 清理前版本在本地 `archive/pre_interference_only_2026-10-14_156b994/`；历史报告、结果和来源哈希保持原样，复现使用对应冻结源码。
