# 归档与测试

- `archive/` 只在本地保存，不提交、不推送，不使用 `git add -f` 绕过忽略规则。
- 日常开发维护 `core/` 与 `experiments/`；不继续维护冻结版源码。
- 默认 `python -m pytest -q` 必须不依赖本地 archive 或 results 数据。
- 依赖本地冻结源码或研究数据的测试标记为 `local_evidence`，在本机显式执行 `python -m pytest -m local_evidence -q`。
