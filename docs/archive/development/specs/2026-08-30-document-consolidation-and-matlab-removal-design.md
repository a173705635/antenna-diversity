# MATLAB 清理与项目文档整合设计

日期：2026-08-30

## 目标

将仓库收敛为以 Python 为唯一开发基线的 GNSS 天线分集与空域抗干扰工程，删除不再维护的 MATLAB 实现，减少 Markdown 重复，同时保留理解当前代码、继续设计和从 Git 历史恢复旧实现所需的信息。

## 清理范围

### 删除 MATLAB 实现

删除整个 `matlab_sim/`。当前 Python 链已经覆盖 GPS L1 的天线、RF/ADC、DFE、SINGLE/EGC/MVDR/Blind Nulling、捕获、跟踪与 C/N0 主线，因此删除 MATLAB 版本不会改变当前仿真能力。

MATLAB 中独有的 Galileo E5/AltBOC 代码不会迁入 Python；如以后重新研究，可从删除前的 Git 提交 `6f5e686` 恢复，再根据届时的 Python 架构重新实现。

### 删除 MATLAB 专属历史设计

删除以下成对的设计与实施计划：

- `2026-08-04-matlab-rf-dfe-trk-chain-design.md` 及对应 plan；
- `2026-08-08-galileo-e5-fullband-joint-tracking-design.md` 及对应 plan；
- `2026-08-09-acquisition-search-visualization-design.md` 及对应 plan。

捕获可视化的当前行为已经由 Python README、DESIGN 和代码测试说明，不再依赖早期 MATLAB 计划。

## 当前文档职责

新增根目录 `README.md`，只作为统一入口，不重复展开技术细节：

- `PROJECT_STATUS.md`：当前进度、已证明结论、风险和下一阶段；
- `python_sim/README.md`：安装、运行、配置和实验入口；
- `python_sim/DESIGN.md`：公式、接口和代码映射；
- `python_sim/ARCHITECTURE.md`：对外架构、数据流、已建模与未建模边界；
- `基础知识/README.md`：理论知识库入口；
- `docs/superpowers/`：仍保留的历史设计和实施记录，不作为当前状态依据。

## 可合并内容

将 `python_sim/docs/diagram-style.md` 合并为 `python_sim/ARCHITECTURE.md` 的“制图与维护约定”附录，然后删除独立文件，并更新 `DESIGN.md` 中的链接。README、DESIGN 和 ARCHITECTURE 只去除明显重复或改为互相引用，不合成巨型文档。

`CODEX_MEMORY_GNSS_ANTENNA_DIVERSITY.md` 属于历史项目记忆，不作为当前状态依据。当前仍有效的阶段结论已经存在于 `PROJECT_STATUS.md`；本次不删除该未跟踪文件，也不把它加入当前文档入口，避免未经明确授权处理用户保存的历史材料。

## 一致性更新

- 从 `PROJECT_STATUS.md` 删除“MATLAB 代码仍作为历史参考”的当前仓库表述，改为“旧 MATLAB/E5 实现可从 Git 历史恢复”；
- 删除所有指向已移除 MATLAB 文件或 `diagram-style.md` 的有效文档链接；
- 根目录 README 明确 Python 是唯一主线、知识库和历史设计的边界；
- 不修改 Python 算法、配置、测试和生成结果。

## 验证

完成后检查：

1. `matlab_sim/` 及六份 MATLAB 专属历史文档均不存在；
2. 根目录 README 的本地链接有效；
3. 当前文档中不存在失效的 `matlab_sim` 或 `diagram-style.md` 链接；
4. Python 测试仍通过，证明纯文档/目录清理未影响主工程；
5. Git 差异只包含已确认的删除和文档整合，不包含未跟踪的 Word、IDE 配置、requirements 或项目记忆。
