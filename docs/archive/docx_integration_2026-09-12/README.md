# Word 整合历史记录

本目录保存 2026-09-12 整理前的构建脚本、输入文本、核验记录、原 DOCX 备份和渲染产物。脚本为历史快照，部分依赖已删除的根目录来源文件和旧路径，不保证直接重跑；恢复构建前需还原输入并调整路径。

- `root_docx_before_full_merge.zip`：七份原文件，可恢复。
- `full_merge_manifest.json`、`full_merge_verification.json`：原文件哈希与全文核验。
- `report_content.md`、`build_report.py`、`compose_full.py`：历史内容和生成过程。
- `render/`、`full_render/`、PDF：版面检查产物。
- `vendor/`：当时安装的合并依赖，不是项目运行依赖。

正式交付文件已归入 `docs/reports/`；原理文档在 `docs/theory/`。归档内来源路径按当时状态保留，未改写历史证据。
