# 005 — docs/simai-syntax.md 初稿（v0.1）

- **日期**：2026-09-10
- **变更**：新增 `docs/simai-syntax.md` v0.1 初稿。
- **理由**：第 2 步调研（simai 语法规范）的成果文档。先以两份已到手的来源交叉撰写：用户提供的 MMFC PDF 第四章（社区教学）+ 解析器源码分析报告（源码级规则），让成果尽早落盘；官方 wiki 与中文社区两份调研返回后再交叉核实升版。
- **来源**：agent 自主判断（调研推进中，先产初稿再合并）。
- **关联**：`docs/simai-syntax.md`、`docs/research/parser-source-analysis.md`、笔记 002/004。

## 初稿要点

- 相对时间体系：BPM + 分音 + 逗号；最细分音 384（分音须为其约数）。
- note 全类型：Tap / Each（`/`、`` ` ``、`0/`）/ Hold（三种时长格式）/ Touch 与 C 区 Touch-Hold / Slide 全 12 形状（含端点几何约束表）/ 修饰符 b·x·f。
- **零报错安全子集**：以 SimaiSharp 为最严格基准、对齐其 Serializer 输出格式；11 条生成器禁用清单（`[##秒]`、C2、slide+x、`@`/`m`/`$`、无冒号秒数、星星时长混合、同头用 `/`、`{#秒}`、单 `|`、连续 BPM/分音标记、缺 `E`）。
