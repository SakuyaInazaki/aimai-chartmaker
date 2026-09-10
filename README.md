# aimai-chartmaker

AI 舞萌（maimai DX）**制谱软工作流 + 制谱经验知识体系**。

> **不训练、不开发独立的谱面生成模型。** 核心形态：**强 AI 模型（coding agent）**拿到一首 mp3 后，
> 依据本项目的**分析工具**与**制谱经验**，直接上手创作符合要求的谱面。

- **输入**：一个 MP3 track 音频文件（BPM / offset / 变速点等元数据可由用户直接提供）
- **输出**：一份 `maidata.txt`，内容为 **simai 语法**的谱面数据（由模型在工作流内创作）

## 项目目标

最终交付的谱面必须满足以下全部要求（硬性验收标准，按优先级排序）：

1. **语法正确**：完全符合 simai 语法规范，通过校验器后 **零报错、零警告提示**；
2. **人类可玩**：谱面可以被真人实际游玩；
3. **手序合理**：符合人类手序（左右手交替、可及范围等）设计；
4. **符合制谱规范**：难度分布与官方谱近似、情感循序渐进（密度/复杂度随情绪演进）、
   节奏对拍准确（note 与音频 onsets 对齐）、段落结构与休息段/高潮段安排合理。

## 当前状态

🛠️ **知识与工具建设期**（阶段划分以 [AGENT.md](AGENT.md) 为准，最后更新 2026-09-10）

**已完成**

- [x] 建立 GitHub 公开仓库
- [x] 调研 simai 语法规范 → [`docs/simai-syntax.md`](docs/simai-syntax.md)（v1.0 定稿）
- [x] 调研 simai 报错/警告判定逻辑（以成熟解析器实际实现为准）→ [`docs/simai-error-checking.md`](docs/simai-error-checking.md)（v1.0 定稿）
- [x] ST 谱面要素年表（只做 FiNALE 及以前的旧框要素）→ [`docs/st-chart-elements.md`](docs/st-chart-elements.md)
- [x] 音频分析管线技术路线定稿（四层工具选型，许可证已核实）→ [`docs/audio-analysis.md`](docs/audio-analysis.md)
- [x] 官方谱配置分布扫描：本地 388 个官方 ST 谱 → [`docs/research/config-usage-survey.md`](docs/research/config-usage-survey.md)（agent 观察，待用户确认）
- [x] 官方谱逐谱精读：本地 14 级 **87 个谱面文件全部读完**（配置构成 / 强度难度分布 / 可复用手法 / 存疑）→ [`docs/research/level14-readings.md`](docs/research/level14-readings.md)
- [x] 制谱认知知识库 **30 条**原子条目（用户逐条讲授 + 官方谱实证验证）→ [`.agent/knowledge/`](.agent/knowledge/)
- [x] 过程记录与变更日志 **33 篇**（编号 001–033）→ [`.agent/notes/`](.agent/notes/)

**进行中 / 下一步**

- [ ] 跨谱通用装置整理进 `.agent/knowledge/`（**星星（slide）相关需等用户讲授判断标准后再定性**）
- [ ] 逐谱精读向 13 级及以下扩展（待用户安排）
- [ ] 分析工具落地实现（节拍 / 分轨 onset / 结构分段 / 强度曲线）
- [ ] 谱面生成 → 校验链路（四层校验蓝本见 `docs/simai-error-checking.md` §10）
- [ ] 拿到 mp3 后的完整操作 SOP（工作流指引）

## 目录导览

| 路径 | 内容 |
|---|---|
| [`AGENT.md`](AGENT.md) | 项目概述、当前阶段、工作准则（coding agent 必读） |
| [`docs/`](docs/) | 定稿文档：simai 语法 / 报错判定 / ST 谱面要素 / 音频分析 |
| [`docs/research/`](docs/research/) | 调研报告 + 官方谱扫描 + 14 级逐谱精读记录 |
| [`.agent/knowledge/`](.agent/knowledge/) | 制谱认知知识库（原子条目，含日期/理由/来源/置信度/修订记录） |
| [`.agent/notes/`](.agent/notes/) | 过程记录与变更日志（编号索引见其 README） |
| `resource/` | 用户提供的参考资料与**本机专用**数据（官方谱面、预览镜像），**不入库** |

## 说明

- **`resource/` 下的官方谱面数据与本地预览环境不入 git 仓库**：版权归 SEGA / 原曲版权方，
  仅作本机研究参考（忽略规则见 [`.gitignore`](.gitignore) 与 `resource/README.md`）。
- 本项目只生成 **ST（通常）谱面**，故只采用 FiNALE（2018-12-13）及以前的旧框要素。
- 知识库内容不是金科玉律，允许被新证据推翻（见 `.agent/knowledge/README.md` 的约定）。

> 更多面向开发者的说明见 [AGENT.md](AGENT.md)。

## 第三方材料与致谢

- 本仓库**不再分发**任何第三方代码、官方谱面数据或歌曲音频：
  - 官方谱面数据（抓取自 mai-notes.com）与本地预览环境仅存于本机 `resource/`，已被 `.gitignore` 排除，版权归 **SEGA / 原曲版权方**；
  - 用户提供的参考资料（如 MMFC《MAIMAI 谱面创作基础学 长篇指南》PDF）同样不入库，仅在本机阅读与消化。
- `docs/` 与 `docs/research/` 中的内容是对**公开资料**（simai 官方 wiki、社区解析器源码、中文社区教程）与**用户提供资料**的**归纳与引用**，来源链接均随文标注；
  未复制第三方源码或许可证不明的大段原文。若需引用本项目对上述资料的整理结论，请自行核对原始来源。
- 本项目与 **SEGA**、maimai（舞萌 DX）官方无任何关联；「maimai」「舞萌」等名称与相关素材的权利归其各自所有者。

## 许可证

[MIT](LICENSE) © 2026 SakuyaInazaki

- 覆盖本仓库全部入库内容：代码、`docs/`、`docs/research/`、`.agent/knowledge/`、`.agent/notes/`。
- 依赖工具与模型的许可证另有约束：默认管线只采用 **MIT / Apache-2.0 / ISC / BSD** 类，
  AGPL/GPL 与非商用（CC BY-NC）权重不进默认管线（见 `AGENT.md` 技术路线、`docs/audio-analysis.md` §7）。
