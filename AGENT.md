# AGENT.md — AI 舞萌制谱工作流（aimai-chartmaker）

> 本文件面向所有参与本项目开发的 coding agent。开始任何工作前请先完整阅读本文件。

## 项目概述

本项目旨在搭建一套**软工作流 + 制谱经验知识体系**——**不训练、不开发独立的谱面生成模型**。核心形态：**强 AI 模型（coding agent）**在拿到一首 mp3 后，依据本项目的分析工具与制谱经验，**直接上手创作**符合要求的谱面。

项目三大组成：

1. **分析工具**（`docs/audio-analysis.md`）：把 mp3 变成结构化特征（节拍/分轨 onset 踩音目标/结构分段/强度曲线），让模型"看懂"乐曲；
2. **制谱经验**（`.agent/knowledge/`）：从官方谱与教程沉淀的制谱意识与规范，模型创作时的依据（**官方谱经验挖掘尚未开始，待用户安排**）；
3. **工作流指引**（待建）：拿到 mp3 后的完整操作 SOP，串联分析与创作。

- **输入**：一个 MP3 格式的 track 音频文件（BPM/offset/变速点等元数据可由用户直接提供）
- **输出**：一份 `maidata.txt`，内容为 **simai 语法** 的谱面数据（由模型在工作流内创作，非模型产物）

最终交付的谱面必须满足以下全部要求（按优先级排序，全部为硬性验收标准）：

1. **语法正确**：完全符合 simai 语法规范，通过校验器后 **零报错、零警告提示**；
2. **人类可玩**：谱面可以被真人实际游玩；
3. **手序合理**：符合人类手序（左右手交替、可及范围等）设计；
4. **符合制谱规范**：包括但不限于——
   - 难度分布与官方谱面近似；
   - 情感循序渐进（谱面密度/复杂度随音乐情绪演进）；
   - 节奏对拍准确（note 与音频 onsets 对齐）；
   - 段落结构、休息段、高潮段安排合理。

## 当前阶段：前期准备（Phase 0）

### 已完成

- [x] 建立 GitHub 公开仓库 `aimai-chartmaker` 并推送初始提交

### 待办（按用户安排推进）

> 注意：以下各项仅在用户明确要求时启动，agent 不得擅自开工。

- [ ] **官方谱制谱经验挖掘**（用户已指出"这块还没弄"，**尚未安排**）：从 388 个官方 ST 谱中提炼配置/手型/段落写法经验，沉淀为知识条目

- [x] 调研 simai 语法规范，产出文档 `docs/simai-syntax.md`（v1.0 定稿，2026-09-10）：
  - 三路调研合流：官方 simai wiki（记法定义者 Celeca 规范）+ 社区解析器源码 + 中文社区教程；
  - 原始调研报告存档于 `docs/research/`（official-spec-research.md / parser-source-analysis.md / chinese-community-research.md）；
  - 结论要点：note 全类型语法、slide 12 形状与端点约束、启动拍机制（60/BPM 固定一拍）、meta 转义规则、零报错安全子集与禁用清单（§6）。
- [x] 调研 simai 报错/警告的判定逻辑，产出文档 `docs/simai-error-checking.md`（v1.0 定稿，2026-09-10）：
  - 调研原则（用户决策）：报错判定逻辑以**成熟解析器的实际实现行为**为准（majdata、visual maimai、miacode 等）；
  - 已收录：MajdataEdit SyntaxCheck（15 组判定，源码行号级）、MajdataView 报错消息、SimaiSharp 5 类异常、MaiLib/maidata-rs、**MiaCode 34 类规则+双模式架构+Muri 检测**（源码级）、Visual Maimai（闭源，文档侧取证，置信度低）；
  - 校验器实现蓝本见该文档 §10（四层校验 + 双检 + MiaCode slide_data.json 白名单与 SimaiParserSpec 回归用例借鉴）；
  - 原始报告：`docs/research/parser-source-analysis.md`、`docs/research/miacode-vm-research.md`。

## 技术路线（2026-09-10 定稿，详见 `docs/audio-analysis.md`）

**路线决策**：特征分析先行、生成器后置——不采用"全频谱直接进 Transformer"的端到端路线。四层管线：

1. **L1 节拍层**（beat_this，MIT）：BPM/拍点/下拍 → 谱面时间轴；用户提供的 BPM/offset 为最高优先级覆盖
2. **L2 音轨层**（Demucs htdemucs_ft + basic-pitch，MIT/Apache）：鼓/人声/贝斯/其他 stems 分离 → 各 stem onset 流 = 踩音候选池
3. **L3 结构层**（all-in-one + FMP/libfmp SSM，MIT/ISC）：段落分段（含 chorus 标签）+ 重复段验证副歌 + librosa 强度曲线/高潮定位
4. **L4 规划层**：结构 × 强度 × 音轨 → 逐段踩音计划（charting_plan.json），规则来自 `.agent/knowledge/` 001-016；生成端只消费结构化特征（LLM 不接触原始音频）
5. **语法与可玩性校验**（`docs/simai-error-checking.md` §10）：四层校验 + SimaiSharp/MajdataEdit/MiaCode 三检，不通过自动迭代修复
6. **输出**：合法 `maidata.txt`（零报错安全子集，`docs/simai-syntax.md` §6）

工具选型红线：默认依赖仅 MIT/Apache-2.0/ISC/BSD；AGPL/GPL/CC BY-NC 权重不进默认管线（详见 `docs/audio-analysis.md` §7）。

## 工作准则（coding agent 必须遵守）

1. **校验优先**：任何生成谱面的代码路径，必须带校验环节；未通过校验的输出不得视为完成。
2. **文档同步**：Phase 0 的调研结论必须先落到 `docs/` 下的 markdown 文档，再进入实现。
3. **语言约定**：文档与代码注释使用中文；提交信息（commit message）可使用中文或英文，需简洁描述变更。
4. **仓库结构约定**：
   - `docs/` — 调研文档与设计文档
   - `.agent/knowledge/` — 制谱意识知识库（谱面设计的稳定认知，跨上下文记忆，见该目录 README.md）
   - `.agent/notes/` — 项目变更记录（变更内容 + 理由，见该目录 README.md）
   - `resource/` — 用户提供的参考资料（本机留存，默认不入库，见该目录 README.md）
   - `AGENT.md` — 本文件（项目对 agent 的说明）
   - 后续实现代码、测试、样例谱面的目录结构在 Phase 0 结束时统一规划
5. **版权与素材**：仓库中不得提交有版权的音频文件；测试用音频使用自创或无版权素材。
6. **社区合规**：调研 simai 语法与校验器时，参考社区公开资料并注明来源；若参考了特定开源项目，需遵守其许可证要求。
7. **变更记录**：每次对项目做实质变更（增删改文件、方向性决策）后，必须在 `.agent/notes/` 下新增一条编号笔记，说明**变更内容与理由**，并与变更在同一提交中推送；格式约定见 `.agent/notes/README.md`。
8. **制谱意识知识库（跨上下文记忆）**：模型上下文长度有限，**绝不允许依赖对话上下文来维持对制谱意识（谱面设计认知）的理解**——跨上下文的理解不合理、不可靠且会产生幻觉。规则：
   - 所有关于谱面设计的稳定认知（无论大小）必须沉淀到 `.agent/knowledge/` 下，作为长期维护的知识条目；
   - **生成谱面、评审谱面、设计生成器规则之前，必须先读取 `.agent/knowledge/` 中相关条目**，不得凭对话中的"印象"工作；
   - 从资料/调研/实验中获得的任何新制谱意识小点，**立即**新增条目并标注置信度与来源；
   - **知识不是金科玉律**：历史条目只是"当前最佳认知"，长期维护中会被新证据推翻——读取时保持批判（`存疑` 条目不得当作硬约束），与新证据冲突时以证据为准，修订或废弃旧条目并记录依据；
   - `.agent/knowledge/` 存稳定认知，`.agent/notes/` 存变更过程，二者勿混淆；格式约定见 `.agent/knowledge/README.md`。

## 参考资源（持续补充）

- 官方 simai wiki（记法定义者 Celeca 的规范）：https://w.atwiki.jp/simai/pages/1003.html （Notations of simai）、pages/1002.html（日文）、pages/25.html（旧版全量手册）、pages/510.html（meta 变量）。⚠️ 该站有 Cloudflare 防护，调研时经 Wayback Machine 快照获取。
- SimaiSharp（astrodx 官方解析器，本项目零报错校验基准）：https://github.com/reflektone-games/SimaiSharp
- MajdataView / MajdataEdit：https://github.com/LingFeng-bbben/MajdataView ；wiki《怎样写谱？》
- MaiLib：https://github.com/Neskol/MaiLib ；maidata-rs：https://github.com/xen0n/maidata-rs
- 中文社区：MMFC《谱面创作基础学》（用户提供 PDF，`resource/` 本机）；无理综述系列（B站）；详见 `docs/research/chinese-community-research.md` 来源清单
- 本仓库三份调研报告：`docs/research/` 下 official-spec-research.md / parser-source-analysis.md / chinese-community-research.md
