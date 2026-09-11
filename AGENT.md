# AGENT.md — AI 舞萌制谱工作流（aimai-chartmaker）

> 本文件面向所有参与本项目开发的 coding agent。开始任何工作前请先完整阅读本文件。

## 项目概述

本项目旨在搭建一套**软工作流 + 制谱经验知识体系**——**不训练、不开发独立的谱面生成模型**。核心形态：**强 AI 模型（coding agent）**在拿到一首 mp3 后，依据本项目的分析工具与制谱经验，**直接上手创作**符合要求的谱面。

项目三大组成：

1. **分析工具**（`docs/audio-analysis.md`）：把 mp3 变成结构化特征（节拍/分轨 onset 踩音目标/结构分段/强度曲线），让模型"看懂"乐曲；
2. **制谱经验**（`.agent/knowledge/`）：从官方谱与教程沉淀的制谱意识与规范，模型创作时的依据（**已建库 30 条**：用户逐条讲授 + 官方谱实证验证；官方谱扫描与 14 级逐谱精读已完成，见下）；
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
- [x] **官方谱制谱经验挖掘（第一批，2026-09-10）**：
  - **配置分布扫描**：脚本扫描本地 388 个官方 ST 谱，用已讲授的 14 类配置当透镜量化分布 → `docs/research/config-usage-survey.md`（**agent 观察，未经用户确认**）；
  - **逐谱精读**：按用户要求"必须逐个谱去看"，把本地 14 级（14.0–14.5）**87 个谱面文件全部读完**，逐谱记录「配置构成 / 强度难度分布 / 可复用手法 / 存疑」→ `docs/research/level14-readings.md`（含各组小结、总总结、跨谱候选装置 46+ 条、账目对账）；
  - **知识库建库**：`.agent/knowledge/` 现有 **30 条**原子条目（用户逐条讲授为来源，官方谱为验证）；
  - **过程记录**：`.agent/notes/` 现有 **33 篇**（编号 001–033）。
- [x] **音频分析攻坚（2026-09-11）**：在"BPM / offset / 变速点由用户直接提供"的前提下，攻克"AI 通过分析 mp3 了解哪里是副歌、哪里是高潮、采鼓点还是人声"（对应 MMFC 第五章强度/踩音/配置理论）：
  - **调研第二轮** → `docs/research/audio-analysis-research-v2.md`（onset→分音网格量化规则、offset 反向校验、J-pop 结构分段证据与日式标签词表、无官方音频的标定协议、切轨可计算代理、song sheet 表示、M4/py3.12 可行性、评测数据；24 条修订建议）；
  - **官方谱逐小节密度基准** → `tools/chart_analysis/`（simai 时间轴解析器，387/388 零错，manifest 对账 97.2%）+ `docs/research/official-chart-density-curves.md`（388 谱密度曲线：五段模板实测、密度地板、末段最强、休息段规律；结论入库为知识 **031**，知识 001 相应修订）；
  - **设计定稿 v1.1** → `docs/audio-analysis.md`（合并上述全部证据与原型实测；当前实现状态见其 §11）；
  - **原型** → `tools/audio_analysis/`（mp3 + BPM/first → 统一解码 → offset 校验 → Demucs 分轨 → 逐 stem onset 量化网格串 → 结构分段 → 强度曲线 → song sheet 双格式 + 复核图；3 首测试曲实跑，合成音频单测全绿）；
  - **官方音频配对标定**（用户提供 8 首官方 ST 谱 zip → 本机 `official-mp3/`，不入库）→ `tools/calibration/` + `docs/research/audio-chart-calibration.md`：强度×密度逐小节 ρ 0.454 / 逐段 0.523，权重留一曲 CV 未显著优于初值故不换；各 stem 命中率（鼓骨架 recall 0.607，只落人声 3.2%）与切轨规则表一致率；候选池 recall 0.82 / precision 0.55；388 谱结尾形态尾杀 65% / 渐弱 1% / 其他 34%（知识 031 §10）；
  - 过程记录：notes 034–041。
- [x] 调研 simai 语法规范，产出文档 `docs/simai-syntax.md`（v1.0 定稿，2026-09-10）：
  - 三路调研合流：官方 simai wiki（记法定义者 Celeca 规范）+ 社区解析器源码 + 中文社区教程；
  - 原始调研报告存档于 `docs/research/`（official-spec-research.md / parser-source-analysis.md / chinese-community-research.md）；
  - 结论要点：note 全类型语法、slide 12 形状与端点约束、启动拍机制（60/BPM 固定一拍）、meta 转义规则、零报错安全子集与禁用清单（§6）。
- [x] 调研 simai 报错/警告的判定逻辑，产出文档 `docs/simai-error-checking.md`（v1.0 定稿，2026-09-10）：
  - 调研原则（用户决策）：报错判定逻辑以**成熟解析器的实际实现行为**为准（majdata、visual maimai、miacode 等）；
  - 已收录：MajdataEdit SyntaxCheck（15 组判定，源码行号级）、MajdataView 报错消息、SimaiSharp 5 类异常、MaiLib/maidata-rs、**MiaCode 34 类规则+双模式架构+Muri 检测**（源码级）、Visual Maimai（闭源，文档侧取证，置信度低）；
  - 校验器实现蓝本见该文档 §10（四层校验 + 双检 + MiaCode slide_data.json 白名单与 SimaiParserSpec 回归用例借鉴）；
  - 原始报告：`docs/research/parser-source-analysis.md`、`docs/research/miacode-vm-research.md`。

### 待办（按用户安排推进）

> 注意：以下各项仅在用户明确要求时启动，agent 不得擅自开工。

- [ ] **跨谱通用装置整理进 `.agent/knowledge/`**（用户已决定：14 级谱面读完后再一起写）
  - ⚠️ 前置条件：**星星（slide）相关的归纳须等用户讲授「如何判断星星的配置」后**才能评估；
    在此之前，`level14-readings.md` 中所有星星类观察（自环星、星链、星-单交替、双星齐奏、往返星等）
    **只算原文记录，不得作为结论写入知识库**。
- [ ] **逐谱精读向 13 级及以下扩展**（待用户安排；14 级已读完）
- [ ] **分析工具迭代与标定**（原型已可用；待做：`basic-pitch` 装包、结构标签与人声侧归因的人工听审复核、RWC-Pop / osu2beat2025 评测协议 A/B 档、更多官方音频下的权重标定；清单见 `docs/audio-analysis.md` §9/§10）
- [ ] **待用户拍板**：① CC BY 4.0 是否纳入依赖白名单（决定 SongFormer 去留）；② 知识 002「副歌全踩人声」在 8 首官方曲的代理实测下降为条件性并标存疑（见知识 032），需用户听审裁定；③ 若能再提供更多官方音频（尤其人声主导曲），可把 n=8 的标定与切轨验证做实
- [ ] **工作流指引**：拿到 mp3 后的完整操作 SOP（串联分析与创作）

## 技术路线（v1.1，2026-09-11；详见 `docs/audio-analysis.md`）

**路线决策**（2026-09-10，用户）：特征分析先行、生成器后置——不采用"全频谱直接进 Transformer"的端到端路线。四层管线（v1.1）：

1. **L1 节拍层**：用户提供的 BPM/offset/变速点为最高优先级；管线只做 **offset 反向校验**（网格相位搜索 + BPM 漂移回归，只报告不覆盖）；全程 ffmpeg 统一转 WAV；beat_this/librosa 仅作缺省与交叉验证
2. **L2 音轨层**：Demucs htdemucs（MIT）四 stems → 逐 stem onset → **拍网格量化**（每小节单一分音、容差 `τ=clamp(0.25·slot,12,30)ms`、三连判别、**`{32}` 红线**）+ 鼓件三频带启发式 + 人声 VAD → 逐小节特征清单 = 踩音候选池
3. **L3 结构层**：all-in-one-infer（MIT）+ libfmp/SSM 重复段三路投票分段、**日式(英文)双标签**、pre_chorus 规则派生；强度 = 响度 / 去重 onset 数 / 鼓能量 / 人声活动 / 谱通量融合，五票定高潮；**强度→密度映射用 388 官方谱实测**（`T_density` + 密度地板，知识 031）
4. **L4 规划层**：结构 × 强度 × 音轨 → **song sheet 双格式**（`song_sheet.md` 给 LLM、`song_analysis.json` 给程序；逐小节 4 轨网格串是候选池不是谱面）+「段落类型 × 特征 → 主踩音轨」规则表（规则来自 `.agent/knowledge/` 001–005/031）；生成端只消费结构化特征（LLM 不接触原始音频）
5. **语法与可玩性校验**（`docs/simai-error-checking.md` §10）：四层校验 + SimaiSharp/MajdataEdit/MiaCode 三检，不通过自动迭代修复
6. **输出**：合法 `maidata.txt`（零报错安全子集，`docs/simai-syntax.md` §6）

工具选型红线：默认依赖仅 MIT/Apache-2.0/ISC/BSD；AGPL/GPL/CC BY-NC 权重不进默认管线（已排除 ADTOF / LarsNet / omnizart / pop-music-highlighter；SongFormer 为 CC BY 4.0 但其 MuQ 权重 NC，仅作非商用第二意见；详见 `docs/audio-analysis.md` §7）。

## 工作准则（coding agent 必须遵守）

1. **校验优先**：任何生成谱面的代码路径，必须带校验环节；未通过校验的输出不得视为完成。
2. **文档同步**：Phase 0 的调研结论必须先落到 `docs/` 下的 markdown 文档，再进入实现。
3. **语言约定**：文档与代码注释使用中文；提交信息（commit message）可使用中文或英文，需简洁描述变更。
4. **仓库结构约定**：
   - `docs/` — 调研文档与设计文档
   - `.agent/knowledge/` — 制谱意识知识库（谱面设计的稳定认知，跨上下文记忆，见该目录 README.md）
   - `.agent/notes/` — 项目变更记录（变更内容 + 理由，见该目录 README.md）
   - `resource/` — 用户提供的参考资料（本机留存，默认不入库，见该目录 README.md）
   - `tools/` — 分析工具代码（`tools/audio_analysis/` 音频分析管线原型；`tools/chart_analysis/` simai 解析器与官方谱密度统计），依赖装在仓库根 `.venv/`（python3.12，不入库）
   - `tests/` — 单元测试（只用自写 simai 片段与 numpy 合成音频，不放真实音频/官方谱原文）
   - `out/` — 分析输出（stems/特征/图，含版权音频衍生物，不入库）
   - `official-mp3/` — 用户提供的官方谱 zip（官方 maidata + 版权 mp3），仅本机标定用，不入库
   - `AGENT.md` — 本文件（项目对 agent 的说明）
   - 样例谱面等其余目录在需要时再统一规划
5. **版权与素材**：仓库中不得提交有版权的音频文件；测试用音频使用自创或无版权素材。
6. **社区合规**：调研 simai 语法与校验器时，参考社区公开资料并注明来源；若参考了特定开源项目，需遵守其许可证要求。
7. **变更记录**：每次对项目做实质变更（增删改文件、方向性决策）后，必须在 `.agent/notes/` 下新增一条编号笔记，说明**变更内容与理由**，并与变更在同一提交中推送；格式约定见 `.agent/notes/README.md`。
8. **制谱意识知识库（跨上下文记忆）**：模型上下文长度有限，**绝不允许依赖对话上下文来维持对制谱意识（谱面设计认知）的理解**——跨上下文的理解不合理、不可靠且会产生幻觉。规则：
   - 所有关于谱面设计的稳定认知（无论大小）必须沉淀到 `.agent/knowledge/` 下，作为长期维护的知识条目；
   - **生成谱面、评审谱面、设计生成器规则之前，必须先读取 `.agent/knowledge/` 中相关条目**，不得凭对话中的"印象"工作；
   - 从资料/调研/实验中获得的任何新制谱意识小点，**立即**新增条目并标注置信度与来源；
   - **知识不是金科玉律**：历史条目只是"当前最佳认知"，长期维护中会被新证据推翻——读取时保持批判（`存疑` 条目不得当作硬约束），与新证据冲突时以证据为准，修订或废弃旧条目并记录依据；
   - `.agent/knowledge/` 存稳定认知，`.agent/notes/` 存变更过程，二者勿混淆；格式约定见 `.agent/knowledge/README.md`。
9. **用户讲授制谱经验时——引文即边界**：用户给出的谱面引文（片段）**就是该配置的完整边界**。只允许在引文范围内核对与归类，**不得向引文之外延伸推理**（引文之后的谱面内容属于其他配置，未经讲授不得擅自归类）；发现疑问时先向用户确认，不得据此质疑用户提供的数据。
10. **不擅自开工**：待办/研究方向仅在用户明确要求时启动；宣布"即将开始某项工作"同样视为越界——先问，再动。
11. **谱面参考只看官方谱**（用户 2026-09-11 明确）：谱面设计的参考、统计与评测 ground truth 一律只用 `resource/official-chart/` 的官方谱；用户的自制谱（`~/Desktop/self-charts` 等）**不得**作为谱面参考或评测依据，其 `track.mp3` 仅可作音频分析的测试音频，且 `maidata.txt` 只读头部 `&first` 与首个 `(BPM)`，正文不读不引用不评价。官方谱目前没有对应音频，"音频 × 官方谱密度"的配对标定须等用户提供音频后再做。

## 参考资源（持续补充）

- 官方 simai wiki（记法定义者 Celeca 的规范）：https://w.atwiki.jp/simai/pages/1003.html （Notations of simai）、pages/1002.html（日文）、pages/25.html（旧版全量手册）、pages/510.html（meta 变量）。⚠️ 该站有 Cloudflare 防护，调研时经 Wayback Machine 快照获取。
- SimaiSharp（astrodx 官方解析器，本项目零报错校验基准）：https://github.com/reflektone-games/SimaiSharp
- MajdataView / MajdataEdit：https://github.com/LingFeng-bbben/MajdataView ；wiki《怎样写谱？》
- MaiLib：https://github.com/Neskol/MaiLib ；maidata-rs：https://github.com/xen0n/maidata-rs
- 中文社区：MMFC《谱面创作基础学》（用户提供 PDF，`resource/` 本机）；无理综述系列（B站）；详见 `docs/research/chinese-community-research.md` 来源清单
- 本仓库三份调研报告：`docs/research/` 下 official-spec-research.md / parser-source-analysis.md / chinese-community-research.md
