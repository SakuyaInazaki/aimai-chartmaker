# 音频分析与踩音规划（管线设计）

> **状态**：v1.0（2026-09-10）。四层工具选型已定稿，依据 `docs/research/audio-analysis-research.md`（SOTA 调研 + prior art，许可证均经一手核实）。
> **目标**：攻克"让 AI 通过分析 mp3 了解哪里是副歌、哪里是高潮、要采鼓点还是人声"——对应 MMFC 教程第五章的强度/踩音/配置理论（知识库 001/002/003）。

## 0. 设计原则（与项目约定一致）

1. **LLM 不"听"原始音频**：模型上下文有限且不可靠（见 AGENT.md 准则 8），谱面决策一律基于**结构化特征**（分段标注、onset 流、能量曲线），而非把音频或长特征喂给 LLM——特征提取用确定性 DSP/深度学习模型完成，LLM 只做规划与决策。
2. **用户提供的数据优先**：BPM、offset、变速点可由用户直接提供（"得益于 ai 的发展这些数据我可以直接告诉你"），管线把这些作为**最高优先级输入**覆盖自动检测结果；自动检测作为缺省路径与交叉验证。
3. **可校验**：每个环节的输出都要能被人复核（分段/强度曲线可视化到本地预览），踩音计划必须先通过知识库 001-003 的规则检查。

## 1. 管线总览（四层）

```
mp3 ──► L1 节拍层 ──► L2 音轨层 ──► L3 结构层 ──► L4 规划层 ──► charting_plan.json ──► 生成器
        BPM/beats/     stems 分离+     段落/副歌/       强度×音轨→
        downbeats      onset 流        强度曲线         逐段踩音计划
```

| 层 | 输入 | 输出 | 对应谱师动作 |
|----|------|------|-------------|
| L1 节拍层 | mp3（+用户提供的 BPM/offset 覆盖） | BPM 曲线、拍点、下拍（小节边界）、offset | "确认 BPM、&offset、有没有变速段" |
| L2 音轨层 | mp3 | 鼓/人声/贝斯/其他 4 个 stems + 各自 onset 流 | "要采什么音：鼓点还是人声"的**候选池** |
| L3 结构层 | mp3 + L1 | 段落边界与类型（intro/verse/pre-chorus/chorus/bridge/outro）、重复段（副歌）、逐小节强度曲线 | "哪里是副歌、哪里是高潮" |
| L4 规划层 | L1+L2+L3 + 知识库 001/002/003 | charting_plan（逐段：采音目标、密度、休息安排） | "按情绪写配置"的蓝图 |

## 2. L1 节拍层：时间轴对齐

- **目标**：BPM（含变速点序列）、每个小节（下拍）的精确时间——谱面逗号槽必须落在拍上。
- **用户覆盖路径**：用户提供 BPM + offset（首拍毫秒）+ 变速点 → 直接构造时间轴；`&first = offset/1000` 秒。
- **自动路径**：**主线 = beat_this（MIT ✅，`pip install beat-this`）**——beat+downbeat 一次输出（downbeat=小节起点）、帧级 logits 置信度、无 DBN、对变速宽容（Ballroom beat F1 97.5/downbeat 95.3）；BPM 序列需自研"拍间隔聚类分段"（几行代码）。**打底 = librosa**（ISC ✅，先跑通管线）；备选 = all-in-one（MIT ✅，见 L3，beat/downbeat 同源输出）。
  - **排除**：madmom（精度好但**模型权重 CC BY-NC-SA 非商用** + numpy2 编译坑）、essentia/tempo-cnn（**AGPL**，开源项目避开）。
  - **BPM 上下限显式设置**：目标 160-220，默认 max_bpm 会在 220 档翻车折叠成 110。
  - **MP3 先转 WAV**（ffmpeg）：all-in-one 官方警告 MP3 解码偏移 20-40ms。
  - 低置信度拍点过滤 + 局部峰值细化；变速检测输出"逐小节 BPM 序列"。
- **质量要求**：音游曲多为电子乐 BPM 130–230，节拍必须逐小节稳定，漂移 ≤ 数 ms；变速检测（如 HECATONCHEIR 155→180）需输出分段 BPM。
- **校验**：把检测出的拍点对齐到 AU 式频谱（火焰条）人工复核；与用户提供的 BPM 不一致时以用户为准并记录差异。

## 3. L2 音轨层：踩音候选池

- **声源分离**：**主线 = Demucs v4 htdemucs_ft（MIT ✅ 代码+权重）**——drums/bass/vocals/other 四路，鼓 SDR 10.08 泄漏少；CPU 弱时迭代期用 htdemucs（快 4 倍），量产期 ft/GPU。**排除**：Spleeter（质量垫底）、BS-RoFormer（权重许可不明）、AudioSep（1.26GB 过度设计）、Open-Unmix umxl（权重 CC BY-NC-SA 非商用）。
- **onset 提取**：对每个 stem 做 onset 检测（鼓 stem 的 onset 就是"鼓点"；人声 stem 的 onset/音高就是"人声"候选）。
- **人声旋律**：**主线 = basic-pitch（Apache-2.0 ✅）**——直接输出 note 事件（onset/offset/pitch/confidence）+ MIDI，省掉整条后处理，CPU ≈19×RTF；后备 RMVPE（Apache-2.0 ✅）。**排除**：CREPE（慢）、essentia Melodia（AGPL）、pYIN（GPL）。
- **输出**：四组 onset 时间序列 + 每小节各 stem 的 onset 密度/能量统计——这就是知识 002"切轨"的候选池。
- **注意**：分离质量影响踩音准确度；鼓 stem onset 通常最可靠（音游谱以鼓为骨架），人声 onset 需后处理去噪。

## 4. L3 结构层：段落 / 副歌 / 高潮

- **分段**：**主线 = all-in-one（MIT ✅，ISMIR 2023）**——tempo/beat/downbeat + 段落边界 + **10 类功能标签（intro/verse/chorus/bridge/outro 等）**一个模型全出；用 openmirlab/all-in-one-infer（纯 PyTorch 替代，避开原版 NATTEN 自编译坑），输入需先跑 Demucs 分离。
- **副歌验证（重复段信息）**：all-in-one 不输出重复段 → 加支线验证"**副歌=重复最多段**"：FMP/libfmp + librosa 纯传统方案（CQT/HPCP → SSM → novelty 边界 → path_enhance/谱聚类提取重复段对，MIT/ISC ✅、零模型、秒级）→ 与 all-in-one 的 chorus 标签投票确认。**排除**：MERT/music2vec 特征（权重 cc-by-nc-4.0 非商用）、DeepChorus（仓库无 LICENSE）。
- **强度曲线**：纯 librosa 配方（无模型、全宽松许可，详见下）。
- **高潮定位**：高潮 ≈ chorus 段 ∩ 能量峰值；谱面强度峰值（密度最高、配置最强）应落在这里。
- **输出**：段落列表（起止小节、类型、是否重复段、强度档位）+ 全曲强度曲线。

### 强度曲线配方（librosa，无模型依赖）

```
1) librosa.load(sr=22050, mono) + pyloudnorm 归一 -14 LUFS（防"谁响谁高潮"）
2) 同 hop=512 提取：rms；onset_strength（librosa 默认=SuperFlux）；spectral_centroid；
   可选第4特征 = Demucs drums stem 逐帧 RMS（鼓强度）
3) 逐特征 5-95 百分位截断 → min-max 归一
4) 融合：intensity = 0.35*rms + 0.30*onset + 0.20*cent + 0.15*drums（权重待标定）
5) 对齐节拍聚合：用 beat 时间数组，beats[::4]=小节边界，逐小节均值/P75
6) 平滑：小节级中值滤波（窗≈2小节）+ 高斯(σ≈1)
7) 高潮定位=四票组合投票：
   V1 结构票 0.4 = chorus/重复段内=1
   V2 novelty 票 0.3 = SuperFlux 局部峰值 ±1 小节三角窗
   V3 能量票 0.2 = rms_n>0.75 且局部极大
   V4 质心票 0.1 = cent_n>0.7（电子乐 drop 高频/白噪上升）
   climax = argmax(vote)；前 k 个峰 = 各次副歌出现时间
8) 模板匹配（弱→较强→较弱→强→渐弱）：T=[0.20,0.45,0.30,1.00,0.10]（可参数化）
   a) 有结构结果：chorus 硬对齐模板第4段；intro/verse/bridge/outro 按序映射
   b) 无结构：PELT/RDP 变点检测分4段 + 分段线性拟合最小残差
   c) 强度曲线归一 [0,1]，与模板 DTW/soft-DTW 对齐
   d) 输出 {小节: 强度∈[0,1]} + 段标签 + 高潮起止 → 谱面端按强度映射 note 密度
```

四票互补原理：结构票覆盖摇滚 chorus=整段反复；novelty 票覆盖 drop/爆发瞬态；能量票覆盖响度/密度；质心票覆盖电子乐 drop。流派差异建议两套权重预设（电子 drop vs 摇滚 chorus）。

## 5. L4 规划层：逐段踩音计划（核心设计）

> 这一层把 MMFC 第五章的谱师决策规则化，输入全部是 L1-L3 的结构化特征 + 知识库规则。

### 5.1 输入

- 段落列表（每段：小节范围、类型、强度档位、各 stem onset 密度）
- 目标难度（→ 目标 NPS 区间，知识 004）
- 知识库规则：001（强度贴合情绪）、002（切轨）、003（配置强度四因素）、005（采音要简）、012-015（可玩性约束）、016（ST 要素黑白名单）

### 5.2 决策步骤（规则化，可被 LLM 执行也可被程序执行）

1. **结构映射**：按段落类型给强度基线——intro/outro 低、verse 中、pre-chorus（build）爬升、chorus 高；与强度曲线交叉验证（曲线明显不符时以曲线为准并标记人工复核）。
2. **踩音目标选择（切轨）**：按知识 002 的模板逐段选主踩音轨——副歌采人声、主歌踩器乐/鼓、间奏安排低密度休息段、build 末尾切回人声铺垫；某 stem onset 过稀时降级到鼓骨架（"哪个响踩哪个"只用于前奏类段落）。
3. **密度分配**：每段强度档位 → 分音密度（{4}/{8}/{16}/{32}）与每小节 note 数，总量控制在目标难度 NPS 区间（知识 004/015）。
4. **配置强度升级手段**（知识 003）：同踩音下用 note 种类（tap→hold→each→slide）、位移、错位来微调强度，尤其第二遍副歌用配置升级（单星星→双手星星）体现递进。
5. **休息与可玩性检查**（知识 012-015）：Hold 尾留空、叠键/外键/撞尾红线、Touch 不适用（ST 谱）。
6. **输出 charting_plan.json**：逐段 {小节范围, 段落类型, 强度档位, 主踩音轨, 副踩音轨, 密度档位, 建议配置类型, 休息小节}——交给谱面生成器。

### 5.3 输出样例（与知识 002 的 SPICY SWINGY STYLE 模板对应）

```json
{
  "sections": [
    {"bars": [1,8],   "type": "intro",     "intensity": "low",    "primary_stem": "drums",  "density": "{8}"},
    {"bars": [9,24],  "type": "verse",     "intensity": "mid",    "primary_stem": "other",  "density": "{8}",  "secondary_stem": "vocals_sparse"},
    {"bars": [25,28], "type": "pre_chorus","intensity": "rising", "primary_stem": "vocals", "density": "{8}"},
    {"bars": [29,44], "type": "chorus",    "intensity": "high",   "primary_stem": "vocals", "density": "{8}/{16}", "repeat_of": null},
    {"bars": [45,52], "type": "interlude", "intensity": "low",    "primary_stem": "other",  "density": "{8}",  "rest": true},
    {"bars": [53,68], "type": "chorus",    "intensity": "peak",   "primary_stem": "vocals", "density": "{8}/{16}", "repeat_of": [29,44], "upgrade": "both_hand_slides"}
  ]
}
```

## 6. 与项目现有资产的对接

- **知识库**：001/002/003（本管线 L4 的规则来源）、004（难度 NPS 校准）、005（采音要简）、012-015（可玩性约束）、016（ST 要素黑白名单）。
- **数据**：`resource/official-chart/` 388 个官方 ST 谱（定数 13.0–14.5）——可作为"音频结构 → 踩音计划"的**标注对照数据**（用官方谱反推各段的踩音目标与密度，验证 L4 规则），前提是获得对应音频（用户提供或另寻无版权测试曲）。
- **预览**：本地预览库（http://127.0.0.1:8099/list.html）用于结构/强度标注的可视化复核。

## 7. 工具选型总表、许可证策略与 prior art

### 7.1 四层主线（全 MIT/Apache/ISC，适合开源发布）

| 层 | 主线 | 许可 | 备选/打底 | 排除（理由） |
|----|------|------|-----------|-------------|
| L1 节拍 | **beat_this**（beat+downbeat+置信度） | MIT ✅ | librosa；all-in-one | madmom（模型 NC）、essentia/tempo-cnn（AGPL） |
| L2 分离 | **Demucs htdemucs_ft**（4 stems） | MIT ✅ | htdemucs（CPU 快）；MDX23C-DrumSep（鼓件细分，权重许可待复核） | Spleeter、BS-RoFormer（权重不明）、AudioSep、umxl（NC） |
| L2 人声 | **basic-pitch**（note 事件直出） | Apache-2.0 ✅ | RMVPE | CREPE（慢）、Melodia（AGPL）、pYIN（GPL） |
| L3 结构 | **all-in-one(-infer)**（10 类标签） | MIT ✅ | FMP/libfmp SSM（重复段验证） | MERT/music2vec（权重 NC）、DeepChorus（无 LICENSE）、msaf（低维护） |
| L3/L4 强度 | **librosa 配方**（§4 八步） | ISC ✅ | musicnn（可选） | madmom features（NC）、pychorus（弃用） |

### 7.2 许可证策略（开源项目红线）

- 默认依赖只允许 **MIT / Apache-2.0 / ISC / BSD**；**AGPL/GPL 与 CC BY-NC(-SA) 权重一律不进默认管线**（可作为个人非商用实验的 optional 插件）。
- 待法务复核项：MDX23C/Kim_Vocal_2 权重（分发源无 LICENSE）、Spleeter 5stems 权重 MIT 覆盖性。
- 全线 CPU RTF 无官方数字 → 落地前在目标机器对 2 分钟音游曲实测（beat_this、htdemucs_ft CPU、basic-pitch）。

### 7.3 prior art：miaChartGen2（最值得研读的参照）

- **Goldgom/miaChartGen2（gitcode 镜像存活；GitHub 原仓库已 404；README 自称 maiChartGen3）**：检索到的**唯一 maimai 专用"MP3→simai"生成式项目**。管线：EnCodec 24kHz（75Hz 帧、8 codebooks）音频 token + BeatTokenizer（librosa/beat_this 双后端——**与我们的节拍选型不谋而合**）+ simai tokenizer → 对齐 75Hz 帧网格 → **5-Stage 级联 Transformer**（S1 谱面骨架逐帧分类 → S2 Hold 时长自回归 → S3 Slide 路径+合法性校验 → S4 Break 二分类 → S5 EX 二分类）→ 密度/类型偏置采样 → simai 输出。
- **重要限制**：无 LICENSE、README 声明"仅用于学术研究和个人学习"、不含 Stage 模型 checkpoint → **只能借鉴设计思路（5-Stage 拆分、75Hz 网格、属性后处理），不可直接商用其代码**。
- **我们的差异化点（经全库检索确认）**：现有开源项目**没有任何一个做"显式声源分离→分轨踩音"**（miaChartGen2 是 EnCodec 全频谱直接进 Transformer）；我们的结构分段（chorus 标签）+ 强度曲线（显式条件信号）+ 分轨踩音 + 知识库规则校验是四层相对 prior art 的增量。
- 其他参照：DDC（MIT，CNN+LSTM 舞步放置）、ARG/softchart-v15（MIT，7.94M 小模型）、TaikoNation（乐句模式=结构分段）、audio2chart（端到端转录，最接近踩音层）、Mug-Diffusion、ChartGenEval（评测框架）。

## 8. 技术路线决策：显式特征分析 vs 端到端（2026-09-10，用户决策）

- **用户判断**："全频谱直接进 transformer 有点扯；对 mp3 进行分析，寻找采音、分段以及情绪强度才是正确的路径。"
- **本项目决策**：**特征分析先行、生成器后置**——L1-L3 确定性提取（节拍 / 分轨 onset 采音目标 / 结构分段 / 强度曲线），L4 规则+规划（注入知识库硬约束），生成端只消费结构化特征；不采用"EnCodec 全频谱 token 直接进 Transformer"的端到端路线（miaChartGen2 路线）。
- **理由**：
  1. **硬约束可保证**：零报错 / 可玩性 / 无理红线（知识 008-015）必须由显式校验强制执行，端到端模型无法承诺这类保证；
  2. **可解释可复核**：每层输出可人工查看与修正，符合本项目"人机协作、写前必读知识库"的工作流；
  3. **数据效率**：端到端需海量谱面数据（本项目仅 388 个官方 ST 谱作参考）；显式特征低维、规则层零数据需求；
  4. **幻觉可控**：LLM 只接触结构化特征，不消费原始音频（AGENT.md 准则 8）；
  5. **差异化**：现有开源项目无一做分轨踩音（调研结论，§7.3）。
- **保留的开放点**：生成端可采用"**以显式特征为条件的小型生成模型**"（借鉴 miaChartGen2 的分阶段骨架→属性设计，但输入换成我们的特征：节拍网格 + 分段/强度 + 各 stem onset），即"特征先行、生成器后置"——这仍是显式分析路线，只是把 L4 之后的具体谱面写作交给模型完成，并保留规则硬校验兜底。

## 9. 存疑 / 待定（落地前必做）

1. **域外泛化**：所有模型基准都是西方流行乐（Ballroom/GTZAN/Harmonix/MUSDB18），对日语 ACG/音游曲（160-220BPM）的精度未知 → 拿 20-50 首人工核对：节拍/分段命中率、"副歌=重复最多段"命中率。
2. **配方系数待标定**：强度融合权重（0.35/0.30/0.20/0.15）、投票权重（0.4/0.3/0.2/0.1）、模板 T=[0.20,0.45,0.30,1.00,0.10] 均为初值，需在目标曲库标定（电子 drop 与摇滚 chorus 两套预设）。
3. **CPU 实测**：各模型 RTF 在目标机器实测；CPU 弱时分离降级 htdemucs。
4. **鼓 onset 干净度**：无公开定量基准 → 项目内 A/B（整曲鼓点→onset 命中率）。
5. **basic-pitch 对高速/声码器人声**的漏音表现；不行换 RMVPE。
6. **结构分段精度不够的 fallback**："SSM 重复段 + 人声密度"启发式。
7. **器乐曲（无副歌）**：高潮 fallback 为纯能量峰值。
8. miaChartGen2 的 HF 模型权重（Goldgom/models）许可未核实——如需参考其思路，只读设计不取代码。
