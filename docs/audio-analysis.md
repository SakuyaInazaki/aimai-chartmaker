# 音频分析与踩音规划（管线设计）

> **状态**：v1.1（2026-09-11）。在 v1.0（2026-09-10，四层工具选型）基础上合并三份新证据：
> ① `docs/research/audio-analysis-research-v2.md`（第二轮调研，落地问题攻关，其 §8 的 R1–R24 逐条采纳）；
> ② `docs/research/official-chart-density-curves.md`（388 个官方 ST 谱逐小节密度实测）；
> ③ `.agent/notes/035-音频分析原型实现.md` + `tools/audio_analysis/README.md`（原型 v0.1 在 3 首真实曲目上的实跑记录）。
> **冲突仲裁原则**：**调研预期与原型实测冲突时，一律以实测为准**，并在原处标注。
> **目标**：攻克"让 AI 通过分析 mp3 了解哪里是副歌、哪里是高潮、要采鼓点还是人声"——对应 MMFC 教程第五章的强度/踩音/配置理论（知识库 001/002/003）。
> **标注约定**：✅ 一手核实（源码/LICENSE/API/论文原文）｜🧪 本项目实测（原型 v0.1 或 388 谱统计）｜⚠️ 部分核实（二手口径）｜❓ 未确证，待实测。

## 0. 设计原则（与项目约定一致）

1. **LLM 不"听"原始音频**：模型上下文有限且不可靠（见 AGENT.md 准则 8），谱面决策一律基于**结构化特征**（分段标注、onset 流、能量曲线），而非把音频或长特征喂给 LLM——特征提取用确定性 DSP/深度学习模型完成，LLM 只做规划与决策。
2. **用户提供的数据优先**：BPM、offset、变速点可由用户直接提供，管线把这些作为**最高优先级输入**；自动检测只做**一致性校验并如实报告差值，绝不覆盖用户值**（🧪 原型已按此实现）。
3. **可校验**：每个环节的输出都要能被人复核（分段/强度曲线出 `plot.png`），踩音计划必须先通过知识库 001-003 的规则检查。
4. **实测优先于设计预期**（v1.1 新增）：本文档中凡标 🧪 的结论来自真实跑通的实验，优先级高于任何调研推测；启发式环节必须在输出里显式标"存疑"，不得让下游把启发式当事实。

## 1. 管线总览（四层）

```
mp3 ──► L1 节拍层 ──► L2 音轨层 ──► L3 结构层 ──► L4 规划层 ──► song sheet 双格式 ──► LLM 创作
        BPM/beats/     stems 分离+     段落/副歌/       结构×强度×音轨→    song_sheet.md
        downbeats      分音网格量化     强度曲线          逐段踩音计划       song_analysis.json
        +offset校验
```

| 层 | 输入 | 输出 | 对应谱师动作 |
|----|------|------|-------------|
| L1 节拍层 | mp3（+用户提供的 BPM/offset/变速点） | 统一 WAV、网格 `t(bar,div,idx)`、offset 反向校验报告 | "确认 BPM、&offset、有没有变速段" |
| §2.5 量化 | L1 网格 + L2 onset | 逐小节 `{x}` 分音判定 + 网格串 | "这一小节是 8 分还是 16 分" |
| L2 音轨层 | 统一 WAV | 4 stems + 各 stem onset 流 + **逐小节特征清单** | "要采什么音：鼓点还是人声"的**候选池** |
| L3 结构层 | WAV + L1 + L2 | 段落边界与类型（日式/英文双标签）、重复段、逐小节强度曲线、高潮定位 | "哪里是副歌、哪里是高潮" |
| L4 规划层 | L1+L2+L3 + 知识库 001/002/003/031 | song sheet（逐段：采音目标、密度、休息安排 + 逐小节网格串） | "按情绪写配置"的蓝图 |

🧪 **整条冷跑 RTF ≈ 0.29**（M4 / 16 GB / macOS 26 / py3.12 / torch 2.14；2 分钟曲约 36 s）；热跑（复用 stems）RTF 0.14–0.15。Demucs + all-in-one 两个深度模型吃掉 >90% 时间，全部 DSP 环节合计不到 2 秒。

## 2. L1 节拍层：时间轴对齐

### 2.1 Step 0 — 统一解码（硬性前置，R2）

**一律 `ffmpeg -i in.mp3 out.wav`（44.1 kHz PCM），全管线只读这一个 WAV。**

- 依据 ✅：all-in-one-infer README「Concerning MP3 Files」实测——**不同 MP3 解码器有约 20–40 ms 的偏移差异**，而 beat tracking 的常规容差只有 70 ms；osu! 官方 timing 指南同样写明"任何音频编辑（包括重新编码）都会改变 timing"。
- 🧪 实测佐证：三首测试曲的 MP3 容器 `start_time` 全部为 **23–25 ms**（ffprobe），正是上述编码器延迟。该值已写入分析单供人工判断。
- 附带收益 ✅：`torchaudio ≥ 2.11` 已不再自带 mp3 解码（需另装 `torchcodec`），统一转 WAV 一次性绕开全部解码器差异与依赖问题。

### 2.2 用户覆盖路径（主线）

用户提供 BPM + offset（`&first`，秒，可为负）+ 变速点 → 直接构造 `t(bar, div, idx)`；变速按**分段线性**（每段常 BPM）处理。参考实现范式：DDC 的 `BeatCalc`（分段 bps 线性积分，`beat_to_time` / `time_to_beat` 双向）✅ MIT，是最简洁的可抄结构。

🧪 原型限制：网格只在**小节边界**换 BPM；若原曲变速点不落在小节线上，其后所有小节时间都会错位（用了变速参数会在 `warnings` 里报警）。变速支持仍是实验性的。

### 2.3 自动路径（缺省 / 交叉验证）

- **主线 = beat_this**（MIT ✅）：beat+downbeat 一次输出、帧级置信度、无 DBN、对变速宽容（Ballroom beat F1 97.5 / downbeat 95.3）。
  - ⚠️ **MPS 用法（R3）**：CLI 只暴露 `--gpu`（CUDA 语义），Apple Silicon 上必须走 Python API：`File2Beats(checkpoint_path="final0", device="mps", dbn=False)` ✅。
- **打底 = librosa**（ISC ✅）；**备选 = all-in-one-infer**（beat/downbeat 与 L3 同源）。
- **排除**：madmom（模型权重 CC BY-NC-SA + 2018 年 Cython 包，py3.12 装不上 ✅）、essentia/tempo-cnn（AGPL ✅）。
- BPM 上下限显式设置：目标 160–220，默认 max_bpm 会在 220 档折叠成 110。
- ⚠️ librosa 1.0 已移除 `librosa.beat.tempo`；`librosa.beat.plp` 仍在（`__all__=["beat_track","plp"]` ✅）。

### 2.4 offset 反向校验（v1.1 新增，R1）

用户给的 offset 错 20 ms 会让全谱皆错，而校验成本极低（🧪 实测 0.85–1.1 s / 曲，RTF ≈ 0.008）。这是**只报告、不覆盖**的环节。

**算法（网格相位搜索）**

```
输入：用户 BPM/offset、统一解码后的 WAV
1) 在 drums stem（或全曲）上做 onset 检测，hop 必须细：sr=44100, hop=256 → 5.8 ms/帧
   （强度曲线用的 hop=512@22050 = 23 ms 太粗，量级与待测偏移相当，会把答案磨平）
2) 取用户网格上的 {8} 点集合 {t_k}（{8} 比 {16} 鲁棒；也可只用 downbeat）
3) 对 φ ∈ [−beat/2, +beat/2)，步长 ~1–5 ms：S(φ) = Σ_k O(t_k + φ)，φ* = argmax S(φ)
4) 判据：|φ*| ≤ 10 ms → 可信直接用；10 < |φ*| ≤ 30 ms → 提示"疑似解码偏移"，
   记录但仍以用户值为准（准则：用户优先）；|φ*| > 30 ms → 报警，交人工确认
5) BPM 漂移检测：全曲按每 16 小节分块逐块求 φ_i，对 (t_i, φ_i) 线性回归，
   斜率 a ≠ 0 表示网格在漂 → BPM_true ≈ BPM_user × (1 − a)；|a| < 2e-4 视为无偏
6) 独立第二意见：librosa.beat.plp(prior=围绕用户 BPM 的窄正态) 的峰位相位，与 φ* 对照
```

**🧪 实测修正（三条，均以实测为准，覆盖 v2 §1.2 的原始设计）**

1. **打分量必须用 onset 时间，不能用 onset 强度包络**。原设计的"整曲 onset 强度包络 beat-synchronous 打分"在三首曲子上一律报 +36~+40 ms；在**合成 click 音频**（真值 `first=0.5`）上实测，包络法给 0.5107（**+10.7 ms 系统性滞后**，来自 librosa `onset_strength` 的帧间谱流滞后），而 backtrack 后的 onset 时间只差 −4.9 ms。→ 改用 **onset-fit**（backtrack 后的 onset 时间对拍线的加权命中率）作主打分，包络法结果作为 `envelope_delta_ms` 一并输出。
   反向验证：用包络法的 +40 ms 重建网格后，量化平均误差从 9.47 ms **劣化**到 13.46 ms、分音直方图从"8 分为主"塌成"24/32 分为主"——证明用户给的值才是对的，+40 ms 是伪信号。
2. **offset 只能定到一拍以内**。整拍平移的分数几乎相同，取全局最大值会把结论错移整整一拍（🧪 TransientTears 曾被报成 −272 ms）。→ 必须把同级峰**分组取峰**，选离用户值最近的那个，并把全部候选列进 `peaks`。
3. **三首实测差值 +8.0 / +10.6 / +13.7 ms（z = 3.0–3.5）**，全部落在"可信"或"疑似解码偏移"档，与 MP3 容器 `start_time` 23–25 ms 同量级。

**方法论旁证** ✅：Osu2MIR（ISMIR 2025 LBD）反向工程 osu! 的未继承 timing point 当 beat/downbeat ground truth，并发现"单个 timing point、或多点间隔 ≥ 5 s 的图标注可靠"——说明**人工确定的 BPM+offset（恒速）可以当 ground truth**，本项目"让用户提供 BPM/offset"的路线在方法论上站得住。

## 2.5 onset → 分音网格量化规则（独立节，R4）

### 2.5.1 前置事实与信息论天花板

- simai 时间模型：`{x}` 表示"一个逗号 = x 分音符"，**槽长 = 240 / BPM / x 秒**（`docs/simai-syntax.md` §1.4/§3.2）。
- 目标 BPM 区间的槽长（ms）：

| BPM | {4} | {8} | {12} | {16} | {24} | {32} | {48} |
|-----|-----|-----|------|------|------|------|------|
| 160 | 375.0 | 187.5 | 125.0 | 93.8 | 62.5 | 46.9 | 31.2 |
| 180 | 333.3 | 166.7 | 111.1 | 83.3 | 55.6 | 41.7 | 27.8 |
| 200 | 300.0 | 150.0 | 100.0 | 75.0 | 50.0 | 37.5 | 25.0 |
| 220 | 272.7 | 136.4 | 90.9 | 68.2 | 45.5 | 34.1 | 22.7 |

- 行业标准容差 ✅：`mir_eval` onset F-measure 窗 = **±50 ms**、beat F-measure 阈 = **70 ms**。
- **关键推论**：BPM 180 的 `{32}` 槽长 41.7 ms、BPM 220 的 34.1 ms，**都小于通用 onset 检测的标准容差窗**——"自动把 onset 量化到 {32}"在本项目的目标 BPM 区间**在信息论上就站不住**。`{24}`（45–62 ms）同样在噪声边缘。

### 2.5.2 量化步骤

**Step 1 拍同步重采样**：不要"固定 hop 提特征再找最近格子"，而是把 onset 包络按拍重采样，**每拍 N=48 个采样点**（= 每小节 192 点，同时整除二分族与三分族，恰好等于 simai 的 `{192}`）。依据 ✅：Yi 2023（arXiv 2311.13687）的 beat-aligned 频谱就是 hop = 1/48 拍。

**Step 2 逐小节推断最小分音**（奥卡姆 + 知识 005「采音要简」）：

```
候选族：二分族 D2 = [4, 8, 16, 32]；三分族 D3 = [12, 24]
容差：τ(d) = clamp(0.25 × slot_ms(d), 12 ms, 30 ms)
从粗到细扫描 d ∈ [4, 8, 12, 16, 24, 32]：
  r(d) = max_i min_k |p_i − k/d| × bar_ms        # 最大残差
  第一个满足 r(d) ≤ τ(d) 的 d 即为该小节的 {x}
```

**Step 3 三连 / 二分判别**：分别用 d=16 与 d=12 拟合，判为三连需**同时**满足 `rms12 ≤ τ(12)` 且 `rms12 ≤ 0.7 × rms16`（音游曲绝大多数是二分体系，三连需显著证据才能推翻默认）。

**Step 4 swing 判别（独立于三连）**：取反拍 onset 算 `s = median((p_i − p_onbeat)/(1/8))`；`s ≈ 0.5` 直、`s ≈ 0.667` 三连 swing → 改用 `{12}`/`{24}`。⚠️ swing ratio 随 tempo 升高近似线性下降（Dittmar/Müller, ISMIR 2015），160–220 BPM 下出现概率低 → **默认关闭**，仅在 `rms12 << rms16` 且 s 稳定落在 0.6–0.72 时启用。

**Step 5 `{32}` 红线**：**默认禁止自动量化产出 `{32}` 及更细。** 唯一例外需三条同时成立：(a) 该小节 onset 数 ≥ 6；(b) 来源是 **drums stem**（瞬态最锐）；(c) `{32}` 下 RMS 残差 ≤ 15 ms。
制谱侧理由同向：`{32}` 在 ST 谱里属装饰性 burst，应由 LLM 按**配置需要**主动使用（知识 003/015），而不是由量化算法"检测"出来。

**Step 6 输出**：每个 onset → `{bar, div, idx, t_ms, residual_ms, stem, strength, confidence}`；每小节 → `{bar, div, n_onset_per_stem, fit_rms_ms}`；网格串 `x...x..x....x...`（`.` 空 / `x` onset / `X` 强 onset）。

### 2.5.3 🧪 实测：这是全管线最不可靠的环节

- 量化误差（原型 v0.1，drums 轨）：平均 6.78 / 7.35 / 9.47 ms，最大 24.9–25.1 ms，"无法解释的小节" 全曲为 0。
- **但 `unresolved_bars = 0` 不等于量化准确**：原型的容差取 `max(25 ms, 1/64 小节)`，**192 BPM 下一小节 1.25 s，24 分与 32 分格子只差 ~13 ms，远小于容差**，于是"最小可解释分音"在 16/24/32 之间**基本是随机挑**。
- → **高 BPM 曲目请把网格串当"节奏轮廓"，不要当谱面分音的依据。** 这与 §2.5.1 的信息论推论完全一致，是本节最需要记住的一条。
- ❓ 待办：容差三参数（0.25 / 12 / 30）与三连阈 0.7 需在 osu2beat2025 子集上标定（见 §10）。

### 2.5.4 成熟做法的取舍

| 来源 | 实际做法 | 对我们的价值 |
|------|----------|--------------|
| **DDC** 推理端 | **根本不做音乐量化**（R22 事实更正）：写死 `_SUBDIV=192, _DT=0.01, _BPM=125`，即**伪造一个 BPM 让 1/192 小节恰好等于 10 ms 帧**，输出时间均匀网格的 .sm ✅ | ❌ 不是 snap 逻辑的参照 |
| **DDC** `beatcalc.py`/`abstime.py` | 分段 BPM/stop ↔ 时间双向换算 | ✅ 时间轴换算的干净范式，可直接照抄结构 |
| **osu! 编辑器** | beat snap divisor 11 档 1/1…1/16，含 1/3、1/6；timing 靠人工敲 T 键 ⚠️ | ✅ 佐证"候选集要同时含二分族与三分族"；❌ 无算法可抄 |
| **MIDI 量化文献** | 规则 → metrical HMM → Transformer（给定 beat 标注量化，ASAP onset F1 97.3%）⚠️ | ✅ 思路可借：网格语法合法性应作硬约束（simai 的 384 约数就是）；❌ 我们的输入是机械制作电子乐 + 已知恒定网格，规则 + 残差判据足够，上 HMM/Transformer 是过度设计 |
| **Yi 2023** | hop = 1/48 拍 beat-aligned 频谱，osu!mania 14648 谱，micro-F1 84.6% ✅ | ✅ 拍同步重采样直接采纳 |

## 3. L2 音轨层：踩音候选池

### 3.1 声源分离

- **主线 = Demucs v4**（MIT ✅ 代码+权重）——drums/bass/vocals/other 四路，鼓 SDR 10.08 泄漏少。
  - 🧪 **实测：htdemucs 在 MPS 上 RTF ≈ 0.10–0.11**（12.6–13.8 s / 2 分钟曲，未回退 CPU，首次含权重下载 40 s）。迭代期用 `htdemucs`，量产期再上 `htdemucs_ft`（慢 4 倍、质量更好）。
  - ⚠️ Demucs `separate.py` 在 CUDA 不可用时会自动选 `mps` ✅；历史上 HTDemucs 在 MPS 有复数张量兼容问题，且 macOS 26.x 上 `torch.backends.mps.is_available()` 有误报 issue（pytorch#177819）→ 实现必须带 **MPS 失败自动回退 CPU** 并记录实际设备（🧪 原型已实现）。加速备选：`demucs-mlx` / `mlx-audio-separator`（MIT ✅）。
  - **排除**：Spleeter（质量垫底）、BS-RoFormer（权重许可不明）、AudioSep（过度设计）、Open-Unmix umxl（权重 CC BY-NC-SA）。
- 内存：M4/16 GB 下建议**分阶段落盘**（分离 → 写 stem wav → 释放 → 再分析），不要四个 stem 全程驻留。

### 3.2 逐小节特征清单（v1.1 新增，R5）

L2 必须为每个小节 b 输出下列特征——它们是 L4 切轨规则表的全部输入：

| 特征 | 计算 | 用途 |
|---|---|---|
| `share_s(b)` | stem s 在小节 b 的能量占比 `E_s / ΣE` | "哪个响" |
| `n_onset_s(b)` | stem s 的 onset 数 | 可踩音数 |
| `grid_fit_s(b)` | stem s 的 onset 落在 `{8}` 网格上的比例 | 律动规整度；低 = 采样/人声长音，不适合当骨架 |
| `voiced_ratio(b)` | vocal stem 有声帧占比 | 人声进出——**切轨最强信号** |
| `vocal_notes(b)` | vocal stem 的 note 数 / 音高中位 / 最高 | 区分"人声长音"（verse）与"密集人声"（chorus） |
| `kick/snare/hihat(b)` | drums stem 三带 onset 数（§3.3） | 段落类型判别 + 骨架 |
| `bass_onsets(b)` | bass stem 的 note onset | 低频律动骨架；kick 不明显时替代 |
| `riff_sim(b, b−4/−8)` | other stem 的 chroma/CQT 小节向量余弦相似 | hook/riff 循环检测与变化点 |
| `sil_run(b)` | 最长静默长度 | 休息段 / 留白识别 |

### 3.3 鼓件三频带启发式（v1.1 新增，R6）

**一手许可证核实：ADT 预训练模型全线非商用，默认管线不能用。**

| 工具 | 许可 | 结论 |
|---|---|---|
| **ADTOF**（5 类鼓件） | **CC BY-NC-SA 4.0** ✅（LICENSE 原文） | ❌ 违反 §7.2 红线 → 排除，仅可做个人非商用插件 |
| **LarsNet**（鼓件分离 5 路） | 仓库 `license: None` ✅，README 称权重 CC BY-NC 4.0 ⚠️ | ❌ 排除 |
| **omnizart**（drum） | 代码 MIT ✅，但 `requires_dist` 含 `madmom>=0.16.1` ✅ | ❌ 排除（许可连坐 + py3.12 装不上，双杀） |
| **drumsep**（Demucs 微调） | 代码 MIT ✅，权重许可未明示 ❓ | ⚠️ 可选插件，用前需核 |

**→ 默认方案：无模型的频带启发式**（在 Demucs `drums` stem 上做，泄漏已被分离器处理掉）：

```
LOW  : 30–120 Hz         （kick 基频，峰值多在 50–80 Hz）
MID  : 120–400 Hz + 1.5–8 kHz 宽带噪声（snare：低频体 + 响弦噪声）
HIGH : 6–16 kHz          （hihat / cymbal）
判据：LOW 占比最大且 HIGH < 0.2 → kick；MID 宽带噪声显著（谱平坦度高）→ snare；
     HIGH 占比最大且衰减快(<80 ms) → closed hihat；衰减慢(>300 ms) → cymbal/crash
```

**🧪 实测强制修正（必须遵守）**：**不能各频带独立检 onset**——原型第一版这么做，结果 kick(715) 比整条 drums 轨(626) 的 onset 还多。正确做法是**先在 drums stem 上检出 onset，再按频带相对能量给每个 onset 贴标签**（一个 onset 可带多个标签），结果必然是 drums onset 的子集（实测 469/382/519）。

⚠️ 可靠性定位：这套的价值**不在转录精度，而在段落判别**——kick 消失 + hihat 密度翻倍 = 典型 build；crash 落在小节 1 拍 = 段落起点。🧪 实测混淆很多：底鼓 vs 低音 tom、军鼓 vs 拍手/边击、hihat vs 镲片与人声齿音泄漏都会混；**密集段里 kick 与 hihat 的网格串常常完全一样**。❓ 在 160–220 BPM 高密度电子鼓组上的可分性需 A/B 实测。

### 3.4 人声活动检测 VAD（v1.1 新增，R7）

- **不要用语音 VAD**：silero-vad（MIT ✅）与 pyannote（MIT ✅）都是**语音域**模型，对歌唱（长音、颤音、和声、声码器）表现未验证 ❓，且 Demucs 的 vocal stem 里本来就只剩人声。
- **推荐（零模型）**：`voiced(t) = [RMS_vocal(t) > θ] ∧ [谐波性 > θ_h]`，θ 取全曲 vocal RMS 的 P40；谐波性用 `librosa.effects.harmonic` 能量比。加滞回（进入 150 ms / 退出 300 ms）去抖。
- 🧪 **实测已知缺陷**：原型阈值取"整曲 RMS 峰值 −32 dB"，**Demucs 的 vocals 轨常混进 lead synth，导致器乐段被误判为有人声**。→ VAD 结论必须标存疑，且在 L4 切轨时要与 `vocal_notes` 交叉验证。❓ 阈值待用 RWC-Pop（有旋律标注）标定。

### 3.5 人声旋律（note 事件）

- **主线 = basic-pitch**（Apache-2.0 ✅）：直接输出 note 事件（onset/offset/pitch/confidence）+ MIDI，一份输出同时解决"人声有没有"和"人声踩哪里"。
- 🧪 **实测：装不上，当前缺席**。`basic-pitch[onnx]` 两次尝试都报 `error: resolution-too-deep`（pip 在 `narwhals` 上无限回溯，加约束仍失败），超时放弃。**已按预案降级为用 vocals stem 的 onset + 活动度代替人声 note 事件**——代价是没有音高/时值，`vocal_notes` 特征目前不可用。
  - ⚠️ 相关背景 ✅：`basic-pitch[tf]` 钉死 `tensorflow-macos<2.15.1`，而该版本无 cp312 wheel → py3.12 上 TF 后端本就装不上；Darwin 默认会拉 `coremltools`（9.0 支持 3.12 ✅）走 CoreML。
  - 后续路径：换 pip 解析策略 / 锁死版本重试 onnx 后端，或改用 **RMVPE**（Apache-2.0 ✅）。❓ 三后端（CoreML/ONNX/TF）数值是否一致需实测，若不一致固定用 ONNX（onset 时间差会直接污染量化结果）。
- **排除**：CREPE（慢）、essentia Melodia（AGPL）、pYIN（GPL）。

### 3.6 输出

四组 onset 时间序列 + §3.2 的逐小节特征表 + 鼓件标签 + VAD 曲线——这就是知识 002"切轨"的候选池。⚠️ 分离质量直接影响踩音准确度；鼓 stem onset 通常最可靠（音游谱以鼓为骨架），人声 onset 需后处理去噪。

## 4. L3 结构层：段落 / 副歌 / 高潮 / 强度

### 4.1 事实更正（R8 / R9，v1.0 有误）

1. **all-in-one 的功能标签只有 8 类**（+ start/end 哨兵）：`intro / outro / break / bridge / inst / solo / verse / chorus` ✅（`src/allin1/config.py: HARMONIX_LABELS`）。
   → **没有 pre-chorus、没有 build**。v1.0 §4 写"10 类标签"、§5.3 样例直接写 `"type": "pre_chorus"` 都是错的——**`pre_chorus` 必须由规则派生**（派生判据见 §4.4）。
2. **all-in-one 出处是 WASPAA 2023，不是 ISMIR 2023**（arXiv 2307.16425，Kim & Nam）⚠️。v1.0 §4/§7.3 两处已更正。

### 4.2 分段：三路投票

| 角色 | 选择 | 理由 |
|---|---|---|
| **主线** | `all-in-one-infer` 3.1.0（PyPI 名，MIT ✅，R20） | 许可干净、py3.12 ✅、同时给 beat/downbeat/段落，与 L1 同源 |
| **二号意见（非商用分支）** | SongFormer（CC BY 4.0 ✅，R10） | 严格边界显著更好；有 prechorus/build/quietchorus 标签。**但前端 MuQ 权重是 CC BY-NC-4.0 ✅ → 只入个人实验分支** |
| **第三票（零风险）** | libfmp/librosa SSM 重复段 | RefraiD 的现代等价物；**器乐曲与主线失效时靠它兜底** |
| **投票规则** | 边界三者取交集 + **吸附到最近下拍**；标签 chorus 需 ≥2 票 | SongFormer 帧率 8.333 Hz（120 ms/帧）✅，边界分辨率天花板就是 120 ms，必须再吸附一次 |

**🧪 关于 all-in-one 的安装实测（推翻 v2 §6.2 的预期）**

- `all-in-one-infer` 3.1.0 **可用**（模块名 `allin1_infer`），且**支持直接喂已跑好的 Demucs stems**（`create_stems_input_from_directory`），省掉一次重复分离——这是实跑发现的重要优化点。
- **`natten` 在 Apple Silicon + py3.12 上源码编译通过**，调研预期的"NATTEN 无 wheel、自编译坑"**未出现**。（但仍推荐 `all-in-one-infer`，因为它把 natten 降级成可选 extra。）
- 官方 `allin1` 1.1.0 装上但 **import 即失败**（硬依赖 madmom，而 madmom 权重是 NC 红线）→ **不要装 `allin1`**。
- 🧪 **许可风险已解除**：`all-in-one-infer` 会拖进 `madmom-infer`（BSD-2 ✅），其 `models.py` 声明会运行时下载 CC BY-NC-SA 权重。实跑后检查 **`~/.cache/madmom_infer/models` 为空** → all-in-one 只用了 madmom 的 DBN 解码器（纯 HMM 算法、无学习权重），**未触及 NC 权重**。对外发布前仍建议每次复核一次此目录。

**🧪 分段可用性体检（原型新增，必须保留）**

两条路径**都会退化**：all-in-one 在 TransientTears 上只给 3 段、最长段占全曲 63%，对制谱毫无信息量；自研 SSM 的第一版也把 13–68 小节合成一个 B 段。
→ 统一体检判据：**段数 ≥ 5 且最长段 ≤ 35% 小节**；不通过就退回 SSM，SSM 内部再对"边界数 × 簇数"自适应重试；被否决的那版保留在 `alt_segments` 供人工比对。
→ ⚠️ **段落表必须人工过目 `plot.png` 复核**，这是当前管线里第二不可靠的环节。

**"副歌 = 重复最多段"的证据（R12）**：Goto (2006) 的 RefraiD 在 **RWC-Pop 100 首中 80 首完全正确、F = 0.938** ⚠️，而 **RWC-Pop 的 100 首里有 80 首是日语 J-pop** ✅。→ 该启发式在**日语流行乐上基本成立**，v1.0 §9.1 的"命中率未知"过于保守，可下调为"J-pop 成立"。**但**：(a) RWC-Pop 是 1990s 风格，与 160–220 BPM 的 ACG/音游曲仍有域差；(b) **器乐向音游曲（东方/hardcore/EDM）常常没有"副歌"概念**，只有 build → drop，此时该启发式失效，必须走能量/drop 通道。
⚠️ 没有任何 MSA 模型公布过日语曲上的分段指标 ✅（Harmonix = 西方流行；SongFormBench 语言字段 `["en","zh"]`，**无 ja**）→ 日语域验证必须自己做（§10 A 档）。

### 4.3 日式（英文）双标签词表（v1.1 新增，R11）

**规定：分析输出一律用「日式(英文)」双标签**——知识库与 MMFC 教程用的是中/日语境，且强模型对 `サビ`/`落ちサビ` 的语义理解比 `chorus` 携带更多制谱含义。

| 日式 | 英文 | all-in-one | SongFormer 细标签 | 本项目标准标签 | 制谱含义 |
|---|---|---|---|---|---|
| イントロ | Intro | `intro` | `intro`/`instintro`/`fadein` | `intro` | 低强度；"哪个响踩哪个" |
| Aメロ | Verse | `verse` | `verse`/`slowverse` | `verse` | 中强度；踩器乐，小节尾混人声 |
| Bメロ | Pre-Chorus | ❌ 无 | `prechorus`/`build` | `pre_chorus` | 爬升；**末尾切回人声铺垫** |
| サビ | Chorus / Hook | `chorus` | `chorus`/`altchorus` | `chorus` | 高强度；vocal 曲全踩人声 |
| 落ちサビ | Quiet Chorus | ❌ | `quietchorus` | `quiet_chorus` | **减压休息段**，密度骤降 |
| 大サビ / ラスサビ | Final Chorus | `chorus` | `chorus`（靠位置判定） | `final_chorus` | 全曲峰值；配置升级（单星→双手星） |
| Cメロ | Bridge | `bridge` | `bridge`/`instbridge` | `bridge` | 转折；换轨 |
| 間奏 | Interlude | `inst`/`solo` | `interlude`/`inst`/`solo` | `interlude` | 踩采样/独奏；靠**位移**提强度 |
| ドロップ | Drop (EDM) | `chorus`/`break` | `breakdown`/`build` | `drop` | 器乐曲的"副歌"，能量票主导 |
| アウトロ | Outro | `outro` | `outro`/`vocaloutro` | `outro` | 与 intro 前后呼应 |

⚠️ 日英并非严格一一对应（`Bメロ` 在部分语境被叫 Bridge），**映射以位置 + 功能判定为准，不以词面为准**。SongFormer 的 128 类细标签需先归并到本表的标准标签。

### 4.4 `pre_chorus` / `build` 派生规则（因主线无此标签）

满足**全部三条**：① 位于 chorus 起点前 4–8 小节；② 强度曲线单调上升；③ kick 密度下降或消失 ∨ hihat 密度上升 ∨ 高频能量单调上升（riser）。
这是 MMFC 5.4 第 3 条"build 末尾切回人声"的定位前提，不能省。

### 4.5 强度曲线配方 v2（R13，替换 v1.0 的融合式）

**v1.0 的问题**：系数全是拍脑袋；**质心权重过高**——质心对失真吉他、riser 敏感，但**对高音人声同样敏感**，容易把 Bメロ 的人声爬升误判成 drop；`onset_strength` 用全曲混音，被最响音轨支配，正是 MMFC 5.4 批评的"哪个响踩哪个"。

```
1) 统一 WAV → librosa.load(sr=22050, mono) + pyloudnorm 归一 −14 LUFS（防"谁响谁高潮"）
2) 同 hop=512 提特征；offset 校验另走 hop=256@44100（§2.4），不共用
3) 逐特征 5–95 百分位截断 → min-max 归一（记为 Z(·)）
4) 融合（新式）：
   I_bar = w1·Z(loudness_bar)     # 逐小节响度
         + w2·Z(n_onset_bar)      # 各 stem 去重合并 onset 数 ——「可踩音上限」
         + w3·Z(E_drums_bar)      # 鼓能量 —— 律动骨架
         + w4·Z(voiced_ratio_bar) # 人声活动率 —— J-pop 的情绪主载体
         + w5·Z(flux_bar)         # 谱通量 —— 瞬态密度
   初值 w = [0.25, 0.30, 0.20, 0.15, 0.10]      ❓ 仍未标定
   相对 v1.0 的改动：onset 从"强度包络"改为"去重 onset 计数"并升为最大权重；
                     新增人声活动项；**质心项删除**（移到 drop 专用票）
5) 小节聚合：直接用 §2.2 构造出来的 downbeat，**不要用 beats[::4]**（只在恒定 4/4 且首拍对齐时成立）
6) 平滑：小节级中值滤波（窗 ≈ 2 小节）+ 高斯(σ≈1)
7) 高潮定位 = 五票投票（§4.6）
8) 映射到谱面密度（§4.7）
```

⚠️ 🧪 原型 v0.1 仍在跑 v1.0 的四项融合式（rms/onset/centroid/drums），**新融合式尚未实现**；权重与投票权重都还是设计初值，**未在本项目曲库上标定**。

### 4.6 高潮定位：四票 → 五票（R14）

| 票 | v1.0 | v1.1 | 理由 |
|---|---|---|---|
| V1 结构票（chorus/重复段内） | 0.40 | **0.45** | 现在有两到三个模型投票，可靠性提高 |
| V2 novelty 票（SuperFlux 局部峰 ±1 小节三角窗） | 0.30 | 0.25 | 保留 |
| V3 能量票（rms_n>0.75 且局部极大） | 0.20 | 0.20 | 保留；**器乐曲 fallback 时升到 0.5** |
| V4 质心票（cent_n>0.7） | 0.10 | **0.05** | 降权，只留给 EDM drop |
| **V5 人声票（新增）** | — | **0.05** | `voiced_ratio > 0.6` 且音高中位处于全曲上四分位 → J-pop 副歌强特征 |

`climax = argmax(vote)`；前 k 个峰 = 各次副歌出现时间。
**结构层必须输出 `repeat_of` 与 `chorus_index`**——MMFC 5.4 第 7 条"第二副歌踩音相同但配置升级"要求能明确区分"第 N 次副歌"。`final_chorus / 大サビ` = 所有 chorus 段中"**最后一个 + 强度最高**"者。

五票互补原理：结构票覆盖摇滚 chorus=整段反复；novelty 票覆盖 drop/爆发瞬态；能量票覆盖响度/密度；质心票覆盖电子乐 drop；人声票覆盖 J-pop。流派差异建议两套权重预设（电子 drop vs 摇滚 chorus）。

### 4.7 强度 → 密度：用官方谱实测取代手写模板（R15，v1.1 最重要的一处修订）

**v1.0 的 `T=[0.20,0.45,0.30,1.00,0.10]` 作为密度目标已被 388 个官方 ST 谱实测推翻**（详见 `docs/research/official-chart-density-curves.md` §4，知识 031）。

**(1) 形状先验换成实测值**

```
T_density = [0.62, 0.77, 0.72, 0.78, 0.82]        # 结构对齐五段口径，归一到峰值=1
95% 自助 CI：[0.60–0.64, 0.75–0.79, 0.70–0.74, 0.76–0.80, 0.80–0.84]（n=388）
等分五段口径：[0.81, 0.92, 0.91, 0.97, 1.00]（相对全曲均值 [0.88,0.99,0.98,1.06,1.09]）
⚠️ 两种口径不可混用，使用时必须注明
```

- **前四段"弱 → 较强 → 较弱 → 强"的定性顺序被 388 谱完整验证**（段序 s1 < s3 < s2 < s4）；
- **第五段被推翻**：实测 s5 = 0.817 是五段最高，不是 T 说的 0.10；
- **T 的动态范围是真实值的约 4.5 倍**——拿 T 当密度目标直接驱动生成，第 5 段只会给出 1/10 峰值的密度，远低于官方谱下限。

**(2) 密度地板（映射的核心非线性）**

音频强度可以逼近 0，**谱面密度不能**——任何一小节只要还在出音，就得给出基本踩音密度。实测归一密度分位（388 谱均值）：**p10 = 0.264 / p50 = 0.514 / p90 = 0.730**；段落尺度上地板被抬高到 **≈ 0.62**（结构对齐五段的最弱段）。

```
density_norm = floor + (1 − floor) · intensity_norm
  段落尺度（给每段定基调）：floor ≈ 0.60
  小节尺度（段内细分）：    floor ≈ 0.25
```

**(3) 六条官方谱密度规律（L4 直接当规则用）**

| 规律 | 实测数字 | 对生成的约束 |
|---|---|---|
| **末段最强，不减压收尾** | 曲末 10% 密度 = **1.084 × 全曲均值**；**24.2% 的谱把高潮段放在最后 10%**（最大众数）；高潮段中位位置 **0.72**，**74.7% 落在后半程** | 禁止"峰在 4/5 处再淡出"；收尾减压只发生在最后 1–2 小节 |
| **第二副歌更密但幅度小** | 后半/前半强度比中位 **1.119（≈ +12%）**，**79.6% 的谱后半更强** | 第二副歌密度 +10%~+15%，不要翻倍 |
| **"强度 ≠ 密度"** | SPICY SWINGY STYLE 第二副歌密度只 **+5.1%**，而知识 002 说它"配置升级" | 强度差异主要由**配置复杂度**（知识 003 四因素）承担，密度只是其中一个维度 |
| **休息靠低密段，不靠空小节** | 休息小节（note ≤ 1）只占全曲 **2.43%**，**35% 的谱一个都没有**；低密段每谱 **5.91 段 × 1.69 小节** | 休息段写成"密度降一档"，不要写成空小节 |
| **三明治结构是主流** | "高密 → 低密 → 高密" **61.9% 的谱至少出现一次**，每谱均值 2.41 次 | 两个副歌之间应安排一次明显的减压 |
| **约 7 个密度台阶** | 每谱密度台阶（≥4 小节同档平台）均值 **7.09、中位 7**；中位 85 小节 → **平均每 12 小节换一次密度基调** | 这是 L4 做段落规划的天然粒度 |

**(4) 绝对量级锚点**（note/小节，13.0→8.03 / 13.5→9.15 / 14.0→10.54；全库 8.96；**峰值小节 ≈ 2.06 × 全曲均值**）——与知识 004 的 note 总数区间、知识 015 的密度红线一起做硬裁剪。

⚠️ 样本限制：只覆盖 ST 谱、定数 13.0–14.5、MASTER/Re:MASTER；12 级及以下与 DX 谱未测，外推需谨慎。
⚠️ 方法学坑：变点检测**必须加最小段长约束**（`min_len_frac=0.10`），否则 37.4% 的谱会被切出一个极短的第 5 段，**凭空制造"尾部渐弱"的假象**。
⚠️ k-means 轮廓系数 < 0.08 → 官方谱密度曲线是**连续谱系，没有清晰可分的"流派"**，聚类结果只能当倾向描述（三簇：全程高密平台型 40.7% / 双段递进型 33.5% / 单峰后置型 25.8%）。

### 4.8 无官方音频时的标定协议（v1.1 新增，R16）

现实：`resource/official-chart/` 有 388 个官方 ST 谱但**没有对应 mp3** → 无法做"音频特征 → 谱面密度"的回归。应对：把标定拆成五块，其中三块不需要音频。

- **(A) 形状先验（不需要音频）** — 已完成 ✅：388 谱逐小节密度 → 归一 → 重采样 32 bin → 变点检测五段 → `T_density`（§4.7）。顺带实证检验了知识 001 的五段走势（前四段成立、第五段推翻）。
- **(B) 分位映射：强度 → NPS（不需要配对数据）**
  ```
  d_target(b) = F_chart⁻¹( F_audio( I_bar(b) ) )
  F_audio  = I_bar 的经验分位函数；F_chart = 官方同定数谱 d_b 的经验分位函数
  ```
  这是统计降尺度的标准 quantile mapping：**只要求两边的排序结构可比，不要求逐条配对**。保证绝对 NPS 水平来自官方谱（难度校准正确），相对起伏来自音频（情绪贴合正确）。再叠加知识 004（总 note 数区间）与知识 015（密度红线）硬裁剪。
- **(C) 谱面内部结构比值（不需要音频）** — 已完成 ✅：见 §4.7(3) 六条规律；变点检测能从密度曲线直接还原乐曲段落边界（SPICY 个案 4 个边界 4/4 命中知识 002 的叙事）。
- **(D) 音频侧人工听审**：选 5–10 首目标域曲逐段听审给强度档（1–5），与 `I_bar` 的段落均值算 Spearman 相关，**目标 ρ ≥ 0.8**。这是当前唯一能验证音频侧的手段。❓ 未做。
- **(E) 将来若用户提供官方音频**：才能做真正的配对标定——逐小节 `I_bar` vs `d_b` 的 Pearson/Spearman、峰值位置误差、对 w1..w5 的岭回归拟合。**这是唯一能把权重从"初值"变成"标定值"的路径**，作为待办挂起（哪怕 10–20 首就能跑通）。

> ⚠️ (A)(B)(C) 全部只使用 `resource/official-chart/` 的官方谱；不引入任何社区/自制谱作为设计或评测基准（用户规矩）。

### 4.9 强度层排除项（R17）

- **pop-music-highlighter**（TISMIR 2018）：**GPL-3.0** ✅ + TF 1.x、2018-10 停更 → **红线排除**。
- Essentia DEAM arousal：AGPL + 模型 CC BY-NC-SA → 排除。
- MERT / MuQ / music2vec 特征：权重 CC BY-NC → 排除。
- → **强度层维持"纯 librosa + pyloudnorm、零模型"的定位是正确的**，本轮没有找到许可干净又明显更好的替代品。

## 5. L4 规划层：逐段踩音计划（核心设计）

> 这一层把 MMFC 第五章的谱师决策规则化，输入全部是 L1–L3 的结构化特征 + 知识库规则。
> 🧪 **当前状态：未实现**（原型 v0.1 只到 L3）。

### 5.1 输入

- 段落列表（每段：小节范围、日式(英文)双标签、强度档位、`chorus_index`、`repeat_of`、各 stem onset 密度）
- 目标难度（→ 目标 NPS 区间与 note 总数区间，知识 004；note/小节量级见知识 031）
- 知识库规则：001（强度贴合情绪）、002（切轨）、003（配置强度四因素）、005（采音要简）、012–015（可玩性约束）、016（ST 要素黑白名单）、**031（官方谱密度曲线实测规律）**

### 5.2 「段落类型 × 特征 → 主踩音轨」规则表（v1.1 新增，R19）

把 MMFC 5.4 的 8 条谱师判断逐条翻译成可计算判据：

| # | 段落类型 | 触发判据（逐小节特征） | 主踩音轨 | 副踩 / 备注 | 出处 |
|---|---|---|---|---|---|
| 1 | `intro` | `voiced_ratio < 0.2`，段内 `argmax_s share_s` 可能切换 | **`argmax_s share_s`（逐 2–4 小节重算）** | "哪个响踩哪个"——**仅前奏/尾声可用** | 5.4-1 |
| 2 | `verse` | `voiced_ratio ∈ [0.3,0.7]` 且 `vocal_notes/bar < 4`（人声长音为主） | **`other`（吉他/小号等）或 `drums`** | **小节尾混人声**：每 4 小节的最后 1–2 拍切 vocal | 5.4-2 |
| 3 | `pre_chorus`/`build` | 强度单调上升 ∧（kick 密度↓ ∨ hihat 密度↑ ∨ 高频能量单调↑） | 段首 `drums`/`other`，**最后 1–2 小节切 `vocals`** | 为副歌铺垫，切轨点必须落在小节线 | 5.4-3 |
| 4 | `chorus` | `voiced_ratio > 0.6` ∧ `vocal_notes/bar ≥ 4` | **`vocals`** | kick 处叠 each/双押作重音 | 5.4-4 |
| 5 | `interlude`（有人声采样） | `voiced_ratio ∈ [0.2,0.5]` ∧ vocal note 短促（中位时长 < 200 ms）∧ 重复性高 | **vocal 采样** | **强度靠位移提升，不靠密度**（知识 003-2） | 5.4-5 |
| 6 | `休息段` | 位于两个高强度段之间 ∧ `I_bar < P40` | **次响音轨**（`share_s` 第二名，通常是 other 里的管乐/synth） | 显式标 `rest:true`，**密度降一档但不要清零**（知识 031：休息小节仅占 2.43%） | 5.4-6 |
| 7 | `final_chorus`/第二副歌 | `repeat_of` 指向前一 chorus ∧ 位置更靠后 | **与被复制段相同**（踩音不变） | **`upgrade` 字段置位**：配置升级（单星 → 双手星）；密度只 +10%~15%（知识 031） | 5.4-7 |
| 8 | `outro` | 与 `intro` 的 SSM 相似度 > θ | **与 intro 同轨**（前后呼应） | ⚠️ **不等于减压**：官方谱末段最强（知识 031） | 5.4-8 |

**通用规则（与段落无关）**

- **切轨触发点**：`argmax_s share_s` 改变且持续 ≥ 2 小节 → 候选切轨点；`voiced_ratio` 跨越 0.5 → **强**切轨点。切轨点一律吸附到小节线（或 4 小节乐句边界）。
- **降级规则**：若选定主轨的 `n_onset_s(b) < 2` 或 `grid_fit_s(b) < 0.5` → 退回 `drums` 骨架（kick+snare）。
- **单轨警告**（反新手护栏，知识 002/005）：若全曲 `primary_stem` 唯一值只有 1 个且曲长 > 90 s → **报警**（这正是 MMFC 批评的"哪个最响就一直踩哪个"）。
  ⚠️ 已知例外：MMFC 5.6 提到《金星》整张谱只踩一条音轨且评价很高 → **这是警告不是硬错误**。

### 5.3 决策步骤（规则化，可被 LLM 执行也可被程序执行）

1. **结构映射**：按段落类型给强度基线，与强度曲线交叉验证（明显不符时以曲线为准并标人工复核）。
2. **踩音目标选择（切轨）**：按 §5.2 规则表逐段选主踩音轨 + 切轨点。
3. **密度分配**：强度档位 → `density_norm = floor + (1−floor)·intensity_norm`（§4.7）→ 分位映射到 NPS（§4.8 B）→ 分音密度与每小节 note 数；总量控制在知识 004 的区间内。**密度基调按约每 12 小节一个台阶规划**（知识 031）。
4. **配置强度升级手段**（知识 003）：同踩音下用 note 种类（tap→hold→each→slide）、位移、错位微调强度；**第二副歌优先用配置升级而非加密**（知识 031"强度≠密度"）。
5. **休息与可玩性检查**（知识 012–015）：Hold 尾留空、叠键/外键/撞尾红线、Touch 不适用（ST 谱）。
6. **输出 song sheet 双格式**（§5.4）。

### 5.4 输出：song sheet 双格式（R18，替换 v1.0 的 charting_plan.json）

**双格式同源**：`song_sheet.md`（给 LLM 读）与 `song_analysis.json`（给程序/校验器读）由同一份数据渲染，**禁止两边手改**。

**为什么是"逐小节逐 stem 网格串"**：① 与 simai 同构——LLM 看到的 `x...x...x...x...` 和它要写的 `1,,,,5,,,,` 在时间轴上一一对应，不需要心算换算（最大收益）；② 符合 ChatMusician 的结论（文本原生 + 高压缩 + 显式重复结构）⚠️；③ 天然可视，人也能一眼复核。
（反例：LLark 把音频 embedding 直接投给 LLM ⚠️ → 与本项目 AGENT §8 路线相反，不采纳。Loop Copilot 的"Global Attribute Table"⚠️ → 对应我们的 Header + Section table，采纳。）

**四条硬约束**

- **(a) 网格分音跟随该小节的量化结论**，不能固定 16；每行必须显式写 `div=16`，否则 LLM 会按 16 分理解一个三连小节。
- **(b) 每行自包含**：`bar 号 | 段落标签 | 强度档 | div | 各轨网格串`。LLM 处理长表格时会丢上下文，绝不能靠"上一行说过"。
- **(c) 字符集与制谱直觉一致**：`x` = onset（该踩的音）、`X` = 重音/强拍、`-` = 延音持续（hold 候选）、`.` = 空。
- **(d) 轨数上限 4**：kick+snare 合并为 `drum`、`vocal`、`bass`、`other/hook`。3 分钟 / BPM 180 ≈ 135 小节 × ~100 字符 ≈ 14 K 字符，可接受；铺 8 条轨会诱导 LLM 采密（违反知识 005）。
- 另：Header 必须**显式声明"网格是候选池不是谱面"**，并给出目标 note 总数区间（知识 004）。

**样例**

```markdown
# SONG SHEET — <title>
## 0. Header
bpm: 180 (const)   offset(&first): 1.234 s   bars: 1..136   time-sig: 4/4
offset_check: φ* = +4 ms (OK, method=onset-fit)   bpm_drift: 1.2e-5 /s (OK)
target: ST Master 定数 13.5 → note 总数 700–850（知识 004）；note/小节 ≈ 9.15（知识 031）
分音使用统计: {8}=61%  {16}=33%  {12}=4%  {24}=2%
⚠️ 下面的网格是**候选池**，不是谱面。按知识 005「删到不能再删」筛选。
⚠️ 高 BPM 下 {16}/{24}/{32} 的判定不可靠，网格串只当节奏轮廓用。

## 1. Sections
| bars   | 段落 (日/英)        | intensity | 建议密度 | 主踩    | 副踩        | repeat_of | 备注 |
|--------|--------------------|-----------|---------|---------|-------------|-----------|------|
| 1–8    | イントロ (intro)     | 0.18      | {8}     | other   | —           | —         | 哪个响踩哪个 |
| 9–24   | Aメロ (verse)       | 0.42      | {8}     | other   | vocal(尾拍) | —         | 人声长音 |
| 25–32  | Bメロ (pre_chorus)  | 0.31→0.78 | {8}→{16}| drums   | vocal       | —         | kick 在 29 小节消失；31–32 切人声 |
| 33–48  | サビ (chorus #1)    | 0.91      | {16}    | vocal   | kick(双押)  | —         | voiced 0.82 |
| 49–56  | 間奏 (interlude)    | 0.55      | {8}     | vocal采样| —          | —         | 强度靠位移（知识 003-2） |
| 57–64  | 落ちサビ (quiet_ch.)| 0.30      | {8}     | other   | —           | —         | **休息段** rest:true（降档不清零） |
| 65–80  | ラスサビ (chorus #2)| 1.00      | {16}    | vocal   | kick        | 33–48     | **upgrade: 单星→双手星**；密度仅 +12% |
| 81–88  | アウトロ (outro)     | 0.55      | {8}     | other   | —           | intro     | 前后呼应；**不减压**，收尾只在最后 1–2 小节 |

## 2. Bars（逐小节）
bar 33 | サビ | I=0.90 | div=16 | drum  X...x...X...x...
                                | vocal .x.x..x..x.x....
                                | bass  x.......x.......
                                | hook  ................
       | note: crash@1拍(段落起点); vocal 入
```

**JSON 侧**（`song_analysis.json`，供校验器/生成器）
- `sections[]`：在 v1.0 `charting_plan` 字段基础上增加 `chorus_index` / `repeat_of` / `upgrade` / `rest` / `labels_ja`；
- 新增 `bars[]`：`{bar, div, intensity, onsets: {drum:[idx...], vocal:[...], bass:[...], hook:[...]}, notes:[...]}`；
- 🧪 原型 v0.1 已产出的 `analysis.json` 字段（`offset_check` / `quantize_stats` / `structure.alt_segments` / `timings_sec` / `tools` 等）保留，作为可复核证据链。

## 6. 与项目现有资产的对接

- **知识库**：001（强度贴合情绪，**已于 2026-09-11 修订**）、002（切轨）、003（配置强度四因素）、004（难度 NPS 校准）、005（采音要简）、012–015（可玩性约束）、016（ST 要素黑白名单）、**031（官方谱密度曲线实测规律，本管线 §4.7 的全部数字出处）**。
- **数据**：`resource/official-chart/` 388 个官方 ST 谱（定数 13.0–14.5）——已用于 §4.8(A)(C) 的无音频标定；若日后获得对应音频即可跑 §4.8(E) 的配对标定。
- **工具**：`tools/chart_analysis/`（simai 解析器 + 密度统计，是官方谱侧 `F_chart` 的来源）、`tools/audio_analysis/`（本管线 L1–L3 的原型实现，见 §11）。
- **预览**：本地预览库（http://127.0.0.1:8099/list.html）用于结构/强度标注的可视化复核。

## 7. 工具选型总表、许可证策略与 prior art

### 7.1 四层主线（全 MIT/Apache/ISC）

| 层 | 主线 | 许可 | 备选/打底 | 排除（理由） |
|----|------|------|-----------|-------------|
| L1 节拍 | **beat_this**（MPS 走 Python API） | MIT ✅ | librosa 1.0；all-in-one-infer | madmom（模型 NC + py3.12 装不上）、essentia/tempo-cnn（AGPL） |
| L2 分离 | **Demucs v4 htdemucs / htdemucs_ft** | MIT ✅ | demucs-mlx / mlx-audio-separator（MIT） | Spleeter、BS-RoFormer（权重不明）、AudioSep、umxl（NC） |
| L2 人声 note | **basic-pitch** ⚠️ 🧪 **当前装不上，暂缺席** | Apache-2.0 ✅ | RMVPE（Apache-2.0）；**当前降级 = vocals stem onset** | CREPE（慢）、Melodia（AGPL）、pYIN（GPL） |
| L2 鼓件 | **三频带启发式（零模型）** | — | drumsep（代码 MIT，权重待核 ❓） | **ADTOF（CC BY-NC-SA ✅）、LarsNet（权重 NC ⚠️）、omnizart（连坐 madmom ✅）全部排除** |
| L3 结构 | **`all-in-one-infer` 3.1.0**（PyPI 名，MIT ✅，py3.12 ✅，natten 可选，🧪 可直接吃已分离 stems） | MIT ✅ | 自研 SSM 退路（必须有）；SongFormer（非商用分支） | 原版 `allin1`（Darwin 强制依赖 natten；import 需 madmom）、MERT/music2vec（NC）、DeepChorus（无 LICENSE）、msaf（低维护） |
| L3/L4 强度 | **librosa + pyloudnorm 配方**（§4.5） | ISC ✅ | — | **pop-music-highlighter（GPL-3.0 ✅）**、madmom features（NC）、pychorus（弃用） |

### 7.2 许可证策略（开源项目红线）

- 默认依赖只允许 **MIT / Apache-2.0 / ISC / BSD**；**AGPL/GPL 与 CC BY-NC(-SA) 权重一律不进默认管线**（可作为个人非商用实验的 optional 插件）。
- **❓ 待用户决策：CC BY 4.0 是否纳入白名单**（影响 SongFormer 代码/权重）。注意即便放开，SongFormer 的前端 **MuQ 权重仍是 CC BY-NC-4.0** ✅，所以它**只能进个人非商用分支**；其另一半前端 MusicFM 是 MIT+Apache-2.0 ✅。
- **madmom-infer 运行时下载风险** ✅：`all-in-one-infer` → `madmom-infer`（BSD-2 ✅），后者 `models.py` 声明会运行时从 `CPJKU/madmom_models` 下载 **CC BY-NC-SA** 权重。🧪 **实测已跑一次并检查 `~/.cache/madmom_infer/models` 为空**（all-in-one 只用纯 HMM 的 DBN 解码器）→ 当前未触及 NC 权重；**对外发布前仍需复核一次**，若非空则只能走 `--no-allin1` 的纯 SSM 路线。
- 待复核项 ❓：MDX23C/Kim_Vocal_2 权重（分发源无 LICENSE）、drumsep 权重、miaChartGen2 的 HF 权重。
- 已明确排除并记录：**ADTOF（CC BY-NC-SA 4.0 ✅）、LarsNet（`license: None` ✅ / README 称 NC ⚠️）、omnizart（requires_dist 含 madmom ✅）、pop-music-highlighter（GPL-3.0 ✅）**。

### 7.3 本机环境速查（M4 / 16 GB / macOS 26 / py3.12 / torch 2.14）

🧪 实测装上且可用：`librosa 1.0.0` / `torch 2.14.0`（MPS 可用）/ `demucs 4.1.0` / `all-in-one-infer 3.1.0` / `natten`（源码编译通过）/ `pyloudnorm` / `matplotlib` / `scipy` / `numpy 2.5.3` / `pytest`。
🧪 实测装不上：`basic-pitch[onnx]`（pip `resolution-too-deep`）；`allin1` 1.1.0（装上但 import 失败）。
⚠️ 其他一手事实：`librosa 1.0` requires_python ≥ 3.12 且已移除 `beat.tempo` ✅；`torchaudio ≥ 2.11` 解 mp3 需 `torchcodec` ✅（已被 §2.1 统一转 WAV 绕开）。

### 7.4 prior art

- **Goldgom/miaChartGen2**（gitcode 镜像存活；GitHub 原仓已 404）：检索到的**唯一 maimai 专用"MP3→simai"生成式项目**。EnCodec 24 kHz token + BeatTokenizer（librosa/beat_this 双后端）+ simai tokenizer → 75 Hz 帧网格 → **5-Stage 级联 Transformer**（骨架 → Hold 时长 → Slide 路径 → Break → EX）→ 密度/类型偏置采样。**无 LICENSE、README 声明"仅学术/个人学习"、不含 checkpoint** → 只借鉴设计思路，不取代码。
- **DDC**（MIT）：⚠️ **事实更正（R22）——DDC 不做音乐量化**（伪造 BPM=125 让 1/192 小节 = 10 ms 帧），它**不是 snap 逻辑的参照**；可抄的是 `BeatCalc`/`abstime` 的时间轴换算结构。
- **Yi 2023**（arXiv 2311.13687）：1/48 拍 beat-aligned 频谱、每 2 拍 96 位置 token、osu!mania 14648 谱 / 3166 曲、micro-F1 84.6% ✅ → **本项目"拍同步重采样"的直接依据**（§2.5 Step 1）。
- 其他参照：ARG/softchart-v15（MIT，7.94M 小模型）、TaikoNation（乐句模式=结构分段）、audio2chart、Mug-Diffusion、ChartGenEval；LLM 侧 ChatMusician / LLark / Loop Copilot / MusicAgent（§5.4 已引）。
- **我们的差异化点**（经全库检索确认）：现有开源项目**没有任何一个做"显式声源分离 → 分轨踩音"**；结构分段（日式双标签 + chorus_index）+ 强度曲线 + 官方谱密度先验 + 知识库规则校验是四层相对 prior art 的增量。

## 8. 技术路线决策：显式特征分析 vs 端到端（2026-09-10，用户决策，v1.1 未变更）

- **用户判断**："全频谱直接进 transformer 有点扯；对 mp3 进行分析，寻找采音、分段以及情绪强度才是正确的路径。"
- **决策**：**特征分析先行、生成器后置**——L1–L3 确定性提取，L4 规则+规划（注入知识库硬约束），生成端只消费结构化特征；不采用 EnCodec 全频谱 token 端到端路线。
- **理由**：① 硬约束可保证（零报错/可玩性/无理红线必须显式校验）；② 可解释可复核（每层输出可人工查看修正）；③ 数据效率（端到端需海量谱面，本项目仅 388 官方 ST 谱）；④ 幻觉可控（LLM 只接触结构化特征）；⑤ 差异化（无人做分轨踩音）。
- **保留的开放点**：生成端可采用"以显式特征为条件的小型生成模型"（借鉴 miaChartGen2 的分阶段骨架→属性设计，输入换成我们的特征），仍属显式分析路线，并保留规则硬校验兜底。

## 9. 存疑 / 待定（v1.1 刷新）

> 已解决项已从 v1.0 清单删除：域外数据"无处可寻"（→ §10 已有 RWC-Pop + osu2beat2025）、CPU/设备 RTF 未知（→ §1 已有实测）、NATTEN 自编译坑（→ 实测编译通过）、madmom NC 权重风险（→ 实测缓存为空）。

**A. 最不可靠的四个环节（🧪 原型实测，按可靠性由低到高）**

1. **高 BPM 下的分音选择近乎随机**：192 BPM 时 24 分与 32 分格子只差 ~13 ms，远小于容差 → 16/24/32 之间基本是随机挑；`unresolved_bars=0` 不代表量化准确（§2.5.3）。这是全管线第一不可靠环节。
2. **结构标签必须人工对图复核**：两条分段路径都会退化（all-in-one 曾只给 3 段、最长段占 63%；SSM 第一版把 56 个小节并成一段），已加"段数 ≥5 且最长段 ≤35%"体检 + SSM 自适应重试兜底（§4.2）。
3. **人声 VAD 受 synth 泄漏影响**：Demucs vocals 轨常混进 lead synth，器乐段被误判为有人声（§3.4）。
4. **鼓件分解是频带启发式不是转录**：密集段里 kick 与 hihat 的网格串常常完全一样（§3.3）。

**B. 未标定 / 未实现**

5. **权重全部未标定** ❓：强度融合 `w=[0.25,0.30,0.20,0.15,0.10]`、投票 `[0.45,0.25,0.20,0.05,0.05]` 都是设计初值；真正标定需 §4.8(E) 的官方音频配对数据，在此之前只能靠 (D) 人工听审做排序检验（目标 Spearman ρ ≥ 0.8，❓ 未做）。
6. 量化容差三参数（`0.25 / 12 ms / 30 ms`）与三连阈 `0.7` ❓ → 在 osu2beat2025 子集上标定。
7. swing 在 160–220 BPM 音游曲里到底出不出现 ❓ → 默认关闭；先统计目标曲库 offbeat 相位分布。
8. 强度配方 v2（§4.5 新融合式）与 §4.7 的密度映射 **尚未实现**；L4 规划层整层未实现（§11）。

**C. 域与数据**

9. **域外泛化**：所有模型基准都是西方流行乐；🧪 原型只跑了 3 首日系曲，样本远不够 → 按 §10 的 A/B 档做 20–50 首人工核对。
10. **器乐曲（无副歌）**："副歌=重复最多段"失效，高潮 fallback 为能量峰值（能量票权重升到 0.5）。
11. **官方 maimai 音频**：❌ 当前无数据，配对标定（§4.8 E）**挂起**；一旦用户能提供 10–20 首立刻可跑。
12. **密度先验的样本边界**：知识 031 只覆盖 ST 谱定数 13.0–14.5，12 级及以下与 DX 谱外推需谨慎。

**D. 许可与工程**

13. **CC BY 4.0 是否纳入白名单** ❓ **需用户决策**（影响 SongFormer 能否进管线；即便放开其 MuQ 权重仍是 NC）。
14. drumsep 权重许可 ❓；MDX23C/Kim_Vocal_2 权重 ❓；miaChartGen2 HF 权重 ❓。
15. basic-pitch 装不上的问题需解决（或换 RMVPE）；三后端数值一致性 ❓。
16. SongFormer 能否在 M4 MPS/CPU 上跑 ❓（官方只给 cuda 示例）。
17. 变速（`--bpm-changes`）仍是实验性：网格只在小节边界换 BPM，变速点不落小节线则其后全错 ❓。

## 10. 评测数据与协议（v1.1 新增，R23）

> 原则重申：**谱面设计参考只用 `resource/official-chart/` 的 388 个官方 ST 谱**。下面的外部数据集**只用于音频分析层（节拍/offset/分段/副歌）的工程验证**，不作为谱面设计基准。

### 10.1 RWC-Pop（2026-02 起已开放在线下载）

- AIST 官网 ✅：**"The RWC Music Database has been available for online download since February 2026."**，指向 `https://zenodo.org/communities/rwc-music/`（DOI 10.5281/zenodo.18656623，**CC BY-NC 4.0**，`RWC-P.zip` 4.07 GB）；标注在 `github.com/rwc-music/rwc-annotations`。
- 内容 ✅：Popular 100 首 = **日语歌词的日本流行乐 80 首** + 英语 20 首；AIST Annotation 提供 **beat structure、melody line、chorus sections、结构段落标签**（intro / verse A / verse B / pre-chorus / chorus A / chorus B / post-chorus / bridge / ending）。
- **这是目前唯一一个"日语流行 + 结构/副歌/节拍标注 + 音频可合法获取"的数据集**，直接解决 v1.0 §9.1 的"域外泛化未知"。
- ⚠️ 限制：CC BY-NC 4.0 → **不得入仓库、不得商用、不得再分发**；本机评测用途没问题。
- 对标基线：RefraiD 在 RWC-Pop 上 100 中 80 正确、F=0.938 ⚠️。

### 10.2 osu2beat2025（Osu2MIR, ISMIR 2025 LBD）

- 仓库 `ziyunliu4444/osu2mir` ✅（无 LICENSE 字段 ❓；论文 CC BY 4.0）。
- 内容 ✅：**741 条用户标注 / 708 首不同音频**的 metered beats（含 downbeat），文件名带 MD5 + BeatmapSetID；**音频不随附**，需自行从 `osu.ppy.sh/beatmapsets/<id>` 下载 `.osz` 解出 mp3 并校验 MD5（合法免费）。
- **曲风正是本项目的域**（anime、Vocaloid、video game music）⚠️。
- 质量筛选经验（可直接复用）✅：只用"单个未继承 timing point"或"多点间隔 ≥ 5 s"的图。
- 用途限定：验证 §2.4 的 offset 相位搜索、BPM 漂移检测、§2.5 的量化残差、beat F1。**不涉及谱面设计**（osu! 谱是社区自制谱）。

### 10.3 三档评测协议

| 档 | 目的 | 数据 | 指标 |
|---|---|---|---|
| **A** 日语结构/副歌 | 验证分段与"副歌=重复最多段" | RWC-Pop 80 首日语（本机，NC） | 边界 HR.5F / HR3F、chorus 段 IoU、副歌命中率（对标 RefraiD 80/100） |
| **B** 域内节拍/offset/量化 | 验证 §2.4 相位搜索与 §2.5 量化 | osu2beat2025 的"单 timing point"子集 | `|φ*|` 分布、BPM 漂移估计误差、beat F1(70 ms)、量化残差分布 |
| **C** 谱面密度基准 | 难度/强度校准 | **仅 388 官方 ST 谱**（无音频） | `T_density`、分位映射表、chorus/verse 比值（§4.8 A/B/C）——**已完成，见知识 031** |

### 10.4 明确不投入的数据

- **BOF / BMS 曲包**：① 逐年逐作者授权条款不一致 ❓；② 谱面是社区自制谱 → 按用户规矩不作谱面参考；③ 音频侧价值已被 osu2beat2025 更干净地覆盖。→ **不投入**。
- **Harmonix Set**：不含音频，只能当标签体系参考（MIT ✅）。**SALAMI**：曲风杂、音频需自取，优先级低。**SongFormBench**：CC BY 4.0 ✅ 但**无日语**（语言字段 `["en","zh"]` ✅）。

## 11. 当前实现状态（v1.1 新增）

### 11.1 `tools/audio_analysis`（原型 v0.2，2026-09-11；v0.1 见 note 035，v0.2 见 note 038）

用法：`python -m tools.audio_analysis --audio <mp3> --bpm <BPM> --first <秒> [--level 13.5] --out out/<song>/`
输出：`track.44k.wav` / `stems/*.wav` / `song_analysis.json` / `song_sheet.md` / `plot.png`（旧名 `analysis.json` / `song-sheet.md` 同时写出保持兼容）。
测试：`tests/test_audio_analysis.py` 65 用例，**全部用 numpy 合成音频**（仓库不放真实音频，AGENT 准则 5）。

**已实现**

| 模块 | 对应本文 | 状态 |
|---|---|---|
| `decode.py` | §2.1 统一解码 + ffprobe `start_time` | ✅ |
| `grid.py` | §2.2 网格构造（含负 first、实验性变速）+ §2.4 offset 校验（onset-fit，整拍歧义分组取峰） | ✅ |
| `stems.py` | §3.1 Demucs 四轨（MPS 优先、失败回退 CPU） | ✅ |
| `onsets.py` | §3.2 逐 stem onset + §3.3 鼓件频带启发式 + §3.4 人声 VAD | ✅（零模型启发式） |
| `quantize.py` | §2.5 全套：1/48 拍重采样、**每小节单一分音**、`τ=clamp(0.25·slot,12,30)`、三连门、**`{32}` 红线**、未落格 onset 如实统计 | ✅（实测未落格 7–19%，见下） |
| `features.py` | §3.2 逐小节特征清单（share / n_onset / grid_fit / voiced_ratio / kick-snare-hihat / riff_sim / sil_run） | ✅ |
| `structure.py` | §4.2 三路边界投票（all-in-one-infer + novelty + 重复段，吸附小节线）+ §4.3 日式(英文)双标签 + §4.4 `pre_chorus` 派生 + `chorus_index` / `repeat_of` / `upgrade` / `rest` / `final_chorus` / `quiet_chorus` / 器乐向 `drop` | ✅（标签须人工对图复核） |
| `intensity.py` | §4.5 五项融合式 + §4.6 五票 + §4.7 密度地板映射（`--level` 取官方均值 note/小节） | ✅（权重为初值） |
| `stemplan.py` | §5.2 八条「段落×特征→主踩音轨」规则 + 切轨触发 / 降级 / 单轨警告 | ✅ |
| `tracks.py` / `sheet.py` | §5.4 song sheet 双格式：4 轨 `drum/vocal/bass/hook`、字符集 `X x - .`、行自包含、Header 硬约束与分音统计 | ✅ |

**v0.2 实测（3 首测试曲，BPM 148–192）**：`{32}`/`{24}` 从输出中消失（红线拦下 9–21 小节/曲）；分音以 `{16}` 为主（92–100%）；**未落格 onset 6.9% / 13.1% / 18.6%**——v0.1 报的"全部可解"是容差过宽造成的假象；78–87% 的小节走"找不到合法分音、压到上限"分支，因为四轨并集每小节中位 15–22 个 onset、非鼓轨定位误差 15–33 ms（上游解法是 basic-pitch 给出人声 note 事件，不是放宽容差）。强度曲线 v0.1 的 raw/smoothed 偏离根因是**双重归一**（bar_raw 未归一、平滑后又 min-max），已修并加回归测试。

**未实现清单**

- §4.8(B) 分位映射 `F_chart⁻¹∘F_audio`（当前只有 §4.7 的地板线性映射）；(D) 人工听审校验；(E) 配对标定（无官方音频）；
- basic-pitch 人声 note 事件（§3.5，`resolution-too-deep` 装不上）；SongFormer 二号意见（§4.2）；
- 器乐向阈值（voiced 能量占比 0.18）、chorus 护栏、器乐曲强度分档阈值只在 3 首上试过，无实证。

### 11.2 `tools/chart_analysis`（已完成，2026-09-11）

simai 解析器 + 逐小节密度统计 + 变点分段 + k-means + 五段模板检验；`validate/summary/analyze/chart` 四个子命令；`tests/test_chart_analysis.py` 42 用例。
**用途**：官方谱侧的 ground truth 提取器——§4.7 的 `T_density`、密度地板、六条密度规律、§4.8(B) 的 `F_chart` 全部由它产出。解析覆盖率 387/388 = 99.74%，与 manifest 官方 note 总数完全一致 376/387 = 97.16%（|Δ| ≤ 3 占 100%）。
