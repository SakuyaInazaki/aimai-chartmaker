# 调研报告 v2：音频分析落地问题（onset 量化 / 结构分段 / 强度标定 / 切轨代理 / LLM 表示 / 本机可行性 / 评测数据）

> **日期**：2026-09-11　**执行**：coding agent（子代理），受主会话委托
> **定位**：v1.0（`docs/research/audio-analysis-research.md`）解决的是"用什么工具"，本轮解决的是"**已知 BPM/offset 的前提下，怎么把它做对**"。不重复 v1 的工具选型对比与许可证核实，只在**与 v1 结论冲突**或**v1 未覆盖**处展开。
> **前提约束**（用户 2026-09-11 明确）：
> - 用户**直接提供 BPM / offset(`&first`) / 变速点**，节拍追踪不是重点；
> - **谱面参考只看官方谱**；用户自制谱不作为谱面参考或评测 ground truth（其 mp3 仅可作测试音频）；
> - 用户**没有官方谱对应的 mp3** → "音频 vs 官方谱密度"的直接配对标定**目前无数据**。
>
> **置信标记**：✅ 一手核实（源码 / LICENSE 原文 / PyPI·GitHub·HF API / 论文原文）｜⚠️ 部分核实（二手摘要，未见原始数字）｜❓ 未确证（推理或需实测）

---

## 1. onset → 拍网格量化（simai 分音）

### 1.0 前置事实

- simai 时间模型：`{x}` 表示"一个逗号 = x 分音符"，**槽长 = 240 / BPM / x 秒**；x 必须是 384 的约数（4/8/12/16/24/32/48/64/96/192/384 合法）。✅（`docs/simai-syntax.md` §1.4、§3.2）
- 每小节槽数 = x/4（`{16}` = 每小节 16 槽 = 每拍 4 槽）。
- 目标 BPM 区间 160–220 下的槽长（自算）：

| BPM | {4} | {8} | {12} | {16} | {24} | {32} | {48} |
|-----|-----|-----|------|------|------|------|------|
| 120 | 500.0 | 250.0 | 166.7 | 125.0 | 83.3 | 62.5 | 41.7 |
| 160 | 375.0 | 187.5 | 125.0 | 93.8 | 62.5 | 46.9 | 31.2 |
| 180 | 333.3 | 166.7 | 111.1 | 83.3 | 55.6 | 41.7 | 27.8 |
| 200 | 300.0 | 150.0 | 100.0 | 75.0 | 50.0 | 37.5 | 25.0 |
| 220 | 272.7 | 136.4 | 90.9 | 68.2 | 45.5 | 34.1 | 22.7 |

（单位 ms）

- **检测精度的天花板**：`mir_eval` 的行业标准容差——**onset F-measure 窗 = ±50 ms**（`mir_eval/onset.py: f_measure(..., window=0.05)`）✅、**beat F-measure 阈 = 70 ms**（`mir_eval/beat.py: f_measure_threshold=0.07`）✅。
- **直接推论（关键）**：BPM 180 时 `{32}` 槽长 41.7 ms、BPM 220 时 34.1 ms，**都小于通用 onset 检测的标准容差窗**。也就是说"自动把 onset 量化到 {32}"在本项目的目标 BPM 区间里**在信息论上就站不住**——它测不出来。`{24}`（45–62 ms）同样处在噪声边缘。

### 1.1 结论：量化规则（可直接实现）

**Step 0 — 统一解码（必须）**
一律 `ffmpeg -i in.mp3 out.wav` 转 PCM，**所有分析环节只读这一个 WAV**。
依据：all-in-one-infer 作者在 README「Concerning MP3 Files」中实测——"**不同 MP3 解码器会有约 20~40 ms 的偏移差异**，而 beat tracking 的常规容差只有 70 ms" ✅。osu! 官方 timing 指南同样写明"**任何音频编辑（包括重新编码）都会改变 timing**" ✅。
→ v1 §2 已写"MP3 先转 WAV"，本轮补上**理由的一手出处**与"分析全程只用同一个 WAV"的强化要求。

**Step 1 — 构造网格**
由用户的 BPM 段列表 + offset 构造 `t(bar, div, idx)`；变速点按分段线性（每段常 BPM）处理。参考实现范式：DDC 的 `BeatCalc`（分段 bps 线性积分，`beat_to_time` / `time_to_beat` 双向）✅——这是最简洁的可抄结构（MIT）。

**Step 2 — 拍同步重采样（对 v1 的方法论修订）**
不要"固定 hop 提特征 → 再去找最近的格子"，而是**把 onset 包络按拍重采样**（beat-synchronous）：以"每拍 N 个采样点"重采样，N 取 48（= 每小节 192 点，能同时整除 2 分族和 3 分族）。
依据：目前最新的音游谱生成工作（Yi, *Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts*, arXiv 2311.13687）就是把 log-Mel 频谱按"**hop = 1/48 拍**"提取，事件用"每 2 拍 96 个时间位置"的 token 表示 ✅。这条工程实践与 simai 的 384 分约数体系天然兼容（每拍 48 = `{192}`）。

**Step 3 — 逐小节推断最小分音（奥卡姆 + 知识 005「采音要简」）**
对第 b 小节内的 onset 集合 `{p_i}`（p = 小节内相对位置 ∈ [0,1)）：

```
候选族：
  二分族 D2 = [4, 8, 16, 32]
  三分族 D3 = [12, 24]          # 12 = 每拍 3 连；24 = 每拍 6 连
容差：τ(d) = clamp(0.25 × slot_ms(d), 12 ms, 30 ms)
  0.25 的理由：残差超过槽长 1/4 就越过"归属歧义带"，可能落到邻槽
  30 ms 上限：不超过 onset 检测本身的可信精度（标准窗 50 ms 的一半量级）
  12 ms 下限：低于此已是解码/包络分辨率噪声

从粗到细扫描 d ∈ [4, 8, 12, 16, 24, 32]：
  r(d) = max_i  min_k |p_i − k/d| × bar_ms      # 最大残差
  第一个满足 r(d) ≤ τ(d) 的 d 即为该小节的 {x}
```

**Step 4 — 三连 / 二分判别**
分别用 d=16 与 d=12 拟合，比较 RMS 残差 `rms16`、`rms12`。
判为三连需**同时**满足：`rms12 ≤ τ(12)` **且** `rms12 ≤ 0.7 × rms16`。
先验理由：音游曲绝大多数是二分体系，三连是少数派，需要显著证据才推翻默认。

**Step 5 — swing 判别（独立于三连）**
取该段所有"反拍"onset（最近网格点 k 为奇数的 8 分位置），计算 `s = median((p_i − p_onbeat)/ (1/8))`。
- `s ≈ 0.5` → 直（straight 8th）
- `s ≈ 0.667` → 三连 swing → 该段改用 `{12}`/`{24}`
文献口径：swing factor 1 = 直，2 = "tied-triplet feel"；且 **swing ratio 随 tempo 升高近似线性下降**（>150 BPM 尤其明显）⚠️（Dittmar/Müller 等，ISMIR 2015 "Automated Estimation of Ride Cymbal Swing Ratios"；swingogram 表示）。
→ 对 160–220 BPM 的音游曲，swing 出现概率低且幅度小，**默认关闭 swing 检测**，仅在 `rms12 << rms16` 且 s 稳定落在 0.6–0.72 时启用。

**Step 6 — {32}/{48} 红线**
默认**禁止**自动量化产出 `{32}` 及更细。允许的唯一例外需三条同时成立：
(a) 该小节 onset 数 ≥ 6；(b) 来源是 **drums stem**（瞬态最锐，定位最准）；(c) `{32}` 下 RMS 残差 ≤ 15 ms。
制谱侧理由同样支持这条：`{32}` 在 ST 谱里属于装饰性 burst，应当由 LLM 按**配置需要**主动使用（知识 003/015），而不是由量化算法"检测"出来。

**Step 7 — 输出**
每个 onset → `{bar, div, idx, t_ms, residual_ms, stem, strength, confidence}`；
每小节 → `{bar, div, n_onset_per_stem, fit_rms_ms}`。

### 1.2 用音频反向校验用户给的 offset（网格相位搜索）

这是 v1 完全没有的环节，**成本极低、收益极高**（offset 错 20 ms 全谱皆错）。

```
输入：用户 BPM/offset、统一解码后的 WAV
1) 用 drums stem（或全曲）算 onset strength envelope O(t)
   hop 必须细：sr=44100, hop=256 → 5.8 ms/帧（v1 的 hop=512@22050 = 23 ms 太粗，
   量级与待测偏移相当，会把答案磨平）
2) 取用户网格上的 {8} 点集合 {t_k}（{8} 比 {16} 鲁棒；也可只用 downbeat）
3) 对 φ ∈ [−beat/2, +beat/2)，步长 1 ms：
       S(φ) = Σ_k O(t_k + φ)        # 亚帧处用线性插值
   φ* = argmax S(φ)
4) 判据：
   |φ*| ≤ 10 ms         → 用户 offset 可信，直接用
   10 ms < |φ*| ≤ 30 ms → 提示"疑似解码偏移"，记录但仍以用户值为准（AGENT 准则：用户优先）
   |φ*| > 30 ms         → 报警：offset 或 BPM 可能有误，交人工确认
5) BPM 交叉校验（漂移检测）：
   把全曲按每 16 小节分块，逐块各求 φ_i；对 (块中心时间 t_i, φ_i) 做线性回归。
   斜率 a ≠ 0 表示网格在漂 → BPM_true ≈ BPM_user × (1 − a)
   （a 的量纲是"秒/秒"，即累计漂移速率）
   |a| < 2e-4（3 分钟累计 < 36 ms）视为无偏
6) 独立第二意见：librosa.beat.plp(onset_envelope, prior=围绕用户 BPM 的窄正态)
   → PLP 脉冲曲线的峰位相位，与 φ* 对照。
   （librosa 1.0 的 beat 模块 __all__ = ["beat_track", "plp"] ✅ 一手核实，plp 仍在；
     注意 librosa 1.0 已移除 librosa.beat.tempo）
```

**为什么这套是对的**：它就是 beat tracking 里"相位估计"这一步的最小化版本——tempo 已知（用户给），只剩相位未知，退化成一维网格搜索，几乎不可能出错。旁证：Osu2MIR（ISMIR 2025 LBD）做的正是反向工程——把 osu! 谱面的未继承 timing point 当作 beat/downbeat ground truth，并发现"**单个未继承 timing point、或多点间隔 ≥ 5 s 的图，标注可靠；间隔 < 5 s 的需要额外筛查**" ✅。这反过来说明：**人工确定的 BPM+offset（恒速）是可以当 ground truth 用的**，我们让用户提供 BPM/offset 的路线在方法论上是站得住的。

### 1.3 成熟做法的取舍（哪些不能抄）

| 来源 | 实际做法 | 对我们的价值 |
|------|----------|--------------|
| **DDC**（chrisdonahue/ddc，MIT） | **根本不做音乐量化**：推理端写死 `_SUBDIV=192, _DT=0.01, _BPM=60×100/192×4=125`，即**伪造一个 BPM 让 1/192 小节恰好等于 10 ms 帧**，输出的是"时间均匀网格"的 .sm ✅（`infer/ddc_server.py`） | ❌ 无 snap 逻辑可借鉴。v1 把 DDC 列为 prior art 没错，但它**不是量化参照** |
| **DDC**（`learn/beatcalc.py`, `dataset/smdataset/abstime.py`） | 分段 BPM/stop → 时间的双向换算 | ✅ 时间轴换算的干净范式，可直接照抄结构（含变速/停顿） |
| **osu! 编辑器** | beat snap divisor 共 11 档 1/1…1/16，含 1/3、1/6（三连/六连）；timing 靠人工敲 T 键 + 节拍器逼近，反复微调 offset 与 BPM ⚠️（osu! wiki） | ✅ 佐证"分音候选集要同时含二分族与三分族"；❌ 其 snap 是人工行为，无算法 |
| **MIDI 量化文献** | 规则法 → HMM/metrical HMM（Nakamura 等，可保证不出现"不完整三连"这类语法错误）→ Transformer（arXiv 2508.19262，给定 beat/downbeat 标注做量化，ASAP onset F1 97.3%，MUSTER 指标 SOTA）⚠️ | ✅ **思路**可借：（a）"网格语法合法性"应作为硬约束（simai 的 384 约数就是这个）；（b）最新工作正是"**已知 beat 标注**下的量化"，与我们场景同构。❌ 但它们解决的是"表现性演奏时值"的难题，我们的输入是**机械制作的电子乐 + 已知恒定网格**，规则 + 残差判据足够，上 HMM/Transformer 是过度设计 |
| **Yi 2023**（arXiv 2311.13687） | hop = 1/48 拍的 beat-aligned 频谱；osu!mania 14648 谱/3166 曲（过滤后 6781/2004，训练 5494 谱/1598 曲/73 小时）；micro-F1 osu!mania 84.6% ✅ | ✅ **拍同步重采样**这一条直接采纳（见 Step 2） |

---

## 2. 结构分段在 J-pop / ACG / 音游曲上的可用性

### 2.1 硬事实更正（v1 有误）

1. **all-in-one 的功能标签只有 8 类**（+ start/end 两个哨兵）：
   `['start','end','intro','outro','break','bridge','inst','solo','verse','chorus']` ✅（`src/allin1/config.py: HARMONIX_LABELS`）。
   → **没有 pre-chorus、没有 build**。v1 §4 写"10 类功能标签（intro/verse/chorus/bridge/outro 等）"并在 L4 §5.3 样例里直接写 `"type": "pre_chorus"`——**这个标签 all-in-one 给不出来，必须由规则派生**。这是 v1 最需要修订的一处事实错误。
2. **all-in-one 的出处是 WASPAA 2023，不是 ISMIR 2023**（arXiv 2307.16425，Kim & Nam，"All-In-One Metrical And Functional Structure Analysis With Neighborhood Attentions on Demixed Audio"）⚠️。v1 §4/§7.3 两处写 ISMIR 2023，应更正。

### 2.2 SongFormer（2025-09/10，本轮新增的最重要候选）

- 论文：arXiv 2510.02797，ASLP@NPU + HKUST + M-A-P。代码 https://github.com/ASLP-lab/SongFormer ，权重 https://huggingface.co/ASLP-lab/SongFormer 。
- **许可证：CC BY 4.0** ✅（GitHub LICENSE 原文首行 "Creative Commons Attribution 4.0 International Public License"；HF 卡片徽章同）。署名即可，**非 NC**。
- 官方 README 实测表 ✅：

| 数据集 | 方法 | ACC | HR.5F | HR3F |
|---|---|---|---|---|
| SongFormBench-HarmonixSet | All-In-One | 0.740 | 0.596 | 0.730 |
| | LinkSeg-7Labels | 0.780 | 0.630 | 0.762 |
| | Gemini 2.5 Pro | 0.748 | 0.423 | **0.813** |
| | **SongFormer (HX)** | 0.795 | **0.703** | 0.784 |
| | **SongFormer (HX+E+H+G)** | **0.807** | 0.696 | 0.780 |
| SongFormBench-CN | All-In-One | 0.834 | 0.563 | 0.771 |
| | **SongFormer (HX+E+H+G)** | **0.891** | 0.688 | 0.851 |

- **严格边界（±0.5 s）比 all-in-one 提升 ~18% 相对**（0.596 → 0.703）——对制谱是最要紧的指标，因为段落边界必须落在小节线上。
- 速度：整曲 2–4 s（NVIDIA L40），比 all-in-one 的 9–12 s 快 ⚠️（作者自测）。
- **标签词表是 Harmonix 原始细标签**（`dataset/label2id.py`，`num_classes=128`）✅，含：
  `intro / verse / prechorus / chorus / postchorus / bridge / inst / solo / outro / break / breakdown / build / interlude / quietchorus / instchorus / chorushalf / mainriff / transition / fadein / silence …`
  → **能直接给出 `prechorus` / `build` / `quietchorus`（≈落ちサビ）/ `postchorus` 这些 all-in-one 给不出的标签**，对制谱的"build 末尾切回人声"（MMFC 5.4 第 3 条）是刚需。
  ⚠️ 代价：128 类细标签在推理时是逐段 argmax（`postprocessing/functional.py`），细标签之间必然存在混淆，**需要做一层"细标签 → 我们的 9 类"的归并映射**（映射表见 §2.4）。
- **⚠️ 许可证隐藏成本（必须写进选型表）**：SongFormer 的前端是 **MuQ + MusicFM** 双 SSL 表征（`config.json` 里 `muq_config2.json` 与 `musicfm/` 目录）✅。
  - MusicFM（ByteDance）：MIT + Apache-2.0 双许可 ✅（HF 仓库内 `musicfm/LICENSE` 原文）；
  - **MuQ 权重：`OpenMuQ/MuQ-large-msd-iter` 标注 `license:cc-by-nc-4.0`** ✅（HF API tags）——代码仓 `tencent-ailab/MuQ` 是 MIT，但**权重是 NC**。
  → 按 AGENT.md §7.2 红线，SongFormer 这条路**不能进"默认/可商用"管线**；作为**个人非商用的第二意见插件**是合适的。
- 帧率 8.333 Hz（120 ms/帧）✅（`configuration_songformer.py: frame_rates=8.333`）→ 边界分辨率天花板就是 120 ms，**必须再做一次"吸附到最近下拍"的后处理**。

### 2.3 J-pop 上"副歌 = 重复最多段"到底成不成立

**这是本轮最有价值的一条证据**：

- Goto (2006) 的 **RefraiD**（*A Chorus Section Detection Method for Musical Audio Signals and Its Application to a Music Listening Station*, IEEE TASLP）在 **RWC-Pop 上 100 首中 80 首完全正确，这 80 首的平均 F-measure = 0.938** ⚠️（二手摘要口径，论文 PDF 在 AIST 站点可取）。
- **RWC-Pop 的 100 首里有 80 首是日语 J-pop**（"Songs with Japanese lyrics performed in the style of Japanese popular music (80 songs)"）✅（AIST 官网原文）。
- RefraiD 的核心机制就是**基于 chroma 自相似的重复段检测**（并额外处理转调副歌）。

→ **结论**：纯重复性方法（= v1 的 libfmp/SSM 支线）在**日语流行乐**上有 ~80% 级别的一手实证支持，"副歌 = 重复最多段"在 J-pop 上**基本成立**，v1 §9.1 把它列为"命中率未知"是过于保守的。
→ **但两点保留**：(a) RWC-Pop 是 1990s 风格 J-pop，与 160–220 BPM 的 ACG/音游曲仍有域差；(b) **器乐向音游曲（BOF 系、东方系、hardcore/EDM）常常没有"副歌"概念**，只有 build → drop，此时该启发式失效，必须走能量/drop 通道（见 §3）。

**没有任何 MSA 模型公布过日语曲上的分段指标** ✅（Harmonix = 西方流行；SongFormBench = en + zh，语言字段 `["en","zh"]`，**无 ja**）。→ 日语域的验证必须我们自己做，数据见 §7。

### 2.4 日式段落词汇 ↔ 谱师用语 ↔ 模型标签 映射表（建议作为分析输出的标准词表）

| 日式（谱师/作曲口语） | 英文 | all-in-one | SongFormer 细标签 | 本项目标准标签 | 制谱含义（知识 001/002） |
|---|---|---|---|---|---|
| イントロ | Intro | `intro` | `intro`/`instintro`/`fadein`/`rhythmlessintro` | `intro` | 低强度；"哪个响踩哪个" |
| Aメロ | Verse | `verse` | `verse`/`slowverse`/`instrumentalverse` | `verse` | 中强度；踩器乐，小节尾混人声 |
| Bメロ | Pre-Chorus | ❌（无） | `prechorus`/`pre-chorus`/`build` | `pre_chorus` | 爬升；**末尾切回人声铺垫** |
| サビ | Chorus / Hook | `chorus` | `chorus`/`altchorus`/`chorushalf` | `chorus` | 高强度；vocal 曲全踩人声 |
| 落ちサビ | Quiet / Breakdown Chorus | ❌ | `quietchorus` | `quiet_chorus` | **减压休息段**，密度骤降 |
| 大サビ / ラスサビ | Final / Last Chorus | `chorus` | `chorus`（靠位置判定） | `final_chorus` | 全曲峰值；配置升级（单星→双手星） |
| Cメロ | Bridge | `bridge` | `bridge`/`instbridge` | `bridge` | 转折；换轨 |
| 間奏 | Interlude / Inst. break | `inst`/`solo` | `interlude`/`inst`/`solo`/`gtr` | `interlude` | 踩采样/独奏；靠**位移**提强度 |
| ドロップ（EDM） | Drop | `chorus`/`break` | `breakdown`/`build` | `drop` | 器乐曲的"副歌"，能量票主导 |
| アウトロ | Outro / Ending | `outro` | `outro`/`bigoutro`/`vocaloutro` | `outro` | 渐弱；与 intro 前后呼应 |

（日英对应的一手依据：日文音乐圈通用对照 ⚠️——Aメロ=Verse、Bメロ=Pre-Chorus、サビ=Chorus/Hook、落ちサビ=Breakdown/Quiet Chorus、大サビ=Bridge/Shout Chorus、ラスサビ=Final Chorus、間奏=Interlude；注意日英并非严格一一对应，`Bメロ` 在部分语境被叫 Bridge，映射时**以位置+功能判定为准，不以词面为准**。）

**建议：分析输出直接用"日式标签 + 英文标签"双写**（如 `サビ(chorus)`），因为：(a) 制谱经验知识库与 MMFC 教程用的是中/日语境；(b) 强模型对 `サビ`/`落ちサビ` 的语义理解是准确的，且比 `chorus` 携带更多制谱含义（落ちサビ 天然暗示"休息段"）。

### 2.5 选型结论

| 角色 | 选择 | 理由 |
|---|---|---|
| **主线** | `all-in-one-infer` 3.1.0（PyPI 名，MIT ✅） | 许可干净、py3.12 ✅、默认路径不需 natten、同时给 beat/downbeat/段落，与 L1 同源 |
| **二号意见（非商用）** | SongFormer | 严格边界显著更好、有 prechorus/build/quietchorus 标签；**MuQ 权重 NC** → 只做个人实验分支 |
| **第三票（零风险）** | libfmp/librosa SSM 重复段 | RefraiD 的现代等价物，在 J-pop 上有实证；**器乐曲失效时靠它兜底** |
| **投票规则** | 边界：三者取交集 + 吸附到最近下拍；标签：chorus 需 ≥2 票 | |
| **pre_chorus/build 派生规则**（因主线无此标签） | 满足全部：① 位于 chorus 起点前 4–8 小节；② 强度曲线单调上升；③ kick 密度下降或消失 / hihat 密度上升 / 高频能量单调上升（riser） | MMFC 5.4 第 3 条要求的"build 末尾切回人声"必须靠它定位 |

---

## 3. 强度曲线与高潮定位（含"无官方音频"下的标定方法）

### 3.1 对 v1 §4 八步配方的修订建议

| v1 做法 | 问题 | 建议修订 |
|---|---|---|
| `intensity = 0.35·rms + 0.30·onset + 0.20·cent + 0.15·drums` | 系数全是拍脑袋；且**质心（centroid）权重过高**——质心对失真吉他、riser、白噪上升敏感，但**对高音人声同样敏感**，容易把 Bメロ 的人声爬升误判成 drop | 把 cent **移出融合项**，只保留在"drop 专用票"里（见 3.3）。融合项换成**与"可踩音数"直接相关**的量 |
| onset_strength 用全曲混音 | 混音 onset 被最响音轨支配，正是 MMFC 5.4 批评的"哪个响踩哪个" | 改用**各 stem onset 合并去重后的数量**（`n_onset_bar`），这是"这一小节最多能踩几个音"的直接代理 |
| hop=512 @ 22050（23 ms） | 对强度曲线够用，但与 §1.2 的 offset 校验共用会磨平精度 | 强度曲线保留 hop=512；**offset 校验单独用 hop=256@44100** |
| `beats[::4]` 当小节边界 | 只在恒定 4/4 且首拍对齐时成立 | 用户已给 offset/BPM → 直接用**构造出来的 downbeat**，不要从 beat 数组切 |
| 模板 `T=[0.20,0.45,0.30,1.00,0.10]` | 来自 MMFC 的定性描述"弱→较强→较弱→强→渐弱"，数值无依据 | 用 388 个官方谱的**密度曲线原型库**替换（见 3.4） |

**建议的新融合式**（权重仍需标定，但每一项都有可解释的制谱含义）：

```
I_bar = w1·Z(loudness_bar)        # pyloudnorm 全曲归一 −14 LUFS 后的逐小节响度
      + w2·Z(n_onset_bar)         # 各 stem 去重合并 onset 数 —— "可踩音上限"
      + w3·Z(E_drums_bar)         # 鼓能量 —— 律动骨架
      + w4·Z(voiced_ratio_bar)    # 人声活动率 —— J-pop 的情绪主载体
      + w5·Z(flux_bar)            # 谱通量 —— 瞬态密度
初值建议：w = [0.25, 0.30, 0.20, 0.15, 0.10]
（相对 v1 的改动：onset 从"强度包络"改成"onset 计数"并升为最大权重；
  新增人声活动项；质心项删除）
```

### 3.2 高潮定位：四票 → 五票

| 票 | v1 权重 | 建议 | 理由 |
|---|---|---|---|
| V1 结构票（chorus/重复段） | 0.40 | **0.45** | 现在有两到三个模型投票，可靠性提高 |
| V2 novelty 票（SuperFlux 峰） | 0.30 | 0.25 | 保留 |
| V3 能量票（rms 局部极大） | 0.20 | 0.20 | 保留；**器乐曲 fallback 时升到 0.5** |
| V4 质心票 | 0.10 | **0.05** | 降权，只留给 EDM drop |
| **V5 人声票（新增）** | — | **0.05** | `voiced_ratio > 0.6` 且音高中位数处于全曲上四分位 → J-pop 副歌的强特征 |

**`最终副歌 / 大サビ` 的定位**：在所有 chorus 段中取"**最后一个 + 强度最高**"者；MMFC 5.4 第 7 条要求的"第二副歌踩音相同但配置升级"需要能明确区分"第 N 次副歌"，因此结构层必须输出 `repeat_of` 与 `chorus_index`。

### 3.3 排除项（v1 未评估）

- **pop-music-highlighter**（remyhuang，TISMIR 2018，"Marking the Emotion Keypoints"）：**GPL-3.0** ✅（GitHub API license 字段），且最后提交 2018-10、TF 1.x → **红线排除**，不要写进管线。
- Essentia DEAM arousal：AGPL + 模型 CC BY-NC-SA（v1 已排除，维持）。
- MERT / MuQ 特征：权重 CC BY-NC（维持排除）。
- → **强度层维持"纯 librosa + pyloudnorm，零模型"的定位是正确的**，本轮没有找到许可干净、又明显更好的替代品。

### 3.4 标定方法：**没有官方音频时怎么办**（本轮重点重写）

现实：`resource/official-chart/` 有 388 个官方 ST 谱（另一条子代理正在算逐小节 note 密度曲线），但**没有对应 mp3** → 无法做"音频特征 → 谱面密度"的回归。
应对：把标定**拆成三块，其中两块不需要音频**。

**(A) 形状先验：官方密度曲线原型库（不需要音频）**
```
1) 每谱 → 逐小节 note 密度 d_b（含 each/hold/slide 的计权，权重按知识 003 的 note 种类强度）
2) 全曲归一：d̂_b = (d_b − min) / (max − min)
3) 相对位置重采样：把每首谱重采样到固定 64 个点（消除曲长差异）
4) k-means / DTW-barycenter 聚类（k = 4~6）→ 得到 4~6 条"官方强度曲线原型"
5) 用途：
   - 替换 v1 的手写模板 T=[0.20,0.45,0.30,1.00,0.10]；
   - 顺便**验证 MMFC 5.2 的"弱→较强→较弱→强→渐弱"到底占多大比例**
     （这是对知识 001 的一次实证检验，结果无论正反都应回写知识库）
6) 分层：按定数（13.0 / 13.5 / 14.0 / 14.5）分别聚类，因为高定数谱的"谷"更浅
```

**(B) 分位映射：强度 → NPS（不需要配对数据）**
```
音频侧：I_bar 的经验分位函数 F_audio
谱面侧：官方同定数谱的 d_b 经验分位函数 F_chart
映射：   d_target(b) = F_chart⁻¹( F_audio(I_bar(b)) )
```
这是气象/统计降尺度里的标准 quantile mapping：**只要求两边的"排序结构"可比，不要求逐条配对**。它保证了：绝对 NPS 水平来自官方谱（正确的难度校准），相对起伏来自音频（正确的情绪贴合）。再叠加知识 004（总 note 数区间）与知识 015（密度红线）做硬裁剪。

**(C) 谱面内部的结构比值（不需要音频）**
官方谱内部可以**无音频地**识别反复段：把逐小节的"note 序列指纹"（位置+类型+分音）做自相似矩阵，找 lag = 8/16 小节的亮带 → 得到谱面自身的 chorus/verse 划分。由此统计出**可直接当规则用的比值**：
- `密度(chorus) / 密度(verse)` 的分布；
- 休息小节（密度 < 全曲 P25）占比；
- 峰值小节的相对位置（验证"大サビ在 75%~90% 处"）；
- 第二副歌相对第一副歌的密度/配置增量（验证 MMFC 5.4 第 7 条）。

**(D) 音频侧只能人工听审**
选 5–10 首目标域曲，由用户或 agent 逐段听审给出强度档（1–5），与 `I_bar` 的段落均值算 Spearman 相关。**目标 ρ ≥ 0.8**。这是当前唯一能验证音频侧的手段，成本可控。

**(E) 将来若用户提供官方音频**
才可做真正的配对标定：逐小节 `I_bar` vs `d_b` 的 Pearson/Spearman、峰值位置误差（小节数）、以及对 w1..w5 的岭回归拟合。**这是唯一能把权重从"初值"变成"标定值"的路径**，应在文档中作为待办挂起。

> ⚠️ 明确：以上 (A)(B)(C) 全部只使用 `resource/official-chart/` 的官方谱；不引入任何社区/自制谱作为设计或评测基准。

---

## 4. "采什么音"（切轨）的可计算代理

### 4.1 逐小节特征清单（L2 需要新增输出）

| 特征 | 计算 | 用途 |
|---|---|---|
| `share_s(b)` | stem s 在小节 b 的能量占比 `E_s / ΣE` | "哪个响" |
| `n_onset_s(b)` | stem s 的 onset 数 | 可踩音数 |
| `grid_fit_s(b)` | stem s 的 onset 落在 `{8}` 网格上的比例 | 律动规整度；低 = 采样/人声长音，不适合当骨架 |
| `voiced_ratio(b)` | vocal stem 有声帧占比 | 人声进出（切轨最强信号） |
| `vocal_notes(b)` | basic-pitch 在 vocal stem 上的 note 数 / 音高中位/最高 | 区分"人声长音"（verse）与"密集人声"（chorus） |
| `kick/snare/hihat(b)` | drums stem 三带 onset 数（见 4.2） | 段落类型判别 + 骨架 |
| `bass_onsets(b)` | bass stem 的 note onset（basic-pitch 或 torchcrepe） | 低频律动骨架；kick 不明显时替代 |
| `riff_sim(b, b−4/−8)` | other stem 的 chroma/CQT 小节向量余弦相似 | hook/riff 循环检测与变化点 |
| `sil_run(b)` | 最长静默长度 | 休息段/留白识别 |

### 4.2 鼓件分类（kick / snare / hihat）

**一手许可证核实结果——ADT 预训练模型全线是非商用：**

| 工具 | 许可 | 结论 |
|---|---|---|
| **ADTOF**（MZehren/ADTOF，5 类 kick/snare/hihat/toms/cymbals） | **CC BY-NC-SA 4.0** ✅（仓库 LICENSE 原文首行 "Attribution-NonCommercial-ShareAlike 4.0 International"） | ❌ 违反 AGENT §7.2 红线 → 默认管线排除；可作个人非商用插件 |
| **LarsNet**（polimi-ispl，鼓件分离 5 路） | 仓库无 license 字段 ✅（GitHub API `license: None`）；README 声明权重 CC BY-NC 4.0 ⚠️ | ❌ 排除 |
| **omnizart**（drum 模块） | 代码 MIT ✅，但 `requires_dist` 含 **`madmom>=0.16.1`** ✅ → 拖入 CC BY-NC-SA 权重 + 2018 年 Cython 包（py3.12 装不上） | ❌ 排除（许可 + 可装性双杀） |
| **drumsep**（inagoy/drumsep，Demucs 微调，kick/snare/toms/cymbals） | 代码 **MIT** ✅ | ⚠️ 权重许可未在仓库明示 ❓ → 可选插件，用前需核 |
| **madmom** 本体 | 无鼓转录功能 | — |

→ **默认方案：无模型的频带启发式**（在 Demucs `drums` stem 上做，泄漏已被分离器处理掉，可靠性远高于在混音上做）：

```
对 drums stem 做 STFT，逐 onset 计算三个能量带的占比：
  LOW   : 30–120 Hz     （kick 基频，峰值多在 50–80 Hz）
  MID   : 120–400 Hz + 1.5–8 kHz 宽带噪声  （snare：低频体 + 响弦噪声）
  HIGH  : 6–16 kHz      （hihat / cymbal）
判据（在该 onset 的短时窗内）：
  LOW 占比最大且 HIGH 占比 < 0.2      → kick
  MID 宽带噪声显著（谱平坦度高）      → snare
  HIGH 占比最大且能量衰减快（< 80 ms）→ hihat（closed）
  HIGH 占比最大且衰减慢（> 300 ms）   → cymbal/crash（常在段落起点 → 结构边界佐证）
```
这套的价值不在"转录精度"，而在**段落判别**：kick 消失 + hihat 密度翻倍 = 典型 build；crash 落在小节 1 拍 = 段落起点。精度要求因此很低，启发式完全够用。❓ 需实测确认在 160–220 BPM 高密度鼓组上的可分性。

### 4.3 人声活动检测（VAD）

- **不要用语音 VAD**：silero-vad（MIT ✅，10k stars）与 pyannote（MIT ✅）都是**语音域**模型，对歌唱（长音、颤音、和声、声码器）表现未验证 ❓，且它们的设计目标是"人声 vs 静音/噪声"，而 Demucs 的 vocal stem 里本来就只剩人声。
- **推荐（零模型）**：在 vocal stem 上做
  `voiced(t) = [RMS_vocal(t) > θ] ∧ [谐波性 > θ_h]`，θ 取全曲 vocal RMS 的 P40；谐波性用 `librosa.effects.harmonic` 能量比或 basic-pitch 的 note 覆盖。加滞回（进入 150 ms / 退出 300 ms）去抖。
- basic-pitch 的 note 事件（Apache-2.0 ✅）同时给出 onset/offset/pitch/confidence → **一份输出同时解决"人声有没有"和"人声踩哪里"**。

### 4.4 段落类型 × 特征 → 主踩音轨（规则表草案）

把 MMFC 5.4 的 8 条谱师判断逐条翻译成可计算判据：

| # | 段落类型 | 触发判据（逐小节特征） | 主踩音轨 | 副踩 / 备注 | MMFC 出处 |
|---|---|---|---|---|---|
| 1 | `intro` | `voiced_ratio < 0.2`，且 `argmax_s share_s` 在段内可能切换 | **`argmax_s share_s`（逐 2–4 小节重算）** | "哪个响踩哪个"——**仅前奏/尾声可用** | 5.4-1 |
| 2 | `verse` | `voiced_ratio ∈ [0.3,0.7]` 且 `vocal_notes/bar < 4`（人声长音为主） | **`other`（吉他/小号等）或 `drums`** | **小节尾混人声**：每 4 小节的最后 1–2 拍切 vocal | 5.4-2 |
| 3 | `pre_chorus` / `build` | 强度单调上升 ∧（kick 密度↓ ∨ hihat 密度↑ ∨ 高频能量单调↑） | 段首 `drums`/`other`，**最后 1–2 小节切 `vocals`** | 为副歌铺垫，切轨点必须落在小节线 | 5.4-3 |
| 4 | `chorus` | `voiced_ratio > 0.6` ∧ `vocal_notes/bar ≥ 4` | **`vocals`**（vocal 曲的最常见做法） | kick 处叠 each/双押作重音 | 5.4-4 |
| 5 | `interlude`（有人声采样） | `voiced_ratio ∈ [0.2,0.5]` ∧ vocal note 短促（median dur < 200 ms）∧ 重复性高 | **vocal 采样** | **强度靠位移提升，不靠密度**（知识 003-2） | 5.4-5 |
| 6 | `休息段` | 位于两个高强度段之间 ∧ `I_bar < P40` | **次响音轨**（`share_s` 第二名，通常是 other 里的管乐/synth） | 显式标 `rest:true`，密度降一档 | 5.4-6 |
| 7 | `final_chorus` / 第二副歌 | `repeat_of` 指向前一 chorus ∧ 位置更靠后 | **与被复制段相同**（踩音不变） | **`upgrade` 字段置位**：配置升级（单星 → 双手星） | 5.4-7 |
| 8 | `outro` | 与 `intro` 的 SSM 相似度 > θ | **与 intro 同轨**（前后呼应） | 收尾可摆 `{24}` 位移交互 | 5.4-8 |

**通用规则（与段落无关）**
- **切轨触发点**：`argmax_s share_s` 改变且持续 ≥ 2 小节 → 候选切轨点；`voiced_ratio` 跨越 0.5 → **强**切轨点。切轨点一律吸附到小节线（或 4 小节乐句边界）。
- **降级规则**：若选定主轨的 `n_onset_s(b) < 2` 或 `grid_fit_s(b) < 0.5` → 退回 `drums` 骨架（kick+snare）。
- **反新手护栏**（知识 002/005）：整曲主踩音轨的**切换次数下限** —— 若全曲 `primary_stem` 唯一值只有 1 个且曲长 > 90 s，**报警**（这正是 MMFC 批评的"哪个最响就一直踩哪个"）。
  ⚠️ 例外已知：MMFC 5.6 提到《金星》整张谱只踩一条音轨且评价很高 → 这是**警告不是硬错误**。

---

## 5. 给 LLM 消费的表示（song sheet）

### 5.1 先验工作的结论

| 项目 | 做法 | 对我们的启示 |
|---|---|---|
| **ChatMusician**（arXiv 2402.16153） | 用 **ABC notation** 这种"面向乐谱、纯文本、高压缩率"的表示训练 LLM；论文强调 ABC **序列比 MIDI 短、且内在编码了重复与结构** ⚠️ | ✅ **给 LLM 的应该是与目标记谱法同构的文本**，不是特征 dump。simai 风格网格串正属此类 |
| **LLark**（Spotify，arXiv 2310.07160） | Jukebox 音频 embedding → 投影层 → Llama2；音频**不转文本** ⚠️ | ❌ 与本项目"LLM 不接触原始音频"（AGENT §8）路线相反，不采纳 |
| **Loop Copilot**（arXiv 2310.12404） | 用音乐 captioning 把输入转成文本；并维护一张**集中式属性表（Global Attribute Table）**跨轮次保存关键属性以"确保音乐连贯性" ⚠️（摘要口径，具体字段未取到 ❓） | ✅ **"一张贯穿全程的属性表"**这个设计直接采纳 → 我们的 song sheet 的 Header + Section table |
| **MusicAgent**（微软） | LLM 做任务调度，调用多个 MIR 模型 ⚠️ | ✅ 佐证"LLM 只做规划、工具做感知"的分工 |
| **Yi 2023**（音游谱生成） | 事件 = "每 2 拍 96 个时间位置"的离散 token ✅ | ✅ 佐证"按拍离散化的位置 token"是有效表示 |

### 5.2 对"逐小节逐 stem 网格字符串"这个想法的评估

**结论：方向正确，建议采纳，但要加四条约束。**

支持理由：
1. 与 simai 同构——LLM 看到的 `x...x...x...x...` 和它要写的 `1,,,,5,,,,` 在时间轴上一一对应，**不需要做心算换算**（这是最大的收益）；
2. 符合 ChatMusician 的结论（文本原生 + 高压缩 + 显式重复结构）；
3. 天然可视，人也能一眼复核（符合 AGENT "可校验"原则）。

必须加的约束：
- **(a) 网格分音要跟随该小节的量化结论**，不能固定 16。每行必须显式写 `div=16`，否则 LLM 会按 16 分去理解一个三连小节。
- **(b) 每行自包含**：`bar 号 | 段落标签 | 强度档 | div | 各轨网格串`。LLM 处理长表格时会丢上下文，绝不能靠"上一行说过"。
- **(c) 字符集与制谱直觉一致**：`x` = onset（该踩的音）、`X` = 重音/强拍、`-` = 延音持续（对应 hold 候选）、`.` = 空。
- **(d) 控制轨数与体量**：3 分钟 / BPM 180 ≈ 135 小节。若 4 轨 × 16 字符 ≈ 每行 ~100 字符 → 全曲约 14 K 字符，可接受。**但轨数上限 4**（kick+snare 合并为 `drum`、`vocal`、`bass`、`other/hook`），不要把 8 条轨全铺出来——那会诱导 LLM 采密（违反知识 005"删到不能再删"）。
- **(e) 明确声明网格串是候选池不是谱面**：在 Header 里写一句硬约束，要求 LLM 从候选中**筛选**，并给出目标 note 总数区间（知识 004）。

### 5.3 推荐的 song sheet 结构（三层 + 双格式）

**双格式同源**：`song_sheet.md`（给 LLM 读）与 `song_analysis.json`（给程序/校验器读）由同一份数据渲染，禁止两边手改。

```markdown
# SONG SHEET — <title>
## 0. Header
bpm: 180 (const)   offset(&first): 1.234 s   bars: 1..136   time-sig: 4/4
offset_check: φ* = +4 ms (OK)   bpm_drift: 1.2e-5 /s (OK)
target: ST Master 定数 13.5 → note 总数 700–850（知识 004）；NPS 上限见知识 015
分音使用统计: {8}=61%  {16}=33%  {12}=4%  {24}=2%
⚠️ 下面的网格是**候选池**，不是谱面。按知识 005「删到不能再删」筛选。

## 1. Sections
| bars   | 段落 (日/英)        | intensity | 建议密度 | 主踩    | 副踩        | repeat_of | 备注 |
|--------|--------------------|-----------|---------|---------|-------------|-----------|------|
| 1–8    | イントロ (intro)     | 0.18      | {8}     | other   | —           | —         | 哪个响踩哪个；0–4 小节仅 pad |
| 9–24   | Aメロ (verse)       | 0.42      | {8}     | other   | vocal(尾拍) | —         | 人声长音，note 2.1/小节 |
| 25–32  | Bメロ (pre_chorus)  | 0.31→0.78 | {8}→{16}| drums   | vocal       | —         | kick 在 29 小节消失；31–32 切人声 |
| 33–48  | サビ (chorus #1)    | 0.91      | {16}    | vocal   | kick(双押)  | —         | voiced 0.82 |
| 49–56  | 間奏 (interlude)    | 0.55      | {8}     | vocal采样| —          | —         | 强度靠位移（知识 003-2） |
| 57–64  | 落ちサビ (quiet_ch.)| 0.30      | {8}     | other   | —           | —         | **休息段** rest:true |
| 65–80  | ラスサビ (chorus #2)| 1.00      | {16}    | vocal   | kick        | 33–48     | **upgrade: 单星→双手星** |
| 81–88  | アウトロ (outro)     | 0.20      | {8}     | other   | —           | intro     | 前后呼应；末尾 {24} 位移交互 |

## 2. Bars（逐小节）
bar 33 | サビ | I=0.90 | div=16 | drum  X...x...X...x...
                                | vocal .x.x..x..x.x....
                                | bass  x.......x.......
                                | hook  ................
       | note: crash@1拍(段落起点); vocal 入
bar 34 | サビ | I=0.92 | div=16 | drum  X...x...X..xX...
       ...
```

**JSON 侧**（供校验器/生成器）保留 v1 的 `charting_plan` 字段并扩展：
`sections[]` 增加 `chorus_index`、`repeat_of`、`upgrade`、`rest`、`labels_ja`；
新增 `bars[]`：`{bar, div, intensity, onsets: {drum:[idx...], vocal:[...], bass:[...], hook:[...]}, notes:[...]}`。

---

## 6. 本机可行性（M4 / 16 GB / macOS 26 / Python 3.12 / torch 2.11 + MPS）

> 本节全部来自 PyPI JSON / GitHub API / 源码一手核实 ✅；**RTF 与实际能否跑通由另一条子代理实测**，这里给"预期与坑"。

### 6.1 包兼容性速查（2026-09-11 快照）

| 包 | 最新版 | requires_python | 许可 | py3.12 | 备注 |
|---|---|---|---|---|---|
| `librosa` | **1.0.0**（2026-08） | **>=3.12** | ISC ✅ | ✅（3.12/3.13/3.14 分类器） | `beat.tempo` 已移除；`beat.plp` 仍在 ✅（`__all__=["beat_track","plp"]`） |
| `demucs` | **4.1.0**（2026-07） | >=3.10 | MIT ✅ | ✅ | 依赖 `sphn>=0.1.12` → cp312 arm64 wheel ✅ |
| `beat-this` | 1.1.0（2026-04） | >=3 | MIT ✅ | ✅ | 依赖仅 torch/torchaudio/einops/rotary-embedding-torch/soxr |
| `all-in-one-infer` | **3.1.0** | >=3.9 | **MIT ✅** | ✅（有 3.12 分类器） | **推荐主线**；natten 变成可选 extra |
| `allin1`（原版） | 1.1.0（**2023-10**） | >=3.8 | — | ❌ 无 3.12 分类器 | **`natten; platform_system=="Darwin"` 是强制依赖** ✅ → 见 6.2 |
| `basic-pitch` | 0.4.0（2024-08） | — | Apache-2.0 | ⚠️ 见 6.3 | Darwin 默认拉 `coremltools` |
| `libfmp` | 1.3.0（2026-03） | >=3.10 | MIT ✅ | ✅ | |
| `pyloudnorm` | 0.2.0（2026-01） | >=3.9 | — | ✅（显式 3.12 分类器） | |
| `madmom` | 0.16.1（**2018**） | — | 代码 BSD / **模型 CC BY-NC-SA** | ❌ | 维持排除 |
| `omnizart` | 0.6.3 | >=3.8 | MIT | ❌ | `requires_dist` 含 `madmom>=0.16.1` ✅ → 连坐 |
| `onnxruntime` | 1.30.0 | >=3.11 | MIT ✅ | ✅ | basic-pitch 的 onnx 后端可用 |
| `coremltools` | 9.0 | — | BSD ✅ | ✅（3.12 分类器） | basic-pitch 在 macOS 的默认后端 |
| `torchcodec` | 0.16.0 | >=3.10 | — | ✅（cp312 macos arm64 wheel） | torchaudio ≥2.11 解 mp3 需要它 |

### 6.2 坑 1：all-in-one 原版在 Apple Silicon 上装不了

- `allin1` 的 `requires_dist` 里写着 **`natten; platform_system == "Darwin"`** ✅ —— macOS 上 natten 是**强制**依赖。
- `natten` 最新 0.21.7 在 PyPI **只有 sdist（`natten-0.21.7.tar.gz`），没有任何 wheel** ✅ → 必须现场编译 C++/CUDA 扩展，且要对齐已安装的 torch 版本。
- 上游 issue 佐证 ✅：`[macOS] Unable to install allin1`（2025-04，open）、`MPS Alternatives to Natten`（2024-06，open）、`Incompatibility issues with NATTEN v0.17.5`（open）。原仓库最后一次 push 是 **2024-05**。
- → **不要装 `allin1`**。改装 `pip install all-in-one-infer`（MIT ✅，import 名 `allin1_infer`，CLI `all-in-one-infer`），它把 natten 降级成可选 extra。

### 6.3 坑 2：basic-pitch 在 py3.12 macOS 上的后端

- `basic-pitch` 的 `[tf]` extra 钉死 `tensorflow-macos<2.15.1`；而 tensorflow-macos **2.14.0 / 2.15.0 只有 cp39/cp310/cp311 wheel，2.16.2 才有 cp312** ✅ → **在 Python 3.12 上 TF 后端装不上**。
- 默认（无 extra）安装在 Darwin 上会拉 `coremltools`（9.0 支持 3.12 ✅）→ 走 CoreML 后端，理论上在 M4 上反而更快（ANE/GPU）。
- 保险做法：`pip install "basic-pitch[onnx]"` → onnxruntime 1.30（3.11–3.14 ✅），跨平台一致、可复现。
- ❓ 三个后端（CoreML / ONNX / TF）输出是否数值一致，需实测（若不一致，onset 时间差会直接污染量化结果 → **建议固定用 ONNX**）。

### 6.4 坑 3：MPS

- `demucs` 的 `separate.py` 会在 **CUDA 不可用时自动选 `mps`** ✅（源码：`else "mps" if th.backends.mps.is_available()`）。
- 但 HTDemucs 在 MPS 上历史上有复数张量 / 自定义算子的兼容问题 ⚠️（社区报告），另有 PyTorch 侧 issue 称 macOS 26.x 上 `torch.backends.mps.is_available()` 误报 ⚠️（pytorch#177819）。
- 应对：① 先跑一次 `mps` 与 `cpu` 的**输出差异检查**（同一首曲，比较 stem 的 RMS 与 onset 时间，差异应 < 1e-3 / < 5 ms）；② 不一致就 `-d cpu`；③ 加速备选 **`ssmall256/demucs-mlx`（MIT ✅）** 或 `ssmall256/mlx-audio-separator`（MIT ✅），MLX 原生跑 Apple Silicon。
- `beat_this` 的 **CLI 只暴露 `--gpu`（CUDA 语义）** ✅，MPS 要走 Python API：`File2Beats(checkpoint_path="final0", device="mps", dbn=False)`。

### 6.5 坑 4：torchaudio ≥ 2.11 不再自带 mp3 解码

all-in-one-infer README（3.1.0+）明确：WAV/FLAC 走 `soundfile`（与 torchaudio 位级一致），**MP3 走 torchaudio，而 torchaudio ≥ 2.11 需要另装 `torchcodec`** ✅。
→ 再次印证 §1 Step 0：**统一 ffmpeg 转 WAV**，一次性绕开所有解码器差异与依赖问题。

### 6.6 坑 5（许可证）：all-in-one-infer → madmom-infer 的运行时下载

- `all-in-one-infer` 依赖 **`madmom-infer`**（BSD-2-Clause ✅，纯 numpy 从头重实现，不含 Cython）。
- 但 `madmom_infer/models.py` 的文件头写明 ✅：该模块会**运行时从 `CPJKU/madmom_models` 下载 `.pkl` 权重，这些权重是 CC BY-NC-SA 4.0（非商用）**，仅在调用者显式请求某个模型（如 `RNNDownBeatProcessor()`）时才触发下载，且从不打包进仓库。
- all-in-one 只用 madmom 的 **DBN 解码器**（纯 HMM 算法，无学习权重）对自家 activation 做后处理 → **理论上不会触发 NC 权重下载**。
- ❓ **必须实测确认**：跑一次分析后检查 `$XDG_CACHE_HOME/madmom_infer/models/` 是否为空。若非空，则这条管线落入非商用范畴，需在 AGENT §7.2 备案。

### 6.7 内存

M4 / 16 GB：htdemucs_ft 四模型串行 CPU/MPS 推理是内存大头（v1 记 GPU 3–7 GB）；basic-pitch 峰值 ~951 MB。建议**分阶段落盘**（分离 → 写 stem wav → 释放 → 再分析），不要把四个 stem 全程驻留。

---

## 7. 可用的评测数据

> 原则重申：**谱面设计参考只用 `resource/official-chart/` 的 388 个官方 ST 谱**。下面的外部数据集只用于**音频分析层（节拍/offset/分段/副歌）的工程验证**，不作为谱面设计基准。

### 7.1 RWC-Pop：**2026 年已开放在线下载**（本轮最大发现）

- AIST 官网首页现在写着：**"The RWC Music Database has been available for online download since February 2026."** ✅，指向 `https://doi.org/10.5334/tismir.326` 与 `https://zenodo.org/communities/rwc-music/`，最新标注在 `https://github.com/rwc-music/rwc-annotations`。
- Zenodo 记录（API 一手）✅：`RWC Music Database`，DOI `10.5281/zenodo.18656623`，**许可 CC BY-NC 4.0**，文件含 **`RWC-P.zip` 4.07 GB**（Popular）、`RWC-C/J/G/R.zip`。
- 内容 ✅：Popular 100 首 = **日语歌词的日本流行乐 80 首** + 英语 20 首；**AIST Annotation 提供 beat structure、melody line、chorus sections、结构段落标签**（intro / verse A / verse B / pre-chorus / chorus A / chorus B / post-chorus / bridge / ending）。
- **这是目前唯一一个"日语流行 + 结构/副歌/节拍标注 + 音频可合法获取"的数据集**，直接解决 v1 §9.1 的"域外泛化未知"。
- 限制：**CC BY-NC 4.0** → 不得入仓库、不得商用、不得再分发；本机评测用途没问题。
- 参考基线：RefraiD 在 RWC-Pop 上 100 首中 80 首正确、F=0.938 ⚠️ → 我们的 SSM 重复段支线应以此为对标。

### 7.2 osu2beat2025（Osu2MIR，ISMIR 2025 LBD）：域最接近的免费节拍/offset 集

- 仓库 `ziyunliu4444/osu2mir` ✅（无 LICENSE 字段 ❓；论文 CC BY 4.0）。
- 内容 ✅：`osu2beat2025_metered_beats.zip` = **741 条用户标注 / 708 首不同音频**的 metered beats（含 downbeat），文件名带 MD5 + BeatmapSetID；**音频不随附**，需自行从 `osu.ppy.sh/beatmapsets/<id>` 下载 `.osz` 解出 mp3 并校验 MD5（合法免费）。
- **曲风正是我们的域**：论文强调 osu! 谱覆盖"anime、Vocaloid、video game music 等在 MIR 数据集里代表性不足的曲风" ⚠️。
- 质量筛选经验（可直接复用）✅：只用"单个未继承 timing point"或"多点间隔 ≥ 5 s"的图；间隔 < 5 s 的需额外筛查。
- **用途限定**：验证 §1.2 的 offset 相位搜索、BPM 漂移检测、beat_this/all-in-one 在 ACG 曲上的 F1。**不涉及谱面设计**（osu! 谱是社区自制谱，按用户规矩不作为制谱参考）。

### 7.3 其他

| 数据集 | 音频 | 标注 | 日语 | 结论 |
|---|---|---|---|---|
| **Harmonix Set** | ❌ 不含音频（只有标注 + 预提特征） | beat/downbeat/段落（912 首西方流行） | ❌ | 仓库 MIT ✅；只能当**标签体系**参考（all-in-one/SongFormer 的标签都源于它） |
| **SALAMI** | 多数需从 Internet Archive 自取 | 多层结构标注 | 极少 | 曲风杂，优先级低 |
| **SongFormBench** | HF 数据集，CC BY 4.0 ✅ | 300 首专家校验结构 | ❌（语言字段 `["en","zh"]` ✅） | 可做西方/中文对照，**无日语** |
| **BOF / BMS 曲包** | 作者自由分发 | 谱面（社区自制） | 多 | GenerationMania 曾用 BOF2011（1454 谱 / 366 曲）⚠️。但 ① 逐年逐作者的授权条款不一致 ❓；② **谱面是社区自制谱 → 按用户规矩不作谱面参考**；③ 音频侧的价值（"音频 + 精确 BPM/offset 元数据"）已被 osu2beat2025 更干净地覆盖。→ **不建议投入** |
| **官方 maimai 音频** | 用户当前没有 | — | ✅ | **唯一能做"音频 ↔ 官方谱密度"配对标定的路径**；作为待办挂起，若用户日后能提供哪怕 10–20 首，就能把 §3.4(E) 跑通 |

### 7.4 建议的三档评测协议

| 档 | 目的 | 数据 | 指标 |
|---|---|---|---|
| **A** 日语结构/副歌 | 验证分段与"副歌=重复最多段" | RWC-Pop 80 首日语（本机，NC） | 边界 HR.5F / HR3F、chorus 段 IoU、副歌命中率（对标 RefraiD 80/100） |
| **B** 域内节拍/offset | 验证 §1.2 相位搜索与量化 | osu2beat2025 的"单 timing point"子集（anime/Vocaloid/game） | `\|φ*\|` 分布、BPM 漂移估计误差、beat F1(70 ms) |
| **C** 谱面密度基准 | 难度/强度校准 | **仅 388 官方 ST 谱**（无音频） | 密度曲线原型库、分位映射表、chorus/verse 比值（§3.4 A/B/C） |

---

## 8. 对 `docs/audio-analysis.md` 的修订建议清单

> 只列建议，**本报告未改动 `docs/audio-analysis.md`**，由主会话审核后合并。

| # | 目标节 | 类型 | 建议 |
|---|---|---|---|
| R1 | §2 L1 | **新增小节** | 加「§2.x 用户 offset 的音频反向校验」——网格相位搜索 + BPM 漂移线性回归 + 三档判据（≤10 / ≤30 / >30 ms），见本报告 §1.2 |
| R2 | §2 L1 | 强化 | "MP3 先转 WAV" 升级为**硬性 Step 0**：全管线只读同一个 ffmpeg 产出的 WAV；补一手理由（解码器差异 20–40 ms vs beat 容差 70 ms） |
| R3 | §2 L1 | 补充 | beat_this 的 MPS 用法：CLI 只有 `--gpu`（CUDA），MPS 需 `File2Beats(device="mps")` |
| R4 | **新增 §2.5 或独立节** | **新增** | 「onset → 分音网格量化」完整规则：拍同步重采样（1/48 拍）、逐小节最小分音扫描、容差 `τ=clamp(0.25·slot,12,30) ms`、三连判别（rms12 ≤ 0.7·rms16）、swing 判别、**{32} 红线**。见本报告 §1.1 |
| R5 | §3 L2 | 扩充输出 | 逐小节特征清单（share/onset 数/grid_fit/voiced_ratio/vocal_notes/kick-snare-hihat/bass/riff_sim/sil_run），见 §4.1 |
| R6 | §3 L2 | **新增** | 鼓件三带启发式（30–120 / 120–400+1.5–8k / 6–16 kHz）；并写明 **ADT 预训练模型全线 NC**：ADTOF = CC BY-NC-SA 4.0 ✅、LarsNet 权重 CC BY-NC ⚠️、omnizart 连坐 madmom ✅ → 默认排除；drumsep（代码 MIT ✅）权重许可待核 |
| R7 | §3 L2 | 新增 | 人声 VAD 用 vocal stem 能量+谐波性+滞回，**不用语音 VAD**（silero/pyannote 虽 MIT 但域不符） |
| R8 | §4 L3 | **事实更正** | all-in-one 功能标签只有 8 类（`intro/outro/break/bridge/inst/solo/verse/chorus`，+start/end），**没有 pre-chorus** ✅；§5.3 样例里的 `"type": "pre_chorus"` 必须改成"由规则派生"并写出派生判据 |
| R9 | §4 / §7.3 | 事实更正 | all-in-one 是 **WASPAA 2023**，不是 ISMIR 2023 |
| R10 | §4 L3 | 新增候选 | **SongFormer（CC BY 4.0 ✅）作为二号意见**：HarmonixSet HR.5F 0.703 vs all-in-one 0.596；标签含 prechorus/build/quietchorus。**但 MuQ 权重 CC BY-NC-4.0 ✅ → 只入个人非商用分支**；帧率 8.333 Hz 需吸附下拍 |
| R11 | §4 L3 | 新增 | 日式↔英文↔模型标签映射表（§2.4），并规定**分析输出用"日式(英文)"双标签** |
| R12 | §4 L3 | 补证据 | "副歌=重复最多段"在日语流行上有实证：RefraiD 在 RWC-Pop（80 首日语）100 中 80 正确、F=0.938 → v1 §9.1 的"未知"可下调为"J-pop 成立，器乐/EDM 曲需 fallback" |
| R13 | §4 强度配方 | **修订系数** | 删除质心融合项（移到 drop 专用票）；onset 从"包络"改为"**去重 onset 计数**"并升权；新增人声活动项。新初值 `w=[0.25,0.30,0.20,0.15,0.10]`（见 §3.1） |
| R14 | §4 高潮投票 | 修订 | 四票 → 五票：结构 0.45 / novelty 0.25 / 能量 0.20 / 质心 0.05 / **人声 0.05**；输出必须带 `chorus_index` 与 `repeat_of` |
| R15 | §4 模板 | **替换** | 手写模板 `T=[0.20,0.45,0.30,1.00,0.10]` 替换为「388 官方谱密度曲线原型库」（归一 + 重采样 64 点 + 按定数分层聚类）；顺带实证检验知识 001 的"五段走势"占比 |
| R16 | **新增 §4.x** | **新增** | 「无官方音频时的标定协议」：(A) 形状先验 (B) 分位映射 F_chart⁻¹∘F_audio (C) 谱面内部结构比值 (D) 人工听审 Spearman ≥0.8 (E) 待用户提供官方音频后的配对标定。见 §3.4 |
| R17 | §4 | 排除项 | pop-music-highlighter = **GPL-3.0 ✅ + TF1.x（2018 停更）→ 红线排除**（v1 未评估此项） |
| R18 | §5 L4 | **扩展输出** | `charting_plan.json` 升级为 **song sheet 双格式**（`song_sheet.md` 给 LLM + `song_analysis.json` 给程序），含 Header / Section table / **逐小节网格串**；四条约束（div 跟随量化、行自包含、字符集 `X x - .`、轨数 ≤4）与"候选池非谱面"声明。见 §5.3 |
| R19 | §5 L4 | 新增 | 「段落类型 × 特征 → 主踩音轨」规则表（8 条，逐条对应 MMFC 5.4）+ 切轨触发判据 + 降级规则 + "整曲单轨"警告（注意《金星》反例 → 警告非错误）。见 §4.4 |
| R20 | §7.1 选型表 | 更新 | L3 主线从 `all-in-one` 改为 **`all-in-one-infer` 3.1.0（PyPI 名，MIT ✅，py3.12 ✅，natten 可选）**；注明原版 `allin1` 在 Darwin 强制依赖 natten 且 natten 无 wheel ✅ |
| R21 | §7.2 许可证策略 | 补充 | ① 新增 **CC BY 4.0 是否纳入白名单**的决策点（SongFormer 代码/权重）；② 记录 `madmom-infer` 运行时可能下载 CC BY-NC-SA 权重的风险与检查方法（`$XDG_CACHE_HOME/madmom_infer/models/` 应为空）；③ 新增 ADTOF/LarsNet/pop-music-highlighter 的排除记录 |
| R22 | §7.3 prior art | 补充/更正 | ① **DDC 不做音乐量化**（伪造 BPM=125 让 1/192 小节 = 10 ms 帧）✅ → 它不是 snap 逻辑的参照；其 `BeatCalc` 才是可抄的部分；② 新增 Yi 2023（arXiv 2311.13687，1/48 拍 beat-aligned 频谱，osu!mania 14648 谱）作为"拍同步重采样"的依据 |
| R23 | **新增 §10** | **新增** | 「评测数据与协议」：A 档 RWC-Pop（**2026-02 起 Zenodo 开放，CC BY-NC 4.0，RWC-P.zip 4.07 GB，80 首日语 + AIST 结构/副歌/beat 标注**）；B 档 osu2beat2025（741 标注/708 曲，anime/Vocaloid/game，音频自取）；C 档仅 388 官方谱。**明确 BOF/BMS 不投入**（授权逐年不一 + 社区自制谱不作谱面参考） |
| R24 | §9 存疑 | 更新 | 删除已解决项（域外数据"无处可寻"→ 已有 RWC-Pop + osu2beat2025）；新增本报告 §10 的待实测项 |

---

## 9. 来源清单

**一手核实 ✅（源码 / LICENSE / API / 官网原文）**

- mir_eval 容差：`mir_eval/onset.py`（window=0.05）、`mir_eval/beat.py`（f_measure_threshold=0.07）— https://github.com/mir-evaluation/mir_eval
- DDC 量化与时间轴：`infer/ddc_server.py`（`_SUBDIV=192,_DT=0.01,_BPM=125`）、`learn/beatcalc.py`、`dataset/smdataset/abstime.py` — https://github.com/chrisdonahue/ddc （MIT）
- all-in-one 标签集：`src/allin1/config.py: HARMONIX_LABELS` — https://github.com/mir-aidj/all-in-one （MIT；last push 2024-05）
- all-in-one issues（natten / macOS / MPS）— GitHub Search API，14 条命中
- all-in-one-infer：README「Concerning MP3 Files」（解码器差 20–40 ms）、Speed、LICENSE(MIT)、PyPI 3.1.0 依赖表 — https://github.com/openmirlab/all-in-one-infer 、https://pypi.org/project/all-in-one-infer/
- madmom-infer：`madmom_infer/models.py` 文件头 LICENSE NOTICE（运行时下载 CC BY-NC-SA 权重）、`NOTICE`、`LICENSE`(BSD-2) — https://github.com/openmirlab/madmom-infer
- SongFormer：GitHub `LICENSE`（CC BY 4.0）、README 性能表与速度表、HF `config.json` / `configuration_songformer.py`（num_classes=128, frame_rates=8.333）/ `dataset/label2id.py`（128 标签）/ `postprocessing/functional.py`（改编自 all-in-one，MIT 注明）/ `musicfm/LICENSE`（MIT+Apache-2.0） — https://github.com/ASLP-lab/SongFormer 、https://huggingface.co/ASLP-lab/SongFormer
- MuQ 权重许可：HF API `OpenMuQ/MuQ-large-msd-iter` → `license:cc-by-nc-4.0`；代码仓 `tencent-ailab/MuQ` MIT
- SongFormBench 数据集：HF API `license: cc-by-4.0`, `language: ["en","zh"]`
- ADTOF LICENSE：`Attribution-NonCommercial-ShareAlike 4.0 International` — https://github.com/MZehren/ADTOF
- LarsNet：GitHub API `license: None` — https://github.com/polimi-ispl/larsnet
- drumsep：MIT — https://github.com/inagoy/drumsep
- pop-music-highlighter：GitHub API `license: GPL-3.0`，last push 2018-10 — https://github.com/remyhuang/pop-music-highlighter
- demucs：`separate.py` 自动选 mps；PyPI 4.1.0（>=3.10, MIT, sphn 依赖）— https://github.com/adefossez/demucs
- demucs-mlx / mlx-audio-separator：MIT — https://github.com/ssmall256/demucs-mlx 、https://github.com/ssmall256/mlx-audio-separator
- beat_this：MIT；README device/`--gpu` 说明 — https://github.com/CPJKU/beat_this
- librosa：`librosa/beat.py: __all__=["beat_track","plp"]`；PyPI 1.0.0 requires_python>=3.12 — https://github.com/librosa/librosa
- PyPI JSON（版本/requires_python/classifiers/requires_dist）：demucs, basic-pitch, allin1, beat-this, libfmp, pyloudnorm, librosa, madmom, omnizart, natten, coremltools, tensorflow-macos(2.14/2.15/2.16 wheel 列表), onnxruntime, silero-vad, sphn, torchcodec, all-in-one-infer, madmom-infer, demucs-infer
- RWC：AIST 官网首页（2026-02 起在线下载 / Popular 100 首含 80 首日语）— https://staff.aist.go.jp/m.goto/RWC-MDB/ ；Zenodo API `communities=rwc-music`（CC BY-NC 4.0，RWC-P.zip 4.07 GB，DOI 10.5281/zenodo.18656623）；标注仓 https://github.com/rwc-music/rwc-annotations ；TISMIR 10.5334/tismir.326
- osu2mir：README（741 标注 / 708 音频、下载流程、三档 timing point 划分）— https://github.com/ziyunliu4444/osu2mir
- Harmonix Set：MIT — https://github.com/urinieto/harmonixset
- osu! wiki《How to time songs》（"任何音频编辑包括重新编码都会改变 timing"）— https://osu.ppy.sh/wiki/en/Guides/How_to_time_songs
- 本仓库：`docs/simai-syntax.md` §1.4/§3.2（384 约数、槽长公式）；`resource/mmfc-guide-extracted.txt` 1292–1520 行（MMFC 第五章）

**部分核实 ⚠️（论文摘要 / 二手数字，未逐一核对原文表格）**

- Kim & Nam, *All-In-One Metrical And Functional Structure Analysis…*, WASPAA 2023 — arXiv 2307.16425
- Hao et al., *SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision*, 2025 — arXiv 2510.02797
- Goto, *A Chorus Section Detection Method…*, IEEE TASLP 2006（RWC-Pop 100 中 80 正确、F=0.938）— https://staff.aist.go.jp/m.goto/PAPER/IEEETASLP200609goto.pdf
- Goto, *AIST Annotation for the RWC Music Database*, ISMIR 2006（段落标签体系）
- Yi, *Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts*, 2023 — arXiv 2311.13687（1/48 拍 hop、96 位置 token、osu!mania 14648 谱/3166 曲、micro-F1 84.6%）
- Liu et al., *Osu2MIR: Beat Tracking Dataset Derived From Osu! Data*, ISMIR 2025 LBD — arXiv 2509.12667
- *Beat-Based Rhythm Quantization of MIDI Performances*, 2025 — arXiv 2508.19262（给定 beat 标注的 Transformer 量化，MUSTER SOTA）
- *Transformer-Based Rhythm Quantization of Performance MIDI Using Beat Annotations*（ASAP onset F1 97.3%）— arXiv 2604.22290
- Nakamura 等，metrical HMM 节奏量化（避免"不完整三连"）
- Dittmar, Müller 等, *Automated Estimation of Ride Cymbal Swing Ratios in Jazz Recordings*, ISMIR 2015（swing factor 1=直 / 2=三连；ratio 随 tempo 下降）
- Yuan et al., *ChatMusician*, 2024 — arXiv 2402.16153（ABC notation 高压缩、内在编码重复与结构）
- Gardner et al., *LLark*, 2023 — arXiv 2310.07160（Jukebox embedding + Llama2）
- Zhang et al., *Loop Copilot*, 2023 — arXiv 2310.12404（Global Attribute Table 维持音乐连贯性）
- *GenerationMania*（BOF2011：1454 谱 / 366 曲）— arXiv 1806.11170
- 日英段落术语对照（Aメロ=Verse / Bメロ=Pre-Chorus / サビ=Chorus / 落ちサビ=Breakdown·Quiet Chorus / 大サビ=Bridge / ラスサビ=Final Chorus / 間奏=Interlude）— tetsu7017 音楽用語和英比較、er-music.jp、note.com 多源交叉
- PyTorch MPS on macOS 26.x 误报 issue（pytorch#177819）；Demucs on MPS 复数张量兼容性（社区报告 + MLX 移植动机）

---

## 10. 存疑 / 待实测

| # | 项 | 标记 | 处理方式 |
|---|---|---|---|
| 1 | 容差 `τ = clamp(0.25·slot, 12, 30) ms` 的 0.25 / 12 / 30 三个数字 | ❓ 设计初值 | 在 osu2beat2025 子集上标定：用已知 BPM/offset 反推 onset 残差分布 |
| 2 | 三连判别阈 `rms12 ≤ 0.7·rms16` | ❓ | 同上 |
| 3 | swing 在 160–220 BPM 音游曲里到底出不出现 | ❓ | 先默认关闭；统计目标曲库的 offbeat 相位分布再决定 |
| 4 | 强度融合新权重 `[0.25,0.30,0.20,0.15,0.10]` | ❓ 初值 | 需 §3.4(E) 的配对数据才能真正标定；在此之前只能靠 (D) 人工听审做排序检验 |
| 5 | 官方谱密度曲线原型库的聚类数 k、以及"五段走势"实际占比 | ❓ | 待另一条子代理的密度曲线产出后跑 |
| 6 | 鼓件三带启发式在高密度电子鼓组上的可分性 | ❓ | 项目内 A/B：人工标 50 个 onset 的鼓件，算混淆矩阵 |
| 7 | vocal stem 的能量+谐波性 VAD 阈值 | ❓ | 用 RWC-Pop（有旋律标注）标定 |
| 8 | basic-pitch 三后端（CoreML / ONNX / TF）数值是否一致 | ❓ | 实测；若不一致固定用 ONNX |
| 9 | Demucs 在 MPS 上输出是否与 CPU 一致 | ❓ | 实测（RMS 差 < 1e-3、onset 差 < 5 ms 为通过） |
| 10 | all-in-one-infer 是否会触发 madmom NC 权重下载 | ❓ **许可风险** | 跑一次后检查 `$XDG_CACHE_HOME/madmom_infer/models/` |
| 11 | SongFormer 能否在 M4 MPS / CPU 上跑（官方只给 cuda 示例） | ❓ | 实测；跑不动就放弃二号意见 |
| 12 | CC BY 4.0 是否纳入项目依赖白名单 | ❓ **需用户决策** | 影响 SongFormer 能否进管线（其 MuQ 权重仍是 NC，即便白名单放开也只能非商用） |
| 13 | drumsep 权重许可 | ❓ | 用前核实 |
| 14 | Loop Copilot 的 Global Attribute Table 具体字段 | ❓ | 只取到摘要口径；若要照搬字段需读全文 PDF |
| 15 | RefraiD 的 80/100 与 F=0.938 未见论文原表 | ⚠️ | PDF 在 AIST 站点可取，需要时补核 |
| 16 | 官方 maimai 音频 | ❌ 当前无数据 | **挂起**：一旦用户能提供 10–20 首，立刻跑 §3.4(E) 配对标定 |
