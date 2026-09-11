# tools/audio_analysis — 歌曲分析单生成器（原型 v0.3）

把一首 mp3 变成 **LLM 可直接消费的结构化「歌曲分析单」**：段落划分（日式标签）、
逐小节强度与建议密度、逐小节四轨 onset 网格串、逐段切轨建议。对应
`docs/audio-analysis.md` v1.0 管线的 **L1–L3 层**（L4 规划层与生成器不在本工具范围内），
实现口径以 `docs/research/audio-analysis-research-v2.md` 为准。

> **前提**：BPM 与 offset（simai `&first`）**由用户给定**。本工具**不做节拍追踪**，
> 只对用户给的 offset 做一致性校验并**如实报告差值，绝不覆盖用户值**。

## 0. v0.3 变更一览（落地 8 首官方音频配对标定的结论，2026-09-11）

依据：`docs/research/audio-chart-calibration.md`（R1–R11）+ 主会话裁定。

| # | 变更 | 修法 | 依据 |
|---|------|------|------|
| **A** | **offset 校验的 ±半拍反拍假警报（真 bug）** | 搜索半径 ±1 拍 → **±0.4 拍**（`MAX_SEARCH_BEATS=0.45` 强制夹紧）；同级峰仍取离用户值最近者；新增「反拍歧义 / 不可判定」判词与 `ratio ≥ 1.15` 显著性门；15–30 ms 的判词改指 **MP3 解码延迟**而非「谱面可能错」 | 报告 §1.2 / R1 / R2 |
| **B** | **单值 `primary_stem` → 骨架轨 + 点缀轨** | `stemplan.py` 重写：`skeleton_stem` / `accent_stems` / `accent_share`（含 `free` 残差 0.19）/ `plan_evidence`；规则 1 判据从能量占比换成 **onset 匹配倾向**；规则 4（副歌踩人声）**条件触发 + 标存疑**；规则 5（间奏人声采样）**加强**；规则 6 **标存疑** | 报告 §5 / R8–R10 |
| **C** | **密度绝对量级锚点 note/小节 → NPS** | `nps_for_level` + `notes_per_bar_from_nps`（NPS × 每小节秒数，变速曲逐小节算）；旧锚保留为对照列 | 报告 §4.2 / R6 |
| **D** | **段落地板参数化 + intro/outro 结构封顶** | `--section-floor`（默认仍 0.60，Header 注明 n=8 最优 0.125）、`--bar-floor`、`--intro-outro-cap`（默认 0.60 × 全曲建议均值） | 报告 §4.1 / R5 + MYTHOS 个案 |
| **E** | **`climax` 正名为「音频能量高潮」** | JSON 加 `audio_climax_bar` / `climax_kind` / `climax_note`；Header 显式写「不是谱面最密处，中位误差 33 小节」 | 报告 §2.3 / R4 |
| **F** | **Header 加候选池参考量级** | 「官方谱只采用候选池约 55%（precision 0.546、recall 0.821）——删到不能再删」 | 报告 §5.2 |

**没改的**：`intensity.py` 的融合权重（标定结论是维持初值，见 §6-6）。

## 0.1 v0.2 变更一览（对 v0.1 的五项返工）

| # | v0.1 的问题（主会话验收） | v0.2 的修法 | 依据 |
|---|--------------------------|-------------|------|
| A | 同一小节各轨网格串长度不一致（kick 4 格 / other 24 格），LLM 得数字符；出现 `{32}`；容差 `max(25ms, 1/64 小节)` 在 192 BPM 下比 24↔32 格线的最小间距（13 ms）还大，16/24/32 之间近乎随机 | **拍同步 1/48 拍重采样**；**每小节只选一个 div**，四轨共用；容差 `τ(d)=clamp(0.25·slot_ms(d),12,30)`；三连判别 `rms12≤τ(12) ∧ rms12≤0.7·rms16`；**`{32}` 红线**（仅 drums 且该小节 drums onset≥6 且 rms≤15ms 才放行，否则压到 16/24 并把落格失败的 onset 计入 `unquantized`）；swing 默认关闭 | v2 §1.1 Step 2–7 |
| B | song sheet 铺 6 条轨，超过上限 4；轨越多越诱导采密 | 压成 **4 轨**（`drum`/`vocal`/`bass`/`hook`），hihat 只给计数列；字符集限定 `X x - .`；每行自包含；Header 加硬约束声明 + 分音使用统计 + `--level` 目标 note 总数区间；`song_sheet.md` 与 `song_analysis.json` 同源 | v2 §5.2 (a)–(e) / §5.3 |
| C | チモシー健康ジャズ 34–48 小节 raw 与 smoothed 系统性偏离 0.2–0.3 | **归一只做一次**（详见 §6 根因）；融合式换成 v2 §3.1 五项（质心移出）；高潮改五票 | v2 §3.1 / §3.2 |
| D | TransientTears 被标成 chorus 13–16（4 小节）+ pre_chorus/build 17–40（24 小节） | **边界三路投票**（all-in-one + SSM novelty + 重复段变化点，≥2 路同意才采纳，吸附到 4 小节乐句线）；标签用 v2 §2.4 日式词表双写；`pre_chorus` 按 §2.5 派生且**最多切 8 小节**；chorus 需 ≥2 路证据 + 45% 全局护栏；器乐曲走 `drop` 能量票 | v2 §2.4 / §2.5 / §3.2 |
| E | 只有一个 `primary_stem`，无规则依据 | 按 v2 §4.4 规则表给每段 `primary_stem`/`secondary_stem` + 逐条规则出处；降级规则；"整曲单轨"警告；逐小节特征补齐 v2 §4.1 清单 | v2 §4.1 / §4.4 |

## 1. 安装

```bash
# 仓库根目录
/opt/homebrew/bin/python3.12 -m venv .venv          # .venv 已在 .gitignore
.venv/bin/pip install -r tools/audio_analysis/requirements.txt
# 另需 ffmpeg / ffprobe：brew install ffmpeg
```

## 2. 用法

```bash
.venv/bin/python -m tools.audio_analysis \
    --audio ~/Desktop/self-charts/TransientTears/track.mp3 \
    --bpm 192 --first 1.875 --level 13.5 \
    --out out/TransientTears/
```

| 参数 | 说明 |
|------|------|
| `--audio` | 输入音频（mp3/wav） |
| `--bpm` | 用户给定 BPM（第 1 小节起） |
| `--first` | simai `&first`：谱面第 1 小节第 1 拍在音频中的秒数，**可为负** |
| `--level` | **新增**，目标定数（如 `13.5`）。给了才输出 note 总数区间（知识 004）与该定数的官方均值 note/小节；缺省用全库量级 9.0 |
| `--bpm-changes "33:180,65:155"` | 变速点（实验性，见 §7 限制） |
| `--beats-per-bar` | 每小节拍数，默认 4 |
| `--model` / `--device` | Demucs 模型与推理设备 |
| `--divisions` | 候选分音扫描顺序，默认 `4,8,12,16,24,32` |
| `--div-outlier-ratio` | **新增**，定 div 时允许多少比例的 onset 超出 τ。**默认 0.0 = 严格照 v2 的 max 残差口径**；调到 0.1 能让分音直方图重新有信息量，但那是偏离调研口径的做法（见 §6） |
| `--no-fine-div` | 彻底禁止 `{32}`（默认已有红线） |
| `--fusion-weights` | 如 `"loudness=0.25,onset=0.30,drums=0.20,voiced=0.15,flux=0.10"` |
| `--vote-weights` | 如 `"structure=0.45,novelty=0.25,energy=0.20,centroid=0.05,vocal=0.05"` |
| `--section-floor` | **v0.3**，段落尺度密度地板（默认 0.60）。n=8 配对标定的池化最优是 **0.125**（0.60 的 SSE 比最优差 10.3%），样本太小未采纳为默认 |
| `--bar-floor` | **v0.3**，小节尺度密度地板（默认 0.25；标定最优 0.020，仅差 2.4% SSE → 保留） |
| `--intro-outro-cap` | **v0.3**，intro/outro 建议密度的结构封顶比例（默认 0.60 × 全曲建议均值；设 0 关闭） |
| `--offset-search-beats` | **v0.3**，offset 搜索半径（拍），默认 0.40；**不得 ≥0.5**（半拍处是反拍），超了会被夹到 0.45 |
| `--no-allin1` | 跳过 all-in-one，边界只用 novelty + 重复段两路 |
| `--align-to-detected` | **默认关**。用 offset 校验找到的偏移构造分析网格 |
| `--force` | 忽略缓存，重跑解码与分离 |
| `--copy-plot-to` / `--copy-sheet-to` | 额外把 `plot.png` / `song_sheet.md` 复制一份到指定路径 |

## 3. 模块结构

| 文件 | 职责 |
|------|------|
| `grid.py` | 由 BPM+first 构造小节/拍/细分网格；变速点；**offset 校验**（纯 numpy） |
| `decode.py` | ffmpeg mp3→44.1k WAV；ffprobe 读时长与容器 `start_time` |
| `stems.py` | Demucs v4 四轨分离（MPS 失败自动回退 CPU），记录设备与耗时 |
| `onsets.py` | 逐 stem onset 检测；鼓件频带启发式分类；人声 VAD |
| `quantize.py` | **拍同步重采样 + 逐小节唯一 div + τ(d) + 三连判别 + {32} 红线** |
| `tracks.py` | **四条逻辑轨的构造与网格串渲染**（`X x - .`）；人声"有音高"判定 |
| `features.py` | **逐小节特征表**（v2 §4.1 清单） |
| `structure.py` | **边界三路投票 + 日式标签词表 + pre_chorus/chorus/final_chorus/rest 派生** |
| `intensity.py` | **五项融合强度 + 高潮五票 + 强度→建议密度映射** |
| `stemplan.py` | **段落 → 骨架轨 + 点缀轨 + 估计占比 + 依据**（v0.3；规则表见模块注释） |
| `sheet.py` | 输出 `song_analysis.json` / `song_sheet.md` / `plot.png` |
| `cli.py` | 串联全流程 |

## 4. 输出

```
out/<song>/
  track.44k.wav          # ffmpeg 统一解码产物（后续全部分析的时间基准）
  stems/{drums,bass,other,vocals}.wav
  song_analysis.json     # 全部机器可读结果（旧名 analysis.json 同时写出，保持兼容）
  song_sheet.md          # 给 LLM/人看的中文分析单（旧名 song-sheet.md 同时写出）
  plot.png               # 强度曲线 + 段落 + stem 活动 + 逐小节 div（人工复核用）
```

### 4.1 四条逻辑轨与字符集

| 轨 | 来源 stem | `X` | `x` | `-` |
|----|-----------|-----|-----|-----|
| `drum` | drums | kick | snare / 其他鼓件 | — |
| `vocal` | vocals | **有音高且强**的 onset | onset | **延音持续**（VAD 有声但无新 onset） |
| `bass` | bass | 强 onset（≥ 本轨 P70） | onset | — |
| `hook` | other | 强 onset | onset | — |

`.` = 空。**字符集只有这四个**。hihat 不单独给串（密集段里它与 kick 的串常常
完全一样），只在逐小节表给一个计数列。串长 = **该小节的 div**，四轨共用，可纵向对齐读。

### 4.2 `song_analysis.json` schema（v0.2）

| 字段 | 内容 |
|------|------|
| `schema_version` | `"0.3"` |
| `song` / `grid` / `warnings` / `stems` / `tools` | 同 v0.1 |
| **`offset_check`** | v0.3 新增 `search_beats`·`search_beats_clamped`·`sec_per_beat`·`half_beat_ms`·**`offbeat_ambiguous`**·**`tied_symmetric`**·`offbeat_reason`·`ratio_gate` |
| `offset_verdict` | v0.3 四档判词：一致 / 疑似 MP3 解码延迟（15–30 ms）/ 反拍歧义·显著性不足 → **不报警** / 报警 |
| **`target`** | `level` / **`anchor`**（`nps`）/ **`nps`**·`nps_source` / `notes_per_bar`（= NPS × 每小节秒数，全曲均值）/ **`notes_per_bar_level_anchor`**（旧口径对照）/ `total_mean`·`total_p10`·`total_p90`（知识 004）/ **`density_floor`**（含 n=8 标定最优值）/ **`structural_cap`**（intro/outro 封顶记录） |
| **`quantize`** | `division_histogram`（全部有 onset 的小节）/ **`division_histogram_resolved`**（真的被选中的）/ `resolved_bars` / `unquantized_onsets` / `unquantized_ratio` / `fine_blocked_bars` / `triplet_bars` / `per_track`（逐轨量化误差） |
| **`bar_divisions[bar]`** | `division` / `resolved` / `n_unquantized` / `triplet` / `fine_blocked` / `rms_ms` / `max_ms` / `reason` |
| `vocal_vad` | 阈值 / `global_voiced_ratio` / **`instrumental`**（器乐向判定） |
| **`structure.boundary_vote`** | 每个边界候选的 `bar` / `votes` / `sources`（哪几路提名） |
| **`structure.segments[]`** | `start_bar`/`end_bar`/`cluster`/`function`/**`label_ja`**/**`label_display`**（`サビ(chorus)`）/`is_repeat`/`repeat_of`/**`chorus_index`**/**`upgrade`**/**`rest`**/`intensity`/`intensity_tier`/`voiced_ratio`/**`density_norm`**/**`suggested_notes_per_bar`**/`suggested_division`/**`skeleton_stem`**/**`accent_stems[]`**/**`accent_share{}`**（含 `free`）/**`plan_evidence[]`**（纯音频特征依据）/`primary_stem`·`secondary_stem`（⚠️ v0.2 兼容字段，已弃用）/**`evidence`**（段落判定证据）/`notes`（规则出处与存疑标注） |
| `intensity` | 权重 / **`audio_climax_bar`**·**`climax_kind`**·**`climax_note`**（音频能量高潮，≠谱面密度峰）/ `climax_bar`·`climax_peaks`（兼容名）/ `bar_intensity`（平滑后）/ **`bar_intensity_raw`**（同基准、未平滑）/ `vote_total` / `density_floor` |
| **`bars[]`** | `division` / `resolved` / `n_unquantized` / `intensity` / `intensity_raw` / `density_norm` / `suggested_notes` / `patterns{drum,vocal,bass,hook}` / **`features{}`**（见下） |
| **`bars[].features`** | `share_*`、`n_onset_*`（含 kick/snare/hihat）、`n_onset_merged`、`grid_fit_*`（v2 的 `{8}` 口径）、`grid_fit_bar_*`（对齐到该小节实际 div）、`voiced_ratio`、`riff_sim_4`、`riff_sim_8`、`sil_run`、`hf_ratio` |

## 5. 实跑记录（2026-09-11，Apple M4 / 16GB / macOS 26 / py3.12 / torch 2.14）

三首曲子取自 `~/Desktop/self-charts/`（只读；BPM 与 `&first` 只从 `maidata.txt`
头部读取，**谱面正文未读取、未解析**）。stems 与 all-in-one 结果复用 v0.1 的缓存，
未重跑分离。全部用 `--level 13.5`。

### 5.1 总览

| 曲目 | 时长/小节 | BPM/first | 结构路径 | 段数 | 高潮小节 | 器乐向 | offset |
|------|-----------|-----------|----------|------|----------|--------|--------|
| TransientTears | 125.9s / 100 | 192 / 1.875 | vote(allin1+novelty+repeat) | 12 | 55（次峰 46、89） | 是（voiced 0.17） | +13.7 ms |
| チモシー健康ジャズ | 124.4s / 75 | 145 / 1.655 | vote(allin1+novelty+repeat) | 9 | 24（次峰 48、59） | 是（voiced 0.16） | +10.6 ms |
| 金魚鉢からの脱走 | 142.9s / 88 | 148 / 1.622 | vote(allin1+novelty+repeat) | 9 | 50（次峰 18、35） | 否（voiced 0.64） | +8.0 ms |

耗时（复用 stems 的热跑）：总 20.5–22.5 s，RTF **0.15–0.16**；其中 all-in-one 推理
17.6–19.2 s 占 86%，全部 DSP 环节（解码/offset/onset/量化/特征/强度）合计 < 3 s。
`--no-allin1` 可把 RTF 压到 ~0.03。

### 5.2 A 项：分音分布的变化

| 曲目 | v0.1（逐轨各选各的，7 条轨的直方图） | v0.2（每小节唯一 div） | 其中**真的被 τ 解释**的 |
|------|----------------------------------|----------------------|----------------------|
| TransientTears | drums `8×55/16×13/24×18/32×4`，other `24×26/32×14`，vocals `4×9…32×12` | `{8}×6`（7%）、`{16}×86`（93%） | 20/92 小节：`{8}×6`、`{16}×14` |
| チモシー健康ジャズ | drums `8×17/16×52/32×4`，bass `32×30`，other `32×32` | `{16}×74`（100%） | 27/74 小节：`{16}×27` |
| 金魚鉢からの脱走 | drums `16×43/32×18`，bass `32×44`，other `32×24`，vocals `32×28` | `{4}×4`、`{8}×3`、`{16}×78` | 11/85 小节：`{4}×4`、`{8}×3`、`{16}×4` |

**结论**：
1. **`{32}` 与 `{24}` 从输出里彻底消失**（v0.1 里 bass/other/vocals 有 30–44 个小节被判 `{32}`，全是噪声）。`{32}` 红线在三首曲子上分别拦下 21 / 9 / 16 个小节。
2. **三连小节 0 个**：三连门（`rms12 ≤ 0.7·rms16`）在这三首曲子上一次都没开，符合"音游曲绝大多数是二分体系"的先验。
3. **代价：`{16}` 变成压倒性多数**。原因是 div 由四条 stem 的 onset 并集决定（每小节中位 **15–22 个 onset**），一条 stem 的离群点就把整小节顶到 16。

### 5.3 A 项：unquantized 比例（v0.1 没有这个指标）

| 曲目 | 未落格 onset | 比例 | 走"压到上限"分支的小节 | 逐轨量化误差（平均，ms） |
|------|-------------|------|----------------------|------------------------|
| TransientTears | 196 / 1498 | **13.1%** | 72 / 92 | drums 10.6、bass 10.4、other 17.6、vocals 15.3 |
| チモシー健康ジャズ | 109 / 1589 | **6.9%** | 47 / 74 | drums 7.4、bass 17.5、other 17.5、vocals 14.6 |
| 金魚鉢からの脱走 | 221 / 1186 | **18.6%** | 74 / 85 | drums 7.7、bass **33.3**、other 15.6、vocals 20.0 |

v0.1 报的是 `unresolved_bars = 0`，那是**容差过宽造成的假象**——25 ms 容差比
`{32}` 格距的一半（19.5 ms）还大，任何 onset 都能被 `{32}` "解释"。v0.2 把这个
假象换成了一个**可被检验的数字**：7–19% 的 onset 落不到 ≤`{16}` 的合法格线上。

主要责任方是**非鼓轨的 onset 定位精度**：drums 平均误差 7–11 ms（好），
bass/other/vocals 是 15–33 ms（差，且 basic-pitch 未装上，没有 note 事件可用）。
`--div-outlier-ratio 0.1` 能把 TransientTears 的 `{8}` 占比从 7% 拉回 16%，
但那**偏离 v2 §1.1 Step 3 的 `max` 残差口径**，因此默认关闭。

### 5.4 C 项：raw/smoothed 系统性偏离的根因（已定位并修复）

**根因 = 两次不同的归一，不是平滑本身。**

v0.1 的 `bar_raw` 是**未归一**的融合值，`bar_intensity` 是「平滑 → 再做一次
min-max」的值；`sheet.plot_overview` 画图时把 raw 画成 `bar_raw / bar_raw.max()`
（下界留在 `min/max`），把 smoothed 画成拉满 `[0,1]` 的那条。两条线基准不同，
偏离量恰好是 `min/max`，**在低强度段最大**。

用 v0.1 的代码路径原样复算（同一份 wav / stems）：

| 曲目 | 全曲 mean(raw_plot − smoothed_plot) | 34–48 小节 | 逐小节偏离 |
|------|-----------------------------------|-----------|-----------|
| チモシー健康ジャズ | +0.115（max +0.626） | **+0.256** | 0.17 / 0.20 / 0.23 / 0.24 / 0.29 / 0.33 / 0.34 / 0.37 / 0.41 / 0.51 … |
| TransientTears | +0.001（max +0.228） | — | — |

**+0.256 精确对上主会话报的"0.2–0.3"**，且只在チモシー这种"弱段占比大"的曲子上
显著（TransientTears 的曲线本来就贴顶，`min/max` 小，全曲均值只有 +0.001）——
这也解释了为什么只有一首曲子暴露了这个 bug。

**修法**：归一只做一次——先把融合值做鲁棒 min-max（P5–P95 截断）得到
`bar_raw ∈ [0,1]`，**再**平滑，平滑后不再归一。修完三首曲子的
`mean(smoothed − raw)` 分别是 −0.0005 / −0.0018 / +0.0030（≈0，无电平漂移），
チモシー 34–48 小节从 +0.256 变成 −0.016。
回归测试：`test_smoothing_does_not_shift_level`（同时把 v0.1 的错误做法作为反例断言）。

### 5.5 D 项：结构标签的变化

**TransientTears**（v0.1 的翻车重灾区）

| | v0.1（ssm-fallback） | v0.2（三路投票） |
|---|---|---|
| 段数 | 11 | 12 |
| 错误 | chorus **13–16（4 小节）**；pre_chorus/build **17–40（24 小节）**；45–68 又是 24 小节的 build | 无 4 小节副歌、无 24 小节 build；**判为器乐向**（voiced 0.17）→ 不再乱贴 chorus，改用 ドロップ |
| 结果 | — | 1–4 イントロ / 5–12 間奏 / 13–16 間奏 / 17–40 Aメロ / 41–56 Aメロ / **57–68 ドロップ(I=0.92)** / 69–72 間奏 / 73–78 間奏 / 79–90 Aメロ / 91–94 間奏 / 95–98 間奏 / 99–100 アウトロ |

高潮小节从 78 移到 **55**（紧邻 57 起的 ドロップ 段），与强度曲线的形状一致
（见 `plot.png`：曲线在 55 冲顶后在 57–68 维持 1.0）。

**チモシー健康ジャズ**：v0.1 是 all-in-one 原样 5 段（intro 1–17 / chorus 18–33 /
solo 34–48 / chorus 49–67 / outro 68–75）。v0.2 给 9 段，并把
**45–48 派生成 Bメロ(pre_chorus)**（强度 0.37，位于 49 起的 ドロップ 前 4 小节，
kick 密度下降）、49–60 与 67–72 两次 ドロップ（第二次 `upgrade=✅`）、
61–66 标 `rest=✅`。这是 v0.1 完全给不出的信息（all-in-one 没有 prechorus 标签）。

**金魚鉢からの脱走**（唯一的人声曲，voiced 0.64）：
1–8 イントロ / 9–24 Aメロ / **25–32 落ちサビ**（rest）/ **33–44 サビ #1** /
**45–54 ラスサビ #2**（upgrade，repeat_of 33–44）/ 55–64 間奏 / 65–72 Aメロ /
**73–84 落ちサビ**（rest）/ 85–88 アウトロ。
v0.1 在这首上给的是 all-in-one 的 `intro/verse/chorus/solo/chorus/end`，
没有 `repeat_of`、没有 `chorus_index`、没有 `upgrade`、没有休息段。

**边界投票的实际效果**：三路提名数（TransientTears）为 all-in-one 7 / novelty 12 /
重复段 42，其中 **11 个边界获 ≥2 路同意**，没有触发 novelty 补齐。

### 5.6 E 项：切轨建议

- **TransientTears**：`hook`（1–68、79–90）↔ `drum`（73–78、91–100）交替，
  切轨 4 次，未触发"整曲单轨"警告；intro 的"哪个响踩哪个"逐 4 小节重算结果也写在备注里。
- **チモシー健康ジャズ**：`drum` 为主，25–44 切 `hook`；45–48 的 Bメロ 备注给出
  "段首踩鼓、最后 1–2 小节切人声"，但随即被副踩可用性检查取消（器乐曲无人声可混）——
  这条降级链在备注里是显式的。
- **金魚鉢からの脱走**：`hook`（intro）→ `drum`（Aメロ）→ **`vocal`（サビ/ラスサビ）**
  → `drum`（間奏）→ `vocal`（落ちサビ），完全符合 MMFC 5.4「副歌全踩人声」。

三首都没有触发"整曲单轨"警告。

### 5.7 v0.3 实跑（2026-09-11，11 首：8 首官方曲包 + 3 首测试曲）

**复用 `out/` 里已缓存的 wav 与 Demucs stems，未重跑分离**；8 首官方曲的 mp3 与谱面在
`out/calib/`（gitignore，不入库），3 首测试曲只用 `~/Desktop/self-charts/*/track.mp3`
的音频，**谱面正文未读**（AGENT 准则 11）。

#### A 项：offset 反拍假警报修复前后

| 曲目 | BPM | 半拍 (ms) | v0.2 φ\* | **v0.3 φ\*** | 同级峰 | ratio | v0.3 判词 |
|---|---:|---:|---:|---:|---:|---:|---|
| Back 2 Back | 150 | 200.0 | +7.7 | **+7.7** | 1 | 1.19 | 一致 |
| MYTHOS | 158 | 189.9 | +10.8 | **+11.0** | 1 | 1.39 | 一致 |
| Mare Maris | 150 | 200.0 | **+27.9** | **+27.9** | 1 | 15.37 | **疑似 MP3 解码延迟**（保留报出，措辞按 R2 改） |
| **Signature** | 128 | 234.4 | **−229.0**（假警报） | **+0.6** | 1 | 1.00 | 一致 ✅ 修复 |
| Xevel | 176→185 | 170.5 | +7.2 | **+7.6** | 6 | 1.03 | 一致（对称同级峰 → 注明参考价值打折） |
| 初音ミクの激唱 | 200 | 150.0 | +4.1 | **+4.1** | 1 | 1.04 | 一致 |
| 幻想のサテライト | 230 | 130.4 | +7.2 | **+7.1** | 1 | 1.16 | 一致 |
| **麒麟** | 220 | 136.4 | **−129.8**（假警报） | **+10.6** | 1 | 1.29 | 一致 ✅ 修复 |
| TransientTears | 192 | 156.3 | +13.7 | **+13.9** | 1 | 1.81 | 一致 |
| チモシー健康ジャズ | 145 | 206.9 | +10.6 | **+10.5** | 1 | 1.50 | 一致 |
| 金魚鉢からの脱走 | 148 | 202.7 | +8.0 | **+8.0** | 1 | 1.21 | 一致 |

- **两个假警报（Signature −229.0 ≈ −半拍 234.4、麒麟 −129.8 ≈ −半拍 136.4）全部消失**，
  φ\* 回到 +0.6 / +10.6 ms；
- **Mare Maris 的 +27.9 ms 仍被报出**（ratio 15.37，远高于 1.15 的显著性门），
  但判词从"建议人工复核"改成"**这个量级几乎都是 MP3 解码/容器延迟**"；
- 其余 9 首的 φ\* 变化 ≤ 0.4 ms（搜索窗变窄不影响真值附近的峰）；
- Xevel 是打分曲线平坦的个案（z=1.7、6 个同级峰对称）：不报「反拍歧义」（φ\* 只有 +7.6 ms），
  但判词里注明参考价值打折。
- 回归测试：`test_offbeat_false_alarm_is_reproduced_with_legacy_window`（合成"反拍略强于
  正拍的 8 分 hi-hat"，用 `search_beats=1.0, allow_offbeat_window=True` **复现**出 ±半拍的
  假警报）+ `test_offbeat_false_alarm_fixed_by_default_window`（默认窗口下 |φ\*| ≤ 25 ms 且不报警）。

#### B 项：骨架 + 点缀的实际输出

11 首共 **122 段**：骨架 **drum 114 段（93.4%）**，例外 8 段（幻想 2 段 hook / TransientTears 4 段
hook / Mare Maris 1 段 vocal / 激唱 1 段 bass，全部是"鼓 onset 太少或不落格"触发的降级）。

**Signature（BPM 128，定数 13.5）** —— 非人声主导曲目的典型：

| 小节 | 段落 | 骨架 | 点缀 | 估计占比 |
|---|---|---|---|---|
| 13–28 | サビ(chorus) #1 | drum | hook ＞ vocal | drum 0.55 / hook 0.15 / vocal 0.11 / free 0.19 |
| 29–36 | ラスサビ(final_chorus) #2 | drum | hook ＞ vocal | drum 0.55 / hook 0.16 / vocal 0.10 / free 0.19 |
| 55–62 | 間奏(interlude) | drum | bass | drum 0.69 / bass 0.12 / free 0.19 |

副歌段的判词：「本段**不是人声主导**（voiced=0.58、vocals/drums onset 密度比=0.43）→ 维持
**drums 骨架**，人声只作重音点缀」+ 存疑标注。v0.2 在这里会无条件答 `vocal`。

**幻想のサテライト（BPM 230，定数 14.1）** —— 标定里**唯一**的"副歌踩人声"正例：

| 小节 | 段落 | 骨架 | 点缀 | 判定 |
|---|---|---|---|---|
| 49–62 | サビ #1 | drum | **vocal ＞ hook** | 人声主导（voiced=1.00、密度比 0.87） |
| 89–96 | サビ #2 | drum | hook | 密度比 0.68 达标，但 vocals 落格率 0.30 < 0.5 → 人声点缀被降级规则去掉 |
| 97–112 | サビ #3 | drum | **vocal ＞ hook** | 人声主导（密度比 0.72） |
| 113–126 | 間奏 | **hook** | — | ⚠️ 骨架降级：该段 drums onset 均值 0.2 < 2 |

→ 条件触发在唯一已知正例上**确实会点亮**，在其余 7 首上一次都没点亮。
⚠️ 但判据阈值（密度比 0.65）是在这 8 首上分出来的（幻想 0.68–0.87 vs 其余 ≤ 0.64），
**属 n=8 调参，存疑**。

#### C 项：NPS 锚 vs note/小节 锚（8 首官方曲，与官方谱实际 note/小节 对比）

| 锚点 | MAPE |
|---|---:|
| **NPS 锚（v0.3 默认）** | **22.5%** |
| note/小节 锚（v0.2，对照列） | 29.3% |

方向与标定报告 §4.2（16.5% vs 21.0%）一致（绝对值不同：本表用"定数取最近档 + 全曲均值"口径）。
改善最大的是高 BPM 曲：幻想 10.54 → **7.45**（实际 6.85）、麒麟 10.54 → **7.79**（实际 7.22）。
⚠️ MYTHOS 反而变差（10.10 → 11.70，实际 9.67），Signature 也略差 —— **NPS 锚不是每首都更准**。

#### D 项：intro/outro 结构封顶

11 首里 **9 首**触发（封顶 3–17 小节不等），MYTHOS 与 TransientTears 因前奏本来就不高而未触发。
例：麒麟封顶 12 小节（intro 1–18 的建议密度被压到 ≤ 3.53 note/小节）。

## 6. 已知限制（按可靠性从低到高）

1. **分音判定被"四轨并集"稀释**。div 由该小节全部 stem 的 onset 并集决定（v2 §1.1 的
   口径），而每小节中位有 15–22 个 onset、非鼓轨定位误差 15–33 ms，导致 78–87% 的小节
   走"找不到合法分音 → 压到 `{16}`"的分支。**`song_sheet.md` 的 Header 会显式区分
   「真的被选中」与「被压到上限」两种 `{16}`**，后者的网格串只是近似，不是分音结论。
   → 真正的解法是提高 bass/vocals 的 onset 精度（装上 basic-pitch，用 note onset 代替
   能量 onset），不是放宽容差。
2. **鼓件分解（kick/snare/hihat）是频带启发式，不是鼓转录**（同 v0.1）。底鼓 vs 低音 tom、
   军鼓 vs 拍手、hihat vs 镲片与齿音泄漏都会混淆。
3. **人声 VAD 与"有音高"判定都是零模型替代**。VAD 阈值是"整曲 vocal RMS 峰值 −32 dB"；
   `X`（有音高且强）用 librosa `yin` + 谐波能量比判定，不是 basic-pitch 的 note 事件。
   实测三首曲子的 vocals stem 能量占比分别是 1.4% / 0.1% / 20%，前两首被正确判为
   **器乐向**，但这个 0.18 的阈值只在 3 首曲子上验证过。
4. **器乐曲的标签只能沿强度轴分档**（I≥P75 → ドロップ、≥P60 → Aメロ、其余 → 間奏）。
   器乐曲没有"副歌"概念，这是信息本身的限制，不是实现问题；但档位阈值是拍脑袋的。
5. **chorus 的 45% 全局护栏是启发式**。金魚鉢 上有 6 段拿到 ≥2 路证据（整曲 voiced 0.64
   让"人声活动"票几乎人人有份），靠这条护栏砍到 4 段。护栏本身没有实证依据。
6. **强度融合权重与五票权重仍是 v2 的初值**——2026-09-11 已用 8 首官方音频×官方谱做过
   配对标定，结论是**维持初值**（LOSO 交叉验证 0.515 vs 初值 0.488，5/8 折胜出、
   符号检验 p≈0.36，不满足"明显优于"；见标定报告 §3）。两条待验假设留在
   `intensity.py` 注释里：`voiced`→0（单项 ρ 0.016）、`flux` 0.10→0.25（单项 ρ 0.524，
   五项最高），**须等 ≥20 首配对数据**再验，不得凭 n=8 改动。
7. **"建议 note/小节"精度只到档位**：8 首配对实测逐小节 MAE 2.07–4.69（均值 3.4
   note/小节，相对误差 30–45%）→ 它是 **±3 note 的粗档**。v0.3 已把绝对量级锚点换成
   NPS（MAPE 22.5% vs 旧口径 29.3%，本机 8 首复算），但**锚点本身仍只在 n=8 上验证过，
   且不是每首都更准**（MYTHOS、Signature 反而变差）。段落 floor 0.60 与 n=8 的池化最优
   0.125 冲突，**默认未改**（388 谱 vs 8 首，以大样本为准）。
8. **intro/outro 的结构封顶比例 0.60 是个案推广**：来源是 MYTHOS 一首（前奏音频强度≈1.0、
   官方谱只放 4 note/小节）。方向（前奏密度由结构定、不跟能量走）有依据，**0.60 这个数
   没有**；`--intro-outro-cap` 可调。
9. **`stemplan` 的「人声主导」阈值是 n=8 调出来的**（voiced ≥0.45 且 vocals/drums onset
   密度比 ≥0.65）：8 首里只有幻想のサテライト在 0.65 以上。**存疑，且 C1 结论本身待人工听审**
   （ground truth 是 onset 代理，人声侧最不可靠）。逐段"估计占比"从未被验证，只是量级提示。
10. **offset 只能定到一拍以内**（同 v0.1）；v0.3 把搜索窗收到 ±0.4 拍后，**真实偏移超过
   0.4 拍的曲子会测不出来**（这是为消除反拍假警报付出的代价，在"用户提供 offset"的前提下
   可接受）；**变速是实验性的**（同 v0.1）。
11. **basic-pitch 仍未装上**（pip `resolution-too-deep`，见 v0.1 记录）。人声/贝斯只有
   能量 onset，没有音高/时值 note 事件——这是限制 1 与 3 的共同上游。

## 7. 测试

```bash
.venv/bin/python -m pytest tests/test_audio_analysis.py -q   # 81 passed
.venv/bin/python -m pytest tests/ -q                          # 164 passed（+ chart_analysis 42 + calibration 41）
```

全部用 **numpy 合成音频/合成事件**（已知 BPM/first/分音的 click 序列与合成结构曲线），
仓库内不放任何真实音频（AGENT.md 准则 5）。

**v0.3 新增覆盖**：
- **offset 反拍回归**：合成"反拍略强的 8 分 hi-hat" → 旧窗口（±1 拍）复现 ±半拍假警报、
  默认窗口（±0.4 拍）修复；搜索半径超 0.45 拍被夹紧；只有反拍有音时判「反拍歧义」不报警；
  判词四档（一致 / 解码延迟 / 显著性不足不报警 / 报警）；对称同级峰只降置信不改判词。
- **骨架+点缀**：骨架默认 drums；副歌人声**条件触发**（主导 / 非主导两条路径）；
  规则 4/5/6 的存疑标注；intro 用匹配倾向而非能量占比；估计占比含 `free` 残差且合计≈1；
  鼓不可用时骨架降级；第 N 次副歌复用整套计划；outro 镜像 intro；单轨警告；依据只含音频特征。
- **密度锚**：NPS 表与知识 031 一致；NPS→note/小节 随 BPM 缩放（高 BPM 不再高估）；
  段落 floor 可参数化（0.60 / 0.125）；intro/outro 结构封顶只封不抬。

v0.2 覆盖：

- **量化**：τ(d) 单调且被 clamp；v0.1 容差为何让 24/32 不可分；1/48 拍重采样的整数格；
  三连门接受真三连 / 拒绝二分小节；`{32}` 红线三条件（非鼓拦下 / 鼓且 onset≥6 放行 /
  onset 不足拦下 / `--no-fine-div`）；**同一小节四轨串长一致**；unquantized 统计；
  末尾吸附；端到端（合成 click → librosa 检测 → 还原 16 分）。
- **网格串**：字符集只含 `X x - .`；同格冲突取强者；`-` 延音；kick→`X`；
  人声需"有音高且强"才 `X`；轨数上限 4。
- **强度**：**C 项 bug 回归**（平滑不得改变电平，且把 v0.1 的两次归一作为反例断言）；
  Z-score 与鲁棒归一；融合权重/票权重与 v2 一致（质心已移出融合项）。
- **密度映射**：地板生效（I=0 时段落 0.60、小节 0.25）；定数 → note/小节 与 note 总数区间。
- **结构**：边界投票需 ≥2 路、补齐记在 `topped_up`、吸附到 4 小节线；日式标签双写；
  `pre_chorus` 派生（**≤8 小节**且前一段被切短而不是整段变 Bメロ）；chorus ≥2 路证据；
  `final_chorus`/`upgrade`/`chorus_index`/`repeat_of`；器乐曲走 `drop`；`rest` 标记。
- **切轨**：副歌踩人声；主歌踩 other + 小节尾人声；outro 与 intro 同轨；
  第 N 次副歌与 `repeat_of` 同轨 + upgrade；不可用轨降级到 drums；整曲单轨警告（含曲短不报警）。
- **逐小节特征**：`grid_fit`、去重合并 onset 计数、能量占比和为 1、静默段检测、
  `pitched_mask` 能分辨正弦与噪声。
