# tools/audio_analysis — 歌曲分析单生成器（原型 v0.1）

把一首 mp3 变成 **LLM 可直接消费的结构化「歌曲分析单」**：段落划分、逐小节强度、
逐小节各 stem 的 onset 网格串。对应 `docs/audio-analysis.md` v1.0 管线的 **L1–L3 层**
（L4 规划层与生成器不在本工具范围内）。

> **前提**：BPM 与 offset（simai `&first`）**由用户给定**。本工具**不做节拍追踪**，
> 只对用户给的 offset 做一致性校验并**如实报告差值，绝不覆盖用户值**。

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
    --bpm 192 --first 1.875 \
    --out out/TransientTears/
```

| 参数 | 说明 |
|------|------|
| `--audio` | 输入音频（mp3/wav） |
| `--bpm` | 用户给定 BPM（第 1 小节起） |
| `--first` | simai `&first`：谱面第 1 小节第 1 拍在音频中的秒数，**可为负** |
| `--bpm-changes "33:180,65:155"` | 变速点（实验性，见 §6 限制） |
| `--beats-per-bar` | 每小节拍数，默认 4 |
| `--model` | Demucs 模型，`htdemucs`（默认，快）/ `htdemucs_ft`（慢、质量更好） |
| `--device` | `auto`（默认，Apple Silicon 上取 mps）/ `mps` / `cpu` / `cuda` |
| `--divisions` | 候选分音，默认 `4,8,12,16,24,32` |
| `--tolerance-ms` | 量化容差下限，默认 25（实际取 `max(该值, 1/64 小节)`） |
| `--fusion-weights` | 强度融合权重，如 `"rms=0.35,onset=0.30,centroid=0.20,drums=0.15"` |
| `--vote-weights` | 高潮投票权重，如 `"structure=0.4,novelty=0.3,energy=0.2,centroid=0.1"` |
| `--no-allin1` | 跳过 all-in-one，直接走自研 SSM 分段 |
| `--skip-stems` | 跳过 Demucs（调试用） |
| `--align-to-detected` | **默认关**。用 offset 校验找到的偏移构造分析网格（小节编号不变，整体平移）。仅在确认存在解码偏移时使用 |
| `--force` | 忽略缓存，重跑解码与分离 |
| `--copy-plot-to` | 额外把 plot.png 复制一份到指定路径 |

## 3. 模块结构

| 文件 | 职责 |
|------|------|
| `grid.py` | 由 BPM+first 构造小节/拍/细分网格；变速点；**offset 校验**（纯 numpy） |
| `decode.py` | ffmpeg mp3→44.1k WAV；ffprobe 读时长与容器 `start_time` |
| `stems.py` | Demucs v4 四轨分离（MPS 失败自动回退 CPU），记录设备与耗时 |
| `onsets.py` | 逐 stem onset 检测（backtrack，按 stem 调参）；鼓件频带启发式分类；人声 VAD |
| `quantize.py` | onset → 拍网格：逐小节选最小可解释分音，输出网格字符串与误差统计 |
| `structure.py` | 结构分段：all-in-one 主线 + 自研 SSM 退路 + 分段可用性体检 |
| `intensity.py` | 强度曲线八步配方 + 高潮四票投票 |
| `sheet.py` | 输出 analysis.json / song-sheet.md / plot.png |
| `cli.py` | 串联全流程 |

## 4. 输出

全部写到 `--out` 指定目录：

```
out/<song>/
  track.44k.wav      # ffmpeg 统一解码产物（后续全部分析的时间基准）
  stems/{drums,bass,other,vocals}.wav
  analysis.json      # 全部机器可读结果
  song-sheet.md      # 给 LLM/人看的中文分析单
  plot.png           # 强度曲线 + 段落边界 + 各 stem 活动度（人工复核用）
```

### `analysis.json` schema（v0.1）

| 字段 | 内容 |
|------|------|
| `song` | 曲名、源文件、wav、时长、容器 `start_time`（MP3 编码器延迟） |
| `grid` | bpm / first（用户值）/ analysis_first（实际用的）/ beats_per_bar / 变速点 / 小节数 |
| `offset_check` | `method`（`onset-fit` 或 `envelope`）、`best_first`、`delta_ms`、`confidence_z`、`peaks`（整拍歧义的同级峰）、`envelope_delta_ms` |
| `offset_verdict` | 一句中文结论 |
| `warnings` | 变速、`--align-to-detected` 等需要人工注意的事项 |
| `stems` | 模型、设备、耗时 |
| `onset_tracks` | 每条 onset 流的数量与是否启发式 |
| `vocal_vad` | 阈值、有人声的小节比例 |
| `quantize_stats` | 逐 stem：onset 数、平均/P95/最大量化误差、无法解释的小节数、分音直方图 |
| `structure` | `method`（`allin1_infer` / `allin1` / `ssm-fallback`）、`notes`、`segments`、`alt_segments`（被否决的那一版） |
| `structure.segments[]` | `start_bar` / `end_bar` / `label` / `function` / `is_repeat` / `repeat_of` / `intensity` / `intensity_tier` / `primary_stem` |
| `intensity` | 权重、响度归一信息、`climax_bar`、`climax_peaks`、`bar_intensity[]`、`vote_total[]` |
| `bars[]` | 逐小节：`start_sec` / `bpm` / `intensity` / `intensity_tier` / `vocal_activity` / `drum_onsets` / `finest_division` / `onsets{}` / `divisions{}` / `patterns{}` / `quant_unresolved[]` |
| `timings_sec` / `rtf` | 各阶段耗时与实时率 |
| `tools` | 各依赖版本（复现实验用） |

### 网格字符串

```
16 分：x...x..x....x...     . = 空   x = 有 onset   X = 强 onset（强度 ≥ 该轨 P70）
```

串长 = **该轨该小节选中的分音**（每小节等分数，对应 simai 的 `{分音}`）。
逐小节表里的「最小分音」= 该小节各轨中最细的那个。

## 5. 实跑记录（2026-09-11，Apple M4 / 16GB / macOS 26 / py3.12 / torch 2.14）

三首曲子取自 `~/Desktop/self-charts/`（只读；BPM 与 `&first` 只从 `maidata.txt`
头部读取，**谱面正文未读取、未解析**）。

| 曲目 | 时长 | 小节 | BPM / first | 结构路径 | 段数 | 高潮小节 | offset 校验 |
|------|------|------|-------------|----------|------|----------|-------------|
| TransientTears | 125.9s | 100 | 192 / 1.875 | **ssm-fallback**（all-in-one 只给 3 段被体检否决） | 11 | 78（次峰 46、90） | +13.7 ms，z=3.04 |
| チモシー健康ジャズ | 124.4s | 75 | 145 / 1.655 | allin1_infer | 5 | 53（次峰 27、61） | +10.6 ms，z=3.49 |
| 金魚鉢からの脱走 | 142.9s | 88 | 148 / 1.622 | allin1_infer | 6 | 38（次峰 47、75） | +8.0 ms，z=3.49 |

### 各阶段耗时（冷跑，含 Demucs；RTF = 耗时 / 音频时长）

| 阶段 | 耗时 | RTF | 备注 |
|------|------|-----|------|
| ffmpeg 解码 | 0.07–0.31 s | ~0.002 | |
| offset 校验 | 0.85–1.1 s | ~0.008 | 整曲 onset 检测 + ±1 拍 / 5ms 步长搜索 |
| Demucs htdemucs | 12.6–13.8 s | **0.10–0.11** | **设备 = MPS**（未回退 CPU）；首次含权重下载 40s |
| 逐 stem onset + 鼓件 + VAD | 0.37–0.43 s | ~0.003 | |
| 量化 | 0.012–0.013 s | ~0.0001 | |
| 结构分段 | 16.4–20.0 s | 0.13–0.14 | all-in-one 推理；纯 SSM 退路只要 ~2 s |
| 强度曲线 | 0.14–0.17 s | ~0.001 | |
| **整条管线（冷跑）** | **36.0 s**（TransientTears） | **0.286** | 热跑（复用 stems）18–21 s，RTF 0.14–0.15 |

耗时结论：**Demucs + all-in-one 两个深度模型吃掉 >90% 的时间**，其余全部 DSP 环节
加起来不到 2 秒。`--no-allin1` 可把总 RTF 压到 0.12 左右。

### offset 校验结论

三首曲子的 `onset-fit` 结果都落在 **+8 ~ +14 ms**（阈值 15ms 内判为一致），
说明用户给的 `&first` 与音频对得上。同时两个发现值得记档：

1. **包络法有 +10ms 系统性滞后**。最初用 librosa `onset_strength` 包络直接打分，
   三首曲子一律报 +36 ~ +40 ms；在**合成 click 音频**（真值已知 first=0.5）上实测，
   包络法给出 0.5107（+10.7 ms），而 backtrack 后的 onset 时间只差 −4.9 ms。
   故改用 **onset-fit**（backtrack 后的 onset 时间对拍线加权命中率）作为主打分方式，
   包络法结果仍作为 `envelope_delta_ms` 一并输出。
2. **MP3 容器 `start_time` ≈ 23–25 ms**（ffprobe 实测三首都有），正是设计文档 §2
   警告的"MP3 解码偏移 20–40ms"。这部分偏移在不同播放器/解码器之间会表现不一致，
   分析单里已显式列出，供人工判断。

反向验证：拿 `--align-to-detected` 用包络法给的 +40ms 重建网格，量化平均误差从
9.47ms **劣化**到 13.46ms、分音直方图从「8 分为主」塌成「24/32 分为主」——
证明用户给的 1.875 才是对的，包络法的 +40ms 是伪信号。

### 量化误差

| 曲目 | drums 平均/最大误差 | 无法解释的小节 |
|------|--------------------|----------------|
| TransientTears | 9.47 / 24.9 ms | 0 |
| チモシー健康ジャズ | 7.35 / 24.8 ms | 0 |
| 金魚鉢からの脱走 | 6.78 / 25.1 ms | 0 |

「无法解释的小节」全曲为 0，但这**不等于量化准确**——见 §6 第 1 条。

### 装包失败清单

| 包 | 结果 | 报错要点 |
|----|------|----------|
| `librosa` / `soundfile` / `torch` / `torchaudio` / `demucs` / `pyloudnorm` / `matplotlib` / `numpy` / `scipy` / `pytest` | ✅ 全部装上 | — |
| `allin1`（官方 all-in-one） | ⚠️ 装上但**不可用** | `import allin1` → `ModuleNotFoundError: No module named 'madmom'`。官方版硬依赖 madmom，而 madmom 未装（且其模型权重是 CC BY-NC-SA，属设计文档 §7.2 的许可证红线） |
| `all-in-one-infer`（openmirlab） | ✅ 可用 | 提供模块 `allin1_infer`，纯 PyTorch，**支持直接喂我们已跑好的 Demucs stems**（`create_stems_input_from_directory`），省掉一次重复分离。⚠️ 但它会拖进 `madmom-infer`，对外发布前需复核该依赖是否触及 NC 权重 |
| `natten`（allin1 依赖） | ✅ 源码编译成功 | Apple Silicon + py3.12 上 `pip` 自建 wheel 通过，没出现设计文档预期的"NATTEN 自编译坑" |
| `basic-pitch[onnx]` | ❌ **未装上** | 两次尝试都是 `error: resolution-too-deep — Dependency resolution exceeded maximum depth`（pip 在 `narwhals` 上无限回溯；第二次已加 `narwhals>=2.26` 与 `numpy>=2` 约束仍然失败，中途还去源码编译 `cffi`）。超出 15 分钟时限后放弃，**核心环境未受影响**。**已按预案用 vocals stem 的 onset 代替人声 note 事件** |

## 6. 已知限制（原型阶段，按可靠性从低到高排列）

1. **量化的分音选择在高 BPM 下不可靠**。容差是 `max(25ms, 1/64 小节)`；192 BPM 下
   一小节 1.25 s，24 分与 32 分格子只差 ~13 ms，**远小于容差**，于是"最小可解释分音"
   在 16/24/32 之间基本是随机挑。`unresolved_bars = 0` 只说明总能找到一个分音，
   不说明那个分音是对的。→ 高 BPM 曲目请把网格串当"节奏轮廓"，别当谱面分音的依据。
2. **鼓件分解（kick/snare/hihat）是频带启发式，不是鼓转录**。做法是先在 drums stem
   上检出 onset，再按 20–150 / 150–800 / >5000 Hz 的相对能量给每个 onset 贴标签
   （一个 onset 可带多个标签）。底鼓 vs 低音 tom、军鼓 vs 拍手/边击、hihat vs
   镲片与人声齿音泄漏都会混淆；密集段里 kick 与 hihat 的网格串常常完全一样。
3. **结构分段两条路都会翻车**。all-in-one 在 TransientTears 上只给 3 段（最长段占
   全曲 63%），已被"分段体检"（段数 ≥5 且最长段 ≤35% 小节）拦下改走 SSM；SSM 退路
   自己也需要在边界数与簇数上自适应重试才能避免"整首歌都是 B 段"。被否决的那一版
   保留在 `structure.alt_segments` 里。**段落表必须人工过目 plot.png 复核。**
4. **人声 VAD 会被合成器/和声泄漏骗**。阈值是"整曲 RMS 峰值 −32 dB"，Demucs 的
   vocals 轨常混进 lead synth，导致器乐段被判为有人声。
5. **offset 只能定到一拍以内**。整拍平移给出几乎相同的分数，所以搜索窗内会出现多个
   同级峰，工具取"离用户值最近"的那个并在 `peaks` 里列出全部候选。
6. **变速（`--bpm-changes`）是实验性的**。网格只在**小节边界**换 BPM；若原曲变速点
   不落在小节线上，其后所有小节时间都会错位。用了该参数会在 `warnings` 里报警。
7. **强度曲线配方的第 8 步（模板 T=[0.20,0.45,0.30,1.00,0.10] 的 DTW 对齐）未实现**，
   段落强度档位直接取曲线的逐小节聚合值。融合权重与投票权重都是设计文档的初值，
   **尚未在本项目曲库上标定**（设计文档 §9 存疑第 2 条仍然成立）。
8. **basic-pitch 缺席**：人声旋律没有 note 事件（音高/时值），只有 vocals stem 的
   onset 与活动度。

## 7. 测试

```bash
.venv/bin/python -m pytest tests/test_audio_analysis.py -q     # 27 passed
```

全部用 **numpy 合成音频**（已知 BPM/first/分音的 click 序列），仓库内不放任何真实音频
（AGENT.md 准则 5）。覆盖：网格时间计算（含负 first、变速点）、分音选择与量化误差、
末尾吸附、offset 校验能找回已知偏移（含整拍歧义）、人声 VAD。
