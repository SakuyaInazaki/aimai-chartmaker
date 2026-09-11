# 043 — 分轨细化：六路分离 + 有音高 note 轨 + 高频瞬态轨（原型 v0.5）

- **日期**：2026-09-12
- **来源**：**用户**（2026-09-12）——「采样的话不能只有人声和鼓点两种」
- **关联**：note **042**（官方音频配对标定 n=40）；报告 `docs/research/stem-refinement-n40.md`

## 理由

两条独立证据指向同一处：

1. **用户的判断**：官方谱大量踩**合成器主旋律、钢琴、吉他 riff、采样音效**，
   现在全塞在 Demucs `other` 一路里；而能量 onset 抓不到**长音 / 分解和弦 / legato 换音**。
2. **n=40 标定留下的两个缺口**（`audio-chart-calibration-n40.md`）：
   `other` recall 只有 0.267；**16.8% 的官方 note 什么 stem 都不落**；
   以及 §9-A.5 自己给出的判断「下一轮最该补的不是更多曲子，是更好的人声 ground truth」。

## 变更

### 新增文件

- **`tools/audio_analysis/pitch_notes.py`**：basic-pitch（Apache-2.0）有音高 note 事件检测。
  **一个文件两种身份** —— 主 `.venv` 里 import 时拼子进程命令，`.venv-pitch` 里被执行时才
  真正 `import basic_pitch`。含最高声部筛选（`lead_mask`）、逐小节 note 数 / 时间覆盖率、
  结果 JSON 缓存。
- **`tools/calibration/stemrefine.py`**：v0.5 的归因对照、**「16.8% 收回」统计**（含随机基线 lift）、
  人声曲副歌复验、曲目类型三套判据交叉表、强度融合换件的 LOSO 对照。
- **`docs/research/stem-refinement-n40.md`**：本轮中文报告（525 行）。
- **`.venv-pitch/`**（不入库，`.gitignore` 已加）：uv 建的 python 3.11 venv，只装 basic-pitch。

### 修改的文件

| 文件 | 改动 |
|---|---|
| `tools/audio_analysis/stems.py` | 支持 **`htdemucs_6s`**（+guitar/piano）；输出目录按模型名分开（`htdemucs` 仍写 `stems/`，**已有 40 首四路缓存不失效**）；新增 `compact` 落盘（单声道 22.05 kHz PCM_16，体积 1/8）；`model` 可外部传入以复用 |
| `tools/audio_analysis/onsets.py` | 新增 **`transient_fx`**（`other` stem 的 >4 kHz 瞬态 → `fx` 轨）；`STEM_ONSET_PARAMS` 补 guitar / piano |
| `tools/audio_analysis/features.py` | 逐小节特征表扩到六轨 + `n_onset_melody/fx` + `n_pnote_*` / `pitched_cov_*` / **`vocal_pitched_ratio`** |
| `tools/audio_analysis/tracks.py` | `TRACK_ORDER_V5 = drum/vocal/melody/bass`；`JSON_ONLY_TRACKS`；`merge_times` / `build_melody_times`（默认只取最高声部） |
| `tools/audio_analysis/stemplan.py` | 候选轨按**角色**分四张名单；排序量乘 **`TRACK_LIFT_PRIOR`**；`piano`/`guitar` 进点缀候选；`fx` 走稀疏规则；人声活动优先读 `vocal_pitched_ratio` |
| `tools/audio_analysis/structure.py` | `Segment` 加 `sparse_accents` 字段 |
| `tools/audio_analysis/sheet.py` | 四轨展示集可切换；`+fx★` 标注；两段必读警示；§1 脚注的 n=8 数字换成 n=40 |
| `tools/audio_analysis/cli.py` | `--model htdemucs_6s` / `--compact-stems` / `--no-pitch-notes` / `--pitch-venv` / `--pitch-backend`；器乐曲判据改 `share_vocals < 0.10 ∧ vocal_pitched_ratio < 0.30` |
| `tools/calibration/{stemhit,loader,cli}.py` | 归因扩到全部新轨（含 lift）；新增 `recovery_breakdown`；`--v5-metrics` / `--v5-only` |
| `tools/audio_analysis/README.md` | 升 v0.5：变更一览、`.venv-pitch` 装法、耗时实测 |
| `tools/calibration/README.md` | `stemrefine.py` 与 v0.5 跑法 |
| `tests/` | **+38 条**用例（全合成数据），全套 **216 passed** |
| `.gitignore` | 加 `.venv-pitch/` |
| `tools/audio_analysis/__init__.py` | `__version__` 0.3.0 → **0.5.0** |

### 40 首实跑

六路分离 8.8 min（中位 14.2 s/曲，MPS，RTF 0.10–0.13，全程未回退 CPU）；
basic-pitch 10.3 min（中位 12.0 s/曲，5 条 stem 一次子进程，**0 次失败**）；
compact 落盘约 1.5 GB（44.1k 立体声会要 6 GB，本机当时只剩 5.4 GB）。

## 结论（详见报告）

**先过可复现性关**：四路的既有数字全部原样复现（drums 0.639/2.73、`other` 0.267/2.30、
「什么都不落」16.77%、逐段最高命中轨 320/43/9/7、骨架一致率 0.8522）。

| 问题 | 结论 |
|---|---|
| basic-pitch | ✅ **装上了**（uv + py3.11 + **`setuptools<81`**，后者是硬性的）；ONNX 与 CoreML 数值实测一致，默认固定 ONNX |
| 六路分离 | **一半值**：`piano` lift **2.44 > other 2.30**、precision 0.563 全表第一 → 赢；`guitar` 1.79 → 不如不拆；**六路模型自己的 `other` 反而变差**（2.30 → 1.81）→ `other` 继续用四路版 |
| 有音高 note | **❌ 作为候选池不值**：`melody` recall 0.698 但 lift 只有 **1.61**、precision 0.338 —— 高 recall 是**靠撒得密**；`pnote_vocals` lift 1.27 **低于** `vocals` 能量 onset 的 2.16 |
| `fx` 瞬态 | **✅ 值，稀有高价值**：recall 0.043 / precision 0.460 / lift 1.99，在 `final_chorus`(prec 0.596)、`interlude`(0.548) 最准 |
| 16.8% 收回 | 表面**收回 77.0%**（未解释 16.8% → 4.0%），但 chance-adjusted lift 只有 melody 1.42 / pnote_vocals 1.27 / fx 1.14，**guitar 0.84、piano 0.84 低于随机** → 大部分是候选池灌水 |
| 人声曲 ρ | **完全没动**：LOSO 40 折，人声曲 **0.338 → 0.338**（NNLS 给有音高覆盖率的权重也是 **0**）。5 种特征组合，三条换权重判据**一条没过** → **权重不换** |
| 曲目类型分类 | 有音高覆盖率 **37/40 判成人声曲**，比能量 VAD 还差 —— 它修掉了底噪失败模式，但 **lead synth 本身就是有音高的** → `share_vocals` 维持 |
| 切轨规划 | 骨架 **0.8522 持平**（常数基线 0.8443）；第一点缀落在段内 lift 前三 **0.528 → 0.662**（瞎猜 0.500），收益全来自 `piano` 进候选 |

**两条被推翻的旧判断**（要写回设计文档）：

1. **n=40 §9-A.5「下一轮最该补的是更好的人声 ground truth」不成立** ——
   补到了真人声 note 事件，ρ 一点没动。瓶颈不在检测器，在「**人声活动**」这个标量本身：
   能量 VAD 0.088、有音高覆盖率 **−0.003**、音节数密度 0.073，单项相关全在 0 附近。
   需要的是「**乐句 / 句读边界**」这一层结构，不是「人声有多活跃」这一个数。
2. **「把新轨直接塞进候选集就行」不成立** —— naive 接法让骨架一致率跌破常数基线（0.826 < 0.844）、
   点缀命中从 0.699 掉到 0.356。根因：**`tendency = onset 密度 × 落格率` 是纯密度量，
   对稠密轨系统性高估**，而 lift 的定义正是「扣掉密度红利之后还剩多少」。
   修法是**按角色分名单**，不是调系数。

**一条方法论**（建议进知识 032）：**比较不同密度的轨时只能看 lift，不能看 recall**。
`melody` 的 recall 0.698 是 `other` 0.267 的 2.6 倍，而 lift 1.61 < 2.30。

## 边界与未做

- **ground truth 仍是代理**（±30 ms 时间重合），**本轮没有人工听审** —— 这仍是第一优先待办。
- basic-pitch 三阈值、`fx` 两参数、`fx` 触发门 0.5/小节、`TRACK_LIFT_PRIOR` 都**只在这 40 首上成立**。
- `stemplan` 的一致率是「复用旧段落划分 + 只重跑规划」，**结构分段没有重跑**。
- 知识 001/002/031/032 与 `docs/audio-analysis.md` **本轮一律只提建议、未改原文**
  （报告 §10 的 R11–R19），等主会话拍板。
- `out/calib/<曲>/stems_htdemucs_6s/` 与 `pitch_notes.json` 是本机缓存，不入库。
