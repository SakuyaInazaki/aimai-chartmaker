# tools/calibration — 官方音频 × 官方谱的配对标定

把 `tools/audio_analysis` 的音频侧输出与 `tools/chart_analysis` 的官方谱侧 ground truth
**逐小节配对**，回答 `docs/audio-analysis.md` §4.8(E) 挂起的问题：

- 音频强度 `I_bar` 到底能不能预测官方逐小节密度 `d_bar`？相关有多强？
- 五项融合权重 `[0.25, 0.30, 0.20, 0.15, 0.10]` 标定出来是多少？值不值得换？
- **官方谱师这一段实际踩了哪条音轨**？`stemplan.py` 的规则表对不对？
- 官方谱的结尾是尾杀还是渐弱？

**结论全部写在 `docs/research/audio-chart-calibration-n160.md`**（n=160，取代 n=40 的
`audio-chart-calibration-n40.md` 与 n=8 的 `audio-chart-calibration.md`），
本 README 只讲怎么跑。

## 模块

| 文件 | 职责 |
|------|------|
| `chartpair.py` | 谱面 ↔ 音频小节对齐、加权密度、Spearman/Pearson、峰值位置、边界命中、密度地板池化拟合 |
| `stemhit.py` | 官方 note 时间 × stem onset 匹配：命中率 / 精确率 / **随机基线 lift** / 全局相位扫描 φ\* |
| `weights.py` | 五项 Z 特征 → 密度的 NNLS / 岭回归 + **留一曲交叉验证** + 手写预设对照 |
| `ending.py` | 结尾形态判据（尾杀 / 渐弱 / 其他）与敏感性 |
| `strata.py` | **（n=40 新增）** 曲目人声画像与分层汇总、精确符号检验、换权重的三条判据、单/双阈值扫描（Youden/AUC）、v0.2 单值规则表复刻、切轨模型一致率表；**（n=160 新增）** 官方 `&genre` **六曲风分层键**（`genre_band` / `genre_short` / `GENRE_ORDER`）、`evaluate_fixed_one`（现役单阈值的对照混淆矩阵） |
| `stemrefine.py` | **（v0.5 新增）** 四轨 vs **六轨 + 有音高 note + fx** 的归因对照、**「16.8% 什么都不落」的收回统计**（含随机基线 lift）、人声曲副歌复验、曲目类型三套判据交叉表、把 `voiced` 换成 `vocal_pitched` 的强度 LOSO 对照 |
| `loader.py` | 从 `out/calib/<曲名>/` 装载音频侧与谱面侧（重跑 onset，不重跑 Demucs） |
| `cli.py` | 命令行入口 |

## 用法

前置：把官方曲包解压到 `out/calib/<曲名>/`（含 `maidata.txt` + `track.mp3`），
先对每首跑一遍音频管线（BPM / `&first` / `&lv_5` 取自 maidata 头部）：

```bash
python -m tools.audio_analysis \
  --audio "out/calib/<曲名>/track.mp3" --bpm <BPM> --first <&first> \
  --level <&lv_5> --out "out/calib/<曲名>/" [--bpm-changes "<小节>:<BPM>"]
```

再跑标定：

```bash
python -m tools.calibration \
  --calib-dir out/calib \
  --csv docs/research/data/audio-chart-calibration-n160-summary.csv \
  --metrics <scratchpad>/metrics160.json \
  --plots   <scratchpad> \
  --csv-no-segments                 # n=160 起必须加，见下
# 可选：--tol-ms 30（切轨匹配容差）--ridge-alpha 1.0 --skip-corpus（跳过 388 谱结尾统计）
#       --overlay-songs "A,B,C"（只画这几首的叠加图，缺省会写 160 张）
```

**n=160 新增的三个开关与它们的理由**

| 开关 | 为什么 |
|---|---|
| `--csv-no-segments` | 160 首有 **1490 个音频段**，逐段行会把 CSV 撑到 600 KB 以上，超过交付物上限。加了之后只写逐曲 160 行，逐段信息压成 `n_segments` / `seg_best_*` / `seg_plan_agree*` 汇总列；**段落明细仍然完整保留在 metrics JSON 里** |
| `--overlay-songs` | 缺省会给每一首写一张 `<曲名>-overlay.png`（160 张 ≈ 45 MB） |
| （自动）`loader.ensure_mix_wav` | 磁盘紧张时管线跑完会删掉 `track.44k.wav`（22 MB/首）。标定装载时按**同一条 ffmpeg 命令**临时重新解码到进程私有临时目录，用完即删 —— 时间基准与管线完全一致 |

**metrics JSON 里的 n=160 新字段**

| 字段 | 含义 |
|---|---|
| `summary.strata.by_genre_*` / `box_rho_by_genre` / `genre_kind_cross` | 官方 `&genre` 六曲风分层与「曲风 × 人声/器乐」交叉表 |
| `summary.calibration_vs_current` / `weight_switch_vs_current` | LOSO **对照现役默认权重**（`calibration` 那一套对照的是 v0.3 初值，与 n=40 报告同口径）。"要不要**再**换一次权重"只能看这一组 |
| `weight_presets.current_default` | 现役权重在预设对照表里的那一行 |
| `vocal_led_sweep.current_share` | 现役 `stemplan.VOCAL_LED_SHARE`（0.14）在本轮样本上的混淆矩阵；`legacy_0.45_0.65` 是 v0.3 旧判据 |
| `songs[].pool.recall_phi_shifted` | 候选池**做过 φ 补偿**的对照值（主值已改为不补偿，见下） |

**v0.5 分轨细化标定**（`docs/research/stem-refinement-n40.md`）：先给每首曲目补上
六路 stem 与 basic-pitch 缓存（**不动已有四路 stems**），再加 `--v5-metrics` 跑：

```bash
# 1) 每首补 stems_htdemucs_6s/ + pitch_notes.json（compact 落盘，约 1.5 GB / 40 首）
python - <<'PY'
from pathlib import Path
from tools.audio_analysis import stems, pitch_notes as pn
from demucs.pretrained import get_model
m = get_model("htdemucs_6s"); m.eval()
for d in sorted(Path("out/calib").iterdir()):
    sd = stems.stems_dir_for(d, "htdemucs_6s")
    stems.separate(d / "track.44k.wav", sd, "htdemucs_6s", compact=True, model=m)
    pn.detect_batch({n: sd / f"{n}.wav"
                     for n in ("vocals", "other", "guitar", "piano", "bass")},
                    cache_path=sd / "pitch_notes.json")
PY

# 2) 跑对照（--v5-only 跳过 n=40 的既有全套）
python -m tools.calibration --calib-dir out/calib \
  --v5-metrics <scratchpad>/v5_metrics.json --csv /dev/null --skip-corpus
```

**变速曲的前置处理（n=40 新增，n=160 沿用）**：`Grid` 只支持"在小节线上换 BPM"。
变速点落在小节**中间**的曲子（**160 首里有 11 首**，n=40 时只有 3 首），
要先用谱面侧解析器的 `bpm_events`（拍位 → BPM）算出每个谱面小节的真实时长、
反解成**该小节的等效 BPM**，再逐小节喂给 `--bpm-changes`。
这样小节边界能精确对齐（实测 **160/160 首最大误差 0.002 ms**），代价是那几个小节
**内部**的分音判定不可信。做法见报告 n160 §1.2 与 R22。
⚠️ **《東方妖々夢 ～the maximum moving about～》自带逐小节 tempo map（117 段变速）**，
对它而言"内部分音不可信"等于全曲都不可信。

**基准 BPM 取哪个（n=160 新增的坑）**：**取谱面正文首个 `(BPM)`，不要取 `&wholebpm`**。
160 首里有 3 首两者不一致，其中《ウッーウッーウマウマ(ﾟ∀ﾟ)》的 `&wholebpm=164` 而
正文是 `164.73`，按 164 跑会让 66 个小节只有 6 个对得上（对齐误差 434 ms）。

依赖：仓库根 `.venv`（python3.12）；用到 numpy / scipy / librosa / matplotlib，
以及 `tools/audio_analysis` 与 `tools/chart_analysis` 两个包。
测试：`python -m pytest tests/test_calibration.py -q`（**73 用例**，全合成数据）。

## 口径说明（引用数字前必读）

- **小节号**：谱面侧 0 起（`chart_analysis`），音频侧 1 起（`audio_analysis.Grid`）。
  对齐用"起始秒最近"匹配，不硬编码 +1。**160 首实测最大对齐误差 0.002 ms**。
- **踩音事件 = 时间槽**：同刻的 each/双押算**一个**踩音事件（逐 note 口径同时输出）。
- **匹配容差 30 ms**：librosa onset 的帧分辨率是 512/22050 = 23.2 ms，30 ms 已接近下限。
- **必须看 lift 不能只看 recall**：鼓轨 onset 天然最多（8 首 479–631 个，人声只有 61–333 个），
  只比命中率的话鼓永远赢。`lift = recall / (1 − exp(−λ·2τ))`。
- **候选池比的是"未做 φ 补偿"的官方时间槽**（n=160 修正）：φ\* 补偿的是"音频 onset 相对
  谱面时钟的 MP3 解码延迟"，拿来跟 stem onset 比是对的；但候选池是**量化落格后的网格位置**，
  网格锚在用户给的 `&first` 上，与官方 note 本来就同一个时钟，再平移一次反而把官方 note
  推离网格。n=40 时最大 |φ\*| 只有 25 ms、被 30 ms 容差盖住；n=160 里
  L4TS2018（φ\* = −37.5 ms）的候选池 recall 被压到 **0.003** 才暴露出来。
  修正后 pool recall 0.8328 → **0.8396**；补偿版保留为 `recall_phi_shifted` 对照列。
- **权重标定只定方向**：融合值后接的 `robust_unit` 是单调变换，
  Spearman 与 w 的整体缩放无关，所以权重归一到 Σw = 1 不损失信息。
- **密度地板拟合只比形状**：两边各除以曲内均值后再拟合 `floor`，
  绝对量级由"定数 → note/小节（或 NPS）"另行锚定。
- **知识 003 加权密度的权重是本项目自定**（tap 1.0 / hold 1.2 / slide 轨 1.5、
  BREAK +0.3、each +0.2）——知识 003 只给了定性顺序 tap < hold < each < slide，
  **没有数值依据，只作稳健性对照**。

## 实跑记录（2026-09-13，160 首官方曲包）

- 音频管线：**160/160 跑通**，均值 41 s/首（M4 / MPS，四路 Demucs + all-in-one，
  `--compact-stems --no-pitch-notes`）；六路分离只留 `piano` 另加 19 s/首；
  谱面解析 160/160 零 error / 零 warning。
- `out/calib/` 最终 **4.9 GB / 160 首**（均值 31 MB/首），全程磁盘余量未低于 6 GB。
- 标定本体 ~8 min（复用已有 stems，只重跑 onset 与五项分量 + 临时解码混音）。
- **`intensity.py` 的默认融合权重不再改动**：LOSO 160 折对照**现役默认**是
  **−0.0015 / 82 折胜（51.3%）/ p = 0.81**，三条判据全不过 → 维持；
  对照 v0.3 初值仍然是 +0.0617 / 124 折（77.5%）/ p ≈ 0 → **n=40 那次替换被确认正确**。
- **`stemplan.VOCAL_LED_SHARE = 0.14` 不再改动**：1401 段扫出的 Youden 最优是 **0.1424**
  （n=40 是 0.142），现役值与最优只差 0.007。
- 详见报告 n160 §3 / §5.4。

## 实跑记录（2026-09-11，8 首官方曲包）

- 音频管线：8/8 跑通，RTF ≈ 0.28（M4 / MPS）；谱面解析 8/8 零 error，
  note 总数与 `resource/official-chart/` 的转录版 8/8 一致。
- 标定本体耗时 ~10 s（复用已有 stems，只重跑 onset 与五项分量）。
- **`intensity.py` 的默认融合权重未改动**：LOSO 8 折 CV Spearman
  **标定值 0.515（中位 0.537）vs 初值 0.488（中位 0.518）**，仅 5/8 折胜出、
  符号检验 p ≈ 0.36 → 不满足"明显优于"，按任务约定保持初值。
  详见报告 §3。
