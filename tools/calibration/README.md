# tools/calibration — 官方音频 × 官方谱的配对标定

把 `tools/audio_analysis` 的音频侧输出与 `tools/chart_analysis` 的官方谱侧 ground truth
**逐小节配对**，回答 `docs/audio-analysis.md` §4.8(E) 挂起的问题：

- 音频强度 `I_bar` 到底能不能预测官方逐小节密度 `d_bar`？相关有多强？
- 五项融合权重 `[0.25, 0.30, 0.20, 0.15, 0.10]` 标定出来是多少？值不值得换？
- **官方谱师这一段实际踩了哪条音轨**？`stemplan.py` 的规则表对不对？
- 官方谱的结尾是尾杀还是渐弱？

**结论全部写在 `docs/research/audio-chart-calibration-n40.md`**（n=40，取代 n=8 的
`audio-chart-calibration.md`），本 README 只讲怎么跑。

## 模块

| 文件 | 职责 |
|------|------|
| `chartpair.py` | 谱面 ↔ 音频小节对齐、加权密度、Spearman/Pearson、峰值位置、边界命中、密度地板池化拟合 |
| `stemhit.py` | 官方 note 时间 × stem onset 匹配：命中率 / 精确率 / **随机基线 lift** / 全局相位扫描 φ\* |
| `weights.py` | 五项 Z 特征 → 密度的 NNLS / 岭回归 + **留一曲交叉验证** + 手写预设对照 |
| `ending.py` | 结尾形态判据（尾杀 / 渐弱 / 其他）与敏感性 |
| `strata.py` | **（n=40 新增）** 曲目人声画像与分层汇总、精确符号检验、换权重的三条判据、单/双阈值扫描（Youden/AUC）、v0.2 单值规则表复刻、切轨模型一致率表 |
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
  --csv docs/research/data/audio-chart-calibration-n40-summary.csv \
  --metrics <scratchpad>/metrics40.json \
  --plots   <scratchpad>            # 逐曲 <曲名>-overlay.png + summary.png（分布与分层箱线图）
# 可选：--tol-ms 30（切轨匹配容差）--ridge-alpha 1.0 --skip-corpus（跳过 388 谱结尾统计）
```

**变速曲的前置处理（n=40 新增）**：`Grid` 只支持"在小节线上换 BPM"。变速点落在小节
**中间**的曲子（40 首里有 2 首），要先用谱面侧解析器的 `bpm_events`（拍位 → BPM）
算出每个谱面小节的真实时长、反解成**该小节的等效 BPM**，再逐小节喂给 `--bpm-changes`。
这样小节边界能精确对齐（实测 40/40 首误差 0.000 ms），代价是那一两个小节**内部**的
分音判定不可信。做法见报告 n40 §1.2 与 R10。

依赖：仓库根 `.venv`（python3.12）；用到 numpy / scipy / librosa / matplotlib，
以及 `tools/audio_analysis` 与 `tools/chart_analysis` 两个包。
测试：`python -m pytest tests/test_calibration.py -q`（**54 用例**，全合成数据）。

## 口径说明（引用数字前必读）

- **小节号**：谱面侧 0 起（`chart_analysis`），音频侧 1 起（`audio_analysis.Grid`）。
  对齐用"起始秒最近"匹配，不硬编码 +1。8 首实测最大对齐误差 **0.000 ms**。
- **踩音事件 = 时间槽**：同刻的 each/双押算**一个**踩音事件（逐 note 口径同时输出）。
- **匹配容差 30 ms**：librosa onset 的帧分辨率是 512/22050 = 23.2 ms，30 ms 已接近下限。
- **必须看 lift 不能只看 recall**：鼓轨 onset 天然最多（8 首 479–631 个，人声只有 61–333 个），
  只比命中率的话鼓永远赢。`lift = recall / (1 − exp(−λ·2τ))`。
- **权重标定只定方向**：融合值后接的 `robust_unit` 是单调变换，
  Spearman 与 w 的整体缩放无关，所以权重归一到 Σw = 1 不损失信息。
- **密度地板拟合只比形状**：两边各除以曲内均值后再拟合 `floor`，
  绝对量级由"定数 → note/小节（或 NPS）"另行锚定。
- **知识 003 加权密度的权重是本项目自定**（tap 1.0 / hold 1.2 / slide 轨 1.5、
  BREAK +0.3、each +0.2）——知识 003 只给了定性顺序 tap < hold < each < slide，
  **没有数值依据，只作稳健性对照**。

## 实跑记录（2026-09-11，8 首官方曲包）

- 音频管线：8/8 跑通，RTF ≈ 0.28（M4 / MPS）；谱面解析 8/8 零 error，
  note 总数与 `resource/official-chart/` 的转录版 8/8 一致。
- 标定本体耗时 ~10 s（复用已有 stems，只重跑 onset 与五项分量）。
- **`intensity.py` 的默认融合权重未改动**：LOSO 8 折 CV Spearman
  **标定值 0.515（中位 0.537）vs 初值 0.488（中位 0.518）**，仅 5/8 折胜出、
  符号检验 p ≈ 0.36 → 不满足"明显优于"，按任务约定保持初值。
  详见报告 §3。
