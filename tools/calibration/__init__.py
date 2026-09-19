"""tools/calibration — 官方音频 × 官方谱的配对标定（docs/audio-analysis.md §4.8(E)）。

用户 2026-09-11 提供 8 首官方 ST 曲包（官方 mp3 + 官方 maidata），解锁了设计文档
一直挂起的 (E) 项：**逐小节音频强度 `I_bar` 与官方逐小节密度 `d_bar` 的配对回归**。

本包只做"分析与标定"，不改动 `tools/audio_analysis` 的运行时行为；
它消费 `tools/audio_analysis` 已经跑出的 `song_analysis.json` + `stems/`，
以及 `tools/chart_analysis` 解析出的官方谱 note 时间轴。

| 模块 | 职责 |
|------|------|
| `chartpair.py` | 谱面 ↔ 音频的小节对齐、逐小节/逐段密度、相关系数、峰值与边界命中 |
| `stemhit.py` | 官方 note 时间 × 各 stem onset 的匹配（命中率 / 精确率 / 全局相位扫描） |
| `weights.py` | 五项 Z 特征 → 密度的非负最小二乘 / 岭回归 + 留一曲交叉验证 |
| `ending.py` | 结尾形态（尾杀 / 渐弱 / 其他）判据与敏感性 |
| `sampling.py` | **采音方式度量**：逐小节看目标音轨的音被踩了多少，落成社区的三个词 **全踩 / 舍音 / 留白**（判不出记 `—`；2026-09-20 术语校正，原 `采全音 / 半采音 / 空音` 是用户自述"我编的"，见 `docs/research/sampling-terminology-survey.md`）+ 随机平移对照；coverage 等数字只作诊断（报告 `docs/research/sampling-mode-study.md`） |
| `loader.py` | 从 `out/calib/<曲名>/` 装载一首曲子的音频侧与谱面侧数据（含 IO） |
| `cli.py` | 命令行入口：跑完全部分析，写 CSV / 图 / 指标 JSON |

报告：`docs/research/audio-chart-calibration.md`
"""

__version__ = "0.1.0"

__all__ = ["chartpair", "ending", "sampling", "stemhit", "weights"]
