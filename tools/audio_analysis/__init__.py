"""音频分析原型包（aimai-chartmaker）。

在用户给定 BPM 与 offset（simai `&first`）的前提下，把 mp3 转成 LLM 可消费的
结构化"歌曲分析单"。管线对应 `docs/audio-analysis.md` v1.0 的 L1–L3 层：

- L1 节拍层：不做节拍追踪，由 `grid` 用用户给的 BPM/first 直接构造网格，
  并对用户 offset 做**只报告不覆盖**的对齐校验；
- L2 音轨层：`stems`（Demucs 四轨）+ `onsets`（逐 stem onset / 鼓件启发式 / 人声 VAD）
  + `quantize`（onset → 拍网格字符串）；
- L3 结构层：`structure`（段落分段）+ `intensity`（逐小节强度曲线与高潮投票）；
- 输出：`sheet`（song_analysis.json / song_sheet.md / plot.png）。

v0.2 的模块补充：`tracks`（四条逻辑轨与网格串渲染）、`features`（逐小节特征表）、
`stemplan`（段落 → 踩音规划）。

v0.3（2026-09-11，落地 8 首官方音频配对标定的结论，见
`docs/research/audio-chart-calibration.md`）：
- `grid.check_offset` **修掉 ±半拍反拍假警报**（搜索窗收到 ±0.4 拍 + 显著性门 +
  「反拍歧义」判词）；
- `stemplan` 从单值「主踩音轨」换成 **骨架轨 + 点缀轨 + 估计占比 + 依据**；
- `intensity` 的绝对量级锚点改走 **NPS**，段落地板可调，**intro/outro 结构封顶**；
- `climax` 正名为**音频能量高潮**（不是谱面密度峰定位器）。
"""

__version__ = "0.3.0"
