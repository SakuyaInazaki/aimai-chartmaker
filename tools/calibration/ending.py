"""结尾形态统计：尾杀 / 渐弱 / 其他。

**判据由本项目自定**（用户 2026-09-11 讲授只给了定性描述：「曲子结尾一般会有尾杀，
当然渐弱淡出也是有的，这两个都有」），因此必须写清楚并做敏感性检验。

给定逐小节密度曲线 `d`（长度 n，单位 note/小节）与尾窗长度 `N`：

```
tail       = d[-N:]
tail_ratio = mean(tail) / mean(d)                 # 相对全曲均值
late_ratio = mean(tail) / mean(最后一个高密平台)   # 相对"最后一个高密段"
trend      = spearman(0..N-1, tail)               # 尾窗内的单调性（−1 = 严格下降）
```

- **尾杀 `tail_kill`**：`tail_ratio >= kill_ratio`（默认 1.0）
  —— 末段密度不低于全曲平均，收尾不减压；
- **渐弱 `fade_out`**：`tail_ratio < fade_ratio`（默认 0.7）**且** `trend <= fade_trend`
  （默认 −0.5，即尾窗内明显下行）；
- 其余为 **其他 `other`**（包括"低于均值但不单调下降"、"介于 0.7–1.0 的温和收束"）。

第二套（更宽松的尾杀判据，报告里作对照）：`late_ratio >= 0.9`，即末段不低于
"最后一个高密平台"的 90%。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chartpair import spearman

DEFAULT_TAIL = 8
KILL_RATIO = 1.0
FADE_RATIO = 0.7
FADE_TREND = -0.5
LATE_KILL_RATIO = 0.9


@dataclass
class EndingShape:
    label: str                 # tail_kill / fade_out / other
    tail_bars: int
    tail_mean: float
    song_mean: float
    tail_ratio: float
    late_ratio: float          # 相对最后一个高密平台
    trend: float               # 尾窗内 spearman 趋势
    strictly_decreasing: bool
    label_late: str            # 用 late_ratio 判的尾杀（对照口径）

    def to_dict(self) -> dict:
        return {"label": self.label, "tail_bars": self.tail_bars,
                "tail_mean": round(self.tail_mean, 3),
                "song_mean": round(self.song_mean, 3),
                "tail_ratio": round(self.tail_ratio, 4),
                "late_ratio": round(self.late_ratio, 4),
                "trend": round(self.trend, 4),
                "strictly_decreasing": self.strictly_decreasing,
                "label_late": self.label_late}


def last_high_plateau_mean(curve, quantile: float = 0.75, min_len: int = 4) -> float:
    """最后一个"高密平台"的均值。

    高密 = 密度 ≥ 全曲 ``quantile`` 分位；取最后一段长度 ≥ ``min_len`` 的连续高密区间；
    找不到就退回全曲 ``quantile`` 分位值本身。
    """
    d = np.asarray(curve, dtype=float)
    if d.size == 0:
        return float("nan")
    thr = float(np.quantile(d, quantile))
    hi = d >= thr
    runs: list[tuple[int, int]] = []
    i = 0
    while i < d.size:
        if hi[i]:
            j = i
            while j + 1 < d.size and hi[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    if not runs:
        return thr
    a, b = runs[-1]
    return float(np.mean(d[a:b + 1]))


def classify_ending(curve, tail_bars: int = DEFAULT_TAIL,
                    kill_ratio: float = KILL_RATIO,
                    fade_ratio: float = FADE_RATIO,
                    fade_trend: float = FADE_TREND,
                    late_kill_ratio: float = LATE_KILL_RATIO) -> EndingShape:
    """按上述判据给一条逐小节密度曲线定结尾形态。"""
    d = np.asarray(curve, dtype=float)
    n = int(d.size)
    N = int(min(max(1, tail_bars), n)) if n else 0
    if n == 0:
        return EndingShape("other", 0, float("nan"), float("nan"), float("nan"),
                           float("nan"), float("nan"), False, "other")
    tail = d[n - N:]
    song_mean = float(np.mean(d))
    tail_mean = float(np.mean(tail))
    ratio = tail_mean / song_mean if song_mean > 0 else float("nan")
    plateau = last_high_plateau_mean(d)
    late = tail_mean / plateau if plateau and np.isfinite(plateau) and plateau > 0 \
        else float("nan")
    trend = spearman(np.arange(N, dtype=float), tail) if N >= 2 else float("nan")
    strict = bool(N >= 2 and np.all(np.diff(tail) <= 0))

    if np.isfinite(ratio) and ratio >= kill_ratio:
        label = "tail_kill"
    elif (np.isfinite(ratio) and ratio < fade_ratio
          and np.isfinite(trend) and trend <= fade_trend):
        label = "fade_out"
    else:
        label = "other"
    label_late = "tail_kill" if (np.isfinite(late) and late >= late_kill_ratio) \
        else "not_kill"
    return EndingShape(label, N, tail_mean, song_mean, float(ratio), float(late),
                       float(trend), strict, label_late)


def summarize(shapes) -> dict:
    """一批 ``EndingShape`` 的比例汇总。"""
    labels = [s.label for s in shapes]
    n = len(labels) or 1
    out = {"n": len(labels)}
    for k in ("tail_kill", "fade_out", "other"):
        out[k] = labels.count(k)
        out[f"{k}_share"] = round(labels.count(k) / n, 4)
    late = [s.label_late for s in shapes]
    out["tail_kill_late"] = late.count("tail_kill")
    out["tail_kill_late_share"] = round(late.count("tail_kill") / n, 4)
    ratios = np.array([s.tail_ratio for s in shapes if np.isfinite(s.tail_ratio)])
    if ratios.size:
        out["tail_ratio_median"] = round(float(np.median(ratios)), 4)
        out["tail_ratio_mean"] = round(float(ratios.mean()), 4)
    return out
