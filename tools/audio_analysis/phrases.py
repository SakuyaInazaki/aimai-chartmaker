"""纯音频的**乐句（句读）边界**探测：人声 stem → 乐句起点 / 句间留白。

为什么要有这一层
----------------
`docs/research/audio-chart-calibration-n160.md` 证明音频强度对官方密度的解释力
**在密度侧已经饱和**，人声曲（ρ 0.28）系统性低于器乐曲（ρ 0.42）；
`docs/research/stem-refinement-n40.md` 又排除了「更细的分轨 / 有音高 note 轨」。
剩下的候选是**句读结构**这一层——它不是"人声有多响"，而是"歌手在哪里开口、在哪里换气"。

本模块给出**不依赖歌词**的乐句边界估计，ground truth 用
`tools/calibration/lyrics_phrases.py` 从公网 LRC 对齐来的逐行时间戳
（评测口径：±1 拍容差的 F1，见 `docs/research/vocal-phrase-alignment.md` §5）。

两个探测器
----------
- `onset_gap`：人声 onset 的**发音间隔（IOI）≥ `gap_beats` 拍**时，把该 onset 记为句首。
  这是"歌手停了一下再开口"的最直接读法；
- `vad_rise`：人声 VAD 的**上升沿**（静音 → 有声）。⚠️ Demucs `vocals` 轨泄漏严重
  （能量 VAD 把 31/51 首器乐曲判成人声曲，知识 002 第三轮修订），
  所以这一路单用时召回偏低。

`detect_phrases()` 默认取两者的并集再按 `min_sep_beats` 合并近邻。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: 默认：IOI ≥ 多少拍算一次"停顿后重新开口"
DEFAULT_GAP_BEATS = 1.5
#: 默认：两个句首至少要隔多少拍（更近的合并成一个）
DEFAULT_MIN_SEP_BEATS = 2.0


@dataclass
class PhraseDetection:
    """探测结果。"""

    starts: np.ndarray = field(default_factory=lambda: np.zeros(0))   # 句首秒
    gaps: list = field(default_factory=list)      # [(t0, t1)] 句间留白
    source: str = "combined"
    n_onsets: int = 0

    def to_dict(self) -> dict:
        return {"starts": [round(float(t), 3) for t in self.starts],
                "gaps": [[round(float(a), 3), round(float(b), 3)] for a, b in self.gaps],
                "source": self.source, "n_onsets": int(self.n_onsets)}


def _beat_sec(grid, t: float) -> float:
    bar = min(max(grid.bar_of(float(t)), 1), max(1, grid.n_bars))
    return 60.0 / float(grid.bar_bpm(bar))


def onset_gap_starts(onsets: Sequence[float], grid,
                     gap_beats: float = DEFAULT_GAP_BEATS) -> np.ndarray:
    """人声 onset 里"前面隔了 ≥ `gap_beats` 拍"的那些 —— 停顿后重新开口。"""
    t = np.sort(np.asarray(onsets, dtype=float))
    if t.size == 0:
        return np.zeros(0)
    keep = [float(t[0])]
    for prev, cur in zip(t, t[1:]):
        if (cur - prev) >= gap_beats * _beat_sec(grid, prev):
            keep.append(float(cur))
    return np.asarray(keep, dtype=float)


def vad_rise_starts(active: Sequence[bool], times: Sequence[float]) -> np.ndarray:
    """VAD 的上升沿时刻（静音 → 有声）。"""
    a = np.asarray(active, dtype=bool)
    ts = np.asarray(times, dtype=float)
    if a.size < 2 or ts.size < a.size:
        return np.zeros(0)
    idx = np.flatnonzero((~a[:-1]) & a[1:]) + 1
    out = ts[idx]
    if a[0]:
        out = np.concatenate(([ts[0]], out))
    return np.asarray(out, dtype=float)


def vad_gaps(active: Sequence[bool], times: Sequence[float], grid,
             min_beats: float = 1.0) -> list[tuple[float, float]]:
    """VAD 的静音段里长度 ≥ `min_beats` 拍的那些 —— 句间留白候选。"""
    a = np.asarray(active, dtype=bool)
    ts = np.asarray(times, dtype=float)
    if a.size < 2:
        return []
    out: list[tuple[float, float]] = []
    edges = np.flatnonzero(np.diff(a.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [a.size]))
    for s, e in zip(bounds, bounds[1:]):
        if s >= a.size or a[s]:
            continue
        t0, t1 = float(ts[s]), float(ts[min(e, ts.size - 1)])
        if (t1 - t0) >= min_beats * _beat_sec(grid, t0):
            out.append((t0, t1))
    return out


def _merge_close(t: np.ndarray, grid, min_sep_beats: float) -> np.ndarray:
    t = np.sort(np.asarray(t, dtype=float))
    if t.size == 0:
        return t
    out = [float(t[0])]
    for x in t[1:]:
        if (x - out[-1]) >= min_sep_beats * _beat_sec(grid, out[-1]):
            out.append(float(x))
    return np.asarray(out, dtype=float)


def detect_phrases(onsets: Sequence[float], vad: dict | None, grid,
                   gap_beats: float = DEFAULT_GAP_BEATS,
                   min_sep_beats: float = DEFAULT_MIN_SEP_BEATS,
                   source: str = "combined") -> PhraseDetection:
    """纯音频的乐句边界估计。

    `source` ∈ {`onset_gap`, `vad_rise`, `combined`}。
    """
    og = onset_gap_starts(onsets, grid, gap_beats=gap_beats)
    vr = (vad_rise_starts(vad.get("active", []), vad.get("times", []))
          if vad else np.zeros(0))
    if source == "onset_gap":
        st = og
    elif source == "vad_rise":
        st = vr
    else:
        st = np.concatenate([og, vr]) if vr.size else og
    st = _merge_close(st, grid, min_sep_beats)
    gaps = vad_gaps(vad.get("active", []), vad.get("times", []), grid) if vad else []
    return PhraseDetection(starts=st, gaps=gaps, source=source,
                           n_onsets=int(np.size(onsets)))


# --------------------------------------------------------------------------
# 评测（对 LRC ground truth）
# --------------------------------------------------------------------------

def boundary_prf(pred: Sequence[float], truth: Sequence[float],
                 tol_sec: float) -> dict:
    """句首边界的 precision / recall / F1（贪心一对一匹配，容差 `tol_sec`）。"""
    p = np.sort(np.asarray(pred, dtype=float))
    g = np.sort(np.asarray(truth, dtype=float))
    if p.size == 0 or g.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_pred": int(p.size), "n_truth": int(g.size), "tp": 0}
    used = np.zeros(g.size, dtype=bool)
    tp = 0
    for x in p:
        d = np.abs(g - x)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_sec:
            used[j] = True
            tp += 1
    prec = tp / p.size
    rec = tp / g.size
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "n_pred": int(p.size), "n_truth": int(g.size), "tp": int(tp)}


# --------------------------------------------------------------------------
# 逐小节特征（接进 song sheet / song_analysis.json 用）
# --------------------------------------------------------------------------

def phrase_features_per_bar(det: PhraseDetection, grid) -> dict:
    """逐小节特征：`phrase_start`（该小节内的句首个数）、`in_gap`（该小节落在句间留白里的比例）。"""
    n = int(grid.n_bars)
    starts = np.zeros(n, dtype=float)
    in_gap = np.zeros(n, dtype=float)
    for t in det.starts:
        b = grid.bar_of(float(t))
        if 1 <= b <= n:
            starts[b - 1] += 1.0
    for b in range(1, n + 1):
        t0 = grid.bar_start(b)
        t1 = t0 + grid.bar_duration(b)
        cov = 0.0
        for a, c in det.gaps:
            cov += max(0.0, min(c, t1) - max(a, t0))
        in_gap[b - 1] = min(1.0, cov / max(t1 - t0, 1e-6))
    return {"phrase_start": starts, "in_gap": in_gap}
