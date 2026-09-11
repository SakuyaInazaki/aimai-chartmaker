#!/usr/bin/env python3
"""逐小节 note 密度统计与全库形状分析。

本模块把 :mod:`chart_analysis.simai_parser` 的解析结果加工成：

1. **逐小节明细**（:class:`MeasureStat`）：note 总数、各类型数、each 数、
   最细分音、是否休息小节；
2. **逐谱曲线**（:class:`ChartDensity`）：归一化密度曲线、32-bin 重采样、
   峰值位置、休息段、密度台阶；
3. **全库汇总**：平均曲线 / 分位带、k-means 形状聚类、五段模板检验。

除 numpy 外无第三方依赖（k-means 为本模块自带实现，避免引入 sklearn）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .simai_parser import NoteEvent, ParseResult, beat_to_time

RESAMPLE_BINS = 32
#: 知识 001 的量化候选模板「弱 → 较强 → 较弱 → 强 → 渐弱」
TEMPLATE_K001 = np.array([0.20, 0.45, 0.30, 1.00, 0.10])


# ---------------------------------------------------------------------------
# 逐小节
# ---------------------------------------------------------------------------


@dataclass
class MeasureStat:
    measure: int
    start_time: float  # 小节起点（秒）
    bpm: float
    notes: int = 0  # 官方口径 note 总数（tap+hold+slide+touch+break）
    taps: int = 0
    holds: int = 0
    slides: int = 0  # 滑轨数
    stars: int = 0  # 星星头数（已含在 taps 或 breaks 里）
    breaks: int = 0
    touches: int = 0
    each_groups: int = 0  # 同刻 >= 2 note 的时间槽数
    each_notes: int = 0  # 参与 each 的 note 数
    groups: int = 0  # 有 note 的时间槽数
    finest_divisor: float = 0.0  # 本小节出现过的最细分音（数值最大者）

    @property
    def is_rest(self) -> bool:
        """休息小节：note 数 <= 1。"""
        return self.notes <= 1


def measure_stats(res: ParseResult) -> list[MeasureStat]:
    """产出**连续**的逐小节统计（含中间空小节），范围 = 首个 note 到末个 note 所在小节。"""
    if not res.notes:
        return []
    first = min(n.measure for n in res.notes)
    last = max(n.measure for n in res.notes)
    stats = {
        m: MeasureStat(
            measure=m,
            start_time=beat_to_time(res, 4.0 * m),
            bpm=_bpm_at(res, 4.0 * m),
        )
        for m in range(first, last + 1)
    }
    seen_groups: dict[int, set[int]] = {m: set() for m in range(first, last + 1)}
    each_groups: dict[int, set[int]] = {m: set() for m in range(first, last + 1)}
    for n in res.notes:
        st = stats[n.measure]
        _accumulate(st, n)
        seen_groups[n.measure].add(n.group_index)
        if n.is_each:
            each_groups[n.measure].add(n.group_index)
    for m, st in stats.items():
        st.groups = len(seen_groups[m])
        st.each_groups = len(each_groups[m])
    return [stats[m] for m in range(first, last + 1)]


def _accumulate(st: MeasureStat, n: NoteEvent) -> None:
    st.notes += 1
    if n.is_break:
        st.breaks += 1
    elif n.kind in ("tap", "slide_star"):
        st.taps += 1
    elif n.kind == "hold":
        st.holds += 1
    elif n.kind == "slide_track":
        st.slides += 1
    elif n.kind in ("touch", "touch_hold"):
        st.touches += 1
    if n.kind == "slide_star":
        st.stars += 1
    if n.is_each:
        st.each_notes += 1
    if n.divisor > st.finest_divisor:
        st.finest_divisor = n.divisor


def _bpm_at(res: ParseResult, beat: float) -> float:
    bpm = res.bpm_events[0][1] if res.bpm_events else 0.0
    for b, v in res.bpm_events:
        if b <= beat:
            bpm = v
        else:
            break
    return bpm


# ---------------------------------------------------------------------------
# 逐谱曲线
# ---------------------------------------------------------------------------


@dataclass
class ChartDensity:
    name: str
    measures: list[MeasureStat] = field(default_factory=list)
    raw: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 逐小节 note 数
    normalized: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 除以峰值 -> [0,1]
    resampled: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 32-bin，除以自身峰值
    peak_measure: int = 0
    peak_position: float = 0.0  # 峰值小节在全曲的相对位置 [0,1]
    peak_position_smooth: float = 0.0  # 4 小节滑动均值最大处的中心位置（"高潮段"）
    peak_notes: int = 0
    rest_runs: list[tuple[int, int]] = field(default_factory=list)  # (起始小节, 长度)
    plateau_count: int = 0
    segment_means: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 等分五段均值
    segment_profile: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 五段，除以峰值

    @property
    def measure_count(self) -> int:
        return len(self.measures)


def chart_density(res: ParseResult, name: str = "") -> ChartDensity:
    stats = measure_stats(res)
    d = ChartDensity(name=name, measures=stats)
    if not stats:
        return d
    raw = np.array([s.notes for s in stats], dtype=float)
    d.raw = raw
    peak = raw.max()
    d.normalized = raw / peak if peak > 0 else raw
    d.resampled = resample_curve(d.normalized, RESAMPLE_BINS)
    idx = int(np.argmax(raw))
    d.peak_measure = stats[idx].measure
    d.peak_notes = int(raw[idx])
    d.peak_position = idx / (len(raw) - 1) if len(raw) > 1 else 0.0
    d.peak_position_smooth = smoothed_peak_position(raw, window=4)
    d.rest_runs = rest_runs(stats)
    d.plateau_count = plateau_count(d.normalized)
    d.segment_means = segment_means(raw, 5)
    smax = d.segment_means.max()
    d.segment_profile = d.segment_means / smax if smax > 0 else d.segment_means
    return d


def smoothed_peak_position(curve: np.ndarray, window: int = 4) -> float:
    """"高潮段"位置：``window`` 小节滑动均值取最大处的窗口中心，归一到 [0,1]。

    比单小节峰值稳健——单小节峰值容易被一次性爆发（收尾大双押、一小节 32 分连打）
    劫持，滑动窗口找的是"持续高密的那一段"。
    """
    n = len(curve)
    if n < 2:
        return 0.0
    w = min(window, n)
    ma = np.convolve(curve, np.ones(w) / w, mode="valid")
    i = int(np.argmax(ma))
    return float((i + (w - 1) / 2.0) / (n - 1))


def resample_curve(curve: np.ndarray, bins: int = RESAMPLE_BINS) -> np.ndarray:
    """把任意长度曲线重采样到 ``bins`` 个等长区间（区间内取均值）。"""
    n = len(curve)
    if n == 0:
        return np.zeros(bins)
    edges = np.linspace(0, n, bins + 1)
    out = np.empty(bins)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        a, b = int(math.floor(lo)), int(math.ceil(hi))
        a = max(0, min(a, n - 1))
        b = max(a + 1, min(b, n))
        # 面积加权：按小节与区间的重叠长度加权平均
        w = np.array([max(0.0, min(hi, j + 1) - max(lo, j)) for j in range(a, b)])
        if w.sum() <= 0:
            out[i] = curve[a]
        else:
            out[i] = float(np.dot(curve[a:b], w) / w.sum())
    peak = out.max()
    return out / peak if peak > 0 else out


def segment_means(curve: np.ndarray, k: int = 5) -> np.ndarray:
    """等分 ``k`` 段，返回各段均值（按面积加权，段长不整除时按比例分摊）。"""
    n = len(curve)
    if n == 0:
        return np.zeros(k)
    edges = np.linspace(0, n, k + 1)
    out = np.empty(k)
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        a, b = int(math.floor(lo)), min(n, int(math.ceil(hi)))
        a = max(0, min(a, n - 1))
        b = max(a + 1, b)
        w = np.array([max(0.0, min(hi, j + 1) - max(lo, j)) for j in range(a, b)])
        out[i] = float(np.dot(curve[a:b], w) / w.sum()) if w.sum() > 0 else 0.0
    return out


def rest_runs(stats: list[MeasureStat]) -> list[tuple[int, int]]:
    """连续休息小节段（note <= 1），返回 ``(起始下标, 长度)`` 列表（下标相对曲线起点）。"""
    runs: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(stats):
        if s.is_rest:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - start))
                start = None
    if start is not None:
        runs.append((start, len(stats) - start))
    return runs


def low_density_runs(curve: np.ndarray, ratio: float = 0.5) -> list[tuple[int, int]]:
    """低密段：归一化密度低于「全曲中位数 × ratio」的连续小节段。"""
    if len(curve) == 0:
        return []
    thr = float(np.median(curve)) * ratio
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(curve):
        if v <= thr:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - start))
                start = None
    if start is not None:
        runs.append((start, len(curve) - start))
    return runs


def plateau_count(curve: np.ndarray, *, level_step: float = 0.125, min_len: int = 4) -> int:
    """密度"台阶"（平台段）数量。

    做法：3 小节中值平滑 → 按 ``level_step`` 量化到离散档位 →
    统计长度 >= ``min_len`` 的同档位连续段个数。
    """
    n = len(curve)
    if n < min_len:
        return 0
    sm = np.array(
        [float(np.median(curve[max(0, i - 1) : min(n, i + 2)])) for i in range(n)]
    )
    levels = np.round(sm / level_step).astype(int)
    count = 0
    run = 1
    for i in range(1, n):
        if levels[i] == levels[i - 1]:
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 1
    if run >= min_len:
        count += 1
    return count


def changepoint_segments(
    curve: np.ndarray, k: int = 5, *, min_len_frac: float = 0.0
) -> list[int] | None:
    """L2 最优 ``k`` 段分割（动态规划变点检测），返回 ``k + 1`` 个边界下标。

    ``min_len_frac`` 为每段的最小长度（占全曲比例）。**强烈建议设为 0.10**：
    不加约束时 DP 会把"末尾一两个收尾小节"单独切成一段（实测 37% 的谱第 5 段
    短于全曲 5%），从而伪造出"尾部渐弱"。段数放不下时返回 ``None``。
    """
    n = len(curve)
    m = max(1, int(math.ceil(min_len_frac * n))) if min_len_frac > 0 else 1
    if n < k * m:
        return None
    pre = np.concatenate([[0.0], np.cumsum(curve)])
    pre2 = np.concatenate([[0.0], np.cumsum(curve**2)])

    def cost(a: int, b: int) -> float:  # [a, b)
        length = b - a
        if length <= 0:
            return 0.0
        s = pre[b] - pre[a]
        s2 = pre2[b] - pre2[a]
        return float(s2 - s * s / length)

    INF = float("inf")
    dp = np.full((k + 1, n + 1), INF)
    back = np.zeros((k + 1, n + 1), dtype=int)
    dp[0][0] = 0.0
    for j in range(1, k + 1):
        for b in range(j * m, n + 1):
            best, arg = INF, (j - 1) * m
            for a in range((j - 1) * m, b - m + 1):
                if dp[j - 1][a] == INF:
                    continue
                c = dp[j - 1][a] + cost(a, b)
                if c < best:
                    best, arg = c, a
            dp[j][b], back[j][b] = best, arg
    if dp[k][n] == INF:
        return None
    bounds = [n]
    b = n
    for j in range(k, 0, -1):
        b = back[j][b]
        bounds.append(b)
    return sorted(bounds)


def structure_profile(
    curve: np.ndarray, k: int = 5, *, min_len_frac: float = 0.10
) -> tuple[np.ndarray, np.ndarray] | None:
    """结构对齐的 ``k`` 段强度画像。

    返回 ``(归一到峰值的各段均值, 各段长度占比)``；无法分割时返回 ``None``。
    """
    b = changepoint_segments(curve, k, min_len_frac=min_len_frac)
    if b is None:
        return None
    vals = np.array([curve[b[i] : b[i + 1]].mean() for i in range(k)])
    lens = np.array([(b[i + 1] - b[i]) / len(curve) for i in range(k)])
    peak = vals.max()
    return (vals / peak if peak > 0 else vals), lens


# ---------------------------------------------------------------------------
# 全库：k-means 形状聚类（自带实现，不依赖 sklearn）
# ---------------------------------------------------------------------------


def kmeans(x: np.ndarray, k: int, *, seed: int = 0, restarts: int = 20, iters: int = 200):
    """朴素 k-means（k-means++ 初始化 + 多次重启取最小 inertia）。

    返回 ``(labels, centers, inertia)``。
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    best = None
    for _ in range(restarts):
        # k-means++ 初始化
        centers = [x[rng.integers(n)]]
        for _ in range(k - 1):
            d2 = np.min(
                ((x[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1
            )
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            centers.append(x[rng.choice(n, p=probs)])
        c = np.array(centers)
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            d2 = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
            new = np.argmin(d2, axis=1)
            if np.array_equal(new, labels):
                labels = new
                break
            labels = new
            for j in range(k):
                if np.any(labels == j):
                    c[j] = x[labels == j].mean(axis=0)
        inertia = float(((x - c[labels]) ** 2).sum())
        if best is None or inertia < best[2]:
            best = (labels.copy(), c.copy(), inertia)
    return best


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """平均轮廓系数（样本量不大，直接算全距离矩阵）。"""
    n = x.shape[0]
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    ks = np.unique(labels)
    if len(ks) < 2:
        return 0.0
    out = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = d[i][same].mean() if same.any() else 0.0
        b = min(
            d[i][labels == kk].mean() for kk in ks if kk != labels[i] and np.any(labels == kk)
        )
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


# ---------------------------------------------------------------------------
# 五段模板检验
# ---------------------------------------------------------------------------


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def template_fit(profiles: np.ndarray, template: np.ndarray = TEMPLATE_K001) -> dict:
    """检验五段模板：逐谱相关系数 + 残差，以及数据估计的模板与自助置信区间。"""
    corrs = np.array([pearson(p, template) for p in profiles])
    resid = profiles - template[None, :]
    rmse = np.sqrt((resid**2).mean(axis=1))
    est = profiles.mean(axis=0)
    rng = np.random.default_rng(20260911)
    boot = np.array(
        [profiles[rng.integers(0, len(profiles), len(profiles))].mean(axis=0) for _ in range(2000)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return {
        "corr": corrs,
        "corr_mean": float(np.nanmean(corrs)),
        "corr_median": float(np.nanmedian(corrs)),
        "corr_positive_ratio": float(np.mean(corrs > 0)),
        "rmse": rmse,
        "rmse_mean": float(rmse.mean()),
        "estimate": est,
        "ci_low": lo,
        "ci_high": hi,
        "median_profile": np.median(profiles, axis=0),
    }
