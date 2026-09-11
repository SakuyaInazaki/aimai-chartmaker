"""谱面 ↔ 音频的配对口径：小节对齐、密度序列、相关系数、峰值与边界命中。

口径约定（与 `tools/chart_analysis` 保持一致，引用时必须注明）：

- **谱面小节号从 0 起**（`chart_analysis.simai_parser.NoteEvent.measure`），
  **音频小节号从 1 起**（`tools/audio_analysis.grid.Grid`）。
  在 `&first = 0` 且 BPM 一致时二者相差 1：谱面 m ↔ 音频 b = m + 1。
  本模块**不假定**这一关系，而是用「起始秒数最近」做匹配（`map_measures_to_bars`），
  这样变速曲与 `&first ≠ 0` 的曲子也能正确对齐。
- **密度 `d_bar`** = 该小节的官方口径 note 数（tap+hold+slide+touch+break）。
- **加权密度 `d_bar_w`** = 按知识 003「note 种类强度递增 tap < hold < each < slide」
  加权后的 note 数；权重见 `KIND_WEIGHT` / `EACH_BONUS`（**本项目自定，知识 003 只给了
  定性顺序没给数值**，故加权变体只作为稳健性对照，不作为主口径）。
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

# 知识 003-1：同踩音下 tap < hold < each < slide。数值为本项目自定（定性顺序的最简量化）。
KIND_WEIGHT: dict[str, float] = {
    "tap": 1.0,
    "hold": 1.2,
    "slide_star": 1.0,     # 星星头本身按 tap 计（与官方计数口径一致）
    "slide_track": 1.5,    # 滑轨
    "touch": 0.8,
    "touch_hold": 1.0,
}
BREAK_BONUS = 0.3          # BREAK 额外加权
EACH_BONUS = 0.2           # 参与 each（同刻 ≥2 note）的每个 note 额外加权


# ---------------------------------------------------------------------------
# 小节对齐
# ---------------------------------------------------------------------------


def map_measures_to_bars(measure_starts: Sequence[float], bar_starts: Sequence[float],
                         tol_sec: float = 0.05) -> dict[int, int]:
    """谱面小节起始秒 → 音频小节号（1 起）的映射，按"起始秒最近且在容差内"匹配。

    参数：
        measure_starts: 第 i 个谱面小节的起始秒（i 为**列表下标**，不是小节号）
        bar_starts:     第 j 个音频小节的起始秒（音频小节号 = j + 1）
    返回：
        ``{列表下标 i: 音频小节号 (1 起)}``，匹配不上（超出音频长度或误差 > tol）的不收录。
    """
    bs = np.asarray(bar_starts, dtype=float)
    out: dict[int, int] = {}
    if bs.size == 0:
        return out
    for i, t in enumerate(measure_starts):
        j = int(np.argmin(np.abs(bs - float(t))))
        if abs(float(bs[j]) - float(t)) <= tol_sec:
            out[i] = j + 1
    return out


def measure_alignment_error(measure_starts: Sequence[float],
                            bar_starts: Sequence[float]) -> float:
    """谱面小节与音频小节起始秒的最大绝对误差（秒），用于核对网格是否同源。"""
    bs = np.asarray(bar_starts, dtype=float)
    if bs.size == 0 or len(measure_starts) == 0:
        return float("nan")
    errs = [abs(float(bs[int(np.argmin(np.abs(bs - t)))]) - float(t))
            for t in measure_starts]
    return float(max(errs))


# ---------------------------------------------------------------------------
# 密度序列
# ---------------------------------------------------------------------------


def weighted_note_value(kind: str, is_break: bool, is_each: bool) -> float:
    """单个 note 的加权值（知识 003 的定性顺序 → 数值，本项目自定）。"""
    v = KIND_WEIGHT.get(kind, 1.0)
    if is_break:
        v += BREAK_BONUS
    if is_each:
        v += EACH_BONUS
    return float(v)


def weighted_density(notes: Iterable, n_measures: int, first_measure: int = 0
                     ) -> np.ndarray:
    """逐小节加权密度。``notes`` 为 ``NoteEvent`` 序列（需有 measure/kind/is_break/is_each）。"""
    out = np.zeros(int(n_measures), dtype=float)
    for n in notes:
        idx = int(getattr(n, "measure")) - int(first_measure)
        if 0 <= idx < out.size:
            out[idx] += weighted_note_value(getattr(n, "kind"),
                                            bool(getattr(n, "is_break", False)),
                                            bool(getattr(n, "is_each", False)))
    return out


# ---------------------------------------------------------------------------
# 相关系数
# ---------------------------------------------------------------------------


def _rank(x: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return float("nan")
    return pearson(_rank(a), _rank(b))


def fisher_mean(rhos: Sequence[float]) -> float:
    """多曲相关系数的汇总：Fisher z 变换后取均值再反变换（比直接平均更合适）。"""
    vals = [float(r) for r in rhos if np.isfinite(r) and abs(float(r)) < 1.0]
    if not vals:
        finite = [float(r) for r in rhos if np.isfinite(r)]
        return float(np.mean(finite)) if finite else float("nan")
    z = np.arctanh(np.asarray(vals, dtype=float))
    return float(np.tanh(z.mean()))


# ---------------------------------------------------------------------------
# 峰值与边界
# ---------------------------------------------------------------------------


def smoothed_peak_index(curve: Sequence[float], window: int = 4) -> int:
    """``window`` 小节滑动均值最大处的窗口中心下标（四舍五入到整数下标）。"""
    c = np.asarray(curve, dtype=float)
    if c.size == 0:
        return 0
    w = int(min(max(1, window), c.size))
    ma = np.convolve(c, np.ones(w) / w, mode="valid")
    i = int(np.argmax(ma))
    return int(round(i + (w - 1) / 2.0))


def boundary_hits(pred: Sequence[int], truth: Sequence[int], tol: int = 1
                  ) -> tuple[int, int, int]:
    """边界命中：返回 ``(命中数, len(pred), len(truth))``。

    命中 = 该 ``truth`` 边界在 ``pred`` 中存在 ``|差| <= tol`` 的对应项（一对一贪心匹配）。
    """
    p = sorted(int(x) for x in pred)
    t = sorted(int(x) for x in truth)
    used = [False] * len(p)
    hits = 0
    for b in t:
        best, bi = None, -1
        for i, q in enumerate(p):
            if used[i]:
                continue
            d = abs(q - b)
            if d <= tol and (best is None or d < best):
                best, bi = d, i
        if bi >= 0:
            used[bi] = True
            hits += 1
    return hits, len(p), len(t)


# ---------------------------------------------------------------------------
# 段落聚合
# ---------------------------------------------------------------------------


def fit_density_floor(pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
                      grid_lo: float = 0.0, grid_hi: float = 0.99,
                      n_grid: int = 199) -> dict:
    """在多曲上池化拟合最优密度地板 ``floor``。

    模型（`docs/audio-analysis.md` §4.7(2)）：
    ``density_norm = floor + (1 − floor)·I``。
    因为绝对量级另由"定数 → note/小节"锚定，这里把两边都**除以各自曲内均值**再比较，
    于是只剩形状：``y = d / mean(d)`` 对 ``ŷ = m / mean(m)``，``m = floor + (1−floor)·I``。

    参数：
        pairs: ``[(I, d), …]``，每首一组（逐小节或逐段均可）
    返回：
        ``{"floor": 最优值, "sse": 该处残差平方和, "curve": [(floor, sse), …]}``
    """
    data = []
    for I, d in pairs:
        Ia = np.asarray(I, dtype=float)
        da = np.asarray(d, dtype=float)
        if Ia.size < 2 or da.size != Ia.size or da.mean() <= 0:
            continue
        data.append((Ia, da / da.mean()))
    if not data:
        return {"floor": float("nan"), "sse": float("nan"), "curve": []}
    grid = np.linspace(grid_lo, grid_hi, int(n_grid))
    curve = []
    for f in grid:
        sse = 0.0
        for Ia, y in data:
            m = f + (1.0 - f) * Ia
            mm = float(m.mean())
            if mm <= 0:
                continue
            sse += float(np.sum((y - m / mm) ** 2))
        curve.append((float(f), sse))
    best = min(curve, key=lambda kv: kv[1])
    return {"floor": best[0], "sse": best[1],
            "curve": [(round(f, 3), round(s, 3)) for f, s in curve]}


def segment_means(values: Sequence[float], spans: Sequence[tuple[int, int]],
                  index_of: Callable[[int], int] | None = None) -> np.ndarray:
    """按 ``spans``（含端点的下标区间）求段均值；越界部分自动裁剪，空段记 NaN。"""
    v = np.asarray(values, dtype=float)
    out = np.full(len(spans), np.nan, dtype=float)
    for i, (a, b) in enumerate(spans):
        lo = a if index_of is None else index_of(a)
        hi = b if index_of is None else index_of(b)
        lo, hi = max(0, lo), min(v.size - 1, hi)
        if hi >= lo and v.size:
            out[i] = float(np.mean(v[lo:hi + 1]))
    return out
