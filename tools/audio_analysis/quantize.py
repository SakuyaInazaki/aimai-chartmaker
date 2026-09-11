"""onset → 拍网格量化。

逐小节在候选分音 {4,8,12,16,24,32} 中选**能在容差内解释该小节全部 onset 的
最小分音**，输出每小节每 stem 的网格字符串（simai 的 `{分音}` 直接对应）：

    16 分：`x...x...x.x.....`   `.` = 空，`x` = 有 onset，`X` = 强 onset

同时保留原始 onset 秒数与量化误差，供人工判断"这一小节是不是真的能对上网格"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_DIVISIONS = (4, 8, 12, 16, 24, 32)
DEFAULT_TOL_SEC = 0.025      # ±25ms
TOL_BAR_FRACTION = 1.0 / 64  # 或 1/64 小节，取大者


@dataclass
class BarQuant:
    """某个 stem 在某一小节的量化结果。"""

    bar: int
    stem: str
    division: int
    pattern: str
    n_onsets: int
    errors_ms: list[float] = field(default_factory=list)
    resolved: bool = True        # 是否在容差内被解释
    raw_times: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bar": self.bar,
            "stem": self.stem,
            "division": self.division,
            "pattern": self.pattern,
            "n_onsets": self.n_onsets,
            "max_err_ms": round(max((abs(e) for e in self.errors_ms), default=0.0), 2),
            "resolved": self.resolved,
            "raw_times": [round(t, 4) for t in self.raw_times],
        }


def tolerance_for_bar(bar_duration: float,
                      base: float = DEFAULT_TOL_SEC,
                      frac: float = TOL_BAR_FRACTION) -> float:
    """容差 = max(±25ms, 1/64 小节)。"""
    return max(base, bar_duration * frac)


def assign_bars(times: np.ndarray, grid, tol: float | None = None) -> dict[int, np.ndarray]:
    """把 onset 秒数按小节分组；落在小节末尾容差内的归到下一小节（避免 idx==division）。"""
    out: dict[int, list[float]] = {}
    for t in np.atleast_1d(times):
        bar = grid.bar_of(float(t))
        if bar <= 0:
            continue
        bd = grid.bar_duration(bar)
        this_tol = tol if tol is not None else tolerance_for_bar(bd)
        end = grid.bar_start(bar) + bd
        if end - float(t) <= this_tol and bar + 1 <= grid.n_bars:
            bar += 1
        out.setdefault(bar, []).append(float(t))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def choose_division(rel: np.ndarray, bar_duration: float,
                    divisions=DEFAULT_DIVISIONS,
                    tol: float | None = None) -> tuple[int, np.ndarray, np.ndarray, bool]:
    """在候选分音中选能解释全部 onset 的最小分音。

    返回 (division, slot_indices, errors_sec, resolved)。
    """
    if tol is None:
        tol = tolerance_for_bar(bar_duration)
    if rel.size == 0:
        return divisions[0], np.zeros(0, dtype=int), np.zeros(0), True

    for d in divisions:
        step = bar_duration / d
        idx = np.rint(rel / step).astype(int)
        err = rel - idx * step
        if np.all(np.abs(err) <= tol) and np.all(idx < d) and np.all(idx >= 0):
            return d, idx, err, True

    # 没有分音能解释 → 用最细的分音尽力拟合，并标记 unresolved
    d = divisions[-1]
    step = bar_duration / d
    idx = np.clip(np.rint(rel / step).astype(int), 0, d - 1)
    err = rel - idx * step
    return d, idx, err, False


def make_pattern(division: int, idx: np.ndarray, strong: np.ndarray | None = None) -> str:
    """生成网格字符串。同一格重复 onset 只保留一个（强者优先）。"""
    chars = ["."] * division
    for i, slot in enumerate(idx):
        s = int(slot)
        if not (0 <= s < division):
            continue
        is_strong = bool(strong[i]) if strong is not None and i < len(strong) else False
        if chars[s] == "X":
            continue
        chars[s] = "X" if is_strong else ("x" if chars[s] == "." else chars[s])
    return "".join(chars)


def quantize_track(
    times: np.ndarray,
    strong: np.ndarray | None,
    stem: str,
    grid,
    divisions=DEFAULT_DIVISIONS,
) -> tuple[dict[int, BarQuant], dict]:
    """把一条 onset 流量化到整曲网格。

    返回 (逐小节结果 dict[bar]→BarQuant, 统计摘要)。
    """
    times = np.atleast_1d(np.asarray(times, dtype=float))
    strong = None if strong is None else np.atleast_1d(np.asarray(strong))
    order = np.argsort(times)
    times = times[order]
    if strong is not None:
        strong = strong[order]

    # 逐 onset 记录所属小节（含末尾吸附）
    bars: dict[int, list[int]] = {}
    for i, t in enumerate(times):
        bar = grid.bar_of(float(t))
        if bar <= 0:
            continue
        bd = grid.bar_duration(bar)
        tol = tolerance_for_bar(bd)
        if (grid.bar_start(bar) + bd) - float(t) <= tol and bar + 1 <= grid.n_bars:
            bar += 1
        bars.setdefault(bar, []).append(i)

    results: dict[int, BarQuant] = {}
    all_err: list[float] = []
    div_hist: dict[int, int] = {}
    unresolved = 0

    for bar in range(1, grid.n_bars + 1):
        ids = bars.get(bar, [])
        bd = grid.bar_duration(bar)
        if not ids:
            results[bar] = BarQuant(bar, stem, divisions[0], "." * divisions[0], 0)
            continue
        rel = times[ids] - grid.bar_start(bar)
        rel = np.clip(rel, 0.0, bd)
        d, idx, err, ok = choose_division(rel, bd, divisions)
        st = None if strong is None else strong[ids]
        pattern = make_pattern(d, idx, st)
        errs_ms = [float(e * 1000.0) for e in err]
        results[bar] = BarQuant(bar, stem, d, pattern, len(ids), errs_ms, ok,
                                [float(t) for t in times[ids]])
        all_err.extend(abs(e) for e in errs_ms)
        div_hist[d] = div_hist.get(d, 0) + 1
        if not ok:
            unresolved += 1

    stats = {
        "stem": stem,
        "n_onsets": int(len(times)),
        "bars_with_onsets": int(sum(1 for b in results.values() if b.n_onsets > 0)),
        "mean_abs_err_ms": round(float(np.mean(all_err)), 2) if all_err else 0.0,
        "p95_abs_err_ms": round(float(np.percentile(all_err, 95)), 2) if all_err else 0.0,
        "max_abs_err_ms": round(float(np.max(all_err)), 2) if all_err else 0.0,
        "unresolved_bars": unresolved,
        "division_histogram": {str(k): v for k, v in sorted(div_hist.items())},
    }
    return results, stats
