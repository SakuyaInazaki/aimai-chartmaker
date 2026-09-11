"""官方 note 时间 × 各 stem onset 的匹配 —— "官方谱师这一段实际踩了哪条轨"。

两个方向的指标（两者**不可互换**）：

- **命中率 recall**：官方 note（时间槽）中，在 ±τ 内能找到该 stem onset 的比例
  → "这一段的官方踩音有多少能被这条轨解释"。
- **精确率 precision**：该 stem 的 onset 中，被官方 note 采用的比例
  → "这条轨的音有多少真被写进谱面"（= 候选池的有效率）。

τ 默认 30 ms（与 `tools/audio_analysis/quantize.py` 的量化容差上限同量级；
librosa onset 的帧分辨率 512/22050 = 23.2 ms，所以 30 ms 已经接近分辨率下限）。

⚠️ 口径：官方谱一个时间槽内常有多个 note（each/双押），对"踩了哪条轨"来说
**同刻的多个 note 只是一个踩音事件**，所以默认以**去重后的时间槽**为单位统计；
逐 note 口径同时给出，供对照。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TOL_SEC = 0.030


def unique_times(times, merge_sec: float = 1e-4) -> np.ndarray:
    """把同刻（或极近）事件去重成时间槽。"""
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if t.size == 0:
        return np.zeros(0)
    t = np.sort(t)
    keep = [float(t[0])]
    for x in t[1:]:
        if float(x) - keep[-1] > merge_sec:
            keep.append(float(x))
    return np.asarray(keep, dtype=float)


def nearest_gap(query, reference) -> np.ndarray:
    """``query`` 中每个时间到 ``reference`` 最近元素的绝对时间差（秒）。

    ``reference`` 为空时全部返回 ``+inf``。
    """
    q = np.atleast_1d(np.asarray(query, dtype=float))
    r = np.atleast_1d(np.asarray(reference, dtype=float))
    if q.size == 0:
        return np.zeros(0)
    if r.size == 0:
        return np.full(q.size, np.inf)
    r = np.sort(r)
    idx = np.searchsorted(r, q)
    left = np.clip(idx - 1, 0, r.size - 1)
    right = np.clip(idx, 0, r.size - 1)
    return np.minimum(np.abs(q - r[left]), np.abs(q - r[right]))


def covered_mask(query, reference, tol: float = DEFAULT_TOL_SEC) -> np.ndarray:
    """``query`` 中每个事件是否在 ±tol 内被 ``reference`` 覆盖。"""
    return nearest_gap(query, reference) <= float(tol)


@dataclass
class HitStat:
    n_events: int = 0          # 官方时间槽数
    n_onsets: int = 0          # 该 stem 的 onset 数
    n_hit: int = 0             # 被该 stem 覆盖的官方时间槽数
    n_used: int = 0            # 被官方采用的该 stem onset 数
    span_sec: float = 0.0      # 统计区间长度（秒），用于随机基线
    tol: float = DEFAULT_TOL_SEC

    @property
    def recall(self) -> float:
        return self.n_hit / self.n_events if self.n_events else float("nan")

    @property
    def precision(self) -> float:
        return self.n_used / self.n_onsets if self.n_onsets else float("nan")

    @property
    def chance_recall(self) -> float:
        """随机基线：把 ``n_onsets`` 个 onset 均匀撒在区间上时的期望命中率。

        ⚠️ **必须看这个数**：鼓轨的 onset 天然最多，只比 recall 的话它永远赢。
        ``1 − exp(−λ·2τ)``，λ = onset 密度（个/秒）。
        """
        if self.span_sec <= 0 or self.n_onsets <= 0:
            return float("nan")
        lam = self.n_onsets / self.span_sec
        return float(1.0 - np.exp(-lam * 2.0 * self.tol))

    @property
    def lift(self) -> float:
        """命中率 / 随机基线。> 1 表示官方踩音确实跟着这条轨，而不只是"轨太密"。"""
        c = self.chance_recall
        if not np.isfinite(c) or c <= 0:
            return float("nan")
        return float(self.recall / c)

    def to_dict(self) -> dict:
        def r(v):
            return round(float(v), 4) if np.isfinite(v) else None
        return {"n_events": self.n_events, "n_onsets": self.n_onsets,
                "n_hit": self.n_hit, "n_used": self.n_used,
                "recall": r(self.recall), "precision": r(self.precision),
                "chance_recall": r(self.chance_recall), "lift": r(self.lift)}


def hit_stat(event_times, onset_times, tol: float = DEFAULT_TOL_SEC,
             span_sec: float = 0.0) -> HitStat:
    """一条 stem 在一个时间区间上的命中率 / 精确率 / 随机基线 lift。"""
    ev = np.atleast_1d(np.asarray(event_times, dtype=float))
    on = np.atleast_1d(np.asarray(onset_times, dtype=float))
    if span_sec <= 0:
        pool = np.concatenate([ev, on]) if (ev.size or on.size) else np.zeros(0)
        span_sec = float(pool.max() - pool.min()) if pool.size >= 2 else 0.0
    return HitStat(
        n_events=int(ev.size), n_onsets=int(on.size),
        n_hit=int(np.sum(covered_mask(ev, on, tol))) if ev.size else 0,
        n_used=int(np.sum(covered_mask(on, ev, tol))) if on.size else 0,
        span_sec=float(span_sec), tol=float(tol),
    )


def best_shift(event_times, onset_times, shifts_sec, tol: float = DEFAULT_TOL_SEC
               ) -> dict:
    """全局相位扫描：把官方 note 时间整体平移 φ，找覆盖率最高的 φ。

    这是**用谱面反过来校验 mp3 对齐**的最直接手段——官方包的 `&first` 是 ground truth，
    所以 φ* 偏离 0 只能来自 mp3 解码/容器偏移。
    """
    ev = np.atleast_1d(np.asarray(event_times, dtype=float))
    shifts = np.atleast_1d(np.asarray(shifts_sec, dtype=float))
    if ev.size == 0 or shifts.size == 0:
        return {"best_shift_sec": 0.0, "best_rate": float("nan"),
                "rate_at_zero": float("nan"), "curve": []}
    rates = []
    for s in shifts:
        rates.append(float(np.mean(covered_mask(ev + float(s), onset_times, tol))))
    rates_arr = np.asarray(rates, dtype=float)
    # 容差比步长大得多时，最优 φ 会是一个**平台**而非单点；取平台中位数，
    # 否则 argmax 只会挑到平台左端（实测偏 −7.5 ms）。
    top = float(rates_arr.max())
    plateau = shifts[rates_arr >= top - 1e-12]
    phi = float(np.median(plateau))
    i = int(np.argmin(np.abs(shifts - phi)))
    j = int(np.argmin(np.abs(shifts)))
    return {
        "best_shift_sec": float(shifts[i]),
        "best_rate": float(rates_arr[i]),
        "plateau_ms": [round(float(plateau.min()) * 1000, 2),
                       round(float(plateau.max()) * 1000, 2)],
        "rate_at_zero": float(rates_arr[j]),
        "curve": [(round(float(s), 4), round(float(r), 4))
                  for s, r in zip(shifts, rates_arr)],
    }


def explain_breakdown(event_times, stem_onsets: dict, tol: float = DEFAULT_TOL_SEC
                      ) -> dict:
    """全曲层面的归因：每个官方时间槽被哪些 stem 解释。

    返回各类占比：``drums_only`` / ``vocals_only`` / ``both_drums_vocals`` /
    ``other_or_bass_only`` / ``none``（谱师自由发挥或装饰音）/ ``any``。
    """
    ev = np.atleast_1d(np.asarray(event_times, dtype=float))
    n = int(ev.size)
    if n == 0:
        return {}
    masks = {k: covered_mask(ev, v, tol) for k, v in stem_onsets.items()}
    d = masks.get("drums", np.zeros(n, dtype=bool))
    v = masks.get("vocals", np.zeros(n, dtype=bool))
    b = masks.get("bass", np.zeros(n, dtype=bool))
    o = masks.get("other", np.zeros(n, dtype=bool))
    any_ = d | v | b | o
    return {
        "n_events": n,
        "any": float(np.mean(any_)),
        "none": float(np.mean(~any_)),
        "drums": float(np.mean(d)),
        "vocals": float(np.mean(v)),
        "bass": float(np.mean(b)),
        "other": float(np.mean(o)),
        "drums_only": float(np.mean(d & ~v & ~b & ~o)),
        "vocals_only": float(np.mean(v & ~d & ~b & ~o)),
        "both_drums_vocals": float(np.mean(d & v)),
        "no_drums": float(np.mean(~d)),
    }


def rank_stems(event_times, stem_onsets: dict, tol: float = DEFAULT_TOL_SEC,
               span_sec: float = 0.0, by: str = "recall") -> list[tuple[str, float]]:
    """按 ``by``（``recall`` 或 ``lift``）降序给出该区间最可能的主踩音轨。"""
    out = []
    for k, v in stem_onsets.items():
        st = hit_stat(event_times, v, tol, span_sec=span_sec)
        val = st.lift if by == "lift" else st.recall
        if np.isfinite(val):
            out.append((k, float(val)))
    return sorted(out, key=lambda kv: kv[1], reverse=True)
