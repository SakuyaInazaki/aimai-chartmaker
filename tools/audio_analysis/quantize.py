"""onset → 拍网格量化（v0.2，按调研 v2 §1.1 Step 2–7 重写）。

与 v0.1 的四点关键区别：

1. **拍同步重采样**：onset 先吸附到 `1/48 拍` 的格点（4/4 下每小节 192 个单位），
   与 simai 的 384 约数体系同源（v2 §1.1 Step 2）；
2. **每小节只选一个 div**：div 是"小节属性"而不是"轨属性"，所有轨都按该 div 渲染，
   杜绝 v0.1「kick 4 格 / other 24 格」这种同小节串长不一致（LLM 无法对齐）；
3. **容差 τ(d) = clamp(0.25·slot_ms(d), 12ms, 30ms)**：随分音变细而收紧，
   取代 v0.1 的 `max(25ms, 1/64 小节)`（后者在 192 BPM 下比 24/32 分格距还大，
   导致 16/24/32 之间近乎随机）；
4. **三连判别与 {32} 红线**：
   - 三连（12/24）需 `rms12 ≤ τ(12)` 且 `rms12 ≤ 0.7·rms16`（v2 Step 4）；
   - `{32}` 仅当 **来源是 drums 且该小节 drums onset ≥ 6 且 rms 残差 ≤ 15 ms**
     时放行（v2 Step 6）；否则上限压到 16/24，落格失败的 onset 计入 `unquantized`。

swing 判别（v2 Step 5）默认**关闭**：目标 BPM 区间（160–220）里 swing 出现概率低、
幅度小，误判代价高于收益。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 候选分音的扫描顺序（从粗到细；12 在 16 之前，靠三连判别把关）
DEFAULT_DIVISIONS = (4, 8, 12, 16, 24, 32)
# 拍同步重采样：每拍 48 个单位（4/4 下每小节 192），能同时整除 2 分族与 3 分族
BEAT_SUBUNITS = 48

TAU_SLOT_FRACTION = 0.25   # τ = 0.25 × 槽长
TAU_MIN_MS = 12.0          # 下限：低于此是解码/包络分辨率噪声
TAU_MAX_MS = 30.0          # 上限：不超过 onset 检测可信精度（标准窗 50ms 的一半量级）

TRIPLET_RMS_RATIO = 0.7    # rms12 ≤ 0.7 × rms16 才判三连
FINE_DIV = 32              # 红线分音
FINE_MIN_DRUM_ONSETS = 6   # {32} 放行条件 (a)
FINE_MAX_RMS_MS = 15.0     # {32} 放行条件 (c)
FALLBACK_DIV_BINARY = 16   # 红线拦下后的二分族上限
FALLBACK_DIV_TRIPLET = 24  # 红线拦下后的三分族上限


# ---------------- 容差与格点 ----------------


def slot_ms(division: int, bar_ms: float) -> float:
    """该分音下一个格子的毫秒数（division = 每小节等分数）。"""
    return bar_ms / float(division)


def tau_ms(division: int, bar_ms: float) -> float:
    """容差 τ(d) = clamp(0.25 × slot_ms(d), 12 ms, 30 ms)（v2 §1.1 Step 3）。"""
    return float(np.clip(TAU_SLOT_FRACTION * slot_ms(division, bar_ms),
                         TAU_MIN_MS, TAU_MAX_MS))


def bar_units(beats_per_bar: int = 4) -> int:
    """一小节的 1/48 拍单位数（4/4 → 192）。"""
    return int(beats_per_bar) * BEAT_SUBUNITS


def resample_to_beat_units(rel_sec: np.ndarray, bar_duration: float,
                           beats_per_bar: int = 4) -> np.ndarray:
    """拍同步重采样：小节内相对秒数 → 最近的 1/48 拍格点（整数单位）。

    v2 §1.1 Step 2：不要"固定 hop 提特征再找最近格子"，而是把事件按拍离散化。
    192 BPM 下一个单位 = 1250/192 ≈ 6.5 ms，远细于任何候选分音，重采样本身
    引入的误差可忽略（≤ 3.3 ms），但把后续所有分音判定都放到同一套整数格上。
    """
    n_units = bar_units(beats_per_bar)
    if np.size(rel_sec) == 0:
        return np.zeros(0, dtype=int)
    u = np.asarray(rel_sec, dtype=float) / max(bar_duration, 1e-12) * n_units
    return np.clip(np.rint(u).astype(int), 0, n_units - 1)


def _fit(rel_sec: np.ndarray, bar_duration: float, division: int
         ) -> tuple[np.ndarray, np.ndarray]:
    """把小节内相对秒数拟合到 `division` 等分网格。返回 (格号, 残差秒)。"""
    step = bar_duration / float(division)
    idx = np.rint(rel_sec / step).astype(int)
    idx = np.clip(idx, 0, division - 1)
    err = rel_sec - idx * step
    return idx, err


def _rms_ms(err_sec: np.ndarray) -> float:
    if err_sec.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(err_sec))) * 1000.0)


def _max_ms(err_sec: np.ndarray) -> float:
    if err_sec.size == 0:
        return 0.0
    return float(np.max(np.abs(err_sec)) * 1000.0)


# ---------------- 逐小节选 div ----------------


@dataclass
class BarDivision:
    """某一小节的分音判定结果（**全轨共用**）。"""

    bar: int
    division: int
    n_onsets: int = 0
    n_drum_onsets: int = 0
    rms_ms: float = 0.0
    max_ms: float = 0.0
    resolved: bool = True           # 是否所有 onset 都落在 τ 内
    n_unquantized: int = 0          # 落格失败的 onset 数
    triplet: bool = False           # 是否判为三连族
    fine_blocked: bool = False      # 是否被 {32} 红线拦下
    reason: str = ""
    candidates: dict = field(default_factory=dict)   # div → {rms_ms, max_ms, tau_ms}

    def to_dict(self) -> dict:
        return {
            "bar": self.bar,
            "division": self.division,
            "n_onsets": self.n_onsets,
            "rms_ms": round(self.rms_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "resolved": self.resolved,
            "n_unquantized": self.n_unquantized,
            "triplet": self.triplet,
            "fine_blocked": self.fine_blocked,
            "reason": self.reason,
        }


def choose_bar_division(
    rel_sec: np.ndarray,
    bar_duration: float,
    beats_per_bar: int = 4,
    divisions=DEFAULT_DIVISIONS,
    n_drum_onsets: int = 0,
    bar: int = 0,
    allow_fine: bool = True,
    outlier_ratio: float = 0.0,
) -> BarDivision:
    """为一个小节挑唯一的分音 `{d}`（v2 §1.1 Step 3/4/6）。

    参数：
        rel_sec: 小节内所有（各轨合并后的）onset 相对秒数
        n_drum_onsets: 该小节 drums stem 的 onset 数——{32} 红线只认鼓轨
        allow_fine: 关掉后 {32} 一律不放行（测试与保守模式用）
        outlier_ratio: 允许多少比例的 onset 超出 τ 仍判该分音成立。
            **默认 0.0 = 严格照 v2 §1.1 Step 3 的 `r(d) = max_i` 口径。**
            实跑发现：四条 stem 合并后每小节中位 15–22 个 onset，而 `other`/`bass`/
            `vocals` 的 onset 定位误差本身就有 15–33 ms，一个离群点就能否掉整个小节
            （实测 78–87% 的小节走了"压到上限"分支）。调到 0.1 左右能让分音直方图
            重新有信息量，但那是**偏离 v2 口径**的做法，因此做成显式开关、默认不开。
    """
    rel_sec = np.atleast_1d(np.asarray(rel_sec, dtype=float))
    rel_sec = rel_sec[np.isfinite(rel_sec)]
    bar_ms = bar_duration * 1000.0
    if rel_sec.size == 0:
        return BarDivision(bar=bar, division=divisions[0], n_onsets=0,
                           reason="空小节")

    # 拍同步重采样（1/48 拍），再用重采样后的位置做分音判定
    units = resample_to_beat_units(rel_sec, bar_duration, beats_per_bar)
    rel_q = units.astype(float) / bar_units(beats_per_bar) * bar_duration

    cand: dict[int, dict] = {}
    for d in divisions:
        idx, err = _fit(rel_q, bar_duration, d)
        cand[d] = {"idx": idx, "err": err, "rms_ms": _rms_ms(err),
                   "max_ms": _max_ms(err), "tau_ms": tau_ms(d, bar_ms)}

    # --- Step 4：三连判别（独立于扫描）---
    rms12 = cand.get(12, {}).get("rms_ms")
    rms16 = cand.get(16, {}).get("rms_ms")
    triplet = False
    if rms12 is not None and rms16 is not None:
        triplet = (rms12 <= tau_ms(12, bar_ms)) and (rms12 <= TRIPLET_RMS_RATIO * rms16)

    # --- Step 6：{32} 红线 ---
    rms32 = cand.get(FINE_DIV, {}).get("rms_ms", float("inf"))
    fine_ok = bool(
        allow_fine
        and rel_sec.size >= FINE_MIN_DRUM_ONSETS
        and n_drum_onsets >= FINE_MIN_DRUM_ONSETS
        and rms32 <= FINE_MAX_RMS_MS
    )

    # --- Step 3：从粗到细扫描 ---
    max_outliers = int(np.floor(max(0.0, outlier_ratio) * rel_sec.size))
    for d in divisions:
        c = cand[d]
        if d in (12, 24) and not triplet:
            continue                      # 三分族没拿到显著证据 → 跳过
        if d >= FINE_DIV and not fine_ok:
            continue                      # 红线
        over = int(np.sum(np.abs(c["err"]) * 1000.0 > c["tau_ms"]))
        if over <= max_outliers:
            return BarDivision(
                bar=bar, division=d, n_onsets=int(rel_sec.size),
                n_drum_onsets=int(n_drum_onsets),
                rms_ms=c["rms_ms"], max_ms=c["max_ms"], resolved=True,
                n_unquantized=over, triplet=triplet and d in (12, 24),
                fine_blocked=False,
                reason=f"τ({d})={c['tau_ms']:.1f}ms 内解释全部 onset",
                candidates={str(k): {"rms_ms": round(v["rms_ms"], 2),
                                     "max_ms": round(v["max_ms"], 2),
                                     "tau_ms": round(v["tau_ms"], 1)}
                            for k, v in cand.items()},
            )

    # --- 没有分音能解释：压到上限，落格失败的 onset 计入 unquantized ---
    cap = FALLBACK_DIV_TRIPLET if triplet else FALLBACK_DIV_BINARY
    cap = max(d for d in divisions if d <= cap) if any(d <= cap for d in divisions) else divisions[-1]
    c = cand[cap]
    bad = int(np.sum(np.abs(c["err"]) * 1000.0 > c["tau_ms"]))
    fine_blocked = (not fine_ok) and rms32 <= cand[cap]["rms_ms"]
    return BarDivision(
        bar=bar, division=cap, n_onsets=int(rel_sec.size),
        n_drum_onsets=int(n_drum_onsets),
        rms_ms=c["rms_ms"], max_ms=c["max_ms"], resolved=False,
        n_unquantized=bad, triplet=triplet, fine_blocked=fine_blocked,
        reason=(f"无分音能在 τ 内解释（上限压到 {cap}）"
                + ("；{32} 被红线拦下" if fine_blocked else "")),
        candidates={str(k): {"rms_ms": round(v["rms_ms"], 2),
                             "max_ms": round(v["max_ms"], 2),
                             "tau_ms": round(v["tau_ms"], 1)}
                    for k, v in cand.items()},
    )


# ---------------- 逐轨渲染 ----------------


def assign_onsets_to_bars(times: np.ndarray, grid) -> dict[int, list[int]]:
    """把 onset 秒数按小节分组（落在小节末尾 τ(8) 内的吸附到下一小节）。"""
    out: dict[int, list[int]] = {}
    times = np.atleast_1d(np.asarray(times, dtype=float))
    for i, t in enumerate(times):
        bar = grid.bar_of(float(t))
        if bar <= 0:
            continue
        bd = grid.bar_duration(bar)
        tol = tau_ms(8, bd * 1000.0) / 1000.0
        if (grid.bar_start(bar) + bd) - float(t) <= tol and bar + 1 <= grid.n_bars:
            bar += 1
        out.setdefault(bar, []).append(i)
    return out


@dataclass
class BarPattern:
    """某条逻辑轨在某小节的网格串。"""

    bar: int
    track: str
    division: int
    pattern: str
    n_onsets: int = 0
    errors_ms: list[float] = field(default_factory=list)
    raw_times: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "division": self.division,
            "pattern": self.pattern,
            "n_onsets": self.n_onsets,
            "max_err_ms": round(max((abs(e) for e in self.errors_ms), default=0.0), 2),
        }


def render_pattern(division: int, idx: np.ndarray, chars: list[str],
                   sustain: np.ndarray | None = None) -> str:
    """把格号 + 字符渲染成网格串。

    字符集严格限定 `X x - .`：
      `X` 重音（kick / 强人声 / 强 onset）、`x` 普通 onset、
      `-` 延音持续（只由 `sustain` 掩码给出）、`.` 空。
    同格重复 onset 取"强者优先"（X > x）。
    """
    out = ["."] * int(division)
    if sustain is not None:
        for s in range(int(division)):
            if s < len(sustain) and sustain[s]:
                out[s] = "-"
    rank = {".": 0, "-": 1, "x": 2, "X": 3}
    for i, slot in enumerate(np.atleast_1d(idx)):
        s = int(slot)
        if not (0 <= s < division):
            continue
        ch = chars[i] if i < len(chars) else "x"
        if rank.get(ch, 0) > rank.get(out[s], 0):
            out[s] = ch
    return "".join(out)


def quantize_song(
    onset_times: dict[str, np.ndarray],
    grid,
    divisions=DEFAULT_DIVISIONS,
    drum_key: str = "drums",
    div_source_keys: tuple[str, ...] | None = None,
    allow_fine: bool = True,
    outlier_ratio: float = 0.0,
) -> tuple[dict[int, BarDivision], dict[str, dict[int, list[int]]], dict]:
    """整曲量化：先逐小节定唯一 div，再把各条 onset 流按该 div 归格。

    参数：
        onset_times: {流名: onset 秒数数组}（含 drums/bass/other/vocals 与鼓件）
        div_source_keys: 参与"定 div"的流（默认取四条 stem，鼓件不重复计入）

    返回：
        (bar_div, per_track_slots, stats)
        - `bar_div[bar]` → BarDivision
        - `per_track_slots[track][bar]` → 该轨在该小节的格号列表（与 onset 顺序一致）
        - `stats` → 整曲量化统计
    """
    if div_source_keys is None:
        div_source_keys = tuple(k for k in ("drums", "bass", "other", "vocals")
                                if k in onset_times)

    per_track_bars = {k: assign_onsets_to_bars(v, grid) for k, v in onset_times.items()}

    bar_div: dict[int, BarDivision] = {}
    for bar in range(1, grid.n_bars + 1):
        bd = grid.bar_duration(bar)
        start = grid.bar_start(bar)
        rel: list[float] = []
        for k in div_source_keys:
            ids = per_track_bars.get(k, {}).get(bar, [])
            if ids:
                rel.extend(float(onset_times[k][i]) - start for i in ids)
        n_drum = len(per_track_bars.get(drum_key, {}).get(bar, []))
        bar_div[bar] = choose_bar_division(
            np.asarray(rel, dtype=float), bd, grid.beats_per_bar, divisions,
            n_drum_onsets=n_drum, bar=bar, allow_fine=allow_fine,
            outlier_ratio=outlier_ratio)

    # 各轨按该小节的 div 归格
    per_track_slots: dict[str, dict[int, list[int]]] = {}
    per_track_err: dict[str, list[float]] = {}
    for name, times in onset_times.items():
        slots: dict[int, list[int]] = {}
        errs: list[float] = []
        for bar, ids in per_track_bars.get(name, {}).items():
            if bar not in bar_div:
                continue
            d = bar_div[bar].division
            bd = grid.bar_duration(bar)
            rel = np.asarray([float(times[i]) - grid.bar_start(bar) for i in ids])
            idx, err = _fit(rel, bd, d)
            slots[bar] = [int(x) for x in idx]
            errs.extend(float(e * 1000.0) for e in err)
        per_track_slots[name] = slots
        per_track_err[name] = errs

    div_hist: dict[int, int] = {}
    div_hist_resolved: dict[int, int] = {}
    n_unq = 0
    n_total_onsets = 0
    bars_unresolved = 0
    for bd_ in bar_div.values():
        if bd_.n_onsets == 0:
            continue
        div_hist[bd_.division] = div_hist.get(bd_.division, 0) + 1
        n_unq += bd_.n_unquantized
        n_total_onsets += bd_.n_onsets
        if bd_.resolved:
            div_hist_resolved[bd_.division] = div_hist_resolved.get(bd_.division, 0) + 1
        else:
            bars_unresolved += 1

    total = sum(div_hist.values()) or 1
    stats = {
        "bars_with_onsets": int(sum(div_hist.values())),
        "division_histogram": {str(k): v for k, v in sorted(div_hist.items())},
        "division_share": {str(k): round(v / total, 4)
                           for k, v in sorted(div_hist.items())},
        # ⚠️ 区分「真的选中」与「找不到分音后被压到上限」——后者不是分音判定结论
        "division_histogram_resolved": {str(k): v
                                        for k, v in sorted(div_hist_resolved.items())},
        "resolved_bars": int(sum(div_hist_resolved.values())),
        "unresolved_bars": int(bars_unresolved),
        "unquantized_onsets": int(n_unq),
        "onsets_considered": int(n_total_onsets),
        "unquantized_ratio": round(n_unq / max(n_total_onsets, 1), 4),
        "outlier_ratio": float(outlier_ratio),
        "fine_blocked_bars": int(sum(1 for b in bar_div.values() if b.fine_blocked)),
        "triplet_bars": int(sum(1 for b in bar_div.values()
                                if b.triplet and b.division in (12, 24))),
        "per_track": {
            name: {
                "n_onsets": int(len(onset_times[name])),
                "mean_abs_err_ms": round(float(np.mean(np.abs(e))), 2) if e else 0.0,
                "p95_abs_err_ms": round(float(np.percentile(np.abs(e), 95)), 2) if e else 0.0,
                "max_abs_err_ms": round(float(np.max(np.abs(e))), 2) if e else 0.0,
            }
            for name, e in per_track_err.items()
        },
    }
    return bar_div, per_track_slots, stats
