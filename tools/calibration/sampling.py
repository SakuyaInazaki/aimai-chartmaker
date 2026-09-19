#!/usr/bin/env python3
"""采音模式度量：官方谱在每个小节**怎么采**骨架轨的音。

问题
----
`stemhit.py` 回答的是"官方踩的是哪条轨"（recall / precision / lift）。
本模块回答的是**同一条轨上、采音的形态**：

- **全采**：把那条轨这一小节的音基本都踩了；
- **半采**：只踩一半（隔一个踩 / 只踩强拍 / 谱面分音 = 音乐分音的一半）；
- **稀采（空音）**：音轨在响，谱面却大面积留白；
- **加花**：谱面写了音轨里没有的音（自由发挥）；
- **静默**：这一小节谱面几乎不写音；
- **混合**：都不像。

⚠️ **这些类别的阈值是 agent 的操作化**，不是用户讲授的术语定义。
官方/社区并没有"全采 / 半采 / 空音"的成文判据；本模块给出的是一套
可复现、可调参、可做敏感性分析的操作化，报告里必须照此说明，并把
各类的谱例小节列出来供用户校正术语。参数集中在 :data:`THRESHOLDS`。

口径
----
- **时间基准**：官方 note 时间 = ``&first + note.time + φ``（φ = 全局相位补偿，
  与 `tools/calibration/cli.py` 的 `best_shift` 同口径，|φ*| ≤ 10 ms 时不补偿）；
- **官方时间槽**：同刻的 each / 双押算**一个采音事件**（`stemhit.unique_times`）；
- **pool（候选池）**：骨架轨在该小节的 onset，按 ``2τ = 60 ms`` 合并去重
  （两个挨得比容差还近的 onset，谱面上只可能对应一个音）；
- **hit**：pool 中被官方时间槽 ±τ 覆盖的个数（= `stemhit` 的 ``n_used``）；
- **extra**：官方时间槽中，**任何一条 stem** 都覆盖不到的个数（自由发挥/装饰/漏检）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import stemhit

#: 参与 "extra" 判定的候选轨（与 n=160 标定的候选池口径一致：四路 + piano）
POOL_STEMS = ("drums", "bass", "other", "vocals", "piano")
#: 可以当骨架的轨
SKELETON_STEMS = ("drums", "bass", "other", "vocals", "piano")

#: 采音模式的操作化阈值（**agent 设定**，可整体替换；敏感性见报告）
THRESHOLDS: dict = {
    "full_coverage": 0.80,      # 全采：coverage ≥
    "full_extra_max": 0.30,     # 全采：extra_ratio <
    "half_lo": 0.35,            # 半采：coverage ∈ [half_lo, half_hi]
    "half_hi": 0.65,
    "sparse_coverage": 0.35,    # 稀采/空音：coverage <
    "flourish_extra": 0.50,     # 加花：extra_ratio ≥
    "silent_slots": 1,          # 静默：官方时间槽 ≤
    "min_pool": 2,              # pool 少于这么多就不给 coverage 下结论
    "merge_sec": 0.060,         # pool 去重granularity（= 2τ）
    "strong_beat_grid": 0.5,    # "只踩强拍"= 全部落在 0.5 拍的整数倍上
}


# ---------------------------------------------------------------------------
# 逐小节记录
# ---------------------------------------------------------------------------


@dataclass
class BarSampling:
    """一小节的采音度量。"""

    bar: int                       # 音频小节号（1 起）
    chart_measure: int             # 谱面小节号（0 起）
    t0: float
    t1: float
    skeleton: str = "drums"
    n_pool: int = 0
    n_slots: int = 0               # 官方时间槽数（同刻算一个）
    n_notes: int = 0               # 官方 note 数
    hit: int = 0
    extra: int = 0
    coverage: float = float("nan")
    extra_ratio: float = float("nan")
    alt_pattern: bool = False
    alt_reason: str = ""
    rest_beats: float = 0.0
    density_ratio: float = float("nan")   # 官方槽数 / pool 大小
    mode: str = "静默"
    per_stem_pool: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def r(v, n=4):
            return None if not np.isfinite(v) else round(float(v), n)
        return {"bar": self.bar, "chart_measure": self.chart_measure,
                "skeleton": self.skeleton, "n_pool": self.n_pool,
                "n_slots": self.n_slots, "n_notes": self.n_notes,
                "hit": self.hit, "extra": self.extra,
                "coverage": r(self.coverage), "extra_ratio": r(self.extra_ratio),
                "alt_pattern": self.alt_pattern, "alt_reason": self.alt_reason,
                "rest_beats": r(self.rest_beats, 3),
                "density_ratio": r(self.density_ratio), "mode": self.mode}


def classify(coverage: float, extra_ratio: float, n_slots: int,
             n_pool: int, alt_pattern: bool, th: dict | None = None) -> str:
    """把一小节的度量落成采音模式（**操作化定义**，见模块 docstring）。

    判定顺序（先判的优先）：
    ``静默`` → ``加花`` → ``全采`` → ``半采`` → ``稀采/空音`` → ``混合``。
    """
    th = THRESHOLDS if th is None else th
    if n_slots <= th["silent_slots"]:
        return "静默"
    if np.isfinite(extra_ratio) and extra_ratio >= th["flourish_extra"]:
        return "加花"
    if n_pool < th["min_pool"] or not np.isfinite(coverage):
        return "混合"
    if coverage >= th["full_coverage"] and extra_ratio < th["full_extra_max"]:
        return "全采"
    if th["half_lo"] <= coverage <= th["half_hi"] and alt_pattern:
        return "半采"
    if coverage < th["sparse_coverage"]:
        return "稀采/空音"
    return "混合"


# ---------------------------------------------------------------------------
# 单小节度量
# ---------------------------------------------------------------------------


def _alt_check(slot_times: np.ndarray, pool: np.ndarray, t0: float,
               sec_per_beat: float, chart_div: float, audio_div: float,
               tol: float, th: dict) -> tuple[bool, str]:
    """`alt_pattern`：官方槽是不是"pool 的隔一个" / "只踩强拍" / "分音减半"。

    三条判据任一成立即算（各自写进 ``alt_reason``，便于事后拆开看）：

    1. **隔一个**：被踩中的 pool 下标构成等差为 2 的序列（≥3 个）；
    2. **只踩强拍**：全部官方槽都落在 ``strong_beat_grid`` 拍的整数倍上，
       而 pool 至少有一个落在半拍之间（否则"只踩强拍"没有区分度）；
    3. **分音减半**：谱面该小节的最细分音 = 音频该小节量化分音的一半。
    """
    reasons: list[str] = []
    if pool.size >= 3 and slot_times.size >= 2:
        hit_idx = [i for i, p in enumerate(pool)
                   if np.min(np.abs(slot_times - p)) <= tol]
        if len(hit_idx) >= 3:
            d = np.diff(hit_idx)
            if np.all(d == 2):
                reasons.append("隔一个")
    if slot_times.size >= 2 and sec_per_beat > 0:
        g = th["strong_beat_grid"]
        pos = ((slot_times - t0) / sec_per_beat) % g
        on_strong = np.all(np.minimum(pos, g - pos) <= tol / sec_per_beat)
        pool_pos = ((pool - t0) / sec_per_beat) % g if pool.size else np.zeros(0)
        pool_off = (np.any(np.minimum(pool_pos, g - pool_pos) > tol / sec_per_beat)
                    if pool.size else False)
        if on_strong and pool_off:
            reasons.append("只踩强拍")
    if chart_div > 0 and audio_div > 0 and abs(chart_div * 2 - audio_div) < 1e-6:
        reasons.append("分音减半")
    return bool(reasons), "+".join(reasons)


def bar_metrics(slot_times: Sequence[float], note_count: int,
                skeleton_pool: Sequence[float], any_onsets: Sequence[float],
                t0: float, t1: float, sec_per_beat: float,
                chart_div: float = 0.0, audio_div: float = 0.0,
                tol: float = stemhit.DEFAULT_TOL_SEC,
                th: dict | None = None) -> dict:
    """一小节的全部采音度量（纯函数，方便单测）。"""
    th = THRESHOLDS if th is None else th
    ev = np.atleast_1d(np.asarray(slot_times, dtype=float))
    pool = np.atleast_1d(np.asarray(skeleton_pool, dtype=float))
    allon = np.atleast_1d(np.asarray(any_onsets, dtype=float))
    n_slots = int(ev.size)
    n_pool = int(pool.size)
    hit = int(np.sum(stemhit.covered_mask(pool, ev, tol))) if n_pool else 0
    extra = int(np.sum(~stemhit.covered_mask(ev, allon, tol))) if n_slots else 0
    coverage = hit / n_pool if n_pool else float("nan")
    extra_ratio = extra / n_slots if n_slots else float("nan")
    alt, reason = _alt_check(ev, pool, t0, sec_per_beat, chart_div, audio_div, tol, th)
    # 最长留白：相邻官方槽之间（含小节两端）的最大间隔，且该区间内 pool 非空
    rest = 0.0
    edges = np.concatenate([[t0], np.sort(ev), [t1]])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 0:
            continue
        if np.any((pool > a + tol) & (pool < b - tol)):
            rest = max(rest, (b - a) / sec_per_beat if sec_per_beat > 0 else 0.0)
    mode = classify(coverage, extra_ratio, n_slots, n_pool, alt, th)
    return {"n_slots": n_slots, "n_notes": int(note_count), "n_pool": n_pool,
            "hit": hit, "extra": extra, "coverage": coverage,
            "extra_ratio": extra_ratio, "alt_pattern": alt, "alt_reason": reason,
            "rest_beats": rest, "mode": mode,
            "density_ratio": (n_slots / n_pool) if n_pool else float("nan")}


# ---------------------------------------------------------------------------
# 整曲
# ---------------------------------------------------------------------------


def pick_skeleton(ev: np.ndarray, stem_onsets: dict, span: float,
                  tol: float = stemhit.DEFAULT_TOL_SEC,
                  min_events: int = 3, min_onsets: int = 3,
                  default: str = "drums", margin: float = 1.25) -> str:
    """按 `lift` 最高选骨架轨（与标定报告 §5 的归因口径一致）。

    两道稳态闸门（**没有这两道，逐小节 argmax 会一节一换、纯噪声**）：

    1. 证据不足（官方槽 < ``min_events``，或该轨 onset < ``min_onsets``）→ 回落 ``default``；
    2. 只有当最高 lift **超过 ``default`` 的 lift ``margin`` 倍**才换轨，否则留在
       ``default``（``default`` 一般是该段的段落骨架）。
    """
    if ev.size < min_events:
        return default
    lifts: dict[str, float] = {}
    for k, v in stem_onsets.items():
        on = np.atleast_1d(np.asarray(v, dtype=float))
        if on.size < min_onsets:
            continue
        lf = stemhit.hit_stat(ev, on, tol, span_sec=span).lift
        if np.isfinite(lf):
            lifts[k] = float(lf)
    if not lifts:
        return default
    best = max(lifts, key=lambda k: lifts[k])
    base = lifts.get(default, 0.0)
    if best != default and lifts[best] < margin * max(base, 1e-9):
        return default
    return best


def song_sampling(bars: Sequence[tuple[int, int, float, float, float]],
                  slot_times: Sequence[float], notes_per_bar: dict,
                  stem_onsets: dict, skeleton_mode: str = "bar",
                  segment_skeleton: dict | None = None,
                  chart_div: dict | None = None, audio_div: dict | None = None,
                  tol: float = stemhit.DEFAULT_TOL_SEC,
                  th: dict | None = None) -> list[BarSampling]:
    """逐小节跑采音度量。

    参数
    ----
    bars
        ``[(音频小节号, 谱面小节号, t0, t1, 每拍秒数), ...]``
    slot_times
        全曲官方时间槽（已做 φ 补偿、同刻已去重），绝对秒。
    notes_per_bar
        ``{谱面小节号: note 数}``。
    stem_onsets
        ``{轨名: np.ndarray(秒)}``。
    skeleton_mode
        ``bar`` = 逐小节按 lift 取（证据不足回落段落/drums）；
        ``segment`` = 用 ``segment_skeleton`` 给的段落骨架；
        其余值 = 固定用该轨（如 ``drums`` / ``vocals``）。
    """
    th = THRESHOLDS if th is None else th
    ev_all = np.atleast_1d(np.asarray(slot_times, dtype=float))
    allon = np.sort(np.concatenate(
        [np.atleast_1d(np.asarray(stem_onsets.get(s, np.zeros(0)), dtype=float))
         for s in POOL_STEMS])) if stem_onsets else np.zeros(0)
    out: list[BarSampling] = []
    for bar, measure, t0, t1, spb in bars:
        ev = ev_all[(ev_all >= t0) & (ev_all < t1)]
        seg_sk = (segment_skeleton or {}).get(bar, "drums")
        if skeleton_mode == "bar":
            per = {k: np.asarray(v, dtype=float) for k, v in stem_onsets.items()
                   if k in SKELETON_STEMS}
            per = {k: v[(v >= t0) & (v < t1)] for k, v in per.items()}
            sk = pick_skeleton(ev, per, max(t1 - t0, 1e-6), tol, default=seg_sk)
        elif skeleton_mode == "segment":
            sk = seg_sk
        else:
            sk = skeleton_mode
        raw = np.atleast_1d(np.asarray(stem_onsets.get(sk, np.zeros(0)), dtype=float))
        pool = stemhit.unique_times(raw[(raw >= t0) & (raw < t1)],
                                    merge_sec=th["merge_sec"])
        m = bar_metrics(ev, notes_per_bar.get(measure, 0), pool,
                        allon[(allon >= t0 - 0.2) & (allon < t1 + 0.2)],
                        t0, t1, spb,
                        (chart_div or {}).get(measure, 0.0),
                        (audio_div or {}).get(bar, 0.0), tol, th)
        bs = BarSampling(bar=bar, chart_measure=measure, t0=t0, t1=t1, skeleton=sk,
                         **{k: m[k] for k in ("n_pool", "n_slots", "n_notes", "hit",
                                              "extra", "coverage", "extra_ratio",
                                              "alt_pattern", "alt_reason",
                                              "rest_beats", "density_ratio",
                                              "mode")})
        out.append(bs)
    return out


def shuffled_control(bars: Sequence[tuple[int, int, float, float, float]],
                     slot_times: Sequence[float], notes_per_bar: dict,
                     stem_onsets: dict, seed: int = 0, **kw) -> list[BarSampling]:
    """对照：把**每条 stem 的 onset 整体循环平移**一个随机量后重跑。

    平移量取 ``[0.1·曲长, 0.9·曲长]`` 上的均匀分布（各轨独立），落在曲长上循环回绕
    （按比例取而不是固定秒数，短曲/长曲都不会退化成常数平移）。
    这样 onset 的**密度与节奏纹理完全保留**，只有"和谱面的对齐关系"被打断——
    如果采音模式的分布在对照上也一样，说明模式只是密度的伪影。
    """
    rng = np.random.default_rng(seed)
    t_end = max(b[3] for b in bars) if bars else 0.0
    fake: dict = {}
    for k, v in stem_onsets.items():
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == 0 or t_end <= 0.0:
            fake[k] = arr
            continue
        shift = float(rng.uniform(0.1 * t_end, 0.9 * t_end))
        fake[k] = np.sort(((arr + shift) % t_end))
    return song_sampling(bars, slot_times, notes_per_bar, fake, **kw)


# ---------------------------------------------------------------------------
# 装载（轻量路径：不解码混音、不重算强度）
# ---------------------------------------------------------------------------


SHIFT_SCAN = np.arange(-0.045, 0.04501, 0.0025)
APPLY_SHIFT_THRESHOLD = 0.010


def load_pair(song_dir: Path, onset_cache: Path | None = None,
              tol: float = stemhit.DEFAULT_TOL_SEC) -> dict:
    """装载一首曲子的"谱面 × 音频"配对，只取采音分析需要的东西。

    与 `loader.load_song` 的差别：**不解码 track.mp3、不重算强度曲线**
    （这两件事占了装载时间的九成以上，而采音分析用不到）。
    onset 从 ``onset_cache/<曲名>.npz`` 读；没有缓存就现场从 `stems/*.wav` 跑
    （参数与管线完全一致）。
    """
    import sys

    repo = Path(__file__).resolve().parents[2]
    for p in (str(repo), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from chart_analysis import configs as cfg_mod
    from chart_analysis.density import chart_density
    from chart_analysis.simai_parser import parse_chart

    from . import chartpair as cp
    from . import loader as loader_mod

    song_dir = Path(song_dir)
    meta = loader_mod.read_maidata_header(song_dir / "maidata.txt")
    analysis = json.loads((song_dir / "song_analysis.json").read_text(encoding="utf-8"))
    res = parse_chart(loader_mod.extract_inote(song_dir / "maidata.txt"),
                      name=song_dir.name)
    dens = chart_density(res, song_dir.name)

    # ---- onset ----
    onsets: dict[str, np.ndarray] = {}
    npz = (Path(onset_cache) / f"{song_dir.name}.npz") if onset_cache else None
    if npz is not None and npz.exists():
        with np.load(npz) as z:
            for k in z.files:
                if not k.endswith("__str"):
                    onsets[k] = np.asarray(z[k], dtype=float)
    else:
        from tools.audio_analysis import onsets as onsets_mod
        from tools.audio_analysis import stems as stems_mod
        for s in ("drums", "bass", "other", "vocals"):
            p = song_dir / "stems" / f"{s}.wav"
            if p.exists():
                y, _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
                onsets[s] = onsets_mod.detect_onsets(
                    y, s, sr=onsets_mod.ANALYSIS_SR, hop=onsets_mod.HOP).times
        p = song_dir / "stems_htdemucs_6s" / "piano.wav"
        if p.exists():
            from tools.audio_analysis import stems as stems_mod2
            y, _ = stems_mod2.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
            onsets["piano"] = onsets_mod.detect_onsets(
                y, "piano", sr=onsets_mod.ANALYSIS_SR, hop=onsets_mod.HOP).times

    # ---- 小节对齐（与 cli.analyze_song 同口径）----
    g = analysis["grid"]
    first = float(g["first"])
    stats = dens.measures
    measure_starts = [first + s.start_time for s in stats]
    bar_starts = _bar_starts(analysis)
    m2b = cp.map_measures_to_bars(measure_starts, bar_starts, tol_sec=0.05)

    note_times = np.array([first + n.time for n in res.notes], dtype=float)
    slot_times = stemhit.unique_times(note_times, merge_sec=1e-3)
    allon = np.sort(np.concatenate(
        [np.atleast_1d(onsets.get(s, np.zeros(0))) for s in POOL_STEMS])) \
        if onsets else np.zeros(0)
    shift = stemhit.best_shift(slot_times, allon, SHIFT_SCAN, tol=tol)
    phi = float(shift["best_shift_sec"])
    applied = phi if abs(phi) > APPLY_SHIFT_THRESHOLD else 0.0

    bars: list[tuple[int, int, float, float, float]] = []
    for i in sorted(m2b):
        bar = m2b[i]
        bpm = float(stats[i].bpm) or float(g["bpm"])
        spb = 60.0 / bpm
        t0 = measure_starts[i] + applied
        bars.append((bar, stats[i].measure, t0, t0 + 4.0 * spb, spb))

    notes_per_bar = {s.measure: s.notes for s in stats}
    chart_div = {s.measure: float(s.finest_divisor) for s in stats}
    audio_div = {int(b["bar"]): float(b.get("division", 0) or 0)
                 for b in analysis.get("bars", [])}
    seg_of_bar, seg_rows = _segments_by_bar(analysis)

    # 段落骨架：段内按 lift 最高的轨（证据比逐小节稳）
    seg_sk: dict[int, str] = {}
    for row in seg_rows:
        a, b = row["start_bar"], row["end_bar"]
        t_a = next((x[2] for x in bars if x[0] >= a), None)
        t_b = next((x[3] for x in reversed(bars) if x[0] <= b), None)
        if t_a is None or t_b is None or t_b <= t_a:
            continue
        ev = slot_times + applied
        ev = ev[(ev >= t_a) & (ev < t_b)]
        per = {k: v[(v >= t_a) & (v < t_b)] for k, v in onsets.items()
               if k in SKELETON_STEMS}
        sk = pick_skeleton(ev, per, t_b - t_a, tol, default="drums")
        for bb in range(a, b + 1):
            seg_sk[bb] = sk

    return {
        "name": song_dir.name,
        "level": float(meta.get("lv_5", 0.0) or 0.0),
        "genre": str(meta.get("genre", "") or ""),
        "bpm": float(g["bpm"]), "first": first,
        "analysis": analysis, "parse": res, "density": dens,
        "onsets": onsets, "bars": bars,
        "slot_times": slot_times + applied,
        "notes_per_bar": notes_per_bar,
        "chart_div": chart_div, "audio_div": audio_div,
        "seg_of_bar": seg_of_bar, "segments": seg_rows,
        "segment_skeleton": seg_sk,
        "phi_ms": round(phi * 1000.0, 2), "phi_applied_ms": round(applied * 1000.0, 2),
        "configs": cfg_mod,
    }


def _bar_starts(analysis: dict) -> list[float]:
    """音频小节起始秒（1 起）。优先用 `bars[].start_sec`，缺了就按网格推。"""
    rows = analysis.get("bars", [])
    if rows and "start_sec" in rows[0]:
        return [float(r["start_sec"]) for r in rows]
    g = analysis["grid"]
    spb = 60.0 / float(g["bpm"])
    return [float(g["first"]) + i * 4.0 * spb for i in range(int(g["n_bars"]))]


def _segments_by_bar(analysis: dict) -> tuple[dict[int, dict], list[dict]]:
    rows = analysis.get("structure", {}).get("segments", [])
    by_bar: dict[int, dict] = {}
    for r in rows:
        for b in range(int(r["start_bar"]), int(r["end_bar"]) + 1):
            by_bar[b] = r
    return by_bar, list(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m tools.calibration.sampling",
        description="采音模式度量（全采/半采/空音/加花），逐小节")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("song", help="单曲逐小节表")
    s1.add_argument("song_dir")
    s1.add_argument("--onset-cache", default=None)
    s1.add_argument("--skeleton", default="bar",
                    help="bar / segment / drums / vocals / …")
    s1.add_argument("--json", default=None, help="逐小节 JSON 输出路径")

    s2 = sub.add_parser("corpus", help="全库汇总（模式分布 + 对照）")
    s2.add_argument("--calib-dir", default="out/calib")
    s2.add_argument("--onset-cache", default=None)
    s2.add_argument("--skeleton", default="bar")
    s2.add_argument("--json", required=True)
    s2.add_argument("--limit", type=int, default=0)

    a = p.parse_args(argv)
    if a.cmd == "song":
        d = load_pair(Path(a.song_dir),
                      Path(a.onset_cache) if a.onset_cache else None)
        rows = song_sampling(d["bars"], d["slot_times"], d["notes_per_bar"],
                             d["onsets"], skeleton_mode=a.skeleton,
                             segment_skeleton=d["segment_skeleton"],
                             chart_div=d["chart_div"], audio_div=d["audio_div"])
        print(f"# {d['name']}  φ*={d['phi_ms']:+.1f}ms  小节 {len(rows)}")
        print("bar  seg              mode      cov   extra  pool slots 骨架")
        for r in rows:
            seg = d["seg_of_bar"].get(r.bar, {}).get("label_ja", "")
            cov = "  n/a" if not np.isfinite(r.coverage) else f"{r.coverage:5.2f}"
            print(f"{r.bar:>3}  {seg:<14} {r.mode:<8} {cov} "
                  f"{r.extra_ratio:5.2f} {r.n_pool:>4} {r.n_slots:>5} {r.skeleton}")
        if a.json:
            Path(a.json).write_text(json.dumps([r.to_dict() for r in rows],
                                               ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        return 0

    from . import loader as loader_mod
    dirs = loader_mod.discover(Path(a.calib_dir))
    if a.limit:
        dirs = dirs[:a.limit]
    out = []
    for i, sd in enumerate(dirs, 1):
        try:
            d = load_pair(sd, Path(a.onset_cache) if a.onset_cache else None)
            rows = song_sampling(d["bars"], d["slot_times"], d["notes_per_bar"],
                                 d["onsets"], skeleton_mode=a.skeleton,
                                 segment_skeleton=d["segment_skeleton"],
                                 chart_div=d["chart_div"], audio_div=d["audio_div"])
            out.append({"name": d["name"], "genre": d["genre"], "level": d["level"],
                        "bpm": d["bpm"], "bars": [r.to_dict() for r in rows]})
        except Exception as exc:                     # noqa: BLE001
            out.append({"name": sd.name, "error": repr(exc)})
        if i % 20 == 0:
            print(f"  {i}/{len(dirs)}", flush=True)
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {a.json}（{len(out)} 曲）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
