#!/usr/bin/env python3
"""采音层：逐段对照**全踩 / 舍音 / 留白**（知识 068）与骨架轨。

⚠️ **口径与 `tools/calibration/sampling.py` 不同，必须写明**：标定那边用的是
`stems/*.wav` 的**原始 onset 秒数**；检查器不要求有音频，只吃
`song_analysis.json` 里**已经落格的候选池网格**（`bars[].patterns`，四轨字符串
`X`/`x`/`-`/`.`）。所以这里的 `coverage` 是「谱面踩到了这一小节骨架轨**网格候选**
的几成」，比标定口径粗；**两套数字不可混用**。

三个词的切点直接复用 `calibration.sampling.THRESHOLDS`（工具内部诊断口径，
不是术语定义，见知识 068/081）。

本层只做两件事，都**交给人看**、不判对错：

1. 「**该有音却整段留白**」——骨架轨整段有候选、而谱面这一段几乎没踩；
2. 「**采空音密集**」——谱面这一小节的音有一大批任何一条轨都对不上
   （社区词：采空音 / 插空音，知识 081）。⚠️ 它**不等于写坏了**：成段的高 extra
   恰恰是「似踩非踩」这种正规写法（知识 086）。
"""

from __future__ import annotations

import numpy as np

from .model import LayerResult

TRACK_KEYS = ("drum", "vocal", "bass", "hook", "melody", "piano", "fx", "guitar")
#: `song_analysis.json` 的轨名 → 标定侧的 stem 名（知识 032 的口径）
TRACK_TO_STEM = {"drum": "drums", "vocal": "vocals", "bass": "bass",
                 "hook": "other", "melody": "other", "piano": "piano"}
STEM_TO_TRACK = {v: k for k, v in TRACK_TO_STEM.items()}


def _grid_times(bar: dict, track: str, first: float = 0.0) -> list[float]:
    """一小节里某条轨的候选池时刻（秒）。`-` 是延音，不算新 onset。

    `first` = `&first`。`song_analysis.json` 的 `start_sec` 是**音频绝对秒**，
    而 `simai_parser` 的 note 时间从**谱面正文起点**（= `&first`）起算——
    两套时钟差一个 offset，必须先平掉（note 065 缺口 G16）。
    """
    pat = (bar.get("patterns") or {}).get(track, "")
    if not pat:
        return []
    t0 = float(bar.get("start_sec", 0.0)) - first
    bpm = float(bar.get("bpm", 0.0)) or 1.0
    bar_sec = 240.0 / bpm
    n = len(pat)
    return [t0 + bar_sec * i / n for i, ch in enumerate(pat) if ch in ("x", "X")]


#: 谱面小节线与 `song_analysis.json` 小节线允许的最大偏差（秒）。
#: 超过这个数就说明两边**不是同一张网格**（典型情形：谱面改成了变速，而 analysis
#: 还是旧的恒定 BPM 版），这时逐小节对照没有意义——**跳过，不报假发现**。
GRID_DRIFT_TOL = 0.050


def _grid_drift(res, analysis: dict, first: float) -> tuple[float, int]:
    """``(最大偏差秒, 最差的小节号)``。谱面小节线 vs analysis 的 `bars[].start_sec`。"""
    starts = getattr(res, "measure_starts", None) or {}
    worst, worst_bar = 0.0, -1
    for bar in analysis.get("bars", []):
        m = int(bar["bar"]) - 1                 # analysis 的 bar 是 1 起
        if m not in starts:
            continue
        d = abs((float(bar.get("start_sec", 0.0)) - first) - starts[m])
        if d > worst:
            worst, worst_bar = d, int(bar["bar"])
    return worst, worst_bar


def check_sampling(res, analysis: dict | None, measure_texts,
                   tol: float = 0.030, first: float = 0.0) -> LayerResult:
    """`first` = `&first`（秒）。见 `_grid_times` 的说明：两套时钟必须先对齐。"""
    out = LayerResult(layer="采音")
    if not analysis:
        out.skipped = "没有给 --analysis song_analysis.json → 采音层不跑"
        return out
    bars = analysis.get("bars") or []
    segs = (analysis.get("structure") or {}).get("segments") or []
    if not bars:
        out.skipped = "song_analysis.json 里没有 bars"
        return out

    drift, bad_bar = _grid_drift(res, analysis, first)
    if drift > GRID_DRIFT_TOL:
        g = analysis.get("grid") or {}
        out.skipped = (
            f"**网格对不上，本层跳过**：谱面小节线与 `song_analysis.json` 的 "
            f"`bars[].start_sec` 最大差 **{drift * 1000:.0f} ms**（最差在第 {bad_bar} 小节，"
            f"容差 {GRID_DRIFT_TOL * 1000:.0f} ms）。"
            f"analysis 的网格是 `first={g.get('first')}` / `bpm={g.get('bpm')}` / "
            f"`bpm_changes={len(g.get('bpm_changes') or [])} 条`——"
            "谱面改成变速之后这份 analysis 就过期了，**逐小节对照会报出一堆假的留白/采空音**。"
            "要么重跑音频分析生成新网格的 analysis，要么别给 `--analysis`。")
        out.stats = {"grid_drift_sec": round(drift, 4), "worst_bar": bad_bar,
                     "first_sec": first}
        return out

    try:
        from calibration.sampling import THRESHOLDS, UNKNOWN, classify
    except Exception as exc:  # pragma: no cover
        out.skipped = f"采音三词口径没加载上（{exc}）"
        return out

    # 谱面时间槽（同刻算一个）
    slots = sorted({round(n.time, 4) for n in res.notes})
    slot_arr = np.asarray(slots, dtype=float)

    seg_of_bar: dict[int, dict] = {}
    for s in segs:
        for b in range(int(s["start_bar"]), int(s["end_bar"]) + 1):
            seg_of_bar[b] = s

    rows = []
    for bar in bars:
        b = int(bar["bar"])
        seg = seg_of_bar.get(b, {})
        skel_track = seg.get("skeleton_stem") or "drum"
        pool = _grid_times(bar, skel_track, first)
        # 所有轨合起来（判 extra 用）
        allpool = sorted({t for tr in TRACK_KEYS for t in _grid_times(bar, tr, first)})
        t0 = float(bar.get("start_sec", 0.0)) - first
        bar_sec = 240.0 / (float(bar.get("bpm", 0.0)) or 1.0)
        ev = slot_arr[(slot_arr >= t0) & (slot_arr < t0 + bar_sec)]
        hit = sum(1 for p in pool if np.any(np.abs(ev - p) <= tol)) if len(ev) else 0
        extra = sum(1 for e in ev
                    if not any(abs(e - p) <= tol for p in allpool))
        cov = hit / len(pool) if pool else float("nan")
        exr = extra / len(ev) if len(ev) else float("nan")
        mode = classify(cov, exr, len(ev), len(pool), False, THRESHOLDS)
        rows.append({"bar": b, "skeleton": skel_track, "n_pool": len(pool),
                     "n_slots": int(len(ev)), "hit": hit, "extra": extra,
                     "coverage": None if not np.isfinite(cov) else round(cov, 4),
                     "extra_ratio": None if not np.isfinite(exr) else round(exr, 4),
                     "mode": mode,
                     "section": seg.get("label_ja") or seg.get("function") or "—"})

    # ---- 逐段汇总 ----
    seg_rows = []
    for s in segs:
        a, z = int(s["start_bar"]), int(s["end_bar"])
        rs = [r for r in rows if a <= r["bar"] <= z]
        if not rs:
            continue
        counts: dict[str, int] = {}
        for r in rs:
            counts[r["mode"]] = counts.get(r["mode"], 0) + 1
        pool = sum(r["n_pool"] for r in rs)
        hit = sum(r["hit"] for r in rs)
        ev = sum(r["n_slots"] for r in rs)
        extra = sum(r["extra"] for r in rs)
        cov = hit / pool if pool else float("nan")
        exr = extra / ev if ev else float("nan")
        label = s.get("label_ja") or s.get("function") or "—"
        seg_rows.append({"bars": f"{a}–{z}", "section": label,
                         "skeleton": s.get("skeleton_stem") or "—",
                         "accents": s.get("accent_stems") or [],
                         "coverage": None if not np.isfinite(cov) else round(cov, 4),
                         "extra_ratio": None if not np.isfinite(exr) else round(exr, 4),
                         "modes": counts, "n_pool": pool, "n_slots": ev})
        top = max(counts, key=lambda k: counts[k]) if counts else UNKNOWN
        out.add("提示", "SMP-SECTION",
                f"{a}–{z} {label}（骨架 {s.get('skeleton_stem') or '—'}）："
                f"coverage {('%.2f' % cov) if np.isfinite(cov) else '—'}、"
                f"采空音比 {('%.2f' % exr) if np.isfinite(exr) else '—'}、"
                f"逐小节以**{top}**为主（"
                + "、".join(f"{k}×{v}" for k, v in sorted(counts.items(),
                                                          key=lambda kv: -kv[1]))
                + "）",
                source="知识 068 / 081")
        # 该有音却整段留白
        if pool >= 8 and np.isfinite(cov) and cov < THRESHOLDS["empty_hi"]:
            out.add("警告", "SMP-EMPTY-SECTION",
                    f"{a}–{z} {label}：骨架轨整段给了 {pool} 个候选，"
                    f"谱面只踩到 {cov:.0%}——「该有音却整段留白」，请人听一遍"
                    "（知识 068：留白是表达手段，但整段留白要有理由）",
                    source="知识 068")
        if ev >= 8 and np.isfinite(exr) and exr >= THRESHOLDS["pseudo_extra_lo"]:
            out.add("提示", "SMP-EXTRA-DENSE",
                    f"{a}–{z} {label}：采空音比 {exr:.0%}（谱面里有一大批音任何一条轨"
                    "都对不上）。⚠️ **这不等于写坏了**——成段的高 extra 正是"
                    "「似踩非踩」这种正规写法（知识 086）；也可能只是 onset 漏检。",
                    source="知识 086 / 081")

    mode_all: dict[str, int] = {}
    for r in rows:
        mode_all[r["mode"]] = mode_all.get(r["mode"], 0) + 1
    out.add("提示", "SMP-OVERALL",
            "全曲逐小节采音方式："
            + "、".join(f"{k} {v}（{v / len(rows):.0%}）"
                        for k, v in sorted(mode_all.items(), key=lambda kv: -kv[1]))
            + "；官谱 160 首 13 535 小节的分布是 全踩 59.2% / 舍音 24.3% / "
              "留白 8.4% / 判不出 8.1%（知识 068，**原始 onset 口径**，"
              "与这里的网格口径不可直接比）",
            source="知识 068")

    out.stats = {"bars": rows, "sections": seg_rows, "mode_counts": mode_all,
                 "tol_sec": tol, "first_sec": first,
                 "口径": "网格候选池（song_analysis.json bars[].patterns），"
                         "非 stems 原始 onset；与 calibration/sampling.py 不可混用"}
    return out
