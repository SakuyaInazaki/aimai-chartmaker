#!/usr/bin/env python3
"""密度层：note 总数 / NPS / 逐小节密度曲线 / 末段形态，对照官谱实测。

对照对象（全部是既有实测条目，不新造阈值）：

- **知识 004**：各定数档 note 总数分布（p10–p90）。给了 `--level` 才对照；
- **知识 031 §1**：`T_density = [0.62, 0.77, 0.72, 0.78, 0.82]`（**结构对齐五段口径**，
  用 `density.changepoint_segments` 切；等分五段口径是另一套数，**不可混用**）；
- **知识 031 §2**：密度地板——归一化逐小节密度分位官谱是 p10 0.264 / p50 0.514 /
  p90 0.730；"任何一小节只要还在出音，就得给出基本踩音密度"；
- **知识 031 §7**：定数 → NPS 锚（**下游一律走 NPS 列再用 BPM 换算**，
  note/小节 锚在 BPM ≥ 200 上系统性高估）；
- **知识 001 / 031 §10**：末段形态（尾杀 65.1% / 渐弱 1.0% / 其他 33.9%，末 8 小节口径），
  判据实现直接调 `tools/calibration/ending.py`，**判据是本项目自定的**，引用要连判据一起引。
"""

from __future__ import annotations

import numpy as np

from ._deps import density_mod as D
from .model import LayerResult

#: 知识 031 §1（结构对齐五段口径）
T_DENSITY = (0.62, 0.77, 0.72, 0.78, 0.82)
#: 知识 031 §2：归一化逐小节密度分位（388 谱均值）
OFFICIAL_QUANTILES = {"p10": 0.264, "p50": 0.514, "p90": 0.730}
#: 知识 031 §7：定数档 → (note/小节 均值, NPS 均值, 峰值小节 note 均值)
LEVEL_ANCHOR = {13.0: (8.03, 5.29, 15.8), 13.5: (9.15, 6.13, 18.3),
                14.0: (10.54, 7.14, 23.2), 14.5: (10.10, 7.70, 24.7)}
#: 知识 004：ST/SD 谱 note 总数 (均值, p10, p90)
LEVEL_TOTAL = {12.0: (492.2, 302, 671), 13.0: (643.7, 445, 828),
               13.5: (767.4, 492, 967), 14.0: (899.8, 685, 1080),
               14.5: (1020.7, 733, 1181)}
#: 知识 031 §3：曲末 10% 密度 = 1.084 × 全曲均值；峰值小节 / 全曲均值 ≈ 2.06
OFFICIAL_PEAK_RATIO = 2.06


def _bucket(level: float | None) -> float | None:
    if level is None:
        return None
    import math
    return math.floor(level * 2.0) / 2.0


def check_density(res, level: float | None, measure_texts) -> LayerResult:
    out = LayerResult(layer="密度")
    dens = D.chart_density(res, "chart")
    curve = np.asarray(dens.raw, dtype=float)
    if curve.size == 0:
        out.skipped = "谱面没有 note"
        return out

    tot = {k: sum(getattr(s, k) for s in dens.measures)
           for k in ("notes", "taps", "holds", "slides", "breaks", "touches")}
    total = tot["notes"]
    nps = total / res.total_seconds if res.total_seconds > 0 else 0.0
    mean_per_bar = float(np.mean(curve))
    peak = float(np.max(curve))
    bucket = _bucket(level)

    out.add("提示", "DEN-TOTAL",
            f"note 总数 {total}（tap {tot['taps']} / hold {tot['holds']} / "
            f"slide {tot['slides']} / break {tot['breaks']} / touch {tot['touches']}）；"
            f"时长 {res.total_seconds:.1f} s；NPS {nps:.2f}；"
            f"小节 {len(curve)}，平均 {mean_per_bar:.2f} note/小节，峰值 {peak:.0f}",
            source="知识 004 / 031 §7")

    # ---- 对照知识 004 的 note 总数区间 ----
    if bucket in LEVEL_TOTAL:
        mean_t, p10, p90 = LEVEL_TOTAL[bucket]
        if total < p10:
            out.add("警告", "DEN-TOTAL-LOW",
                    f"note 总数 {total} 低于定数 {bucket} 档的 p10（{p10}，均值 {mean_t}）"
                    "——知识 004",
                    source="知识 004")
        elif total > p90:
            out.add("警告", "DEN-TOTAL-HIGH",
                    f"note 总数 {total} 高于定数 {bucket} 档的 p90（{p90}，均值 {mean_t}）"
                    "——知识 004",
                    source="知识 004")
        else:
            out.add("提示", "DEN-TOTAL-OK",
                    f"note 总数 {total} 落在定数 {bucket} 档的 p10–p90（{p10}–{p90}）内",
                    source="知识 004")
    if bucket in LEVEL_ANCHOR:
        npb_a, nps_a, peak_a = LEVEL_ANCHOR[bucket]
        out.add("提示", "DEN-ANCHOR",
                f"定数 {bucket} 档官谱均值：NPS {nps_a}（本谱 {nps:.2f}）、"
                f"note/小节 {npb_a}（本谱 {mean_per_bar:.2f}）、"
                f"峰值小节 {peak_a}（本谱 {peak:.0f}）"
                "；⚠️ 知识 031 §7 的口径是**先走 NPS 再用 BPM 换算**",
                source="知识 031 §7")

    # ---- 五段形状对照 T_density（结构对齐口径） ----
    try:
        prof_len = D.structure_profile(curve, k=5, min_len_frac=0.10)
        if prof_len is None:
            raise ValueError("小节太少，切不出 5 段")
        prof = np.asarray(prof_len[0], dtype=float)
        diff = prof - np.asarray(T_DENSITY)
        out.add("提示", "DEN-SHAPE",
                "结构对齐五段（变点检测，min_len_frac=0.10）归一形状 "
                + "[" + ", ".join(f"{v:.2f}" for v in prof) + "]；"
                + "官谱 T_density [" + ", ".join(f"{v:.2f}" for v in T_DENSITY) + "]；"
                + "逐段差 [" + ", ".join(f"{v:+.2f}" for v in diff) + "]"
                + "。⚠️ 两种五段口径（结构对齐 / 等分）不可混用。",
                source="知识 031 §1")
        if prof[4] < prof.max() - 0.15:
            out.add("提示", "DEN-TAIL-SOFT",
                    f"末段（s5={prof[4]:.2f}）明显低于全曲最强段——官谱里 s5 是五段中"
                    "最强的一段（0.82）。知识 001：尾杀为常态、渐弱淡出也存在，"
                    "**按曲子实际结尾判断**。",
                    source="知识 001 / 031 §1")
    except Exception as exc:  # pragma: no cover - 变点检测对超短谱会失败
        out.add("提示", "DEN-SHAPE-SKIP", f"五段形状对照没跑（{exc}）",
                source="知识 031 §1")
        prof = None

    # ---- 密度地板 ----
    norm = curve / (peak or 1.0)
    q10, q50, q90 = (float(np.percentile(norm, p)) for p in (10, 50, 90))
    out.add("提示", "DEN-FLOOR",
            f"归一化逐小节密度分位 p10 {q10:.3f} / p50 {q50:.3f} / p90 {q90:.3f}；"
            f"官谱 p10 {OFFICIAL_QUANTILES['p10']} / p50 {OFFICIAL_QUANTILES['p50']} / "
            f"p90 {OFFICIAL_QUANTILES['p90']}（知识 031 §2）",
            source="知识 031 §2")
    empty = [i for i, v in enumerate(curve) if v <= 1]
    if empty:
        out.add("提示", "DEN-REST",
                f"休息小节（note ≤ 1）{len(empty)} 个（{100 * len(empty) / len(curve):.1f}%）："
                + "、".join(f"m{i:03d}" for i in empty[:12])
                + ("…" if len(empty) > 12 else "")
                + "；官谱只占 2.43%，35% 的谱一个都没有，休息靠**低密段**不靠空小节"
                "（知识 031 §5）",
                source="知识 031 §5")
    if peak > 0 and peak / (mean_per_bar or 1.0) > OFFICIAL_PEAK_RATIO * 1.5:
        out.add("提示", "DEN-PEAK-SPIKE",
                f"峰值小节 / 全曲均值 = {peak / mean_per_bar:.2f}，官谱这个比值约 "
                f"{OFFICIAL_PEAK_RATIO}（知识 031 §7）",
                source="知识 031 §7")

    # ---- 末段形态 ----
    ending = None
    try:
        from calibration.ending import classify_ending
        ending = classify_ending(curve)
        label_cn = {"tail_kill": "尾杀", "fade_out": "渐弱", "other": "其他"}[ending.label]
        out.add("提示", "DEN-ENDING",
                f"末 {ending.tail_bars} 小节形态：**{label_cn}**"
                f"（tail_ratio {ending.tail_ratio:.3f}、trend {ending.trend:+.2f}）；"
                "官谱 387 谱 尾杀 65.1% / 渐弱 1.0% / 其他 33.9%（末 8 小节口径）。"
                "⚠️ 判据是本项目自定的三个阈值（1.00 / 0.70 / −0.50），引用要连判据一起引。",
                source="知识 001 / 031 §10")
    except Exception as exc:  # pragma: no cover
        out.add("提示", "DEN-ENDING-SKIP", f"末段形态没跑（{exc}）", source="知识 031 §10")

    # ---- 后半 / 前半 ----
    half = len(curve) // 2
    if half >= 4:
        a, b = float(np.mean(curve[:half])), float(np.mean(curve[half:]))
        if a > 0:
            out.add("提示", "DEN-HALVES",
                    f"后半 / 前半 平均密度比 {b / a:.3f}；官谱中位 1.119（+12%），"
                    "79.6% 的谱后半强于前半（知识 031 §4）。"
                    "⚠️ 知识 031 §8：**强度提升 ≠ 密度提升**，更大的强度差由配置复杂度承担。",
                    source="知识 031 §4/§8")

    out.stats = {
        "total_notes": total, "taps": tot["taps"], "holds": tot["holds"],
        "slides": tot["slides"], "breaks": tot["breaks"], "touches": tot["touches"],
        "each_groups": res.each_groups,
        "seconds": round(res.total_seconds, 3), "nps": round(nps, 3),
        "n_measures": len(curve), "mean_per_bar": round(mean_per_bar, 3),
        "peak_per_bar": peak,
        "curve": [float(v) for v in curve],
        "norm_quantiles": {"p10": round(q10, 4), "p50": round(q50, 4),
                           "p90": round(q90, 4)},
        "shape5": [round(float(v), 4) for v in prof] if prof is not None else None,
        "T_density": list(T_DENSITY),
        "ending": ending.to_dict() if ending is not None else None,
        "level": level, "level_bucket": bucket,
    }
    return out
