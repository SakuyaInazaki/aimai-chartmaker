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
  判据实现直接调 `tools/calibration/ending.py`，**判据是本项目自定的**，引用要连判据一起引；
- **知识 088**：**note 种类配比**——谱级按定数档、段落级按日式标签，对照官谱分布
  （`e2e-trial-01` 的缺口 G1：note 总数与 NPS 都在区间内，却差着三倍的 slide 与十倍的 hold，
  而五层里没有任何一条报得出来）。⚠️ 088 是 agent 观察条目，**只作对照，不判对错**；
  单曲整段偏离是常态（知识 **092**）。
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

#: 知识 088：388 官谱**谱级** note 种类占比，定数档 → {种类: (中位, Q1, Q3)}。
#: 分母都是 note 总数；`slide` = 滑轨条数、`each` = 参与同刻多押的 note 数。
#: **只作对照，不作判据**（agent 观察，未经用户确认）。
LEVEL_NOTEMIX = {
    # 口径 = `_bucket()` 的 floor(定数×2)/2 档，与本文件其余表一致（n = 168/135/73/12）
    13.0: {"n": 168, "slide": (0.146, 0.111, 0.176), "hold": (0.058, 0.032, 0.082),
           "each": (0.590, 0.486, 0.657), "break": (0.030, 0.020, 0.050)},
    13.5: {"n": 135, "slide": (0.123, 0.092, 0.166), "hold": (0.045, 0.025, 0.067),
           "each": (0.544, 0.454, 0.649), "break": (0.034, 0.021, 0.055)},
    14.0: {"n": 73, "slide": (0.103, 0.083, 0.153), "hold": (0.044, 0.026, 0.072),
           "each": (0.486, 0.418, 0.605), "break": (0.032, 0.018, 0.067)},
    14.5: {"n": 12, "slide": (0.107, 0.081, 0.144), "hold": (0.055, 0.039, 0.067),
           "each": (0.458, 0.373, 0.554), "break": (0.033, 0.012, 0.057)},
}

#: 知识 088：160 首**段落级**分布（日式标签 → 中位 [Q1, Q3]）。
#: `slide` / `hold` 是占 note 比，`slide_per_bar` 是每小节滑轨条数，
#: `zero_slide` / `zero_hold` 是"整段一根都没有"的段落占比。
SECTION_NOTEMIX = {
    "イントロ": {"n": 155, "slide": (0.075, 0.009, 0.171), "hold": (0.038, 0.000, 0.110),
              "slide_per_bar": (0.50, 0.07, 1.04), "npb": (6.82, 4.62, 8.52),
              "zero_slide": 0.25, "zero_hold": 0.39},
    "Aメロ": {"n": 355, "slide": (0.122, 0.058, 0.227), "hold": (0.035, 0.000, 0.109),
            "slide_per_bar": (1.00, 0.50, 1.63), "npb": (7.88, 6.25, 9.89),
            "zero_slide": 0.09, "zero_hold": 0.36},
    "Bメロ": {"n": 58, "slide": (0.144, 0.047, 0.209), "hold": (0.025, 0.000, 0.099),
            "slide_per_bar": (1.06, 0.50, 1.50), "npb": (7.50, 5.31, 10.59),
            "zero_slide": 0.17, "zero_hold": 0.47},
    "サビ": {"n": 215, "slide": (0.167, 0.092, 0.233), "hold": (0.017, 0.000, 0.054),
           "slide_per_bar": (1.33, 0.87, 2.00), "npb": (8.88, 7.25, 10.42),
           "zero_slide": 0.07, "zero_hold": 0.42},
    "落ちサビ": {"n": 183, "slide": (0.154, 0.095, 0.251), "hold": (0.015, 0.000, 0.054),
             "slide_per_bar": (1.27, 0.75, 2.00), "npb": (8.10, 6.75, 9.94),
             "zero_slide": 0.08, "zero_hold": 0.43},
    "ラスサビ": {"n": 46, "slide": (0.165, 0.076, 0.218), "hold": (0.022, 0.001, 0.062),
             "slide_per_bar": (1.44, 0.79, 2.05), "npb": (9.38, 7.84, 11.00),
             "zero_slide": 0.09, "zero_hold": 0.26},
    "間奏": {"n": 362, "slide": (0.101, 0.036, 0.189), "hold": (0.021, 0.000, 0.077),
           "slide_per_bar": (0.88, 0.26, 1.50), "npb": (8.75, 6.75, 10.81),
           "zero_slide": 0.18, "zero_hold": 0.42},
    "ドロップ": {"n": 81, "slide": (0.102, 0.031, 0.190), "hold": (0.013, 0.000, 0.069),
             "slide_per_bar": (1.00, 0.38, 2.00), "npb": (10.75, 9.06, 12.75),
             "zero_slide": 0.19, "zero_hold": 0.42},
    "アウトロ": {"n": 32, "slide": (0.000, 0.000, 0.199), "hold": (0.000, 0.000, 0.010),
             "slide_per_bar": (0.00, 0.00, 2.00), "npb": (6.00, 2.00, 11.00),
             "zero_slide": 0.53, "zero_hold": 0.75},
}


def _nearest_level_key(bucket: float | None) -> float | None:
    """`LEVEL_NOTEMIX` 只有四个键（13.0/13.5/14.0/14.5），把定数档落到最近的一个。"""
    if bucket is None:
        return None
    return min(LEVEL_NOTEMIX, key=lambda k: abs(k - float(bucket)))


def _q3(ref: dict, key: str) -> tuple[float, float, float]:
    return ref[key]


def _band(v: float, ref: tuple[float, float, float]) -> str:
    """一个值落在官谱四分位区间的哪一侧（只描述位置，不判对错）。"""
    med, q1, q3 = ref
    if v < q1:
        return "低于 Q1"
    if v > q3:
        return "高于 Q3"
    return "在 Q1–Q3 内"


def _bucket(level: float | None) -> float | None:
    if level is None:
        return None
    import math
    return math.floor(level * 2.0) / 2.0


def check_density(res, level: float | None, measure_texts,
                  analysis: dict | None = None) -> LayerResult:
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

    # ---- note 种类配比（知识 088）：谱级 ----
    each_notes = sum(1 for n in res.notes if n.is_each)
    mix = {"slide": tot["slides"] / total if total else 0.0,
           "hold": tot["holds"] / total if total else 0.0,
           "each": each_notes / total if total else 0.0,
           "break": tot["breaks"] / total if total else 0.0}
    key = _nearest_level_key(bucket)
    if key is not None:
        ref = LEVEL_NOTEMIX[key]
        bits = []
        for k, cn in (("slide", "slide 轨"), ("hold", "hold"),
                      ("each", "each note"), ("break", "break")):
            med, q1, q3 = ref[k]
            bits.append(f"{cn} {mix[k]:.1%}（官谱 {med:.1%}"
                        f" [{q1:.1%}–{q3:.1%}]，{_band(mix[k], ref[k])}）")
        out.add("提示", "DEN-NOTEMIX",
                f"note 种类配比（分母 = note 总数）：" + "；".join(bits)
                + f"。对照口径 = 388 官谱定数 {key} 档（n={ref['n']}，知识 088）。"
                "⚠️ 知识 070：**升定数靠密度与细分音，不靠堆多押与星星**——"
                "slide / each 占比随定数**下降**是官谱的常态，"
                "这条对照只是让你知道自己写在分布的哪一侧，**不判对错**。",
                source="知识 088 / 070")
        for k, cn in (("slide", "slide 轨"), ("hold", "hold")):
            med, q1, q3 = ref[k]
            if mix[k] < q1:
                out.add("提示", f"DEN-NOTEMIX-{k.upper()}-LOW",
                        f"{cn} 占比 {mix[k]:.1%} 低于定数 {key} 档官谱的 Q1（{q1:.1%}，"
                        f"中位 {med:.1%}）——知识 088。"
                        "**这不是错误**（官谱自己也有四分之一在 Q1 以下），"
                        "但它是 `e2e-trial-01` 写浅的主要形态，值得回头看一眼。",
                        source="知识 088 / 093")

    # ---- note 种类配比（知识 088）：段落级 ----
    segs = ((analysis or {}).get("structure") or {}).get("segments") or []
    sec_rows = []
    if segs:
        per_bar = {s.measure: s for s in dens.measures}
        for s in segs:
            a, z = int(s["start_bar"]), int(s["end_bar"])
            ms = [per_bar[m] for m in range(a - 1, z) if m in per_bar]
            if not ms:
                continue
            n_notes = sum(x.notes for x in ms)
            if n_notes == 0:
                continue
            label = s.get("label_ja") or s.get("function") or "—"
            row = {"bars": f"{a}–{z}", "section": label, "n_bars": len(ms),
                   "notes": n_notes,
                   "slide_ratio": round(sum(x.slides for x in ms) / n_notes, 4),
                   "hold_ratio": round(sum(x.holds for x in ms) / n_notes, 4),
                   "slide_per_bar": round(sum(x.slides for x in ms) / len(ms), 3),
                   "notes_per_bar": round(n_notes / len(ms), 3)}
            ref = SECTION_NOTEMIX.get(label)
            if ref:
                row["official_slide_ratio"] = ref["slide"]
                row["official_slide_per_bar"] = ref["slide_per_bar"]
                out.add("提示", "DEN-NOTEMIX-SECTION",
                        f"{a}–{z} {label}：slide 占比 {row['slide_ratio']:.1%}"
                        f"（官谱同段落 {ref['slide'][0]:.1%} [{ref['slide'][1]:.1%}–"
                        f"{ref['slide'][2]:.1%}]，{_band(row['slide_ratio'], ref['slide'])}）、"
                        f"每小节滑轨 {row['slide_per_bar']:.2f}（官谱 {ref['slide_per_bar'][0]:.2f}）、"
                        f"hold 占比 {row['hold_ratio']:.1%}（官谱 {ref['hold'][0]:.1%}；"
                        f"官谱有 {ref['zero_hold']:.0%} 的{label}一根 hold 都没有）、"
                        f"note/小节 {row['notes_per_bar']:.2f}（官谱 {ref['npb'][0]:.2f}）"
                        f"；n={ref['n']} 段。⚠️ 知识 092：**段落分布不是处方**，"
                        "单曲整段偏离是常态。",
                        source="知识 088 / 092")
            sec_rows.append(row)

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
        "note_mix": {k: round(v, 4) for k, v in mix.items()},
        "note_mix_official": LEVEL_NOTEMIX.get(key) if key is not None else None,
        "sections": sec_rows,
    }
    return out
