"""命令行入口：官方音频 × 官方谱配对标定（docs/audio-analysis.md §4.8(E)）。

    python -m tools.calibration --calib-dir out/calib \
        --csv docs/research/data/audio-chart-calibration-summary.csv \
        --plots /tmp/.../calib --metrics /tmp/.../calib/metrics.json

子步骤（全部可单独关掉）：

1. **对齐核对**：谱面小节起始秒 vs 音频网格；官方 note 时间 vs 各 stem onset 的
   全局相位扫描 φ*（用谱面反过来校验 mp3 对齐）；
2. **强度 × 密度**：逐小节 / 逐段 Spearman、Pearson；峰值位置误差；结构边界命中；
3. **权重标定**：五项 Z 特征 → 密度的 NNLS/岭回归 + 留一曲交叉验证；
4. **密度地板检验**：`suggested_notes` vs 官方实际 note/小节；反解 floor；
5. **切轨验证**：逐段各 stem 命中率 / 精确率；与 `stemplan.py` 的一致率；
6. **结尾形态**：388 官方谱语料统计 + 这 8 首的"音频淡出 vs 谱面不淡出"对照。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from . import chartpair as cp
from . import ending as ending_mod
from . import loader as loader_mod
from . import stemhit
from . import weights as weights_mod

STEMS = ("drums", "bass", "other", "vocals")
TRACK_OF_STEM = {"drums": "drum", "vocals": "vocal", "bass": "bass", "other": "hook"}
SHIFT_SCAN = np.arange(-0.045, 0.04501, 0.0025)
APPLY_SHIFT_THRESHOLD = 0.010     # |φ*| 超过 10 ms 才在后续分析里补偿


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_paths() -> None:
    repo = _repo_root()
    for p in (str(repo), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# 单曲分析
# ---------------------------------------------------------------------------


def analyze_song(bundle, tol: float = stemhit.DEFAULT_TOL_SEC) -> dict:
    _ensure_paths()
    from chart_analysis.density import changepoint_segments

    grid = bundle.grid
    dens = bundle.density
    stats = dens.measures
    if not stats:
        return {"name": bundle.name, "error": "谱面无 note"}

    # ---- 1. 小节对齐 ----
    measure_starts = [float(bundle.first) + s.start_time for s in stats]
    bar_starts = [grid.bar_start(b) for b in range(1, grid.n_bars + 1)]
    m2b = cp.map_measures_to_bars(measure_starts, bar_starts, tol_sec=0.05)
    align_err_ms = cp.measure_alignment_error(measure_starts, bar_starts) * 1000.0

    idx = sorted(m2b)                       # 谱面小节的列表下标（连续）
    bars = [m2b[i] for i in idx]            # 对应的音频小节号
    d_bar = np.array([stats[i].notes for i in idx], dtype=float)
    d_bar_w = cp.weighted_density(bundle.parse.notes, len(stats),
                                  first_measure=stats[0].measure)[idx]
    I_smooth = np.array([bundle.bar_intensity[b - 1] for b in bars], dtype=float)
    I_raw = np.array([bundle.bar_intensity_raw[b - 1] for b in bars], dtype=float)

    # ---- 2. 官方 note 时间（绝对秒）+ 相位扫描 ----
    note_times = np.array([float(bundle.first) + n.time for n in bundle.parse.notes],
                          dtype=float)
    slot_times = stemhit.unique_times(note_times, merge_sec=1e-3)
    all_onsets = np.sort(np.concatenate(
        [np.atleast_1d(bundle.onset_times.get(s, np.zeros(0))) for s in STEMS]))
    shift = stemhit.best_shift(slot_times, all_onsets, SHIFT_SCAN, tol=tol)
    phi = float(shift["best_shift_sec"])
    applied = phi if abs(phi) > APPLY_SHIFT_THRESHOLD else 0.0
    slot_adj = slot_times + applied
    note_adj = note_times + applied

    # ---- 3. 逐小节相关 ----
    corr = {
        "spearman_raw": cp.spearman(I_raw, d_bar),
        "spearman_smooth": cp.spearman(I_smooth, d_bar),
        "spearman_smooth_weighted": cp.spearman(I_smooth, d_bar_w),
        "pearson_raw": cp.pearson(I_raw, d_bar),
        "pearson_smooth": cp.pearson(I_smooth, d_bar),
        "n_bars": int(len(d_bar)),
    }
    # 单特征相关：哪一项 Z 特征自己就最能预测密度
    corr["per_feature_spearman"] = {
        k: round(cp.spearman(
            np.asarray(bundle.components[k], dtype=float)[[b - 1 for b in bars]], d_bar), 4)
        for k in weights_mod.FEATURE_ORDER if k in bundle.components}

    # ---- 4. 峰值位置误差 ----
    peak_audio = cp.smoothed_peak_index(I_smooth, window=4)
    peak_chart = cp.smoothed_peak_index(d_bar, window=4)
    peak_audio8 = cp.smoothed_peak_index(I_smooth, window=8)
    peak_chart8 = cp.smoothed_peak_index(d_bar, window=8)
    climax_bar = int(bundle.analysis["intensity"].get("climax_bar", 0))
    climax_idx = bars.index(climax_bar) if climax_bar in bars else None
    top20 = d_bar >= np.percentile(d_bar, 80)
    peak = {
        "peak_audio_idx": int(peak_audio), "peak_chart_idx": int(peak_chart),
        "peak_err_bars": int(abs(peak_audio - peak_chart)),
        "peak_err_bars_w8": int(abs(peak_audio8 - peak_chart8)),
        "climax_vote_idx": climax_idx,
        "climax_vote_err_bars": (abs(climax_idx - peak_chart)
                                 if climax_idx is not None else None),
        "peak_audio_pos": round(peak_audio / max(1, len(d_bar) - 1), 4),
        "peak_chart_pos": round(peak_chart / max(1, len(d_bar) - 1), 4),
        # 更宽容的口径：音频峰所在小节是否落在谱面密度前 20% 的小节里
        "audio_peak_in_chart_top20": bool(top20[min(peak_audio, len(d_bar) - 1)]),
        "climax_in_chart_top20": (bool(top20[climax_idx])
                                  if climax_idx is not None else None),
    }

    # ---- 5. 结构边界 vs 密度变点 ----
    audio_bounds = [s["start_bar"] for s in bundle.segments if s["start_bar"] > 1]
    audio_bounds_idx = [bars.index(b) for b in audio_bounds if b in bars]
    cps = changepoint_segments(d_bar, k=5, min_len_frac=0.10)
    chart_bounds_idx = [int(x) for x in (cps or [])[1:-1]]
    hits1, n_pred, n_true = cp.boundary_hits(audio_bounds_idx, chart_bounds_idx, tol=1)
    hits2, _, _ = cp.boundary_hits(audio_bounds_idx, chart_bounds_idx, tol=2)
    boundary = {"audio_bounds": audio_bounds_idx, "chart_bounds": chart_bounds_idx,
                "hits_tol1": hits1, "hits_tol2": hits2,
                "n_pred": n_pred, "n_true": n_true,
                "hit_rate_tol1": round(hits1 / n_true, 4) if n_true else None,
                "hit_rate_tol2": round(hits2 / n_true, 4) if n_true else None}

    # ---- 6. 密度地板检验 ----
    npb_target = float(bundle.analysis["target"].get("notes_per_bar", 9.0))
    suggested = np.array([bundle.analysis["bars"][b - 1]["suggested_notes"]
                          for b in bars], dtype=float)
    d_norm_actual = d_bar / npb_target
    A = np.column_stack([np.ones_like(I_smooth), I_smooth])
    coef, *_ = np.linalg.lstsq(A, d_norm_actual, rcond=None)
    floor_fit = float(coef[0])
    # 经验"地板"：把小节按 I 分 5 档，看最低档的实际归一密度还剩多少
    d_peaknorm = d_bar / max(d_bar.max(), 1.0)
    qs = np.quantile(I_smooth, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        sel = (I_smooth >= lo) & (I_smooth <= hi) if i == 4 \
            else (I_smooth >= lo) & (I_smooth < hi)
        bins.append(round(float(d_peaknorm[sel].mean()), 4) if sel.any() else None)
    floor_check = {
        "notes_per_bar_target": npb_target,
        "actual_notes_per_bar_mean": float(d_bar.mean()),
        "mae": float(np.mean(np.abs(suggested - d_bar))),
        "bias": float(np.mean(suggested - d_bar)),
        "mape": float(np.mean(np.abs(suggested - d_bar) / np.maximum(d_bar, 1.0))),
        "floor_fit_bar": round(floor_fit, 4),
        "slope_fit_bar": round(float(coef[1]), 4),
        "d_norm_p10": round(float(np.percentile(d_peaknorm, 10)), 4),
        "d_norm_p50": round(float(np.percentile(d_peaknorm, 50)), 4),
        "d_norm_p90": round(float(np.percentile(d_peaknorm, 90)), 4),
        "d_norm_by_intensity_quintile": bins,
    }

    # ---- 7. 切轨验证 ----
    stem_onsets = {s: np.atleast_1d(bundle.onset_times.get(s, np.zeros(0)))
                   for s in STEMS}
    song_span = float(measure_starts[idx[-1]] - measure_starts[idx[0]]) or 1.0
    global_break = stemhit.explain_breakdown(slot_adj, stem_onsets, tol=tol)
    global_stats = {s: stemhit.hit_stat(slot_adj, stem_onsets[s], tol,
                                        span_sec=song_span).to_dict()
                    for s in STEMS}

    def seg_rows(spans, kind: str, extra=None) -> list[dict]:
        rows = []
        for k, (a_idx, b_idx) in enumerate(spans):
            t0 = measure_starts[idx[a_idx]] + applied
            last = idx[b_idx]
            last_bpm = float(stats[last].bpm) or float(bundle.bpm)
            t1 = measure_starts[last] + 4.0 * 60.0 / last_bpm + applied
            sel = (slot_adj >= t0) & (slot_adj < t1)
            ev = slot_adj[sel]
            span = max(t1 - t0, 1e-6)
            per = {}
            for s in STEMS:
                on = stem_onsets[s]
                on_seg = on[(on >= t0) & (on < t1)]
                per[s] = stemhit.hit_stat(ev, on_seg, tol, span_sec=span).to_dict()
            ranked = sorted(((s, per[s]["recall"] or 0.0) for s in STEMS),
                            key=lambda kv: kv[1], reverse=True)
            ranked_lift = sorted(((s, per[s]["lift"] or 0.0) for s in STEMS),
                                 key=lambda kv: kv[1], reverse=True)
            row = {
                "kind": kind, "seg_index": k,
                "start_idx": a_idx, "end_idx": b_idx,
                "start_bar": bars[a_idx], "end_bar": bars[b_idx],
                "t0": round(t0, 3), "t1": round(t1, 3),
                "n_events": int(ev.size),
                "d_bar_mean": round(float(d_bar[a_idx:b_idx + 1].mean()), 3),
                "I_mean": round(float(I_smooth[a_idx:b_idx + 1].mean()), 4),
                "best_stem": ranked[0][0] if ranked else "",
                "best_recall": round(ranked[0][1], 4) if ranked else None,
                "second_stem": ranked[1][0] if len(ranked) > 1 else "",
                "second_recall": round(ranked[1][1], 4) if len(ranked) > 1 else None,
                "best_stem_lift": ranked_lift[0][0] if ranked_lift else "",
                "best_lift": round(ranked_lift[0][1], 4) if ranked_lift else None,
                "second_stem_lift": ranked_lift[1][0] if len(ranked_lift) > 1 else "",
                "per_stem": per,
            }
            if extra:
                row.update(extra[k])
            rows.append(row)
        return rows

    # (a) 音频侧段落
    audio_spans, audio_extra = [], []
    for s in bundle.segments:
        try:
            a = bars.index(max(s["start_bar"], bars[0]))
            b = bars.index(min(s["end_bar"], bars[-1]))
        except ValueError:
            continue
        if b < a:
            continue
        audio_spans.append((a, b))
        audio_extra.append({"function": s["function"], "label_ja": s.get("label_ja", ""),
                            "plan_primary": s.get("primary_stem", ""),
                            "plan_secondary": s.get("secondary_stem", ""),
                            "intensity": s.get("intensity"),
                            "rest": s.get("rest"), "upgrade": s.get("upgrade")})
    audio_rows = seg_rows(audio_spans, "audio_segment", audio_extra)
    for r in audio_rows:
        r["plan_agree"] = bool(r["plan_primary"]
                               and r["plan_primary"] == TRACK_OF_STEM.get(r["best_stem"]))
        r["plan_agree_top2"] = bool(
            r["plan_primary"] and r["plan_primary"] in
            (TRACK_OF_STEM.get(r["best_stem"]), TRACK_OF_STEM.get(r["second_stem"])))
        r["plan_agree_lift"] = bool(
            r["plan_primary"]
            and r["plan_primary"] == TRACK_OF_STEM.get(r["best_stem_lift"]))

    # (b) 谱面密度变点段落
    cps5 = changepoint_segments(d_bar, k=5, min_len_frac=0.10)
    chart_spans = [(cps5[i], cps5[i + 1] - 1) for i in range(5)] if cps5 else []
    chart_rows = seg_rows(chart_spans, "chart_changepoint")

    # 段落级强度 × 密度（音频段落口径）
    seg_I = np.array([r["I_mean"] for r in audio_rows], dtype=float)
    seg_d = np.array([r["d_bar_mean"] for r in audio_rows], dtype=float)
    corr["section_spearman"] = cp.spearman(seg_I, seg_d) if seg_I.size >= 3 else float("nan")
    corr["section_pearson"] = cp.pearson(seg_I, seg_d) if seg_I.size >= 3 else float("nan")
    corr["n_sections"] = int(seg_I.size)

    # ---- 8. 候选池有效率（量化落格 onset 被官方采用的比例）----
    from tools.audio_analysis import quantize as q_mod

    bar_div, per_track_slots, _ = q_mod.quantize_song(
        {s: stem_onsets[s] for s in STEMS}, grid)
    cand: list[float] = []
    for trk, slots in per_track_slots.items():
        for bar, sl in slots.items():
            if bar < 1 or bar > grid.n_bars:
                continue
            bd = grid.bar_duration(bar)
            d = bar_div[bar].division
            t0 = grid.bar_start(bar)
            cand.extend(t0 + (np.asarray(sl, dtype=float) * bd / d))
    cand_t = stemhit.unique_times(np.asarray(cand, dtype=float), merge_sec=0.005)
    pool = stemhit.hit_stat(slot_adj, cand_t, tol=tol, span_sec=song_span)
    pool_stat = {"n_candidates": int(cand_t.size), **pool.to_dict()}

    # ---- 9. 结尾形态（本曲）----
    endings = {str(n): ending_mod.classify_ending(d_bar, tail_bars=n).to_dict()
               for n in (6, 8, 12)}
    loud = bundle.components.get("loudness")
    audio_tail = None
    if loud is not None and len(loud):
        lo = np.array([loud[b - 1] for b in bars], dtype=float)
        N = min(8, lo.size)
        audio_tail = {
            "loudness_tail_minus_mean_db": round(float(lo[-N:].mean() - lo.mean()), 3),
            "loudness_trend": round(cp.spearman(np.arange(N, dtype=float), lo[-N:]), 4),
            "intensity_tail_ratio": round(
                float(I_smooth[-N:].mean() / max(I_smooth.mean(), 1e-9)), 4),
            "density_tail_ratio": round(
                float(d_bar[-N:].mean() / max(d_bar.mean(), 1e-9)), 4),
        }

    return {
        "name": bundle.name, "level": bundle.level, "bpm": bundle.bpm,
        "first": bundle.first,
        "n_bars_audio": grid.n_bars, "n_measures_chart": len(stats),
        "n_bars_paired": len(bars),
        "notes_total": int(bundle.parse.counts["notes"]),
        "n_slots": int(slot_times.size),
        "parse_errors": len(bundle.parse.errors),
        "align_err_ms": round(align_err_ms, 3),
        "offset_check_delta_ms": round(
            float(bundle.analysis["offset_check"].get("delta_ms", float("nan"))), 2),
        "offset_verdict": bundle.analysis.get("offset_verdict", ""),
        "phi_star_ms": round(phi * 1000.0, 2),
        "phi_applied_ms": round(applied * 1000.0, 2),
        "match_rate_at_zero": round(shift["rate_at_zero"], 4),
        "match_rate_at_best": round(shift["best_rate"], 4),
        "recompute_max_abs_diff": float(
            bundle.components.get("_recompute_max_abs_diff", [float("nan")])[0]),
        "corr": corr, "peak": peak, "boundary": boundary, "floor": floor_check,
        "global_breakdown": global_break, "global_stem": global_stats,
        "pool": pool_stat, "endings": endings, "audio_tail": audio_tail,
        "audio_segments": audio_rows, "chart_segments": chart_rows,
        "_series": {"bars": bars, "d_bar": d_bar.tolist(),
                    "d_bar_w": d_bar_w.tolist(),
                    "I_smooth": I_smooth.tolist(), "I_raw": I_raw.tolist(),
                    "suggested": suggested.tolist()},
        "_features": {k: np.asarray(bundle.components[k])[[b - 1 for b in bars]].tolist()
                      for k in weights_mod.FEATURE_ORDER if k in bundle.components},
    }


# ---------------------------------------------------------------------------
# 388 谱语料：结尾形态
# ---------------------------------------------------------------------------


def corpus_endings(tails=(6, 8, 12)) -> dict:
    _ensure_paths()
    from chart_analysis.corpus import discover as discover_charts
    from chart_analysis.density import chart_density
    from chart_analysis.simai_parser import parse_chart

    charts = discover_charts()
    out: dict = {"n_files": len(charts), "by_tail": {}, "per_chart": []}
    parsed = []
    for cf in charts:
        try:
            res = parse_chart(Path(cf.path).read_text(encoding="utf-8", errors="replace"),
                              name=cf.name)
        except Exception:
            continue
        if res.errors or not res.notes:
            continue
        d = chart_density(res, cf.name)
        if d.measure_count < 16:
            continue
        parsed.append((cf, d))
    out["n_parsed"] = len(parsed)
    for N in tails:
        shapes = [ending_mod.classify_ending(d.raw, tail_bars=N) for _, d in parsed]
        out["by_tail"][str(N)] = ending_mod.summarize(shapes)
        if N == 8:
            out["per_chart"] = [
                {"name": cf.name, "level": cf.internal_level,
                 **ending_mod.classify_ending(d.raw, tail_bars=N).to_dict()}
                for cf, d in parsed]
    return out


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def write_csv(path: Path, songs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scope", "song", "level", "bpm", "kind", "seg_index", "start_bar", "end_bar",
            "function", "label_ja", "n_bars", "notes_total", "n_events",
            "align_err_ms", "phi_star_ms", "offset_delta_ms",
            "spearman_raw", "spearman_smooth", "spearman_weighted", "pearson_smooth",
            "peak_err_bars", "boundary_hit_tol1", "boundary_hit_tol2",
            "d_bar_mean", "I_mean", "suggested_notes", "floor_fit",
            "mae_notes", "bias_notes",
            "plan_primary", "plan_secondary", "best_stem", "best_recall",
            "second_stem", "second_recall", "best_stem_lift", "best_lift",
            "plan_agree", "plan_agree_top2", "plan_agree_lift",
            "recall_drums", "recall_bass", "recall_other", "recall_vocals",
            "prec_drums", "prec_bass", "prec_other", "prec_vocals",
            "lift_drums", "lift_bass", "lift_other", "lift_vocals",
            "none_share", "pool_precision", "ending_label", "tail_ratio",
            "section_spearman"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in songs:
            if s.get("error"):
                continue
            e8 = s["endings"]["8"]
            w.writerow({
                "scope": "song", "song": s["name"], "level": s["level"], "bpm": s["bpm"],
                "kind": "", "n_bars": s["n_bars_paired"],
                "notes_total": s["notes_total"], "n_events": s["n_slots"],
                "align_err_ms": s["align_err_ms"], "phi_star_ms": s["phi_star_ms"],
                "offset_delta_ms": s["offset_check_delta_ms"],
                "spearman_raw": round(s["corr"]["spearman_raw"], 4),
                "spearman_smooth": round(s["corr"]["spearman_smooth"], 4),
                "spearman_weighted": round(s["corr"]["spearman_smooth_weighted"], 4),
                "pearson_smooth": round(s["corr"]["pearson_smooth"], 4),
                "peak_err_bars": s["peak"]["peak_err_bars"],
                "boundary_hit_tol1": s["boundary"]["hit_rate_tol1"],
                "boundary_hit_tol2": s["boundary"]["hit_rate_tol2"],
                "d_bar_mean": round(s["floor"]["actual_notes_per_bar_mean"], 3),
                "suggested_notes": round(s["floor"]["notes_per_bar_target"], 3),
                "floor_fit": s["floor"]["floor_fit_bar"],
                "mae_notes": round(s["floor"]["mae"], 3),
                "bias_notes": round(s["floor"]["bias"], 3),
                "none_share": round(s["global_breakdown"].get("none", float("nan")), 4),
                "pool_precision": s["pool"]["precision"],
                "ending_label": e8["label"], "tail_ratio": e8["tail_ratio"],
                "section_spearman": round(s["corr"]["section_spearman"], 4),
                **{f"recall_{k}": s["global_stem"][k]["recall"] for k in STEMS},
                **{f"prec_{k}": s["global_stem"][k]["precision"] for k in STEMS},
                **{f"lift_{k}": s["global_stem"][k]["lift"] for k in STEMS},
            })
            for r in s["audio_segments"] + s["chart_segments"]:
                w.writerow({
                    "scope": "segment", "song": s["name"], "level": s["level"],
                    "bpm": s["bpm"], "kind": r["kind"], "seg_index": r["seg_index"],
                    "start_bar": r["start_bar"], "end_bar": r["end_bar"],
                    "function": r.get("function", ""), "label_ja": r.get("label_ja", ""),
                    "n_bars": r["end_idx"] - r["start_idx"] + 1,
                    "n_events": r["n_events"],
                    "d_bar_mean": r["d_bar_mean"], "I_mean": r["I_mean"],
                    "plan_primary": r.get("plan_primary", ""),
                    "plan_secondary": r.get("plan_secondary", ""),
                    "best_stem": r["best_stem"], "best_recall": r["best_recall"],
                    "second_stem": r["second_stem"],
                    "second_recall": r["second_recall"],
                    "best_stem_lift": r["best_stem_lift"],
                    "best_lift": r["best_lift"],
                    "plan_agree": r.get("plan_agree", ""),
                    "plan_agree_top2": r.get("plan_agree_top2", ""),
                    "plan_agree_lift": r.get("plan_agree_lift", ""),
                    **{f"recall_{k}": r["per_stem"][k]["recall"] for k in STEMS},
                    **{f"prec_{k}": r["per_stem"][k]["precision"] for k in STEMS},
                    **{f"lift_{k}": r["per_stem"][k]["lift"] for k in STEMS},
                })


def write_overlay(path: Path, song: dict) -> None:
    """逐小节 I_bar vs d_bar 叠加图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = song["_series"]
    x = np.arange(len(s["d_bar"]))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(x, s["d_bar"], color="#c8d6e5", label="官方密度 d_bar (note/小节)")
    ax.plot(x, s["suggested"], color="#e17055", lw=1.2, ls="--",
            label="管线建议 note/小节（floor=0.25 线性映射）")
    ax.set_ylabel("note / 小节")
    ax.set_xlabel("小节（谱面口径，0 起）")
    ax2 = ax.twinx()
    ax2.plot(x, s["I_smooth"], color="#0984e3", lw=1.6, label="音频强度 I_bar (smoothed)")
    ax2.plot(x, s["I_raw"], color="#74b9ff", lw=0.8, alpha=0.6, label="I_bar (raw)")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("音频强度 [0,1]")
    for b in song["boundary"]["chart_bounds"]:
        ax.axvline(b, color="#2d3436", lw=0.8, ls=":", alpha=0.7)
    for b in song["boundary"]["audio_bounds"]:
        ax.axvline(b, color="#00b894", lw=0.8, alpha=0.6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7, ncol=2)
    c = song["corr"]
    ax.set_title(f"{song['name']} — ρ(smooth)={c['spearman_smooth']:.3f} "
                 f"ρ(raw)={c['spearman_raw']:.3f} r={c['pearson_smooth']:.3f} | "
                 f"绿=音频段落边界 黑虚=密度变点", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _setup_cjk_font() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS",
                                   "PingFang SC", "Heiti SC", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.calibration",
        description="官方音频 × 官方谱配对标定（docs/audio-analysis.md §4.8(E)）")
    p.add_argument("--calib-dir", default="out/calib", help="标定曲目录")
    p.add_argument("--csv", default="docs/research/data/audio-chart-calibration-summary.csv")
    p.add_argument("--metrics", default=None, help="全量指标 JSON 输出路径")
    p.add_argument("--plots", default=None, help="叠加图输出目录")
    p.add_argument("--tol-ms", type=float, default=30.0, help="切轨匹配容差（毫秒）")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--skip-corpus", action="store_true", help="跳过 388 谱结尾形态统计")
    return p


def run(args: argparse.Namespace) -> dict:
    _ensure_paths()
    _setup_cjk_font()
    tol = float(args.tol_ms) / 1000.0
    dirs = loader_mod.discover(Path(args.calib_dir))
    if not dirs:
        raise SystemExit(f"{args.calib_dir} 下没有跑完管线的曲目")

    songs: list[dict] = []
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for d in dirs:
        print(f"[标定] {d.name} …", flush=True)
        b = loader_mod.load_song(d)
        s = analyze_song(b, tol=tol)
        songs.append(s)
        feats = s.get("_features", {})
        if len(feats) == len(weights_mod.FEATURE_ORDER):
            X = np.column_stack([feats[k] for k in weights_mod.FEATURE_ORDER])
            datasets[s["name"]] = (X, np.asarray(s["_series"]["d_bar"], dtype=float))
        print(f"       ρ={s['corr']['spearman_smooth']:.3f} "
              f"φ*={s['phi_star_ms']:+.1f}ms 主踩="
              f"{[(r['function'], r['best_stem']) for r in s['audio_segments'][:3]]}")

    calib = weights_mod.calibrate(datasets, alpha=args.ridge_alpha)
    presets = weights_mod.evaluate_presets(datasets, {
        "initial": dict(weights_mod.INITIAL_WEIGHTS),
        "calibrated_all": calib.weights_all or dict(weights_mod.INITIAL_WEIGHTS),
        # 可解释候选：把对密度毫无贡献的 voiced 权重挪给 flux
        "voiced_to_flux": {"loudness": 0.25, "onset": 0.30, "drums": 0.20,
                           "voiced": 0.0, "flux": 0.25},
        "flux_only": {"loudness": 0.0, "onset": 0.0, "drums": 0.0,
                      "voiced": 0.0, "flux": 1.0},
        "equal": {k: 0.2 for k in weights_mod.FEATURE_ORDER},
    })

    corpus = {} if args.skip_corpus else corpus_endings()

    summary = {
        "n_songs": len(songs),
        "spearman_smooth_fisher": cp.fisher_mean(
            [s["corr"]["spearman_smooth"] for s in songs]),
        "spearman_raw_fisher": cp.fisher_mean(
            [s["corr"]["spearman_raw"] for s in songs]),
        "spearman_weighted_fisher": cp.fisher_mean(
            [s["corr"]["spearman_smooth_weighted"] for s in songs]),
        "pearson_smooth_fisher": cp.fisher_mean(
            [s["corr"]["pearson_smooth"] for s in songs]),
        "section_spearman_fisher": cp.fisher_mean(
            [s["corr"]["section_spearman"] for s in songs]),
        "section_spearman_per_song": [round(s["corr"]["section_spearman"], 4)
                                      for s in songs],
        "spearman_smooth_per_song": [round(s["corr"]["spearman_smooth"], 4)
                                     for s in songs],
        "song_names": [s["name"] for s in songs],
        "peak_err_bars": [s["peak"]["peak_err_bars"] for s in songs],
        "boundary_hit_tol1": [s["boundary"]["hit_rate_tol1"] for s in songs],
        "mae_notes": [round(s["floor"]["mae"], 3) for s in songs],
        "bias_notes": [round(s["floor"]["bias"], 3) for s in songs],
        "floor_fit": [s["floor"]["floor_fit_bar"] for s in songs],
        "calibration": calib.to_dict(),
        "weight_presets": presets,
    }
    # 逐段落类型的切轨归因（切轨验证的核心表）
    by_fn: dict[str, dict] = {}
    for s in songs:
        for r in s["audio_segments"]:
            fn = r.get("function", "?")
            e = by_fn.setdefault(fn, {"n_seg": 0, "n_events": 0,
                                      **{f"hit_{k}": 0 for k in STEMS},
                                      **{f"on_{k}": 0 for k in STEMS},
                                      **{f"used_{k}": 0 for k in STEMS},
                                      "best": {}})
            e["n_seg"] += 1
            e["n_events"] += r["n_events"]
            for k in STEMS:
                e[f"hit_{k}"] += r["per_stem"][k]["n_hit"]
                e[f"on_{k}"] += r["per_stem"][k]["n_onsets"]
                e[f"used_{k}"] += r["per_stem"][k]["n_used"]
            e["best"][r["best_stem"]] = e["best"].get(r["best_stem"], 0) + 1
    for fn, e in by_fn.items():
        for k in STEMS:
            e[f"recall_{k}"] = round(e[f"hit_{k}"] / e["n_events"], 4) \
                if e["n_events"] else None
            e[f"prec_{k}"] = round(e[f"used_{k}"] / e[f"on_{k}"], 4) \
                if e[f"on_{k}"] else None
    summary["by_function"] = by_fn

    # 密度地板池化拟合（小节尺度 / 段落尺度）
    bar_pairs = [(s["_series"]["I_smooth"], s["_series"]["d_bar"]) for s in songs]
    sec_pairs = [([r["I_mean"] for r in s["audio_segments"]],
                  [r["d_bar_mean"] for r in s["audio_segments"]])
                 for s in songs if len(s["audio_segments"]) >= 3]
    fb = cp.fit_density_floor(bar_pairs)
    fs_ = cp.fit_density_floor(sec_pairs)
    sse_at = dict(fb["curve"])
    sse_at_sec = dict(fs_["curve"])

    def _sse(curve: dict, f: float) -> float:
        k = min(curve, key=lambda x: abs(x - f))
        return curve[k]

    summary["density_floor_fit"] = {
        "bar_best_floor": round(fb["floor"], 3), "bar_best_sse": round(fb["sse"], 2),
        "bar_sse_at_0.25": round(_sse(sse_at, 0.25), 2),
        "bar_sse_at_0.60": round(_sse(sse_at, 0.60), 2),
        "section_best_floor": round(fs_["floor"], 3),
        "section_best_sse": round(fs_["sse"], 3),
        "section_sse_at_0.60": round(_sse(sse_at_sec, 0.60), 3),
        "section_sse_at_0.25": round(_sse(sse_at_sec, 0.25), 3),
        "note": "两边各除以曲内均值后只比形状；绝对量级由「定数 → note/小节」另行锚定",
    }
    # 定数锚点：note/小节 vs NPS 哪个更准
    K = {13.0: (8.03, 5.29), 13.5: (9.15, 6.13), 14.0: (10.54, 7.14),
         14.5: (10.10, 7.70)}
    npb_err, nps_err = [], []
    for s in songs:
        key = min(K, key=lambda k: abs(k - s["level"]))
        dur = s["n_bars_paired"] * 4 * 60.0 / s["bpm"]
        npb = s["floor"]["actual_notes_per_bar_mean"]
        nps = s["notes_total"] / dur if dur > 0 else float("nan")
        npb_err.append(abs(npb - K[key][0]) / K[key][0])
        nps_err.append(abs(nps - K[key][1]) / K[key][1])
    summary["level_anchor"] = {
        "notes_per_bar_mape": round(float(np.mean(npb_err)), 4),
        "nps_mape": round(float(np.mean(nps_err)), 4),
        "source": "知识 031 §7 的定数 → note/小节 与 NPS 均值",
    }
    agree = [r["plan_agree"] for s in songs for r in s["audio_segments"]]
    agree2 = [r["plan_agree_top2"] for s in songs for r in s["audio_segments"]]
    agreeL = [r["plan_agree_lift"] for s in songs for r in s["audio_segments"]]
    summary["stemplan_agree"] = round(float(np.mean(agree)), 4) if agree else None
    summary["stemplan_agree_top2"] = round(float(np.mean(agree2)), 4) if agree2 else None
    summary["stemplan_agree_lift"] = round(float(np.mean(agreeL)), 4) if agreeL else None
    summary["n_segments"] = len(agree)
    # 全曲层面的各 stem 汇总（按事件数加权）
    for key in ("recall", "precision", "lift", "chance_recall"):
        summary[f"global_{key}"] = {}
        for st in STEMS:
            vals = [(s["global_stem"][st][key], s["n_slots"]) for s in songs
                    if s["global_stem"][st].get(key) is not None]
            if vals:
                tot = sum(w for _, w in vals)
                summary[f"global_{key}"][st] = round(
                    sum(v * w for v, w in vals) / tot, 4)
    summary["global_breakdown"] = {
        k: round(float(np.mean([s["global_breakdown"][k] for s in songs])), 4)
        for k in ("any", "none", "drums", "vocals", "bass", "other",
                  "drums_only", "vocals_only", "both_drums_vocals", "no_drums")}
    summary["pool_precision"] = round(float(np.mean(
        [s["pool"]["precision"] for s in songs])), 4)
    summary["pool_recall"] = round(float(np.mean(
        [s["pool"]["recall"] for s in songs])), 4)
    summary["per_feature_spearman"] = {
        k: round(cp.fisher_mean([s["corr"]["per_feature_spearman"][k] for s in songs]), 4)
        for k in weights_mod.FEATURE_ORDER
        if all(k in s["corr"]["per_feature_spearman"] for s in songs)}
    summary["peak_err_bars_w8"] = [s["peak"]["peak_err_bars_w8"] for s in songs]
    summary["audio_peak_in_chart_top20"] = [s["peak"]["audio_peak_in_chart_top20"]
                                            for s in songs]

    out = {"summary": summary, "songs": songs, "corpus_endings": corpus}

    write_csv(Path(args.csv), songs)
    print(f"[输出] CSV → {args.csv}")
    if args.plots:
        pd = Path(args.plots)
        for s in songs:
            if not s.get("error"):
                write_overlay(pd / f"{s['name']}-overlay.png", s)
        print(f"[输出] 叠加图 → {pd}")
    if args.metrics:
        mp = Path(args.metrics)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float),
                      encoding="utf-8")
        print(f"[输出] 指标 JSON → {mp}")
    return out


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
