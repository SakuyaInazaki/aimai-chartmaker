"""v0.5 分轨细化的标定：**四轨 vs 六轨 + 音高 note + fx** 的归因对照。

回答 `.agent/notes/043` 提出的三个问题：

1. 六路分离（htdemucs_6s 的 guitar/piano）、**有音高 note**（basic-pitch）、
   **高频瞬态**（fx）这三条新轨，各自能解释多少官方踩音？lift 是多少？
2. n=40 报告 §5.2 里"**16.8% 的官方 note 什么 stem 都不落**"，新轨收回了多少？
3. 人声曲的瓶颈（逐小节 ρ 0.146）——把"人声活动"从**能量 VAD** 换成
   **有音高 note 覆盖率**之后，强度融合的 LOSO CV 有没有改善？

口径与 n=40 报告完全一致（时间槽去重、±30 ms、`lift = recall / (1 − e^{−2λτ})`），
**新轨只是加进同一张表**，四路的既有数字必须能复现。

⚠️ ground truth 仍然是**代理**（官方 note 是否落在某条轨的事件 ±30 ms 内）。
有音高 note 轨的代理比能量 onset 更接近"谱师听到的音"，但它也有自己的失败模式：
basic-pitch 是**复音**转录器，一个和弦吐 3–5 个 note，不筛声部的话候选池密度
比鼓轨高一个数量级，随机基线被抬爆 —— 所以默认只取最高声部（`lead_mask`）。
"""

from __future__ import annotations

import numpy as np

from . import chartpair as cp
from . import stemhit
from . import weights as weights_mod

#: 四路基线轨（"原来"）
BASE4 = ("drums", "bass", "other", "vocals")
#: v0.5 新增的轨（"现在"）
NEW_TRACKS = ("guitar", "piano", "melody", "fx", "pnote_vocals")
#: 归因表里要逐条列出的全部轨
ALL_TRACKS = BASE4 + NEW_TRACKS + (
    "melody_all", "melody_c50", "melody_c60", "melody_c70",
    "vocal_plus", "drums_6s", "bass_6s", "other_6s", "vocals_6s",
    "pnote_other", "pnote_lead_other", "pnote_guitar", "pnote_piano", "pnote_bass",
    "pnote_lead_vocals")
#: "收回 16.8%" 那一问里算作新轨的集合（不含与四路同源的 `*_6s` 复刻）
RECOVERY_TRACKS = ("guitar", "piano", "melody", "fx", "pnote_vocals")

#: 曲目类型分层的音频侧代理（与 n=40 报告 §5.3 一致）
VOCAL_SONG_SHARE = 0.10
#: v0.5 的新判据：人声有音高 note 的时间覆盖率
VOCAL_SONG_PITCHED = 0.15

SHIFT_SCAN = np.arange(-0.045, 0.04501, 0.0025)
APPLY_SHIFT_THRESHOLD = 0.010


def _ts(bundle, name) -> np.ndarray:
    return np.atleast_1d(np.asarray(bundle.onset_times.get(name, np.zeros(0)),
                                    dtype=float))


def analyze_song_v5(bundle, tol: float = stemhit.DEFAULT_TOL_SEC) -> dict:
    """单曲的 v0.5 归因（全曲 + 逐段 + 收回统计）。"""
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for p in (str(repo), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)

    grid = bundle.grid
    stats = bundle.density.measures
    if not stats:
        return {"name": bundle.name, "error": "谱面无 note"}

    measure_starts = [float(bundle.first) + s.start_time for s in stats]
    bar_starts = [grid.bar_start(b) for b in range(1, grid.n_bars + 1)]
    m2b = cp.map_measures_to_bars(measure_starts, bar_starts, tol_sec=0.05)
    idx = sorted(m2b)
    if not idx:
        return {"name": bundle.name, "error": "小节对齐失败"}
    bars = [m2b[i] for i in idx]

    note_times = np.array([float(bundle.first) + n.time for n in bundle.parse.notes],
                          dtype=float)
    slot_times = stemhit.unique_times(note_times, merge_sec=1e-3)
    all_onsets = np.sort(np.concatenate([_ts(bundle, s) for s in BASE4]))
    shift = stemhit.best_shift(slot_times, all_onsets, SHIFT_SCAN, tol=tol)
    phi = float(shift["best_shift_sec"])
    applied = phi if abs(phi) > APPLY_SHIFT_THRESHOLD else 0.0
    slot_adj = slot_times + applied

    span = float(measure_starts[idx[-1]] - measure_starts[idx[0]]) or 1.0
    present = [k for k in ALL_TRACKS if _ts(bundle, k).size or k in BASE4]
    onsets_map = {k: _ts(bundle, k) for k in present}

    global_stem = {k: stemhit.hit_stat(slot_adj, v, tol, span_sec=span).to_dict()
                   for k, v in onsets_map.items()}
    breakdown = stemhit.explain_breakdown(slot_adj, onsets_map, tol=tol,
                                          base_tracks=BASE4)
    recovery = stemhit.recovery_breakdown(
        slot_adj, onsets_map, base_tracks=BASE4,
        new_tracks=tuple(k for k in RECOVERY_TRACKS if k in onsets_map), tol=tol)

    # ---- 曲级画像 ----
    bar_feat = _bar_features(bundle.analysis)
    share_vocals = float(np.mean(bar_feat.get("share_vocals", np.zeros(1))))
    voiced_vad = float(np.mean(bar_feat.get("voiced_ratio", np.zeros(1))))
    pitched_cov = _pitched_cov_mean(bundle, grid)
    profile = {
        "share_vocals": round(share_vocals, 4),
        "voiced_ratio_vad": round(voiced_vad, 4),
        "vocal_pitched_ratio": round(pitched_cov, 4) if pitched_cov == pitched_cov
        else None,
        "vocal_song_by_share": bool(share_vocals >= VOCAL_SONG_SHARE),
        "vocal_song_by_vad": bool(voiced_vad >= 0.5),
        "vocal_song_by_pitched": (bool(pitched_cov >= VOCAL_SONG_PITCHED)
                                  if pitched_cov == pitched_cov else None),
    }

    # ---- 逐段 ----
    seg_rows = []
    for s in bundle.segments:
        try:
            a = bars.index(max(int(s["start_bar"]), bars[0]))
            b = bars.index(min(int(s["end_bar"]), bars[-1]))
        except ValueError:
            continue
        if b < a:
            continue
        t0 = measure_starts[idx[a]] + applied
        last = idx[b]
        last_bpm = float(stats[last].bpm) or float(bundle.bpm)
        t1 = measure_starts[last] + 4.0 * 60.0 / last_bpm + applied
        ev = slot_adj[(slot_adj >= t0) & (slot_adj < t1)]
        if ev.size == 0:
            continue
        per = {}
        for k, v in onsets_map.items():
            seg_on = v[(v >= t0) & (v < t1)]
            per[k] = stemhit.hit_stat(ev, seg_on, tol,
                                      span_sec=max(t1 - t0, 1e-6)).to_dict()
        ranked = sorted(((k, per[k]["recall"] or 0.0) for k in BASE4),
                        key=lambda kv: kv[1], reverse=True)
        # v0.5 的"最高命中轨"：候选集扩到骨架可用的六条（`fx` 与 `pnote_*` 不参选，
        # 前者定义即点缀、后者是 melody 的原料）
        v5_cand = [k for k in BASE4 + ("guitar", "piano", "melody") if k in per]
        ranked5 = sorted(((k, per[k]["recall"] or 0.0) for k in v5_cand),
                         key=lambda kv: kv[1], reverse=True)
        seg_rows.append({
            "song": bundle.name, "function": s.get("function", ""),
            "start_bar": int(s["start_bar"]), "end_bar": int(s["end_bar"]),
            "n_events": int(ev.size),
            "best_stem_base4": ranked[0][0], "best_stem_v5": ranked5[0][0],
            "skeleton_plan": s.get("skeleton_stem", ""),
            "accent_plan": list(s.get("accent_stems", []) or []),
            "per_track": per,
            "seg_recovery": stemhit.recovery_breakdown(
                ev, {k: v[(v >= t0) & (v < t1)] for k, v in onsets_map.items()},
                base_tracks=BASE4,
                new_tracks=tuple(k for k in RECOVERY_TRACKS if k in onsets_map),
                tol=tol),
        })

    return {
        "name": bundle.name, "level": bundle.level, "bpm": bundle.bpm,
        "n_slots": int(slot_times.size), "phi_applied_ms": round(applied * 1000, 2),
        "span_sec": round(span, 2),
        "profile": profile,
        "global_stem": global_stem,
        "breakdown": breakdown,
        "recovery": recovery,
        "segments": seg_rows,
        "track_counts": {k: int(v.size) for k, v in onsets_map.items()},
        "v5_available": dict(bundle.v5 or {}),
    }


def _bar_features(analysis: dict) -> dict[str, np.ndarray]:
    bars = analysis.get("bars", [])
    keys: set[str] = set()
    for b in bars:
        keys.update((b.get("features") or {}).keys())
    return {k: np.array([float((b.get("features") or {}).get(k, 0.0) or 0.0)
                         for b in bars], dtype=float) for k in sorted(keys)}


def _pitched_cov_mean(bundle, grid) -> float:
    """人声有音高 note 的全曲时间覆盖率（`vocal_pitched_ratio` 的曲级汇总）。"""
    from tools.audio_analysis import pitch_notes as pn_mod

    n = (bundle.pitch_notes or {}).get("vocals")
    if n is None or not n.available:
        return float("nan")
    return float(np.mean(pn_mod.pitched_coverage_per_bar(n, grid)))


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def pool_tracks(songs: list[dict]) -> dict:
    """全曲层面的池化归因表（按**事件数**加权，与 n=40 报告口径一致）。"""
    out: dict[str, dict] = {}
    for track in ALL_TRACKS:
        hit = ons = ev = used = 0
        chances, lifts = [], []
        wl = wr = wc = 0.0          # 事件数加权（与 n=40 报告 §5.2 的口径一致）
        for s in songs:
            g = (s.get("global_stem") or {}).get(track)
            if not g:
                continue
            ev += g["n_events"]; hit += g["n_hit"]
            ons += g["n_onsets"]; used += g["n_used"]
            if g.get("chance_recall"):
                chances.append(g["chance_recall"])
                wc += g["chance_recall"] * g["n_events"]
            if g.get("lift"):
                lifts.append(g["lift"])
                wl += g["lift"] * g["n_events"]
                wr += g["n_events"]
        if ev == 0:
            continue
        rec = hit / ev
        chance = float(np.mean(chances)) if chances else float("nan")
        out[track] = {
            "n_songs": sum(1 for s in songs if track in (s.get("global_stem") or {})),
            "n_events": ev, "n_onsets": ons,
            "recall": round(rec, 4),
            "precision": round(used / ons, 4) if ons else None,
            "chance_recall": round(chance, 4) if chance == chance else None,
            "lift_pooled": round(rec / chance, 4) if chance == chance and chance > 0
            else None,
            "lift_mean_per_song": round(float(np.mean(lifts)), 4) if lifts else None,
            # **引用时用这一列**：与 n=40 报告 §5.2 同口径（逐曲 lift 按事件数加权）
            "lift_event_weighted": round(wl / wr, 4) if wr else None,
            "chance_event_weighted": round(wc / ev, 4) if ev else None,
        }
    return out


def pool_recovery(songs: list[dict], chances: dict | None = None) -> dict:
    """"原来什么都不落的 16.8% 现在被收回多少"（池化 + 逐曲分布）。"""
    n_ev = n_un = n_rec = 0
    by_track: dict[str, int] = {}
    by_excl: dict[str, int] = {}
    per_song = []
    for s in songs:
        r = s.get("recovery") or {}
        if not r:
            continue
        n_ev += r["n_events"]; n_un += r["n_unexplained_base"]
        n_rec += int(round(r["recovered"] * r["n_unexplained_base"]))\
            if r["n_unexplained_base"] else 0
        for k, v in (r.get("by_track") or {}).items():
            by_track[k] = by_track.get(k, 0) + int(round(v * r["n_unexplained_base"]))
        for k, v in (r.get("by_track_exclusive") or {}).items():
            by_excl[k] = by_excl.get(k, 0) + int(round(v * r["n_unexplained_base"]))
        per_song.append({"song": s["name"],
                         "share_unexplained": round(r["share_unexplained_base"], 4),
                         "recovered": round(r["recovered"], 4),
                         "still_none": round(r["still_none"], 4)})
    return {
        "n_events": n_ev, "n_unexplained_base": n_un,
        "share_unexplained_base": round(n_un / n_ev, 4) if n_ev else None,
        "recovered": round(n_rec / n_un, 4) if n_un else None,
        "recovered_share_of_all": round(n_rec / n_ev, 4) if n_ev else None,
        "still_none_share_of_all": round((n_un - n_rec) / n_ev, 4) if n_ev else None,
        "by_track": {k: round(v / n_un, 4) for k, v in sorted(by_track.items())}
        if n_un else {},
        "by_track_exclusive": {k: round(v / n_un, 4)
                               for k, v in sorted(by_excl.items())} if n_un else {},
        # **必须看这一列**：melody 这类稠密轨光靠"撒得多"就能覆盖掉大半未解释事件。
        # chance = 1 − exp(−λ·2τ) 用该轨的池化随机基线，lift = 实际收回率 / chance。
        "by_track_lift": ({k: (round(v / n_un / chances[k], 3)
                               if n_un and chances.get(k) else None)
                           for k, v in sorted(by_track.items())} if chances else {}),
        "per_song": sorted(per_song, key=lambda d: -d["recovered"]),
    }


def pool_by_function(songs: list[dict], tracks=None) -> dict:
    """逐段落类型的 recall 表（新轨加进 n=40 报告 §5.5 的同一张表）。"""
    tracks = tracks or (BASE4 + ("guitar", "piano", "melody", "fx"))
    acc: dict[str, dict] = {}
    for s in songs:
        for r in s.get("segments", []):
            fn = r["function"]
            a = acc.setdefault(fn, {"n_seg": 0, "n_events": 0,
                                    "hit": {t: 0 for t in tracks},
                                    "used": {t: 0 for t in tracks},
                                    "ons": {t: 0 for t in tracks}})
            a["n_seg"] += 1
            a["n_events"] += r["n_events"]
            for t in tracks:
                p = r["per_track"].get(t)
                if p:
                    a["hit"][t] += p["n_hit"]; a["used"][t] += p["n_used"]
                    a["ons"][t] += p["n_onsets"]
    out = {}
    for fn, a in sorted(acc.items(), key=lambda kv: -kv[1]["n_events"]):
        row = {"n_seg": a["n_seg"], "n_events": a["n_events"]}
        for t in tracks:
            row[f"recall_{t}"] = round(a["hit"][t] / a["n_events"], 4) \
                if a["n_events"] else None
            row[f"prec_{t}"] = round(a["used"][t] / a["ons"][t], 4) \
                if a["ons"][t] else None
        out[fn] = row
    return out


def vocal_song_chorus(songs: list[dict],
                      tracks=("drums", "vocals", "pnote_vocals", "melody",
                              "vocal_plus")) -> dict:
    """**人声曲的副歌段**：vocal / melody 的有音高 onset recall 与 lift。

    这是 n=40 报告 §5.4 的 C1（"副歌全踩人声"）在**真人声 note 事件**下的复验 ——
    旧结论的 ground truth 是 Demucs vocals 轨的能量 onset，而器乐曲的 vocals 轨
    是纯泄漏也能拿到 0.339 的 recall（报告 §5.7），代理会**制造**假的"踩人声"。
    """
    out: dict[str, dict] = {}
    for label, pick in (("vocal_songs", True), ("instrumental_songs", False)):
        acc = {t: {"hit": 0, "ons": 0, "used": 0} for t in tracks}
        n_ev = n_seg = 0
        lifts = {t: [] for t in tracks}
        best_counts: dict[str, int] = {}
        for s in songs:
            prof = s.get("profile") or {}
            is_vocal = bool(prof.get("vocal_song_by_share"))
            if is_vocal != pick:
                continue
            for r in s.get("segments", []):
                if r["function"] not in ("chorus", "final_chorus"):
                    continue
                n_seg += 1; n_ev += r["n_events"]
                for t in tracks:
                    p = r["per_track"].get(t)
                    if not p:
                        continue
                    acc[t]["hit"] += p["n_hit"]; acc[t]["ons"] += p["n_onsets"]
                    acc[t]["used"] += p["n_used"]
                    if p.get("lift"):
                        lifts[t].append(p["lift"])
                cand = {t: (r["per_track"].get(t) or {}).get("lift") or 0.0
                        for t in tracks}
                if cand:
                    b = max(cand, key=cand.get)
                    best_counts[b] = best_counts.get(b, 0) + 1
        rows = {}
        for t in tracks:
            rows[t] = {
                "recall": round(acc[t]["hit"] / n_ev, 4) if n_ev else None,
                "precision": round(acc[t]["used"] / acc[t]["ons"], 4)
                if acc[t]["ons"] else None,
                "lift_mean": round(float(np.mean(lifts[t])), 4) if lifts[t] else None,
                "n_onsets": acc[t]["ons"],
            }
        out[label] = {"n_segments": n_seg, "n_events": n_ev, "tracks": rows,
                      "best_lift_track_counts": best_counts}
    return out


def song_type_confusion(songs: list[dict]) -> dict:
    """三套"这首是不是人声曲"判据的交叉表（能量占比 / 能量 VAD / 有音高覆盖率）。"""
    rows = []
    for s in songs:
        p = s.get("profile") or {}
        rows.append({"song": s["name"], **p})
    def _x(a, b):
        t = {"tt": 0, "tf": 0, "ft": 0, "ff": 0}
        for r in rows:
            if r.get(a) is None or r.get(b) is None:
                continue
            t["tt" if (r[a] and r[b]) else "tf" if r[a] else
              "ft" if r[b] else "ff"] += 1
        return t
    return {
        "rows": rows,
        "share_vs_vad": _x("vocal_song_by_share", "vocal_song_by_vad"),
        "share_vs_pitched": _x("vocal_song_by_share", "vocal_song_by_pitched"),
    }


# ---------------------------------------------------------------------------
# 强度融合：把 `voiced`（能量 VAD）换成 `vocal_pitched`（有音高覆盖率）
# ---------------------------------------------------------------------------

#: v0.5 的特征顺序（第 4 项换件）
FEATURE_ORDER_V5 = ("loudness", "onset", "drums", "vocal_pitched", "flux")
#: 现行默认（n=40 标定值，`docs/audio-analysis.md` §4.5）
CURRENT_WEIGHTS = {"loudness": 0.00, "onset": 0.19, "drums": 0.07,
                   "voiced": 0.00, "flux": 0.74}


def build_intensity_datasets(bundles: dict, songs_meta: dict) -> dict:
    """给权重标定准备 `{曲名: (X, y)}`，X 的第 4 列是**有音高覆盖率**。"""
    from tools.audio_analysis import pitch_notes as pn_mod

    out = {}
    for name, b in bundles.items():
        meta = songs_meta.get(name) or {}
        bars = meta.get("_bars")
        d_bar = meta.get("_d_bar")
        if not bars or d_bar is None:
            continue
        pv = (b.pitch_notes or {}).get("vocals")
        cov = (pn_mod.pitched_coverage_per_bar(pv, b.grid) if pv is not None
               and pv.available else np.zeros(b.grid.n_bars))
        cols = []
        for k in FEATURE_ORDER_V5:
            if k == "vocal_pitched":
                arr = cov
            else:
                arr = b.components.get(k)
            if arr is None:
                arr = np.zeros(b.grid.n_bars)
            cols.append(np.asarray([arr[i - 1] for i in bars], dtype=float))
        out[name] = (np.column_stack(cols), np.asarray(d_bar, dtype=float))
    return out


def compare_intensity_weights(ds_v4: dict, ds_v5: dict, alpha: float = 1.0) -> dict:
    """v0.4（`voiced` = 能量 VAD）vs v0.5（`vocal_pitched`）的 LOSO 对照。

    换权重的三条判据（与 n=40 报告 §3.3 相同，不得放宽）：
    ① 留一 CV 的 Spearman 均值明显提高；② 多数折胜出；③ 符号检验显著。
    """
    res_v4 = weights_mod.calibrate(ds_v4, alpha=alpha,
                                   feature_order=weights_mod.FEATURE_ORDER,
                                   initial=CURRENT_WEIGHTS)
    res_v5 = weights_mod.calibrate(ds_v5, alpha=alpha,
                                   feature_order=FEATURE_ORDER_V5,
                                   initial={k: CURRENT_WEIGHTS.get(
                                       "voiced" if k == "vocal_pitched" else k, 0.0)
                                       for k in FEATURE_ORDER_V5})
    common = sorted(set(f.held_out for f in res_v4.folds)
                    & set(f.held_out for f in res_v5.folds))
    a = np.array([next(f.rho_fit for f in res_v4.folds if f.held_out == n)
                  for n in common], dtype=float)
    b = np.array([next(f.rho_fit for f in res_v5.folds if f.held_out == n)
                  for n in common], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    wins = int(np.sum(b[ok] > a[ok]))
    return {
        "v4": res_v4.to_dict(), "v5": res_v5.to_dict(),
        "n_folds": int(ok.sum()),
        "rho_v4_mean": round(float(a[ok].mean()), 4) if ok.any() else None,
        "rho_v5_mean": round(float(b[ok].mean()), 4) if ok.any() else None,
        "delta_mean": round(float((b[ok] - a[ok]).mean()), 4) if ok.any() else None,
        "n_wins_v5": wins,
        "sign_test_p": _sign_test_p(wins, int(ok.sum())),
        "per_fold": [{"song": n, "rho_v4": round(float(x), 4),
                      "rho_v5": round(float(y), 4),
                      "delta": round(float(y - x), 4)}
                     for n, x, y in zip(common, a, b)],
    }


def _sign_test_p(wins: int, n: int) -> float | None:
    """双侧精确符号检验（p = 2·P(X ≥ wins)，X ~ Bin(n, 0.5)）。"""
    if n <= 0:
        return None
    from math import comb

    k = max(wins, n - wins)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return round(min(1.0, 2.0 * tail), 6)


def stratified_rho(songs_meta: dict, songs: list[dict], key: str) -> dict:
    """按曲目类型分层的逐小节 ρ（用来看人声曲瓶颈有没有动）。"""
    prof = {s["name"]: (s.get("profile") or {}) for s in songs}
    groups: dict[str, list[float]] = {"vocal": [], "instrumental": []}
    for name, meta in songs_meta.items():
        r = meta.get(key)
        if r is None or not np.isfinite(r):
            continue
        p = prof.get(name, {})
        groups["vocal" if p.get("vocal_song_by_share") else "instrumental"].append(
            float(r))
    return {g: {"n": len(v), "mean": round(float(np.mean(v)), 4) if v else None,
                "median": round(float(np.median(v)), 4) if v else None}
            for g, v in groups.items()}
