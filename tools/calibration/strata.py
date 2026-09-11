"""分层统计与阈值扫描（n=40 扩样时新增，2026-09-11）。

n=8 那一轮的结论全是**池化**数字（8 首汇总一个 ρ、85 段汇总一个一致率），
样本一大就必须回答"**对谁成立**"：

- 人声曲 vs 器乐曲（知识 002「副歌全踩人声」只在人声主导曲上成立的假设）；
- 定数档、BPM 段（n=8 的 BPM ≥ 200 三首被系统性高估，是分层现象还是巧合）；
- `stemplan.py` 的人声主导阈值（`VOCAL_LED_VOICED=0.45` / `VOCAL_LED_ONSET_RATIO=0.65`）
  在 n=8 上只有 1 个正例，阈值是"凑出来的"；n=40 才第一次有条件扫最优分割点。

本模块**只做统计**，不碰音频也不碰谱面文件：输入是 `cli.analyze_song` 的产物
（纯 dict / 数组），因此全部函数都能用合成数据单测。

⚠️ 口径：这里的"人声曲"是**音频侧代理**（Demucs vocals 轨的 VAD 活跃率），
不是"这首歌有没有人唱"。Demucs vocals 轨会混进 lead synth，所以器乐曲也可能拿到
不低的 `voiced_ratio` —— 引用分层结论时必须连这条一起引。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .chartpair import fisher_mean

STEMS = ("drums", "bass", "other", "vocals")

# 人声曲判据（曲级）：**Demucs vocals 轨的逐小节能量占比 `share_vocals` 均值**。
#
# 为什么不用 `voiced_ratio`（管线的人声 VAD）：n=40 上它**完全分不开人声曲与器乐曲**
# —— 器乐曲《超絶！Superlative》的 `voiced_ratio` 是 **1.000**（vocals 轨近乎静音时，
# 相对 dB 的 VAD 把底噪全判成"有人声"），而它的 `share_vocals` 只有 **0.003**。
# 换成能量占比后，n=40 的排序与曲目元数据（`&genre` / `&artist` 的 feat./CV 标记）基本一致。
#
# 阈值 0.10 取自 n=40 的分布：≤0.041 是一个干净的器乐簇（9 首），0.10–0.15 之间是过渡区。
# ⚠️ **已知一处错分**：《僕は空気が嫁ない》(0.070) 是人声曲但被判成器乐（该曲器乐极密）。
# **换阈值结论会变，引用这条分层时必须连阈值与错分一起引。**
VOCAL_SONG_SHARE = 0.10
# 旧判据保留为对照列（报告 §5.3 用它说明 VAD 为什么不能当分类器）
VOCAL_SONG_VOICED = 0.35


# ---------------------------------------------------------------------------
# 分层键
# ---------------------------------------------------------------------------


def level_band(level: float) -> str:
    """定数分档（与知识 031 §7 的四档对齐）。"""
    lv = float(level)
    if lv < 13.25:
        return "13.0-13.2"
    if lv < 13.75:
        return "13.3-13.7"
    if lv < 14.25:
        return "13.8-14.2"
    return "14.3-14.5"


def bpm_band(bpm: float) -> str:
    """BPM 分段（n=8 报告里"BPM ≥ 200 被系统性高估"的那条分界保留成一档）。"""
    b = float(bpm)
    if b < 150:
        return "<150"
    if b < 180:
        return "150-179"
    if b < 200:
        return "180-199"
    return ">=200"


def vocal_profile(song: dict) -> dict:
    """曲级人声画像：vocals 轨能量占比、VAD 活跃率、人声/鼓 onset 数比、人声 onset 天花板。

    参数 `song` 是 `cli.analyze_song` 的返回值（需要 `global_stem` / `n_slots` /
    `_features` / `audio_profile`）。`kind` 用 `VOCAL_SONG_SHARE` 对
    `share_vocals` 二分；`kind_by_voiced` 是旧 VAD 判据的对照列。
    """
    voiced = np.asarray(song.get("_features", {}).get("voiced", []), dtype=float)
    voiced_mean = float(voiced.mean()) if voiced.size else float("nan")
    share = song.get("audio_profile", {}).get("share_vocals_mean")
    share = float(share) if share is not None else float("nan")
    gs = song.get("global_stem", {})
    n_voc = float(gs.get("vocals", {}).get("n_onsets", 0) or 0)
    n_dr = float(gs.get("drums", {}).get("n_onsets", 0) or 0)
    n_slots = float(song.get("n_slots", 0) or 0)
    return {
        "share_vocals_mean": round(share, 4) if np.isfinite(share) else None,
        "voiced_mean": round(voiced_mean, 4) if np.isfinite(voiced_mean) else None,
        "vocals_onsets": int(n_voc), "drums_onsets": int(n_dr),
        "vocals_over_drums": round(n_voc / n_dr, 4) if n_dr > 0 else None,
        "vocal_ceiling": round(n_voc / n_slots, 4) if n_slots > 0 else None,
        "kind": ("vocal" if (np.isfinite(share) and share >= VOCAL_SONG_SHARE)
                 else "instrumental"),
        "kind_by_voiced": ("vocal" if (np.isfinite(voiced_mean)
                                       and voiced_mean >= VOCAL_SONG_VOICED)
                           else "instrumental"),
    }


# ---------------------------------------------------------------------------
# 分层汇总
# ---------------------------------------------------------------------------


def stratify(items: list[dict], key: str, value: str,
             agg: str = "fisher") -> dict[str, dict]:
    """按 `item[key]` 分组，对 `item[value]` 做 Fisher-z 汇总（或均值）。

    返回 ``{组名: {"n": …, "value": …, "min": …, "max": …}}``。
    """
    groups: dict[str, list[float]] = {}
    for it in items:
        k = it.get(key)
        v = it.get(value)
        if k is None or v is None:
            continue
        v = float(v)
        if not np.isfinite(v):
            continue
        groups.setdefault(str(k), []).append(v)
    out: dict[str, dict] = {}
    for k in sorted(groups):
        vals = groups[k]
        agg_v = fisher_mean(vals) if agg == "fisher" else float(np.mean(vals))
        out[k] = {"n": len(vals), "value": round(float(agg_v), 4),
                  "median": round(float(np.median(vals)), 4),
                  "min": round(float(min(vals)), 4),
                  "max": round(float(max(vals)), 4)}
    return out


# 分层键的显示顺序（字典序会把 "<150" 排到 ">=200" 后面，必须显式给序）
GROUP_ORDER = {
    "bpm_band": ("<150", "150-179", "180-199", ">=200"),
    "level_band": ("13.0-13.2", "13.3-13.7", "13.8-14.2", "14.3-14.5"),
    "kind": ("instrumental", "vocal"),
}


def boxplot_groups(items: list[dict], key: str, value: str) -> dict[str, list[float]]:
    """给箱线图用的原始分组值（不聚合），按 `GROUP_ORDER` 排序。"""
    groups: dict[str, list[float]] = {}
    for it in items:
        k, v = it.get(key), it.get(value)
        if k is None or v is None:
            continue
        v = float(v)
        if np.isfinite(v):
            groups.setdefault(str(k), []).append(v)
    order = GROUP_ORDER.get(key)
    keys = ([k for k in order if k in groups] + sorted(set(groups) - set(order))
            if order else sorted(groups))
    return {k: sorted(groups[k]) for k in keys}


# ---------------------------------------------------------------------------
# 符号检验
# ---------------------------------------------------------------------------


def sign_test_p(n_win: int, n: int) -> float:
    """双侧精确符号检验（二项，p=0.5）。n=0 时返回 nan。"""
    if n <= 0:
        return float("nan")
    k = int(min(n_win, n - n_win))

    def cdf(kk: int) -> float:
        return sum(math.comb(n, i) for i in range(kk + 1)) / (2.0 ** n)

    return float(min(1.0, 2.0 * cdf(k)))


def weight_switch_verdict(rho_fit: list[float], rho_init: list[float],
                          min_gain: float = 0.03, min_win_frac: float = 0.70,
                          max_p: float = 0.05) -> dict:
    """换默认权重的三条判据（任务约定）：CV 提升 ≥ min_gain、胜出折数 ≥ 70%、符号检验 p < 0.05。

    **三条同时满足才换**；返回逐条通过情况与最终结论。
    """
    a = np.asarray(rho_fit, dtype=float)
    b = np.asarray(rho_init, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = int(a.size)
    if n == 0:
        return {"n_folds": 0, "switch": False, "reason": "无有效折"}
    wins = int(np.sum(a > b))
    gain = float(a.mean() - b.mean())
    p = sign_test_p(wins, n)
    c1, c2, c3 = gain >= min_gain, wins / n >= min_win_frac, p < max_p
    return {
        "n_folds": n, "n_wins": wins, "win_frac": round(wins / n, 4),
        "mean_fit": round(float(a.mean()), 4), "mean_init": round(float(b.mean()), 4),
        "median_fit": round(float(np.median(a)), 4),
        "median_init": round(float(np.median(b)), 4),
        "gain_mean": round(gain, 4),
        "gain_median": round(float(np.median(a) - np.median(b)), 4),
        "sign_test_p": round(p, 4),
        "criteria": {f"gain>={min_gain}": bool(c1),
                     f"win_frac>={min_win_frac}": bool(c2),
                     f"p<{max_p}": bool(c3)},
        "switch": bool(c1 and c2 and c3),
    }


# ---------------------------------------------------------------------------
# 阈值扫描（人声主导判据）
# ---------------------------------------------------------------------------


@dataclass
class Split:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else float("nan")

    @property
    def tpr(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def fpr(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else float("nan")

    @property
    def youden(self) -> float:
        t, f = self.tpr, self.fpr
        return (t - f) if (np.isfinite(t) and np.isfinite(f)) else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    def to_dict(self) -> dict:
        return {"threshold": round(self.threshold, 4), "tp": self.tp, "fp": self.fp,
                "fn": self.fn, "tn": self.tn,
                "accuracy": round(self.accuracy, 4),
                "tpr": round(self.tpr, 4) if np.isfinite(self.tpr) else None,
                "fpr": round(self.fpr, 4) if np.isfinite(self.fpr) else None,
                "precision": (round(self.precision, 4)
                              if np.isfinite(self.precision) else None),
                "youden": round(self.youden, 4) if np.isfinite(self.youden) else None}


def sweep_threshold(scores, labels, grid=None, criterion: str = "youden") -> dict:
    """单阈值扫描：`score >= t` 判正，找最优 t。

    `criterion` ∈ {"youden", "accuracy"}。返回最优分割点 + 全部候选点的曲线 +
    AUC（梯形法，按 FPR 排序）。样本全同类时返回 `best=None`。
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if s.size == 0 or y.all() or (~y).all():
        return {"best": None, "n": int(s.size), "n_pos": int(y.sum()),
                "curve": [], "auc": None,
                "note": "样本全为同一类或为空，无法扫阈值"}
    if grid is None:
        uniq = np.unique(s)
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        grid = np.concatenate([[uniq[0] - 1e-6], mids, [uniq[-1] + 1e-6]])
    splits = []
    for t in np.asarray(grid, dtype=float):
        pred = s >= t
        splits.append(Split(tp=int(np.sum(pred & y)), fp=int(np.sum(pred & ~y)),
                            fn=int(np.sum(~pred & y)), tn=int(np.sum(~pred & ~y)),
                            threshold=float(t)))
    key = (lambda sp: sp.youden) if criterion == "youden" else (lambda sp: sp.accuracy)
    best = max(splits, key=lambda sp: (key(sp) if np.isfinite(key(sp)) else -9e9))
    pts = sorted({(sp.fpr, sp.tpr) for sp in splits if np.isfinite(sp.fpr)})
    auc = float(np.trapezoid([p[1] for p in pts], [p[0] for p in pts])) \
        if len(pts) >= 2 else None
    return {"best": best.to_dict(), "n": int(s.size), "n_pos": int(y.sum()),
            "criterion": criterion,
            "auc": round(auc, 4) if auc is not None else None,
            "curve": [sp.to_dict() for sp in splits]}


def sweep_two_thresholds(score_a, score_b, labels,
                         grid_a=None, grid_b=None) -> dict:
    """两条件同时成立（`a >= ta and b >= tb`）的联合扫描，对应 `stemplan._vocal_led`。"""
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, y = a[ok], b[ok], y[ok]
    if a.size == 0 or y.all() or (~y).all():
        return {"best": None, "n": int(a.size), "n_pos": int(y.sum()),
                "note": "样本全为同一类或为空，无法扫阈值"}

    def _grid(x, g):
        if g is not None:
            return np.asarray(g, dtype=float)
        u = np.unique(x)
        mids = (u[:-1] + u[1:]) / 2.0
        return np.concatenate([[u[0] - 1e-6], mids, [u[-1] + 1e-6]])

    ga, gb = _grid(a, grid_a), _grid(b, grid_b)
    best = None
    for ta in ga:
        for tb in gb:
            pred = (a >= ta) & (b >= tb)
            sp = Split(threshold=float("nan"),
                       tp=int(np.sum(pred & y)), fp=int(np.sum(pred & ~y)),
                       fn=int(np.sum(~pred & y)), tn=int(np.sum(~pred & ~y)))
            j = sp.youden
            if best is None or (np.isfinite(j) and j > best[0]):
                best = (j, float(ta), float(tb), sp)
    j, ta, tb, sp = best
    return {"best": {"threshold_a": round(ta, 4), "threshold_b": round(tb, 4),
                     **sp.to_dict()},
            "n": int(a.size), "n_pos": int(y.sum())}


def evaluate_fixed_two(score_a, score_b, labels, ta: float, tb: float) -> dict:
    """给定一组固定阈值（如 `stemplan` 的 0.45/0.65）的混淆矩阵。"""
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    y = np.asarray(labels, dtype=bool)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, y = a[ok], b[ok], y[ok]
    pred = (a >= ta) & (b >= tb)
    sp = Split(threshold=float("nan"),
               tp=int(np.sum(pred & y)), fp=int(np.sum(pred & ~y)),
               fn=int(np.sum(~pred & y)), tn=int(np.sum(~pred & ~y)))
    return {"threshold_a": ta, "threshold_b": tb, **sp.to_dict()}


# ---------------------------------------------------------------------------
# 切轨模型一致率（骨架模型 / 常数基线 / v0.2 单值规则表）
# ---------------------------------------------------------------------------


def plan_v02_primary(segments: list[dict], bar_features: dict) -> list[str]:
    """**复刻** v0.2 的单值 `primary_stem` 规则表，只为给 n=40 提供对照基线。

    原实现见 git `e2e83de:tools/audio_analysis/stemplan.py`（已被 v0.3 的
    骨架 + 点缀模型替换）。这里按段落 dict + 逐小节特征数组重写同一套判据：
    判据是**能量占比 `share_*`**（v0.3 已换成 onset 匹配倾向），降级判据
    `n_onset < 2` 或 `grid_fit_bar < 0.5` 保持一致。

    ⚠️ 这是**基线复刻，不是生产代码**：只返回逻辑轨名列表，不写回段落，
    也不复刻 v0.2 的 note 文本与整曲单轨警告。
    """
    track_of = {"drums": "drum", "vocals": "vocal", "bass": "bass", "other": "hook"}

    def sl(key: str, a: int, b: int) -> np.ndarray:
        arr = np.asarray(bar_features.get(key, []), dtype=float)
        return arr[a - 1:min(b, len(arr))]

    def mean(key: str, a: int, b: int, default: float = 0.0) -> float:
        s = sl(key, a, b)
        return float(s.mean()) if s.size else float(default)

    def loudest(a: int, b: int) -> list[str]:
        sc = {st: mean(f"share_{st}", a, b) for st in STEMS}
        return sorted(sc, key=sc.get, reverse=True)

    def usable(st: str, a: int, b: int) -> bool:
        if mean(f"n_onset_{st}", a, b, 99.0) < 2.0:
            return False
        fit_key = (f"grid_fit_bar_{st}" if f"grid_fit_bar_{st}" in bar_features
                   else f"grid_fit_{st}")
        return mean(fit_key, a, b, 1.0) >= 0.5

    out: list[str] = []
    by_span: dict[tuple, dict] = {}
    intro_stem = ""
    for s in segments:
        a, b = int(s["start_bar"]), int(s["end_bar"])
        fn = s.get("function", "")
        ranked = loudest(a, b)
        primary, secondary = "", ""
        if fn == "intro":
            # v0.2 逐 4 小节重算最响轨，第一窗即 primary
            primary = loudest(a, min(b, a + 3))[0] if ranked else ""
            intro_stem = primary
        elif fn == "outro":
            primary = intro_stem or (ranked[0] if ranked else "")
        elif fn == "verse":
            primary = "other" if "other" in ranked[:2] else (ranked[0] if ranked else "")
            if primary != "other" and "drums" in ranked[:2]:
                primary = "drums"
            secondary = "vocals"
        elif fn == "pre_chorus":
            primary = "drums" if "drums" in ranked[:2] else (ranked[0] if ranked else "")
            secondary = "vocals"
        elif fn in ("chorus", "final_chorus"):
            primary, secondary = "vocals", "drums"      # 规则 4：副歌全踩人声
        elif fn == "quiet_chorus":
            primary = ranked[1] if len(ranked) > 1 else (ranked[0] if ranked else "")
            secondary = "vocals"
        elif fn == "interlude":
            v = mean("voiced_ratio", a, b)
            primary = "vocals" if 0.2 <= v <= 0.5 else "other"
        elif fn == "drop":
            primary, secondary = (ranked[0] if ranked else ""), "drums"
        else:
            primary = ranked[0] if ranked else ""
        if s.get("rest") and fn != "quiet_chorus":
            alt = [x for x in ranked if x != primary]
            if alt:
                secondary, primary = primary, alt[0]
        ref_span = tuple(s["repeat_of"]) if s.get("repeat_of") else None
        if ref_span and fn in ("chorus", "final_chorus", "drop"):
            ref = by_span.get(ref_span)
            if ref and ref["function"] in ("chorus", "final_chorus", "drop"):
                primary = ref["stem"]
        if primary and not usable(primary, a, b):
            primary = "drums"
        by_span[(a, b)] = {"stem": primary, "function": fn}
        out.append(track_of.get(primary, primary))
    return out


def agreement_table(rows: list[dict], truth_key: str = "best_stem") -> dict:
    """一次算清三个模型对 `truth_key` 的一致率。

    每行需要 `plan_primary`（v0.3 骨架轨的逻辑轨名）、可选 `plan_v02`（v0.2 规则表的
    逻辑轨名）、`best_stem` / `best_stem_lift`（实测）、`second_stem`。
    """
    track_of = {"drums": "drum", "vocals": "vocal", "bass": "bass", "other": "hook"}
    n = len(rows)
    if not n:
        return {"n": 0}

    def frac(pred_key: str) -> float | None:
        vals = [r for r in rows if r.get(pred_key)]
        if not vals:
            return None
        hit = sum(1 for r in vals
                  if r[pred_key] == track_of.get(r.get(truth_key, ""), ""))
        return round(hit / len(vals), 4)

    const = round(sum(1 for r in rows
                      if r.get(truth_key) == "drums") / n, 4)
    top2 = [r for r in rows if r.get("plan_primary")]
    top2_rate = round(sum(1 for r in top2 if r["plan_primary"] in (
        track_of.get(r.get(truth_key, ""), ""),
        track_of.get(r.get("second_stem", ""), ""))) / len(top2), 4) if top2 else None
    # 点缀轨命中：实测第二高轨是否在 accent 列表里
    acc_rows = [r for r in rows if r.get("accent_stems")]
    acc_rate = round(sum(1 for r in acc_rows
                         if track_of.get(r.get("second_stem", ""), "")
                         in r["accent_stems"]) / len(acc_rows), 4) if acc_rows else None
    return {
        "n": n,
        "skeleton_top1": frac("plan_primary"),
        "skeleton_top2": top2_rate,
        "accent_hits_second": acc_rate,
        "n_accent_rows": len(acc_rows),
        "v02_rule_top1": frac("plan_v02"),
        "n_v02_rows": sum(1 for r in rows if r.get("plan_v02")),
        "constant_drums": const,
        "truth_dist": {s: sum(1 for r in rows if r.get(truth_key) == s)
                       for s in STEMS},
    }
