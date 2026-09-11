"""逐小节强度曲线、高潮五票、强度→建议密度映射（v0.2）。

## 相对 v0.1 的三处改动

### 1. 融合式换成调研 v2 §3.1

```
I = 0.25·Z(loudness) + 0.30·Z(n_onset 去重) + 0.20·Z(E_drums)
  + 0.15·Z(voiced_ratio) + 0.10·Z(flux)
```

- **质心移出融合项**：质心对失真吉他/riser 敏感，但对高音人声同样敏感，
  会把 Bメロ 的人声爬升误判成 drop。质心只保留在高潮票里（权重 0.05）；
- **onset 从"强度包络"换成"去重合并后的 onset 计数"**，并升为最大权重——
  它是"这一小节最多能踩几个音"的直接代理，而混音 onset 包络会被最响音轨支配
  （正是 MMFC 5.4 批评的"哪个响踩哪个"）；
- **新增人声活动率**：J-pop 的情绪主载体。

### 2. 修掉 v0.1 的 raw/smoothed 系统性偏离（主会话验收发现的 C 项）

**根因**：v0.1 的 `bar_raw` 是**未归一**的融合值，而 `bar_intensity` 是
「平滑后再做一次 min-max」的值；画图时 raw 只被 `/max` 归一（下界留在
`min/max`），smoothed 却被拉到 `[0,1]`。二者基准不同，偏离量恰好是
`min/max`（低强度段最大），所以在チモシー健康ジャズ 34–48 这种弱段
能看到 0.2–0.3 的"系统性偏离"——**平滑本身没有改变电平，是两次不同的归一
造成的假象**。

**修法**：归一只做一次。先把融合值做鲁棒 min-max（P5–P95 截断）得到
`bar_raw ∈ [0,1]`，**再**平滑；平滑后不再归一。于是 raw 与 smoothed 同基准，
平滑只能改变形状不能改变电平。

### 3. 新增强度 → 建议密度映射（官方谱密度曲线报告 §4.3）

```
density_norm = floor + (1 − floor) · I      # 段落 floor 0.60、小节 floor 0.25
建议 note/小节 = density_norm × 该定数的官方均值 note/小节
```

官方谱密度有**可玩性地板**：音频强度可以逼近 0，谱面密度不能
（实测归一密度 p10 = 0.264、段落尺度最弱段 ≈ 0.62）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOP = 512
ANALYSIS_SR = 22050
TARGET_LUFS = -14.0

# v2 §3.1 新融合式。
# 🧪 **2026-09-11 已用 8 首官方音频×官方谱做过配对标定，结论是维持初值不变**
#    （`docs/research/audio-chart-calibration.md` §3）：留一曲交叉验证的 Spearman
#    标定值 0.515（中位 0.537）vs 初值 0.488（中位 0.518），仅 5/8 折胜出、
#    符号检验 p ≈ 0.36，不满足"明显优于"。
#    同一份报告留下两条**待验假设**（拿到 ≥20 首配对数据后优先复验，勿凭 n=8 改动）：
#      H1 `voiced` → 0.0（它与官方逐小节密度的 Spearman 只有 0.016）；
#      H2 `flux` 0.10 → 0.25（它是五项里单项最强的，Spearman 0.524，高于整条融合式的 0.454）。
DEFAULT_FUSION_WEIGHTS = {
    "loudness": 0.25,
    "onset": 0.30,      # 去重合并 onset 计数
    "drums": 0.20,
    "voiced": 0.15,
    "flux": 0.10,
}
# v2 §3.2 四票 → 五票
DEFAULT_VOTE_WEIGHTS = {
    "structure": 0.45,
    "novelty": 0.25,
    "energy": 0.20,
    "centroid": 0.05,
    "vocal": 0.05,
}

# 官方谱逐小节密度绝对量级（docs/research/official-chart-density-curves.md §2.4）
# ⚠️ v0.3 起**不再作为主锚点**，只保留作对照列：标定实测这条锚在 BPM ≥ 200 的曲子上
#    系统性高估 3.1–3.7 note/小节（note/小节 口径 MAPE 21.0%）。
LEVEL_NOTES_PER_BAR = {13.0: 8.03, 13.5: 9.15, 14.0: 10.54, 14.5: 10.10}
DEFAULT_NOTES_PER_BAR = 9.0
# **v0.3 主锚点：NPS**（知识 031 §7 的 NPS 列）。标定报告 §4.2：NPS 口径 MAPE 16.5%
# 优于 note/小节口径的 21.0%，幻想のサテライト（BPM 230）从 −35% 改善到 −8%。
# note/小节 = NPS × 每小节秒数 = NPS × beats_per_bar × 60 / BPM。
LEVEL_NPS = {13.0: 5.29, 13.5: 6.13, 14.0: 7.14, 14.5: 7.70}
# note 总数区间（知识 004，ST/SD 谱：均值 (p10–p90)）
LEVEL_TOTAL_NOTES = {
    13.0: (643.7, 445, 828),
    13.5: (767.4, 492, 967),
    14.0: (899.8, 685, 1080),
    14.5: (1020.7, 733, 1181),
}
# 密度地板（官方谱密度曲线报告 §4.3 / 知识 031 §2）
# ⚠️ 段落尺度 0.60 与 n=8 配对标定的池化最优值 **0.125** 冲突（标定报告 §4.1：
#    0.60 的 SSE 比最优差 10.3%）。**不改默认**——0.60 来自 388 谱的段间落差统计，
#    n=8 不足以推翻；但 CLI 开了 `--section-floor` 让使用者自己试 0.125。
SECTION_DENSITY_FLOOR = 0.60
BAR_DENSITY_FLOOR = 0.25
CALIBRATED_SECTION_FLOOR = 0.125    # n=8 实测最优（仅记录，不作默认）
# 结构封顶：intro/outro 的密度基线**由结构决定**，不跟能量曲线走。
# 依据（个案，主会话 2026-09-11 看叠加图）：MYTHOS 小节 8–19 音频强度 ≈ 1.0，
# 官方谱却只放 4 note/小节 —— 谱师按「这是前奏」压密度，不按响度。
INTRO_OUTRO_CAP_RATIO = 0.60
STRUCTURAL_CAP_FUNCTIONS = ("intro", "outro")


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS
                       ) -> tuple[np.ndarray, dict]:
    """pyloudnorm 归一到目标 LUFS。失败（音频太短等）时原样返回。"""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        if not np.isfinite(loudness):
            return y, {"applied": False, "reason": "integrated_loudness 非有限值"}
        out = pyln.normalize.loudness(y, loudness, target_lufs)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0:
            out = out / peak
        return out.astype(np.float32), {
            "applied": True, "input_lufs": float(loudness), "target_lufs": target_lufs,
        }
    except Exception as exc:
        return y, {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}


def zscore(x: np.ndarray) -> np.ndarray:
    """标准化（全曲逐小节）。方差为 0 时返回全 0。"""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    s = float(x.std())
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / s


def robust_unit(x: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """P5–P95 截断后的 min-max，把任意量纲压到 [0,1]。**全流程只做这一次归一。**"""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = float(np.percentile(x, lo_pct)), float(np.percentile(x, hi_pct))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _bar_rms_db(y: np.ndarray, sr: int, grid) -> np.ndarray:
    """逐小节响度（dBFS），用于 `Z(loudness)`。"""
    out = np.full(grid.n_bars, -80.0, dtype=float)
    if y is None or np.size(y) == 0:
        return out
    for bar in range(1, grid.n_bars + 1):
        a = int(round(max(0.0, grid.bar_start(bar)) * sr))
        b = int(round(max(0.0, grid.bar_start(bar) + grid.bar_duration(bar)) * sr))
        a, b = max(0, min(a, len(y))), max(0, min(b, len(y)))
        if b > a:
            r = float(np.sqrt(np.mean(np.square(y[a:b], dtype=np.float64))))
            out[bar - 1] = 20.0 * np.log10(max(r, 1e-8))
    return out


@dataclass
class IntensityResult:
    bar_intensity: np.ndarray          # 平滑后（与 bar_raw 同基准，均 ∈ [0,1]）
    bar_raw: np.ndarray                # 归一后、未平滑
    components: dict[str, np.ndarray]  # 逐小节原始分量（未 Z 化）
    components_z: dict[str, np.ndarray]
    votes: dict[str, np.ndarray] = field(default_factory=dict)
    vote_total: np.ndarray = field(default_factory=lambda: np.zeros(0))
    climax_bar: int = 0
    climax_peaks: list[int] = field(default_factory=list)
    loudness_info: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)


def compute_intensity(
    grid,
    y_mix: np.ndarray,
    sr: int,
    bar_features: dict[str, np.ndarray],
    y_drums: np.ndarray | None = None,
    fusion_weights: dict | None = None,
    smooth_median_bars: int = 3,
    smooth_sigma: float = 1.0,
) -> IntensityResult:
    """按 v2 §3.1 的五项融合式算逐小节强度。

    所有分量都在**小节尺度**上算（不是帧尺度再聚合），因为 BPM/offset 由用户给定，
    小节边界是精确已知的（v2 §3.1 对 v1「`beats[::4]` 当小节边界」的修订）。
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    fw = dict(DEFAULT_FUSION_WEIGHTS)
    if fusion_weights:
        fw.update(fusion_weights)

    y_norm, loud = loudness_normalize(y_mix, sr)

    from . import features as feat_mod

    comp = {
        "loudness": _bar_rms_db(y_norm, sr, grid),
        "onset": np.asarray(bar_features.get("n_onset_merged",
                                             np.zeros(grid.n_bars)), dtype=float),
        "drums": feat_mod.energy_per_bar(y_drums, sr, grid) if y_drums is not None
        else np.zeros(grid.n_bars),
        "voiced": np.asarray(bar_features.get("voiced_ratio",
                                              np.zeros(grid.n_bars)), dtype=float),
        "flux": feat_mod.spectral_flux_per_bar(y_norm, sr, grid),
    }
    # 鼓能量量纲跨度大 → 先取 dB 再 Z，避免单个爆音主导
    comp["drums"] = 10.0 * np.log10(np.maximum(comp["drums"], 1e-10))

    if y_drums is None or np.size(y_drums) == 0:
        fw = {k: v for k, v in fw.items() if k != "drums"}

    comp_z = {k: zscore(v) for k, v in comp.items() if k in fw}
    wsum = sum(fw.values()) or 1.0
    fused = sum(fw[k] * comp_z[k] for k in comp_z) / wsum

    # ⚠️ 归一只做这一次；平滑在归一之后，因此平滑不会改变电平（C 项 bug 的修法）
    bar_raw = robust_unit(fused)
    win = max(1, int(smooth_median_bars) | 1)
    sm = median_filter(bar_raw, size=win, mode="nearest")
    bar_intensity = np.clip(gaussian_filter1d(sm, sigma=smooth_sigma, mode="nearest"),
                            0.0, 1.0)

    return IntensityResult(
        bar_intensity=bar_intensity,
        bar_raw=bar_raw,
        components=comp,
        components_z=comp_z,
        loudness_info=loud,
        weights={"fusion": fw, "note": "权重为 v2 §3.1 初值，未在本项目曲库标定"},
    )


def vote_climax(
    res: IntensityResult,
    segments,
    grid,
    y_mix: np.ndarray | None = None,
    sr: int = ANALYSIS_SR,
    bar_features: dict[str, np.ndarray] | None = None,
    vote_weights: dict | None = None,
    top_k: int = 3,
    instrumental: bool = False,
) -> IntensityResult:
    """高潮五票（v2 §3.2）：结构 0.45 / novelty 0.25 / 能量 0.20 / 质心 0.05 / 人声 0.05。

    器乐曲（全曲 voiced_ratio 低）走 `instrumental=True`：能量票升到 0.5，
    结构票降到 0.25——此时"副歌"概念不成立，只有 build → drop。
    """
    vw = dict(DEFAULT_VOTE_WEIGHTS)
    if instrumental:
        vw.update({"structure": 0.25, "energy": 0.50, "vocal": 0.0, "novelty": 0.20,
                   "centroid": 0.05})
    if vote_weights:
        vw.update(vote_weights)
    n = grid.n_bars
    bf = bar_features or {}

    # V1 结构票：chorus 家族 / drop / 重复段
    v1 = np.zeros(n)
    for s in segments:
        fn = (getattr(s, "function", "") or "").lower()
        if "chorus" in fn or fn == "drop" or getattr(s, "is_repeat", False):
            v1[s.start_bar - 1: s.end_bar] = 1.0

    # V2 novelty 票：强度曲线的一阶差分正峰（段落进入点）
    v2 = np.zeros(n)
    I = res.bar_raw
    if n >= 3:
        d = np.diff(I, prepend=I[0])
        thr = float(np.percentile(d, 85))
        for i in range(1, n - 1):
            if d[i] >= thr and d[i] > 0:
                v2[i] = 1.0
                v2[min(n - 1, i + 1)] = max(v2[min(n - 1, i + 1)], 0.5)

    # V3 能量票：响度分量高且局部极大
    loud_u = robust_unit(res.components.get("loudness", np.zeros(n)))
    v3 = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if loud_u[i] > 0.75 and loud_u[i] >= loud_u[lo:hi].max() - 1e-9:
            v3[i] = 1.0

    # V4 质心票（只给 EDM drop 用，权重已降到 0.05）
    v4 = np.zeros(n)
    if y_mix is not None and np.size(y_mix):
        import librosa

        cent = librosa.feature.spectral_centroid(y=y_mix, sr=sr, hop_length=HOP)[0]
        ft = librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=HOP)
        bar_cent = np.zeros(n)
        for bar in range(1, n + 1):
            t0 = grid.bar_start(bar)
            sel = (ft >= t0) & (ft < t0 + grid.bar_duration(bar))
            if sel.any():
                bar_cent[bar - 1] = float(np.mean(cent[sel]))
        v4 = (robust_unit(bar_cent) > 0.7).astype(float)

    # V5 人声票（新增）：voiced_ratio > 0.6
    voiced = np.asarray(bf.get("voiced_ratio", np.zeros(n)), dtype=float)
    v5 = (voiced > 0.6).astype(float)

    total = (vw["structure"] * v1 + vw["novelty"] * v2 + vw["energy"] * v3
             + vw["centroid"] * v4 + vw["vocal"] * v5)
    ranked = total * (0.5 + 0.5 * res.bar_intensity)

    climax_bar = int(np.argmax(ranked)) + 1 if n else 0
    peaks: list[int] = []
    for i in np.argsort(ranked)[::-1]:
        if len(peaks) >= top_k:
            break
        if all(abs(int(i) - (p - 1)) >= 8 for p in peaks):
            peaks.append(int(i) + 1)

    res.votes = {"structure": v1, "novelty": v2, "energy": v3, "centroid": v4,
                 "vocal": v5}
    res.vote_total = total
    res.climax_bar = climax_bar
    res.climax_peaks = sorted(peaks)
    res.weights["vote"] = vw
    res.weights["instrumental"] = bool(instrumental)
    return res


# ---------------- 强度 → 建议密度 ----------------


def notes_per_bar_for_level(level: float | None) -> tuple[float, str]:
    """定数 → 官方均值 note/小节（**v0.3 起只作对照列**，主锚点见 `nps_for_level`）。"""
    if level is None:
        return DEFAULT_NOTES_PER_BAR, "无 --level，用全库量级 9.0 note/小节"
    key = min(LEVEL_NOTES_PER_BAR, key=lambda k: abs(k - float(level)))
    return LEVEL_NOTES_PER_BAR[key], f"定数 {key} 的官方均值（密度曲线报告 §2.4）"


def nps_for_level(level: float | None) -> tuple[float | None, str]:
    """定数 → 官方均值 **NPS**（知识 031 §7）。这是 v0.3 的主锚点。"""
    if level is None:
        return None, "无 --level → 无 NPS 锚，退回 note/小节 全库量级"
    key = min(LEVEL_NPS, key=lambda k: abs(k - float(level)))
    return LEVEL_NPS[key], f"定数 {key} 的官方均值 NPS {LEVEL_NPS[key]}（知识 031 §7）"


def notes_per_bar_from_nps(nps: float, bar_seconds) -> np.ndarray | float:
    """NPS → note/小节：`NPS × 每小节秒数`（变速曲逐小节各算各的）。

    标定报告 §4.2：换成 NPS 锚后 MAPE 从 21.0% 降到 16.5%，BPM ≥ 200 的三首
    （激唱 / 幻想 / 麒麟）不再被系统性高估 3.1–3.7 note/小节。
    """
    return float(nps) * np.asarray(bar_seconds, dtype=float)


def apply_structural_cap(segments, bar_suggested: np.ndarray,
                         cap_ratio: float = INTRO_OUTRO_CAP_RATIO,
                         functions: tuple[str, ...] = STRUCTURAL_CAP_FUNCTIONS,
                         ) -> tuple[list, np.ndarray, dict]:
    """**intro/outro 的密度由结构封顶，不跟能量曲线走。**

    上限 = `cap_ratio × 全曲建议 note/小节 的均值`。段落级与逐小节级同时封顶。
    依据：MYTHOS 个案（音频强度 ≈ 1.0 的前奏，官方谱只放 4 note/小节）。
    """
    bar_suggested = np.asarray(bar_suggested, dtype=float).copy()
    if bar_suggested.size == 0:
        return list(segments), bar_suggested, {"applied": False, "reason": "无逐小节建议密度"}
    cap = float(cap_ratio) * float(np.mean(bar_suggested))
    capped_segments: list[str] = []
    n_bars_capped = 0
    for s in segments:
        if (getattr(s, "function", "") or "") not in functions:
            continue
        lo = max(0, s.start_bar - 1)
        hi = min(len(bar_suggested), s.end_bar)
        hit = bar_suggested[lo:hi] > cap
        n_bars_capped += int(np.count_nonzero(hit))
        bar_suggested[lo:hi] = np.minimum(bar_suggested[lo:hi], cap)
        if getattr(s, "suggested_notes_per_bar", 0.0) > cap:
            s.suggested_notes_per_bar = cap
            s.notes.append(
                f"**结构封顶**：{s.function} 的建议密度按结构压到 ≤ {cap:.2f} note/小节"
                f"（= 全曲建议均值 × {cap_ratio:.2f}）——前奏/尾奏的密度基线由结构决定，"
                f"**不跟能量曲线走**（MYTHOS 个案：音频强度≈1.0 的前奏官方只放 4 note/小节）")
            capped_segments.append(f"{s.start_bar}–{s.end_bar} {s.function}")
    return list(segments), bar_suggested, {
        "applied": bool(capped_segments or n_bars_capped),
        "cap_ratio": float(cap_ratio),
        "cap_notes_per_bar": round(cap, 3),
        "segments": capped_segments,
        "bars_capped": n_bars_capped,
        "functions": list(functions),
    }


def density_map(intensity: np.ndarray, floor: float, notes_per_bar: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """`density_norm = floor + (1−floor)·I`，再乘官方均值 note/小节。

    返回 (density_norm, 建议 note/小节)。**均为初值**——真正的标定需要
    "官方音频 + 官方谱"的配对数据（v2 §3.4(E)），目前没有。
    """
    I = np.clip(np.asarray(intensity, dtype=float), 0.0, 1.0)
    dn = floor + (1.0 - floor) * I
    # `notes_per_bar` 可以是标量，也可以是**逐小节数组**（NPS 锚在变速曲上逐小节不同）
    return dn, dn * np.asarray(notes_per_bar, dtype=float)


def total_notes_range(level: float | None) -> tuple[float, float, float] | None:
    """定数 → note 总数（均值, p10, p90），知识 004。"""
    if level is None:
        return None
    key = min(LEVEL_TOTAL_NOTES, key=lambda k: abs(k - float(level)))
    return LEVEL_TOTAL_NOTES[key]
