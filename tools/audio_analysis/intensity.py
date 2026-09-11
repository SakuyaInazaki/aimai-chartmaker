"""逐小节强度曲线与高潮定位。

严格按 `docs/audio-analysis.md` §4「强度曲线配方」实现（权重全部做成参数）：

1. pyloudnorm 归一到 -14 LUFS（防"谁响谁高潮"）
2. 同 hop 提取 rms / onset_strength / spectral_centroid / drums stem RMS
3. 逐特征 5–95 百分位截断 → min-max 归一
4. 加权融合 intensity = 0.35*rms + 0.30*onset + 0.20*cent + 0.15*drums
5. 逐小节聚合（均值 / P75）
6. 平滑：小节级中值滤波（窗≈2 小节）+ 高斯(σ≈1)
7. 高潮四票投票：结构 0.4 / novelty 0.3 / 能量 0.2 / 质心 0.1

第 8 步（模板 T=[0.20,0.45,0.30,1.00,0.10] 的 DTW 对齐）本原型未实现，
段落强度档位直接由第 5–6 步的曲线给出。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

HOP = 512
ANALYSIS_SR = 22050

DEFAULT_FUSION_WEIGHTS = {"rms": 0.35, "onset": 0.30, "centroid": 0.20, "drums": 0.15}
DEFAULT_VOTE_WEIGHTS = {"structure": 0.4, "novelty": 0.3, "energy": 0.2, "centroid": 0.1}
TARGET_LUFS = -14.0


def loudness_normalize(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS) -> tuple[np.ndarray, dict]:
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


def _clip_minmax(x: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """5–95 百分位截断 + min-max 归一。"""
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _agg_per_bar(values: np.ndarray, times: np.ndarray, grid,
                 stat: str = "mean") -> np.ndarray:
    out = np.zeros(grid.n_bars, dtype=float)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        t1 = t0 + grid.bar_duration(bar)
        sel = (times >= t0) & (times < t1)
        if not sel.any():
            continue
        v = values[sel]
        out[bar - 1] = float(np.percentile(v, 75)) if stat == "p75" else float(v.mean())
    return out


@dataclass
class IntensityResult:
    bar_intensity: np.ndarray
    bar_raw: np.ndarray
    components: dict[str, np.ndarray]
    votes: dict[str, np.ndarray]
    vote_total: np.ndarray
    climax_bar: int
    climax_peaks: list[int]
    loudness_info: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)


def compute_intensity(
    y_mix: np.ndarray,
    sr: int,
    grid,
    y_drums: np.ndarray | None = None,
    fusion_weights: dict | None = None,
    vote_weights: dict | None = None,
    aggregate: str = "mean",
    smooth_median_bars: int = 3,
    smooth_sigma: float = 1.0,
) -> IntensityResult:
    """执行强度曲线配方的第 1–6 步（高潮投票见 `vote_climax`）。"""
    import librosa
    from scipy.ndimage import gaussian_filter1d, median_filter

    fw = dict(DEFAULT_FUSION_WEIGHTS)
    if fusion_weights:
        fw.update(fusion_weights)

    # 步骤 1
    y_norm, loud = loudness_normalize(y_mix, sr)

    # 步骤 2
    rms = librosa.feature.rms(y=y_norm, hop_length=HOP)[0]
    onset = librosa.onset.onset_strength(y=y_norm, sr=sr, hop_length=HOP)
    cent = librosa.feature.spectral_centroid(y=y_norm, sr=sr, hop_length=HOP)[0]
    n = min(len(rms), len(onset), len(cent))
    rms, onset, cent = rms[:n], onset[:n], cent[:n]
    ft = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP)

    if y_drums is not None and y_drums.size:
        drums = librosa.feature.rms(y=y_drums, hop_length=HOP)[0]
        drums = np.interp(ft, librosa.frames_to_time(np.arange(len(drums)), sr=sr,
                                                     hop_length=HOP), drums)
    else:
        drums = np.zeros(n)
        fw = {k: v for k, v in fw.items() if k != "drums"}

    # 步骤 3
    comp_frame = {
        "rms": _clip_minmax(rms),
        "onset": _clip_minmax(onset),
        "centroid": _clip_minmax(cent),
        "drums": _clip_minmax(drums) if drums.any() else drums,
    }

    # 步骤 4
    wsum = sum(fw.values()) or 1.0
    fused = sum(fw.get(k, 0.0) * comp_frame[k] for k in comp_frame) / wsum

    # 步骤 5
    bar_raw = _agg_per_bar(fused, ft, grid, stat=aggregate)
    components_bar = {k: _agg_per_bar(v, ft, grid, stat=aggregate)
                      for k, v in comp_frame.items()}

    # 步骤 6
    win = max(1, int(smooth_median_bars) | 1)
    sm = median_filter(bar_raw, size=win, mode="nearest")
    sm = gaussian_filter1d(sm, sigma=smooth_sigma, mode="nearest")
    lo, hi = float(sm.min()), float(sm.max())
    bar_intensity = (sm - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(sm)

    return IntensityResult(
        bar_intensity=bar_intensity,
        bar_raw=bar_raw,
        components=components_bar,
        votes={},
        vote_total=np.zeros(grid.n_bars),
        climax_bar=0,
        climax_peaks=[],
        loudness_info=loud,
        weights={"fusion": fw, "aggregate": aggregate},
    )


def vote_climax(
    res: IntensityResult,
    segments,
    grid,
    vote_weights: dict | None = None,
    top_k: int = 3,
) -> IntensityResult:
    """步骤 7：四票组合投票定高潮。"""
    vw = dict(DEFAULT_VOTE_WEIGHTS)
    if vote_weights:
        vw.update(vote_weights)
    n = grid.n_bars

    # V1 结构票：chorus/重复段内 = 1
    v1 = np.zeros(n)
    for s in segments:
        is_chorus = ("chorus" in (s.function or "").lower()) or s.is_repeat
        if is_chorus:
            v1[s.start_bar - 1: s.end_bar] = 1.0

    # V2 novelty 票：onset 分量局部峰 ±1 小节三角窗
    on = res.components.get("onset", np.zeros(n))
    v2 = np.zeros(n)
    if on.size == n and n >= 3:
        for i in range(1, n - 1):
            if on[i] >= on[i - 1] and on[i] >= on[i + 1] and on[i] > np.percentile(on, 70):
                v2[i] += 1.0
                v2[i - 1] = max(v2[i - 1], 0.5)
                v2[i + 1] = max(v2[i + 1], 0.5)

    # V3 能量票：rms_n > 0.75 且局部极大
    rms = res.components.get("rms", np.zeros(n))
    v3 = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        if rms[i] > 0.75 and rms[i] >= rms[lo:hi].max() - 1e-9:
            v3[i] = 1.0

    # V4 质心票：cent_n > 0.7
    cent = res.components.get("centroid", np.zeros(n))
    v4 = (cent > 0.7).astype(float)

    total = (vw["structure"] * v1 + vw["novelty"] * v2
             + vw["energy"] * v3 + vw["centroid"] * v4)
    # 与强度曲线相乘做最终排序（避免选到"投票高但整体很弱"的小节）
    ranked = total * (0.5 + 0.5 * res.bar_intensity)

    climax_bar = int(np.argmax(ranked)) + 1 if n else 0
    peaks: list[int] = []
    order = np.argsort(ranked)[::-1]
    for i in order:
        if len(peaks) >= top_k:
            break
        if all(abs(int(i) - (p - 1)) >= 8 for p in peaks):
            peaks.append(int(i) + 1)

    res.votes = {"structure": v1, "novelty": v2, "energy": v3, "centroid": v4}
    res.vote_total = total
    res.climax_bar = climax_bar
    res.climax_peaks = sorted(peaks)
    res.weights["vote"] = vw
    return res
