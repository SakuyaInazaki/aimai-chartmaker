"""逐 stem onset 检测 / 鼓件频带启发式分解 / 人声活动度（VAD）。

对应设计文档 §3：每个 stem 的 onset 流就是踩音候选池。

⚠️ 鼓件分解（kick/snare/hihat）是**频带能量启发式**，不是训练出来的鼓转录模型
（设计文档 §7.1 提到的 MDX23C-DrumSep 权重许可待复核，故不进默认管线）。
分件结果只能当"倾向性提示"用，不能当作准确的鼓谱。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ANALYSIS_SR = 22050
HOP = 256  # 11.6ms @22050，足以分辨 192BPM 的 1/32 音（39ms）
N_FFT = 1024

# 逐 stem 的 peak-pick 参数（delta 越大越保守；wait 为最小间隔帧数）
STEM_ONSET_PARAMS: dict[str, dict] = {
    # 鼓：瞬态最干净，可以松一点、允许密集
    "drums": dict(delta=0.06, wait=2, pre_max=3, post_max=3, pre_avg=10, post_avg=10),
    # 贝斯：低频起音钝，需要更保守 + 更长间隔，否则一个音会检出好几次
    "bass": dict(delta=0.12, wait=5, pre_max=4, post_max=4, pre_avg=14, post_avg=14),
    # other（吉他/合成器/键盘）：介于两者之间
    "other": dict(delta=0.08, wait=3, pre_max=3, post_max=3, pre_avg=12, post_avg=12),
    # 人声：连奏多、颤音会误触发，最保守
    "vocals": dict(delta=0.14, wait=6, pre_max=4, post_max=4, pre_avg=16, post_avg=16),
    "_default": dict(delta=0.08, wait=3, pre_max=3, post_max=3, pre_avg=12, post_avg=12),
}

# 鼓件频带（Hz）——启发式
DRUM_BANDS = {
    "kick": (20.0, 150.0),
    "snare": (150.0, 800.0),
    "hihat": (5000.0, 11025.0),
}
SNARE_BROADBAND = (2000.0, 8000.0)  # 军鼓的宽带噪声成分


@dataclass
class OnsetTrack:
    """一条 onset 流。"""

    name: str
    times: np.ndarray            # 秒
    strengths: np.ndarray        # 该 onset 处的 onset_strength 值
    env: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    env_times: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    heuristic: bool = False      # 是否为启发式产物（鼓件分解）

    @property
    def count(self) -> int:
        return int(len(self.times))

    def strong_mask(self, quantile: float = 0.7) -> np.ndarray:
        """强 onset 掩码：强度 ≥ 本条流的 `quantile` 分位。"""
        if self.count == 0:
            return np.zeros(0, dtype=bool)
        thr = float(np.quantile(self.strengths, quantile))
        return self.strengths >= thr


def frame_times(n_frames: int, sr: int = ANALYSIS_SR, hop: int = HOP) -> np.ndarray:
    import librosa

    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)


def onset_envelope(y: np.ndarray, sr: int = ANALYSIS_SR, hop: int = HOP) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)


def detect_onsets(
    y: np.ndarray,
    name: str,
    sr: int = ANALYSIS_SR,
    hop: int = HOP,
    backtrack: bool = True,
    params: dict | None = None,
) -> OnsetTrack:
    """对单个 stem 做 onset 检测（librosa，backtrack 回退到能量上升起点）。"""
    import librosa

    env = onset_envelope(y, sr=sr, hop=hop)
    cfg = dict(STEM_ONSET_PARAMS.get(name, STEM_ONSET_PARAMS["_default"]))
    if params:
        cfg.update(params)
    if env.size == 0 or float(np.max(env)) <= 0:
        return OnsetTrack(name, np.zeros(0), np.zeros(0), env, frame_times(len(env), sr, hop))

    env_n = env / (float(np.max(env)) + 1e-12)
    frames = librosa.onset.onset_detect(
        onset_envelope=env_n, sr=sr, hop_length=hop,
        backtrack=backtrack, units="frames", normalize=False, **cfg,
    )
    frames = np.asarray(frames, dtype=int)
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    # backtrack 后的帧能量可能很低 → 强度取回溯前后一个小窗的最大值
    strengths = np.array([
        float(env_n[max(0, f - 1): min(len(env_n), f + 4)].max()) if len(env_n) else 0.0
        for f in frames
    ])
    return OnsetTrack(name, times, strengths, env_n, frame_times(len(env), sr, hop))


def _band_energy(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """给定频带内的逐帧幅度和。"""
    sel = (freqs >= lo) & (freqs < hi)
    if not sel.any():
        return np.zeros(S.shape[1])
    return S[sel, :].sum(axis=0)


def drum_components(
    y: np.ndarray,
    drums_track: OnsetTrack | None = None,
    sr: int = ANALYSIS_SR,
    hop: int = HOP,
    n_fft: int = N_FFT,
    window_sec: float = 0.045,
    kick_thr: float = 0.30,
    snare_thr: float = 0.30,
    hihat_thr: float = 0.28,
) -> dict[str, OnsetTrack]:
    """鼓 stem 的频带启发式分件：kick / snare / hihat。

    做法：**先在 drums stem 上检出 onset，再对每个 onset 按频带能量分类**
    （而不是各频带独立检 onset）——这样分件结果必然是 drums onset 的子集，
    不会出现"kick 数量比整条鼓轨还多"的荒谬情况。一个 onset 可以同时带多个
    标签（底鼓与踩镲同时敲是常态）。

    ⚠️ 启发式：底鼓与低音 tom、军鼓与拍手/军鼓边击、hihat 与镲片/齿音泄漏
    都可能混淆；分件只当倾向性提示，不能当鼓谱。
    """
    import librosa

    if drums_track is None:
        drums_track = detect_onsets(y, "drums", sr=sr, hop=hop)
    ft_all = drums_track.env_times
    empty = {n: OnsetTrack(n, np.zeros(0), np.zeros(0), np.zeros(0), ft_all, heuristic=True)
             for n in ("kick", "snare", "hihat")}
    if drums_track.count == 0:
        return empty

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    ft = frame_times(S.shape[1], sr, hop)
    bands = {
        "low": _band_energy(S, freqs, *DRUM_BANDS["kick"]),
        "mid": _band_energy(S, freqs, *DRUM_BANDS["snare"]),
        "high": _band_energy(S, freqs, *DRUM_BANDS["hihat"]),
        "wide": _band_energy(S, freqs, *SNARE_BROADBAND),
    }

    win = max(1, int(round(window_sec * sr / hop)))
    n_on = drums_track.count
    feat = {k: np.zeros(n_on) for k in bands}
    for i, t in enumerate(drums_track.times):
        f0 = int(np.searchsorted(ft, t))
        f1 = min(len(ft), f0 + win)
        f0 = max(0, min(f0, len(ft) - 1))
        if f1 <= f0:
            f1 = f0 + 1
        for k, arr in bands.items():
            feat[k][i] = float(arr[f0:f1].max())

    # 每个频带按自身 90 分位归一 → 阈值对整体音量不敏感
    norm = {}
    for k, v in feat.items():
        ref = float(np.percentile(v, 90)) or 1.0
        norm[k] = np.clip(v / ref, 0.0, 1.5)

    # 相对门：某频带还必须在该 onset 的三条频带里占到一定份额，
    # 否则密集段里每个 onset 都会被同时贴上 kick+snare+hihat 三个标签。
    mx = np.maximum.reduce([norm["low"], norm["mid"], norm["high"]]) + 1e-12
    labels = {
        # 底鼓：低频强，且低频不被中频压过
        "kick": (norm["low"] > kick_thr) & (norm["low"] >= norm["mid"] * 0.8)
        & (norm["low"] >= 0.55 * mx),
        # 军鼓：中低频"鼓体" + 宽带噪声二者都在
        "snare": (norm["mid"] > snare_thr) & (norm["wide"] > snare_thr * 0.8)
        & (norm["mid"] >= 0.50 * mx),
        # 踩镲/镲片：高频强
        "hihat": (norm["high"] > hihat_thr) & (norm["high"] >= 0.55 * mx),
    }

    out: dict[str, OnsetTrack] = {}
    for name, mask in labels.items():
        out[name] = OnsetTrack(
            name,
            drums_track.times[mask],
            drums_track.strengths[mask],
            drums_track.env,
            ft_all,
            heuristic=True,
        )
    return out


def vocal_activity(
    y: np.ndarray,
    sr: int = ANALYSIS_SR,
    hop: int = HOP,
    n_fft: int = 2048,
    rel_db: float = 32.0,
    min_gap_sec: float = 0.12,
    min_seg_sec: float = 0.08,
) -> dict:
    """人声逐帧活动度（简易 VAD：RMS 阈值 + 平滑 + 段落长度过滤）。

    阈值取"整曲 RMS 峰值 - rel_db"，因此对整体音量不敏感，但对分离残留敏感
    （Demucs 的 vocals 轨常混入合成器/和声，会被判为有人声）。
    """
    import librosa
    from scipy.ndimage import median_filter

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    ft = frame_times(len(rms), sr, hop)
    if rms.size == 0 or float(np.max(rms)) <= 0:
        return {"active": np.zeros(0, dtype=bool), "times": ft, "rms": rms,
                "threshold_db": None}

    db = librosa.amplitude_to_db(rms, ref=np.max)
    thr = -abs(rel_db)
    active = db > thr
    # 中值滤波去毛刺（窗 ≈ 60ms）
    win = max(3, int(round(0.06 * sr / hop)) | 1)
    active = median_filter(active.astype(np.uint8), size=win).astype(bool)
    active = _fill_gaps(active, int(round(min_gap_sec * sr / hop)))
    active = _drop_short(active, int(round(min_seg_sec * sr / hop)))
    return {"active": active, "times": ft, "rms": rms, "threshold_db": float(thr)}


def _runs(mask: np.ndarray, value: bool):
    """返回 mask 中所有取值为 value 的连续区间 [start, end)。"""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], (mask == value).view(np.int8), [0]))))
    return list(zip(idx[0::2], idx[1::2]))


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask, False):
        if 0 < s and e < len(mask) and (e - s) <= max_gap:
            out[s:e] = True
    return out


def _drop_short(mask: np.ndarray, min_len: int) -> np.ndarray:
    out = mask.copy()
    for s, e in _runs(mask, True):
        if (e - s) < min_len:
            out[s:e] = False
    return out


def activity_per_bar(active: np.ndarray, times: np.ndarray, grid) -> np.ndarray:
    """把帧级活动掩码聚合成逐小节的活动比例 ∈ [0,1]。"""
    out = np.zeros(grid.n_bars, dtype=float)
    if active.size == 0:
        return out
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        t1 = t0 + grid.bar_duration(bar)
        sel = (times >= t0) & (times < t1)
        if sel.any():
            out[bar - 1] = float(active[sel].mean())
    return out


def onsets_per_bar(track: OnsetTrack, grid) -> np.ndarray:
    """逐小节 onset 计数。"""
    counts = np.zeros(grid.n_bars, dtype=int)
    if track.count == 0:
        return counts
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        t1 = t0 + grid.bar_duration(bar)
        counts[bar - 1] = int(((track.times >= t0) & (track.times < t1)).sum())
    return counts
