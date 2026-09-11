"""逐小节特征（调研 v2 §4.1 清单）。

这些特征是"采什么音（切轨）"与"段落类型判别"的可计算代理，全部写进
`song_analysis.json` 的 `bars[]`，供 L4 规划层与 LLM 复核：

| 特征 | 含义 |
|------|------|
| `share_s` | stem s 在该小节的能量占比 —— "哪个响" |
| `n_onset_s` | stem s 的 onset 数 —— 可踩音上限 |
| `grid_fit_s` | stem s 的 onset 落在 `{8}` 网格上的比例 —— 律动规整度 |
| `voiced_ratio` | vocal stem 有声帧占比 —— 切轨最强信号 |
| `kick/snare/hihat` | drums 三带 onset 数（频带启发式） |
| `riff_sim` | other stem 的 chroma 小节向量与前 4/8 小节的余弦相似 |
| `sil_run` | 该小节内最长静默秒数 —— 休息段/留白 |
"""

from __future__ import annotations

import numpy as np

from .quantize import tau_ms

SILENCE_REL_DB = 40.0   # 相对全曲峰值多少 dB 以下算静默


def energy_per_bar(y: np.ndarray, sr: int, grid) -> np.ndarray:
    """逐小节能量（平方和）。"""
    out = np.zeros(grid.n_bars, dtype=float)
    if y is None or np.size(y) == 0:
        return out
    for bar in range(1, grid.n_bars + 1):
        a = int(round(max(0.0, grid.bar_start(bar)) * sr))
        b = int(round(max(0.0, grid.bar_start(bar) + grid.bar_duration(bar)) * sr))
        a, b = max(0, min(a, len(y))), max(0, min(b, len(y)))
        if b > a:
            out[bar - 1] = float(np.sum(np.square(y[a:b], dtype=np.float64)))
    return out


def share_per_bar(stem_energy: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """各 stem 的逐小节能量占比（和为 1）。"""
    if not stem_energy:
        return {}
    names = list(stem_energy)
    stack = np.vstack([stem_energy[n] for n in names])
    total = stack.sum(axis=0) + 1e-12
    return {n: stack[i] / total for i, n in enumerate(names)}


def grid_fit_per_bar(times: np.ndarray, grid, division: int = 8) -> np.ndarray:
    """逐小节：该轨 onset 落在 `{division}` 网格容差内的比例（无 onset 记 NaN→0）。"""
    out = np.zeros(grid.n_bars, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.size == 0:
        return out
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        bd = grid.bar_duration(bar)
        sel = times[(times >= t0) & (times < t0 + bd)]
        if sel.size == 0:
            continue
        step = bd / division
        rel = sel - t0
        err = np.abs(rel - np.rint(rel / step) * step) * 1000.0
        out[bar - 1] = float(np.mean(err <= tau_ms(division, bd * 1000.0)))
    return out


def grid_fit_vs_bar_division(times: np.ndarray, grid,
                             bar_division: dict[int, int]) -> np.ndarray:
    """逐小节：该轨 onset 落在**该小节实际选中的分音**上的比例。

    v2 §4.1 的 `grid_fit_s` 定义在 `{8}` 上，用来识别"采样/长音"这类不规整的轨。
    但 16 分 riff 在 `{8}` 上天然只有 ~0.5 的命中率——拿它当"这条轨能不能当骨架"
    的判据会把所有密集器乐轨都判成不可用（实测 TransientTears 上 other 轨被连续
    误降级）。所以切轨的降级判据改用**这一版**（对齐到该小节真正要写的网格），
    `{8}` 那一版仍按 v2 原样输出，供结构判别使用。
    """
    out = np.zeros(grid.n_bars, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.size == 0:
        return out
    for bar in range(1, grid.n_bars + 1):
        d = int(bar_division.get(bar, 8))
        t0 = grid.bar_start(bar)
        bd = grid.bar_duration(bar)
        sel = times[(times >= t0) & (times < t0 + bd)]
        if sel.size == 0 or d <= 0:
            continue
        step = bd / d
        rel = sel - t0
        err = np.abs(rel - np.rint(rel / step) * step) * 1000.0
        out[bar - 1] = float(np.mean(err <= tau_ms(d, bd * 1000.0)))
    return out


def riff_similarity(y_other: np.ndarray, sr: int, grid,
                    lags: tuple[int, ...] = (4, 8)) -> dict[str, np.ndarray]:
    """other stem 的逐小节 chroma 向量与前 `lag` 小节的余弦相似度。"""
    out = {f"riff_sim_{l}": np.zeros(grid.n_bars, dtype=float) for l in lags}
    if y_other is None or np.size(y_other) == 0:
        return out
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y_other, sr=sr, hop_length=hop)
    ft = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    V = np.zeros((grid.n_bars, chroma.shape[0]))
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        sel = (ft >= t0) & (ft < t0 + grid.bar_duration(bar))
        if sel.any():
            V[bar - 1] = np.median(chroma[:, sel], axis=1)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    for l in lags:
        arr = out[f"riff_sim_{l}"]
        for b in range(l, grid.n_bars):
            arr[b] = float(np.dot(Vn[b], Vn[b - l]))
    return out


def silence_runs(y: np.ndarray, sr: int, grid, rel_db: float = SILENCE_REL_DB,
                 frame: int = 1024, hop: int = 512) -> np.ndarray:
    """逐小节最长静默时长（秒）。"""
    out = np.zeros(grid.n_bars, dtype=float)
    if y is None or np.size(y) == 0:
        return out
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    if rms.size == 0 or float(np.max(rms)) <= 0:
        return out
    db = librosa.amplitude_to_db(rms, ref=np.max)
    quiet = db < -abs(rel_db)
    ft = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    dt = hop / float(sr)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        sel = (ft >= t0) & (ft < t0 + grid.bar_duration(bar))
        if not sel.any():
            continue
        q = quiet[sel]
        best = run = 0
        for v in q:
            run = run + 1 if v else 0
            best = max(best, run)
        out[bar - 1] = best * dt
    return out


def merged_onset_count(onset_times: dict[str, np.ndarray], grid,
                       keys=("drums", "bass", "other", "vocals"),
                       merge_ms: float = 30.0) -> np.ndarray:
    """各 stem onset 合并去重后的逐小节计数 —— "这一小节最多能踩几个音"。

    v2 §3.1 指出 v1 用混音 onset_strength 会被最响音轨支配（正是 MMFC 批评的
    "哪个响踩哪个"），应改用去重合并的 onset **计数**。
    """
    allt = np.concatenate([np.atleast_1d(np.asarray(onset_times.get(k, np.zeros(0)),
                                                    dtype=float)) for k in keys]) \
        if keys else np.zeros(0)
    out = np.zeros(grid.n_bars, dtype=float)
    if allt.size == 0:
        return out
    allt = np.sort(allt)
    keep = [allt[0]]
    thr = merge_ms / 1000.0
    for t in allt[1:]:
        if t - keep[-1] > thr:
            keep.append(float(t))
    keep = np.asarray(keep)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        out[bar - 1] = float(np.sum((keep >= t0) & (keep < t0 + grid.bar_duration(bar))))
    return out


def spectral_flux_per_bar(y: np.ndarray, sr: int, grid, hop: int = 512) -> np.ndarray:
    """逐小节谱通量均值（瞬态密度）。"""
    out = np.zeros(grid.n_bars, dtype=float)
    if y is None or np.size(y) == 0:
        return out
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    flux = np.sqrt(np.sum(np.square(np.maximum(0.0, np.diff(S, axis=1))), axis=0))
    ft = librosa.frames_to_time(np.arange(len(flux)) + 1, sr=sr, hop_length=hop)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        sel = (ft >= t0) & (ft < t0 + grid.bar_duration(bar))
        if sel.any():
            out[bar - 1] = float(np.mean(flux[sel]))
    return out


def high_freq_ratio(y: np.ndarray, sr: int, grid, lo: float = 6000.0,
                    hop: int = 512) -> np.ndarray:
    """逐小节高频（>lo Hz）能量占比 —— riser / hihat 上升的代理（build 判据）。"""
    out = np.zeros(grid.n_bars, dtype=float)
    if y is None or np.size(y) == 0:
        return out
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hi = S[freqs >= lo, :].sum(axis=0)
    tot = S.sum(axis=0) + 1e-12
    r = hi / tot
    ft = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        sel = (ft >= t0) & (ft < t0 + grid.bar_duration(bar))
        if sel.any():
            out[bar - 1] = float(np.mean(r[sel]))
    return out


#: 参与"能量占比 / 落格率"统计的 stem（v0.5：六路分离时自动带上 guitar/piano）
BASE_STEMS = ("drums", "bass", "other", "vocals")
EXTRA_STEMS_6S = ("guitar", "piano")
#: v0.5 新增的派生轨（不是 Demucs stem，是本项目算出来的）
DERIVED_TRACKS = ("melody", "fx")


def build_bar_features(grid, stem_audio: dict[str, np.ndarray], sr: int,
                       onset_times: dict[str, np.ndarray],
                       vocal_act: np.ndarray,
                       y_mix: np.ndarray | None = None,
                       bar_division: dict[int, int] | None = None,
                       pitch_notes: dict | None = None) -> dict[str, np.ndarray]:
    """汇总 v2 §4.1 的逐小节特征表（v0.5 扩到六路 stem + 音高 note 轨 + fx 轨）。

    参数：
        pitch_notes: {stem 名: `pitch_notes.PitchNotes`}，来自 basic-pitch。
            给了就额外产出 `n_pnote_<stem>` / `pitched_cov_<stem>`，并把
            **`vocal_pitched_ratio`**（人声有音高 note 的时间覆盖率）作为
            能量 VAD `voiced_ratio` 的替代品输出。
    """
    feats: dict[str, np.ndarray] = {}
    energies = {k: energy_per_bar(v, sr, grid) for k, v in stem_audio.items()}
    for name, arr in share_per_bar(energies).items():
        feats[f"share_{name}"] = arr

    counted = tuple(BASE_STEMS) + EXTRA_STEMS_6S + ("kick", "snare", "hihat") \
        + DERIVED_TRACKS
    fitted = tuple(BASE_STEMS) + EXTRA_STEMS_6S + DERIVED_TRACKS
    for name in counted:
        t = onset_times.get(name)
        if t is None:
            continue
        t = np.atleast_1d(np.asarray(t, dtype=float))
        feats[f"n_onset_{name}"] = _count_per_bar(t, grid)
        if name in fitted:
            feats[f"grid_fit_{name}"] = grid_fit_per_bar(t, grid, division=8)
            if bar_division:
                feats[f"grid_fit_bar_{name}"] = grid_fit_vs_bar_division(
                    t, grid, bar_division)

    # ---- 有音高 note（basic-pitch）----
    if pitch_notes:
        from . import pitch_notes as pn_mod

        for stem, notes in pitch_notes.items():
            if notes is None or not getattr(notes, "available", False):
                continue
            feats[f"n_pnote_{stem}"] = pn_mod.notes_per_bar(notes, grid)
            feats[f"pitched_cov_{stem}"] = pn_mod.pitched_coverage_per_bar(notes, grid)
        vocals_notes = pitch_notes.get("vocals")
        if vocals_notes is not None and getattr(vocals_notes, "available", False):
            # **人声活动的新口径**（设计文档 §3.4 / §9-A.3 的坏探测器替换件）：
            # 能量 VAD 用相对 dB 阈值，vocals 轨近乎静音时底噪也超阈 ——
            # n=40 实测把 10/12 首器乐曲判成人声曲。改用"有音高 note 的时间覆盖率"，
            # 纯泄漏/底噪不会被 basic-pitch 判成 note，没有这个失败模式。
            feats["vocal_pitched_ratio"] = feats["pitched_cov_vocals"]

    feats["voiced_ratio"] = np.asarray(vocal_act, dtype=float)   # 兼容字段（已知不可靠）
    feats["n_onset_merged"] = merged_onset_count(
        onset_times, grid, keys=tuple(k for k in BASE_STEMS if k in onset_times))
    feats.update(riff_similarity(stem_audio.get("other"), sr, grid))
    feats["sil_run"] = silence_runs(y_mix if y_mix is not None else stem_audio.get("other"),
                                    sr, grid)
    feats["hf_ratio"] = high_freq_ratio(y_mix if y_mix is not None else stem_audio.get("other"),
                                        sr, grid)
    return feats


def _count_per_bar(t: np.ndarray, grid) -> np.ndarray:
    counts = np.zeros(grid.n_bars, dtype=float)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        counts[bar - 1] = float(np.sum((t >= t0) & (t < t0 + grid.bar_duration(bar))))
    return counts
