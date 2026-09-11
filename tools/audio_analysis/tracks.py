"""四条逻辑轨的构造与网格串渲染（调研 v2 §5.2 约束 c/d）。

v0.1 的 song sheet 铺了 6 条轨（kick/snare/hat/vocal/bass/other），超过 v2 §5.2(d)
「轨数上限 4」的约束——轨越多越诱导 LLM 采密，违反知识 005「删到不能再删」。
v0.2 压成 **4 条**：

| 轨 | 内容 | 字符含义 |
|----|------|----------|
| `drum` | drums stem | `X`=kick（重音骨架）、`x`=snare/其他鼓件 |
| `vocal` | vocals stem | `X`=有音高 onset 且强、`x`=onset、`-`=延音持续（VAD 有声但无新 onset） |
| `bass` | bass stem | `X`=强 onset、`x`=onset |
| `hook` | other stem（吉他/合成器/riff） | `X`=强 onset、`x`=onset |

hihat **不单独给串**，只在逐小节表里给一个计数列（它在密集段里与 kick 的串常常
完全一样，铺出来是纯噪声）。字符集严格限定 `X x - .`。
"""

from __future__ import annotations

import numpy as np

from .quantize import BarPattern, render_pattern

# 逻辑轨 → 来源 stem
TRACK_SOURCE = {"drum": "drums", "vocal": "vocals", "bass": "bass", "hook": "other"}
TRACK_ORDER = ("drum", "vocal", "bass", "hook")

CHARSET = ("X", "x", "-", ".")


def _slot_windows(grid, bar: int, division: int) -> tuple[np.ndarray, np.ndarray]:
    """该小节按 division 等分后每个格子的 [起, 止) 秒数。"""
    start = grid.bar_start(bar)
    step = grid.bar_duration(bar) / float(division)
    t0 = start + np.arange(division, dtype=float) * step
    return t0, t0 + step


def _sustain_mask(grid, bar: int, division: int,
                  vad_active: np.ndarray, vad_times: np.ndarray,
                  threshold: float = 0.5) -> np.ndarray:
    """逐格的人声"有声"掩码（该格窗口内 VAD 激活比例 > threshold）。"""
    out = np.zeros(division, dtype=bool)
    if vad_active is None or np.size(vad_active) == 0:
        return out
    t0, t1 = _slot_windows(grid, bar, division)
    for s in range(division):
        sel = (vad_times >= t0[s]) & (vad_times < t1[s])
        if sel.any():
            out[s] = float(np.mean(vad_active[sel])) > threshold
    return out


def build_drum_chars(n: int, kick_mask: np.ndarray | None) -> list[str]:
    """drums onset → 字符：kick 记 `X`，其余鼓件记 `x`。"""
    if kick_mask is None:
        return ["x"] * n
    return ["X" if bool(kick_mask[i]) else "x" for i in range(n)]


def build_strong_chars(n: int, strong_mask: np.ndarray | None) -> list[str]:
    """通用轨：强 onset（≥ 本轨 P70）记 `X`，其余 `x`。"""
    if strong_mask is None:
        return ["x"] * n
    return ["X" if bool(strong_mask[i]) else "x" for i in range(n)]


def build_vocal_chars(n: int, strong_mask: np.ndarray | None,
                      pitched_mask: np.ndarray | None) -> list[str]:
    """人声轨：**有音高且强** 记 `X`，其余 onset 记 `x`。"""
    out = []
    for i in range(n):
        strong = bool(strong_mask[i]) if strong_mask is not None and i < len(strong_mask) else False
        pitched = bool(pitched_mask[i]) if pitched_mask is not None and i < len(pitched_mask) else False
        out.append("X" if (strong and pitched) else "x")
    return out


def build_track_patterns(
    grid,
    bar_div,
    onset_times: dict[str, np.ndarray],
    per_track_slots: dict[str, dict[int, list[int]]],
    char_masks: dict[str, dict],
    vad: dict | None = None,
) -> dict[str, dict[int, BarPattern]]:
    """渲染四条逻辑轨的逐小节网格串。

    参数：
        char_masks: {stem: {"strong": mask, "kick": mask, "pitched": mask}}
        vad: `onsets.vocal_activity` 的返回（给 vocal 轨的 `-` 延音用）
    """
    out: dict[str, dict[int, BarPattern]] = {t: {} for t in TRACK_ORDER}
    vad_active = np.asarray(vad.get("active")) if vad else np.zeros(0, dtype=bool)
    vad_times = np.asarray(vad.get("times")) if vad else np.zeros(0)

    for track in TRACK_ORDER:
        stem = TRACK_SOURCE[track]
        times = np.atleast_1d(np.asarray(onset_times.get(stem, np.zeros(0)), dtype=float))
        slots = per_track_slots.get(stem, {})
        masks = char_masks.get(stem, {})
        for bar in range(1, grid.n_bars + 1):
            d = bar_div[bar].division
            ids = slots.get(bar, [])
            # 该小节该轨的 onset 在全曲数组中的下标（assign 的顺序即 times 顺序）
            order = _bar_onset_indices(times, grid, bar)
            n = len(order)
            if track == "drum":
                chars = build_drum_chars(n, _sub(masks.get("kick"), order))
            elif track == "vocal":
                chars = build_vocal_chars(n, _sub(masks.get("strong"), order),
                                          _sub(masks.get("pitched"), order))
            else:
                chars = build_strong_chars(n, _sub(masks.get("strong"), order))
            sustain = None
            if track == "vocal" and vad_active.size:
                sustain = _sustain_mask(grid, bar, d, vad_active, vad_times)
            pat = render_pattern(d, np.asarray(ids, dtype=int), chars, sustain)
            errs = []
            if n:
                step = grid.bar_duration(bar) / d
                rel = times[order] - grid.bar_start(bar)
                errs = [float((rel[i] - ids[i] * step) * 1000.0) for i in range(min(n, len(ids)))]
            out[track][bar] = BarPattern(bar=bar, track=track, division=d, pattern=pat,
                                         n_onsets=n, errors_ms=errs,
                                         raw_times=[round(float(times[i]), 4) for i in order])
    return out


def _sub(mask, order):
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.size == 0:
        return None
    return m[order] if len(order) else m[:0]


def _bar_onset_indices(times: np.ndarray, grid, bar: int) -> np.ndarray:
    """与 `quantize.assign_onsets_to_bars` 完全一致的归属规则（含末尾吸附）。"""
    from .quantize import tau_ms

    if times.size == 0:
        return np.zeros(0, dtype=int)
    out = []
    for i, t in enumerate(times):
        b = grid.bar_of(float(t))
        if b <= 0:
            continue
        bd = grid.bar_duration(b)
        tol = tau_ms(8, bd * 1000.0) / 1000.0
        if (grid.bar_start(b) + bd) - float(t) <= tol and b + 1 <= grid.n_bars:
            b += 1
        if b == bar:
            out.append(i)
    return np.asarray(out, dtype=int)


def pitched_mask(y: np.ndarray, times: np.ndarray, sr: int,
                 fmin: float = 65.0, fmax: float = 1200.0) -> np.ndarray:
    """人声 onset 是否"有音高"：onset 后 120 ms 窗内 pYIN/自相关判定为浊音。

    basic-pitch 未装上（见 README §5 装包清单），这里用 librosa 的 `yin` +
    谐波能量比做零模型替代——只区分"有明确音高" vs "气声/辅音/打击噪声"，
    不追求音高精度。
    """
    import librosa

    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.size == 0 or y is None or np.size(y) == 0:
        return np.zeros(times.size, dtype=bool)
    win = int(round(0.12 * sr))
    out = np.zeros(times.size, dtype=bool)
    for i, t in enumerate(times):
        a = int(round(float(t) * sr))
        b = min(len(y), a + win)
        a = max(0, min(a, len(y) - 1))
        if b - a < 512:
            continue
        seg = y[a:b]
        if float(np.max(np.abs(seg))) < 1e-4:
            continue
        try:
            harm = librosa.effects.harmonic(seg, margin=2.0)
        except Exception:
            continue
        ratio = float(np.sum(harm ** 2) / (np.sum(seg ** 2) + 1e-12))
        if ratio < 0.35:
            continue
        try:
            f0 = librosa.yin(seg, fmin=fmin, fmax=fmax, sr=sr,
                             frame_length=min(2048, len(seg)))
        except Exception:
            continue
        f0 = f0[np.isfinite(f0)]
        if f0.size == 0:
            continue
        # 音高稳定（帧间相对波动小）才算"有音高"
        med = float(np.median(f0))
        if med <= 0:
            continue
        jitter = float(np.median(np.abs(f0 - med)) / med)
        out[i] = jitter < 0.12
    return out
