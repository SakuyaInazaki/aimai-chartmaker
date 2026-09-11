"""拍网格构造与 offset 校验（纯 numpy，不依赖 librosa）。

本模块**不做节拍追踪**：BPM 与 offset 由用户给定，网格由二者直接构造。
`&first` 语义按 simai：谱面第 1 小节第 1 拍在音频中的秒数（可为负）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass
class BpmChange:
    """变速点：从第 `bar`（1 起）小节开始改用 `bpm`。"""

    bar: int
    bpm: float


@dataclass
class Grid:
    """由 BPM + first 构造的小节/拍网格。

    属性：
        bpm: 初始 BPM（第 1 小节起）
        first: 第 1 小节第 1 拍的音频秒数（可为负）
        beats_per_bar: 每小节拍数（默认 4）
        bpm_changes: 变速点列表（按小节号升序，小节号 > 1）
        duration: 音频总时长（秒），用于确定小节数
    """

    bpm: float
    first: float
    beats_per_bar: int = 4
    bpm_changes: Sequence[BpmChange] = field(default_factory=tuple)
    duration: float = 0.0

    # 派生：每小节的起始秒数与该小节的 BPM
    bar_starts: np.ndarray = field(init=False, repr=False)
    bar_bpms: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError("BPM 必须为正数")
        if self.beats_per_bar <= 0:
            raise ValueError("beats_per_bar 必须为正整数")
        changes = sorted(self.bpm_changes, key=lambda c: c.bar)
        for c in changes:
            if c.bar <= 1:
                raise ValueError("变速点小节号必须 > 1（第 1 小节用 --bpm 指定）")
            if c.bpm <= 0:
                raise ValueError("变速点 BPM 必须为正数")
        self.bpm_changes = tuple(changes)

        starts: list[float] = []
        bpms: list[float] = []
        t = float(self.first)
        cur_bpm = float(self.bpm)
        change_map = {c.bar: c.bpm for c in self.bpm_changes}
        bar = 1
        # 至少铺满到音频结束；上限防御性截断（10 万小节）
        limit = 100000
        while bar <= limit:
            if bar in change_map:
                cur_bpm = float(change_map[bar])
            starts.append(t)
            bpms.append(cur_bpm)
            bar_dur = self.beats_per_bar * 60.0 / cur_bpm
            t += bar_dur
            if self.duration > 0 and t > self.duration:
                break
            if self.duration <= 0 and bar >= 1:
                # 未给时长时只构造 1 小节，调用方应显式给 duration
                break
            bar += 1
        self.bar_starts = np.asarray(starts, dtype=float)
        self.bar_bpms = np.asarray(bpms, dtype=float)

    # ---------- 基本查询 ----------

    @property
    def n_bars(self) -> int:
        return int(len(self.bar_starts))

    @property
    def has_bpm_changes(self) -> bool:
        return len(self.bpm_changes) > 0

    def bar_start(self, bar: int) -> float:
        """第 `bar`（1 起）小节的起始秒数。允许外推到网格之外。"""
        if 1 <= bar <= self.n_bars:
            return float(self.bar_starts[bar - 1])
        if bar < 1:
            # 向前外推（用第 1 小节的 BPM）
            dur = self.beats_per_bar * 60.0 / float(self.bpm)
            return float(self.first + (bar - 1) * dur)
        last = self.n_bars
        dur = self.bar_duration(last)
        return float(self.bar_starts[last - 1] + (bar - last) * dur)

    def bar_duration(self, bar: int) -> float:
        """第 `bar` 小节的时长（秒）。"""
        bpm = self.bar_bpm(bar)
        return self.beats_per_bar * 60.0 / bpm

    def bar_bpm(self, bar: int) -> float:
        if 1 <= bar <= self.n_bars:
            return float(self.bar_bpms[bar - 1])
        if bar < 1:
            return float(self.bpm)
        return float(self.bar_bpms[-1])

    def beat_time(self, bar: int, beat: int) -> float:
        """第 `bar` 小节第 `beat`（0 起）拍的秒数。"""
        return self.bar_start(bar) + beat * 60.0 / self.bar_bpm(bar)

    def subdivision_time(self, bar: int, division: int, index: int) -> float:
        """第 `bar` 小节按 `division` 等分后第 `index`（0 起）个格子的秒数。

        `division` 是"每小节等分数"（16 = 16 分音符 × 4 拍 = 一小节 16 格）。
        """
        if division <= 0:
            raise ValueError("division 必须为正整数")
        return self.bar_start(bar) + index * self.bar_duration(bar) / division

    def subdivision_times(self, bar: int, division: int) -> np.ndarray:
        start = self.bar_start(bar)
        step = self.bar_duration(bar) / division
        return start + np.arange(division, dtype=float) * step

    def bar_of(self, t: float) -> int:
        """秒数所在小节号（1 起）；早于第 1 小节返回 0。"""
        if t < self.bar_starts[0]:
            return 0
        idx = int(np.searchsorted(self.bar_starts, t, side="right"))
        return idx  # searchsorted 返回的是 1-based 小节号

    def time_to_bar_pos(self, t: float) -> tuple[int, float]:
        """秒数 → (小节号, 小节内相对位置 ∈ [0,1))。"""
        bar = self.bar_of(t)
        if bar == 0:
            return 0, 0.0
        start = self.bar_start(bar)
        return bar, (t - start) / self.bar_duration(bar)

    def all_beat_times(self) -> np.ndarray:
        """全曲所有拍点的秒数（含每小节的每一拍）。"""
        out = []
        for bar in range(1, self.n_bars + 1):
            start = self.bar_start(bar)
            spb = 60.0 / self.bar_bpm(bar)
            for b in range(self.beats_per_bar):
                out.append(start + b * spb)
        return np.asarray(out, dtype=float)

    def to_dict(self) -> dict:
        return {
            "bpm": self.bpm,
            "first": self.first,
            "beats_per_bar": self.beats_per_bar,
            "bpm_changes": [{"bar": c.bar, "bpm": c.bpm} for c in self.bpm_changes],
            "duration": self.duration,
            "n_bars": self.n_bars,
            "bar_starts_head": [round(float(x), 4) for x in self.bar_starts[:8]],
        }


def parse_bpm_changes(spec: str | None) -> tuple[BpmChange, ...]:
    """解析 `--bpm-changes "33:180,65:155"` 形式的变速点参数。"""
    if not spec:
        return ()
    out: list[BpmChange] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"变速点格式错误（应为 小节号:BPM）：{chunk!r}")
        bar_s, bpm_s = chunk.split(":", 1)
        out.append(BpmChange(bar=int(bar_s.strip()), bpm=float(bpm_s.strip())))
    return tuple(sorted(out, key=lambda c: c.bar))


# ---------- offset 校验 ----------


def _sample_envelope(env: np.ndarray, times: np.ndarray, frame_times: np.ndarray) -> float:
    """在给定秒数处线性插值采样 onset 包络并求和。"""
    if len(times) == 0:
        return 0.0
    vals = np.interp(times, frame_times, env, left=0.0, right=0.0)
    return float(vals.sum() / len(times))


def check_offset(
    onset_env: np.ndarray,
    frame_times: np.ndarray,
    bpm: float,
    first: float,
    beats_per_bar: int = 4,
    duration: float | None = None,
    search_beats: float = 1.0,
    step: float = 0.005,
    subdivision: int = 1,
    tie_ratio: float = 0.90,
    onset_times: np.ndarray | None = None,
    onset_weights: np.ndarray | None = None,
    tolerance: float = 0.025,
) -> dict:
    """beat-synchronous 对齐搜索：在用户 first 附近找最能解释 onset 的偏移。

    **只报告，不覆盖用户值。**

    两种打分方式：

    - **onset-fit（首选）**：用 backtrack 后的 onset 时间，算"落在拍线容差内"的
      加权命中率。backtrack 已把包络滞后修掉，结论与后续量化用的是同一批时间戳。
    - **envelope（后备）**：直接在 onset 强度包络上按拍采样求和。实现简单，但
      librosa 的 onset_strength 是帧间谱流，峰值比真实瞬态**晚约 10ms**
      （已在合成 click 上实测），因此这条路会系统性地把 first 估大 ~10ms。

    参数：
        onset_env / frame_times: 整曲 onset 强度包络及其帧时间（envelope 路必需）
        bpm/first/beats_per_bar: 用户给定的网格参数
        onset_times/onset_weights: 检出的 onset 秒数与权重（给 onset-fit 路用）
        search_beats: 搜索半径（拍），默认 ±1 拍
        step: 搜索步长（秒），默认 5ms
        tolerance: onset-fit 的容差（秒），默认 25ms
        subdivision: envelope 路每拍再细分多少格（默认 1 = 严格 beat-synchronous）

    ⚠️ offset 只能确定到**一拍以内**：整拍平移会给出几乎相同的分数。
    因此把 ≥tie_ratio×最高分的候选分组取峰，再选离用户值最近的那个。
    """
    onset_env = np.asarray(onset_env, dtype=float) if onset_env is not None else np.zeros(0)
    frame_times = np.asarray(frame_times, dtype=float) if frame_times is not None else np.zeros(0)
    onset_times = np.asarray(onset_times, dtype=float) if onset_times is not None else np.zeros(0)
    if onset_env.size == 0 and onset_times.size == 0:
        return {"available": False, "reason": "既无 onset 包络也无 onset 时间"}

    if duration is None:
        duration = float(frame_times[-1]) if frame_times.size else float(onset_times.max())
    sec_per_beat = 60.0 / bpm
    radius = search_beats * sec_per_beat
    candidates = np.arange(first - radius, first + radius + step * 0.5, step)

    # --- envelope 路 ---
    env_scores = None
    if onset_env.size and frame_times.size:
        n_beats = max(1, int((duration - max(first, 0.0)) / sec_per_beat))
        rel = np.arange(n_beats * subdivision, dtype=float) * (sec_per_beat / subdivision)
        env_scores = np.empty(len(candidates), dtype=float)
        for i, cand in enumerate(candidates):
            times = cand + rel
            times = times[(times >= 0.0) & (times <= duration)]
            env_scores[i] = _sample_envelope(onset_env, times, frame_times)

    # --- onset-fit 路 ---
    fit_scores = None
    if onset_times.size >= 8:
        w = (np.ones_like(onset_times) if onset_weights is None
             else np.asarray(onset_weights, dtype=float))
        w = w / (w.mean() + 1e-12)
        fit_scores = np.empty(len(candidates), dtype=float)
        for i, cand in enumerate(candidates):
            r = (onset_times - cand) % sec_per_beat
            d = np.minimum(r, sec_per_beat - r)
            fit_scores[i] = float(np.mean(w * np.maximum(0.0, 1.0 - d / tolerance)))

    if fit_scores is not None:
        scores, method = fit_scores, "onset-fit"
    else:
        scores, method = env_scores, "envelope"

    result = _summarize_alignment(scores, candidates, first, step, tie_ratio)
    result.update({
        "available": True,
        "method": method,
        "user_first": float(first),
        "search_radius_sec": float(radius),
        "step_sec": float(step),
        "tolerance_sec": float(tolerance),
        "note": "仅报告，不覆盖用户给定的 --first；offset 只能定到一拍以内",
    })
    if env_scores is not None and method != "envelope":
        env_res = _summarize_alignment(env_scores, candidates, first, step, tie_ratio)
        result["envelope_best_first"] = env_res["best_first"]
        result["envelope_delta_ms"] = env_res["delta_ms"]
        result["envelope_note"] = "包络法系统性偏晚约 10ms（librosa onset_strength 的帧间滞后）"
    return result


def _summarize_alignment(scores: np.ndarray, candidates: np.ndarray, first: float,
                         step: float, tie_ratio: float) -> dict:
    """从打分曲线里挑峰、做整拍歧义处理与抛物线细化。"""
    peak = float(scores.max())
    tie_idx = np.flatnonzero(scores >= peak * tie_ratio)
    groups: list[list[int]] = []
    for i in tie_idx:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    peak_idx = [g[int(np.argmax(scores[g]))] for g in groups]
    global_best_i = int(np.argmax(scores))
    best_i = int(peak_idx[int(np.argmin(np.abs(candidates[peak_idx] - first)))])
    best_first = float(candidates[best_i])
    user_score = float(np.interp(first, candidates, scores))
    mean, std = float(scores.mean()), float(scores.std())
    confidence = (scores[best_i] - mean) / std if std > 1e-12 else 0.0

    # 抛物线插值细化峰位
    if 0 < best_i < len(scores) - 1:
        y0, y1, y2 = scores[best_i - 1], scores[best_i], scores[best_i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            shift = 0.5 * (y0 - y2) / denom
            if abs(shift) <= 1.0:
                best_first = float(candidates[best_i] + shift * step)

    return {
        "best_first": best_first,
        "delta_sec": best_first - float(first),
        "delta_ms": (best_first - float(first)) * 1000.0,
        "score_at_user": user_score,
        "score_at_best": float(scores[best_i]),
        "ratio": float(scores[best_i] / user_score) if user_score > 1e-12 else float("inf"),
        "confidence_z": float(confidence),
        "global_best_first": float(candidates[global_best_i]),
        "global_best_delta_ms": float((candidates[global_best_i] - first) * 1000.0),
        "n_tied_peaks": len(peak_idx),
        "peaks": [{"first": float(candidates[i]),
                   "delta_ms": float((candidates[i] - first) * 1000.0),
                   "score": float(scores[i])} for i in peak_idx],
    }


def offset_verdict(check: dict, tolerance_ms: float = 15.0,
                   container_start_ms: float | None = None) -> str:
    """把 offset 校验结果翻成一句中文结论。"""
    if not check.get("available"):
        return "offset 校验不可用：" + str(check.get("reason", ""))
    d = abs(check["delta_ms"])
    z = check["confidence_z"]
    if z < 1.0:
        conf = "对齐峰不显著（该曲 onset 分布平坦，校验参考价值低）"
    elif z < 2.0:
        conf = "对齐峰较弱"
    else:
        conf = "对齐峰显著"
    extra = f"（打分方式 {check.get('method', 'envelope')}）"
    if check.get("n_tied_peaks", 1) > 1:
        extra += f"；搜索窗内有 {check['n_tied_peaks']} 个同级峰（整拍平移歧义），已取离用户值最近的一个"
    if container_start_ms and 5.0 <= d <= 60.0:
        extra += (f"；注意音频容器 start_time={container_start_ms:.1f} ms（MP3 编码器延迟），"
                  f"可解释其中一部分差值——设计文档 §2 已警示 MP3 解码偏移 20–40ms")
    if d <= tolerance_ms:
        return (f"用户 first 与自动最佳偏移相差 {check['delta_ms']:+.1f} ms"
                f"（≤{tolerance_ms:.0f}ms），一致；{conf}{extra}。")
    return (
        f"用户 first 与自动最佳偏移相差 {check['delta_ms']:+.1f} ms（>{tolerance_ms:.0f}ms），"
        f"建议人工复核；{conf}{extra}。已按用户值构造网格。"
    )
