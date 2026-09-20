"""拍网格构造、offset 校验与 **tempo map 复核**。

本模块**不做节拍追踪**：BPM 与 offset 由用户给定，网格由二者直接构造。
`&first` 语义按 simai：谱面第 1 小节第 1 拍在音频中的秒数（可为负）。

**依赖口径**：全部打分/统计函数是**纯 numpy**；只有三个直接吃波形的辅助函数
（`high_res_onsets` / `band_flux` / `_beat_this_beats`）在函数体里**惰性** import
librosa / beat_this，模块级仍然只依赖 numpy。

---

## v1.5（2026-09-20）：tempo map 复核换代（`tempo_map`）

**被替换的旧做法**：`docs/audio-analysis.md` §2.4 第 5 步的「全曲按 8/16 小节分块、
在 **32 分网格**（槽长 36.6 ms @205BPM、搜索半径 ±17 ms）上逐块求相位 φ_i，
再对 (t_i, φ_i) 线性回归」。它有**两个致命盲点**，在 test-01 上被坐实：

1. **对「整格错位」完全不敏感**。打分以一格为周期，半拍（146 ms）、16 分（73 ms）、
   32 分（36.6 ms）在 32 分网格里**全部混叠成 0**。于是谱面整体错半拍、
   中途插入一个 16 分的接缝、乐句错一格——这套复核一个都看不见。
2. **测不出 >0.4% 的 BPM 误差**。一块 8 小节 ≈ 9.4 s，BPM 差 0.5% 就让相位在块内
   绕过一整格，回归到的斜率是**绕圈后的残值**，看起来反而很小。
   test-01 里「bars 73–80 +13.8 ms / bars 105–112 +15.0 ms 偏大」就是绕圈的痕迹，
   换成 8 分量程后这两块是平的（+8.6…+10.3 / +9.9…+12.9 ms）。

**新做法（本模块实现）**：

- **BPM 走「无相位」滑窗全域扫描**（`window_bpm_scan`）：每个窗在 BPM 全域上
  对**相位取最大**后打分，因此结论与相位无关，绕圈问题不存在；
  再加**主体段高精度拟合**（`fine_bpm_fit`）与**前后半独立拟合一致性**
  （`half_split_consistency`）。三者一致才判「恒定」。
- **相位走 8 分量程**（`bar_phase_curve`，量程 ±半拍），能看见任何 ≥1 个 16 分的
  离散跳格；`detect_phase_jumps` 专门找接缝。
- **半拍归属**（`half_beat_attribution`）用**分带 flux**（kick/hat 对 bass），
  而且**只在鼓最干净的小节上判**（`cleanest_drum_bars`）——全曲平均会被 drop 里
  打满 8/16 分的 kick 抹平（test-01 全曲平均正/反比 1.00，干净段 2.14）。
- **无鼓段不做 onset 验证**：旧的「无鼓段单独测相位 → 确认同一网格」**作废**
  （实测该 onset 列表在混响 pad 上比随机还差：到最近 16 分线 |中位| 19.1 ms，
  随机基线 9.1 ms）。这些段一律报 `unmeasurable`，按主体网格外推。
- **beat_this 可选**（装了才跑），**librosa 路必须带 hop 帧量化告警**，
  两者都只作旁证，不得单独作结论。

判据与阈值见 `TEMPO_*` 常量。
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

# v0.3 反拍假警报修复（`docs/research/audio-chart-calibration.md` §1.2 / R1）
#
# **bug**：v0.2 的搜索半径是 ±1 拍，而 onset-fit 的打分函数本身以**一拍为周期**，
# 于是窗口里除了真值峰，还含有 φ*±1 拍的复制峰**与 φ*±半拍的「反拍」峰**。
# 在鼓点规整（8 分 hi-hat 正反拍等强）的曲子上，反拍与正拍得分几乎并列，
# 只要反拍略高就会被选中 —— 实测假警报：Signature（BPM 128）−229.0 ms
# ≈ −半拍 234.4 ms；麒麟（BPM 220）−129.8 ms ≈ −半拍 136.4 ms。
#
# **修法（三条同时）**：
#   ① 搜索半径收到 **±0.4 拍**（= 半拍 − 0.1 拍的保护边），反拍根本进不了窗口；
#   ② 同级峰仍然「取离用户值最近的那个」（v0.2 已有，保留）；
#   ③ |φ*| 逼近半拍（≥ 0.35 拍）或出现**对称并列峰** → 判 **「反拍歧义 / 不可判定」**，
#      **不报警**；且 |φ*| > 容差时还要过一道 `ratio ≥ 1.15` 的显著性门才报警。
DEFAULT_SEARCH_BEATS = 0.40
MAX_SEARCH_BEATS = 0.45          # 再大就会把反拍放进窗口
OFFBEAT_FRAC = 0.35              # |φ*| ≥ 0.35 拍 → 视为逼近反拍
SYMMETRIC_TOL_MS = 20.0          # 两个同级峰关于用户值对称的容差
ALARM_RATIO_GATE = 1.15          # φ* 必须比 φ=0 的得分高出这么多倍才允许报警
DECODE_HINT_MS = 30.0            # 15–30 ms：指向 MP3 解码延迟而不是"谱面可能错"


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
    search_beats: float = DEFAULT_SEARCH_BEATS,
    step: float = 0.005,
    subdivision: int = 1,
    tie_ratio: float = 0.90,
    onset_times: np.ndarray | None = None,
    onset_weights: np.ndarray | None = None,
    tolerance: float = 0.025,
    allow_offbeat_window: bool = False,
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
        search_beats: 搜索半径（拍），默认 **±0.40 拍**。
            ⚠️ **不得 ≥ 0.5 拍**：打分函数以一拍为周期，半拍处正是"反拍"，
            在正反拍等强的曲子上会并列夺峰（v0.2 的 ±1 拍就是 Signature/麒麟
            假警报的根因）。超过 `MAX_SEARCH_BEATS` 会被夹回并在结果里注明。
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
    clamped = float(search_beats) > MAX_SEARCH_BEATS and not allow_offbeat_window
    if clamped:
        search_beats = MAX_SEARCH_BEATS
    search_beats = float(search_beats)
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
        "search_beats": float(search_beats),
        "search_beats_clamped": bool(clamped),
        "sec_per_beat": float(sec_per_beat),
        "half_beat_ms": float(sec_per_beat * 500.0),
        "ratio_gate": float(ALARM_RATIO_GATE),
        "step_sec": float(step),
        "tolerance_sec": float(tolerance),
        "note": "仅报告，不覆盖用户给定的 --first；offset 只能定到一拍以内",
    })
    result.update(_offbeat_flags(result, sec_per_beat))
    if env_scores is not None and method != "envelope":
        env_res = _summarize_alignment(env_scores, candidates, first, step, tie_ratio)
        result["envelope_best_first"] = env_res["best_first"]
        result["envelope_delta_ms"] = env_res["delta_ms"]
        result["envelope_note"] = "包络法系统性偏晚约 10ms（librosa onset_strength 的帧间滞后）"
    return result


def _offbeat_flags(result: dict, sec_per_beat: float) -> dict:
    """判定「反拍歧义」：φ* 逼近半拍，或出现关于用户值对称的并列峰。

    这两种情形下 offset **不可判定**（真值附近不是局部极大，算法只是在两个反拍
    候选里二选一），按 R1-③ 应报「不可判定」而不是报警。
    """
    offbeat_ms = OFFBEAT_FRAC * sec_per_beat * 1000.0
    near_offbeat = abs(float(result["delta_ms"])) >= offbeat_ms
    deltas = [float(p["delta_ms"]) for p in result.get("peaks", [])]
    symmetric = any(
        a < 0.0 < b and abs(a + b) <= SYMMETRIC_TOL_MS
        and min(abs(a), abs(b)) >= 0.25 * sec_per_beat * 1000.0
        for a in deltas for b in deltas)
    return {
        "offbeat_threshold_ms": float(offbeat_ms),
        # φ* 本身就贴在反拍上 → 这次校验没有意义
        "offbeat_ambiguous": bool(near_offbeat),
        # 同级峰关于用户值对称（打分曲线平坦、正反拍并列）→ 参考价值低，
        # 但若 φ* 仍落在容差内，结论"一致"照样成立，只是置信度打折
        "tied_symmetric": bool(symmetric),
        "offbeat_reason": ("φ* 逼近半拍（反拍）" if near_offbeat else
                           ("同级峰关于用户值对称（正反拍并列）" if symmetric else "")),
    }


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
                   container_start_ms: float | None = None,
                   decode_hint_ms: float = DECODE_HINT_MS,
                   ratio_gate: float = ALARM_RATIO_GATE) -> str:
    """把 offset 校验结果翻成一句中文结论（v0.3：四档，含「反拍歧义」）。

    档位：**一致 ≤15 ms** / **疑似解码延迟 15–30 ms**（R2：措辞指向 MP3 容器
    `start_time`，不是"谱面可能错"）/ **反拍歧义（不可判定，不报警）** /
    **报警 >30 ms 且 `ratio ≥ 1.15`**。
    """
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

    # 反拍歧义优先于一切：这种情况下 φ* 本身没有意义
    if check.get("offbeat_ambiguous"):
        return (f"⚠️ **反拍歧义，offset 不可判定**：φ* = {check['delta_ms']:+.1f} ms，"
                f"半拍 = {check.get('half_beat_ms', 0.0):.1f} ms"
                f"（{check.get('offbeat_reason', '')}）。正拍与反拍得分接近时算法只是在两个"
                f"反拍候选里二选一，**不作为报警**；已按用户值构造网格。{conf}{extra}。")
    if check.get("tied_symmetric"):
        extra += "；打分曲线平坦且同级峰关于用户值对称（正反拍并列），本次校验参考价值打折"
    if d <= tolerance_ms:
        return (f"用户 first 与自动最佳偏移相差 {check['delta_ms']:+.1f} ms"
                f"（≤{tolerance_ms:.0f}ms），一致；{conf}{extra}。")
    if d <= decode_hint_ms:
        return (f"用户 first 与自动最佳偏移相差 {check['delta_ms']:+.1f} ms"
                f"（{tolerance_ms:.0f}–{decode_hint_ms:.0f}ms）——**这个量级几乎都是 MP3 "
                f"解码/容器延迟**（容器 start_time 常见 23 ms），**不是「谱面可能错」**；"
                f"以用户值为准，如需逐帧对齐可人工核一次。{conf}{extra}。")
    ratio = float(check.get("ratio", 0.0) or 0.0)
    if ratio < ratio_gate or check.get("tied_symmetric"):
        return (f"φ* = {check['delta_ms']:+.1f} ms（>{decode_hint_ms:.0f}ms）但得分只比用户值"
                f"高 {ratio:.2f}×（< 显著性门 {ratio_gate:.2f}×）→ **判为不可判定，不报警**；"
                f"已按用户值构造网格。{conf}{extra}。")
    return (
        f"用户 first 与自动最佳偏移相差 {check['delta_ms']:+.1f} ms（>{decode_hint_ms:.0f}ms，"
        f"得分 {ratio:.2f}× 用户值），建议人工复核；{conf}{extra}。已按用户值构造网格。"
    )


# =====================================================================
# tempo map 复核（v1.5，替换旧的「8/16 小节块 + 32 分网格 ±17 ms」）
# =====================================================================

# --- 判据阈值（全部可由调用方覆盖）---
TEMPO_SCAN_LO = 100.0            # 全域扫描下限（BPM）
TEMPO_SCAN_HI = 260.0            # 全域扫描上限（BPM）
TEMPO_COARSE_STEP = 0.25         # 一阶粗扫步长
TEMPO_FINE_STEP = 0.02           # 二阶细扫步长（围绕粗扫峰 ±TEMPO_FINE_SPAN）
TEMPO_FINE_SPAN = 1.5            # 二阶细扫半径（BPM）
TEMPO_SIGMA_SEC = 0.012          # onset-fit 高斯核 σ
TEMPO_N_PHASES_COARSE = 24       # 粗扫的相位格数
TEMPO_N_PHASES_FINE = 128        # 细扫的相位格数
TEMPO_WINDOW_BARS = (4, 2)       # 滑窗长度（小节），步长恒为 1 小节
TEMPO_MIN_ONSETS_WINDOW = 6      # 一个窗至少要这么多 onset 才算有效
TEMPO_CONST_TOL_BPM = 0.30       # 「落在中位 ±这个值」算同速
TEMPO_CONST_HIT_RATIO = 0.90     # 有效窗里至少这个比例达标才判恒定
TEMPO_HALF_SPLIT_TOL = 0.05      # 前后半独立拟合的允许差（BPM）
TEMPO_SLOPE_GATE = 2e-4          # 相位回归斜率门槛（s/s），沿用旧口径
TEMPO_JUMP_FRAC = 0.20           # 前后各 TEMPO_JUMP_WIN 小节的相位中位差 ≥ 这么多「格」判接缝
TEMPO_JUMP_WIN = 3               # 接缝判定的前后窗口（小节）——单小节抖动不算接缝
TEMPO_NO_DRUM_MIN_ONSETS = 3     # 一小节 drums onset 少于这个数 → 视为无鼓
TEMPO_MEASURED_RATIO_GATE = 0.90 # 可测小节占比低于此值时，**禁止**报 constant（见 v1.5.1）
TEMPO_CLEAN_BARS = 16            # 半拍归属取多少个「最干净」的小节
TEMPO_CLEAN_KICK_LO = 3          # 干净小节的 kick 事件数下界（4-on-the-floor）
TEMPO_CLEAN_KICK_HI = 5          # 上界（再多就是打满 8/16 分的 drop，不能用）
TEMPO_ATTR_RATIO_GATE = 1.20     # 正/反拍能量比超过它才敢下半拍归属结论

# 分带（Hz）：kick / hat 判拍相位，bass 只作反证（offbeat bass 是常见曲风特征）
TEMPO_BANDS = {
    "kick": (35.0, 120.0),
    "snare": (180.0, 900.0),
    "hat": (6000.0, 15000.0),
    "bass": (30.0, 300.0),
}


# ---------- 纯 numpy 核心：onset-fit 打分 ----------

def _fit_scores(rel_times: np.ndarray, weights: np.ndarray, periods: np.ndarray,
                n_phases: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """对每个候选周期，返回「相位取最大」后的加权命中得分与对应相位。

    这是整套复核的地基：**得分对相位取过最大值，所以与相位无关**——
    旧做法「固定相位/固定网格求残差」才会被整格错位与绕圈欺骗。

    参数：
        rel_times: onset 相对窗口起点的秒数（升序不必）
        weights:   onset 权重（会被归一化到和为 1）
        periods:   候选网格周期（秒）。拍格传 60/bpm，8 分格传 30/bpm
        n_phases:  相位搜索格数（均匀覆盖一个周期）
        sigma:     高斯核宽度（秒）——落在 ±σ 内算「命中」

    返回 (scores, best_phase_sec)，长度都等于 len(periods)。
    """
    rel_times = np.asarray(rel_times, dtype=float)
    weights = np.asarray(weights, dtype=float)
    periods = np.asarray(periods, dtype=float)
    if rel_times.size == 0 or periods.size == 0:
        return np.zeros(periods.size), np.zeros(periods.size)
    w = weights / (weights.sum() + 1e-12)
    phases = (np.arange(n_phases, dtype=float) / n_phases)[None, :]
    scores = np.empty(periods.size, dtype=float)
    best_ph = np.empty(periods.size, dtype=float)
    # 分块：把 (n_onset × n_period × n_phase) 的峰值内存压到 ~2e7 个元素
    chunk = max(1, int(2e7 / max(rel_times.size * n_phases, 1)))
    for c0 in range(0, periods.size, chunk):
        per = periods[c0:c0 + chunk]
        r = (rel_times[:, None] / per[None, :]) % 1.0          # (n, C)
        d = np.abs(r[:, :, None] - phases[None, :, :])          # (n, C, P)
        d = np.minimum(d, 1.0 - d) * per[None, :, None]         # 折成秒
        sc = (w[:, None, None] * np.exp(-(d ** 2) / (2.0 * sigma ** 2))).sum(axis=0)
        j = np.argmax(sc, axis=1)
        scores[c0:c0 + chunk] = sc[np.arange(sc.shape[0]), j]
        best_ph[c0:c0 + chunk] = j / n_phases * per
        del r, d, sc
    return scores, best_ph


def fine_bpm_fit(onset_times, onset_weights, t_start: float, t_end: float,
                 bpm_lo: float, bpm_hi: float, step: float = TEMPO_FINE_STEP,
                 subdivision: int = 2, sigma: float = TEMPO_SIGMA_SEC,
                 n_phases: int = TEMPO_N_PHASES_FINE,
                 report_at: Sequence[float] = ()) -> dict:
    """某一段上的高精度 BPM 拟合（相位无关）。

    `subdivision=2` 表示拟合 **8 分格**（周期 30/bpm），这是音游曲最稳的口径；
    `subdivision=1` 是拍格。`report_at` 里的 BPM 会额外回报其得分，
    方便回答「@205 的得分是不是就等于峰值」。

    返回 dict：`bpm` / `score` / `phase_sec` / `n` / `peak95_lo` / `peak95_hi`
    （得分 ≥ 95% 峰值的 BPM 区间，等价于峰宽）/ `score_at`。
    """
    t = np.asarray(onset_times, dtype=float)
    w = (np.ones_like(t) if onset_weights is None
         else np.asarray(onset_weights, dtype=float))
    m = (t >= t_start) & (t < t_end)
    if int(m.sum()) < 4 or bpm_hi < bpm_lo:
        return {"available": False, "n": int(m.sum()),
                "reason": "该段 onset 太少，无法拟合（无鼓段/渐弱段属正常）"}
    bpms = np.arange(float(bpm_lo), float(bpm_hi) + 1e-9, float(step))
    if bpms.size == 0:
        bpms = np.array([float(bpm_lo)])
    periods = (60.0 / bpms) / float(subdivision)
    scores, phases = _fit_scores(t[m] - t_start, w[m], periods, n_phases, sigma)
    i = int(np.argmax(scores))
    lo_s = float(scores.min())
    thr = scores[i] - (scores[i] - lo_s) * 0.05
    ok = bpms[scores >= thr]
    out = {
        "available": True,
        "bpm": float(bpms[i]),
        "score": float(scores[i]),
        "phase_sec": float(phases[i]),
        "n": int(m.sum()),
        "subdivision": int(subdivision),
        "peak95_lo": float(ok.min()) if ok.size else float(bpms[i]),
        "peak95_hi": float(ok.max()) if ok.size else float(bpms[i]),
        "scan_lo": float(bpms[0]), "scan_hi": float(bpms[-1]), "step": float(step),
    }
    out["score_at"] = {f"{x:.3f}": float(scores[int(np.argmin(np.abs(bpms - x)))])
                       for x in report_at}
    return out


def _two_stage_best(rel_t: np.ndarray, w: np.ndarray, subdivision: int,
                    scan_lo: float, scan_hi: float, coarse_step: float,
                    fine_step: float, fine_span: float, sigma: float) -> tuple[float, float]:
    """粗扫全域 → 围绕粗峰细扫，返回 (best_bpm, best_score)。"""
    coarse = np.arange(scan_lo, scan_hi + 1e-9, coarse_step)
    sc, _ = _fit_scores(rel_t, w, (60.0 / coarse) / subdivision,
                        TEMPO_N_PHASES_COARSE, sigma)
    b0 = float(coarse[int(np.argmax(sc))])
    fine = np.arange(max(scan_lo, b0 - fine_span), min(scan_hi, b0 + fine_span) + 1e-9,
                     fine_step)
    if fine.size == 0:
        return b0, float(sc.max())
    sc2, _ = _fit_scores(rel_t, w, (60.0 / fine) / subdivision,
                         TEMPO_N_PHASES_FINE, sigma)
    j = int(np.argmax(sc2))
    return float(fine[j]), float(sc2[j])


def window_bpm_scan(onset_times, onset_weights, bpm: float, first: float,
                    beats_per_bar: int = 4, n_bars: int | None = None,
                    duration: float | None = None,
                    window_bars: int = 4, hop_bars: int = 1,
                    scan_lo: float = TEMPO_SCAN_LO, scan_hi: float = TEMPO_SCAN_HI,
                    coarse_step: float = TEMPO_COARSE_STEP,
                    fine_step: float = TEMPO_FINE_STEP,
                    fine_span: float = TEMPO_FINE_SPAN,
                    subdivision: int = 2, sigma: float = TEMPO_SIGMA_SEC,
                    local_span: float = 10.0,
                    min_onsets: int = TEMPO_MIN_ONSETS_WINDOW) -> list[dict]:
    """**无相位滑窗全域 BPM 扫描**（本模块的主判据）。

    每个窗给两个数：
    - `best`：在 [scan_lo, scan_hi] **全域**上的最佳 BPM（能看见半/倍速与跑飞）；
    - `best_local`：限制在用户 BPM ±`local_span` 内的最佳 BPM（用来画曲线、判漂移）。

    因为得分对相位取过最大值，**这条曲线与 `first` 无关**，
    也就不会像旧的固定网格残差那样被整格错位骗过去。

    窗长固定为 `window_bars` 小节、步长 `hop_bars` 小节（默认 1）。
    onset 少于 `min_onsets` 的窗标 `valid=False`（无鼓段会大量落在这里，属正常）。
    """
    t = np.asarray(onset_times, dtype=float)
    w = (np.ones_like(t) if onset_weights is None
         else np.asarray(onset_weights, dtype=float))
    bar_sec = beats_per_bar * 60.0 / float(bpm)
    if n_bars is None:
        span = (duration if duration else (float(t.max()) if t.size else 0.0)) - first
        n_bars = max(1, int(span / bar_sec))
    out: list[dict] = []
    for bar in range(1, int(n_bars) - int(window_bars) + 2, int(hop_bars)):
        a = first + (bar - 1) * bar_sec
        b = a + window_bars * bar_sec
        m = (t >= a) & (t < b)
        rec = {"bar": int(bar), "t": float(a), "window_bars": int(window_bars),
               "n": int(m.sum())}
        if int(m.sum()) < min_onsets:
            rec.update({"valid": False, "best": None, "best_local": None,
                        "score": None,
                        "reason": "窗内 onset 不足（无鼓段/留白属正常，不作为变速证据）"})
            out.append(rec)
            continue
        rel, ww = t[m] - a, w[m]
        gb, gs = _two_stage_best(rel, ww, subdivision, scan_lo, scan_hi,
                                 coarse_step, fine_step, fine_span, sigma)
        lo = max(scan_lo, bpm - local_span)
        hi = min(scan_hi, bpm + local_span)
        loc = np.arange(lo, hi + 1e-9, fine_step)
        sc, _ = _fit_scores(rel, ww, (60.0 / loc) / subdivision,
                            TEMPO_N_PHASES_FINE, sigma)
        j = int(np.argmax(sc))
        rec.update({"valid": True, "best": gb, "score": gs,
                    "best_local": float(loc[j]), "score_local": float(sc[j])})
        out.append(rec)
    return out


def bar_phase_curve(onset_times, onset_weights, bpm: float, first: float,
                    beats_per_bar: int = 4, n_bars: int | None = None,
                    duration: float | None = None, subdivision: int = 2,
                    min_onsets: int = 3) -> dict:
    """**8 分量程**的逐小节相位曲线（加权圆均值）。

    `subdivision=2` 指「一拍两格」，一格 = 一个 8 分 = 半拍，圆均值折到 ±半格，
    所以**量程 = ±(15/bpm) 秒 = 205 BPM 下 ±73 ms**——正是旧做法
    （32 分格、±17 ms）看不见的那一档。`subdivision=1`（拍格）量程翻倍到 ±146 ms。

    返回 dict：`phase_sec`（每小节一个，无效为 nan）、`n`、`slope`（相位对时间的
    线性回归斜率，s/s）、`bpm_from_slope`、`median_ms`、`std_ms`、`period_sec`。
    """
    t = np.asarray(onset_times, dtype=float)
    w = (np.ones_like(t) if onset_weights is None
         else np.asarray(onset_weights, dtype=float))
    bar_sec = beats_per_bar * 60.0 / float(bpm)
    period = (60.0 / float(bpm)) / float(subdivision)
    if n_bars is None:
        span = (duration if duration else (float(t.max()) if t.size else 0.0)) - first
        n_bars = max(1, int(span / bar_sec))
    ph = np.full(int(n_bars), np.nan)
    cnt = np.zeros(int(n_bars), dtype=int)
    for i in range(int(n_bars)):
        a = first + i * bar_sec
        m = (t >= a) & (t < a + bar_sec)
        cnt[i] = int(m.sum())
        if cnt[i] < min_onsets:
            continue
        ang = 2.0 * np.pi * ((t[m] - first) % period) / period
        z = (w[m] * np.exp(1j * ang)).sum() / (w[m].sum() + 1e-12)
        ph[i] = float(np.angle(z)) / (2.0 * np.pi) * period
    ok = ~np.isnan(ph)
    tb = first + np.arange(int(n_bars)) * bar_sec
    slope = float(np.polyfit(tb[ok], ph[ok], 1)[0]) if int(ok.sum()) >= 3 else float("nan")
    return {
        "phase_sec": ph, "n": cnt, "bar_times": tb, "valid_bars": int(ok.sum()),
        "period_sec": float(period), "subdivision": int(subdivision),
        "median_ms": float(np.nanmedian(ph) * 1000.0) if ok.any() else float("nan"),
        "std_ms": float(np.nanstd(ph) * 1000.0) if ok.any() else float("nan"),
        "slope": slope,
        "bpm_from_slope": float(bpm * (1.0 - slope)) if slope == slope else float("nan"),
        "slope_gate": TEMPO_SLOPE_GATE,
        "range_ms": float(period * 500.0),
        "note": "量程 ±半格；旧法用 32 分格（±17 ms），整格错位会被混叠成 0",
    }


def detect_phase_jumps(curve: dict, jump_frac: float = TEMPO_JUMP_FRAC,
                       win: int = TEMPO_JUMP_WIN) -> list[dict]:
    """在逐小节相位曲线里找**离散跳格**（接缝）。

    这是旧做法**结构上无法发现**的一类故障：整体错半拍、中途插入一个 16 分、
    乐句错一格——在 32 分网格上全部混叠成 0。

    判据是「**持续的台阶**」而不是「相邻两小节的差」：拿候选接缝前 `win` 个
    有效小节的相位中位数，跟后 `win` 个的中位数比，折到 ±半格后超过
    `jump_frac` 个格才算。单小节的抖动（细分音密集、采样泄漏）不该算接缝——
    test-01 里 bar43(−19 ms)→bar44(+27 ms) 的 46 ms 单点跳就是这种噪声，
    用相邻差会误报，用中位台阶则正确地判为无接缝。

    相邻的多个候选会合并成一条（一个接缝只报一次）。
    """
    ph = np.asarray(curve["phase_sec"], dtype=float)
    period = float(curve["period_sec"])
    tb = np.asarray(curve["bar_times"], dtype=float)
    idx = np.flatnonzero(~np.isnan(ph))
    raw: list[dict] = []
    for k in range(1, idx.size):
        a = idx[max(0, k - win):k]
        b = idx[k:k + win]
        if a.size < min(win, 2) or b.size < min(win, 2):
            continue
        d = float(np.median(ph[b]) - np.median(ph[a]))
        d = (d + period / 2.0) % period - period / 2.0
        if abs(d) >= jump_frac * period:
            raw.append({"from_bar": int(idx[k - 1] + 1), "to_bar": int(idx[k] + 1),
                        "t": float(tb[idx[k]]), "jump_ms": float(d * 1000.0),
                        "jump_frac_of_slot": float(d / period)})
    # 合并相邻候选：同一个接缝会在 win 个位置上连续触发，只保留幅度最大的那个
    out: list[dict] = []
    for j in raw:
        if out and j["to_bar"] - out[-1]["to_bar"] <= win:
            if abs(j["jump_ms"]) > abs(out[-1]["jump_ms"]):
                out[-1] = j
            continue
        out.append(j)
    return out


def beat_residual_curve(onset_times, onset_weights, bpm: float, first: float,
                        beats_per_bar: int = 4, duration: float | None = None,
                        half_window_beats: float = 0.25) -> dict:
    """**逐拍**残差（不是 8 小节块均值）：每条拍线取 ±`half_window_beats` 内最强的 onset。

    这条曲线只用来画图与人工复核；判据以 `window_bpm_scan` 与 `bar_phase_curve` 为准。
    """
    t = np.asarray(onset_times, dtype=float)
    w = (np.ones_like(t) if onset_weights is None
         else np.asarray(onset_weights, dtype=float))
    order = np.argsort(t)
    t, w = t[order], w[order]
    spb = 60.0 / float(bpm)
    end = duration if duration else (float(t.max()) if t.size else first)
    n = max(0, int((end - first) / spb))
    lines = first + np.arange(n) * spb
    hw = half_window_beats * spb
    res = np.full(n, np.nan)
    lo = np.searchsorted(t, lines - hw)
    hi = np.searchsorted(t, lines + hw)
    for i in range(n):
        if hi[i] > lo[i]:
            k = lo[i] + int(np.argmax(w[lo[i]:hi[i]]))
            res[i] = t[k] - lines[i]
    ok = ~np.isnan(res)
    return {"beat_times": lines, "residual_sec": res, "hit_ratio": float(ok.mean()) if n else 0.0,
            "median_ms": float(np.nanmedian(res) * 1000.0) if ok.any() else float("nan"),
            "mad_ms": float(np.nanmedian(np.abs(res - np.nanmedian(res))) * 1000.0)
                      if ok.any() else float("nan")}


def half_split_consistency(onset_times, onset_weights, t_start: float, t_end: float,
                           bpm: float, span: float = 0.5,
                           step: float = TEMPO_FINE_STEP, subdivision: int = 2,
                           sigma: float = TEMPO_SIGMA_SEC) -> dict:
    """把主体段一刀两半，各自独立拟合 BPM，看两半是否一致。

    变速（哪怕只是缓慢漂移）一定让两半分开；恒定则两半在分辨率内重合。
    """
    mid = 0.5 * (t_start + t_end)
    a = fine_bpm_fit(onset_times, onset_weights, t_start, mid,
                     bpm - span, bpm + span, step, subdivision, sigma)
    b = fine_bpm_fit(onset_times, onset_weights, mid, t_end,
                     bpm - span, bpm + span, step, subdivision, sigma)
    if not (a.get("available") and b.get("available")):
        return {"available": False, "first_half": a, "second_half": b,
                "reason": "有一半的 onset 太少"}
    d = abs(a["bpm"] - b["bpm"])
    return {"available": True, "first_half": a, "second_half": b,
            "split_t": float(mid), "delta_bpm": float(d),
            "tolerance": float(TEMPO_HALF_SPLIT_TOL),
            "consistent": bool(d <= TEMPO_HALF_SPLIT_TOL)}


# ---------- 吃波形的辅助（惰性 import librosa / beat_this）----------

TEMPO_HOP = 128                  # 2.90 ms @44.1kHz —— 必须比 onsets.HOP(11.6ms) 细
TEMPO_NFFT_ONSET = 512
TEMPO_NFFT_BAND = 1024
TEMPO_SR = 44100


def high_res_onsets(y, sr: int = TEMPO_SR, hop: int = TEMPO_HOP,
                    n_fft: int = TEMPO_NFFT_ONSET,
                    fmin: float | None = None, fmax: float | None = None,
                    delta: float = 0.05, wait_ms: float = 20.0) -> dict:
    """高分辨率 onset 提取（hop=128 → 2.90 ms）。

    ⚠️ **不要用 `onsets.HOP`（11.6 ms @22.05kHz）做 tempo 复核**：
    那个量级和待测偏移同量级，会把答案磨平（`docs/audio-analysis.md` §2.4 已警示）。

    返回 {"times", "weights", "env", "frame_times", "sr", "hop"}。
    """
    import librosa  # 惰性：让模块级保持纯 numpy
    y = np.asarray(y, dtype=float)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    if fmin is not None or fmax is not None:
        fr = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        lo = 0 if fmin is None else int(np.searchsorted(fr, fmin))
        hi = len(fr) if fmax is None else int(np.searchsorted(fr, fmax))
        S = S[lo:hi]
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max),
                                       sr=sr, hop_length=hop, lag=2, aggregate=np.median)
    ft = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop)
    wait = max(1, int(round(wait_ms / 1000.0 * sr / hop)))
    avg = max(1, int(0.10 * sr / hop))
    idx = np.asarray(librosa.util.peak_pick(
        env, pre_max=wait, post_max=wait, pre_avg=avg, post_avg=avg + 1,
        delta=delta, wait=wait), dtype=int)
    return {"times": ft[idx], "weights": env[idx], "env": env,
            "frame_times": ft, "sr": int(sr), "hop": int(hop)}


BAND_FLUX_FLOOR_DB = 40.0        # 见下：动态范围地板，防「静音带」上的假 flux


def band_flux(y, sr: int = TEMPO_SR, f_lo: float = 35.0, f_hi: float = 120.0,
              hop: int = TEMPO_HOP, n_fft: int = TEMPO_NFFT_BAND,
              diff_frames: int = 1,
              floor_rel_db: float = BAND_FLUX_FLOOR_DB) -> dict:
    """分带能量的正向差分（dB/帧）——用来判 kick / hat / bass 落在哪条线上。

    用**能量 dB 的差分**而不是线性幅度和：后者会被高频的 bin 数压倒
    （5–16 kHz 占了大半个谱），把所有鼓件都判成 hat。

    ⚠️ **必须有动态范围地板**（`floor_rel_db`，默认 40 dB）。
    dB 差分在「这一带几乎没有能量」的时候会爆炸：实测一个 8 kHz 的 hat 泄漏到
    35–120 Hz 只有 −0.9 dB（比 kick 低 46 dB），但因为该带底噪是 −81 dB，
    **dB 上涨了 63.6 dB**，于是反拍上凭空多出一排「kick」。
    真实曲子的 breakdown / 低通渐弱段同理会造出幻影瞬态。
    做法：把能量曲线在 `p99 − floor_rel_db` 处截断再差分。
    """
    import librosa  # 惰性
    y = np.asarray(y, dtype=float)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    fr = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    lo, hi = int(np.searchsorted(fr, f_lo)), int(np.searchsorted(fr, f_hi))
    e = librosa.power_to_db(S[max(lo, 0):max(hi, lo + 1)].sum(axis=0) + 1e-12)
    if floor_rel_db and floor_rel_db > 0 and e.size:
        e = np.maximum(e, float(np.percentile(e, 99)) - float(floor_rel_db))
    d = int(max(1, diff_frames))
    fl = np.concatenate([np.zeros(d), np.maximum(0.0, e[d:] - e[:-d])])
    ft = librosa.frames_to_time(np.arange(len(fl)), sr=sr, hop_length=hop)
    return {"flux": fl, "energy_db": e, "frame_times": ft,
            "floor_rel_db": float(floor_rel_db)}


def sample_flux_at(flux, frame_times, times, half_window: float = 0.030) -> np.ndarray:
    """在给定秒数附近取 flux 的局部最大（容忍检测器的几毫秒抖动）。"""
    flux = np.asarray(flux, dtype=float)
    frame_times = np.asarray(frame_times, dtype=float)
    times = np.asarray(times, dtype=float)
    lo = np.searchsorted(frame_times, times - half_window)
    hi = np.searchsorted(frame_times, times + half_window)
    out = np.zeros(times.size)
    for i in range(times.size):
        if hi[i] > lo[i]:
            out[i] = float(flux[lo[i]:hi[i]].max())
    return out


def cleanest_drum_bars(kick_flux, frame_times, bpm: float, first: float,
                       beats_per_bar: int = 4, n_bars: int = 0,
                       want: int = TEMPO_CLEAN_BARS,
                       lo_hits: int = TEMPO_CLEAN_KICK_LO,
                       hi_hits: int = TEMPO_CLEAN_KICK_HI,
                       delta_db: float = 4.0,
                       rel_frac: float = 0.40,
                       beat_tol: float = 0.25) -> list[int]:
    """挑出「鼓最干净」的小节——**判半拍归属只能在这些小节上做**。

    判据是 **grid 无关** 的：数这一小节里 kick 的击打次数，只保留
    `lo_hits ≤ n ≤ hi_hits`（默认 3–5，即接近 4-on-the-floor）的小节，
    再按 kick 总能量从高到低取前 `want` 个。

    光数「几下」还不够：drop 里挑出的 3–5 下可能是 16 分连打里最响的几下
    （test-01 的 bar22 实测击打间隔是 0.40/0.36/0.44/0.21 拍）。
    所以还要求**击打间隔本身接近一拍**：中位间隔落在 1±`beat_tol` 拍内，
    且至少 n−2 个间隔也落在这个范围。这一条只用到**拍的周期**、
    完全不用拍的**相位**，所以不会偏袒 `first` 或 `first+半拍` 中的任何一个。

    为什么必须这么挑：drop 段的 kick 会打满 8 分甚至 16 分，正拍与反拍都有货，
    **全曲平均出来的正/反比会被抹平到 1.00**（test-01 实测：全曲 1.03，
    按本函数挑出的干净小节 2.32）——这正是上一轮把半拍归属判成「分不开」的根源。
    """
    flux = np.asarray(kick_flux, dtype=float)
    ft = np.asarray(frame_times, dtype=float)
    bar_sec = beats_per_bar * 60.0 / float(bpm)
    if n_bars <= 0:
        n_bars = max(1, int((float(ft[-1]) - first) / bar_sec)) if ft.size else 0
    min_gap = int(max(1, 0.045 * (len(ft) / max(ft[-1], 1e-9)))) if ft.size else 1
    cand: list[tuple[float, int]] = []
    for b in range(1, int(n_bars) + 1):
        a = first + (b - 1) * bar_sec
        i0, i1 = int(np.searchsorted(ft, a)), int(np.searchsorted(ft, a + bar_sec))
        if i1 - i0 < 8:
            continue
        seg = flux[i0:i1]
        # 峰计数：既要过绝对门 `delta_db`，也要 ≥ 本小节最大 flux 的 `rel_frac`
        # （绝对门单独用不行——曲子各段电平差很远；相对门单独用也不行——
        #   安静小节里的噪声会被当成 4 个击打）
        thr = max(float(delta_db), float(rel_frac) * float(seg.max()))
        hits, last, pos = 0, -10 ** 9, []
        for k in range(len(seg)):
            if seg[k] < thr:
                continue
            a0, a1 = max(0, k - min_gap), min(len(seg), k + min_gap + 1)
            if seg[k] >= seg[a0:a1].max() and k - last > min_gap:
                hits += 1
                last = k
                pos.append(float(ft[i0 + k]))
        if not (lo_hits <= hits <= hi_hits) or len(pos) < 2:
            continue
        iv = np.diff(np.asarray(pos, dtype=float))          # 相邻击打的间隔（秒）
        spb = 60.0 / float(bpm)
        near = np.abs(iv / spb - 1.0) <= beat_tol
        if not (abs(float(np.median(iv)) / spb - 1.0) <= beat_tol
                and int(near.sum()) >= max(1, hits - 2)):
            continue
        cand.append((float(near.mean()), float(seg.sum()), b))
    cand.sort(reverse=True)
    return sorted(b for _, _, b in cand[:want])


def half_beat_attribution(band_fluxes: dict, frame_times, bpm: float, first: float,
                          bars: Sequence[int], beats_per_bar: int = 4,
                          half_window: float = 0.030,
                          ratio_gate: float = TEMPO_ATTR_RATIO_GATE) -> dict:
    """**半拍归属**：`first` 到底是拍线，还是差半拍的那条反拍线？

    对每个频带，比较 `first` 的整拍线与 `first + 半拍` 的线上的平均 flux。

    - **kick 与 hat 说了算**：这两件几乎总在拍上（4-on-the-floor + 8 分 hat）；
    - **bass 只作反证**：很多曲风（本曲即是）用 **offbeat bass**——
      kick 在拍上、bass stab 在 8 分反拍上。bass 指向反拍**不是**反例，
      反而是 kick 判定正确的佐证；因此 bass 不参与投票，只写进理由。

    返回 dict：逐带的 `on` / `off` / `ratio`，`verdict`（"first" / "first+half" /
    "undecided"）、`alt_first`、`reason`。
    """
    ft = np.asarray(frame_times, dtype=float)
    spb = 60.0 / float(bpm)
    bar_sec = beats_per_bar * spb
    lines = np.concatenate([[first + (b - 1) * bar_sec + k * spb
                             for k in range(int(beats_per_bar))] for b in bars]) \
        if len(bars) else np.zeros(0)
    per_band: dict[str, dict] = {}
    for name, fl in band_fluxes.items():
        if lines.size == 0:
            per_band[name] = {"on": 0.0, "off": 0.0, "ratio": float("nan"), "n": 0}
            continue
        on = float(sample_flux_at(fl, ft, lines, half_window).mean())
        off = float(sample_flux_at(fl, ft, lines + spb / 2.0, half_window).mean())
        per_band[name] = {"on": on, "off": off,
                          "ratio": float(on / off) if off > 1e-9 else float("inf"),
                          "n": int(lines.size)}
    votes_on = [n for n in ("kick", "hat")
                if n in per_band and per_band[n]["ratio"] >= ratio_gate]
    votes_off = [n for n in ("kick", "hat")
                 if n in per_band and per_band[n]["ratio"] <= 1.0 / ratio_gate]
    if votes_on and not votes_off:
        verdict = "first"
    elif votes_off and not votes_on:
        verdict = "first+half"
    else:
        verdict = "undecided"
    bass = per_band.get("bass")
    bass_note = ""
    if bass and bass["ratio"] == bass["ratio"]:
        if verdict == "first" and bass["ratio"] <= 1.0 / ratio_gate:
            bass_note = ("；bass 指向反拍（比 %.2f）——这是 **offbeat bass** 曲风特征"
                         "（kick 在拍上、bass stab 在 8 分反拍上），不是反例"
                         % bass["ratio"])
        elif verdict == "first+half" and bass["ratio"] >= ratio_gate:
            bass_note = "；bass 指向原 first 线，同样可用 offbeat bass 解释"
    detail = "，".join(f"{k} 正/反 {v['on']:.2f}/{v['off']:.2f} dB（比 {v['ratio']:.2f}）"
                      for k, v in per_band.items())
    return {
        "available": bool(len(bars)),
        "bands": per_band,
        "bars_used": [int(b) for b in bars],
        "n_bars_used": int(len(bars)),
        "ratio_gate": float(ratio_gate),
        "verdict": verdict,
        "alt_first": float(first + spb / 2.0),
        "half_beat_ms": float(spb * 500.0),
        "reason": (f"只在 {len(bars)} 个鼓最干净的小节上判（4-on-the-floor 档，"
                   f"不用全曲平均）：{detail}{bass_note}"),
    }


def _beat_this_beats(wav_path: str, device: str = "mps",
                     checkpoint: str = "final0") -> dict:
    """beat_this（MIT）拍点/下拍——**可选**路，没装就如实回报，不抛异常。"""
    try:
        from beat_this.inference import File2Beats  # 惰性、可选
    except Exception as e:                                   # pragma: no cover
        return {"available": False,
                "reason": f"beat_this 未安装（{type(e).__name__}）；"
                          f"`pip install beat-this` 后可用，dbn=True 另需 madmom"}
    last = "未知错误"
    for dev in (device, "cpu"):
        try:
            beats, downbeats = File2Beats(checkpoint_path=checkpoint, device=dev,
                                          dbn=False)(wav_path)
            return {"available": True, "device": dev,
                    "beats": np.asarray(beats, dtype=float),
                    "downbeats": np.asarray(downbeats, dtype=float),
                    "quantization_note": "beat_this 输出量化到 10 ms，"
                                         "逐间隔求 BPM 会有 ±7 BPM 锯齿，须滑动拟合"}
        except Exception as e:                               # pragma: no cover
            last = f"{type(e).__name__}: {e}"
    return {"available": False, "reason": f"beat_this 运行失败：{last}"}


def _librosa_beats(y, sr: int, start_bpm: float, hop: int = 512) -> dict:
    """librosa 拍点——**必须带 hop 帧量化告警**，不得单独作结论。"""
    try:
        import librosa  # 惰性
        env = librosa.onset.onset_strength(y=np.asarray(y, dtype=float), sr=sr,
                                           hop_length=hop, aggregate=np.median)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=env, sr=sr, hop_length=hop, start_bpm=float(start_bpm),
            tightness=400, units="time", trim=False)
    except Exception as e:                                   # pragma: no cover
        return {"available": False, "reason": f"librosa 拍点失败：{type(e).__name__}: {e}"}
    frame_ms = hop / sr * 1000.0
    n = max(1.0, round(60.0 / float(start_bpm) / (hop / sr)))
    quant = 60.0 / (n * hop / sr)
    return {
        "available": True, "tempo": float(np.atleast_1d(tempo)[0]),
        "beats": np.asarray(beats, dtype=float), "hop": int(hop),
        "frame_ms": float(frame_ms),
        "quantized_bpm_near_user": float(quant),
        "warning": (f"⚠️ librosa `beat_track` 的 tempo 被 hop={hop} 的帧栅格量化："
                    f"一帧 {frame_ms:.2f} ms，用户 BPM 附近只能取到 {quant:.3f}。"
                    f"**这个数不是真值，不得单独作结论**；拍点本身仍可用来看相位。"),
    }


def instantaneous_bpm(beat_times, expect_bpm: float, window: int = 17,
                      subdivision: int = 2) -> dict:
    """拍点序列 → 逐拍瞬时 BPM（滑动线性拟合，抗输出量化）。

    先把相邻间隔折算成「最近的整数个 1/subdivision 拍」，再在 `window` 个拍点上
    做线性拟合取斜率。直接用相邻间隔求 BPM 会被 beat_this 的 10 ms 输出栅格
    打成 ±7 BPM 的锯齿（test-01 实测），**必须滑动拟合**。
    """
    t = np.asarray(sorted(np.asarray(beat_times, dtype=float)))
    if t.size < max(8, window // 2):
        return {"available": False, "reason": "拍点太少"}
    unit = 60.0 / float(expect_bpm) / float(subdivision)
    k = np.maximum(1.0, np.round(np.diff(t) / unit))
    idx = np.concatenate([[0.0], np.cumsum(k)])
    bpm = np.full(t.size, np.nan)
    for i in range(t.size):
        a, b = max(0, i - window // 2), min(t.size, i + window // 2 + 1)
        if b - a >= 8 and idx[b - 1] - idx[a] > 4:
            slope = float(np.polyfit(idx[a:b], t[a:b], 1)[0])
            if slope > 1e-9:
                bpm[i] = 60.0 / (slope * float(subdivision))
    ok = ~np.isnan(bpm)
    if not ok.any():
        return {"available": False, "reason": "滑动拟合无有效点"}
    return {"available": True, "times": t[ok], "bpm": bpm[ok],
            "median": float(np.median(bpm[ok])),
            "p5": float(np.percentile(bpm[ok], 5)),
            "p95": float(np.percentile(bpm[ok], 95)),
            "off_by_1p5_ratio": float(np.mean(np.abs(bpm[ok] - expect_bpm) > 1.5))}


# ---------- 变速分段（判为变速时才用）----------

def segment_tempo(windows: Sequence[dict], bpm: float, first: float,
                  beats_per_bar: int = 4, tol: float = TEMPO_CONST_TOL_BPM,
                  min_run: int = 4) -> list[dict]:
    """把逐窗 BPM 曲线切成若干个「常速段」，输出 `(bpm)` 标记序列的素材。

    做法刻意简单（中值平滑 + 贪心聚段 + 丢掉过短的段），因为一旦判为变速，
    **精确变速点必须人工确认**；本函数只给候选。

    返回 [{start_bar, end_bar, bpm, n_windows}]。
    """
    vals = [(w["bar"], w["best_local"]) for w in windows
            if w.get("valid") and w.get("best_local") is not None]
    if len(vals) < min_run:
        return []
    bars = np.array([v[0] for v in vals])
    y = np.array([v[1] for v in vals], dtype=float)
    # 中值平滑（长度 5，纯 numpy）
    k = 5
    pad = np.pad(y, (k // 2, k // 2), mode="edge")
    ys = np.array([np.median(pad[i:i + k]) for i in range(y.size)])
    segs: list[dict] = []
    s = 0
    for i in range(1, ys.size + 1):
        if i == ys.size or abs(ys[i] - np.median(ys[s:i])) > tol:
            segs.append({"start_bar": int(bars[s]), "end_bar": int(bars[i - 1]),
                         "bpm": float(np.median(ys[s:i])), "n_windows": int(i - s)})
            s = i
    segs = [g for g in segs if g["n_windows"] >= min_run]
    # 相邻同速段合并
    merged: list[dict] = []
    for g in segs:
        if merged and abs(merged[-1]["bpm"] - g["bpm"]) <= tol:
            merged[-1]["end_bar"] = g["end_bar"]
            merged[-1]["n_windows"] += g["n_windows"]
            continue
        merged.append(dict(g))
    return merged


def bpm_changes_from_segments(segments: Sequence[dict], base_bpm: float,
                              decimals: int = 2) -> list[BpmChange]:
    """常速段列表 → `BpmChange`（可直接喂 `Grid` / 写成 simai `(bpm)` 标记）。"""
    out: list[BpmChange] = []
    prev = round(float(base_bpm), decimals)
    for g in segments:
        b = round(float(g["bpm"]), decimals)
        if g["start_bar"] > 1 and abs(b - prev) > 10.0 ** (-decimals):
            out.append(BpmChange(bar=int(g["start_bar"]), bpm=b))
            prev = b
    return out


def bar_start_table(bpm: float, first: float, beats_per_bar: int = 4,
                    bpm_changes: Sequence[BpmChange] = (), n_bars: int = 128) -> list[dict]:
    """按（可能变速的）tempo map 重算 1..n_bars 的小节起点秒。"""
    t = float(first)
    cur = float(bpm)
    cmap = {c.bar: float(c.bpm) for c in bpm_changes}
    rows: list[dict] = []
    for bar in range(1, int(n_bars) + 1):
        if bar in cmap:
            cur = cmap[bar]
        rows.append({"bar": bar, "start_sec": round(t, 4), "bpm": cur})
        t += beats_per_bar * 60.0 / cur
    return rows


# ---------- 总装 ----------

def tempo_map(wav_path: str | None = None, *, y_mix=None, y_drums=None, y_bass=None,
              sr: int = TEMPO_SR, bpm: float = 0.0, first: float = 0.0,
              beats_per_bar: int = 4, duration: float | None = None,
              body_range: tuple[float, float] | None = None,
              scan_lo: float = TEMPO_SCAN_LO, scan_hi: float = TEMPO_SCAN_HI,
              coarse_step: float = TEMPO_COARSE_STEP,
              fine_step: float = TEMPO_FINE_STEP,
              window_bars: Sequence[int] = TEMPO_WINDOW_BARS,
              use_beat_this: bool = True, use_librosa: bool = True,
              beat_this_device: str = "mps",
              n_bars: int | None = None) -> dict:
    """**tempo map 复核总装**：恒定 / 变速判定 + 半拍归属 + 无鼓段标注。

    输入（三选一，优先级从上到下）：
      - `y_drums` + `y_mix`：已解码的单声道波形（推荐，管线里 stems 已经有了）；
      - `y_mix` 单独给：全混回退，结论置信度自动降一档；
      - `wav_path`：自己读（只在没给波形时）。

    判「恒定」需要**三条同时成立**：
      ① 有效窗里 ≥`TEMPO_CONST_HIT_RATIO` 落在逐窗中位 ±`TEMPO_CONST_TOL_BPM`；
      ② 主体段高精度拟合的峰落在用户 BPM ±`TEMPO_CONST_TOL_BPM`；
      ③ 前后半独立拟合差 ≤ `TEMPO_HALF_SPLIT_TOL`。
    另外相位曲线不得有跳格（`detect_phase_jumps` 为空）且回归斜率过门槛。

    返回一个可直接塞进 `song_analysis.json` 的 dict（见 `verdict` / `bpm` /
    `windows` / `phase` / `half_beat` / `no_drum_bars` / `cross_check`）。
    """
    import numpy as _np  # 局部别名，避免与调用方的 np 混淆
    notes: list[str] = []
    if bpm <= 0:
        raise ValueError("tempo_map 需要用户给定的 bpm")
    bar_sec = beats_per_bar * 60.0 / float(bpm)

    # --- 1. 取波形 ---
    if y_mix is None and wav_path:
        import librosa  # 惰性
        y_mix, sr = librosa.load(str(wav_path), sr=TEMPO_SR, mono=True)
    if y_mix is None and y_drums is None:
        return {"available": False, "reason": "既没有波形也没有 wav_path"}

    primary_name = "drums" if y_drums is not None else "mix"
    if y_drums is None:
        notes.append("⚠️ 没有 drums stem，退回全混：全混 onset 在混响 pad / 渐弱段上"
                     "噪声很大，逐窗曲线的离散度不可当变速证据。")

    if duration is None:
        ref = y_drums if y_drums is not None else y_mix
        duration = float(len(ref)) / float(sr)
    if n_bars is None:
        n_bars = max(1, int((duration - first) / bar_sec))

    # --- 2. 高分辨率 onset（主 = drums，辅 = 全混带通）---
    o_main = high_res_onsets(y_drums if y_drums is not None else y_mix, sr=sr)
    o_mix = high_res_onsets(y_mix, sr=sr, fmin=60.0, fmax=6000.0) if y_mix is not None else None

    # --- 3. 无鼓段标注（这些小节一律不做 onset 验证）---
    tt = o_main["times"]
    per_bar = _np.array([int(((tt >= first + i * bar_sec) &
                              (tt < first + (i + 1) * bar_sec)).sum())
                         for i in range(int(n_bars))])
    no_drum = [int(i + 1) for i in range(int(n_bars))
               if per_bar[i] < TEMPO_NO_DRUM_MIN_ONSETS]
    if no_drum:
        notes.append(f"{len(no_drum)}/{n_bars} 小节判为无鼓/无瞬态，**不做 onset 验证**"
                     f"（旧版的「无鼓段单独测相位 → 确认同一网格」已作废："
                     f"那条 onset 列表在混响 pad 上比随机还差）。这些小节按主体网格外推。")

    # --- 4. 主体段（连续的有鼓区间，用来做高精度拟合）---
    if body_range is None:
        has = _np.flatnonzero(per_bar >= TEMPO_NO_DRUM_MIN_ONSETS)
        if has.size >= 8:
            body_range = (float(first + has[0] * bar_sec),
                          float(first + (has[-1] + 1) * bar_sec))
        else:
            body_range = (float(first), float(first + n_bars * bar_sec))
    body = fine_bpm_fit(o_main["times"], o_main["weights"], body_range[0], body_range[1],
                        max(scan_lo, bpm - 5.0), min(scan_hi, bpm + 5.0), fine_step,
                        subdivision=2, report_at=(bpm,))
    # 一致性检验要围绕**实测**的 BPM 做：用户值错了的时候，
    # 围绕用户值的窄窗会把两半都顶到窗边，检验就失去意义。
    split_center = body["bpm"] if body.get("available") else bpm
    split = half_split_consistency(o_main["times"], o_main["weights"],
                                   body_range[0], body_range[1], split_center)

    # --- 5. 无相位滑窗全域扫描 ---
    windows: dict[str, list[dict]] = {}
    for wb in window_bars:
        windows[f"{wb}bar"] = window_bpm_scan(
            o_main["times"], o_main["weights"], bpm=bpm, first=first,
            beats_per_bar=beats_per_bar, n_bars=n_bars, window_bars=int(wb),
            scan_lo=scan_lo, scan_hi=scan_hi, coarse_step=coarse_step,
            fine_step=fine_step)
    if o_mix is not None and primary_name == "drums":
        windows["4bar_mix"] = window_bpm_scan(
            o_mix["times"], o_mix["weights"], bpm=bpm, first=first,
            beats_per_bar=beats_per_bar, n_bars=n_bars, window_bars=4,
            scan_lo=scan_lo, scan_hi=scan_hi, coarse_step=coarse_step,
            fine_step=fine_step)

    ref = windows.get(f"{window_bars[0]}bar", [])
    loc = _np.array([w["best_local"] for w in ref if w.get("valid")], dtype=float)
    stats = {}
    if loc.size:
        med = float(_np.median(loc))
        hit = float(_np.mean(_np.abs(loc - med) <= TEMPO_CONST_TOL_BPM))
        stats = {"n_valid": int(loc.size), "n_windows": len(ref),
                 "median": med, "mean": float(loc.mean()), "std": float(loc.std()),
                 "min": float(loc.min()), "max": float(loc.max()),
                 "hit_ratio_within_tol": hit,
                 "hit_ratio_around_user": float(_np.mean(
                     _np.abs(loc - bpm) <= TEMPO_CONST_TOL_BPM))}

    # --- 6. 相位（8 分量程）与跳格 ---
    phase = bar_phase_curve(o_main["times"], o_main["weights"], bpm=bpm, first=first,
                            beats_per_bar=beats_per_bar, n_bars=n_bars, subdivision=2)
    jumps = detect_phase_jumps(phase)
    residual = beat_residual_curve(o_main["times"], o_main["weights"], bpm=bpm,
                                   first=first, beats_per_bar=beats_per_bar,
                                   duration=duration)

    # --- 7. 半拍归属（只在鼓最干净的小节上判）---
    half = {"available": False, "reason": "没有 drums stem，半拍归属不判（全混分不开鼓件）"}
    if y_drums is not None:
        fl = {n: band_flux(y_drums if n != "bass" else (y_bass if y_bass is not None else y_mix),
                           sr=sr, f_lo=lo_, f_hi=hi_)
              for n, (lo_, hi_) in TEMPO_BANDS.items()
              if n != "bass" or (y_bass is not None or y_mix is not None)}
        ft = fl["kick"]["frame_times"]
        clean = cleanest_drum_bars(fl["kick"]["flux"], ft, bpm=bpm, first=first,
                                   beats_per_bar=beats_per_bar, n_bars=int(n_bars))
        if clean:
            half = half_beat_attribution({k: v["flux"] for k, v in fl.items()}, ft,
                                         bpm=bpm, first=first, bars=clean,
                                         beats_per_bar=beats_per_bar)
        else:
            half = {"available": False,
                    "reason": f"找不到 kick 打 {TEMPO_CLEAN_KICK_LO}–{TEMPO_CLEAN_KICK_HI} "
                              f"下的干净小节（全曲要么无鼓要么 kick 打满）；"
                              f"**不在全曲平均上硬判**"}

    # --- 8. 旁证：beat_this / librosa ---
    cross: dict[str, dict] = {}
    if use_beat_this:
        bt = _beat_this_beats(str(wav_path), device=beat_this_device) if wav_path else \
            {"available": False, "reason": "需要 wav_path 才能跑 beat_this"}
        if bt.get("available"):
            bt["instantaneous"] = instantaneous_bpm(bt["beats"], bpm)
            bt["beats"] = bt["beats"].tolist()
            bt["downbeats"] = bt["downbeats"].tolist()
            inst = bt["instantaneous"]
            if inst.get("available"):
                inst["times"] = inst["times"].tolist()
                inst["bpm"] = inst["bpm"].tolist()
        cross["beat_this"] = bt
    else:
        cross["beat_this"] = {"available": False, "reason": "调用方关闭（--no-beat-this）"}
    if use_librosa and y_mix is not None:
        lb = _librosa_beats(y_mix, sr=sr, start_bpm=bpm)
        if lb.get("available"):
            lb["instantaneous"] = instantaneous_bpm(lb["beats"], bpm)
            lb["beats"] = lb["beats"].tolist()
            inst = lb["instantaneous"]
            if inst.get("available"):
                inst["times"] = inst["times"].tolist()
                inst["bpm"] = inst["bpm"].tolist()
        cross["librosa"] = lb
    cross["note"] = ("beat_this 与 librosa 都只作**旁证**：前者输出量化到 10 ms，"
                     "后者 tempo 被 hop 帧栅格量化，**任何一条都不得单独作结论**。")

    # --- 9. 判定 ---
    c1 = bool(stats) and stats["hit_ratio_within_tol"] >= TEMPO_CONST_HIT_RATIO
    c2 = bool(body.get("available")) and abs(body["bpm"] - bpm) <= TEMPO_CONST_TOL_BPM
    c3 = bool(split.get("consistent"))
    c4 = not jumps
    slope = phase.get("slope", float("nan"))
    c5 = (slope != slope) or abs(slope) < TEMPO_SLOPE_GATE
    constant = c1 and c2 and c3 and c4 and c5
    criteria = {
        "windows_within_tol": {"pass": bool(c1),
                               "value": stats.get("hit_ratio_within_tol"),
                               "gate": TEMPO_CONST_HIT_RATIO},
        "body_fit_matches_user": {"pass": bool(c2), "value": body.get("bpm"),
                                  "gate": TEMPO_CONST_TOL_BPM},
        "half_split_consistent": {"pass": bool(c3), "value": split.get("delta_bpm"),
                                  "gate": TEMPO_HALF_SPLIT_TOL},
        "no_phase_jump": {"pass": bool(c4), "value": len(jumps)},
        "phase_slope_ok": {"pass": bool(c5), "value": slope, "gate": TEMPO_SLOPE_GATE},
    }

    segments: list[dict] = []
    changes: list[BpmChange] = []
    table: list[dict] = []
    if not constant and ref:
        segments = segment_tempo(ref, bpm=bpm, first=first, beats_per_bar=beats_per_bar)
        changes = bpm_changes_from_segments(segments, base_bpm=(
            segments[0]["bpm"] if segments else bpm))
        table = bar_start_table(
            bpm=(segments[0]["bpm"] if segments else bpm), first=first,
            beats_per_bar=beats_per_bar, bpm_changes=changes, n_bars=int(n_bars))

    # 「曲子本身恒定，但不等于用户给的 BPM」要单独成一档——
    # 这正是旧法（32 分网格块相位回归）完全测不出的那类错误：
    # 差 0.5% 时一块之内相位就绕过一整格，回归斜率看起来反而很小。
    detected = body.get("bpm") if body.get("available") else None
    constant_elsewhere = (
        not constant and c1 and c3 and detected is not None
        and abs(detected - bpm) > TEMPO_CONST_TOL_BPM
        and abs(float(stats.get("median", detected)) - detected) <= TEMPO_CONST_TOL_BPM)
    # v1.5.1（test-01 事故）：**可测小节占比不够时不得报 constant**。
    # 旧行为把「无鼓段测不到」直接当成「无鼓段也是这个速度」外推，
    # 结果把一首前奏 9 小节 195 BPM、尾奏 7 小节 221→108 的曲子判成「全曲恒定 205」。
    measured_ratio = 1.0 - (len(no_drum) / max(int(n_bars), 1))
    enough_measured = measured_ratio >= TEMPO_MEASURED_RATIO_GATE
    if constant and not enough_measured:
        constant = False
        constant_in_measured = True
    else:
        constant_in_measured = False
    if constant:
        verdict = "constant"
    elif constant_in_measured:
        verdict = "constant_in_measured"
    elif constant_elsewhere:
        verdict = "constant_other_bpm"
    elif segments and len(segments) >= 2:
        verdict = "variable"
    else:
        verdict = "undetermined"
    drift_ms = (abs((detected or bpm) - bpm) / bpm * (duration - first) * 1000.0
                if detected else 0.0)

    return _json_safe({
        "available": True,
        "version": "1.5",
        "verdict": verdict,
        "user_bpm": float(bpm), "user_first": float(first),
        "detected_bpm": detected,
        "drift_vs_user_ms": float(drift_ms),
        "beats_per_bar": int(beats_per_bar), "n_bars": int(n_bars),
        "primary_source": primary_name,
        "body_range_sec": [float(body_range[0]), float(body_range[1])],
        "body_fit": body, "half_split": split,
        "window_stats": stats, "windows": windows,
        "phase": {k: (v.tolist() if isinstance(v, _np.ndarray) else v)
                  for k, v in phase.items()},
        "phase_jumps": jumps,
        "beat_residual": {k: (v.tolist() if isinstance(v, _np.ndarray) else v)
                          for k, v in residual.items()},
        "half_beat": half,
        "no_drum_bars": no_drum,
        "no_drum_policy": "无鼓段不做 onset 验证，按主体网格外推（v1.5 起）",
        "criteria": criteria,
        "measured_ratio": float(measured_ratio),
        "measured_ratio_gate": float(TEMPO_MEASURED_RATIO_GATE),
        "unmeasured_ranges": _contiguous(no_drum),
        "measured_ranges": _contiguous([b for b in range(1, int(n_bars) + 1)
                                        if b not in set(no_drum)]),
        "segments": segments,
        "bpm_changes": [{"bar": c.bar, "bpm": c.bpm} for c in changes],
        "bar_start_table": table,
        "scan": {"lo": float(scan_lo), "hi": float(scan_hi),
                 "coarse_step": float(coarse_step), "fine_step": float(fine_step),
                 "window_bars": [int(x) for x in window_bars],
                 "phase_free": True},
        "cross_check": cross,
        "notes": notes,
        "replaces": "旧的「8/16 小节块 + 32 分网格 ±17 ms 相位回归」——"
                    "该法对整格错位与 >0.4% 的 BPM 误差都不敏感，已作废",
    })


def _contiguous(bars: Sequence[int]) -> list[list[int]]:
    """把小节号列表压成连续区间 [[a,b], ...]。"""
    out: list[list[int]] = []
    for b in sorted(int(x) for x in bars):
        if out and b == out[-1][1] + 1:
            out[-1][1] = b
        else:
            out.append([b, b])
    return out


def _json_safe(obj):
    """把 NaN / ±Inf / numpy 标量换成 JSON 里安全的值。

    `song_analysis.json` 被 `tools/chart_check` 等下游直接吃，
    写出 `NaN` / `Infinity` 这种非标准字面量会坑到别的语言的解析器。
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if f != f:
            return None
        if f == float("inf"):
            return 1e9
        if f == float("-inf"):
            return -1e9
        return f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def tempo_map_verdict(tm: dict) -> str:
    """把 `tempo_map` 的结果翻成一句中文结论（给 CLI 与分析单用）。"""
    if not tm.get("available"):
        return "tempo map 不可用：" + str(tm.get("reason", ""))
    st = tm.get("window_stats") or {}
    body = tm.get("body_fit") or {}
    sp = tm.get("half_split") or {}
    head = ""
    if tm["verdict"] == "constant":
        head = (f"**BPM {tm['user_bpm']:.3f} 恒定，无变速**（四条判据全过）")
    elif tm["verdict"] == "constant_in_measured":
        rng = "、".join(f"{a}–{b}" for a, b in (tm.get("unmeasured_ranges") or [])[:6])
        head = (f"**BPM {tm['user_bpm']:.3f} 在【可测区间】内恒定**——"
                f"但只有 {tm.get('measured_ratio', 0)*100:.0f}% 的小节可测"
                f"（门槛 {tm.get('measured_ratio_gate', 0)*100:.0f}%），"
                f"**不可测小节：{rng}**。"
                f"⚠️ **这些小节的速度既没测出来、也不得外推为同一 BPM**——"
                f"前奏/尾奏很可能是变速（test-01 就栽在这里）；"
                f"请用别的证据（抄谱、官方 BPM 表、人工听）补，或用 "
                f"`test_tempo_hypothesis` 检验候选 tempo map")
    elif tm["verdict"] == "constant_other_bpm":
        head = (f"⚠️ **曲子速度是稳的，但不等于用户给的 BPM**："
                f"实测 **{tm['detected_bpm']:.3f}**，用户给的是 {tm['user_bpm']:.3f}"
                f"（差 {tm['detected_bpm'] - tm['user_bpm']:+.3f}）。"
                f"照用户值铺网格，到曲末会累计偏离约 "
                f"**{tm['drift_vs_user_ms']:.0f} ms**，请改 --bpm 后重跑")
    elif tm["verdict"] == "variable":
        segs = "；".join(f"bars {g['start_bar']}–{g['end_bar']} ≈ {g['bpm']:.2f}"
                         for g in tm.get("segments", []))
        head = f"**检出变速**：{segs}。`(bpm)` 标记见 `bpm_changes`，小节起点秒表见 `bar_start_table`"
    else:
        fail = [k for k, v in (tm.get("criteria") or {}).items() if not v.get("pass")]
        head = (f"**无法判定**（未过的判据：{', '.join(fail) if fail else '证据不足'}）；"
                f"已按用户 BPM 构造网格，请人工复核")
    detail = ""
    if st:
        detail += (f"；{st['window_bars'] if 'window_bars' in st else ''}"
                   f"逐窗最佳 BPM 中位 {st['median']:.3f}、std {st['std']:.3f}、"
                   f"{st['n_valid']}/{st['n_windows']} 窗有效，"
                   f"{st['hit_ratio_within_tol']*100:.0f}% 落在中位 ±{TEMPO_CONST_TOL_BPM}")
    if body.get("available"):
        detail += (f"；主体段拟合 {body['bpm']:.3f}"
                   f"（95% 峰宽 {body['peak95_lo']:.3f}–{body['peak95_hi']:.3f}）")
    if sp.get("available"):
        detail += (f"；前后半独立拟合差 {sp['delta_bpm']:.3f}"
                   f"（门槛 {sp['tolerance']:.2f}）")
    jm = tm.get("phase_jumps") or []
    detail += ("；相位曲线（8 分量程 ±%.0f ms）无跳格" % (tm.get("phase", {}).get("range_ms", 0.0))
               if not jm else
               "；⚠️ 相位曲线检出 %d 处跳格：%s" % (
                   len(jm), ", ".join(f"bar{j['to_bar']}({j['jump_ms']:+.0f}ms)" for j in jm[:5])))
    hb = tm.get("half_beat") or {}
    if hb.get("available"):
        if hb["verdict"] == "first":
            detail += (f"；半拍归属 **维持 first**（kick/hat 在拍上，"
                       f"只用了 {hb['n_bars_used']} 个干净小节）")
        elif hb["verdict"] == "first+half":
            detail += (f"；⚠️ 半拍归属指向 **first+半拍 = {hb['alt_first']:.4f}s**，"
                       f"当前 first 可能整体错半拍")
        else:
            detail += "；半拍归属**不可判定**（kick/hat 正反拍能量接近）"
    else:
        detail += f"；半拍归属未判（{hb.get('reason', '')}）"
    nd = tm.get("no_drum_bars") or []
    if nd:
        detail += (f"；{len(nd)} 个小节无鼓/无瞬态 → **不可测，按主体网格外推**"
                   f"（不作为变速证据）")
    return head + detail + "。"


# =====================================================================
# v1.5.1：候选 tempo map 假设检验（test-01 事故的直接产物）
# =====================================================================
#
# 背景：前奏 / 尾奏这类无瞬态段落**自己测不出速度**。旧做法（外推为同一 BPM）
# 在 test-01 上把「前奏 9 小节 195 + 减速 8 小节 + 尾奏 7 小节 221→108」
# 整个抹平成「全曲恒定 205」。正确姿势是：从**别处**拿一个候选 tempo map
# （抄谱、官方 BPM 表、人工听），在自己的音频上逐段检验它 ——
# 分得开就报实测值，分不开就如实写「音频无法分辨，采用候选值」。


def tempo_segments_to_beats(first: float, segments: Sequence[tuple[float, float]],
                            beats_per_bar: int = 4) -> tuple[np.ndarray, list[dict], float]:
    """分段 BPM 表 → 拍线数组。

    `segments` 是 [(bpm, n_bars), ...]，`n_bars` 允许半小节（0.5）——
    真实曲子的减速段常常半小节换一次速（test-01 的 bar12/bar13 就是）。

    返回 (beat_times, seg_info, end_time)。
    """
    t = float(first)
    beats: list[float] = []
    info: list[dict] = []
    for bpm, nb in segments:
        spb = 60.0 / float(bpm)
        n = int(round(float(nb) * beats_per_bar))
        info.append({"bpm": float(bpm), "n_bars": float(nb), "start_sec": t,
                     "dur_sec": n * spb, "end_sec": t + n * spb,
                     "start_bar": len(beats) // beats_per_bar + 1,
                     "start_beat_in_bar": len(beats) % beats_per_bar})
        beats.extend(t + k * spb for k in range(n))
        t += n * spb
    return np.asarray(beats, dtype=float), info, float(t)


def _line_score(lines: np.ndarray, t: np.ndarray, w: np.ndarray,
                t0: float, t1: float, sigma: float) -> tuple[float | None, int]:
    m = (t >= t0) & (t < t1)
    if int(m.sum()) < 5:
        return None, int(m.sum())
    tt, ww = t[m], w[m] / max(w[m].sum(), 1e-12)
    L = np.sort(lines)
    i = np.searchsorted(L, tt)
    d = np.full(tt.size, np.inf)
    for off in (-1, 0):
        j = np.clip(i + off, 0, L.size - 1)
        d = np.minimum(d, np.abs(tt - L[j]))
    return float((ww * np.exp(-(d ** 2) / (2 * sigma ** 2))).sum()), int(m.sum())


def _median_dist(a: np.ndarray, lines: np.ndarray) -> float:
    if a.size == 0 or lines.size == 0:
        return float("nan")
    L = np.sort(lines)
    i = np.searchsorted(L, a)
    d = np.full(a.size, np.inf)
    for off in (-1, 0):
        j = np.clip(i + off, 0, L.size - 1)
        d = np.minimum(d, np.abs(a - L[j]))
    return float(np.median(d))


def test_tempo_hypothesis(onset_times, onset_weights, first: float,
                          segments: Sequence[tuple[float, float]],
                          beats_per_bar: int = 4, beat_times=None,
                          sigma: float = 0.018,
                          baseline_bpm: float | None = None,
                          local_fit_span: float = 12.0,
                          min_onsets_fit: int = 40,
                          decisive_margin: float = 0.15,
                          min_dur_decisive: float = 4.0,
                          min_onsets_decisive: int = 80,
                          min_score_decisive: float = 0.35) -> dict:
    """检验一个**候选 tempo map**，并与「恒速外推」对照。

    参数：
        onset_times / onset_weights: 本曲的 onset（建议用 `high_res_onsets`）
        first: 候选 map 里第 1 小节第 1 拍的秒数
        segments: [(bpm, n_bars), ...]，n_bars 可为 0.5
        beat_times: 可选，训练过的跟踪器给的拍点（如 beat_this），作旁证
        baseline_bpm: 恒速对照用的 BPM（默认取 segments 里占小节最多的那个）
        decisive_margin: 候选段得分比恒速高出这个相对比例才算「分得开」

    每段的判定：
      - `measured`：该段自己独立细扫出来的 BPM 与候选值一致（且事件够多）；
      - `confirmed`：独立扫不动，但候选网格明显优于恒速外推；
      - `adopted`：分不开 → **音频无法分辨，采用候选值**；
      - `contradicted`：恒速外推明显更好 → 候选值可疑。

    **只有同时满足三条「可判据」的段才允许出 `confirmed` / `contradicted`**，
    否则一律 `adopted`：
      ① 段长 ≥ `min_dur_decisive`（默认 4 s）——1 小节（约 1.2 s）在物理上
         就分不开 185 与 205，硬判出来的都是噪声；
      ② 段内 onset ≥ `min_onsets_decisive`（默认 80）；
      ③ 两套网格里**较好的那个**得分 ≥ `min_score_decisive`（默认 0.35）——
         得分都很低说明这段的 onset 本身就是噪声（混响 pad / 低通渐弱段的
         onset 列表实测比随机还差），谁高谁低没有意义。
    test-01 的教训：不加这三条，17 段里会有 9 段被误判成 `contradicted`。

    ⚠️ 本函数**不替你做决定**：`adopted` 就是 `adopted`，下游必须照实写明
    「音频无法分辨、采用候选值」，不得升格成实测结论。
    """
    t = np.asarray(onset_times, dtype=float)
    w = (np.ones_like(t) if onset_weights is None
         else np.asarray(onset_weights, dtype=float))
    beats, info, end = tempo_segments_to_beats(first, segments, beats_per_bar)
    if beats.size < 4:
        return {"available": False, "reason": "候选 map 太短"}
    mid = (beats[:-1] + beats[1:]) / 2.0
    lines_h = np.sort(np.concatenate([beats, mid]))

    if baseline_bpm is None:
        tally: dict[float, float] = {}
        for bpm, nb in segments:
            tally[float(bpm)] = tally.get(float(bpm), 0.0) + float(nb)
        baseline_bpm = max(tally.items(), key=lambda kv: kv[1])[0]
    n_beats_base = int(np.ceil((end - first) / (60.0 / baseline_bpm))) + 4
    b_base = first + np.arange(n_beats_base) * (60.0 / baseline_bpm)
    lines_b = np.sort(np.concatenate([b_base, (b_base[:-1] + b_base[1:]) / 2.0]))

    bt = np.asarray(beat_times, dtype=float) if beat_times is not None else None
    rows: list[dict] = []
    for s in info:
        a, z = s["start_sec"], s["end_sec"]
        sh, n = _line_score(lines_h, t, w, a, z, sigma)
        sb, _ = _line_score(lines_b, t, w, a, z, sigma)
        row = {**s, "n_onsets": n, "score_hypothesis": sh, "score_baseline": sb}
        # 本段能不能自己扫出 BPM
        local = None
        if n >= min_onsets_fit and (z - a) >= 4.0:
            local = fine_bpm_fit(t, w, a, z,
                                 max(40.0, s["bpm"] - local_fit_span),
                                 s["bpm"] + local_fit_span, TEMPO_FINE_STEP,
                                 subdivision=2, report_at=(s["bpm"], baseline_bpm))
        row["local_fit"] = local
        if bt is not None:
            m = (bt >= a) & (bt < z)
            row["beat_tracker_median_ms_hyp"] = _median_dist(bt[m], beats) * 1000.0
            row["beat_tracker_median_ms_base"] = _median_dist(bt[m], b_base) * 1000.0
        decisive = (sh is not None and sb is not None
                    and (z - a) >= min_dur_decisive
                    and n >= min_onsets_decisive
                    and max(sh, sb) >= min_score_decisive)
        row["decisive"] = bool(decisive)
        local_ok = False
        if local and local.get("available"):
            sc_c = local["score_at"].get(f"{s['bpm']:.3f}", 0.0)
            sc_b = local["score_at"].get(f"{baseline_bpm:.3f}", 0.0)
            local_ok = (abs(local["bpm"] - s["bpm"]) <= 0.5
                        and abs(local["bpm"] - baseline_bpm) > 0.5
                        and sc_c > sc_b * (1.0 + decisive_margin))
        if sh is None or sb is None:
            row["verdict"] = "no_evidence"
            row["note"] = "该段事件太少，音频给不出任何判据 → 采用候选值"
        elif local_ok:
            row["verdict"] = "measured"
            row["note"] = (f"本段独立细扫得 {local['bpm']:.2f}，与候选值 {s['bpm']:.0f} 相符"
                           f"（@候选 {local['score_at'].get(f'{s['bpm']:.3f}', 0):.3f} vs "
                           f"@恒速 {local['score_at'].get(f'{baseline_bpm:.3f}', 0):.3f}）")
        elif decisive and sb > 0 and (sh - sb) / max(sb, 1e-9) > decisive_margin:
            row["verdict"] = "confirmed"
            row["note"] = f"候选网格得分比恒速外推高 {(sh - sb) / sb * 100:.0f}%"
        elif decisive and sh > 0 and (sb - sh) / max(sh, 1e-9) > decisive_margin:
            row["verdict"] = "contradicted"
            row["note"] = f"恒速外推反而高 {(sb - sh) / sh * 100:.0f}%，候选值可疑"
        elif not decisive:
            why = []
            if (z - a) < min_dur_decisive:
                why.append(f"段长 {z - a:.2f}s < {min_dur_decisive}s（物理上分不开）")
            if n < min_onsets_decisive:
                why.append(f"onset 只有 {n} 个")
            if sh is not None and sb is not None and max(sh, sb) < min_score_decisive:
                why.append(f"两套网格得分都只有 {max(sh, sb):.2f}（onset 本身是噪声）")
            row["verdict"] = "adopted"
            row["note"] = "**音频无法分辨，采用候选值**：" + "；".join(why)
        else:
            row["verdict"] = "adopted"
            row["note"] = ("候选与恒速外推的差 < "
                           f"{decisive_margin*100:.0f}% → **音频无法分辨，采用候选值**")
        rows.append(row)

    all_h, n_all = _line_score(lines_h, t, w, first, end, sigma)
    all_b, _ = _line_score(lines_b, t, w, first, end, sigma)
    out = {
        "available": True, "first": float(first), "end_sec": float(end),
        "n_bars": float(sum(nb for _, nb in segments)),
        "baseline_bpm": float(baseline_bpm),
        "overall_score_hypothesis": all_h, "overall_score_baseline": all_b,
        "overall_n_onsets": n_all,
        "segments": rows,
        "counts": {k: sum(1 for r in rows if r["verdict"] == k)
                   for k in ("measured", "confirmed", "adopted",
                             "contradicted", "no_evidence")},
        "decisive_margin": float(decisive_margin),
        "policy": "adopted / no_evidence 的段必须如实标注「音频无法分辨，采用候选值」，"
                  "不得当作实测结论",
    }
    if bt is not None:
        out["beat_tracker_median_ms_hyp"] = _median_dist(
            bt[(bt >= first) & (bt < end)], beats) * 1000.0
        out["beat_tracker_median_ms_base"] = _median_dist(
            bt[(bt >= first) & (bt < end)], b_base) * 1000.0
    return _json_safe(out)


def tempo_hypothesis_verdict(res: dict) -> str:
    """把 `test_tempo_hypothesis` 的结果翻成一句中文结论。"""
    if not res.get("available"):
        return "候选 tempo map 检验不可用：" + str(res.get("reason", ""))
    c = res["counts"]
    parts = [f"{res['n_bars']:.0f} 小节 / {len(res['segments'])} 段",
             f"实测 {c['measured']}", f"网格占优 {c['confirmed']}",
             f"**分不开(采用候选值) {c['adopted'] + c['no_evidence']}**",
             f"被反证 {c['contradicted']}"]
    head = "候选 tempo map：" + "，".join(parts)
    if res.get("overall_score_baseline"):
        d = ((res["overall_score_hypothesis"] - res["overall_score_baseline"])
             / res["overall_score_baseline"] * 100.0)
        head += f"；全曲 8 分格得分比恒速 {res['baseline_bpm']:.0f} 高 {d:+.0f}%"
    if res.get("beat_tracker_median_ms_hyp") is not None:
        head += (f"；拍点跟踪器到候选拍线的中位距 "
                 f"{res['beat_tracker_median_ms_hyp']:.0f} ms vs 恒速 "
                 f"{res['beat_tracker_median_ms_base']:.0f} ms")
    if c["contradicted"]:
        head += "。⚠️ 有段被音频反证，必须人工复核"
    return head + "。"


# =====================================================================
# v1.5.2：下拍相位四选一（test-01 第三次事故的产物）
# =====================================================================
#
# 拍网格定死之后，**哪一拍是第 1 拍**仍然是未定的：四个候选相差整数拍。
# 选错不影响 note 的绝对时刻，但**整谱的强拍位置全错**，小节线、乐句、
# 配置的起手全部偏一拍。test-01 就栽过：我们把乐曲硬切入当成 bar1 beat1，
# 官谱把它当成 bar2 beat1（前面一整小节空拍），整整差一拍。
#
# **最可靠的音乐学判据是 backbeat**：军鼓 / 拍手落在第 2、4 拍。
# 但必须挑**只有军鼓/拍手、没有 4-on-the-floor kick 的段落**来数
# （test-01 的「薄 build」抽掉了 sub 与 kick，只剩 clap）——
# 在 drop 里数会被打满的 kick 和层叠的 layer 抹平（实测 0.64 对 1.56 的差别）。
# backbeat 只能定到 **mod 2**；mod 4 要靠下拍跟踪器、和声变化点、段落进入点。

TEMPO_BACKBEAT_RATIO_GATE = 1.25   # (2&4)/(1&3) 超过它才算「有 backbeat」


def downbeat_phase(band_fluxes: dict, frame_times, bpm: float, first: float,
                   beats_per_bar: int = 4, t_start: float | None = None,
                   t_end: float | None = None,
                   clap_band: str = "snare", kick_band: str = "kick",
                   downbeat_times=None, chroma_change_times=None,
                   section_entry_times=None, half_window: float = 0.030,
                   ratio_gate: float = TEMPO_BACKBEAT_RATIO_GATE) -> dict:
    """在 `beats_per_bar` 个候选相位里挑下拍相位。

    候选 k（k = 0..beats_per_bar−1）表示「真正的小节线比 `first` 早 k 拍」。

    证据（有几路用几路，缺的自动跳过）：
      - **backbeat**（`band_fluxes[clap_band]`）：第 2、4 拍对第 1、3 拍的能量比，
        **只能定 mod 2**；
      - **下拍跟踪器**（`downbeat_times`，如 beat_this）：到各候选小节线的中位距；
      - **和声变化点**（`chroma_change_times`）：同上；
      - **段落进入点**（`section_entry_times`）：同上。

    ⚠️ 调用方应把 `t_start` / `t_end` 限制在**只有军鼓/拍手的段落**上，
    否则 backbeat 一路会被 4-on-the-floor 的 kick 抹平。
    `clap_band` 建议传一条 2–9 kHz 的自定义带（`TEMPO_BANDS` 里没有），
    拍手的噪声爆在那儿最干净。

    ⚠️ **不要拿别人的谱面当兜底**：别人的 `&first` 未必对准过他自己的音频，
    那只是一个**候选假设**。分不开时返回 `undecided`，把候选并列交人裁定。

    返回 dict：逐候选的 `backbeat_ratio` / 各路中位距、`verdict`（"keep" 或
    "shift_k"）、`shift_beats`、`new_first`、`reason`。
    """
    ft = np.asarray(frame_times, dtype=float)
    spb = 60.0 / float(bpm)
    bar = beats_per_bar * spb
    lo = first if t_start is None else float(t_start)
    hi = (float(ft[-1]) if ft.size else lo) if t_end is None else float(t_end)
    n = int(beats_per_bar)

    def bar_lines(p0):
        k0 = int(np.ceil((lo - p0) / bar))
        k1 = int(np.floor((hi - p0) / bar))
        return p0 + np.arange(k0, max(k1, k0) + 1) * bar

    # 拍手事件（主判据用）：在 clap 频带的 flux 上做峰拾取
    clap_ev = np.zeros(0)
    if clap_band in band_fluxes:
        fl0 = np.asarray(band_fluxes[clap_band], dtype=float)
        msk = (ft >= lo) & (ft < hi)
        if msk.sum() > 10 and fl0[msk].max() > 0:
            wgap = max(1, int(0.08 * msk.sum() / max(hi - lo, 1e-9)))
            thr = float(np.percentile(fl0[msk][fl0[msk] > 0], 90)) if (fl0[msk] > 0).any() else 0.0
            cand_i = np.flatnonzero(msk & (fl0 >= max(thr, 4.0)))
            last = -10 ** 9
            keep = []
            for i in cand_i:
                a0, a1 = max(0, i - wgap), min(fl0.size, i + wgap + 1)
                if fl0[i] >= fl0[a0:a1].max() and i - last > wgap:
                    keep.append(i)
                    last = i
            clap_ev = ft[np.asarray(keep, dtype=int)] if keep else np.zeros(0)

    cands: list[dict] = []
    for k in range(n):
        p0 = first - k * spb
        b = bar_lines(p0)
        row = {"shift_beats": k, "phase_first": float(p0), "n_bars": int(b.size)}
        if clap_band in band_fluxes and b.size:
            fl = band_fluxes[clap_band]
            odd = np.concatenate([sample_flux_at(fl, ft, b + i * spb, half_window)
                                  for i in range(0, n, 2)])
            even = np.concatenate([sample_flux_at(fl, ft, b + i * spb, half_window)
                                   for i in range(1, n, 2)])
            row["backbeat_energy_on"] = float(odd.mean())
            row["backbeat_energy_off"] = float(even.mean())
            row["backbeat_energy_ratio"] = float(even.mean() / max(odd.mean(), 1e-9))
            # **主判据是「数事件」而不是「平均能量」**：能量平均会被 hat / 延音
            # 摊平（test-01 实测 1.12 对 1.56 的差别），数拍手落在哪一拍才够锐。
            if clap_ev.size:
                pos = ((clap_ev - p0) % bar) / spb
                idx = np.round(pos).astype(int) % n
                h = np.bincount(idx, minlength=n)
                on = int(h[0::2].sum()); off = int(h[1::2].sum())
                row["backbeat_hist"] = [int(x) for x in h]
                row["backbeat_n_events"] = int(h.sum())
                row["backbeat_ratio"] = float(off / max(on, 1e-9))
            else:
                row["backbeat_ratio"] = row["backbeat_energy_ratio"]
        for nm, arr in (("downbeat", downbeat_times),
                        ("chroma", chroma_change_times),
                        ("section", section_entry_times)):
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float)
            a = a[(a >= lo) & (a < hi)]
            row[f"{nm}_n"] = int(a.size)
            row[f"{nm}_median_ms"] = (_median_dist(a, b) * 1000.0
                                      if a.size and b.size else float("nan"))
        cands.append(row)

    # --- 投票 ---
    bb = [c.get("backbeat_ratio") for c in cands]
    has_bb = all(x is not None for x in bb) and max(bb) >= ratio_gate
    allowed = ([c["shift_beats"] for c in cands
                if c["backbeat_ratio"] >= ratio_gate] if has_bb
               else [c["shift_beats"] for c in cands])
    votes: dict[int, list[str]] = {c["shift_beats"]: [] for c in cands}
    for nm in ("downbeat", "chroma", "section"):
        key = f"{nm}_median_ms"
        vals = [(c["shift_beats"], c.get(key)) for c in cands
                if c.get(key) is not None and c.get(key) == c.get(key)]
        vals = [(k, v) for k, v in vals if k in allowed] or vals
        if len(vals) >= 2:
            best = min(vals, key=lambda kv: kv[1])[0]
            votes[best].append(nm)
    tally = {k: len(v) for k, v in votes.items()}
    top = max(tally, key=lambda k: tally[k]) if tally else 0
    if has_bb and top not in allowed:
        top = allowed[0]
    # backbeat 把候选收窄到 mod 2 之后，只要再有**一路** mod-4 证据指向同一个，
    # 就算定案；没有 backbeat 时要求至少两路一致。
    decided = ((has_bb and tally.get(top, 0) >= 1)
               or tally.get(top, 0) >= 2
               or (has_bb and len(allowed) == 1))
    verdict = ("keep" if top == 0 else f"shift_{top}") if decided else "undecided"
    reason_bits = []
    if has_bb:
        reason_bits.append(
            "backbeat（只定 mod 2）允许 " + "/".join(f"早{k}拍" for k in allowed)
            + "：" + "，".join(f"早{c['shift_beats']}拍 比 {c['backbeat_ratio']:.2f}"
                              for c in cands))
    else:
        reason_bits.append("backbeat 无差别（该区间可能有 4-on-the-floor kick，"
                           "请把 t_start/t_end 限制到只有军鼓/拍手的段落）")
    for nm in ("downbeat", "chroma", "section"):
        key = f"{nm}_median_ms"
        if any(c.get(key) is not None for c in cands):
            reason_bits.append(nm + "：" + "，".join(
                f"早{c['shift_beats']}拍 {c.get(key, float('nan')):.0f}ms" for c in cands))
    return _json_safe({
        "available": True,
        "candidates": cands,
        "votes": {str(k): v for k, v in votes.items()},
        "backbeat_allowed": allowed,
        "verdict": verdict,
        "shift_beats": int(top) if decided else None,
        "new_first": float(first - top * spb) if decided else None,
        "ratio_gate": float(ratio_gate),
        "reason": "；".join(reason_bits),
        "note": "backbeat 只能定 mod 2；mod 4 靠下拍跟踪器/和声/段落进入点。"
                "几路全分不开时**不要替人做主**：返回 undecided，由调用方把两种候选"
                "并列写出（各自的证据强弱 + 音乐上的合理性），交人裁定",
    })
