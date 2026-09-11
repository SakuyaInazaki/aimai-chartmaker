"""`tools/audio_analysis` 的单元测试（v0.2）。

全部使用 **numpy 合成音频 / 合成事件**（已知 BPM / first / 分音），
不引用任何真实音频文件——仓库内不得出现有版权素材（AGENT.md 准则 5）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audio_analysis import features, intensity, onsets, quantize, stemplan, tracks  # noqa: E402
from tools.audio_analysis.grid import (  # noqa: E402
    BpmChange, Grid, check_offset, parse_bpm_changes,
)
from tools.audio_analysis import structure as struct_mod  # noqa: E402

SR = onsets.ANALYSIS_SR


# ---------------- 合成工具 ----------------


def synth_clicks(times: np.ndarray, duration: float, sr: int = SR,
                 decay: float = 0.035, seed: int = 0) -> np.ndarray:
    """在给定秒数处放置短促噪声爆音（模拟鼓点）。"""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(round(duration * sr)), dtype=np.float32)
    n = int(round(decay * sr))
    env = np.exp(-np.linspace(0, 6, n))
    for t in np.atleast_1d(times):
        i = int(round(float(t) * sr))
        if i < 0 or i >= len(y):
            continue
        m = min(n, len(y) - i)
        y[i:i + m] += rng.standard_normal(m).astype(np.float32) * env[:m]
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.8).astype(np.float32)


def grid_times(grid: Grid, division: int, slots_per_bar, bars) -> np.ndarray:
    out = []
    for bar in bars:
        for slot in slots_per_bar:
            out.append(grid.subdivision_time(bar, division, slot))
    return np.asarray(sorted(out), dtype=float)


def rel_at(division: int, slots, bar_dur: float) -> np.ndarray:
    """小节内理想相对秒数。"""
    return np.asarray([s / division * bar_dur for s in slots], dtype=float)


class FakeSeg:
    """段落桩（只用到 stemplan / structure 需要的字段）。"""

    def __init__(self, start_bar, end_bar, function="", label="A", **kw):
        self.start_bar, self.end_bar = start_bar, end_bar
        self.function, self.label = function, label
        self.is_repeat = kw.get("is_repeat", False)
        self.repeat_of = kw.get("repeat_of")
        self.rest = kw.get("rest", False)
        self.upgrade = kw.get("upgrade", False)
        self.intensity = kw.get("intensity", 0.5)
        self.voiced_ratio = kw.get("voiced_ratio", 0.0)
        self.chorus_index = kw.get("chorus_index")
        self.primary_stem = ""
        self.secondary_stem = ""
        self.notes = []
        self.evidence = []

    @property
    def n_bars(self):
        return self.end_bar - self.start_bar + 1


# ---------------- 1. 网格时间计算 ----------------


def test_grid_basic_times():
    g = Grid(bpm=120.0, first=1.0, beats_per_bar=4, duration=20.0)
    assert g.bar_duration(1) == pytest.approx(2.0)
    assert g.bar_start(1) == pytest.approx(1.0)
    assert g.bar_start(5) == pytest.approx(9.0)
    assert g.beat_time(2, 2) == pytest.approx(4.0)
    assert g.n_bars == 10


def test_grid_negative_first():
    g = Grid(bpm=192.0, first=-0.25, beats_per_bar=4, duration=10.0)
    assert g.bar_start(1) == pytest.approx(-0.25)
    assert g.bar_start(2) == pytest.approx(-0.25 + 1.25)


def test_grid_subdivision_times():
    g = Grid(bpm=150.0, first=0.5, beats_per_bar=4, duration=30.0)
    bar_dur = 1.6
    t = g.subdivision_times(3, 16)
    assert len(t) == 16 and t[0] == pytest.approx(g.bar_start(3))
    assert t[1] - t[0] == pytest.approx(bar_dur / 16)
    assert g.subdivision_times(3, 12)[1] - g.subdivision_times(3, 12)[0] \
        == pytest.approx(bar_dur / 12)


def test_grid_bpm_changes():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4,
             bpm_changes=(BpmChange(bar=3, bpm=240.0),), duration=20.0)
    assert g.bar_duration(3) == pytest.approx(1.0)
    assert g.bar_start(4) == pytest.approx(5.0)
    assert g.has_bpm_changes


def test_parse_bpm_changes():
    assert [(c.bar, c.bpm) for c in parse_bpm_changes("33:180, 65:155")] \
        == [(33, 180.0), (65, 155.0)]
    assert parse_bpm_changes(None) == ()
    with pytest.raises(ValueError):
        parse_bpm_changes("33-180")


def test_grid_time_to_bar_pos():
    g = Grid(bpm=120.0, first=1.0, beats_per_bar=4, duration=20.0)
    bar, pos = g.time_to_bar_pos(4.0)
    assert bar == 2 and pos == pytest.approx(0.5)
    assert g.bar_of(0.5) == 0


# ---------------- 2. 容差 τ(d) 与拍同步重采样 ----------------


def test_tau_is_monotone_and_clamped():
    """τ(d) = clamp(0.25·slot_ms, 12, 30)：随分音变细单调收紧，两端被夹住。"""
    bar_ms = 1250.0                      # 192 BPM 4/4
    taus = [quantize.tau_ms(d, bar_ms) for d in (4, 8, 12, 16, 24, 32)]
    assert taus == sorted(taus, reverse=True)
    assert taus[0] == pytest.approx(30.0)          # 0.25×312ms → 上限 30
    assert taus[-1] == pytest.approx(12.0)         # 0.25×39ms  → 下限 12
    assert taus[3] == pytest.approx(0.25 * bar_ms / 16)   # {16} 未被夹


def test_tau_fixes_v01_random_division_problem():
    """v0.1 的容差 max(25ms, 1/64 小节) 在 192BPM 下比 24/32 格距还大。"""
    bar_ms = 1250.0
    old_tol = max(25.0, bar_ms / 64)                # v0.1 实际取值 = 25ms
    # (1) 容差超过「32 分格距的一半」→ 任意 onset 都能被 {32} "解释"
    assert old_tol > quantize.slot_ms(32, bar_ms) / 2      # 19.5ms < 25ms
    # (2) 24 分格线与 32 分格线的最小非零间距 = 2/192 小节 = 13ms < 25ms
    #     → 25ms 容差下 24 与 32 在信息上不可分，选哪个近乎随机
    min_sep = 2.0 / quantize.bar_units(4) * bar_ms
    assert min_sep == pytest.approx(13.02, abs=0.1) and min_sep < old_tol
    assert quantize.slot_ms(24, bar_ms) == pytest.approx(52.08, abs=0.1)
    # v0.2：τ(24)=13ms、τ(32)=12ms，都 < 各自格距的一半 → 不再随机可换
    for d in (16, 24, 32):
        assert quantize.tau_ms(d, bar_ms) < quantize.slot_ms(d, bar_ms) / 2


def test_beat_unit_resampling():
    """拍同步重采样：1/48 拍 → 4/4 每小节 192 个整数单位。"""
    assert quantize.bar_units(4) == 192
    bar_dur = 1.25
    rel = rel_at(16, [0, 4, 7, 12], bar_dur)
    u = quantize.resample_to_beat_units(rel, bar_dur, 4)
    assert list(u) == [0, 48, 84, 144]              # 192/16 = 12 单位一格
    # 三连也整除
    u12 = quantize.resample_to_beat_units(rel_at(12, [0, 1, 2], bar_dur), bar_dur, 4)
    assert list(u12) == [0, 16, 32]


# ---------------- 3. 逐小节选 div（单 div 渲染 / 三连 / {32} 红线） ----------------


def test_choose_division_picks_coarsest_that_explains():
    bar_dur = 1.25
    d = quantize.choose_bar_division(rel_at(4, [0, 1, 2, 3], bar_dur), bar_dur)
    assert d.division == 4 and d.resolved
    # `x...x...x.x.....` 其实落在 8 分网格上 → 选 8，不选 16
    d = quantize.choose_bar_division(rel_at(16, [0, 4, 8, 10], bar_dur), bar_dur)
    assert d.division == 8 and d.resolved


def test_choose_division_sixteenth():
    bar_dur = 1.25
    d = quantize.choose_bar_division(rel_at(16, [0, 4, 7, 12], bar_dur), bar_dur)
    assert d.division == 16 and d.resolved and not d.triplet


def test_triplet_gate_accepts_real_triplets():
    """真三连：rms12 ≈ 0 且 rms12 ≤ 0.7·rms16 → 判 {12}。"""
    bar_dur = 1.6
    d = quantize.choose_bar_division(rel_at(12, [0, 1, 2, 3, 6, 9], bar_dur), bar_dur)
    assert d.division == 12 and d.triplet and d.resolved


def test_triplet_gate_rejects_binary_bar():
    """纯二分小节不得被判成三连（12 在扫描顺序里排在 16 之前，靠门把关）。"""
    bar_dur = 1.6
    d = quantize.choose_bar_division(rel_at(16, [0, 4, 7, 12], bar_dur), bar_dur)
    assert not d.triplet and d.division == 16


def test_fine_division_redline_blocks_32_without_drums():
    """{32} 红线：非鼓来源（drum onset 数不足）一律拦下，压到 16 并计 unquantized。"""
    bar_dur = 1.25
    rel = rel_at(32, [0, 1, 3, 5, 7, 9, 11, 13], bar_dur)
    d = quantize.choose_bar_division(rel, bar_dur, n_drum_onsets=0)
    assert d.division <= 16
    assert d.fine_blocked and not d.resolved
    assert d.n_unquantized > 0


def test_fine_division_redline_allows_dense_drums():
    """三条件全满足（drums 来源 + onset≥6 + rms≤15ms）才放行 {32}。"""
    bar_dur = 1.25
    rel = rel_at(32, [0, 1, 3, 5, 7, 9, 11, 13], bar_dur)
    d = quantize.choose_bar_division(rel, bar_dur, n_drum_onsets=len(rel))
    assert d.division == 32 and d.resolved and not d.fine_blocked


def test_fine_division_redline_needs_six_onsets():
    bar_dur = 1.25
    rel = rel_at(32, [0, 1, 3], bar_dur)          # 只有 3 个 onset
    d = quantize.choose_bar_division(rel, bar_dur, n_drum_onsets=3)
    assert d.division <= 16 and not d.resolved


def test_no_fine_div_switch():
    bar_dur = 1.25
    rel = rel_at(32, [0, 1, 3, 5, 7, 9, 11, 13], bar_dur)
    d = quantize.choose_bar_division(rel, bar_dur, n_drum_onsets=8, allow_fine=False)
    assert d.division <= 16


def test_single_division_per_bar_across_tracks():
    """**同一小节所有轨共用一个 div**（v0.1 的 kick 4 格 / other 24 格问题）。"""
    g = Grid(bpm=192.0, first=0.0, beats_per_bar=4, duration=10.0)
    # drums 只在四分位置，other 在 16 分位置 → 该小节 div 必须统一
    drums = grid_times(g, 4, [0, 1, 2, 3], bars=[2])
    other = grid_times(g, 16, [0, 4, 7, 12], bars=[2])
    bar_div, slots, stats = quantize.quantize_song(
        {"drums": drums, "other": other, "bass": np.zeros(0), "vocals": np.zeros(0)}, g)
    assert bar_div[2].division == 16
    patterns = tracks.build_track_patterns(
        g, bar_div, {"drums": drums, "other": other, "bass": np.zeros(0),
                     "vocals": np.zeros(0)}, slots, {}, None)
    lens = {t: len(patterns[t][2].pattern) for t in tracks.TRACK_ORDER}
    assert set(lens.values()) == {16}, lens


def test_quantize_song_stats_and_unquantized_ratio():
    g = Grid(bpm=192.0, first=1.875, beats_per_bar=4, duration=20.0)
    times = grid_times(g, 16, [0, 4, 7, 12], bars=[2, 3, 4, 5])
    bar_div, slots, stats = quantize.quantize_song({"drums": times}, g)
    assert all(bar_div[b].division == 16 for b in (2, 3, 4, 5))
    assert stats["division_histogram"] == {"16": 4}
    assert stats["unquantized_onsets"] == 0
    assert stats["unquantized_ratio"] == 0.0
    assert stats["per_track"]["drums"]["mean_abs_err_ms"] < 5.0   # 仅 1/48 拍重采样误差


def test_quantize_reports_unquantized_when_nothing_fits():
    """刻意放一个落在任何格线之间的 onset → 计入 unquantized。

    取 1/48 拍格点第 7 单位：7 不是 6（{32}）也不是 8（{24}）的倍数，
    对 {4}/{8}/{12}/{16} 的残差都超过各自 τ；{24} 被三连门挡住、
    {32} 被红线挡住 → 唯一去处是压到 {16} 并计入 unquantized。
    """
    g = Grid(bpm=192.0, first=0.0, beats_per_bar=4, duration=10.0)
    bd = g.bar_duration(2)
    times = np.array([g.bar_start(2), g.bar_start(2) + 7 / 192 * bd])
    bar_div, _, stats = quantize.quantize_song({"drums": times}, g)
    assert not bar_div[2].resolved
    assert stats["unquantized_onsets"] >= 1
    assert stats["unquantized_ratio"] > 0


def test_end_of_bar_snaps_forward():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=12.0)
    t = g.bar_start(3) - 0.010
    bar_div, slots, _ = quantize.quantize_song({"drums": np.array([t])}, g)
    assert slots["drums"].get(3) == [0] and 2 not in slots["drums"]


def test_onset_detection_then_quantization_recovers_division():
    """端到端：合成 16 分鼓点 → librosa 检测 → 量化，应还原 16 分。"""
    g = Grid(bpm=150.0, first=0.5, beats_per_bar=4, duration=14.0)
    bars = list(range(2, 8))
    times = grid_times(g, 16, [0, 4, 7, 12], bars=bars)
    y = synth_clicks(times, duration=14.0)
    tr = onsets.detect_onsets(y, "drums")
    assert tr.count >= len(times) * 0.9
    bar_div, slots, stats = quantize.quantize_song({"drums": tr.times}, g)
    ok = [b for b in bars if bar_div[b].division == 16]
    assert len(ok) >= len(bars) - 1, {b: bar_div[b].division for b in bars}


# ---------------- 4. 网格串渲染（字符集 X x - .） ----------------


def test_render_pattern_charset_and_precedence():
    p = quantize.render_pattern(16, np.array([0, 4, 8, 10]),
                                ["X", "x", "x", "x"])
    assert p == "X...x...x.x....."
    assert set(p) <= set("Xx-.")
    # 同格冲突取强者
    assert quantize.render_pattern(4, np.array([0, 0]), ["x", "X"]) == "X..."


def test_render_pattern_sustain_dash():
    sustain = np.array([False, True, True, False])
    p = quantize.render_pattern(4, np.array([0]), ["x"], sustain)
    assert p == "x--."
    assert set(p) <= set("Xx-.")


def test_drum_chars_kick_is_capital_X():
    chars = tracks.build_drum_chars(3, np.array([True, False, False]))
    assert chars == ["X", "x", "x"]


def test_vocal_chars_need_pitch_and_strength():
    chars = tracks.build_vocal_chars(
        3, np.array([True, True, False]), np.array([True, False, True]))
    assert chars == ["X", "x", "x"]


def test_four_tracks_only():
    assert tracks.TRACK_ORDER == ("drum", "vocal", "bass", "hook")
    assert len(tracks.TRACK_ORDER) <= 4          # v2 §5.2(d) 轨数上限


# ---------------- 5. 强度：归一只做一次（C 项 bug 回归） ----------------


def test_smoothing_does_not_shift_level():
    """C 项 bug 回归：raw 与 smoothed 必须同基准，平滑不得系统性抬/压电平。

    v0.1 的 bar_raw 未归一、bar_intensity 平滑后又做了一次 min-max，
    画图时二者基准不同，低强度段偏离恰好是 min/max（实测 0.2–0.3）。
    """
    from scipy.ndimage import gaussian_filter1d, median_filter

    rng = np.random.default_rng(3)
    curve = np.concatenate([np.full(20, 0.85), np.full(15, 0.30), np.full(25, 0.9)])
    curve = np.clip(curve + rng.normal(0, 0.02, curve.size), 0, 1)
    raw = intensity.robust_unit(curve)
    sm = np.clip(gaussian_filter1d(median_filter(raw, size=3, mode="nearest"),
                                   sigma=1.0, mode="nearest"), 0, 1)
    # 弱段（第 20–35 小节）上 raw 与 smoothed 的差必须接近 0
    weak = slice(22, 33)
    assert abs(float(np.mean(sm[weak] - raw[weak]))) < 0.03
    assert abs(float(np.mean(sm - raw))) < 0.02
    # 反例：v0.1 的两次不同归一会造成系统性偏离
    v01_raw_plot = curve / curve.max()
    v01_sm = (sm - sm.min()) / (sm.max() - sm.min())
    assert float(np.mean(v01_raw_plot[weak] - v01_sm[weak])) > 0.15


def test_zscore_and_robust_unit():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = intensity.zscore(x)
    assert float(np.mean(z)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(z)) == pytest.approx(1.0)
    assert intensity.zscore(np.full(5, 2.0)).tolist() == [0.0] * 5
    u = intensity.robust_unit(np.arange(100, dtype=float))
    assert u.min() == 0.0 and u.max() == 1.0


def test_fusion_weights_match_research_v2():
    w = intensity.DEFAULT_FUSION_WEIGHTS
    assert w == {"loudness": 0.25, "onset": 0.30, "drums": 0.20,
                 "voiced": 0.15, "flux": 0.10}
    assert sum(w.values()) == pytest.approx(1.0)
    assert "centroid" not in w          # 质心已移出融合项（只留在 drop 票）


def test_vote_weights_are_five():
    w = intensity.DEFAULT_VOTE_WEIGHTS
    assert set(w) == {"structure", "novelty", "energy", "centroid", "vocal"}
    assert w["structure"] == 0.45 and w["vocal"] == 0.05
    assert sum(w.values()) == pytest.approx(1.0)


# ---------------- 6. 密度映射 ----------------


def test_density_map_floor():
    """`density_norm = floor + (1−floor)·I`：强度 0 也不得掉到 0（可玩性地板）。"""
    I = np.array([0.0, 0.5, 1.0])
    dn, notes = intensity.density_map(I, intensity.SECTION_DENSITY_FLOOR, 9.15)
    assert dn.tolist() == pytest.approx([0.60, 0.80, 1.00])
    assert notes.tolist() == pytest.approx([0.60 * 9.15, 0.80 * 9.15, 9.15])
    dn_bar, _ = intensity.density_map(I, intensity.BAR_DENSITY_FLOOR, 9.15)
    assert dn_bar[0] == pytest.approx(0.25)
    assert (dn_bar <= dn).all()          # 小节尺度地板更低


def test_notes_per_bar_for_level():
    assert intensity.notes_per_bar_for_level(13.0)[0] == pytest.approx(8.03)
    assert intensity.notes_per_bar_for_level(13.5)[0] == pytest.approx(9.15)
    assert intensity.notes_per_bar_for_level(14.0)[0] == pytest.approx(10.54)
    assert intensity.notes_per_bar_for_level(None)[0] == pytest.approx(9.0)
    # 非表内定数吸附到最近档
    assert intensity.notes_per_bar_for_level(13.4)[0] == pytest.approx(9.15)


def test_total_notes_range_from_knowledge_004():
    assert intensity.total_notes_range(None) is None
    mean, p10, p90 = intensity.total_notes_range(13.5)
    assert p10 < mean < p90 and (p10, p90) == (492, 967)


# ---------------- 7. 结构：边界投票与标签派生 ----------------


def test_vote_boundaries_requires_two_sources():
    """≥2 路同意才采纳；单票候选只在段数不足时补齐。"""
    sources = {"allin1": [0, 16, 32, 48], "novelty": [16, 32, 48, 70],
               "repeat": [17, 33, 90]}
    bounds, detail = vote = struct_mod.vote_boundaries(sources, 100)
    for b in (0, 16, 32, 48):
        assert b in bounds
    assert detail["multi_vote"] >= 3
    # 90 只有一路提名，但为凑够 min_segments 可能被补进来 —— 必须记在 topped_up
    assert detail["topped_up"] == max(0, len(bounds) - 1 - detail["multi_vote"])


def test_vote_boundaries_snaps_to_phrase_line():
    """边界吸附到最近 4 小节乐句线（容差 1）。"""
    sources = {"a": [15], "b": [17]}
    bounds, _ = struct_mod.vote_boundaries(sources, 64, min_segments=1)
    assert 16 in bounds


def test_label_ja_double_writing():
    assert struct_mod.label_ja("chorus") == "サビ(chorus)"
    assert struct_mod.label_ja("pre_chorus") == "Bメロ(pre_chorus)"
    assert struct_mod.label_ja("final_chorus") == "ラスサビ(final_chorus)"
    assert struct_mod.label_ja("quiet_chorus") == "落ちサビ(quiet_chorus)"
    assert struct_mod.label_ja("drop") == "ドロップ(drop)"


def _synthetic_song():
    """合成一首 intro→verse→build→chorus→休息→chorus#2→outro 的结构（72 小节）。"""
    n = 72
    I = np.concatenate([
        np.full(8, 0.15),                    # 1–8   intro
        np.full(8, 0.35),                    # 9–16  verse
        np.linspace(0.35, 0.80, 8),          # 17–24 build（单调上升）
        np.full(16, 0.95),                   # 25–40 chorus
        np.full(8, 0.20),                    # 41–48 休息
        np.full(16, 1.00),                   # 49–64 chorus#2（最强）
        np.full(8, 0.12),                    # 65–72 outro
    ])
    voiced = np.concatenate([np.full(8, 0.05), np.full(8, 0.40), np.full(8, 0.25),
                             np.full(16, 0.85), np.full(8, 0.10), np.full(16, 0.90),
                             np.full(8, 0.05)])
    kick = np.concatenate([np.full(8, 4.0), np.full(8, 4.0),
                           np.concatenate([np.full(4, 4.0), np.full(4, 0.5)]),  # kick↓
                           np.full(16, 4.0), np.full(8, 1.0), np.full(16, 4.0),
                           np.full(8, 2.0)])
    feats = {"voiced_ratio": voiced, "n_onset_kick": kick,
             "n_onset_hihat": np.full(n, 8.0), "hf_ratio": np.full(n, 0.1)}
    return I, feats


def _synthetic_segments():
    return [struct_mod.Segment(1, 8, label="A"), struct_mod.Segment(9, 24, label="B"),
            struct_mod.Segment(25, 40, label="C"), struct_mod.Segment(41, 48, label="D"),
            struct_mod.Segment(49, 64, label="C"), struct_mod.Segment(65, 72, label="A")]


class _FakeGrid:
    def __init__(self, n):
        self.n_bars = n
        self.beats_per_bar = 4


def test_pre_chorus_derivation():
    """Bメロ 派生：chorus 前 4–8 小节 ∧ 强度单调上升 ∧ kick 下降。"""
    I, feats = _synthetic_song()
    segs = _synthetic_segments()
    out, notes = struct_mod.assign_functions(segs, I, feats, _FakeGrid(72))
    fns = [(s.start_bar, s.end_bar, s.function) for s in out]
    pre = [s for s in out if s.function == "pre_chorus"]
    assert pre, fns
    assert pre[0].n_bars <= 8, fns              # 绝不能像 v0.1 那样吃掉 24 小节
    assert pre[0].end_bar == 24
    # 前一段被切短而不是整段变成 Bメロ
    verse = [s for s in out if s.function == "verse"]
    assert verse and verse[0].start_bar == 9


def test_chorus_needs_two_evidence_and_final_chorus():
    I, feats = _synthetic_song()
    segs = _synthetic_segments()
    out, _ = struct_mod.assign_functions(segs, I, feats, _FakeGrid(72))
    chorus_like = [s for s in out if s.function in struct_mod.CHORUS_FAMILY]
    assert len(chorus_like) >= 2, [(s.start_bar, s.function) for s in out]
    for s in chorus_like:
        assert len(s.evidence) >= 2          # ≥2 路证据
    last = chorus_like[-1]
    assert last.function == "final_chorus"   # 最后一个且强度最高
    assert last.upgrade and last.chorus_index >= 2
    assert last.repeat_of == [25, 40]


def test_instrumental_uses_drop():
    I, feats = _synthetic_song()
    feats = dict(feats, voiced_ratio=np.full(72, 0.02))
    segs = _synthetic_segments()
    out, notes = struct_mod.assign_functions(segs, I, feats, _FakeGrid(72),
                                             instrumental=True)
    assert any(s.function == "drop" for s in out)
    assert not any(s.function == "chorus" for s in out)


def test_rest_flag_between_high_segments():
    I, feats = _synthetic_song()
    segs = _synthetic_segments()
    out, _ = struct_mod.assign_functions(segs, I, feats, _FakeGrid(72))
    rest = [s for s in out if s.rest]
    assert any(s.start_bar == 41 for s in rest), [(s.start_bar, s.rest) for s in out]


def test_tier_and_suggest_division():
    assert struct_mod.tier_of(0.9) == "peak" and struct_mod.tier_of(0.1) == "low"
    assert struct_mod.suggest_division("peak") == "{16}"
    assert struct_mod.suggest_division("low") == "{8}"


# ---------------- 8. 切轨规则表 ----------------


def _stem_feats(n=32, **over):
    f = {}
    for s in ("drums", "bass", "other", "vocals"):
        f[f"share_{s}"] = np.full(n, 0.25)
        f[f"n_onset_{s}"] = np.full(n, 6.0)
        f[f"grid_fit_{s}"] = np.full(n, 0.9)
    f["voiced_ratio"] = np.full(n, 0.7)
    f.update(over)
    return f


def test_chorus_primary_is_vocal():
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, warn = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    assert out[1].primary_stem == "vocal"


def test_verse_uses_other_with_vocal_tail():
    f = _stem_feats()
    f["share_other"] = np.full(32, 0.5)
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "verse"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[1].primary_stem == "hook" and out[1].secondary_stem == "vocal"


def test_outro_mirrors_intro():
    f = _stem_feats()
    f["share_bass"] = np.full(32, 0.7)
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[2].primary_stem == out[0].primary_stem


def test_repeat_chorus_uses_same_stem_and_upgrade():
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 16, "chorus"),
            FakeSeg(17, 24, "interlude"),
            FakeSeg(25, 32, "final_chorus", repeat_of=[9, 16], upgrade=True)]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    assert out[3].primary_stem == out[1].primary_stem
    assert "upgrade" in "".join(out[3].notes)


def test_degrades_to_drums_when_track_unusable():
    f = _stem_feats()
    f["n_onset_vocals"] = np.full(32, 0.5)        # 人声几乎没有 onset
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[1].primary_stem == "drum"
    assert "降级" in "".join(out[1].notes)


def test_single_track_warning():
    f = _stem_feats()
    f["share_vocals"] = np.full(32, 0.9)
    segs = [FakeSeg(1, 16, "chorus"), FakeSeg(17, 32, "chorus")]
    out, warn = stemplan.plan_stems(segs, f, _FakeGrid(32), 150.0)
    assert warn and "哪个最响就一直踩哪个" in warn[0]
    # 曲短则不报警
    _, warn2 = stemplan.plan_stems(segs, f, _FakeGrid(32), 60.0)
    assert not warn2


def test_rest_segment_switches_to_second_loudest():
    f = _stem_feats()
    f["share_vocals"] = np.full(32, 0.5)
    f["share_other"] = np.full(32, 0.3)
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "interlude", rest=True),
            FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert "休息段" in "".join(out[1].notes)


# ---------------- 9. 逐小节特征 ----------------


def test_grid_fit_per_bar():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=12.0)
    on_grid = grid_times(g, 8, [0, 2, 4, 6], bars=[1])
    assert features.grid_fit_per_bar(on_grid, g, 8)[0] == pytest.approx(1.0)
    off = on_grid + 0.09                       # 偏 90ms，远超 τ(8)=30ms
    assert features.grid_fit_per_bar(off, g, 8)[0] < 0.5


def test_merged_onset_count_dedupes():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=8.0)
    t = np.array([0.0, 0.5, 1.0])
    counts = features.merged_onset_count(
        {"drums": t, "bass": t + 0.005, "other": np.zeros(0), "vocals": np.zeros(0)}, g)
    assert counts[0] == 3                      # 5ms 内视为同一个可踩点


def test_share_per_bar_sums_to_one():
    e = {"drums": np.array([1.0, 2.0]), "bass": np.array([1.0, 0.0]),
         "other": np.array([2.0, 2.0]), "vocals": np.array([0.0, 0.0])}
    sh = features.share_per_bar(e)
    tot = sum(sh[k] for k in sh)
    assert tot == pytest.approx([1.0, 1.0])


def test_silence_run_detects_gap():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=8.0)
    sr = SR
    y = np.zeros(int(8.0 * sr), dtype=np.float32)
    y[:int(2.0 * sr)] = 0.5 * np.sin(2 * np.pi * 200 * np.arange(int(2.0 * sr)) / sr)
    runs = features.silence_runs(y, sr, g)
    assert runs[0] < 0.5 and runs[1] > 1.0     # 第 2 小节（2–4s）全静音


# ---------------- 10. offset 校验 ----------------


def _offset_env(times, duration):
    y = synth_clicks(times, duration=duration)
    env = onsets.onset_envelope(y)
    return env, onsets.frame_times(len(env))


@pytest.mark.parametrize("true_first,user_error", [(0.5, 0.0), (0.5, 0.040), (0.5, -0.055)])
def test_check_offset_recovers_known_offset(true_first, user_error):
    bpm, duration = 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    env, ft = _offset_env(g.all_beat_times(), duration)
    res = check_offset(env, ft, bpm=bpm, first=true_first + user_error, duration=duration)
    assert res["available"]
    assert abs(res["best_first"] - true_first) < 0.020
    assert res["delta_sec"] == pytest.approx(-user_error, abs=0.020)


def test_check_offset_reports_only_does_not_override():
    bpm, duration = 150.0, 20.0
    g = Grid(bpm=bpm, first=0.5, beats_per_bar=4, duration=duration)
    env, ft = _offset_env(g.all_beat_times(), duration)
    res = check_offset(env, ft, bpm=bpm, first=0.56, duration=duration)
    assert res["user_first"] == 0.56
    assert Grid(bpm=bpm, first=0.56, beats_per_bar=4,
                duration=duration).bar_start(1) == pytest.approx(0.56)


def test_check_offset_empty_envelope():
    assert check_offset(np.zeros(0), np.zeros(0), bpm=150.0, first=0.5)["available"] is False


@pytest.mark.parametrize("user_error", [0.0, 0.035, -0.045])
def test_check_offset_onset_fit_path(user_error):
    true_first, bpm, duration = 0.5, 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    y = synth_clicks(g.all_beat_times(), duration=duration)
    env = onsets.onset_envelope(y)
    tr = onsets.detect_onsets(y, "drums")
    res = check_offset(env, onsets.frame_times(len(env)), bpm=bpm,
                       first=true_first + user_error, duration=duration,
                       onset_times=tr.times, onset_weights=tr.strengths)
    assert res["method"] == "onset-fit"
    assert abs(res["best_first"] - true_first) < 0.015
    assert "envelope_best_first" in res


def test_check_offset_prefers_peak_nearest_user_value():
    true_first, bpm, duration = 0.5, 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    tr = onsets.detect_onsets(synth_clicks(g.all_beat_times(), duration=duration), "drums")
    res = check_offset(None, None, bpm=bpm, first=true_first, duration=duration,
                       onset_times=tr.times, onset_weights=tr.strengths)
    assert res["n_tied_peaks"] >= 2
    assert abs(res["best_first"] - true_first) < 0.015


# ---------------- 11. 逐小节聚合 / VAD ----------------


def test_onsets_per_bar_counts():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=12.0)
    tr = onsets.OnsetTrack("drums", grid_times(g, 4, [0, 1, 2, 3], bars=[1, 2]),
                           np.ones(8))
    counts = onsets.onsets_per_bar(tr, g)
    assert counts[0] == 4 and counts[1] == 4 and counts[2] == 0


def test_vocal_activity_on_synthetic_tone():
    sr, dur = SR, 8.0
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t, dtype=np.float32)
    half = len(t) // 2
    y[:half] = (0.5 * np.sin(2 * np.pi * 220 * t[:half])).astype(np.float32)
    vad = onsets.vocal_activity(y)
    act, times = vad["active"], vad["times"]
    assert act[times < 3.5].mean() > 0.9
    assert act[times > 4.5].mean() < 0.05


def test_pitched_mask_separates_tone_from_noise():
    """有音高的正弦 → True；宽带噪声爆音 → False。"""
    sr, dur = SR, 2.0
    t = np.arange(int(sr * dur)) / sr
    tone = (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    noise = synth_clicks(np.array([0.0]), duration=dur, decay=0.5)
    assert bool(tracks.pitched_mask(tone, np.array([0.2]), sr)[0])
    assert not bool(tracks.pitched_mask(noise, np.array([0.0]), sr)[0])
