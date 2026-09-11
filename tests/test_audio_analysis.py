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
    BpmChange, Grid, check_offset, offset_verdict, parse_bpm_changes,
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
        self.skeleton_stem = ""
        self.accent_stems = []
        self.accent_share = {}
        self.plan_evidence = []
        self.primary_stem = ""       # 兼容字段（v0.2）
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


def test_fusion_weights_are_the_n40_calibrated_values():
    """v0.4：默认权重换成 40 首配对标定的 LOSO 值（报告 n40 §3）。"""
    w = intensity.DEFAULT_FUSION_WEIGHTS
    assert w == {"loudness": 0.00, "onset": 0.19, "drums": 0.07,
                 "voiced": 0.00, "flux": 0.74}
    assert sum(w.values()) == pytest.approx(1.0)
    assert "centroid" not in w          # 质心已移出融合项（只留在 drop 票）
    # `flux` 是标定里的最强单项，必须是最大权重；`voiced` 被 NNLS 判为 0
    assert max(w, key=w.get) == "flux"
    assert w["voiced"] == 0.0


def test_legacy_fusion_weights_kept_for_rollback():
    """旧初值保留为对照/回退（`--fusion-weights` 可取回）。"""
    w = intensity.LEGACY_FUSION_WEIGHTS
    assert w == {"loudness": 0.25, "onset": 0.30, "drums": 0.20,
                 "voiced": 0.15, "flux": 0.10}
    assert sum(w.values()) == pytest.approx(1.0)


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


# ---- v0.3：NPS 锚 / 段落地板可调 / intro-outro 结构封顶 ----


def test_nps_anchor_matches_knowledge_031():
    assert intensity.nps_for_level(13.5)[0] == pytest.approx(6.13)
    assert intensity.nps_for_level(14.0)[0] == pytest.approx(7.14)
    assert intensity.nps_for_level(None)[0] is None


def test_notes_per_bar_from_nps_scales_with_bpm():
    """高 BPM 下每小节更短 → note/小节更少（这正是 note/小节锚高估的根因）。"""
    bar_sec_150 = 4 * 60.0 / 150.0          # 1.6 s
    bar_sec_230 = 4 * 60.0 / 230.0          # 1.043 s
    npb_slow = intensity.notes_per_bar_from_nps(6.13, [bar_sec_150])[0]
    npb_fast = intensity.notes_per_bar_from_nps(6.13, [bar_sec_230])[0]
    assert npb_slow == pytest.approx(6.13 * 1.6, rel=1e-6)
    assert npb_fast < npb_slow
    # 定数 13.5 的 note/小节 对照锚是 9.15：BPM 230 时 NPS 锚明显更低（不再高估）
    assert npb_fast < intensity.notes_per_bar_for_level(13.5)[0] - 2.0


def test_section_floor_is_a_parameter_not_a_constant():
    I = np.array([0.0, 1.0])
    dn_default, _ = intensity.density_map(I, intensity.SECTION_DENSITY_FLOOR, 9.15)
    dn_calib, _ = intensity.density_map(I, intensity.CALIBRATED_SECTION_FLOOR, 9.15)
    assert dn_default[0] == pytest.approx(0.60)
    assert dn_calib[0] == pytest.approx(0.365)   # n=40 池化最优（n=8 的 0.125 已被推翻）
    assert dn_default[1] == dn_calib[1] == pytest.approx(1.0)


def test_structural_cap_limits_intro_outro_density():
    """MYTHOS 个案：前奏音频强度≈1.0，官方谱只放 4 note/小节 → 结构封顶。"""
    segs = [FakeSeg(1, 4, "intro"), FakeSeg(5, 12, "chorus"), FakeSeg(13, 16, "outro")]
    for s_, v in zip(segs, (10.0, 10.0, 10.0)):
        s_.suggested_notes_per_bar = v
    bar_suggested = np.full(16, 10.0)
    segs, capped, info = intensity.apply_structural_cap(segs, bar_suggested,
                                                        cap_ratio=0.6)
    assert info["applied"] and info["cap_notes_per_bar"] == pytest.approx(6.0)
    assert capped[:4].max() == pytest.approx(6.0)      # intro 被压
    assert capped[4:12].max() == pytest.approx(10.0)   # 副歌不动
    assert capped[12:].max() == pytest.approx(6.0)     # outro 被压
    assert segs[0].suggested_notes_per_bar == pytest.approx(6.0)
    assert "结构封顶" in "".join(segs[0].notes)


def test_structural_cap_does_not_raise_low_intro():
    """封顶只封不抬：本来就低的前奏不受影响。"""
    segs = [FakeSeg(1, 4, "intro"), FakeSeg(5, 16, "chorus")]
    segs[0].suggested_notes_per_bar, segs[1].suggested_notes_per_bar = 2.0, 10.0
    bar_suggested = np.concatenate([np.full(4, 2.0), np.full(12, 10.0)])
    segs, capped, info = intensity.apply_structural_cap(segs, bar_suggested,
                                                        cap_ratio=0.6)
    assert info["cap_notes_per_bar"] == pytest.approx(0.6 * 8.0)   # 均值 8.0
    assert capped[:4].max() == pytest.approx(2.0)                  # 低前奏不被抬
    assert segs[0].suggested_notes_per_bar == pytest.approx(2.0)


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


# ---------------- 8. 踩音规划：骨架轨 + 点缀轨（v0.3） ----------------


def _stem_feats(n=32, **over):
    """默认：四轨都可用、人声活跃（= 人声主导型曲目）。"""
    f = {}
    for s in ("drums", "bass", "other", "vocals"):
        f[f"share_{s}"] = np.full(n, 0.25)
        f[f"n_onset_{s}"] = np.full(n, 6.0)
        f[f"grid_fit_{s}"] = np.full(n, 0.9)
        f[f"grid_fit_bar_{s}"] = np.full(n, 0.9)
    f["voiced_ratio"] = np.full(n, 0.7)
    f.update(over)
    return f


def test_skeleton_defaults_to_drums():
    """v0.3 的核心改动：骨架默认是鼓（实测 drums recall 0.607、70/85 段最高）。"""
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    assert [s.skeleton_stem for s in out] == ["drum", "drum", "drum"]


def test_chorus_vocal_accent_is_conditional():
    """C1：副歌踩人声只在**人声主导**时成立，否则人声只作点缀。"""
    # (a) 人声主导：share_vocals 0.25 ≥ VOCAL_LED_SHARE（v0.4 判据）
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    led = out[1]
    assert led.skeleton_stem == "drum"
    assert led.accent_stems[0] == "vocal"
    assert "人声主导" in "".join(led.notes)

    # (b) 非人声主导：v0.4 判据只看 vocals 轨能量占比，压到阈值以下即不领衔
    f = _stem_feats()
    f["voiced_ratio"] = np.full(32, 0.20)
    f["n_onset_vocals"] = np.full(32, 2.5)
    f["share_vocals"] = np.full(32, 0.05)          # < VOCAL_LED_SHARE (0.14)
    f["share_other"] = np.full(32, 0.45)
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out2, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out2[1].skeleton_stem == "drum"
    assert out2[1].accent_stems[0] != "vocal"
    assert "不是人声主导" in "".join(out2[1].notes)


def test_chorus_rule_is_marked_suspect():
    """C1 被裁定为「条件性 + 存疑，待人工听审」——输出必须带存疑标注。"""
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    joined = "".join(out[1].notes)
    assert "存疑" in joined and "人工听审" in joined


def test_interlude_vocal_sample_is_strengthened():
    """C4：间奏人声采样是标定唯一正面支持的规则（precision 0.715）。"""
    f = _stem_feats()
    f["voiced_ratio"] = np.full(32, 0.30)
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "interlude"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[1].skeleton_stem == "drum" and out[1].accent_stems[0] == "vocal"
    assert "0.715" in "".join(out[1].notes)


def test_intro_uses_match_tendency_not_energy_share():
    """C3：intro 判据从能量占比换成 onset 匹配倾向（能量口径会选错轨）。"""
    f = _stem_feats()
    f["share_other"] = np.full(32, 0.90)       # 最响
    f["n_onset_other"] = np.full(32, 2.0)      # 但可踩音少
    f["grid_fit_bar_other"] = np.full(32, 0.55)
    f["n_onset_bass"] = np.full(32, 9.0)       # 匹配倾向最高
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[0].accent_stems[:1] == ["bass"]


def test_accent_share_includes_free_residual():
    """估计占比必须含 18.6%「什么都不落」的残差，且合计 ≈ 1。"""
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    share = out[1].accent_share
    assert share["free"] == pytest.approx(stemplan.FREE_PLAY_SHARE)
    assert sum(share.values()) == pytest.approx(1.0, abs=0.03)


def test_skeleton_degrades_when_drums_unusable():
    f = _stem_feats()
    f["n_onset_drums"] = np.full(32, 0.5)      # 鼓几乎没有 onset
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[1].skeleton_stem != "drum"
    assert "骨架降级" in "".join(out[1].notes)


def test_repeat_chorus_reuses_whole_plan_and_upgrade():
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 16, "chorus"),
            FakeSeg(17, 24, "interlude"),
            FakeSeg(25, 32, "final_chorus", repeat_of=[9, 16], upgrade=True)]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    assert out[3].skeleton_stem == out[1].skeleton_stem
    assert out[3].accent_stems == out[1].accent_stems
    assert "upgrade" in "".join(out[3].notes)


def test_outro_mirrors_intro():
    f = _stem_feats()
    f["n_onset_bass"] = np.full(32, 9.0)
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, f, _FakeGrid(32), 120.0)
    assert out[2].skeleton_stem == out[0].skeleton_stem
    assert out[2].accent_stems == out[0].accent_stems


def test_quiet_chorus_marked_suspect():
    """C5：休息段规则数据不支持也不反对（13 段，一致率 0.62）→ 标存疑。"""
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "quiet_chorus"),
            FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    assert "存疑" in "".join(out[1].notes)


def test_single_track_warning():
    """反新手护栏：全曲「骨架 ∪ 点缀」只有一条轨才报警（且曲短不报）。"""
    f = _stem_feats()
    for s in ("bass", "other", "vocals"):
        f[f"n_onset_{s}"] = np.full(32, 0.5)   # 只有鼓可用
    segs = [FakeSeg(1, 16, "chorus"), FakeSeg(17, 32, "chorus")]
    out, warn = stemplan.plan_stems(segs, f, _FakeGrid(32), 150.0)
    assert warn and "哪个最响就一直踩哪个" in warn[0]
    _, warn2 = stemplan.plan_stems(segs, f, _FakeGrid(32), 60.0)
    assert not warn2


def test_plan_evidence_is_audio_only():
    """依据必须是纯音频特征（推理期没有谱面）。"""
    segs = [FakeSeg(1, 8, "intro"), FakeSeg(9, 24, "chorus"), FakeSeg(25, 32, "outro")]
    out, _ = stemplan.plan_stems(segs, _stem_feats(), _FakeGrid(32), 120.0)
    ev = " ".join(out[1].plan_evidence)
    assert "onset 密度" in ev and "落格率" in ev and "voiced_ratio" in ev


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
    """整拍平移歧义：宽窗里会出现多个同级峰，必须取离用户值最近的那个。"""
    true_first, bpm, duration = 0.5, 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    tr = onsets.detect_onsets(synth_clicks(g.all_beat_times(), duration=duration), "drums")
    wide = check_offset(None, None, bpm=bpm, first=true_first, duration=duration,
                        onset_times=tr.times, onset_weights=tr.strengths,
                        search_beats=1.0, allow_offbeat_window=True)
    assert wide["n_tied_peaks"] >= 2
    assert abs(wide["best_first"] - true_first) < 0.015
    # v0.3 默认窗（±0.4 拍）里整拍复制峰根本进不来
    narrow = check_offset(None, None, bpm=bpm, first=true_first, duration=duration,
                          onset_times=tr.times, onset_weights=tr.strengths)
    assert narrow["n_tied_peaks"] == 1
    assert abs(narrow["best_first"] - true_first) < 0.015


# ---- v0.3 回归：±半拍「反拍」假警报（标定报告 §1.2 / R1）----


def _offbeat_hihat_clicks(bpm=128.0, first=0.0, duration=30.0, offbeat_gain=1.5):
    """合成 8 分 hi-hat：正拍在拍线上、反拍在半拍处，**反拍略强**（模拟真实曲目里
    正反拍几乎等强、反拍因检测噪声略占优的情形）。

    这正是 Signature（BPM 128，φ* 被报成 −229 ms ≈ −半拍 234.4 ms）与
    麒麟（BPM 220，−129.8 ms ≈ −半拍 136.4 ms）假警报的成因。
    """
    g = Grid(bpm=bpm, first=first, beats_per_bar=4, duration=duration)
    beats = g.all_beat_times()
    half = 30.0 / bpm                       # 半拍
    y = synth_clicks(beats, duration=duration, seed=1)
    y = y + offbeat_gain * synth_clicks(beats + half, duration=duration, seed=2)
    return g, (y / (float(np.max(np.abs(y))) or 1.0) * 0.8).astype(np.float32)


def test_offbeat_false_alarm_is_reproduced_with_legacy_window():
    """反例断言：v0.2 的 ±1 拍窗口会把反拍选成 φ*（= 被修掉的那个 bug）。"""
    bpm, duration = 128.0, 30.0
    g, y = _offbeat_hihat_clicks(bpm=bpm, duration=duration)
    tr = onsets.detect_onsets(y, "drums")
    legacy = check_offset(None, None, bpm=bpm, first=0.0, duration=duration,
                          onset_times=tr.times, onset_weights=tr.strengths,
                          search_beats=1.0, allow_offbeat_window=True)
    half_ms = 30_000.0 / bpm               # 234.4 ms
    assert abs(abs(legacy["delta_ms"]) - half_ms) < 25.0, legacy["delta_ms"]


def test_offbeat_false_alarm_fixed_by_default_window():
    """修后：默认 ±0.4 拍窗口里反拍进不来，φ* 回到真值附近且不报警。"""
    bpm, duration = 128.0, 30.0
    g, y = _offbeat_hihat_clicks(bpm=bpm, duration=duration)
    tr = onsets.detect_onsets(y, "drums")
    res = check_offset(None, None, bpm=bpm, first=0.0, duration=duration,
                       onset_times=tr.times, onset_weights=tr.strengths)
    assert abs(res["delta_ms"]) <= 25.0, res["delta_ms"]
    assert res["offbeat_ambiguous"] is False
    assert "建议人工复核" not in offset_verdict(res)


def test_search_window_is_clamped_below_half_beat():
    """搜索半径不得 ≥0.5 拍（半拍处就是反拍）——超了要被夹回并注明。"""
    bpm, duration = 128.0, 20.0
    g = Grid(bpm=bpm, first=0.0, beats_per_bar=4, duration=duration)
    tr = onsets.detect_onsets(synth_clicks(g.all_beat_times(), duration=duration), "drums")
    res = check_offset(None, None, bpm=bpm, first=0.0, duration=duration,
                       onset_times=tr.times, onset_weights=tr.strengths,
                       search_beats=1.0)
    assert res["search_beats_clamped"] is True
    assert res["search_beats"] <= 0.45


def test_offbeat_ambiguity_is_flagged_not_alarmed():
    """只有反拍有音时，φ* 贴到窗口边缘 → 判「反拍歧义 / 不可判定」，不报警。"""
    bpm, duration = 128.0, 30.0
    g = Grid(bpm=bpm, first=0.0, beats_per_bar=4, duration=duration)
    half = 30.0 / bpm
    y = synth_clicks(g.all_beat_times() + half, duration=duration)
    tr = onsets.detect_onsets(y, "drums")
    res = check_offset(None, None, bpm=bpm, first=0.0, duration=duration,
                       onset_times=tr.times, onset_weights=tr.strengths)
    assert res["offbeat_ambiguous"] is True
    v = offset_verdict(res)
    assert "反拍歧义" in v and "建议人工复核" not in v


def test_verdict_bands_decode_delay_and_ratio_gate():
    """判词四档：一致 / 疑似解码延迟 / 显著性不足 → 不报警 / 才报警。"""
    base = {"available": True, "confidence_z": 3.0, "method": "onset-fit",
            "n_tied_peaks": 1, "offbeat_ambiguous": False, "half_beat_ms": 234.4}
    assert "一致" in offset_verdict({**base, "delta_ms": 8.0, "ratio": 1.2})
    assert "解码" in offset_verdict({**base, "delta_ms": 27.9, "ratio": 15.4})
    assert "不报警" in offset_verdict({**base, "delta_ms": 80.0, "ratio": 1.02})
    assert "建议人工复核" in offset_verdict({**base, "delta_ms": 80.0, "ratio": 1.9})


def test_symmetric_ties_downgrade_confidence_but_not_the_verdict():
    """打分曲线平坦、同级峰对称（如 Xevel）：φ* 仍在容差内就还是「一致」，
    只是注明参考价值打折；**不得**因此误报成「反拍歧义」。"""
    base = {"available": True, "confidence_z": 1.7, "method": "onset-fit",
            "n_tied_peaks": 6, "offbeat_ambiguous": False, "tied_symmetric": True,
            "half_beat_ms": 170.5}
    v_ok = offset_verdict({**base, "delta_ms": 7.6, "ratio": 1.03})
    assert "一致" in v_ok and "反拍歧义" not in v_ok and "打折" in v_ok
    v_big = offset_verdict({**base, "delta_ms": 90.0, "ratio": 3.0})
    assert "不报警" in v_big


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
