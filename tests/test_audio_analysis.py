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
from tools.audio_analysis import pitch_notes as pn_mod  # noqa: E402
from tools.audio_analysis import stems as stems_mod  # noqa: E402
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
        self.sparse_accents = []
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
    # n=160 池化最优（n=8 的 0.125、n=40 的 0.365 依次被更大样本取代）
    assert dn_calib[0] == pytest.approx(0.330)
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


# ---------------- 12. v0.5 分轨细化：六路 / 有音高 note / fx 瞬态 ----------------


def test_stems_model_registry_and_dir_naming():
    """六路模型多两轨；`htdemucs` 的目录名必须保持历史路径（已有缓存不能失效）。"""
    assert stems_mod.stems_for("htdemucs") == ("drums", "bass", "other", "vocals")
    assert stems_mod.stems_for("htdemucs_6s") == (
        "drums", "bass", "other", "vocals", "guitar", "piano")
    assert stems_mod.stems_for("不认识的模型") == stems_mod.STEM_NAMES
    root = Path("/x/y")
    assert stems_mod.stems_dir_for(root, "htdemucs") == root / "stems"
    assert stems_mod.stems_dir_for(root, "htdemucs_6s") == root / "stems_htdemucs_6s"


def test_compact_stem_write_is_mono_22k_pcm16(tmp_path):
    import soundfile as sf

    arr = np.column_stack([np.sin(np.linspace(0, 40, 44100)),
                           np.sin(np.linspace(0, 40, 44100))]).astype(np.float32)
    p = tmp_path / "s.wav"
    stems_mod._write_stem(p, arr, 44100, compact=True)
    info = sf.info(str(p))
    assert info.samplerate == stems_mod.COMPACT_SR
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def _hf_burst(times, duration, sr=SR, freq=8000.0, dur=0.02):
    """高频短爆（风铃 / crash 的代理）。"""
    y = np.zeros(int(round(duration * sr)), dtype=np.float32)
    n = int(round(dur * sr))
    env = np.exp(-np.linspace(0, 5, n))
    t = np.arange(n) / sr
    tone = (np.sin(2 * np.pi * freq * t) * env).astype(np.float32)
    for tt in np.atleast_1d(times):
        i = int(round(float(tt) * sr))
        if 0 <= i < len(y) - n:
            y[i:i + n] += tone
    return y


def test_transient_fx_finds_high_band_events():
    times = np.array([0.5, 1.5, 2.5, 3.5])
    y = _hf_burst(times, 4.5)
    tr = onsets.transient_fx(y, sr=SR)
    assert tr.heuristic is True
    assert tr.count >= 3
    for t in times[:3]:
        assert np.min(np.abs(tr.times - t)) < 0.06


def test_transient_fx_ignores_low_frequency_only_events():
    """低频瞬态（底鼓）不该进 `fx` —— 相对门要求高频占比够高。"""
    t = np.arange(int(4.0 * SR)) / SR
    y = np.zeros_like(t, dtype=np.float32)
    for tt in (0.5, 1.5, 2.5):
        m = (t >= tt) & (t < tt + 0.08)
        y[m] += (np.sin(2 * np.pi * 60.0 * t[m])
                 * np.exp(-12 * (t[m] - tt))).astype(np.float32)
    assert onsets.transient_fx(y, sr=SR).count <= 1


def test_transient_fx_empty_input():
    assert onsets.transient_fx(np.zeros(0), sr=SR).count == 0


# ---- pitch_notes（不依赖 .venv-pitch，只测纯 numpy 侧）----


def _pn(on, off, pit, conf=None):
    on = np.asarray(on, dtype=float)
    return pn_mod.PitchNotes(
        "vocals", on, np.asarray(off, dtype=float), np.asarray(pit, dtype=float),
        np.asarray(conf if conf is not None else np.full(on.size, 0.6), dtype=float))


def test_pitch_notes_dict_roundtrip():
    a = _pn([0.0, 1.0], [0.5, 1.5], [60, 64], [0.9, 0.3])
    b = pn_mod.PitchNotes.from_dict(a.to_dict())
    assert b.count == 2
    assert np.allclose(b.onsets, a.onsets)
    assert np.allclose(b.pitches, a.pitches)
    assert b.available is True


def test_pitch_notes_unavailable_is_empty_not_crashing():
    u = pn_mod.PitchNotes.unavailable("vocals", "没装")
    assert u.count == 0 and u.available is False and "没装" in u.error
    assert u.strong_mask().size == 0


def test_lead_mask_keeps_top_voice_only():
    """一个三音和弦只留最高音（basic-pitch 是复音转录器，不筛会把候选池撑爆）。"""
    n = _pn([0.0, 0.0, 0.0, 1.0], [0.9, 0.9, 0.9, 1.9], [60, 64, 67, 72])
    m = pn_mod.lead_mask(n)
    assert list(m) == [False, False, True, True]
    assert pn_mod.filtered(n, lead_only=True).count == 2


def test_filtered_by_confidence_and_duration():
    n = _pn([0.0, 1.0, 2.0], [0.05, 1.8, 2.8], [60, 62, 64], [0.9, 0.2, 0.8])
    assert pn_mod.filtered(n, min_confidence=0.5).count == 2
    assert pn_mod.filtered(n, min_duration_sec=0.1).count == 2


def test_pitched_coverage_per_bar_is_union_not_sum():
    """和声/复音会重叠 —— 覆盖率必须按区间**并集**算，否则会超过 1。"""
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=4.0)   # 1 小节 = 2s
    n = _pn([0.0, 0.0, 0.5], [1.0, 1.0, 1.0], [60, 64, 67])
    cov = pn_mod.pitched_coverage_per_bar(n, g)
    assert 0.49 < cov[0] < 0.51
    assert cov.max() <= 1.0


def test_notes_per_bar_counts_by_onset():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=8.0)
    n = _pn([0.1, 0.2, 2.5], [0.3, 0.4, 2.9], [60, 62, 64])
    npb = pn_mod.notes_per_bar(n, g)
    assert npb[0] == 2 and npb[1] == 1


def test_as_activity_matches_note_intervals():
    n = _pn([0.0, 2.0], [1.0, 3.0], [60, 62])
    ft = np.arange(0, 4, 0.1)
    act = pn_mod.as_activity(n, ft)["active"]
    assert act[0] and not act[15] and act[21]


def test_pitch_change_times_drops_repeated_pitch():
    n = _pn([0.0, 0.5, 1.0], [0.4, 0.9, 1.4], [60, 60, 64])
    assert np.allclose(pn_mod.pitch_change_times(n), [0.0, 1.0])


def test_last_json_line_ignores_progress_noise():
    assert pn_mod._last_json_line('Predicting MIDI...\n{"results": []}\n') == {
        "results": []}
    assert pn_mod._last_json_line("boom\n") is None


def test_detect_batch_without_venv_returns_unavailable(tmp_path):
    out = pn_mod.detect_batch({"vocals": tmp_path / "v.wav"},
                              venv=tmp_path / "no-such-venv")
    assert out["vocals"].available is False
    assert ".venv-pitch" in out["vocals"].error


# ---- melody / 轨集 ----


def test_merge_times_dedupes_within_window():
    m = tracks.merge_times([0.0, 0.01, 0.5], [0.02, 1.0])
    assert np.allclose(m, [0.0, 0.5, 1.0])


def test_build_melody_times_merges_lead_voices():
    notes = {
        "other": _pn([0.0, 0.0, 1.0], [0.9, 0.9, 1.9], [60, 67, 62]),
        "piano": _pn([2.0], [2.5], [72]),
    }
    m = tracks.build_melody_times(notes, lead_only=True)
    assert np.allclose(m, [0.0, 1.0, 2.0])
    assert tracks.build_melody_times(notes, lead_only=False).size == 3


def test_track_order_v5_is_still_four_tracks():
    assert len(tracks.TRACK_ORDER_V5) == 4
    assert tracks.TRACK_ORDER_V5 == ("drum", "vocal", "melody", "bass")
    assert "fx" in tracks.JSON_ONLY_TRACKS


def test_build_track_patterns_renders_melody_track():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=2.0)
    div = {b: quantize.BarDivision(bar=b, division=8)
           for b in range(1, g.n_bars + 1)}
    times = {"drums": np.zeros(0), "vocals": np.zeros(0), "bass": np.zeros(0),
             "melody": np.array([0.0, 1.0])}
    slots = {"melody": {1: [0, 4]}}
    out = tracks.build_track_patterns(g, div, times, slots, {}, None,
                                      track_order=tracks.TRACK_ORDER_V5)
    assert set(out) == set(tracks.TRACK_ORDER_V5)
    assert out["melody"][1].pattern == "x...x..."


# ---- features / stemplan 的 v0.5 扩展 ----


def test_build_bar_features_adds_pitched_columns():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=8.0)
    notes = {"vocals": _pn([0.0], [1.0], [60])}
    feats = features.build_bar_features(
        g, {}, SR, {"melody": np.array([0.1]), "fx": np.array([0.2]),
                    "guitar": np.array([0.3])},
        np.zeros(g.n_bars), pitch_notes=notes)
    assert "n_pnote_vocals" in feats and "pitched_cov_vocals" in feats
    assert "vocal_pitched_ratio" in feats
    assert "n_onset_melody" in feats and "n_onset_fx" in feats
    assert "n_onset_guitar" in feats


def test_detect_stem_set_switches_to_v5():
    assert stemplan.detect_stem_set({"n_onset_drums": np.zeros(4)}) == stemplan.STEMS
    got = stemplan.detect_stem_set({f"n_onset_{k}": np.zeros(4)
                                    for k in stemplan.STEMS_V5})
    assert set(got) == set(stemplan.STEMS_V5)


def _stem_feats_v5(n=32, **over):
    f = _stem_feats(n)
    for s in ("melody", "fx", "guitar", "piano"):
        f[f"share_{s}"] = np.full(n, 0.1)
        f[f"n_onset_{s}"] = np.full(n, 6.0 if s != "fx" else 3.0)
        f[f"grid_fit_{s}"] = np.full(n, 0.9)
        f[f"grid_fit_bar_{s}"] = np.full(n, 0.9)
    f["vocal_pitched_ratio"] = np.full(n, 0.6)
    f.update(over)
    return f


def test_fx_never_becomes_skeleton():
    """`fx` 是点缀音的定义，任何情况下都不该当骨架。"""
    feats = _stem_feats_v5(n_onset_drums=np.zeros(32), n_onset_fx=np.full(32, 99.0))
    segs = [FakeSeg(1, 32, "interlude")]
    out, _ = stemplan.plan_stems(segs, feats, _FakeGrid(32), 120.0)
    assert out[0].skeleton_stem != "fx"


def test_verse_accent_can_pick_piano():
    """v0.5 标定：`piano` 的 lift 2.44 是非鼓轨最高 → 进 verse 的点缀候选。"""
    feats = _stem_feats_v5(n_onset_other=np.full(32, 1.0),
                           n_onset_piano=np.full(32, 9.0))
    segs = [FakeSeg(1, 32, "verse")]
    out, _ = stemplan.plan_stems(segs, feats, _FakeGrid(32), 120.0)
    assert out[0].accent_stems[0] == "piano"


def test_melody_never_enters_accent_ranking():
    """`melody` 密度虚高（lift 1.61 全表最低）→ 只当候选池，不进 accent 排序。

    实测依据：让它参与排序时 379 段里 232 段把它排成第一点缀，
    「第一点缀落在实测 lift 前三」从 0.699 掉到 0.356（报告 §8.1）。
    """
    feats = _stem_feats_v5(n_onset_melody=np.full(32, 99.0))
    for fn in ("verse", "chorus", "interlude", "intro", "pre_chorus", "quiet_chorus"):
        out, _ = stemplan.plan_stems([FakeSeg(1, 32, fn)], feats, _FakeGrid(32), 120.0)
        assert "melody" not in out[0].accent_stems, fn
        assert out[0].skeleton_stem != "melody", fn


def test_fx_is_sparse_rule_triggered_not_density_ranked():
    """`fx` recall 0.043 但 precision 0.460 —— 按密度永远排不上，必须走专门规则。"""
    feats = _stem_feats_v5()
    out, _ = stemplan.plan_stems([FakeSeg(1, 32, "verse")], feats, _FakeGrid(32), 120.0)
    assert "fx" not in out[0].accent_stems


def test_lift_prior_downweights_dense_low_value_tracks():
    prior = stemplan.lift_prior
    assert prior("drums") == 1.0
    assert prior("piano") > prior("other") > prior("vocals") > prior("melody")


def test_plan_uses_pitched_ratio_when_available():
    """人声活动优先读有音高覆盖率（能量 VAD 是 n=40 认定的坏探测器）。"""
    feats = _stem_feats_v5(voiced_ratio=np.full(32, 1.0),
                           vocal_pitched_ratio=np.full(32, 0.02))
    segs = [FakeSeg(1, 32, "chorus")]
    out, _ = stemplan.plan_stems(segs, feats, _FakeGrid(32), 120.0)
    ev = " ".join(out[0].plan_evidence)
    assert "vocal_pitched_ratio=0.02" in ev
    assert "已知不可靠" in ev


def test_fx_lights_up_as_sparse_accent_when_present():
    """`fx` 不占 accent_stems 名额，走 `sparse_accents`。"""
    feats = _stem_feats_v5(n_onset_fx=np.full(32, 2.0))
    segs = [FakeSeg(1, 32, "final_chorus")]
    out, _ = stemplan.plan_stems(segs, feats, _FakeGrid(32), 120.0)
    assert out[0].sparse_accents == ["fx"]
    assert "fx" not in out[0].accent_stems


def test_fx_stays_dark_when_absent():
    feats = _stem_feats_v5(n_onset_fx=np.zeros(32))
    out, _ = stemplan.plan_stems([FakeSeg(1, 32, "verse")], feats,
                                 _FakeGrid(32), 120.0)
    assert out[0].sparse_accents == []


# ---------------------------------------------------------------------------
# song sheet 的「写谱提示层」（sheet.py v0.6，2026-09-20；只加不删）
# ---------------------------------------------------------------------------

from tools.audio_analysis import sheet as sheet_mod  # noqa: E402
from tools.audio_analysis.structure import Segment  # noqa: E402


def _seg(start, end, func, tier="mid", inten=0.5, skeleton="drum"):
    return Segment(start_bar=start, end_bar=end, function=func,
                   intensity=inten, intensity_tier=tier, skeleton_stem=skeleton)


def _rows(n, **counts):
    return [{"bar": b, "features": {k: v for k, v in counts.items()}}
            for b in range(1, n + 1)]


def test_writing_hints_quotes_entry_numbers_and_has_no_threshold():
    h = sheet_mod.writing_hints(_seg(1, 8, "chorus", "high", 0.8), 13.6, "偏多")
    assert set(h) == {"配置", "采音", "手序"}
    # 三条都必须引用条目号
    assert "知识 074" in h["配置"] and "知识 073" in h["配置"]
    assert "知识 069" in h["采音"] and "知识 005" in h["采音"]
    for k in ("064", "065", "067", "030", "082"):
        assert k in h["手序"], k
    # 不出现阈值：整段文字里不允许出现"≥ / ≤ / ms / 阈值"这类量化门槛
    joined = "".join(h.values())
    for bad in ("≥", "≤", "阈值", " ms", "毫秒"):
        assert bad not in joined, bad


def test_writing_hints_intro_points_to_vertical_family():
    h = sheet_mod.writing_hints(_seg(1, 8, "intro", "low", 0.2), 13.6, "偏少")
    assert "纵连" in h["配置"]


def test_writing_hints_star_family_switches_with_level():
    low = sheet_mod.writing_hints(_seg(1, 8, "chorus", "high", 0.8), 13.2, "偏多")
    high = sheet_mod.writing_hints(_seg(1, 8, "chorus", "high", 0.8), 14.2, "偏多")
    assert "绕圈星星" in low["配置"] and "接得顺" in low["配置"]
    assert "三叉戟" in high["配置"] and "要分手" in high["配置"]


def test_writing_hints_sampling_cells_follow_069():
    dense_hot = sheet_mod.writing_hints(_seg(1, 8, "chorus", "peak", 0.9), 13.5, "偏多")
    sparse_hot = sheet_mod.writing_hints(_seg(1, 8, "chorus", "peak", 0.9), 13.5, "偏少")
    assert "只踩其中一部分" in dense_hot["采音"]
    assert "都写成双押" in sparse_hot["采音"]


def test_writing_hints_quiet_chorus_mentions_chuzhang():
    h = sheet_mod.writing_hints(_seg(1, 8, "quiet_chorus", "mid", 0.5), 13.5, "居中")
    assert "出张最集中" in h["手序"]


def test_segment_pool_rank_is_relative_within_song():
    segs = [_seg(1, 2, "intro"), _seg(3, 4, "verse"), _seg(5, 6, "chorus")]
    rows = ([{"bar": b, "features": {"n_onset_drums": 1.0}} for b in (1, 2)]
            + [{"bar": b, "features": {"n_onset_drums": 5.0}} for b in (3, 4)]
            + [{"bar": b, "features": {"n_onset_drums": 9.0}} for b in (5, 6)])
    rank = sheet_mod.segment_pool_rank(segs, rows)
    assert rank[0] == "偏少" and rank[2] == "偏多"


def test_song_sheet_md_contains_writing_hints_section():
    payload = {
        "song": {"name": "t", "duration_sec": 10.0},
        "grid": {"bpm": 160.0, "bpm_changes": [], "first": 0.0, "n_bars": 2,
                 "beats_per_bar": 4},
        "offset_check": {"available": False},
        "structure": {"method": "test"},
        "intensity": {"climax_bar": 1, "climax_peaks": [1]},
        "quantize": {"division_share": {"16": 1.0}, "resolved_bars": 2,
                     "bars_with_onsets": 2, "division_histogram": {"16": 2},
                     "division_histogram_resolved": {"16": 2},
                     "unquantized_onsets": 0, "onsets_considered": 10,
                     "unquantized_ratio": 0.0, "fine_blocked_bars": 0,
                     "triplet_bars": 0, "unresolved_bars": 0, "per_track": {}},
        "target": {"level": 13.6, "total_p10": 492, "total_p90": 967,
                   "total_mean": 767.4, "nps": 6.13, "nps_source": "知识 031 §7",
                   "notes_per_bar": 9.0, "notes_per_bar_level_anchor": 9.15,
                   "density_floor": {"section": 0.6, "bar": 0.25,
                                     "section_calibrated_optimum": 0.365},
                   "structural_cap": {"applied": False}},
        "warnings": [],
    }
    segs = [_seg(1, 1, "intro", "low", 0.2), _seg(2, 2, "chorus", "peak", 0.9)]
    rows = [{"bar": 1, "intensity": 0.2, "division": 16, "patterns": {},
             "features": {"n_onset_drums": 2.0}},
            {"bar": 2, "intensity": 0.9, "division": 16, "patterns": {},
             "features": {"n_onset_drums": 8.0}}]
    md = sheet_mod.build_song_sheet_md(payload, segs, rows)
    assert "## 1.5 写谱提示（按段落）" in md
    assert "建议配置族" in md and "采音建议" in md and "手序提示" in md


# ==================================================================
# tempo map 复核（grid v1.5）——两个「旧方法抓不住、新方法必须抓住」的用例
# ==================================================================
#
# 旧方法 = `docs/audio-analysis.md` §2.4 第 5 步：8 小节一块，在 **32 分网格**
# （±半格 ≈ ±17 ms @205BPM）上求相位 φ_i，再对 (t_i, φ_i) 线性回归。
# 本节用合成信号把它的两个盲点钉死：
#   ① 半拍错位 —— 半拍 = 16 个 32 分格，在 32 分网格上**混叠成 0**，旧法读数完美；
#   ② BPM +0.5% —— 一块之内相位就绕过一整格，旧法回归到的斜率是绕圈残值。

from tools.audio_analysis import grid as grid_mod  # noqa: E402

TSR = grid_mod.TEMPO_SR


def _tone_burst(freqs, n, sr, decay):
    t = np.arange(n) / sr
    env = np.exp(-t / decay)
    y = np.zeros(n)
    for f in np.atleast_1d(freqs):
        y += np.sin(2 * np.pi * float(f) * t)
    return (y / len(np.atleast_1d(freqs))) * env


def _bandlimit(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """对**整条信号**做理想带限（rfft 置零）。

    必须整条做，不能逐个 patch 做：瞬态 patch 的起始跳变会把宽带能量溅到
    其它频带（实测 8 kHz 的 hat 会在 35–120 Hz 的 kick 带里造出假击打），
    那样测的就不是算法而是合成信号的缺陷了。
    """
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1.0 / sr)
    Y[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(Y, n=len(y))


def _place(y, times, patch, sr, gain=1.0):
    for t in np.atleast_1d(times):
        i = int(round(float(t) * sr))
        if i < 0 or i >= len(y):
            continue
        m = min(len(patch), len(y) - i)
        y[i:i + m] += gain * patch[:m]


def synth_drumkit(bpm: float, first: float, n_bars: int, sr: int = TSR,
                  beats_per_bar: int = 4):
    """合成一套**分带干净**的鼓组：kick 四分四拍 / hat 八分 / bass 八分反拍。

    - kick 55 Hz，整条带限到 40–110 Hz（落在 `TEMPO_BANDS['kick']` 35–120 内）
    - hat 8k+11k Hz，带限到 6.5–14 kHz（落在 6–15 kHz 内）
    - bass 180 Hz，带限到 140–290 Hz（落在 30–300 内，但**避开** kick 的 35–120）

    bass 故意打在八分反拍上，复刻真实曲风里的 **offbeat bass**——
    半拍归属必须靠 kick/hat 判，bass 指向反拍不算反例。
    """
    spb = 60.0 / float(bpm)
    bar = beats_per_bar * spb
    dur = first + n_bars * bar + 1.0
    n = int(round(dur * sr))
    kick = np.zeros(n)
    hat = np.zeros(n)
    bass = np.zeros(n)
    kick_p = _tone_burst([55.0], int(0.13 * sr), sr, 0.030)
    hat_p = _tone_burst([8000.0, 11000.0], int(0.03 * sr), sr, 0.006)
    bass_p = _tone_burst([180.0], int(0.11 * sr), sr, 0.030)
    beats = first + np.arange(n_bars * beats_per_bar) * spb
    _place(kick, beats, kick_p, sr, 1.0)                  # 整拍：kick
    _place(hat, beats, hat_p, sr, 1.0)                    # 整拍：hat
    _place(hat, beats + spb / 2.0, hat_p, sr, 0.55)       # 反拍：只有 hat（更弱）
    _place(bass, beats + spb / 2.0, bass_p, sr, 1.0)      # 反拍：bass
    # 轻度带限：削掉瞬态起跳溅到别的频带的部分，但保留真实的攻击形状
    kick = _bandlimit(kick, sr, 30.0, 400.0)
    hat = _bandlimit(hat, sr, 5000.0, 16000.0)
    bass = _bandlimit(bass, sr, 130.0, 900.0)
    drums = kick / (np.max(np.abs(kick)) or 1.0) + 0.35 * hat / (np.max(np.abs(hat)) or 1.0)
    drums = drums / (np.max(np.abs(drums)) or 1.0) * 0.8
    bass = bass / (np.max(np.abs(bass)) or 1.0) * 0.8
    # 底噪：真实音频不会有「数字静音的频带」，没有底噪时 dB flux 会在空带上爆炸
    rng = np.random.default_rng(7)
    drums = drums + rng.standard_normal(n) * 1e-4
    bass = bass + rng.standard_normal(n) * 1e-4
    return drums.astype(np.float32), bass.astype(np.float32), float(dur)


def _legacy_block_phase(onset_times, bpm: float, first: float,
                        beats_per_bar: int = 4, bars_per_block: int = 8,
                        division: int = 32):
    """复刻**旧做法**：8 小节一块 + 32 分网格 ±半格相位搜索 + 线性回归。

    只在测试里存在，用来证明新法确实堵住了旧法的盲点。
    返回 (block_times, block_phases_sec, slope)。
    """
    t = np.asarray(onset_times, dtype=float)
    bar = beats_per_bar * 60.0 / float(bpm)
    slot = bar / float(division)
    block = bars_per_block * bar
    n_blocks = max(1, int((t.max() - first) / block))
    cand = np.linspace(-slot / 2.0, slot / 2.0, 201)
    ts, ph = [], []
    for b in range(n_blocks):
        a = first + b * block
        m = (t >= a) & (t < a + block)
        if int(m.sum()) < 8:
            continue
        # 每个候选相位下，onset 到最近格线的命中率
        score = [float(np.mean(np.abs(((t[m] - first - c + slot / 2.0) % slot)
                                      - slot / 2.0) < slot * 0.2)) for c in cand]
        ts.append(a + block / 2.0)
        ph.append(float(cand[int(np.argmax(score))]))
    slope = float(np.polyfit(ts, ph, 1)[0]) if len(ts) >= 3 else 0.0
    return np.array(ts), np.array(ph), slope, slot


def test_tempo_half_beat_offset_caught_by_new_blind_to_legacy():
    """用例①：`&first` 整体错半拍 —— 旧法完全看不见，新法必须指出来。"""
    bpm, first, n_bars = 180.0, 0.5, 24
    drums, bass, dur = synth_drumkit(bpm, first, n_bars)
    onsets_hi = grid_mod.high_res_onsets(drums, sr=TSR)
    half = 0.5 * 60.0 / bpm
    wrong_first = first + half

    # --- 旧法：32 分网格块相位，对半拍错位完全免疫（混叠成 0）---
    _, ph_ok, _, slot = _legacy_block_phase(onsets_hi["times"], bpm, first)
    _, ph_bad, _, _ = _legacy_block_phase(onsets_hi["times"], bpm, wrong_first)
    assert np.max(np.abs(ph_bad)) <= slot / 2.0
    # 错半拍与不错半拍，旧法给出的相位读数几乎一模一样 → 它分不开
    assert abs(float(np.median(np.abs(ph_bad))) - float(np.median(np.abs(ph_ok)))) < 0.005

    # --- 新法：分带 flux + 只在干净小节上判 ---
    bands = {n: grid_mod.band_flux(drums if n != "bass" else bass, sr=TSR,
                                   f_lo=lo, f_hi=hi)
             for n, (lo, hi) in grid_mod.TEMPO_BANDS.items()}
    ft = bands["kick"]["frame_times"]
    flux = {k: v["flux"] for k, v in bands.items()}

    clean_ok = grid_mod.cleanest_drum_bars(flux["kick"], ft, bpm=bpm, first=first,
                                           n_bars=n_bars)
    assert clean_ok, "合成信号是标准 4-on-the-floor，必须能挑出干净小节"
    res_ok = grid_mod.half_beat_attribution(flux, ft, bpm=bpm, first=first, bars=clean_ok)
    assert res_ok["verdict"] == "first"
    assert res_ok["bands"]["kick"]["ratio"] > grid_mod.TEMPO_ATTR_RATIO_GATE
    # bass 打在反拍（offbeat bass）：指向反拍但**不**推翻结论
    assert res_ok["bands"]["bass"]["ratio"] < 1.0
    assert "offbeat bass" in res_ok["reason"]

    clean_bad = grid_mod.cleanest_drum_bars(flux["kick"], ft, bpm=bpm,
                                            first=wrong_first, n_bars=n_bars - 1)
    res_bad = grid_mod.half_beat_attribution(flux, ft, bpm=bpm, first=wrong_first,
                                             bars=clean_bad)
    assert res_bad["verdict"] == "first+half", res_bad["reason"]
    assert abs(res_bad["alt_first"] - (wrong_first + half)) < 1e-9


def test_tempo_bpm_half_percent_error_caught_by_new_blind_to_legacy():
    """用例②：真实 BPM 比用户值高 0.5% —— 旧法因绕圈而失明，新法必须测出来。"""
    user_bpm = 180.0
    true_bpm = user_bpm * 1.005          # 180.9
    first, n_bars = 0.5, 24
    drums, bass, dur = synth_drumkit(true_bpm, first, n_bars)
    onsets_hi = grid_mod.high_res_onsets(drums, sr=TSR)

    # --- 旧法：相位在一块之内就绕过一整格，读数被折回 ±半格 ---
    ts, ph, slope, slot = _legacy_block_phase(onsets_hi["times"], user_bpm, first)
    drift_over_block = 8 * 4 * 60.0 / user_bpm * 0.005
    assert drift_over_block > slot, "构造前提：一块的真实漂移必须超过一个 32 分格"
    assert np.max(np.abs(ph)) <= slot / 2.0 + 1e-9      # 被折回，看不出累积
    # 旧法回归到的斜率远小于真实的 0.005，至少低估 5 倍 → 判成「无漂移」
    assert abs(slope) < 1e-3
    assert abs(slope) < 0.2 * 0.005
    assert abs(user_bpm * (1 - slope) - true_bpm) > 0.5   # 反推的 BPM 也是错的

    # --- 新法之一：无相位滑窗扫描，逐窗都指向真实 BPM ---
    wins = grid_mod.window_bpm_scan(onsets_hi["times"], onsets_hi["weights"],
                                    bpm=user_bpm, first=first, n_bars=n_bars,
                                    window_bars=4)
    loc = np.array([w["best_local"] for w in wins if w["valid"]], dtype=float)
    assert loc.size >= 8
    assert abs(float(np.median(loc)) - true_bpm) < 0.3, float(np.median(loc))
    assert abs(float(np.median(loc)) - user_bpm) > grid_mod.TEMPO_CONST_TOL_BPM

    # --- 新法之二：主体段高精度拟合 ---
    fit = grid_mod.fine_bpm_fit(onsets_hi["times"], onsets_hi["weights"],
                                first, first + n_bars * 4 * 60.0 / user_bpm,
                                user_bpm - 5, user_bpm + 5, report_at=(user_bpm,))
    assert fit["available"] and abs(fit["bpm"] - true_bpm) < 0.15

    # --- 总装：verdict 不得是 constant，且必须点名是哪条判据没过 ---
    tm = grid_mod.tempo_map(y_mix=drums + bass, y_drums=drums, y_bass=bass, sr=TSR,
                            bpm=user_bpm, first=first, duration=dur,
                            use_beat_this=False, use_librosa=False)
    assert tm["available"] and tm["verdict"] != "constant"
    # 曲子本身是稳的，只是用户给的 BPM 错了 → 单独一档，并报出实测值与累计偏离
    assert tm["verdict"] == "constant_other_bpm", tm["criteria"]
    assert tm["criteria"]["body_fit_matches_user"]["pass"] is False
    assert abs(tm["detected_bpm"] - true_bpm) < 0.15
    assert tm["drift_vs_user_ms"] > 100.0
    v = grid_mod.tempo_map_verdict(tm)
    assert "不等于用户给的 BPM" in v and "请改 --bpm" in v


def test_tempo_map_constant_positive_control():
    """正对照：BPM / first 都正确时必须判「恒定」，半拍归属维持 first。"""
    bpm, first, n_bars = 180.0, 0.5, 24
    drums, bass, dur = synth_drumkit(bpm, first, n_bars)
    tm = grid_mod.tempo_map(y_mix=drums + bass, y_drums=drums, y_bass=bass, sr=TSR,
                            bpm=bpm, first=first, duration=dur,
                            use_beat_this=False, use_librosa=False)
    assert tm["verdict"] == "constant", tm["criteria"]
    assert all(c["pass"] for c in tm["criteria"].values())
    assert tm["half_beat"]["verdict"] == "first"
    assert tm["phase_jumps"] == []
    assert abs(tm["body_fit"]["bpm"] - bpm) < 0.1
    assert tm["scan"]["phase_free"] is True
    v = grid_mod.tempo_map_verdict(tm)
    assert "恒定" in v and "无跳格" in v


@pytest.mark.parametrize("shift_div, label", [(4, "16 分"), (8, "32 分")])
def test_tempo_phase_jump_detected(shift_div, label):
    """中途插入一个 16 分 / 32 分的接缝 —— 8 分量程的相位曲线必须看得见跳格。

    这正是旧法（32 分网格 ±17 ms）**结构上**看不见的一类故障：
    16 分 = 2 个 32 分格、32 分 = 1 个 32 分格，在那张网格上都混叠成 0。
    """
    bpm, first, n_bars = 180.0, 0.5, 20
    spb = 60.0 / bpm
    shift = spb / shift_div
    beats = first + np.arange(n_bars * 4) * spb
    onsets_ = np.concatenate([beats, beats + spb / 2.0])      # 四分 + 八分反拍
    onsets_ = np.sort(onsets_)
    shifted = onsets_.copy()
    shifted[onsets_ >= first + (n_bars // 2) * 4 * spb] += shift
    curve = grid_mod.bar_phase_curve(shifted, np.ones_like(shifted), bpm=bpm,
                                     first=first, n_bars=n_bars, min_onsets=3)
    jumps = grid_mod.detect_phase_jumps(curve)
    assert jumps, f"{label}接缝必须被检出"
    # 8 分格的量程是 ±半格；16 分恰好是半格（符号简并），故只比幅度
    assert any(abs(abs(j["jump_ms"]) - shift * 1000.0) < 6.0 for j in jumps), \
        [round(j["jump_ms"], 1) for j in jumps]

    # 旧法对照：32 分网格上，这个接缝是整数个格 → 相位读数不动
    _, ph, _, slot = _legacy_block_phase(shifted, bpm, first, bars_per_block=4)
    assert np.max(np.abs(ph)) <= slot / 2.0 + 1e-9


def test_tempo_no_drum_bars_are_not_evidence():
    """无鼓段只标 `no_drum_bars`，不进 onset 验证，**也不得被外推成 constant**。

    v1.5.1（test-01 事故）：旧行为在这里报 `constant`，等于宣称
    「挖空的那几小节也是这个速度」——正是把一首前奏 195 / 尾奏 221→108 的曲子
    判成「全曲恒定 205」的那个 bug。
    """
    bpm, first, n_bars = 180.0, 0.5, 24
    drums, bass, dur = synth_drumkit(bpm, first, n_bars)
    bar = 4 * 60.0 / bpm
    silent = drums.copy()
    a, b = int((first + 4 * bar) * TSR), int((first + 8 * bar) * TSR)
    silent[a:b] = 0.0                              # bars 5–8 挖空
    tm = grid_mod.tempo_map(y_mix=silent + bass, y_drums=silent, y_bass=bass, sr=TSR,
                            bpm=bpm, first=first, duration=dur,
                            use_beat_this=False, use_librosa=False)
    assert set(range(5, 9)) <= set(tm["no_drum_bars"])
    assert tm["verdict"] == "constant_in_measured", (tm["verdict"], tm["criteria"])
    assert tm["measured_ratio"] < grid_mod.TEMPO_MEASURED_RATIO_GATE
    assert [5, 8] in tm["unmeasured_ranges"]
    assert "不做 onset 验证" in " ".join(tm["notes"])
    v = grid_mod.tempo_map_verdict(tm)
    assert "可测区间" in v and "不得外推" in v and "5–8" in v


def test_tempo_slow_intro_plus_constant_body_is_not_constant():
    """**前奏慢速 + 主体恒速**：绝不能判 `constant`（test-01 的原始事故）。

    两个变体都测：
      (a) 前奏有鼓但慢 → 滑窗能看见 → 判据直接不过；
      (b) 前奏没鼓（真实曲子的样子）→ 不可测占比超门槛 → `constant_in_measured`。
    """
    intro_bpm, body_bpm, first = 150.0, 200.0, 0.5
    n_intro, n_body = 8, 24
    d1, b1, _ = synth_drumkit(intro_bpm, first, n_intro)
    intro_end = first + n_intro * 4 * 60.0 / intro_bpm
    d2, b2, _ = synth_drumkit(body_bpm, intro_end, n_body)
    n = max(len(d1), len(d2))
    def pad(x):
        y = np.zeros(n, dtype=np.float32); y[:len(x)] += x; return y
    # 前奏段只保留 d1、主体段只保留 d2
    cut = int(intro_end * TSR)
    drums = pad(d1).copy(); drums[cut:] = 0.0
    tail = pad(d2).copy(); tail[:cut] = 0.0
    drums = drums + tail
    bass = pad(b1).copy(); bass[cut:] = 0.0
    bass = bass + np.where(np.arange(n) >= cut, pad(b2), 0).astype(np.float32)
    dur = n / TSR

    tm = grid_mod.tempo_map(y_mix=drums + bass, y_drums=drums, y_bass=bass, sr=TSR,
                            bpm=body_bpm, first=first, duration=dur,
                            use_beat_this=False, use_librosa=False)
    assert tm["verdict"] != "constant", (tm["verdict"], tm["criteria"])

    # (b) 把前奏的鼓挖掉 —— 这才是 test-01 的真实形态
    silent = drums.copy(); silent[:cut] = 0.0
    tm2 = grid_mod.tempo_map(y_mix=silent + bass, y_drums=silent, y_bass=bass, sr=TSR,
                             bpm=body_bpm, first=first, duration=dur,
                             use_beat_this=False, use_librosa=False)
    assert tm2["verdict"] != "constant", (tm2["verdict"], tm2["criteria"])
    assert tm2["verdict"] == "constant_in_measured", tm2["verdict"]
    assert tm2["unmeasured_ranges"] and tm2["unmeasured_ranges"][0][0] == 1


def test_tempo_hypothesis_scores_candidate_map():
    """候选 tempo map 检验：真值分段必须优于恒速外推，且短段一律判 `adopted`。"""
    intro_bpm, body_bpm, first = 150.0, 200.0, 0.5
    n_intro, n_body = 8, 24
    d1, b1, _ = synth_drumkit(intro_bpm, first, n_intro)
    intro_end = first + n_intro * 4 * 60.0 / intro_bpm
    d2, b2, _ = synth_drumkit(body_bpm, intro_end, n_body)
    n = max(len(d1), len(d2))
    cut = int(intro_end * TSR)
    y = np.zeros(n, dtype=np.float32)
    y[:cut] += d1[:cut]
    y[cut:len(d2)] += d2[cut:]
    o = grid_mod.high_res_onsets(y, sr=TSR)

    segs = [(intro_bpm, n_intro), (body_bpm, n_body)]
    res = grid_mod.test_tempo_hypothesis(o["times"], o["weights"], first, segs)
    assert res["available"]
    assert res["overall_score_hypothesis"] > res["overall_score_baseline"] * 1.2
    assert res["baseline_bpm"] == body_bpm          # 小节最多的那段
    assert res["counts"]["contradicted"] == 0
    assert res["segments"][0]["verdict"] in ("measured", "confirmed")

    # 短段（1 小节）在物理上分不开 → 必须 adopted，不得硬判
    short = [(intro_bpm, 1), (170.0, 1), (190.0, 1), (body_bpm, n_body)]
    res2 = grid_mod.test_tempo_hypothesis(o["times"], o["weights"], first, short)
    for r in res2["segments"][:3]:
        assert r["verdict"] == "adopted", (r["bpm"], r["verdict"], r["note"])
        assert "无法分辨" in r["note"]
    assert "采用候选值" in grid_mod.tempo_hypothesis_verdict(res2)


def test_tempo_segments_to_beats_supports_half_bars():
    """半小节段（真实减速段常见）必须能表达。"""
    beats, info, end = grid_mod.tempo_segments_to_beats(
        0.0, [(160.0, 0.5), (120.0, 0.5), (200.0, 1)])
    assert len(beats) == 8
    assert info[1]["start_bar"] == 1 and info[1]["start_beat_in_bar"] == 2
    assert info[2]["start_bar"] == 2 and info[2]["start_beat_in_bar"] == 0
    assert abs(end - (2 * 60 / 160 + 2 * 60 / 120 + 4 * 60 / 200)) < 1e-9


def test_tempo_map_beat_this_optional_and_librosa_warned():
    """beat_this 可关；librosa 路必须带 hop 帧量化告警。"""
    bpm, first, n_bars = 180.0, 0.5, 12
    drums, bass, dur = synth_drumkit(bpm, first, n_bars)
    tm = grid_mod.tempo_map(y_mix=drums + bass, y_drums=drums, sr=TSR, bpm=bpm,
                            first=first, duration=dur,
                            use_beat_this=False, use_librosa=True)
    assert tm["cross_check"]["beat_this"]["available"] is False
    assert "关闭" in tm["cross_check"]["beat_this"]["reason"]
    lb = tm["cross_check"]["librosa"]
    if lb.get("available"):
        assert "量化" in lb["warning"] and "不得单独作结论" in lb["warning"]
    assert "不得单独作结论" in tm["cross_check"]["note"]


def test_tempo_segments_and_bar_table_on_variable_tempo():
    """判为变速时要给出 `(bpm)` 段落表与重算的小节起点秒。"""
    first = 0.0
    a_bpm, b_bpm, n_a, n_b = 180.0, 200.0, 16, 16
    spb_a, spb_b = 60.0 / a_bpm, 60.0 / b_bpm
    beats = list(first + np.arange(n_a * 4) * spb_a)
    t = beats[-1] + spb_a
    beats += list(t + np.arange(n_b * 4) * spb_b)
    times = np.array(beats)
    wins = grid_mod.window_bpm_scan(times, np.ones_like(times), bpm=a_bpm, first=first,
                                    n_bars=n_a + n_b, window_bars=4, local_span=20.0)
    segs = grid_mod.segment_tempo(wins, bpm=a_bpm, first=first)
    assert len(segs) >= 2, segs
    assert abs(segs[0]["bpm"] - a_bpm) < 1.0
    assert abs(segs[-1]["bpm"] - b_bpm) < 1.5
    changes = grid_mod.bpm_changes_from_segments(segs, base_bpm=segs[0]["bpm"])
    assert changes and changes[0].bar > 1
    table = grid_mod.bar_start_table(segs[0]["bpm"], first, bpm_changes=changes,
                                     n_bars=n_a + n_b)
    assert len(table) == n_a + n_b
    assert table[0]["start_sec"] == 0.0
    # 变速之后的小节应该比恒速外推得更密
    const = grid_mod.bar_start_table(a_bpm, first, n_bars=n_a + n_b)
    assert table[-1]["start_sec"] < const[-1]["start_sec"]


def synth_backbeat(bpm: float, first: float, n_bars: int, sr: int = TSR,
                   with_kick: bool = False):
    """只有军鼓/拍手（在第 2、4 拍）的鼓组——判下拍相位的标准测试台。

    `with_kick=True` 时再叠一层 4-on-the-floor kick，用来复现「在 drop 里数
    backbeat 会被抹平」的失败模式。
    """
    spb = 60.0 / float(bpm)
    dur = first + n_bars * 4 * spb + 1.0
    n = int(round(dur * sr))
    y = np.zeros(n)
    clap = _tone_burst([600.0, 1200.0, 2500.0, 4000.0, 6000.0],
                       int(0.05 * sr), sr, 0.010)
    kick = _tone_burst([55.0], int(0.13 * sr), sr, 0.030)
    bars_ = first + np.arange(n_bars) * 4 * spb
    for k in (1, 3):                       # 第 2、4 拍
        _place(y, bars_ + k * spb, clap, sr, 1.0)
    y = _bandlimit(y, sr, 400.0, 12000.0)   # 真实拍手从 ~700 Hz 起有能量
    if with_kick:
        k4 = np.zeros(n)
        for k in range(4):
            _place(k4, bars_ + k * spb, kick, sr, 1.0)
        y = y / (np.max(np.abs(y)) or 1.0) + 1.2 * _bandlimit(k4, sr, 30.0, 400.0) / (
            np.max(np.abs(_bandlimit(k4, sr, 30.0, 400.0))) or 1.0)
    rng = np.random.default_rng(3)
    y = y / (np.max(np.abs(y)) or 1.0) * 0.8 + rng.standard_normal(n) * 1e-4
    return y.astype(np.float32), float(dur)


@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_downbeat_phase_picks_backbeat(shift):
    """军鼓在 2/4 拍：给一个错了 `shift` 拍的 first，必须指回正确相位。"""
    bpm, true_first, n_bars = 180.0, 0.5, 24
    y, dur = synth_backbeat(bpm, true_first, n_bars)
    spb = 60.0 / bpm
    given = true_first + shift * spb          # 故意报晚 shift 拍
    fl = {n: grid_mod.band_flux(y, sr=TSR, f_lo=lo, f_hi=hi)
          for n, (lo, hi) in grid_mod.TEMPO_BANDS.items()}
    ft = fl["snare"]["frame_times"]
    res = grid_mod.downbeat_phase({k: v["flux"] for k, v in fl.items()}, ft,
                                  bpm=bpm, first=given, t_start=given,
                                  t_end=true_first + n_bars * 4 * spb,
                                  downbeat_times=true_first + np.arange(n_bars) * 4 * spb)
    assert res["available"]
    assert res["shift_beats"] == shift, (shift, res["reason"])
    assert abs(res["new_first"] - true_first) < 1e-6
    assert res["verdict"] == ("keep" if shift == 0 else f"shift_{shift}")
    # backbeat 只定 mod 2 —— 允许集合里必须同时含 shift 与 shift+2
    assert sorted(res["backbeat_allowed"]) == sorted([shift, (shift + 2) % 4])


def test_downbeat_phase_backbeat_flattened_by_four_on_the_floor():
    """叠上 4-on-the-floor kick 后 backbeat 被抹平 —— 必须如实报「无差别」。

    这正是 test-01 在 drop 段数 backbeat 得到 0.64/1.56 却在全曲平均上
    得到 1.00 的原因：**必须挑只有军鼓/拍手的段落**。
    """
    bpm, first, n_bars = 180.0, 0.5, 24
    y, dur = synth_backbeat(bpm, first, n_bars, with_kick=True)
    fl = {n: grid_mod.band_flux(y, sr=TSR, f_lo=lo, f_hi=hi)
          for n, (lo, hi) in grid_mod.TEMPO_BANDS.items()}
    ft = fl["snare"]["frame_times"]
    res = grid_mod.downbeat_phase({k: v["flux"] for k, v in fl.items()}, ft,
                                  bpm=bpm, first=first, t_start=first,
                                  t_end=first + n_bars * 4 * 60.0 / bpm)
    ratios = [c["backbeat_ratio"] for c in res["candidates"]]
    assert max(ratios) < grid_mod.TEMPO_BACKBEAT_RATIO_GATE * 1.6
    assert res["verdict"] == "undecided" or res["shift_beats"] == 0
    assert "军鼓/拍手" in res["reason"] or "backbeat" in res["reason"]
