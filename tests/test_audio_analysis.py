"""`tools/audio_analysis` 的单元测试。

全部使用 **numpy 合成音频**（已知 BPM / first / 分音的 click 序列），
不引用任何真实音频文件——仓库内不得出现有版权素材（AGENT.md 准则 5）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audio_analysis import onsets, quantize  # noqa: E402
from tools.audio_analysis.grid import (  # noqa: E402
    BpmChange, Grid, check_offset, parse_bpm_changes,
)

SR = onsets.ANALYSIS_SR


# ---------------- 合成音频工具 ----------------


def synth_clicks(times: np.ndarray, duration: float, sr: int = SR,
                 decay: float = 0.035, seed: int = 0) -> np.ndarray:
    """在给定秒数处放置短促噪声爆音（模拟鼓点），返回单声道信号。"""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(round(duration * sr)), dtype=np.float32)
    n = int(round(decay * sr))
    env = np.exp(-np.linspace(0, 6, n))
    for t in np.atleast_1d(times):
        i = int(round(float(t) * sr))
        if i < 0 or i >= len(y):
            continue
        m = min(n, len(y) - i)
        burst = rng.standard_normal(m).astype(np.float32) * env[:m]
        y[i:i + m] += burst
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.8).astype(np.float32)


def grid_times(grid: Grid, division: int, slots_per_bar, bars) -> np.ndarray:
    """按 (bar, division, slot 列表) 生成理想 onset 秒数。"""
    out = []
    for bar in bars:
        for slot in slots_per_bar:
            out.append(grid.subdivision_time(bar, division, slot))
    return np.asarray(sorted(out), dtype=float)


# ---------------- 1. 网格时间计算 ----------------


def test_grid_basic_times():
    g = Grid(bpm=120.0, first=1.0, beats_per_bar=4, duration=20.0)
    # 120 BPM 4/4 → 一小节 2.0s
    assert g.bar_duration(1) == pytest.approx(2.0)
    assert g.bar_start(1) == pytest.approx(1.0)
    assert g.bar_start(2) == pytest.approx(3.0)
    assert g.bar_start(5) == pytest.approx(9.0)
    assert g.beat_time(2, 2) == pytest.approx(3.0 + 1.0)
    assert g.n_bars == 10  # 1.0 起、每 2s 一小节，到 20s


def test_grid_negative_first():
    g = Grid(bpm=192.0, first=-0.25, beats_per_bar=4, duration=10.0)
    assert g.bar_start(1) == pytest.approx(-0.25)
    assert g.bar_duration(1) == pytest.approx(4 * 60.0 / 192.0)
    assert g.bar_start(2) == pytest.approx(-0.25 + 1.25)


def test_grid_subdivision_times():
    g = Grid(bpm=150.0, first=0.5, beats_per_bar=4, duration=30.0)
    bar_dur = 4 * 60.0 / 150.0  # 1.6s
    t = g.subdivision_times(3, 16)
    assert len(t) == 16
    assert t[0] == pytest.approx(g.bar_start(3))
    assert t[1] - t[0] == pytest.approx(bar_dur / 16)
    # 三连分音
    t12 = g.subdivision_times(3, 12)
    assert t12[1] - t12[0] == pytest.approx(bar_dur / 12)


def test_grid_bpm_changes():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4,
             bpm_changes=(BpmChange(bar=3, bpm=240.0),), duration=20.0)
    assert g.bar_duration(1) == pytest.approx(2.0)
    assert g.bar_duration(3) == pytest.approx(1.0)
    assert g.bar_start(3) == pytest.approx(4.0)
    assert g.bar_start(4) == pytest.approx(5.0)
    assert g.bar_bpm(2) == 120.0 and g.bar_bpm(3) == 240.0
    assert g.has_bpm_changes


def test_parse_bpm_changes():
    ch = parse_bpm_changes("33:180, 65:155")
    assert [(c.bar, c.bpm) for c in ch] == [(33, 180.0), (65, 155.0)]
    assert parse_bpm_changes(None) == ()
    with pytest.raises(ValueError):
        parse_bpm_changes("33-180")


def test_grid_time_to_bar_pos():
    g = Grid(bpm=120.0, first=1.0, beats_per_bar=4, duration=20.0)
    bar, pos = g.time_to_bar_pos(4.0)
    assert bar == 2 and pos == pytest.approx(0.5)
    assert g.bar_of(0.5) == 0  # 早于第 1 小节


# ---------------- 2. onset → 网格量化 ----------------


def test_choose_division_picks_minimum():
    bar_dur = 1.25  # 192BPM 4/4
    # 四分音：0, 1/4, 2/4, 3/4
    rel = np.array([0.0, 0.25, 0.5, 0.75]) * bar_dur
    d, idx, err, ok = quantize.choose_division(rel, bar_dur)
    assert d == 4 and ok
    assert list(idx) == [0, 1, 2, 3]
    assert np.max(np.abs(err)) < 1e-9


def test_choose_division_prefers_coarsest_that_explains():
    """`x...x...x.x.....` 这种 16 分写法其实落在 8 分网格上 → 应选 8，不选 16。"""
    bar_dur = 1.25
    rel = np.array([0, 4, 8, 10]) / 16 * bar_dur
    d, idx, _, ok = quantize.choose_division(rel, bar_dur)
    assert d == 8 and ok and list(idx) == [0, 2, 4, 5]


def test_choose_division_sixteenth_and_triplet():
    bar_dur = 1.6  # 150BPM 4/4，1/48 小节 = 33ms > 25ms 容差，16 与 12 可区分
    rel16 = np.array([0, 4, 7, 12]) / 16 * bar_dur   # 第 7 格是纯 16 分位置
    d, idx, _, ok = quantize.choose_division(rel16, bar_dur)
    assert d == 16 and ok and list(idx) == [0, 4, 7, 12]

    rel12 = np.array([0, 1, 2, 3, 6, 9]) / 12 * bar_dur
    d, idx, _, ok = quantize.choose_division(rel12, bar_dur)
    assert d == 12 and ok and list(idx) == [0, 1, 2, 3, 6, 9]


def test_choose_division_unresolved():
    """刻意放一个既不在 12 也不在 32 分网格上的音 → 标记 unresolved。"""
    bar_dur = 1.25
    rel = np.array([0.0, 0.137 * bar_dur])  # 0.137 落在 32 分（0.125/0.15625）之间
    d, _, _, ok = quantize.choose_division(rel, bar_dur, tol=0.010)
    assert d == 32 and not ok


def test_tolerance_rule():
    # 25ms 下限 vs 1/64 小节，取大者
    assert quantize.tolerance_for_bar(1.25) == pytest.approx(0.025)      # 1.25/64 = 19.5ms
    assert quantize.tolerance_for_bar(2.0) == pytest.approx(2.0 / 64)    # = 31.25ms


def test_make_pattern_strings():
    p = quantize.make_pattern(16, np.array([0, 4, 8, 10]),
                              np.array([True, False, False, False]))
    assert p == "X...x...x.x....."
    assert len(p) == 16


def test_quantize_track_on_ideal_times():
    g = Grid(bpm=192.0, first=1.875, beats_per_bar=4, duration=20.0)
    # 第 2–5 小节：真 16 分（第 7 格不是 8 分位置）
    slots = [0, 4, 7, 12]
    times = grid_times(g, 16, slots, bars=[2, 3, 4, 5])
    # 192BPM 下 1/48 小节 ≈ 26ms，与 25ms 容差过于接近 → 本用例排除三连分音
    res, stats = quantize.quantize_track(times, None, "drums", g,
                                         divisions=(4, 8, 16, 32))
    for bar in (2, 3, 4, 5):
        assert res[bar].division == 16
        assert res[bar].pattern == "x...x..x....x..."
        assert res[bar].resolved
    assert res[1].n_onsets == 0 and res[1].pattern == "." * 4
    assert stats["n_onsets"] == 16
    assert stats["mean_abs_err_ms"] < 1e-6
    assert stats["unresolved_bars"] == 0
    assert stats["division_histogram"] == {"16": 4}


def test_quantize_track_end_of_bar_snaps_forward():
    """落在小节末尾容差内的 onset 应归到下一小节第 0 格，而不是本小节第 d 格。"""
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=12.0)
    t = g.bar_start(3) - 0.010  # 比第 3 小节起点早 10ms
    res, _ = quantize.quantize_track(np.array([t]), None, "drums", g)
    assert res[3].n_onsets == 1
    assert res[3].pattern.startswith("x")
    assert res[2].n_onsets == 0


def test_onset_detection_then_quantization_recovers_division():
    """端到端：合成 16 分鼓点 → librosa 检测 → 量化，应还原出 16 分与正确网格串。"""
    g = Grid(bpm=150.0, first=0.5, beats_per_bar=4, duration=14.0)
    slots = [0, 4, 7, 12]
    bars = list(range(2, 8))
    times = grid_times(g, 16, slots, bars=bars)
    y = synth_clicks(times, duration=14.0)
    tr = onsets.detect_onsets(y, "drums")
    assert tr.count >= len(times) * 0.9
    res, stats = quantize.quantize_track(tr.times, tr.strong_mask(), "drums", g)
    expected = "x...x..x....x..."
    ok_bars = [b for b in bars
               if res[b].division == 16 and res[b].pattern.replace("X", "x") == expected]
    assert len(ok_bars) >= len(bars) - 1, {b: (res[b].division, res[b].pattern) for b in bars}
    assert stats["p95_abs_err_ms"] <= 25.0


def test_quantize_error_stats_reflect_jitter():
    """给理想时间加 ±12ms 抖动，量化仍应成功，但误差统计要如实反映。"""
    rng = np.random.default_rng(7)
    g = Grid(bpm=192.0, first=1.875, beats_per_bar=4, duration=20.0)
    times = grid_times(g, 8, [0, 2, 3, 6], bars=[2, 3, 4, 5])  # 第 3 格是纯 8 分位置
    jitter = rng.uniform(-0.012, 0.012, size=times.shape)
    res, stats = quantize.quantize_track(times + jitter, None, "drums", g,
                                         divisions=(4, 8, 16, 32))
    assert all(res[b].division == 8 for b in (2, 3, 4, 5))
    assert 1.0 < stats["mean_abs_err_ms"] <= 13.0
    assert stats["unresolved_bars"] == 0


# ---------------- 3. offset 校验 ----------------


def _offset_env(times, duration):
    y = synth_clicks(times, duration=duration)
    env = onsets.onset_envelope(y)
    return env, onsets.frame_times(len(env))


@pytest.mark.parametrize("true_first,user_error", [(0.5, 0.0), (0.5, 0.040), (0.5, -0.055)])
def test_check_offset_recovers_known_offset(true_first, user_error):
    """已知 first 的合成曲：无论用户值偏多少，搜索都应找回真实 first。"""
    bpm, duration = 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    # 每拍一个 click（真实网格）
    times = g.all_beat_times()
    env, ft = _offset_env(times, duration)
    user_first = true_first + user_error
    res = check_offset(env, ft, bpm=bpm, first=user_first, duration=duration)
    assert res["available"]
    assert abs(res["best_first"] - true_first) < 0.020, res
    # delta 应该指向"用户值需要挪多少"
    assert res["delta_sec"] == pytest.approx(-user_error, abs=0.020)
    assert res["confidence_z"] > 1.0


def test_check_offset_reports_only_does_not_override():
    bpm, duration = 150.0, 20.0
    g = Grid(bpm=bpm, first=0.5, beats_per_bar=4, duration=duration)
    env, ft = _offset_env(g.all_beat_times(), duration)
    res = check_offset(env, ft, bpm=bpm, first=0.56, duration=duration)
    assert res["user_first"] == 0.56
    # 网格仍然按用户值构造
    g_user = Grid(bpm=bpm, first=0.56, beats_per_bar=4, duration=duration)
    assert g_user.bar_start(1) == pytest.approx(0.56)


def test_check_offset_empty_envelope():
    res = check_offset(np.zeros(0), np.zeros(0), bpm=150.0, first=0.5)
    assert res["available"] is False


@pytest.mark.parametrize("user_error", [0.0, 0.035, -0.045])
def test_check_offset_onset_fit_path(user_error):
    """onset-fit 路（用 backtrack 后的 onset 时间）应比包络路更准地找回真实 first。"""
    true_first, bpm, duration = 0.5, 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    y = synth_clicks(g.all_beat_times(), duration=duration)
    env = onsets.onset_envelope(y)
    ft = onsets.frame_times(len(env))
    tr = onsets.detect_onsets(y, "drums")
    res = check_offset(env, ft, bpm=bpm, first=true_first + user_error,
                       duration=duration, onset_times=tr.times,
                       onset_weights=tr.strengths)
    assert res["method"] == "onset-fit"
    assert abs(res["best_first"] - true_first) < 0.015, res
    # 包络路作为参考同时给出，且其系统偏差为正（峰值偏晚）
    assert "envelope_best_first" in res


def test_check_offset_prefers_peak_nearest_user_value():
    """整拍平移歧义：同级峰里要选离用户值最近的，而不是全局最大。"""
    true_first, bpm, duration = 0.5, 150.0, 24.0
    g = Grid(bpm=bpm, first=true_first, beats_per_bar=4, duration=duration)
    y = synth_clicks(g.all_beat_times(), duration=duration)
    tr = onsets.detect_onsets(y, "drums")
    res = check_offset(None, None, bpm=bpm, first=true_first, duration=duration,
                       onset_times=tr.times, onset_weights=tr.strengths)
    # 搜索窗 ±1 拍内必然出现 ≥2 个等价峰（真值、真值±1拍）
    assert res["n_tied_peaks"] >= 2
    assert abs(res["best_first"] - true_first) < 0.015


# ---------------- 4. 逐小节聚合 ----------------


def test_onsets_per_bar_counts():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4, duration=12.0)
    times = grid_times(g, 4, [0, 1, 2, 3], bars=[1, 2])
    tr = onsets.OnsetTrack("drums", times, np.ones(len(times)))
    counts = onsets.onsets_per_bar(tr, g)
    assert counts[0] == 4 and counts[1] == 4 and counts[2] == 0


def test_vocal_activity_on_synthetic_tone():
    """前半段有正弦（"人声"），后半段静音 → VAD 应只在前半段激活。"""
    sr, dur = SR, 8.0
    t = np.arange(int(sr * dur)) / sr
    y = np.zeros_like(t, dtype=np.float32)
    half = len(t) // 2
    y[:half] = (0.5 * np.sin(2 * np.pi * 220 * t[:half])).astype(np.float32)
    vad = onsets.vocal_activity(y)
    act, times = vad["active"], vad["times"]
    assert act[times < 3.5].mean() > 0.9
    assert act[times > 4.5].mean() < 0.05
