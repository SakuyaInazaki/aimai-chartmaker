"""tools/calibration 的单元测试 —— **全部用合成数据**，不碰真实音频/官方谱。

与 `tests/test_audio_analysis.py` 同规矩（AGENT.md 准则 5）：仓库里不放版权素材，
所以这里只测纯函数部分（相关系数、匹配、权重拟合、结尾判据），
真实数据上的结论写在 `docs/research/audio-chart-calibration.md`。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.calibration import chartpair as cp          # noqa: E402
from tools.calibration import ending as ending_mod      # noqa: E402
from tools.calibration import stemhit                   # noqa: E402
from tools.calibration import weights as weights_mod    # noqa: E402


# ---------------------------------------------------------------------------
# chartpair：相关系数
# ---------------------------------------------------------------------------


def test_pearson_perfect_and_anti():
    x = np.arange(10, dtype=float)
    assert cp.pearson(x, 2 * x + 3) == pytest.approx(1.0, abs=1e-9)
    assert cp.pearson(x, -x) == pytest.approx(-1.0, abs=1e-9)


def test_spearman_is_rank_based():
    x = np.arange(10, dtype=float)
    # 单调但非线性 → Pearson < 1，Spearman = 1
    y = np.exp(x)
    assert cp.spearman(x, y) == pytest.approx(1.0, abs=1e-9)
    assert cp.pearson(x, y) < 0.95


def test_spearman_handles_ties():
    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([5.0, 5.0, 6.0, 7.0])
    assert cp.spearman(a, b) == pytest.approx(1.0, abs=1e-9)


def test_correlation_degenerate_returns_nan():
    assert np.isnan(cp.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert np.isnan(cp.pearson([1.0], [2.0]))


def test_fisher_mean_between_min_and_max():
    v = cp.fisher_mean([0.2, 0.6, 0.8])
    assert 0.2 < v < 0.8
    # 与算术平均接近但不相等（Fisher z 更重视高相关）
    assert v == pytest.approx(0.5816, abs=1e-3)


# ---------------------------------------------------------------------------
# chartpair：小节对齐与加权密度
# ---------------------------------------------------------------------------


def test_map_measures_to_bars_offset_by_one():
    # 谱面小节 0 起、音频小节 1 起，&first=0、BPM 150 → 每小节 1.6 s
    ms = [0.0, 1.6, 3.2, 4.8]
    bs = [0.0, 1.6, 3.2, 4.8, 6.4]
    m = cp.map_measures_to_bars(ms, bs)
    assert m == {0: 1, 1: 2, 2: 3, 3: 4}


def test_map_measures_to_bars_drops_out_of_range():
    m = cp.map_measures_to_bars([0.0, 100.0], [0.0, 1.6], tol_sec=0.05)
    assert m == {0: 1}


def test_measure_alignment_error():
    assert cp.measure_alignment_error([0.0, 1.61], [0.0, 1.6]) == pytest.approx(0.01)


@dataclass
class _Note:
    measure: int
    kind: str
    is_break: bool = False
    is_each: bool = False


def test_weighted_density_orders_kinds():
    notes = [_Note(0, "tap"), _Note(1, "hold"), _Note(2, "slide_track")]
    w = cp.weighted_density(notes, 3)
    assert w[0] < w[1] < w[2]          # 知识 003：tap < hold < slide


def test_weighted_density_each_and_break_bonus():
    plain = cp.weighted_density([_Note(0, "tap")], 1)[0]
    each = cp.weighted_density([_Note(0, "tap", is_each=True)], 1)[0]
    brk = cp.weighted_density([_Note(0, "tap", is_break=True)], 1)[0]
    assert each == pytest.approx(plain + cp.EACH_BONUS)
    assert brk == pytest.approx(plain + cp.BREAK_BONUS)


def test_weighted_density_ignores_out_of_range_measures():
    assert cp.weighted_density([_Note(9, "tap")], 3).sum() == 0.0


# ---------------------------------------------------------------------------
# chartpair：峰值与边界
# ---------------------------------------------------------------------------


def test_smoothed_peak_index_finds_plateau_center():
    c = np.zeros(20)
    c[8:12] = 10.0
    assert cp.smoothed_peak_index(c, window=4) == pytest.approx(9, abs=1)


def test_boundary_hits_tolerance():
    hits, n_p, n_t = cp.boundary_hits([10, 20, 31], [10, 21, 40], tol=1)
    assert (hits, n_p, n_t) == (2, 3, 3)


def test_boundary_hits_is_one_to_one():
    # 两个 truth 边界挤在一起时，一个 pred 不能同时认领两个
    hits, _, _ = cp.boundary_hits([10], [10, 11], tol=1)
    assert hits == 1


def test_segment_means_clips_and_handles_empty():
    v = [1.0, 2.0, 3.0, 4.0]
    out = cp.segment_means(v, [(0, 1), (2, 99)])
    assert out[0] == pytest.approx(1.5)
    assert out[1] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# stemhit
# ---------------------------------------------------------------------------


def test_unique_times_merges_each_group():
    t = np.array([1.0, 1.0, 1.00005, 2.0])
    assert stemhit.unique_times(t, merge_sec=1e-3).tolist() == [1.0, 2.0]


def test_nearest_gap_empty_reference_is_inf():
    g = stemhit.nearest_gap([1.0, 2.0], [])
    assert np.all(np.isinf(g))


def test_covered_mask_respects_tolerance():
    ev = np.array([1.000, 1.050])
    on = np.array([1.020])
    m = stemhit.covered_mask(ev, on, tol=0.030)
    assert m.tolist() == [True, False]


def test_hit_stat_recall_precision():
    ev = np.array([0.0, 1.0, 2.0, 3.0])
    on = np.array([0.01, 1.01, 9.0])
    st = stemhit.hit_stat(ev, on, tol=0.030, span_sec=10.0)
    assert st.n_hit == 2 and st.recall == pytest.approx(0.5)
    assert st.n_used == 2 and st.precision == pytest.approx(2 / 3)


def test_hit_stat_chance_and_lift():
    """onset 越密，随机基线越高 → lift 才是"真跟着这条轨"的判据。"""
    ev = np.arange(0, 10, 1.0)
    dense = np.arange(0, 10, 0.05)          # 200 个 onset / 10 s
    st = stemhit.hit_stat(ev, dense, tol=0.030, span_sec=10.0)
    assert st.recall == pytest.approx(1.0)
    assert st.chance_recall > 0.6           # 这么密，随机撒点也几乎全中
    assert st.lift < 1.7


def test_best_shift_recovers_known_offset():
    ev = np.arange(0, 20, 0.5)
    on = ev + 0.020
    r = stemhit.best_shift(ev, on, np.arange(-0.045, 0.0451, 0.0025), tol=0.010)
    assert r["best_shift_sec"] == pytest.approx(0.020, abs=0.0026)
    assert r["best_rate"] > r["rate_at_zero"]


def test_explain_breakdown_partition():
    ev = np.array([0.0, 1.0, 2.0])
    br = stemhit.explain_breakdown(
        ev, {"drums": np.array([0.0]), "vocals": np.array([1.0]),
             "bass": np.zeros(0), "other": np.zeros(0)}, tol=0.03)
    assert br["drums_only"] == pytest.approx(1 / 3)
    assert br["vocals_only"] == pytest.approx(1 / 3)
    assert br["none"] == pytest.approx(1 / 3)
    assert br["any"] + br["none"] == pytest.approx(1.0)


def test_rank_stems_by_lift_can_differ_from_recall():
    ev = np.array([0.0, 1.0, 2.0, 3.0])
    stems = {"drums": np.arange(0, 4, 0.02),        # 极密，recall 高但 lift 低
             "vocals": np.array([0.0, 1.0, 2.0, 3.0])}
    assert stemhit.rank_stems(ev, stems, tol=0.03, span_sec=4.0)[0][0] == "drums"
    assert stemhit.rank_stems(ev, stems, tol=0.03, span_sec=4.0,
                              by="lift")[0][0] == "vocals"


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def test_fit_nnls_is_nonnegative_and_recovers_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([2.0, 0.0, 1.0])
    w = weights_mod.fit_nnls(X, y)
    assert np.all(w >= -1e-9)
    assert w[0] > w[2] > w[1]


def test_fit_ridge_nonneg_clips_negative_direction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = X @ np.array([1.0, -1.0])
    w = weights_mod.fit_ridge(X, y, alpha=1.0, nonneg=True)
    assert w[1] == pytest.approx(0.0, abs=1e-6)


def test_normalize_weights_sums_to_one():
    w = weights_mod.normalize_weights([1.0, 3.0])
    assert w.sum() == pytest.approx(1.0)
    assert weights_mod.normalize_weights([0.0, 0.0]).tolist() == [0.5, 0.5]


def test_fuse_matches_intensity_formula():
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = weights_mod.fuse(X, np.array([3.0, 1.0]))
    assert out.tolist() == pytest.approx([0.75, 0.25])


def test_calibrate_loso_recovers_dominant_feature():
    rng = np.random.default_rng(7)
    ds = {}
    for i in range(4):
        X = rng.normal(size=(60, 5))
        y = 3.0 * X[:, 1] + 0.2 * rng.normal(size=60)   # 只有 "onset" 有信号
        ds[f"song{i}"] = (X, y)
    res = weights_mod.calibrate(ds, alpha=0.5)
    assert len(res.folds) == 4
    assert max(res.weights_all, key=res.weights_all.get) == "onset"
    assert res.rho_fit_mean > 0.8


def test_calibrate_empty_returns_blank_result():
    res = weights_mod.calibrate({})
    assert res.folds == [] and res.weights_all == {}


def test_zscore_constant_column_is_zero():
    assert weights_mod.zscore([2.0, 2.0, 2.0]).tolist() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# ending
# ---------------------------------------------------------------------------


def _curve(head, tail):
    return np.array(list(head) + list(tail), dtype=float)


def test_classify_tail_kill():
    c = _curve([8.0] * 30, [12.0] * 8)
    s = ending_mod.classify_ending(c, tail_bars=8)
    assert s.label == "tail_kill"
    assert s.tail_ratio > 1.0


def test_classify_fade_out():
    c = _curve([10.0] * 30, [7, 6, 5, 4, 3, 2, 1, 0])
    s = ending_mod.classify_ending(c, tail_bars=8)
    assert s.label == "fade_out"
    assert s.strictly_decreasing
    assert s.trend == pytest.approx(-1.0)


def test_classify_other_when_low_but_not_monotone():
    c = _curve([10.0] * 30, [2, 8, 1, 7, 2, 6, 1, 5])
    s = ending_mod.classify_ending(c, tail_bars=8)
    assert s.label == "other"


def test_classify_other_for_mild_taper():
    # 末段是全曲均值的 0.8：不到尾杀也不到渐弱
    c = _curve([10.0] * 30, [8.0] * 8)
    assert ending_mod.classify_ending(c, tail_bars=8).label == "other"


def test_classify_empty_curve():
    assert ending_mod.classify_ending([], tail_bars=8).label == "other"


def test_tail_bars_clipped_to_curve_length():
    s = ending_mod.classify_ending([5.0, 5.0, 5.0], tail_bars=8)
    assert s.tail_bars == 3


def test_last_high_plateau_mean_picks_last_run():
    c = np.array([10.0] * 5 + [1.0] * 10 + [20.0] * 5)
    assert ending_mod.last_high_plateau_mean(c, quantile=0.75, min_len=4) \
        == pytest.approx(20.0)


def test_last_high_plateau_falls_back_to_quantile():
    c = np.arange(20, dtype=float)   # 没有长度 ≥4 的高密平台？有，仍应返回有限值
    assert np.isfinite(ending_mod.last_high_plateau_mean(c))


def test_summarize_shares_sum_to_one():
    shapes = [ending_mod.classify_ending(_curve([8.0] * 20, [12.0] * 8), 8),
              ending_mod.classify_ending(_curve([10.0] * 20, [7, 6, 5, 4, 3, 2, 1, 0]), 8),
              ending_mod.classify_ending(_curve([10.0] * 20, [8.0] * 8), 8)]
    out = ending_mod.summarize(shapes)
    assert out["n"] == 3
    total = out["tail_kill_share"] + out["fade_out_share"] + out["other_share"]
    assert total == pytest.approx(1.0, abs=1e-3)


def test_fit_density_floor_recovers_synthetic_floor():
    rng = np.random.default_rng(3)
    pairs = []
    for _ in range(4):
        I = rng.uniform(0, 1, size=120)
        d = 0.4 + 0.6 * I            # 真值 floor = 0.4
        pairs.append((I, d))
    out = cp.fit_density_floor(pairs)
    assert out["floor"] == pytest.approx(0.4, abs=0.02)
    assert out["sse"] == pytest.approx(0.0, abs=1e-6)


def test_fit_density_floor_empty_input():
    out = cp.fit_density_floor([])
    assert np.isnan(out["floor"]) and out["curve"] == []


# ---------------------------------------------------------------------------
# strata.py（n=40 扩样时新增的分层统计与阈值扫描）
# ---------------------------------------------------------------------------

from tools.calibration import strata as strata_mod      # noqa: E402


def test_level_band_and_bpm_band_boundaries():
    assert strata_mod.level_band(13.0) == "13.0-13.2"
    assert strata_mod.level_band(13.2) == "13.0-13.2"
    assert strata_mod.level_band(13.3) == "13.3-13.7"
    assert strata_mod.level_band(13.9) == "13.8-14.2"
    assert strata_mod.level_band(14.5) == "14.3-14.5"
    assert strata_mod.bpm_band(128) == "<150"
    assert strata_mod.bpm_band(150) == "150-179"
    assert strata_mod.bpm_band(185) == "180-199"
    assert strata_mod.bpm_band(230) == ">=200"


def test_vocal_profile_classifies_by_share_vocals():
    hi = {"_features": {"voiced": [0.6, 0.7, 0.5]}, "n_slots": 100,
          "audio_profile": {"share_vocals_mean": 0.25},
          "global_stem": {"vocals": {"n_onsets": 200}, "drums": {"n_onsets": 400}}}
    lo = {"_features": {"voiced": [0.02, 0.0, 0.05]}, "n_slots": 100,
          "audio_profile": {"share_vocals_mean": 0.01},
          "global_stem": {"vocals": {"n_onsets": 10}, "drums": {"n_onsets": 500}}}
    a, b = strata_mod.vocal_profile(hi), strata_mod.vocal_profile(lo)
    assert a["kind"] == "vocal" and b["kind"] == "instrumental"
    assert a["vocals_over_drums"] == pytest.approx(0.5)
    assert a["vocal_ceiling"] == pytest.approx(2.0)


def test_vocal_profile_share_beats_broken_vad():
    """n=40 的真实故障形态：vocals 轨近乎静音，VAD 却报 voiced_ratio=1.0。"""
    broken = {"_features": {"voiced": [1.0, 1.0, 1.0]}, "n_slots": 100,
              "audio_profile": {"share_vocals_mean": 0.003},
              "global_stem": {"vocals": {"n_onsets": 30}, "drums": {"n_onsets": 500}}}
    p = strata_mod.vocal_profile(broken)
    assert p["kind"] == "instrumental"          # 能量占比判对
    assert p["kind_by_voiced"] == "vocal"       # VAD 判错（对照列）


def test_stratify_groups_and_aggregates():
    items = [{"g": "a", "v": 0.5}, {"g": "a", "v": 0.5}, {"g": "b", "v": 0.1},
             {"g": "b", "v": None}, {"g": None, "v": 0.9}]
    out = strata_mod.stratify(items, "g", "v")
    assert out["a"]["n"] == 2 and out["a"]["value"] == pytest.approx(0.5, abs=1e-6)
    assert out["b"]["n"] == 1
    assert set(out) == {"a", "b"}


def test_boxplot_groups_returns_sorted_raw_values():
    items = [{"g": "a", "v": 0.3}, {"g": "a", "v": 0.1}, {"g": "b", "v": 0.2}]
    out = strata_mod.boxplot_groups(items, "g", "v")
    assert out["a"] == [0.1, 0.3] and out["b"] == [0.2]


def test_sign_test_p_known_values():
    assert strata_mod.sign_test_p(5, 8) == pytest.approx(0.7266, abs=1e-3)
    assert strata_mod.sign_test_p(8, 8) == pytest.approx(0.0078, abs=1e-3)
    assert strata_mod.sign_test_p(30, 40) == pytest.approx(0.0022, abs=1e-3)
    assert np.isnan(strata_mod.sign_test_p(0, 0))


def test_weight_switch_verdict_requires_all_three_criteria():
    # 提升够大、胜出比够高、p 够小 → 换
    fit = [0.6] * 30 + [0.3] * 10
    init = [0.5] * 30 + [0.4] * 10
    v = strata_mod.weight_switch_verdict(fit, init)
    assert v["n_folds"] == 40 and v["n_wins"] == 30
    assert v["switch"] is True
    # 只差一点点 → 不换（gain 判据挂掉）
    v2 = strata_mod.weight_switch_verdict([0.51] * 40, [0.50] * 40)
    assert v2["switch"] is False
    assert v2["criteria"]["gain>=0.03"] is False


def test_sweep_threshold_finds_perfect_split():
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    labels = [False, False, False, True, True, True]
    out = strata_mod.sweep_threshold(scores, labels)
    assert out["best"]["youden"] == pytest.approx(1.0)
    assert 0.3 < out["best"]["threshold"] < 0.8
    assert out["auc"] == pytest.approx(1.0, abs=1e-6)


def test_sweep_threshold_degenerate_labels():
    out = strata_mod.sweep_threshold([1.0, 2.0], [True, True])
    assert out["best"] is None and out["auc"] is None


def test_sweep_two_thresholds_and_fixed_eval():
    a = [0.1, 0.9, 0.9, 0.2]
    b = [0.9, 0.9, 0.1, 0.1]
    y = [False, True, False, False]
    out = strata_mod.sweep_two_thresholds(a, b, y)
    assert out["best"]["tp"] == 1 and out["best"]["fp"] == 0
    fixed = strata_mod.evaluate_fixed_two(a, b, y, 0.45, 0.65)
    assert fixed["tp"] == 1 and fixed["fp"] == 0 and fixed["tn"] == 3


def test_agreement_table_counts_models_separately():
    rows = [
        {"plan_primary": "drum", "plan_v02": "vocal", "best_stem": "drums",
         "second_stem": "other", "accent_stems": ["hook"]},
        {"plan_primary": "drum", "plan_v02": "vocal", "best_stem": "vocals",
         "second_stem": "drums", "accent_stems": ["vocal"]},
    ]
    out = strata_mod.agreement_table(rows)
    assert out["n"] == 2
    assert out["skeleton_top1"] == pytest.approx(0.5)
    assert out["v02_rule_top1"] == pytest.approx(0.5)
    assert out["constant_drums"] == pytest.approx(0.5)
    assert out["accent_hits_second"] == pytest.approx(0.5)
    assert out["truth_dist"]["drums"] == 1


def test_plan_v02_chorus_rule_is_vocals_when_usable():
    n = 8
    bf = {"share_drums": np.full(n, 0.5), "share_bass": np.full(n, 0.2),
          "share_other": np.full(n, 0.2), "share_vocals": np.full(n, 0.1),
          "n_onset_drums": np.full(n, 10.0), "n_onset_bass": np.full(n, 10.0),
          "n_onset_other": np.full(n, 10.0), "n_onset_vocals": np.full(n, 10.0),
          "grid_fit_bar_drums": np.full(n, 0.9), "grid_fit_bar_bass": np.full(n, 0.9),
          "grid_fit_bar_other": np.full(n, 0.9), "grid_fit_bar_vocals": np.full(n, 0.9),
          "voiced_ratio": np.full(n, 0.6)}
    segs = [{"start_bar": 1, "end_bar": 8, "function": "chorus"}]
    assert strata_mod.plan_v02_primary(segs, bf) == ["vocal"]
    # 人声轨不可用（onset 太少）→ v0.2 的降级规则把它退回 drums
    bf2 = dict(bf, n_onset_vocals=np.zeros(n))
    assert strata_mod.plan_v02_primary(segs, bf2) == ["drum"]


def test_plan_v02_intro_uses_energy_share():
    n = 8
    bf = {"share_drums": np.full(n, 0.1), "share_bass": np.full(n, 0.1),
          "share_other": np.full(n, 0.7), "share_vocals": np.full(n, 0.1),
          "voiced_ratio": np.zeros(n)}
    for st in ("drums", "bass", "other", "vocals"):
        bf[f"n_onset_{st}"] = np.full(n, 10.0)
        bf[f"grid_fit_bar_{st}"] = np.full(n, 0.9)
    segs = [{"start_bar": 1, "end_bar": 8, "function": "intro"}]
    assert strata_mod.plan_v02_primary(segs, bf) == ["hook"]
