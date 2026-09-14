"""`tools/calibration/lyrics_phrases.py` + `tools/audio_analysis/phrases.py` 的单元测试。

规矩同 `tests/test_calibration.py`（AGENT.md 准则 5）：**全部用合成数据**——
自造的 LRC 时间戳串、自造的 onset 序列、自造的 note 事件，
不碰真实音频、不碰官方谱、**不出现任何歌词文本**（版权约束见模块 docstring）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audio_analysis import phrases as ph          # noqa: E402
from tools.audio_analysis.grid import BpmChange, Grid    # noqa: E402
from tools.calibration import lyrics_phrases as lp       # noqa: E402


# ---------------------------------------------------------------------------
# 曲名 / 艺术家归一化
# ---------------------------------------------------------------------------


def test_strip_st_prefix():
    assert lp.strip_st("[ST] Foo Bar") == "Foo Bar"
    assert lp.strip_st("Foo Bar") == "Foo Bar"


def test_fold_ignores_punctuation_and_width():
    assert lp._fold("Ｆｏｏ  Bar!") == lp._fold("foo-bar")
    # ✪ 是 O 的装饰写法
    assert lp._fold("D✪N'T ST✪P") == lp._fold("dont stop")


def test_title_variants_covers_bracket_and_wrapper():
    v = lp.title_variants("[ST] -OutsideR:RequieM- (Short Ver.)")
    assert any("OutsideR" in x and "(" not in x for x in v)
    assert any(not x.startswith("-") for x in v)


def test_artist_tokens_drops_cv_and_album():
    toks = lp.artist_tokens("somebody(CV:someone)「album name」 feat. other")
    assert lp._fold("somebody") in toks
    assert all("album" not in t for t in toks)


def test_score_candidate_prefers_exact_title_and_artist():
    s_exact = lp._score_candidate("Foo", "someone", "Foo", [lp._fold("someone")])
    s_title = lp._score_candidate("Foo", "nobody", "Foo", [lp._fold("someone")])
    s_other = lp._score_candidate("Completely Else", "nobody", "Foo",
                                  [lp._fold("someone")])
    assert s_exact > s_title > s_other


# ---------------------------------------------------------------------------
# LRC 解析：只保留时间戳与计数
# ---------------------------------------------------------------------------

SYNTH_LRC = "\n".join([
    "[ti:synthetic]",
    "[00:01.00]あいうえお",
    "[00:03.50]かきくけこ",
    "[00:06.00]",                 # 空行 = 上一句的终点
    "[00:10.00]abcdef",
    "[00:12.00]漢字テスト",
])


def test_parse_lrc_keeps_only_timestamps_and_counts():
    lines = lp.parse_lrc(SYNTH_LRC)
    assert [l.idx for l in lines] == [0, 1, 2, 3]
    assert lines[0].t == pytest.approx(1.0)
    assert lines[0].t_end == pytest.approx(3.5)
    # 空行把第 2 行的终点钉在 6.0（而不是延到 10.0）
    assert lines[1].t_end == pytest.approx(6.0)
    # 只留计数，没有任何文本字段
    for l in lines:
        assert not any(isinstance(v, str) for v in vars(l).values())
    assert lines[0].n_kana == 5 and lines[0].n_chars == 5
    assert lines[2].n_latin == 6
    assert lines[3].n_kanji == 2


def test_parse_lrc_multi_timestamp_line_expands():
    lines = lp.parse_lrc("[00:01.00][00:20.00]あい")
    assert len(lines) == 2
    assert [round(l.t) for l in lines] == [1, 20]


def test_parse_lrc_metadata_tag_ignored():
    assert lp.parse_lrc("[ar:someone]\n[al:something]") == []


def test_syllable_estimate_ignores_small_kana():
    big = lp._classify("きや")["syl_est"]
    small = lp._classify("きゃ")["syl_est"]
    assert big == pytest.approx(2.0)
    assert small == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 对齐：合成 onset + 已知偏移，必须还原出那个偏移
# ---------------------------------------------------------------------------


def _synthetic_lines(starts, dur=2.0):
    return [lp.PhraseLine(idx=i, t=float(t), t_end=float(t) + dur, n_chars=10,
                          n_kana=10, n_kanji=0, n_latin=0, n_digit=0, syl_est=10.0)
            for i, t in enumerate(starts)]


def test_align_recovers_known_offset():
    rng = np.random.default_rng(0)
    # ⚠️ 行首必须**不等距**：等距行首在"平移一个行距"处会得到同样满分的退化解
    # （这正是真实数据上对齐失败的机理之一，见 vocal-phrase-alignment.md §4）
    lyric_starts = np.array([20.0, 23.5, 27.0, 33.0, 36.5, 41.0, 47.5, 51.0,
                             56.0, 62.5, 66.0, 71.0, 77.5, 81.0, 86.0, 92.5])
    delta = 17.3                                        # 音频 t ↔ 原曲 t + delta
    audio_dur = 120.0
    # 音频侧：行首处一定有 onset，再撒一堆背景 onset
    hits = lyric_starts - delta
    hits = hits[(hits >= 0) & (hits < audio_dur)]
    bg = rng.uniform(0, audio_dur, 120)
    onsets = np.sort(np.concatenate([hits, bg]))
    al = lp.align_lines(_synthetic_lines(lyric_starts), onsets, audio_dur)
    assert al.ok
    # 命中栅格按 ±HIT_TOL 膨胀，所以还原精度以容差为限
    assert al.delta == pytest.approx(delta, abs=lp.HIT_TOL + 0.02)
    assert al.hit_rate > 0.9


def test_align_rejects_unrelated_onsets():
    rng = np.random.default_rng(1)
    lyric_starts = np.arange(10.0, 90.0, 4.0)
    onsets = np.sort(rng.uniform(0, 120.0, 150))        # 与歌词毫无关系
    al = lp.align_lines(_synthetic_lines(lyric_starts), onsets, 120.0)
    assert not al.ok
    assert al.reason in ("low_z", "low_hit_rate", "too_few_lines")


def test_align_detects_splice():
    """两段各自偏移不同（游戏剪辑版剪掉了中间一段），分块对齐应当认出来。"""
    rng = np.random.default_rng(2)
    blk_a = 10.0 + np.cumsum([0, 2.5, 2.3, 2.7, 2.4, 2.6, 2.5, 2.3, 2.7, 2.4])
    blk_b = 80.0 + np.cumsum([0, 2.2, 2.8, 2.4, 2.6, 2.3, 2.7, 2.5, 2.2, 2.8])
    d_a, d_b = 5.0, 45.0
    audio_dur = 80.0
    hits = np.concatenate([blk_a - d_a, blk_b - d_b])
    onsets = np.sort(np.concatenate([hits, rng.uniform(0, audio_dur, 40)]))
    # 行长 2.0 s → 块内间隙 ≤ 0.7 s（不切块），块间间隙 ≈ 45 s（切块）
    al = lp.align_lines(_synthetic_lines(np.concatenate([blk_a, blk_b]), dur=2.0),
                        onsets, audio_dur)
    assert al.ok
    assert al.spliced
    deltas = sorted({round(b.delta, 1) for b in al.blocks if b.kept})
    assert len(deltas) == 2
    assert deltas[0] == pytest.approx(d_a, abs=lp.HIT_TOL + 0.02)
    assert deltas[1] == pytest.approx(d_b, abs=lp.HIT_TOL + 0.02)


def test_split_blocks_cuts_on_long_gap():
    lines = _synthetic_lines([0.0, 1.2, 2.4, 30.0, 31.2], dur=1.0)
    blocks = lp.split_blocks(lines, gap=1.5)
    assert [(b.i0, b.i1) for b in blocks] == [(0, 3), (3, 5)]


def test_lyric_to_audio_uses_block_offset():
    al = lp.AlignResult(ok=True, delta=1.0, blocks=[
        lp.Block(i0=0, i1=2, delta=1.0, kept=True),
        lp.Block(i0=2, i1=4, delta=9.0, kept=True),
        lp.Block(i0=4, i1=6, delta=0.0, kept=False),
    ])
    assert lp.lyric_to_audio(10.0, al, line_idx=0) == pytest.approx(9.0)
    assert lp.lyric_to_audio(10.0, al, line_idx=3) == pytest.approx(1.0)
    assert lp.lyric_to_audio(10.0, al, line_idx=5) is None


# ---------------------------------------------------------------------------
# 区间 lift + 随机平移对照
# ---------------------------------------------------------------------------


def test_interval_lift_flat_when_notes_uniform():
    rng = np.random.default_rng(11)
    times = np.sort(rng.uniform(0.0, 100.0, 4000))      # 与区间无关的随机 note
    iv = [(t, t + 0.4) for t in np.arange(0.0, 100.0, 5.0)]
    r = lp.interval_lift(times, iv, 0.0, 100.0, n_null=50, seed=0)
    assert r["lift"] == pytest.approx(1.0, abs=0.25)
    assert r["p"] > 0.05


def test_interval_lift_detects_concentration():
    # note 全部堆在区间里
    iv = [(t, t + 0.4) for t in np.arange(0.0, 100.0, 5.0)]
    times = np.array([a + 0.2 for a, _ in iv], dtype=float)
    r = lp.interval_lift(times, iv, 0.0, 100.0, n_null=100, seed=0)
    assert r["lift"] > 5.0
    assert r["p"] < 0.05
    assert r["null_mean"] == pytest.approx(1.0, abs=0.4)


def test_merge_intervals_unions_overlaps():
    out = lp._merge_intervals([(0, 2), (1, 3), (5, 6)])
    assert out == [(0.0, 3.0), (5.0, 6.0)]


def test_wrap_preserves_total_length():
    iv = [(1.0, 3.0), (8.0, 9.0)]
    out = lp._wrap(iv, 4.5, 0.0, 10.0)
    assert sum(b - a for a, b in out) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 乐句表 / 逐小节特征
# ---------------------------------------------------------------------------


def _grid(bpm=120.0, dur=60.0):
    return Grid(bpm=bpm, first=0.0, beats_per_bar=4, duration=dur)


def test_build_phrases_maps_to_bars():
    g = _grid()                                  # 120 BPM → 每小节 2 s
    lines = _synthetic_lines([10.0, 14.0], dur=2.0)
    al = lp.AlignResult(ok=True, delta=6.0,
                        blocks=[lp.Block(i0=0, i1=2, delta=6.0, kept=True)])
    out = lp.build_phrases(lines, al, g, audio_dur=60.0)
    assert [round(p.t_start, 2) for p in out] == [4.0, 8.0]
    assert [round(p.bar_start, 3) for p in out] == [3.0, 5.0]
    assert out[0].dur_beats == pytest.approx(4.0)


def test_syllables_per_bar_splits_across_bars():
    g = _grid()                                  # 小节 = 2 s
    p = lp.Phrase(line=0, t_start=1.0, t_end=3.0, bar_start=1.5, bar_end=2.5,
                  beat_start=2.0, beat_end=6.0, dur_beats=4.0, n_chars=10,
                  syl_est=10.0)
    syl = lp.syllables_per_bar([p], g, g.n_bars)
    assert syl[0] == pytest.approx(5.0)
    assert syl[1] == pytest.approx(5.0)
    assert syl[2:].sum() == pytest.approx(0.0)


def test_head_and_pre_windows_scale_with_bpm():
    g = _grid(bpm=120.0)
    p = lp.Phrase(line=0, t_start=10.0, t_end=12.0, bar_start=6.0, bar_end=7.0,
                  beat_start=20.0, beat_end=24.0, dur_beats=4.0, n_chars=4,
                  syl_est=4.0)
    (a, b), = lp.head_windows([p], g, 0.5)
    assert (b - a) == pytest.approx(0.5)          # 半拍 = 0.25 s，前后共 0.5 s
    (c, d), = lp.pre_phrase_windows([p], g)
    assert c == pytest.approx(10.0 - 0.75) and d == pytest.approx(10.0 - 0.25)


def test_gap_intervals_only_real_gaps():
    g = _grid()
    a = lp.Phrase(0, 0.0, 2.0, 1.0, 2.0, 0.0, 4.0, 4.0, 4, 4.0)
    b = lp.Phrase(1, 2.0, 4.0, 2.0, 3.0, 4.0, 8.0, 4.0, 4, 4.0)   # 无间隙
    c = lp.Phrase(2, 10.0, 12.0, 6.0, 7.0, 20.0, 24.0, 4.0, 4, 4.0)
    assert lp.gap_intervals([a, b], g) == []
    assert lp.gap_intervals([a, b, c], g) == [(4.0, 10.0)]


def test_abs_beat_handles_bpm_change():
    g = Grid(bpm=120.0, first=0.0, beats_per_bar=4,
             bpm_changes=(BpmChange(bar=3, bpm=240.0),), duration=30.0)
    # 前 2 小节各 2 s；第 3 小节起每小节 1 s
    assert lp._abs_beat(g, 4.0) == pytest.approx(8.0)
    assert lp._abs_beat(g, 4.5) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# T3 / T4 的纯计算部分
# ---------------------------------------------------------------------------


def test_ols_r2_perfect_and_null():
    y = np.arange(20, dtype=float)
    assert lp._ols_r2(y, y[:, None]) == pytest.approx(1.0, abs=1e-9)
    rng = np.random.default_rng(3)
    assert lp._ols_r2(y, rng.normal(size=(20, 1))) < 0.5


def test_phrase_starts_per_bar_counts():
    g = _grid()
    ps = [lp.Phrase(i, t, t + 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 4, 4.0)
          for i, t in enumerate([0.1, 0.9, 5.0])]
    out = lp.phrase_starts_per_bar(ps, g, g.n_bars)
    assert out[0] == 2.0 and out[2] == 1.0


# ---------------------------------------------------------------------------
# 纯音频乐句探测器
# ---------------------------------------------------------------------------


def test_onset_gap_starts_picks_gaps():
    g = _grid(bpm=120.0)                 # 1 拍 = 0.5 s；1.5 拍 = 0.75 s
    onsets = np.array([1.0, 1.2, 1.4, 5.0, 5.2, 9.0])
    st = ph.onset_gap_starts(onsets, g, gap_beats=1.5)
    assert list(np.round(st, 2)) == [1.0, 5.0, 9.0]


def test_vad_rise_starts():
    times = np.arange(0, 10, 0.1)
    active = (times > 2) & (times < 4) | (times > 6) & (times < 7)
    st = ph.vad_rise_starts(active, times)
    assert len(st) == 2
    assert st[0] == pytest.approx(2.1, abs=0.15)
    assert st[1] == pytest.approx(6.1, abs=0.15)


def test_vad_gaps_min_length():
    g = _grid(bpm=120.0)
    times = np.arange(0, 12, 0.1)
    active = ~(((times >= 3) & (times < 3.3)) | ((times >= 6) & (times < 9)))
    gaps = ph.vad_gaps(active, times, g, min_beats=1.0)
    assert len(gaps) == 1
    assert gaps[0][0] == pytest.approx(6.0, abs=0.15)


def test_boundary_prf_exact_and_shifted():
    truth = np.array([1.0, 5.0, 9.0])
    assert ph.boundary_prf(truth, truth, 0.2)["f1"] == pytest.approx(1.0)
    r = ph.boundary_prf(truth + 0.5, truth, 0.2)
    assert r["f1"] == pytest.approx(0.0)
    r2 = ph.boundary_prf(np.array([1.1, 5.05]), truth, 0.2)
    assert r2["tp"] == 2 and r2["recall"] == pytest.approx(2 / 3)


def test_boundary_prf_one_to_one():
    """同一个真值不能被两个预测重复认领。"""
    r = ph.boundary_prf(np.array([1.0, 1.05]), np.array([1.0]), 0.2)
    assert r["tp"] == 1 and r["precision"] == pytest.approx(0.5)


def test_detect_phrases_merges_close_starts():
    g = _grid(bpm=120.0)                 # 2 拍 = 1.0 s
    onsets = np.array([1.0, 1.3, 4.0, 4.2, 10.0])
    det = ph.detect_phrases(onsets, None, g, gap_beats=1.5, min_sep_beats=2.0,
                            source="onset_gap")
    assert list(np.round(det.starts, 2)) == [1.0, 4.0, 10.0]


def test_phrase_features_per_bar():
    g = _grid(bpm=120.0, dur=20.0)       # 小节 2 s
    det = ph.PhraseDetection(starts=np.array([0.5, 2.5, 2.9]),
                             gaps=[(4.0, 5.0)])
    f = ph.phrase_features_per_bar(det, g)
    assert f["phrase_start"][0] == 1.0
    assert f["phrase_start"][1] == 2.0
    assert f["in_gap"][2] == pytest.approx(0.5)
    assert f["in_gap"][0] == 0.0
