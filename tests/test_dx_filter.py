#!/usr/bin/env python3
"""``tools/chart_analysis/dx_filter.py`` 的单元测试。

**测试素材全部为自写的小段 simai 片段**，不复制任何官方谱 / 参考谱原文。
覆盖：touch 丢弃、EX 摘除、Break HOLD / 滑条绝赞降级、拼接星星丢轨、
星星头两种口径、时间轴不被改动、`each` 重算、`split_inote` 切正文。

运行：``.venv/bin/python -m pytest tests/test_dx_filter.py -q``
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis.dx_filter import (  # noqa: E402
    ST_SLIDE_SHAPES,
    filter_to_st,
    split_inote,
)
from chart_analysis.simai_parser import parse_chart  # noqa: E402

APPROX = dict(rel=0, abs=1e-9)


def _st(src: str, **kw):
    return filter_to_st(parse_chart(src), **kw)


# ---------------------------------------------------------------------------
# touch
# ---------------------------------------------------------------------------


def test_touch与touchhold整个丢掉():
    fr = _st("(120){4}1,C,2,E1h[4:1],E")
    kinds = [n.kind for n in fr.result.notes]
    assert kinds == ["tap", "tap"]
    assert fr.result.counts["touch"] == 0
    assert [r.reason for r in fr.dropped] == ["touch", "touch"]
    assert [r.bar for r in fr.dropped] == [1, 1]


def test_touch丢掉后不再算each():
    """`1/C` 原本是 each，touch 摘掉之后只剩一个 tap。"""
    r = parse_chart("(120){4}1/C,2,E")
    assert r.notes[0].is_each
    fr = filter_to_st(r)
    assert len(fr.result.notes) == 2
    assert all(not n.is_each for n in fr.result.notes)
    assert fr.result.each_groups == 0


# ---------------------------------------------------------------------------
# 修饰符降级
# ---------------------------------------------------------------------------


def test_EX修饰符被摘掉但note保留():
    fr = _st("(120){4}1x,2xh[4:1],3x-5[4:1],E")
    assert all(not n.is_ex for n in fr.result.notes)
    assert [n.kind for n in fr.result.notes] == ["tap", "hold", "slide_star", "slide_track"]
    # EX 标在星头上（`3x-5`），轨道本身不带 x：tap / hold / 星头 各一
    assert fr.summary()["demoted"]["ex"] == 3


def test_BreakHOLD降级成普通HOLD():
    fr = _st("(120){4}1bh[4:1],E")
    n = fr.result.notes[0]
    assert n.kind == "hold" and not n.is_break
    assert fr.summary()["demoted"]["break_hold"] == 1
    assert fr.result.counts["hold"] == 1 and fr.result.counts["breaks"] == 0


def test_滑条绝赞降级成普通slide而星头绝赞保留():
    """`1b-5[4:1]b`：头上的 b（ST 合法）留着，轨道上的 b（FESTiVAL）摘掉。"""
    fr = _st("(120){4}1b-5[4:1]b,E")
    star, track = fr.result.notes
    assert star.kind == "slide_star" and star.is_break        # 星星头绝赞 = ST 合法
    assert track.kind == "slide_track" and not track.is_break  # 滑条绝赞 = 降级
    assert fr.summary()["demoted"]["break_slide"] == 1


def test_BreakTAP原样保留():
    fr = _st("(120){4}1b,E")
    assert fr.result.notes[0].is_break
    assert not fr.demoted and not fr.dropped


# ---------------------------------------------------------------------------
# 拼接（连锁）星星
# ---------------------------------------------------------------------------


def test_拼接星星丢掉滑轨星头默认按tap计():
    fr = _st("(120){4}1-3-5[4:1],E")
    assert [n.kind for n in fr.result.notes] == ["tap"]
    assert fr.result.counts["slide"] == 0
    assert fr.result.counts["taps"] == 1
    assert [r.reason for r in fr.dropped] == ["chain"]
    assert [r.reason for r in fr.demoted] == ["chain"]


def test_拼接星星星头可按星星头计():
    fr = _st("(120){4}1-3-5[4:1],E", head_as="star")
    assert [n.kind for n in fr.result.notes] == ["slide_star"]
    assert fr.result.counts["slide"] == 0
    assert fr.head_as == "star"


def test_同头多星星只丢被判死的那一条():
    """`*` 同头两条：一条是单段（留），一条是拼接（丢）→ 星头还有轨道，不降级。"""
    fr = _st("(120){4}1-5[4:1]*-3-7[4:1],E")
    kinds = [n.kind for n in fr.result.notes]
    assert kinds == ["slide_star", "slide_track"]
    assert fr.result.notes[0].kind == "slide_star"  # 头没有被改成 tap
    assert [r.reason for r in fr.dropped] == ["chain"]
    assert not [r for r in fr.demoted if r.reason == "chain"]


def test_head_as只接受两种口径():
    with pytest.raises(ValueError):
        _st("(120){4}1,E", head_as="break")


# ---------------------------------------------------------------------------
# ST 形状白名单
# ---------------------------------------------------------------------------


def test_12种ST形状全部保留():
    for shape, end in [("-", "5"), ("^", "3"), ("<", "4"), (">", "4"), ("v", "5"),
                       ("V", "35"), ("p", "5"), ("q", "5"), ("pp", "5"),
                       ("qq", "5"), ("s", "5"), ("z", "5"), ("w", "5")]:
        fr = _st(f"(120){{4}}1{shape}{end}[4:1],E")
        assert fr.result.counts["slide"] == 1, shape
        assert not fr.dropped, shape
    assert ST_SLIDE_SHAPES == frozenset("- ^ < > v V p q pp qq s z w".split())


# ---------------------------------------------------------------------------
# 时间轴必须原样保住
# ---------------------------------------------------------------------------


def test_时间轴与小节划分不因过滤而改变():
    src = "(120){4}1,C,{8}2x,E3h[4:1],(150)3,4,5,6,7,8,1,2,3,E"
    raw = parse_chart(src)
    fr = filter_to_st(raw)
    assert fr.result.tempo_segments == raw.tempo_segments
    assert fr.result.measure_starts == raw.measure_starts
    assert fr.result.total_beats == pytest.approx(raw.total_beats, **APPROX)
    assert fr.result.total_seconds == pytest.approx(raw.total_seconds, **APPROX)
    assert fr.result.bpm_events == raw.bpm_events
    # 留下来的 note 时间/小节号逐个不变
    kept = {(n.time, n.key, n.kind) for n in raw.notes if n.kind not in ("touch", "touch_hold")}
    for n in fr.result.notes:
        assert any(abs(n.time - t) < 1e-9 for t, _, _ in kept)


def test_统计口径前后对账():
    src = "(120){4}1x,C,2bh[4:1],3-5-7[4:1],4b-8[4:1]b,E"
    fr = _st(src)
    s = fr.summary()
    assert s["counts_before"]["touch"] == 1
    assert s["counts_after"]["touch"] == 0
    assert s["n_dropped"] == 2          # touch + 拼接轨道
    assert s["dropped"] == {"chain": 1, "touch": 1}
    assert set(s["demoted"]) == {"ex", "break_hold", "break_slide", "chain"}


# ---------------------------------------------------------------------------
# split_inote
# ---------------------------------------------------------------------------


def test_split_inote按编号切正文():
    txt = ("&title=T\n&first=0\n"
           "&inote_4=(120){4}1,2,E\n"
           "&inote_5=(120){4}3,4,E\n"
           "&des=x\n")
    assert "3,4" in split_inote(txt, 5)
    assert "1,2" in split_inote(txt, 4)
    assert "&des" not in split_inote(txt, 5)
    # 不指定编号时取最后一个
    assert "3,4" in split_inote(txt)


def test_split_inote容忍BOM与无inote():
    assert split_inote("﻿&inote_5=(120){4}1,E").strip().startswith("(120)")
    assert split_inote("(120){4}1,E") == "(120){4}1,E"


# ---------------------------------------------------------------------------
# 逐小节回写
# ---------------------------------------------------------------------------


def test_render_bars回写分音与note():
    from chart_analysis.dx_filter import render_bars
    fr = _st("(120){4}1,2,3,4,{8}5,6,7,8,1,2,3,4,E")
    out = render_bars(fr.result)
    assert out[1] == "{4}1,2,3,4,"
    assert out[2] == "{8}5,6,7,8,1,2,3,4,"


def test_render_bars写回hold与slide时长():
    """时长写成**最简**的 `[x:y]`（`[4:2]` = 2 拍 → `[2:1]`），等价但不一定同形。"""
    from chart_analysis.dx_filter import render_bars
    fr = _st("(120){4}1h[4:2],2b,3-7[8:1],4b-8[4:1],E")
    assert render_bars(fr.result)[1] == "{4}1h[2:1],2b,3-7[8:1],4b-8[4:1],"


def test_render_bars把touch与拼接星星摘掉后再写():
    from chart_analysis.dx_filter import render_bars
    fr = _st("(120){4}1/C,2x,3-5-7[4:1],E4h[4:1],E")
    # touch 没了；拼接星星只剩一个 tap；EX 没了
    assert render_bars(fr.result)[1] == "{4}1,2,3,,"


def test_render_bars同头多星星用星号():
    """分音取**该小节实际落点的最小可行值**：只有一个落点就是 `{1}`。"""
    from chart_analysis.dx_filter import render_bars
    fr = _st("(120){4}1-5[4:1]*-3[4:1],,,,E")
    assert render_bars(fr.result)[1] == "{1}1-5[4:1]*-3[4:1],"


def test_render_bars空小节给空串():
    from chart_analysis.dx_filter import render_bars
    fr = _st("(120){1},,1,E")
    out = render_bars(fr.result, 1, 3)
    assert out[1] == "" and out[2] == "" and out[3] == "{1}1,"
