#!/usr/bin/env python3
"""``tools/chart_analysis`` 的单元测试。

**测试素材全部为自写的小段 simai 片段**，不复制官方谱原文（官方谱为版权数据，
仅本机参考）。覆盖：变速、分音切换、``{#秒}``、hold/slide 时长、each 计数、
小节划分、note 计数口径，以及逐小节密度统计。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_chart_analysis.py -q``
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis.density import (  # noqa: E402
    chart_density,
    changepoint_segments,
    low_density_runs,
    measure_stats,
    resample_curve,
    segment_means,
    structure_profile,
)
from chart_analysis.simai_parser import parse_chart, parse_duration  # noqa: E402

APPROX = dict(rel=0, abs=1e-9)


# ---------------------------------------------------------------------------
# 时间轴：分音、变速、{#秒}
# ---------------------------------------------------------------------------


def test_基本槽长与小节划分():
    """(120){4} 每槽 0.5 秒；4 个四分音符 = 1 小节。"""
    r = parse_chart("(120){4}1,2,3,4,1,E")
    assert not r.errors
    assert r.has_end_marker
    times = [n.time for n in r.notes]
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0], **APPROX)
    # 前 4 个在第 0 小节，第 5 个进入第 1 小节
    assert [n.measure for n in r.notes] == [0, 0, 0, 0, 1]
    assert [n.beat_in_measure for n in r.notes] == pytest.approx([0, 1, 2, 3, 0], **APPROX)
    assert r.total_seconds == pytest.approx(2.5, **APPROX)
    assert r.total_beats == pytest.approx(5.0, **APPROX)


def test_分音切换():
    """{4} -> {8}：槽长减半，拍推进减半。"""
    r = parse_chart("(120){4}1,{8}2,3,E")
    assert not r.errors
    assert [n.time for n in r.notes] == pytest.approx([0.0, 0.5, 0.75], **APPROX)
    assert [n.beat for n in r.notes] == pytest.approx([0.0, 1.0, 1.5], **APPROX)


def test_非2的幂分音():
    """{12} 三连音：一个逗号 = 1/3 拍。"""
    r = parse_chart("(180){12}1,2,3,4,E")
    assert not r.errors
    assert [n.beat for n in r.notes] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0], **APPROX)
    # 180 BPM 下一拍 = 1/3 秒，一槽 = 1/9 秒
    assert [n.time for n in r.notes] == pytest.approx([0.0, 1 / 9, 2 / 9, 1 / 3], **APPROX)


def test_变速():
    """BPM 切换后槽长改变，但小节仍是 4 拍。"""
    r = parse_chart("(120){4}1,2,3,4,(240)5,6,7,8,1,E")
    assert not r.errors
    times = [n.time for n in r.notes]
    # 前 4 槽各 0.5s -> 第 5 个 note 在 2.0s；此后每槽 0.25s
    assert times == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 2.75, 3.0], **APPROX)
    assert [n.measure for n in r.notes] == [0, 0, 0, 0, 1, 1, 1, 1, 2]
    assert len(r.bpm_events) == 2


def test_秒数槽():
    """``{#秒}`` 直接指定每槽秒数；拍位按当前 BPM 折算。"""
    r = parse_chart("(120){#0.25}1,2,3,E")
    assert not r.errors
    assert [n.time for n in r.notes] == pytest.approx([0.0, 0.25, 0.5], **APPROX)
    # 120 BPM 下 0.25s = 0.5 拍
    assert [n.beat for n in r.notes] == pytest.approx([0.0, 0.5, 1.0], **APPROX)


def test_缺少结束标记会告警():
    r = parse_chart("(120){4}1,2,")
    assert any("E" in w for w in r.warnings)
    assert not r.has_end_marker


# ---------------------------------------------------------------------------
# 时长括号
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body,bpm,move,wait",
    [
        ("4:1", 120, 0.5, None),  # 240/120/4*1
        ("8:3", 120, 0.75, None),  # 240/120/8*3
        ("#1.25", 120, 1.25, None),
        ("60#4:1", 120, 1.0, 1.0),  # 指定 BPM 60：移动 240/60/4=1s，等待 60/60=1s
        ("60#0.4", 120, 0.4, 1.0),
        ("0.3##0.7", 120, 0.7, 0.3),
        ("0.3##4:1", 120, 0.5, 0.3),
        ("0.3##60#4:1", 120, 1.0, 0.3),
    ],
)
def test_时长括号六种形式(body, bpm, move, wait):
    m, w = parse_duration(body, bpm)
    assert m == pytest.approx(move, **APPROX)
    if wait is None:
        assert w is None
    else:
        assert w == pytest.approx(wait, **APPROX)


def test_hold时长():
    r = parse_chart("(120){4}1h[4:1],2h[#1.5],3h[60#2:1],E")
    assert not r.errors
    holds = [n for n in r.notes if n.kind == "hold"]
    assert [h.duration for h in holds] == pytest.approx([0.5, 1.5, 2.0], **APPROX)


def test_省略时长的hold是伪tap():
    """``3h`` 无时长 = 伪 TAP，内部等价 [1280:1]，仍按 HOLD 计数。"""
    r = parse_chart("(120){4}3h,E")
    assert not r.errors
    assert r.counts["hold"] == 1
    assert r.notes[0].duration == pytest.approx(240.0 / 120.0 / 1280.0, **APPROX)


def test_修饰符与h顺序任意():
    for src in ("(120){4}5hb[2:1],E", "(120){4}5bh[2:1],E"):
        r = parse_chart(src)
        assert not r.errors, src
        assert r.notes[0].kind == "hold"
        assert r.notes[0].is_break
        assert r.notes[0].duration == pytest.approx(1.0, **APPROX)


# ---------------------------------------------------------------------------
# SLIDE
# ---------------------------------------------------------------------------


def test_slide拆成星头与滑轨():
    r = parse_chart("(120){4}1-4[8:3],E")
    assert not r.errors
    kinds = [n.kind for n in r.notes]
    assert kinds == ["slide_star", "slide_track"]
    track = r.notes[1]
    assert track.duration == pytest.approx(0.75, **APPROX)
    assert track.wait == pytest.approx(60.0 / 120.0, **APPROX)  # 启动拍 = 一拍
    assert track.shape == "-"
    assert track.end_key == "4"
    assert r.counts == {"taps": 1, "hold": 0, "slide": 1, "touch": 0, "breaks": 0, "notes": 2}


def test_同头多slide只有一个星头():
    """``*`` 连接：一个星头 + 两条滑轨 = 3 note。"""
    r = parse_chart("(120){4}1-4[4:3]*-6[8:5],E")
    assert not r.errors
    assert [n.kind for n in r.notes] == ["slide_star", "slide_track", "slide_track"]
    assert r.counts["taps"] == 1
    assert r.counts["slide"] == 2
    assert r.counts["notes"] == 3


def test_连锁slide整体算一条():
    """首尾相接的连锁 slide（总时长写在最后）= 1 星头 + 1 滑轨。"""
    r = parse_chart("(120){4}1-4q7-2[1:2],E")
    assert not r.errors
    assert r.counts["slide"] == 1
    assert r.counts["taps"] == 1
    assert r.slide_chain_segments == 3  # 分段口径下是 3 段
    assert r.notes[1].shape == "-q-"
    assert r.notes[1].end_key == "2"


def test_V形三键记法():
    r = parse_chart("(120){4}1V36[8:1],E")
    assert not r.errors
    assert r.counts["slide"] == 1
    assert r.notes[1].end_key == "6"  # 1 起点、3 转折、6 终点


def test_双字符形状pp_qq():
    r = parse_chart("(120){4}7pp4[4:3],2qq5[4:3],E")
    assert not r.errors
    shapes = [n.shape for n in r.notes if n.kind == "slide_track"]
    assert shapes == ["pp", "qq"]


def test_滑轨break两种写法等价():
    for src in ("(120){4}1-4[8:3]b,E", "(120){4}1-4b[8:3],E"):
        r = parse_chart(src)
        assert not r.errors, src
        assert r.counts["breaks"] == 1  # 滑轨 BREAK 计入 breaks
        assert r.counts["slide"] == 0
        assert r.counts["taps"] == 1  # 星头仍是普通 TAP


def test_星头break与滑轨分开计():
    r = parse_chart("(120){4}1b-4[4:1],E")
    assert not r.errors
    assert r.counts["breaks"] == 1  # 星头 BREAK
    assert r.counts["slide"] == 1
    assert r.counts["taps"] == 0


# ---------------------------------------------------------------------------
# EACH / 多押
# ---------------------------------------------------------------------------


def test_斜杠each():
    r = parse_chart("(120){4}1/5,2,E")
    assert not r.errors
    assert r.each_groups == 1
    assert [n.is_each for n in r.notes] == [True, True, False]


def test_纯tap多押可省略斜杠():
    r = parse_chart("(120){4}12,345,E")
    assert not r.errors
    assert r.counts["taps"] == 5
    assert r.each_groups == 2
    assert [n.key for n in r.notes] == ["1", "2", "3", "4", "5"]


def test_slide与tap同刻构成each():
    r = parse_chart("(120){4}1-4[8:1]/5,E")
    assert not r.errors
    assert r.each_groups == 1
    assert r.counts["notes"] == 3  # 星头 + 滑轨 + tap


def test_伪each用反引号不计each():
    """``1`2`` 为伪 EACH（后者晚 1ms），本解析器按独立成员拆开。"""
    r = parse_chart("(120){4}1`2,E")
    assert not r.errors
    assert r.counts["taps"] == 2


def test_break不计入taps():
    r = parse_chart("(120){4}1b,2,3bx,E")
    assert not r.errors
    assert r.counts == {"taps": 1, "hold": 0, "slide": 0, "touch": 0, "breaks": 2, "notes": 3}
    assert r.notes[2].is_ex


def test_touch单独统计():
    r = parse_chart("(120){4}C,A3,Ch[4:1],E1f,E")
    assert not r.errors
    assert r.counts["touch"] == 4
    assert sum(1 for n in r.notes if n.kind == "touch_hold") == 1


# ---------------------------------------------------------------------------
# 错误处理：不静默跳过
# ---------------------------------------------------------------------------


def test_未知字符记入errors():
    r = parse_chart("(120){4}1,2Ω,3,E")
    assert r.errors
    assert any("Ω" in e for e in r.errors)
    # 其余 note 仍被解析出来
    assert r.counts["taps"] == 3


def test_缺bpm时报错():
    r = parse_chart("{4}1,2,E")
    assert r.errors


def test_注释被忽略():
    r = parse_chart("(120){4}1, || 这是注释 2,3,\n4,E")
    assert not r.errors
    assert r.counts["taps"] == 2  # 只剩 1 和 4


def test_tap带时长括号按hold处理并告警():
    r = parse_chart("(120){4}1[4:1],E")
    assert not r.errors
    assert r.warnings
    assert r.counts["hold"] == 1


# ---------------------------------------------------------------------------
# 逐小节密度
# ---------------------------------------------------------------------------


def _four_measures() -> str:
    """自写 4 小节片段：密度 4 / 8 / 1 / 8。"""
    return (
        "(120){4}1,2,3,4,"  # 第 0 小节：4 note
        "{8}1,2,3,4,5,6,7,8,"  # 第 1 小节：8 note
        "{4}1,,,,"  # 第 2 小节：1 note（休息小节）
        "{8}1,2,3,4,5,6,7,8,"  # 第 3 小节：8 note
        "E"
    )


def test_逐小节统计():
    r = parse_chart(_four_measures())
    assert not r.errors
    stats = measure_stats(r)
    assert [s.measure for s in stats] == [0, 1, 2, 3]
    assert [s.notes for s in stats] == [4, 8, 1, 8]
    assert [s.is_rest for s in stats] == [False, False, True, False]
    assert [s.finest_divisor for s in stats] == [4, 8, 4, 8]
    assert [round(s.start_time, 6) for s in stats] == [0.0, 2.0, 4.0, 6.0]


def test_曲线与峰值位置():
    d = chart_density(parse_chart(_four_measures()))
    assert list(d.raw) == [4, 8, 1, 8]
    assert list(d.normalized) == pytest.approx([0.5, 1.0, 0.125, 1.0])
    assert d.peak_measure == 1
    assert d.peak_notes == 8
    assert d.peak_position == pytest.approx(1 / 3)
    assert d.rest_runs == [(2, 1)]


def test_重采样保长度且归一():
    import numpy as np

    c = np.array([1.0, 2.0, 3.0, 4.0])
    out = resample_curve(c, 8)
    assert len(out) == 8
    assert out.max() == pytest.approx(1.0)
    # 单调递增曲线重采样后仍单调不降
    assert all(out[i] <= out[i + 1] + 1e-9 for i in range(7))


def test_等分五段均值():
    import numpy as np

    c = np.array([0.0] * 2 + [1.0] * 2 + [2.0] * 2 + [3.0] * 2 + [4.0] * 2)
    assert list(segment_means(c, 5)) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_变点检测能找到真实边界():
    import numpy as np

    c = np.array([0.1] * 10 + [0.9] * 10 + [0.2] * 10)
    b = changepoint_segments(c, 3, min_len_frac=0.1)
    assert b == [0, 10, 20, 30]


def test_变点检测最小段长约束生效():
    import numpy as np

    # 末尾一个突降小节：不加约束会被单独切段，加了 10% 约束则不会
    c = np.array([0.8] * 19 + [0.05])
    b_free = changepoint_segments(c, 2, min_len_frac=0.0)
    assert b_free == [0, 19, 20]
    b_cons = changepoint_segments(c, 2, min_len_frac=0.10)
    assert b_cons is not None and (b_cons[1] - b_cons[0]) >= 2 and (b_cons[2] - b_cons[1]) >= 2


def test_结构画像归一到峰值():
    import numpy as np

    c = np.array([0.2] * 10 + [1.0] * 10)
    sp = structure_profile(c, 2, min_len_frac=0.1)
    assert sp is not None
    prof, lens = sp
    assert list(prof) == pytest.approx([0.2, 1.0])
    assert list(lens) == pytest.approx([0.5, 0.5])


def test_低密段识别():
    import numpy as np

    c = np.array([1.0, 1.0, 0.1, 0.1, 1.0, 1.0])
    assert low_density_runs(c, ratio=0.5) == [(2, 2)]
