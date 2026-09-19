#!/usr/bin/env python3
"""``tools/chart_analysis/hands.py``（左右手分配器）的单元测试。

**测试素材全部为自写的合成 simai 片段**，不复制官方谱原文。
每类配置至少钉住一条"手级"断言（谁打什么、谁不打什么），
无理类至少一正一负例（负例钉住"不该报"的形态）。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_hands.py -q``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis import hands as hd  # noqa: E402
from chart_analysis.simai_parser import parse_chart  # noqa: E402


def _assign(body: str):
    res = parse_chart(body, name="t")
    assert not res.errors, res.errors
    return hd.assign(res)


def _hands(body: str) -> str:
    """把分配结果压成 ``键+手`` 串，便于一眼核对。"""
    ha = _assign(body)
    return " ".join(f"{n.key}{n.hand}" for n in ha.notes if n.note_type != "slide_track")


def _kinds(ha) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in ha.muri:
        out[m.kind] = out.get(m.kind, 0) + 1
    return out


# ---------------------------------------------------------------------------
# 键位几何（知识 006 / 023）
# ---------------------------------------------------------------------------


def test_环距与步进():
    assert hd.cdist(1, 2) == 1
    assert hd.cdist(1, 5) == 4
    assert hd.cdist(8, 1) == 1
    assert hd.cstep(1, 2) == 1
    assert hd.cstep(1, 8) == -1


def test_分页归属_知识006():
    assert [hd.home_side(k) for k in (1, 2, 3, 4)] == ["R"] * 4
    assert [hd.home_side(k) for k in (5, 6, 7, 8)] == ["L"] * 4


def test_越界深度_知识023():
    # 右手够得到 5 与 8（深度 1），够不到 6/7（深度 2）
    assert hd.page_depth("R", 3) == 0
    assert hd.page_depth("R", 5) == 1
    assert hd.page_depth("R", 8) == 1
    assert hd.page_depth("R", 6) == 2
    assert hd.page_depth("R", 7) == 2
    assert hd.page_depth("L", 4) == 1
    assert hd.page_depth("L", 2) == 2


# ---------------------------------------------------------------------------
# slide 轨道几何（近似）
# ---------------------------------------------------------------------------


def test_直星不经过途中A区():
    assert hd.slide_path_areas("-", 1, 5) == []
    assert hd.slide_path_areas("v", 1, 4) == []


def test_贴边弧经过沿途A区():
    areas = [a for _, a in hd.slide_path_areas("^", 7, 4)]
    assert areas == [6, 5]          # 7→6→5→4，两端不计
    areas = [a for _, a in hd.slide_path_areas("^", 2, 7)]
    assert areas == [1, 8]          # 2→1→8→7


def test_wifi三端点():
    assert hd.wifi_ends(4) == (3, 4, 5)


# ---------------------------------------------------------------------------
# 基础手序：交互 / 纵连 / 定拍（知识 018 / 019 / 027）
# ---------------------------------------------------------------------------


def test_普通交互_逐音换手_知识018():
    assert _hands("(150){16}1,8,1,8,1,8,1,8,E") == "1R 8L 1R 8L 1R 8L 1R 8L"


def test_交互的交替率为1():
    ha = _assign("(150){16}1,8,1,8,1,8,1,8,E")
    bars = [b for b in ha.bars if b.n_tasks]
    assert all(b.alt_rate == 1.0 for b in bars)
    assert ha.total_cost == 0.0          # 完美分页 + 完美交替 = 零代价


def test_高速同键纵连必须两手拆打_知识019():
    # 185BPM {16} ≈ 81 ms/音，单手做不到（知识 015 的 83.3 ms 下界）
    out = _hands("(185){16}1,1,1,1,1,1,1,1,E")
    assert out == "1R 1L 1R 1L 1R 1L 1R 1L"


def test_慢速定拍留在一只手_知识027():
    # 200BPM {8} = 150 ms/音，一只手守得住 → 不该被拆
    ha = _assign("(200){8}5,5,5,5,5,5,5,5,E")
    hands = {n.hand for n in ha.notes}
    assert hands == {"L"}, hands


def test_定拍与另一手分工_知识027():
    # 一手固定敲 2、另一手 6→8（知识 027 初音ミクの消失 形）
    out = _hands("(240){8}26,2,26,2,26,2,28,2,28,2,E")
    assert out.count("2R") == 10 and out.count("6L") == 3 and out.count("8L") == 2


# ---------------------------------------------------------------------------
# 几何配置：大宇宙 / 方向盘（知识 024 / 023）
# ---------------------------------------------------------------------------


def test_大宇宙_每手守一对相邻键_知识024():
    ha = _assign("(200){16}3,8,2,1,3,8,2,1,3,8,2,1,E")
    left = {k for _, k, _ in hd.bar_hand_sequences(ha)[0]["L"]}
    right = {k for _, k, _ in hd.bar_hand_sequences(ha)[0]["R"]}
    assert right == {2, 3} and left == {8, 1}


def test_方向盘_两手始终180度对位_知识023():
    ha = _assign("(200){16}1,5,2,6,3,7,4,8,5,1,4,8,3,7,2,6,E")
    for _, _, lk, rk in ha.pairs:
        assert False, "方向盘是交替单点，不应出现同刻双押"
    seq = hd.bar_hand_sequences(ha)[0]
    L = [k for _, k, _ in seq["L"]]
    R = [k for _, k, _ in seq["R"]]
    assert R == [1, 2, 3, 4, 5, 4, 3, 2]
    assert L == [5, 6, 7, 8, 1, 8, 7, 6]
    assert all(hd.cdist(a, b) == 4 for a, b in zip(L, R))


# ---------------------------------------------------------------------------
# 双押 / 多押（知识 008 / 012）
# ---------------------------------------------------------------------------


def test_双押各占一手():
    ha = _assign("(150){8}18,18,18,18,E")
    assert len(ha.pairs) == 4
    for _, _, lk, rk in ha.pairs:
        assert (lk, rk) == ("8", "1")
    assert not ha.infeasible


def test_三押判多押无理_知识008():
    ha = _assign("(150){8}123,,456,,E")
    assert _kinds(ha).get("多押") == 2
    assert len(ha.infeasible) == 2
    assert sum(1 for n in ha.notes if n.hand == "-") == 2


def test_双押不算多押_负例():
    ha = _assign("(150){8}12,,56,,E")
    assert "多押" not in _kinds(ha)


def test_Hold尾撞双押_知识012():
    # Hold 一直按到与双押同刻 → 需要三只手
    ha = _assign("(150){8}1h[1:1]/5,,,,,,,,37,,,,,,,,E")
    assert _kinds(ha).get("Hold尾多押", 0) >= 1


# ---------------------------------------------------------------------------
# 叠键（知识 009）
# ---------------------------------------------------------------------------


def test_叠键绝对无理_知识009():
    # 150BPM {64} = 25 ms < 33.3 ms
    ha = _assign("(150){64}1,1,1,1,E")
    assert _kinds(ha).get("叠键") == 3
    assert all(m.level == "绝对" for m in ha.muri if m.kind == "叠键")


def test_同键42ms不算叠键_负例_知识009():
    # 177.6BPM 的 32 分 = 42.2 ms，是官谱实测最快同键纵连，不是叠键
    ha = _assign("(177.6){32}1,1,1,1,E")
    assert "叠键" not in _kinds(ha)


# ---------------------------------------------------------------------------
# slide：启动拍 / 拍划 / 占手 / 一笔画（知识 006 / 007 / 014）
# ---------------------------------------------------------------------------


def test_启动拍固定一拍与分音无关_知识014():
    for div in ("{4}", "{8}", "{16}"):
        ha = _assign(f"(120){div}1-5[8:1],,,,,,,,,,,,,,,,E")
        track = next(n for n in ha.notes if n.note_type == "slide_track")
        assert abs(track.t_hand - track.t - 0.5) < 1e-6, div   # 60/120 = 0.5 s


def test_拍划并入划动手法不另占手_知识006():
    # 1-5 的星星头在 slot0，启动拍（1 拍后 = slot2 @ {8}）上写同键 tap 1
    ha = _assign("(120){8}1-5[8:1],7,1,6,E")
    merged = [n for n in ha.notes if n.merged]
    assert len(merged) == 1 and merged[0].key == "1"
    track = next(n for n in ha.notes if n.note_type == "slide_track")
    assert merged[0].hand == track.hand


def test_错位音由划星的那只手自己拍_知识017():
    # 知识 017 海底谭基础形：左手 头8 → 错位音7 → 启动拍8 → 划 8-4
    ha = _assign("(120){8}2,1/8-4[8:1],7,1-5[8:1]/8,2,1/8-4[8:1],7,1-5[8:1]/8,E")
    first = [n for n in ha.notes if n.t < 0.8]
    star8 = next(n for n in first if n.note_type == "slide_star" and n.key == "8")
    tap7 = next(n for n in first if n.key == "7")
    track84 = next(n for n in first if n.note_type == "slide_track" and n.key == "8")
    tapslide = next(n for n in ha.notes if n.merged and n.key == "8")
    assert star8.hand == tap7.hand == track84.hand == tapslide.hand


def test_轨道途中的音只能给另一只手():
    # 1-5 划动期间（启动拍后）出现的 tap 必须落到另一只手
    ha = _assign("(120){8}1-5[2:1]/2,3,4,2,E")
    track = next(n for n in ha.notes if n.note_type == "slide_track")
    during = [n for n in ha.notes
              if n.note_type == "tap" and track.t_hand < n.t < track.occupancy_until]
    assert during and all(n.hand != track.hand for n in during)


def test_一笔画_星链续划不换手_知识007():
    # 前一条轨道停在下一条的头键上、时间首尾相接 → 同一只手连续划完
    ha = _assign("(120){4}1-5[4:1],5-1[4:1],1-5[4:1],,,,E")
    tracks = [n for n in ha.notes if n.note_type == "slide_track"]
    assert len(tracks) == 3
    assert len({t.hand for t in tracks}) == 1


def test_星链不报多押_负例():
    ha = _assign("(120){4}1-5[4:1],5-1[4:1],1-5[4:1],,,,E")
    assert "多押" not in _kinds(ha) and not ha.infeasible


# ---------------------------------------------------------------------------
# 外键 / 撞尾（知识 010 / 011）
# ---------------------------------------------------------------------------


def test_撞尾踩在结束时刻为绝对无理_知识011():
    # 1-5 @120BPM {8}：头 slot0、启动拍 slot2、[8:1] 走 1 槽 → slot3 到 5
    ha = _assign("(120){8}1-5[8:1],7,6,5,E")
    tail = [m for m in ha.muri if m.kind == "撞尾"]
    assert tail and tail[0].level == "绝对"


def test_撞尾窗口外不报_负例_知识011():
    ha = _assign("(120){8}1-5[8:1],7,6,,,,5,E")
    assert "撞尾" not in _kinds(ha)


def test_外键同头判定_知识010():
    # 启动拍后半拍（250 ms > 200 ms）不该报；同头同刻紧跟的才报
    ha = _assign("(120){8}1-5[8:1],7,6,,1,E")
    assert "外键" not in _kinds(ha)


# ---------------------------------------------------------------------------
# 超速（知识 015）
# ---------------------------------------------------------------------------


def test_单手极限内不报超速_负例_知识015():
    ha = _assign("(200){16}1,5,2,6,3,7,4,8,E")   # 13.3 键/秒，两手各 6.7
    assert "超速" not in _kinds(ha)


def test_扫键不判不可行_知识015_036():
    # 96 分圈（知识 015 白日舞 102BPM ≈ 24.5 ms）应当走扫键模式而不是判死
    ha = _assign("(102){96}1,2,3,4,5,6,7,8,E")
    assert not ha.infeasible


# ---------------------------------------------------------------------------
# 对外 API 形状
# ---------------------------------------------------------------------------


def test_rows_是六元组():
    ha = _assign("(150){8}1,8,1,8,E")
    assert ha.rows and len(ha.rows[0]) == 6
    t, bar, key, kind, hand, until = ha.rows[0]
    assert isinstance(bar, int) and hand in ("L", "R", "LR", "-")
    assert until >= t
    assert list(ha) == ha.rows          # 可直接迭代


def test_左右手序列与双押对():
    ha = _assign("(150){8}1,8,15,8,E")
    assert [k for _, _, k, _ in ha.left_seq] == ["8", "8", "5"] or \
           [k for _, _, k, _ in ha.left_seq] == ["8", "5", "8"]
    assert len(ha.pairs) == 1
    seq = hd.bar_hand_sequences(ha)
    assert set(seq[0]) == {"L", "R", "pairs"}


def test_逐小节指标字段齐全():
    ha = _assign("(150){8}1,8,1,8,1,8,1,8,E")
    d = ha.bars[0].to_dict()
    for k in ("alt_rate", "max_same_run", "cross_rate" if False else "cross_count",
              "max_speed_l", "max_speed_r", "conflicts", "hand_hardness"):
        assert k in d


def test_chart_summary_口径():
    ha = _assign("(150){8}1,8,1,8,E")
    s = hd.chart_summary(ha)
    for k in ("n_tasks", "hand_hardness", "alt_rate", "n_infeasible", "n_muri"):
        assert k in s


def test_接受note列表():
    res = parse_chart("(150){8}1,8,1,8,E", name="t")
    a = hd.assign(res)
    b = hd.assign(res.notes)
    assert a.rows == b.rows


def test_空谱不炸():
    ha = hd.assign([])
    assert ha.rows == [] and ha.bars == [] and ha.total_cost == 0.0


# ---------------------------------------------------------------------------
# 参数可整体替换（操作化层与知识层分离）
# ---------------------------------------------------------------------------


def test_参数可覆盖():
    body = "(150){16}1,2,3,4,5,6,7,8,E"
    a = hd.assign(parse_chart(body, name="t"))
    b = hd.assign(parse_chart(body, name="t"), {"w_page": 0.0})
    assert b.total_cost <= a.total_cost
