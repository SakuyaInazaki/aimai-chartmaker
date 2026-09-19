#!/usr/bin/env python3
"""``tools/chart_analysis/configs.py``（配置识别器 + 逐小节硬度）的单元测试。

**测试素材全部为自写的合成 simai 片段**，不复制官方谱原文。
每个配置至少一正一负例；负例用来钉住"不该被误判成该配置"的形态
（多数负例正是首轮实测里抓到的假阳性类型）。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_configs.py -q``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis import configs as cf  # noqa: E402
from chart_analysis.simai_parser import parse_chart  # noqa: E402


def _cfgs(body: str) -> set[str]:
    res = parse_chart(body, name="t")
    assert not res.errors, res.errors
    return {h.config for h in cf.detect_all(res)}


def _hits(body: str, name: str) -> list[cf.ConfigHit]:
    res = parse_chart(body, name="t")
    assert not res.errors, res.errors
    return [h for h in cf.detect_all(res) if h.config == name]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def test_环形距离与步进():
    assert cf.cdist(1, 2) == 1
    assert cf.cdist(1, 5) == 4
    assert cf.cdist(8, 1) == 1        # 跨 8→1 只有一格
    assert cf.cstep(8, 1) == 1
    assert cf.cstep(1, 8) == -1
    assert cf.cstep(1, 3) == 2


def test_弧跨度():
    assert cf._arc_span({3, 4}) == 1
    assert cf._arc_span({2, 3, 4}) == 2
    assert cf._arc_span({1, 3, 4}) == 3
    assert cf._arc_span({6, 7, 1}) == 3   # 跨 8 的弧
    assert cf._arc_span({1, 5}) == 4


def test_时间槽归并():
    res = parse_chart("(120){8}1/5,2,3,E", name="t")
    slots = cf.build_slots(res)
    assert [s.keys for s in slots] == [(1, 5), (2,), (3,)]
    assert slots[0].is_double and not slots[1].is_double


def test_匀拍段切分_遇到不匀即断():
    # 前 7 个音都隔 0.5 拍（分音标记换在最后一个音之后才生效）；第 8 个音隔 1 拍 → 断
    res = parse_chart("(120){8}1,8,1,8,1,8,{4}1,8,E", name="t")
    runs = cf.uniform_runs(cf.build_slots(res), min_len=4)
    assert runs and runs[0] == (0, 6)


# ---------------------------------------------------------------------------
# 019 纵连
# ---------------------------------------------------------------------------


def test_纵连_正例():
    hits = _hits("(150){16}3,3,3,3,8,1,8,1,E", "纵连")
    assert len(hits) == 1
    assert hits[0].detail["key"] == 3 and hits[0].detail["length"] == 4


def test_长纵连_正例():
    hits = _hits("(150){16}" + "4," * 10 + "E", "长纵连")
    assert len(hits) == 1 and hits[0].detail["length"] == 10


def test_纵连_负例_只有两连是子弹不是纵连():
    body = "(150){16}3,3,8,1,8,1,8,1,E"
    assert "纵连" not in _cfgs(body)
    assert "子弹" in _cfgs(body)


def test_纵连_负例_同键星星串不算纵连():
    body = "(150){4}8-6[8:1],8-3[8:1],8-4[8:1],8-2[8:1],E"
    assert "纵连" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 030 连续双押 / 双押纵 / 侧边双押
# ---------------------------------------------------------------------------


def test_连续双押_正例_绕圈():
    hits = _hits("(150){8}1/2,2/3,3/4,4/5,E", "连续双押")
    assert len(hits) == 1
    assert hits[0].detail["subtype"] == "绕圈"


def test_双押纵_正例():
    hits = _hits("(150){16}1/8,1/8,1/8,1/8,E", "双押纵")
    assert len(hits) == 1 and hits[0].detail["length"] == 4


def test_连续双押_负例_只有两个双押():
    assert "连续双押" not in _cfgs("(150){8}1/2,2/3,5,6,7,8,E")


def test_侧边双押_红线与引导代理():
    # 周边无同键音 → guided_proxy False
    h = _hits("(150){4}2/3,5,8,1,E", "侧边双押")
    assert len(h) == 1 and h[0].detail["guided_proxy"] is False
    # 周边有同键音 → True
    h2 = _hits("(150){8}3,5,2/3,5,8,1,E", "侧边双押")
    assert len(h2) == 1 and h2[0].detail["guided_proxy"] is True
    # 12 不是侧边双押（知识 030：只指 23 与 67）
    assert "侧边双押" not in _cfgs("(150){4}1/2,5,8,1,E")


# ---------------------------------------------------------------------------
# 021 轴交互
# ---------------------------------------------------------------------------


def test_轴交互_正例_BABCB():
    hits = _hits("(150){16}3,4,3,2,3,4,3,2,3,E", "轴交互")
    assert hits and hits[0].detail["axis"] == 3
    assert set(hits[0].detail["others"]) == {2, 4}


def test_轴交互_负例_另一手不变就是纵连式而非轴():
    # 3,4,3,4,3,4 → 另一手只有一个键，不构成"轴 + 变化"
    assert "轴交互" not in _cfgs("(150){16}3,4,3,4,3,4,3,4,E")


# ---------------------------------------------------------------------------
# 020 三角交互
# ---------------------------------------------------------------------------


def test_三角交互_正例_占2键():
    hits = _hits("(150){12}3,4,3,6,5,6,3,4,3,6,5,6,E", "三角交互")
    assert hits and hits[0].detail["n_groups"] >= 4
    assert hits[0].detail["occupancy"][0] == 2


def test_三角交互_正例_占3键():
    hits = _hits("(150){12}3,4,2,6,5,7,2,1,3,7,8,6,E", "三角交互")
    assert hits and hits[0].detail["occupancy"][0] == 3


def test_三角交互_负例_单向音阶不是三角():
    # 7,6,5,4,3,2 是单向下行（每组不转向）
    assert "三角交互" not in _cfgs("(150){16}7,6,5,4,3,2,1,8,E")


def test_三角交互_负例_星星链不是三角():
    body = "(150){8}1-5[8:1],6,7,8-4[8:1],3,2,1-5[8:1],6,7,8-4[8:1],3,2,E"
    assert "三角交互" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 022 楼梯 / 023 逆楼梯（方向盘）
# ---------------------------------------------------------------------------


def test_楼梯交互_正例_不重复直接走():
    hits = _hits("(150){16}2,1,3,8,4,7,5,6,E", "楼梯交互")
    assert hits and hits[0].detail["n_pairs"] >= 3
    assert hits[0].detail["style"] == "不重复直接走"


def test_楼梯交互_正例_每对重复两次():
    hits = _hits("(150){16}1,8,1,8,2,7,2,7,3,6,3,6,4,5,4,5,E", "楼梯交互")
    assert hits and hits[0].detail["style"] == "每对重复"


def test_方向盘_正例_同向且始终对位():
    hits = _hits("(150){16}1,5,2,6,3,7,4,8,5,1,4,8,3,7,2,6,E", "逆楼梯/方向盘")
    assert hits and hits[0].detail["n_pairs"] >= 6
    assert hits[0].detail["reversal"] is True


def test_方向盘_负例_楼梯不应被判成方向盘():
    assert "逆楼梯/方向盘" not in _cfgs("(150){16}2,1,3,8,4,7,5,6,E")


def test_楼梯_负例_方向盘不应被判成楼梯():
    assert "楼梯交互" not in _cfgs("(150){16}1,5,2,6,3,7,4,8,E")


# ---------------------------------------------------------------------------
# 024 大宇宙
# ---------------------------------------------------------------------------


def test_大宇宙_正例():
    hits = _hits("(150){16}2,1,3,8,2,1,3,8,2,1,3,8,E", "大宇宙")
    assert hits
    pa, pb = hits[0].detail["pair_a"], hits[0].detail["pair_b"]
    assert sorted(pa + pb) == [1, 2, 3, 8]


def test_大宇宙_负例_四键但不是两对相邻():
    # {1,3,5,7} 每手那对不相邻 → 不是大宇宙
    assert "大宇宙" not in _cfgs("(150){16}1,5,3,7,1,5,3,7,1,5,3,7,E")


# ---------------------------------------------------------------------------
# 025 散点
# ---------------------------------------------------------------------------


def test_散点_正例():
    hits = _hits("(150){16}6,5,7,4,5,2,4,3,1,8,E", "散点")
    assert hits and hits[0].detail["n_keys"] >= 5


def test_散点_负例_楼梯有规律不算散点():
    assert "散点" not in _cfgs("(150){16}1,8,2,7,3,6,4,5,3,6,2,7,E")


def test_散点_负例_带星星不算散点():
    body = "(150){16}6-2[8:1],5,7,4,5,2,4,3,1,8,E"
    assert "散点" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 027 定拍
# ---------------------------------------------------------------------------


def test_定拍_正例_同刻双押型():
    hits = _hits("(150){8}2/6,2,2/6,2,2/8,2,2/8,2,2/6,2,E", "定拍")
    assert hits and hits[0].detail["key"] == 2


def test_定拍_正例_另一手会歇():
    # 5 每拍一次从不歇；另一手 6 只在部分拍上出现（会歇）
    hits = _hits("(150){8}" + "5,6,5,," * 5 + "E", "定拍")
    assert hits and hits[0].detail["key"] == 5


def test_定拍_负例_严格交互不是定拍():
    # 3,4,3,2,… 每两音一个轴音、另一手每拍都在 → 属轴交互不属定拍
    body = "(150){16}3,4,3,2,3,4,3,2,3,4,3,2,3,4,3,2,E"
    assert "定拍" not in _cfgs(body)
    assert "轴交互" in _cfgs(body)


# ---------------------------------------------------------------------------
# 028 子弹
# ---------------------------------------------------------------------------


def test_子弹_正例_同键相邻两音():
    hits = _hits("(150){16}8,8,5,3,1,6,E", "子弹")
    assert len(hits) == 1 and hits[0].detail["carrier"] == "同键相邻两音"


def test_子弹_正例_短双押串():
    hits = _hits("(150){16}1/2,1/2,5,7,3,8,E", "子弹")
    assert len(hits) == 1 and hits[0].detail["carrier"] == "短双押串"


def test_子弹_负例_三连就是纵连():
    assert "子弹" not in _cfgs("(150){16}8,8,8,5,3,1,E")


def test_子弹_负例_间隔太远():
    # {2} = 半小节一个音，远超"短时间内"
    assert "子弹" not in _cfgs("(150){2}8,8,E")


# ---------------------------------------------------------------------------
# 029 单双 / 双单
# ---------------------------------------------------------------------------


def test_单双_正例_一比一():
    hits = _hits("(150){8}3,2/4,3,1/2,3,2/6,3,2/4,E", "单双/双单")
    assert hits and hits[0].detail["ratio"] in ("1:1",)


def test_单双_正例_一双两单():
    hits = _hits("(150){8}3/4,2,2,3/4,2,2,3/4,2,2,E", "单双/双单")
    assert hits and hits[0].detail["period"] == 3


def test_单双_负例_全是单点():
    assert "单双/双单" not in _cfgs("(150){8}3,2,3,1,3,2,3,4,E")


# ---------------------------------------------------------------------------
# 017 错位
# ---------------------------------------------------------------------------


def test_错位_正例_海底谭形():
    hits = _hits("(120){8}2,1/8-4[8:1],7,1-5[8:1]/8,2,1/8-4[8:1],7,E", "错位")
    assert hits
    assert any(h.detail["n_inner"] >= 1 for h in hits)


def test_错位_负例_间隙里没有音():
    assert "错位" not in _cfgs("(120){4}1-5[8:1],,,,E")


def test_错位_负例_同刻不算错位():
    # 间隙里什么都没有，启动拍同刻的音不构成错位
    assert "错位" not in _cfgs("(120){4}1-5[8:1],3,E")


def test_错位_负例_双手同刻同任务是双押():
    # 两条滑轨同刻起手、间隙里的音也是双押 → 知识 017 的"双押不是错位"
    body = "(120){8}4-1[8:1]/5-8[8:1],4/5,3,6,E"
    assert "错位" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 018 普通交互
# ---------------------------------------------------------------------------


def test_普通交互_正例():
    hits = _hits("(180){16}1,8,1,8,1,8,1,8,E", "普通交互")
    assert hits and hits[0].detail["length"] == 8
    assert abs(hits[0].detail["nps"] - 12.0) < 0.5   # (16/4)*180/60 = 12


def test_普通交互_负例_纵连不算普通交互():
    assert "普通交互" not in _cfgs("(180){16}1,1,1,1,1,1,1,1,E")


def test_普通交互_负例_星星链不算():
    body = ("(180){8}1-5[8:1],6,2-6[8:1],7,3-7[8:1],8,4-8[8:1],1,E")
    assert "普通交互" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 026 跳拍（粗判）
# ---------------------------------------------------------------------------


def test_跳拍_正例_反复插空():
    hits = _hits("(150){8}1,,8,2,,7,3,,6,4,E", "跳拍")
    assert hits


def test_跳拍_负例_匀拍():
    assert "跳拍" not in _cfgs("(150){8}1,8,2,7,3,6,4,5,E")


def test_跳拍_负例_只有一处留白():
    assert "跳拍" not in _cfgs("(150){8}1,8,2,,7,3,6,4,E")


# ---------------------------------------------------------------------------
# 无理代理
# ---------------------------------------------------------------------------


def test_无理_叠键():
    # 600BPM 的 16 分 = 25 ms < 33.3 ms
    res = parse_chart("(600){16}3,3,3,E", name="t")
    kinds = {m.kind for m in cf.detect_muri(res)}
    assert "叠键" in kinds


def test_无理_外键与撞尾():
    # 启动后立刻同侧 tap → 外键；滑条终点键在结束时刻出现 tap → 撞尾
    res = parse_chart("(150){8}1-5[8:1],,,2,E", name="t")
    assert "外键" in {m.kind for m in cf.detect_muri(res)}
    res2 = parse_chart("(150){4}1-5[8:1],,5,E", name="t")
    assert "撞尾" in {m.kind for m in cf.detect_muri(res2)}


def test_无理_干净谱面无命中():
    res = parse_chart("(150){8}1,4,2,7,3,6,4,5,E", name="t")
    assert cf.detect_muri(res) == []


# ---------------------------------------------------------------------------
# 逐小节硬度
# ---------------------------------------------------------------------------


def test_硬度_高速大位移小节高于低速小位移小节():
    body = ("(180){4}1,2,1,2,"          # 小节 0：慢、位移小
            "{16}1,5,2,6,3,7,4,8,1,5,2,6,3,7,4,8,"   # 小节 1：快、位移大
            "E")
    res = parse_chart(body, name="t")
    bars = {b.measure: b for b in cf.bar_hardness(res)}
    assert bars[1].raw > bars[0].raw
    assert bars[1].term_speed > bars[0].term_speed
    assert bars[1].term_move > bars[0].term_move


def test_硬度_归一与三分位():
    body = "(180){4}1,2,1,2,{8}1,5,2,6,1,5,2,6,{16}1,5,2,6,3,7,4,8,1,5,2,6,3,7,4,8,E"
    res = parse_chart(body, name="t")
    bars = cf.bar_hardness(res)
    vals = [b.hardness for b in bars if b.n_notes]
    assert min(vals) == 0.0 and max(vals) == 1.0
    assert {b.hardness_class for b in bars if b.n_notes} <= {"低", "中", "高"}


def test_硬度_错位小节的种类项被抬高():
    plain = parse_chart("(120){8}2,1,7,8,2,1,7,8,E", name="t")
    mis = parse_chart("(120){8}2,1/8-4[8:1],7,1-5[8:1]/8,2,1/8-4[8:1],7,8,E", name="t")
    b0 = cf.bar_hardness(plain)[0]
    b1 = cf.bar_hardness(mis)[0]
    assert b1.term_kind > b0.term_kind
    assert b1.term_config > b0.term_config


def test_硬度_空小节不参与归一():
    body = "(150){4}1,2,3,4,{1},{4}5,6,7,8,E"
    res = parse_chart(body, name="t")
    bars = cf.bar_hardness(res)
    empty = [b for b in bars if b.n_notes == 0]
    assert empty and all(b.hardness_class == "" for b in empty)


# ---------------------------------------------------------------------------
# 汇总接口
# ---------------------------------------------------------------------------


def test_汇总与逐小节配置集合():
    body = "(150){16}1,8,1,8,1,8,1,8,,,,,3,3,3,3,E"
    res = parse_chart(body, name="t")
    hits = cf.detect_all(res)
    s = cf.summarize(hits)
    assert "普通交互" in s and "纵连" in s
    bar_map = dict(cf.iter_bar_configs(hits))
    assert bar_map and all(isinstance(v, set) for v in bar_map.values())


def test_一个片段可同时命中多个配置():
    # 轴交互同时也是普通交互
    body = "(180){16}3,4,3,2,3,4,3,2,3,4,3,2,E"
    got = _cfgs(body)
    assert {"轴交互", "普通交互"} <= got
