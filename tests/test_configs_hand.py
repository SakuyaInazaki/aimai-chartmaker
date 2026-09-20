#!/usr/bin/env python3
"""``tools/chart_analysis/configs_hand.py``（**手序版**配置识别器）的单元测试。

**测试素材全部为自写的合成 simai 片段**，不复制官方谱原文。
每类配置至少一正一负例；`test_键位版假阳性_*` 一组把键位版 `tests/test_configs.py`
里那 7 类假阳性负例（`docs/research/sampling-mode-study.md` §2.2 的表）原样搬过来，
确认手序版同样不误报。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_configs_hand.py -q``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis import configs_hand as ch  # noqa: E402
from chart_analysis import hands as H  # noqa: E402
from chart_analysis.simai_parser import parse_chart  # noqa: E402


def _parse(body: str):
    res = parse_chart(body, name="t")
    assert not res.errors, res.errors
    return res


def _cfgs(body: str, params: dict | None = None) -> set[str]:
    return {h.config for h in ch.detect_all(_parse(body), None, None, params)}


def _hits(body: str, name: str, params: dict | None = None) -> list[ch.ConfigHitH]:
    return [h for h in ch.detect_all(_parse(body), None, None, params)
            if h.config == name]


# ---------------------------------------------------------------------------
# 基础结构
# ---------------------------------------------------------------------------


def test_手级时间槽归并():
    ctx = ch.make_ctx(_parse("(120){8}1/5,2,3,E"))
    assert [s.hit_key_set for s in ctx.slots] == [{1, 5}, {2}, {3}]
    assert ctx.slots[0].is_pair and not ctx.slots[1].is_pair
    # 每个槽的两个任务必须分到不同的手
    assert set(ctx.slots[0].hit_hands) == {"L", "R"}


def test_击打视图排除划到一半的轨道():
    # 1-5 的轨道在启动拍（+1 拍）开始，正好落在 tap 3 那一槽上
    ctx = ch.make_ctx(_parse("(120){4}1-5[4:1],3,8,7,E"))
    slot = [s for s in ctx.slots if abs(s.beat - 1.0) < 1e-6][0]
    assert slot.n_tasks == 2 and slot.n_hits == 1   # 轨道不算"击打"
    assert slot.hit_keys == (3,)


def test_slide_unit_拆成头与轨道():
    ctx = ch.make_ctx(_parse("(120){4}1-5[4:1],,,,E"))
    us = ch.slide_units(ctx)
    assert len(us) == 1
    u = us[0]
    assert u.key == 1 and u.end == 5
    assert abs(u.t_start - u.t_head - 0.5) < 1e-3     # 启动拍 = 60/BPM
    assert u.hand in ("L", "R") and u.head_hand in ("L", "R")


# ---------------------------------------------------------------------------
# 017 错位
# ---------------------------------------------------------------------------


def test_错位_正例_同一只手头到错位音到启动拍():
    # 海底谭形（知识 017 引文同构）：两条星星线相位差一拍、错位音塞在缝里
    body = "(120){8}1/8-4[8:1],7,1-5[8:1]/8,2,1/8-4[8:1],7,1-5[8:1]/8,2,E"
    hits = _hits(body, "错位")
    assert hits
    h = hits[0]
    assert h.detail["hand"] in ("L", "R")
    assert h.detail["inner_keys"] and h.detail["n_inner"] >= 1
    # 手级核心：命中片段里的错位音**全部**由划轨的那只手自己拍（知识 017）；
    # 落到另一只手的那些不计入本片段（`n_inner_other_hand` 只作记录）
    ctx = ch.make_ctx(_parse(body))
    tr = [t for t in ctx.ha.tasks if t.kind == "slide" and t.key == h.detail["head_key"]]
    assert tr
    inner_hands = {ctx.hand(j) for j, x in enumerate(ctx.ha.tasks)
                   if x.key in h.detail["inner_keys"]
                   and any(t.index in x.wait_of for t in tr)}
    assert h.detail["hand"] in inner_hands


def test_错位_负例_间隙里没有音():
    assert "错位" not in _cfgs("(120){4}1-5[4:1],,,,E")


def test_错位_负例_同刻不算错位():
    # 与星星头同刻的音是双押，不是错位（知识 017 的界限）
    assert "错位" not in _cfgs("(120){4}1-5[4:1]/7,,,,E")


def test_错位_负例_双手同刻同任务是双押():
    # 两条滑轨同起 + 间隙音也是双押 → 两手在做同一件事
    assert "错位" not in _cfgs("(120){4}4-1[4:1]/5-8[4:1],4/5,,,E")


# ---------------------------------------------------------------------------
# 018 普通交互 / 019 纵连
# ---------------------------------------------------------------------------


def test_普通交互_正例_逐音换手():
    hits = _hits("(150){16}1,8,1,8,1,8,1,8,E", "普通交互")
    assert len(hits) == 1
    assert hits[0].detail["alt_rate"] == 1.0 and hits[0].detail["cross"] == 0


def test_普通交互_负例_纵连不算():
    assert "普通交互" not in _cfgs("(150){16}3,3,3,3,3,3,3,3,E")


def test_普通交互_负例_星星链不算():
    body = "(150){4}8-6[8:1],8-3[8:1],8-4[8:1],8-2[8:1],8-6[8:1],8-3[8:1],E"
    assert "普通交互" not in _cfgs(body)


def test_纵连_正例_两手拆与一手连打():
    fast = _hits("(240){16}3,3,3,3,8,1,8,1,E", "纵连")
    assert len(fast) == 1 and fast[0].detail["key"] == 3
    # 240BPM 的 16 分 = 62.5 ms < 83.3 ms 单手下界 → 必须拆
    assert fast[0].detail["below_single_hand_limit"] is True
    slow = _hits("(80){8}3,3,3,8,1,8,E", "纵连")
    assert len(slow) == 1 and slow[0].detail["below_single_hand_limit"] is False


def test_长纵连_正例():
    hits = _hits("(150){16}" + "4," * 10 + "E", "长纵连")
    assert len(hits) == 1 and hits[0].detail["length"] == 10


def test_纵连_负例_只有两连是子弹():
    body = "(150){16}3,3,8,1,8,1,8,1,E"
    assert "纵连" not in _cfgs(body)
    assert "子弹" in _cfgs(body)


def test_纵连_负例_同键星星串不算纵连():
    body = "(150){4}8-6[8:1],8-3[8:1],8-4[8:1],8-2[8:1],E"
    assert "纵连" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 020 三角 / 021 轴 / 027 定拍
# ---------------------------------------------------------------------------


def test_三角交互_正例_组内手序XYX():
    hits = _hits("(222){12}3,4,3,6,5,6,3,4,3,6,5,6,E", "三角交互")
    assert hits
    assert all(p[0] == p[2] and p[0] != p[1]
               for p in hits[0].detail["hand_patterns"])


def test_三角交互_负例_单向音阶不是三角():
    assert "三角交互" not in _cfgs("(150){16}7,6,5,4,3,2,1,8,E")


def test_三角交互_负例_星星链不是三角():
    body = "(150){8}1-5[8:1],6,7,8-4[8:1],3,2,1-5[8:1],6,7,8-4[8:1],3,2,E"
    assert "三角交互" not in _cfgs(body)


def test_轴交互_正例():
    hits = _hits("(191){16}3,4,3,2,3,4,3,2,3,4,E", "轴交互")
    assert hits
    d = hits[0].detail
    assert d["axis_key"] == 3 and set(d["other_keys"]) >= {2, 4}


def test_轴交互_负例_另一手也在轴键上是纵连不是轴():
    # 同一个键被两手交替敲 = 019 的"拆"，知识 021 明说与纵连无关
    assert "轴交互" not in _cfgs("(150){16}" + "1," * 12 + "E")


def test_定拍_正例_一手从不歇():
    hits = _hits("(240){8}26,2,26,2,26,2,26,2,26,2,E", "定拍")
    assert hits
    d = hits[0].detail
    assert d["count"] >= 8 and d["other_idle"] >= 1


def test_定拍_负例_严格交互不是定拍():
    assert "定拍" not in _cfgs("(150){16}" + "1,8," * 8 + "E")


# ---------------------------------------------------------------------------
# 022 楼梯 / 023 方向盘 / 024 大宇宙 / 025 散点
# ---------------------------------------------------------------------------


def test_方向盘_正例_两手同向且始终对位():
    hits = _hits("(200){16}1,5,2,6,3,7,4,8,3,7,2,6,1,5,E", "逆楼梯/方向盘")
    assert hits and hits[0].detail["opposite"] is True


def test_方向盘_负例_楼梯不应被判成方向盘():
    assert "逆楼梯/方向盘" not in _cfgs("(199){16}2,1,3,8,4,7,5,6,E")


def test_楼梯_正例_两手反向推进():
    hits = _hits("(199){16}2,1,3,8,4,7,5,6,E", "楼梯交互")
    assert hits and hits[0].detail["dir_l"] == -hits[0].detail["dir_r"]


def test_楼梯_负例_方向盘不应被判成楼梯():
    assert "楼梯交互" not in _cfgs("(200){16}1,5,2,6,3,7,4,8,3,7,2,6,1,5,E")


def test_大宇宙_正例_两手各在一对相邻键横跳():
    hits = _hits("(240){16}1,7,2,8,1,7,2,8,1,7,2,8,E", "大宇宙")
    assert hits
    d = hits[0].detail
    assert sorted(d["pair_l"] + d["pair_r"]) == [1, 2, 7, 8]
    assert d["key_alt_l"] >= 0.6 and d["key_alt_r"] >= 0.6


def test_大宇宙_负例_四键但不是两对相邻():
    assert "大宇宙" not in _cfgs("(240){16}1,5,3,7,1,5,3,7,1,5,3,7,E")


def test_大宇宙_负例_每手不横跳只是分段停留():
    # `3,5,3,5,4,6,4,6`：每只手先敲两下 A 再敲两下 B，不是 2 键 trill
    assert "大宇宙" not in _cfgs("(191){16}3,5,3,5,4,6,4,6,3,5,3,5,4,6,4,6,E")


def test_散点_正例():
    hits = _hits("(190){16}2,7,4,1,6,3,8,5,2,6,4,8,3,7,1,5,E", "散点")
    assert hits and hits[0].detail["n_keys_l"] >= 4


def test_散点_负例_楼梯有规律不算散点():
    assert "散点" not in _cfgs("(199){16}2,1,3,8,4,7,5,6,1,8,2,7,3,6,4,5,E")


def test_散点_负例_带星星不算():
    body = "(150){8}1-4[8:1],,1,,7-4[8:1],,7,,1-4[8:1],,1,,7-4[8:1],,7,,E"
    assert "散点" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 028 子弹 / 029 单双 / 030 连续双押
# ---------------------------------------------------------------------------


def test_子弹_正例_记录同手还是分摊():
    hits = _hits("(150){16}8,8,3,5,2,6,E", "子弹")
    assert hits and "same_hand" in hits[0].detail


def test_子弹_负例_三连就是纵连():
    assert "子弹" not in _cfgs("(150){16}8,8,8,3,5,2,E")


def test_子弹_负例_间隔太远():
    assert "子弹" not in _cfgs("(150){4}8,8,3,5,E")


def test_子弹_负例_星星头加同键tap不算子弹():
    assert "子弹" not in _cfgs("(150){8}2-5[8:1],,2,,6,4,E")


def test_单双_正例_骨架可识别():
    # キミノヨゾラ 型：双押固定 4/5、单点变化（知识 029 的第一个变化维度）
    hits = _hits("(185){8}4/5,6,4/5,3,4/5,6,4/5,3,E", "单双/双单")
    assert hits
    assert hits[0].detail["skeleton_hand"] in ("L", "R", "双押", "单点")


def test_单双_负例_全是单点():
    assert "单双/双单" not in _cfgs("(150){16}1,8,1,8,1,8,1,8,E")


def test_连续双押_正例_绕圈():
    hits = _hits("(150){8}1/2,2/3,3/4,4/5,E", "连续双押")
    assert hits and hits[0].detail["subtype"] == "绕圈"


def test_双押纵_正例_两手键都常量():
    hits = _hits("(150){16}1/8,1/8,1/8,1/8,E", "双押纵")
    assert len(hits) == 1 and hits[0].detail["length"] == 4


def test_连续双押_负例_只有两个双押():
    assert "连续双押" not in _cfgs("(150){8}1/2,2/3,5,6,7,8,E")


def test_侧边双押_无引导进复核清单_知识030():
    # 判据来自 hands.side_double_events：标**是哪一种引导**（八型），"突然"不用数字定义
    h = _hits("(120){4}5,6,7,2/3,E", "侧边双押")
    assert len(h) == 1 and h[0].detail["guided"] is False
    assert h[0].detail["guide_types"] == [] and h[0].detail["guide_level"] == ""
    assert "redline" not in h[0].detail
    assert "sudden" not in h[0].detail and "disp_cfg" not in h[0].detail


def test_侧边双押_共享键步进有引导_负例_知识030B型():
    h = {tuple(x.detail["keys"]): x for x in _hits("(120){4}1/8,8/7,7/6,6/5,5/4,E", "侧边双押")}
    assert (6, 7) in h
    assert h[(6, 7)].detail["guided"] is True
    assert h[(6, 7)].detail["guide_type"] == "A/B"


def test_侧边双押_等距轮转有引导_负例_知识030C型():
    h = {tuple(x.detail["keys"]): x for x in _hits("(120){4}1/8,2/3,4/5,6/7,E", "侧边双押")}
    assert (2, 3) in h and (6, 7) in h
    assert all(v.detail["guide_type"] == "C" and v.detail["guided"] for v in h.values())


def test_侧边双押_普查新归纳的五型也进detail_多标签():
    # 报告 §3.1：原来判不出的 307 次全部落进 F1/F2/H/E/G 五型
    h = _hits("(120){8}2/3,1/2,4,5,6/7,E", "侧边双押")
    d = {tuple(x.detail["keys"]): x.detail for x in h}
    assert d[(6, 7)]["guide_type"] == "F1"
    assert set(d[(6, 7)]["guide_types"]) >= {"F1", "F2", "E"}


# ---------------------------------------------------------------------------
# 026 跳拍（时间层）
# ---------------------------------------------------------------------------


def test_跳拍_正例_反复插空():
    hits = _hits("(150){8}1,,8,1,,8,1,,8,1,,8,E", "跳拍")
    assert hits and hits[0].hands_pattern == "—"


def test_跳拍_负例_匀拍():
    assert "跳拍" not in _cfgs("(150){16}1,8,1,8,1,8,1,8,E")


# ---------------------------------------------------------------------------
# 星星篇：057 / 049 / 053 / 050 / 060 / 061 / 058 / 062
# ---------------------------------------------------------------------------


def test_一笔画_正例_一只手一路划完():
    body = "(167){4}1-6[8:1],6-2[8:1],2-5[8:1],5-1[8:1],E"
    hits = _hits(body, "一笔画")
    assert hits
    d = hits[0].detail
    assert d["length"] == 4 and d["closed"] is True
    # 手级增量：这条链由哪只（几只）手划完必须被报出来
    assert isinstance(d["one_hand"], bool) and len(d["track_hands"]) == 4


def test_一笔画_负例_头尾不相连():
    body = "(167){4}1-6[8:1],5-2[8:1],3-8[8:1],7-4[8:1],E"
    assert "一笔画" not in _cfgs(body)


def test_连续拍滑_正例_拍头手等于划轨手且两手交替():
    body = "(150){4}2-6[8:1],2/7-4[8:1],7/3-8[8:1],3/6-2[8:1],6,,E"
    hits = _hits(body, "连续拍滑")
    assert hits
    d = hits[0].detail
    assert d["n_units"] >= 3 and d["lag"] == 1
    assert all(d["hands"][i] != d["hands"][i + 1] for i in range(len(d["hands"]) - 1))


def test_连续拍滑_负例_两头同刻是同相位家族():
    body = "(160){4}4-2[8:1]/5-1[8:1],4-8[8:1]/5-7[8:1],4-2[8:1]/5-1[8:1],4,5,E"
    assert "连续拍滑" not in _cfgs(body)


def test_双压连续拍滑_正例():
    body = ("(160){4}4-2[8:1]/5-1[8:1],4-8[8:1]/5-7[8:1],"
            "4-2[8:1]/5-1[8:1],4-8[8:1]/5-7[8:1],4/5,E")
    hits = _hits(body, "双压连续拍滑")
    assert hits
    d = hits[0].detail
    assert {d["key_l"], d["key_r"]} == {4, 5} and d["n_units"] >= 3


def test_夹键拍滑_正例_同键连吃三个八分():
    body = "(140){8}1/8-4[8:1],8,3-1[8:1]/8,3,3/5-7[8:1],5,1-5[8:1]/5,1,E"
    hits = _hits(body, "夹键拍滑")
    assert hits and hits[0].detail["n_clamp_units"] >= 2


def test_夹键拍滑_负例_把夹键抽掉就退回连续拍滑():
    body = "(140){8}1/8-4[8:1],,3-1[8:1]/8,,3/5-7[8:1],,1-5[8:1]/5,,E"
    cfgs = _cfgs(body)
    assert "夹键拍滑" not in cfgs
    assert "连续拍滑" in cfgs


def test_同起点拍滑_正例_两手交替各领一根():
    body = "(137){8}3,5-3[8:1],5-7[8:1],5-2[8:1],5-8[8:1],5-1[8:1],,,E"
    hits = _hits(body, "同起点拍滑")
    assert hits
    d = hits[0].detail
    assert d["head_key"] == 5 and d["n_roots"] >= 4 and d["alt_rate"] >= 0.75


def test_三叉戟_正例_恰三根全给一只手():
    # LANCE 型：三根同头 + 另一只手在对侧打自己的线
    # v0.2：尾巴那个 tap 改写在左手线上——原来的 `3` 与第三根的启动拍同刻，
    # 舒适区模型下（知识 064）左手拍 3 是出张，分配器会把第三根改派左手，
    # 与官谱 LANCE m072–075 的实际形态不符（那里三根始终归一只手，见验证报告 v0.2）
    body = "(156){8}1-5[8:1]/8,,1^3[8:1]/7,,1-4[8:1]/6,,7,,E"
    hits = _hits(body, "三叉戟")
    assert hits
    d = hits[0].detail
    assert d["n_roots"] == 3 and d["canonical"] is True


def test_三叉戟_负例_四根退回同起点拍滑():
    body = "(137){8}3,5-3[8:1],5-7[8:1],5-2[8:1],5-8[8:1],5-1[8:1],,,E"
    assert "三叉戟" not in _cfgs(body)


def test_挥手段_正例_固定手甩两边另一手独立线():
    body = ("(155){4}2^4[8:1]/7,2^8[8:1],2^4[8:1]/5,2^8[8:1],"
            "2^4[8:1]/7,2^8[8:1],2^4[8:1]/5,2^8[8:1],E")
    hits = _hits(body, "挥手段")
    assert hits
    d = hits[0].detail
    assert d["head_key"] == 2 and d["n_roots"] >= 3 and d["n_other_tasks"] >= 3


def test_鼓动段_正例_六根以上():
    # 鼓動 m043 型：一小节 8 根、头在 1 与 8 之间逐根对调
    body = ("(121){8}1-4[8:1],8-5[8:1],1-3[8:1],8-4[8:1],"
            "1-5[8:1],8-6[8:1],1-4[8:1],8-3[8:1],E")
    hits = _hits(body, "鼓动段")
    assert hits and hits[0].detail["n_units"] >= 6


def test_鼓动段_负例_三根不够():
    body = "(121){8}1-6[8:1],1,2-5[8:1],2,8-3[8:1],8,E"
    assert "鼓动段" not in _cfgs(body)


# ---------------------------------------------------------------------------
# 第二期：033 / 035 / 036 / 037 / 043 / 039–041
# ---------------------------------------------------------------------------


def test_死镰段_正例_一手走圈一手划弧():
    body = ("(182){8}7^4[8:1],8,1,2^7[8:1],3,4,5^2[8:1],6,"
            "7,8^5[8:1],1,2,3^8[8:1],4,5,6,E")
    hits = _hits(body, "死镰段")
    assert hits
    d = hits[0].detail
    assert d["walk_len"] >= 8 and d["n_slides"] >= 2


def test_死镰段_负例_只有走圈没有星星():
    body = "(182){8}" + ",".join(str((i % 8) + 1) for i in range(16)) + ",E"
    assert "死镰段" not in _cfgs(body)


def test_如龙段_正例_细胞逐格平移():
    body = "(145){16}2/4,3,4,3/5,4,5,4/6,5,6,5/7,6,7,E"
    hits = _hits(body, "如龙段")
    assert hits and hits[0].detail["n_cells"] >= 2


def test_如龙段_负例_没有双押的同向扫不是如龙():
    assert "如龙段" not in _cfgs("(145){16}2,3,4,3,4,5,4,5,6,5,6,7,E")


def test_二连扫_正例_键集不封闭():
    hits = _hits("(180){12}8,7,6,5,4,3,2,1,8,7,6,5,E", "二连扫")
    assert hits and hits[0].detail["n_keys"] > 4


def test_二连扫_负例_封闭四键是大宇宙():
    body = "(240){16}1,7,2,8,1,7,2,8,1,7,2,8,E"
    cfgs = _cfgs(body)
    assert "二连扫" not in cfgs and "大宇宙" in cfgs


def test_反手_正例_同半圈且有一只手出张_知识037与064():
    # 200BPM {16} 的 2↔3 快速交替：一手必须出张到 {2,3} 才打得出来 = 037「出张的极端形」
    hits = _hits("(200){16}2,3,2,3,2,3,2,3,E", "反手")
    assert hits and hits[0].detail["length"] >= 4


def test_反手_负例_同半圈但两手都在舒适区_知识064():
    # v0.1 把"两手同在右半圈"一律算反手；064 下 L 打 1/4、R 打 2/3 各自舒适，不是反手
    assert "反手" not in _cfgs("(150){8}1,2,3,4,1,2,3,4,E")


def _cfgs_with_hands(body: str, hp: dict) -> set[str]:
    """用一套指定的分配器参数算手序，再跑检测器（只为把手序钉死）。"""
    res = _parse(body)
    ha = H.assign(res, {**H.PARAMS, **hp})
    return {h.config for h in ch.detect_all(res, None, ha, None)}


def test_反手_负例_交叉但落点全在共享键_v0_3收紧crossed支():
    # 报告 §5.5：`detect_backhand` 的 064 收紧原来**只打在"同半圈"那一支**，
    # `crossed` 支仍按 006 中线 → 388 官谱报出的 266 段里 217 段（81.6%）没有任何
    # 出张落点，全是 `4,5,4,5` / `1,8,1,8` 这种两手在 8–4 / 8–1 轴两侧交替。
    # 知识 064：L→1/4 与 R→5/8 都在共享区，既不是出张、更不是"出张的极端形"。
    for body in ("(200){16}4,5,4,5,4,5,4,5,E", "(200){16}1,8,1,8,1,8,1,8,E"):
        assert "反手" not in _cfgs_with_hands(body, {"force_first_hand": "L"})


def test_反手_正例_交叉且有一只手出张_v0_3收紧后仍命中():
    # 同样是交叉（L 在 1234、R 在 5678），但 R 真的落到 6 = 出张 → 037「出张的极端形」
    # （`w_chuzhang` 调 0 只是为了让分配器**愿意**给出这套交叉手序——官谱里它由上下文
    #   逼出，合成片段逼不出来；`is_chuzhang` 的判定与这个权重无关。）
    cfg = _cfgs_with_hands("(200){16}4,6,4,6,4,6,4,6,E",
                           {"force_first_hand": "L", "w_chuzhang": 0.0})
    assert "反手" in cfg


def test_反手_负例_正常分页交互():
    assert "反手" not in _cfgs("(173){16}1,8,2,7,1,8,2,7,E")


def test_反手_负例_两手敲同一个键是纵连的拆():
    assert "反手" not in _cfgs("(240){16}" + "1," * 10 + "E")


def test_出张_正例_按064键位定义():
    # 左手被 Hold 钉在 7（右手只好去打 6 = 出张），右手位上还有一个 2 落到左手
    hits = _hits("(120){8}7h[2:1]/2,6,6,6,6,E", "出张")
    assert hits and hits[0].detail["n_cross"] >= 2


def test_出张_负例_共享键不算出张_知识064():
    # 同一形态换成共享键 5/8 —— v0.1 的 page_depth≥1 会误判，064 口径下是 0
    assert "出张" not in _cfgs("(120){8}7h[2:1]/2,5,8,5,8,E")


def test_出张_负例_落点全在共享区():
    # 没有手被钉住**也**没有任何出张落点（L 打 8/7、R 打 1/2 全在舒适区）
    assert "出张" not in _cfgs("(150){16}1,8,2,7,1,8,2,7,E")


def test_出张_正例_纯单点出张_v0_3新增成段方式():
    # 报告 §4.2 类 6：388 官谱 5 739 个出张落点里**纯单点出张 30.2% 是最大的一类**
    # ——没有任何手被长条/星星占住，就是把这一下交给了对侧的手。
    hits = _hits("(200){16}2,3,2,3,2,3,2,3,E", "出张")
    assert hits and any(h.detail.get("mode") == "纯单点出张" for h in hits)
    h = [x for x in hits if x.detail.get("mode") == "纯单点出张"][0]
    assert h.detail["n_cross"] >= ch.PARAMS["chuzhang_min_cross"]


def test_出张_两种成段方式互斥_被占手那支仍标被占手():
    hits = _hits("(120){8}7h[2:1]/2,6,6,6,6,E", "出张")
    assert hits and all(h.detail.get("mode") == "被占手逼出" for h in hits)


def test_出张_负例_纯单点但落点不够成段():
    # 只有一个出张落点：`chuzhang_min_cross` 是"几个落点才值得单独记一段"的计数口径
    assert "出张" not in _cfgs("(150){8}1,8,7,6,5,4,8,1,E")


def test_纵连_短纵连标单手可行_知识019补充():
    h = _hits("(185){16}1,1,1,,,,,,E", "纵连")
    assert h and h[0].detail["single_hand_ok"] is True
    h2 = _hits("(185){16}1,1,1,1,1,1,1,1,E", "长纵连")
    assert h2 and h2[0].detail["single_hand_ok"] is False


def test_2加1_正例():
    hits = _hits("(216){12}8,8,1,8,8,1,8,8,1,8,8,1,E", "2+1")
    assert hits and hits[0].detail["n_seq"] == [2, 2, 2, 2][:len(hits[0].detail["n_seq"])]


def test_3加1_正例():
    hits = _hits("(175){8}7,7,7,6,7,7,7,6,E", "3+1")
    assert hits and set(hits[0].detail["n_seq"]) == {3}


def test_N加1_正例_n在段内变化():
    hits = _hits("(191){16}4,4,4,4,4,5,6,6,6,6,7,8,8,8,8,1,E", "N+1")
    assert hits and len(set(hits[0].detail["n_seq"])) >= 2


def test_N加1_负例_n不变就是2加1或3加1():
    assert "N+1" not in _cfgs("(216){12}8,8,1,8,8,1,8,8,1,8,8,1,E")


# ---------------------------------------------------------------------------
# 键位版 7 类假阳性（`sampling-mode-study.md` §2.2）——手序版也不得误报
# ---------------------------------------------------------------------------


def test_键位版假阳性1_单向音阶不是三角():
    assert "三角交互" not in _cfgs("(150){16}7h[16:1],6,5,4h[16:1],3,2,8,1,E")


def test_键位版假阳性2_星星链不是三角():
    body = "(150){8}1-5[8:1],6,7,8-4[8:1],3,2,1-5[8:1],6,7,8-4[8:1],3,2,E"
    assert "三角交互" not in _cfgs(body)


def test_键位版假阳性3_双手同刻同任务不是错位():
    assert "错位" not in _cfgs("(150){4}4-1[4:1]/5-8[4:1],4/5,,,E")


def test_键位版假阳性4_滑条加tap交替链不是散点():
    body = "(150){8}1-4[8:1],,1,,7-4[8:1],,7,,1-4[8:1],,1,,7-4[8:1],,7,,E"
    assert "散点" not in _cfgs(body)


def test_键位版假阳性5_星星头加同键tap不是子弹():
    assert "子弹" not in _cfgs("(150){8}2-5[8:1],,2,,6,4,E")


def test_键位版假阳性6_密度变化不是跳拍():
    # 8 分块接 16 分块（倍数 [2,2,2,1,1,2,2,2]）= 一处切换，不是反复插空
    assert "跳拍" not in _cfgs("(150){8}1,8,1,{16}8,1,{8}8,1,8,E")


def test_键位版假阳性7_星星串不算普通交互或纵连():
    body = "(150){4}8b-6[8:1],8b-3[8:1],8b-4[8:1],8b-2[8:1],8b-6[8:1],8b-3[8:1],E"
    cfgs = _cfgs(body)
    assert "纵连" not in cfgs and "普通交互" not in cfgs


# ---------------------------------------------------------------------------
# 两个分数并存
# ---------------------------------------------------------------------------


def test_两个分数并存且不合成():
    res = _parse("(150){16}1,8,1,8,1,8,1,8,{8}1/2,2/3,3/4,4/5,E")
    rows = ch.bar_scores(res)
    assert rows
    live = [r for r in rows if r.n_notes > 0]
    assert all(0.0 <= r.physical_hardness <= 1.0 for r in live)
    assert all(0.0 <= r.hand_difficulty <= 1.0 for r in live)
    d = live[0].to_dict()
    assert "physical_hardness" in d and "hand_difficulty" in d
    # 不得出现任何合成列（用户已撤回 (D+H)/2 排序）
    assert not any("combined" in k or "overall" in k for k in d)


def test_逐小节输出包含手序指标():
    res = _parse("(200){8}8-5[8:1],2,8,1,4-7[8:1],2,4,3,E")
    rows = ch.bar_scores(res)
    r = [x for x in rows if x.n_notes > 0][0]
    for k in ("alt_rate", "max_same_run", "cross_count", "chuzhang",
              "scrape", "n_switch"):
        assert hasattr(r, k)


def test_chart_summary_口径():
    res = _parse("(167){4}1-6[8:1],6-2[8:1],2-5[8:1],5-1[8:1],E")
    d = ch.chart_summary(res)
    assert d["n_slides"] == 4 and d["n_configs"] >= 1
    assert "一笔画" in d["configs"]


def test_检测器可注入_bar_config_table():
    from chart_analysis.config_profiles import bar_config_table
    res = _parse("(167){4}1-6[8:1],6-2[8:1],2-5[8:1],5-1[8:1],E")
    table = bar_config_table(res, None, detector=ch.detect_all)
    assert table and any("一笔画" in v for v in table.values())


def test_每个配置都写明了条目依据():
    res = _parse("(150){16}1,8,1,8,1,8,1,8,E")
    for h in ch.detect_all(res):
        assert h.source, f"{h.config} 没写依据条目号"
        assert h.config in ch.IMPLEMENTED


def test_参数可整体替换():
    body = "(167){4}1-6[8:1],6-2[8:1],2-5[8:1],5-1[8:1],E"
    assert "一笔画" in _cfgs(body)
    assert "一笔画" not in _cfgs(body, {"onestroke_min_len": 9})


def test_待用户对齐的参数已清空_用户20260920():
    """agent 自造的阈值/分档不得作为问题抛给用户（用户 2026-09-20 五条批评）。

    `PENDING_USER_PARAMS` 清空；原来的量化项全部转成内部操作化，登记在
    `RESOLVED_USER_PARAMS` 里备查。
    """
    assert ch.PENDING_USER_PARAMS == ()
    assert "chuzhang_min_cross" in ch.RESOLVED_USER_PARAMS
    for k in ch.PENDING_USER_PARAMS:
        assert k in ch.PARAMS, k
