#!/usr/bin/env python3
"""`tools/chart_check`（谱面检查器）的单元测试。

**素材全部为自写的合成 simai 片段**，不复制官方谱原文。
每条规则至少一个正例；容易误报的规则（分音标记连写、`<`/`>` 同键绕圈、
官谱也会踩的硬级无理）另配**负例**，钉住"不该报"。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_chart_check.py -q``
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.chart_check import check_chart  # noqa: E402
from tools.chart_check.maidata import parse_maidata  # noqa: E402
from tools.chart_check.report import render_json, render_markdown  # noqa: E402

HEAD = ("&title=t\n&artist=a\n&des=aimai-chartmaker trial\n"
        "&first=0\n&lv_5=13.5\n&inote_5=\n")


def _check(body: str, **kw):
    md = parse_maidata(HEAD + body, path="<t>")
    return check_chart(md, **kw)


def codes(rep, layer: str | None = None, level: str | None = None) -> set[str]:
    return {i.code for i in rep.issues
            if (layer is None or i.layer == layer)
            and (level is None or i.level == level)}


# ---------------------------------------------------------------------------
# 干净底稿（所有"不该报"的负例都以它为基线）
# ---------------------------------------------------------------------------

CLEAN = "(160){8}\n1,2,3,4,5,6,7,8,\n{16}1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,\nE\n"


def test_clean_chart_has_no_syntax_error():
    rep = _check(CLEAN)
    assert codes(rep, "语法", "错误") == set(), codes(rep, "语法", "错误")
    assert rep.ok


def test_division_change_between_notes_is_not_double_directive():
    """`{8}…,{16}…` 是官谱每首都在写的写法，**不是"连写两个分音标记"**。"""
    rep = _check(CLEAN)
    assert "SYN-DOUBLE-DIRECTIVE" not in codes(rep)


def test_adjacent_double_division_is_reported():
    rep = _check("(160){8}{16}1,2,3,4,E\n")
    assert "SYN-DOUBLE-DIRECTIVE" in codes(rep)


def test_division_then_bpm_order_reported():
    rep = _check("{8}(160)1,2,3,4,E\n")
    assert "SYN-ORDER" in codes(rep)


# ---------------------------------------------------------------------------
# 语法层：安全子集禁用清单（docs/simai-syntax.md §6-3）
# ---------------------------------------------------------------------------


def test_missing_end_marker():
    rep = _check("(160){8}1,2,3,4,\n")
    assert "SYN-NO-E" in codes(rep)


def test_lowercase_end_marker():
    rep = _check("(160){8}1,2,3,4,e\n")
    assert "SYN-LOWER-E" in codes(rep)


def test_single_pipe_is_error():
    rep = _check("(160){8}1,2,3,4,E\n| 这是错的注释\n")
    assert "SYN-PIPE" in codes(rep)


def test_double_pipe_comment_is_ok():
    rep = _check("(160){8}1,2,3,4,E\n|| 这是对的注释\n")
    assert "SYN-PIPE" not in codes(rep)


def test_fullwidth_digit_is_error():
    rep = _check("(160){8}1,２,3,4,E\n")
    assert "SYN-FULLWIDTH" in codes(rep)


def test_seconds_slot_banned():
    rep = _check("(160){#0.35}1,2,E\n")
    assert "SYN-SECONDS-SLOT" in codes(rep)


def test_non_384_divisor_is_warning():
    rep = _check("(160){5}1,2,3,4,5,E\n")
    assert "SYN-DIV-384" in codes(rep, level="警告")


def test_tap_with_duration_is_error():
    rep = _check("(160){8}1[8:1],2,E\n")
    assert "SYN-TAP-DURATION" in codes(rep)


def test_hold_double_hash_duration_banned():
    rep = _check("(160){8}1h[##2.5],2,E\n")
    assert "SYN-HOLD-DUR" in codes(rep)


def test_slide_single_value_duration_banned():
    rep = _check("(160){8}1-5[2],2,E\n")
    assert "SYN-SLIDE-DUR" in codes(rep)


def test_same_head_slide_with_slash_banned():
    rep = _check("(160){8}1-4[8:1]/1-6[8:1],2,E\n")
    assert "SYN-SAMEHEAD-SLASH" in codes(rep)


def test_empty_each_member_is_error():
    rep = _check("(160){8}1//2,3,E\n")
    assert "SYN-EMPTY-MEMBER" in codes(rep)


def test_meta_unescaped_ampersand():
    md = parse_maidata("&title=A&B\n&artist=a\n&des=d\n&first=0\n&inote_5=\n"
                       + CLEAN, path="<t>")
    rep = check_chart(md)
    assert "SYN-META-ESCAPE" in codes(rep)


def test_missing_meta_is_warning():
    md = parse_maidata("&inote_5=\n" + CLEAN, path="<t>")
    rep = check_chart(md)
    assert "SYN-META-MISSING" in codes(rep, level="警告")


# ---------------------------------------------------------------------------
# 语法层：ST 黑名单（docs/st-chart-elements.md §三）
# ---------------------------------------------------------------------------


def test_touch_is_st_blacklisted():
    rep = _check("(160){8}C,1,2,E\n")
    assert "ST-TOUCH" in codes(rep)


def test_ex_modifier_is_st_blacklisted():
    rep = _check("(160){8}1x,2,E\n")
    assert "ST-EX" in codes(rep)


def test_break_slide_is_st_blacklisted():
    rep = _check("(160){8}1-5[8:1]b,2,E\n")
    assert "ST-BREAKSLIDE" in codes(rep)


def test_star_head_break_is_allowed():
    """星星头绝赞 `1b-5[8:1]` 是 PiNK 要素，**ST 合法**，不该报。"""
    rep = _check("(160){8}1b-5[8:1],2,3,4,E\n")
    assert "ST-BREAKSLIDE" not in codes(rep)
    assert codes(rep, "语法", "错误") == set()


def test_chain_slide_is_st_blacklisted():
    rep = _check("(160){8}1-4q7-2[1:2],2,E\n")
    assert "ST-CHAIN" in codes(rep)


def test_same_head_star_is_allowed():
    """同头星星 `*` 是 PiNK 要素，ST 合法。"""
    rep = _check("(160){8}1-4[8:1]*-6[8:1],2,3,4,E\n")
    assert codes(rep, "语法", "错误") == set(), codes(rep, "语法", "错误")


def test_pseudo_each_backtick_blacklisted():
    rep = _check("(160){8}1`2,3,E\n")
    assert "ST-PSEUDOEACH" in codes(rep)


def test_star_tap_dollar_blacklisted():
    rep = _check("(160){8}1$,2,E\n")
    assert "ST-STARTAP" in codes(rep)


def test_restore_at_blacklisted():
    rep = _check("(160){8}1@-5[8:1],2,E\n")
    assert "ST-RESTORE" in codes(rep)


# ---------------------------------------------------------------------------
# 语法层：slide 端点几何（docs/simai-error-checking.md §2）
# ---------------------------------------------------------------------------


def test_straight_slide_too_close():
    rep = _check("(160){8}1-2[8:1],3,E\n")
    assert "SYN-SLIDE-GEOM" in codes(rep, level="错误")


def test_arc_slide_to_opposite_is_error():
    rep = _check("(160){8}1^5[8:1],3,E\n")
    assert "SYN-SLIDE-GEOM" in codes(rep, level="错误")


def test_lightning_slide_must_be_opposite():
    rep = _check("(160){8}1s3[8:1],4,E\n")
    assert "SYN-SLIDE-GEOM" in codes(rep, level="错误")


def test_v_slide_pivot_rule():
    rep = _check("(160){8}1V26[8:1],4,E\n")
    assert "SYN-SLIDE-GEOM" in codes(rep, level="错误")


def test_v_slide_valid():
    rep = _check("(160){8}1V36[8:1],4,5,6,E\n")
    assert codes(rep, "语法", "错误") == set(), codes(rep, "语法", "错误")


def test_full_circle_arc_is_only_a_hint():
    """`3<3` 绕一整圈：官方无硬限制、官谱写过 → 只给提示，不算错误。"""
    rep = _check("(160){8}3<3[8:3],5,6,7,E\n")
    assert "SYN-SLIDE-GEOM" in codes(rep, level="提示")
    assert "SYN-SLIDE-GEOM" not in codes(rep, level="错误")


# ---------------------------------------------------------------------------
# 手序层
# ---------------------------------------------------------------------------


def test_multi_press_is_absolute_muri():
    """同刻三个 tap = 需要三只手（知识 008）。"""
    rep = _check("(160){8}1/3/5,2,4,6,E\n")
    assert "PLAY-MULTI" in codes(rep, "手序", "错误")
    assert not rep.ok


def test_normal_alternation_has_no_muri():
    rep = _check(CLEAN)
    play = rep.layer("手序")
    assert play.stats["n_absolute"] == 0
    assert play.stats["n_infeasible"] == 0


def test_chuzhang_listed_when_hand_crosses():
    """一手被长条钉住 → 另一只手只能跨过去打，落点进出张清单（知识 064）。"""
    rep = _check("(160){8}1h[1:2]/6,7,6,7,2,3,2,3,E\n")
    play = rep.layer("手序")
    assert play.stats["n_chuzhang"] >= 1
    assert "PLAY-CHUZHANG" in codes(rep, "手序", "提示")


def test_side_double_is_listed_not_muri():
    """侧边双押只进复核清单，**不记无理**（知识 030/080/082）。"""
    rep = _check("(160){8}1/8,8/7,7/6,6/5,2/3,4,5,6,E\n")
    play = rep.layer("手序")
    assert play.stats["n_side_double"] >= 1
    assert "PLAY-SIDEDOUBLE" in codes(rep, "手序")
    assert play.stats["n_absolute"] == 0


def test_hard_muri_is_warning_not_error():
    """硬级无理记警告：388 官谱自己也有几十次（知识 077）。"""
    rep = _check("(160){8}1h[8:4],,,,3/7,2,4,6,E\n")
    play = rep.layer("手序")
    assert play.stats["n_hard"] >= 1
    assert "PLAY-HOLDTAIL" in codes(rep, "手序", "警告")
    assert "PLAY-HOLDTAIL" not in codes(rep, "手序", "错误")


# ---------------------------------------------------------------------------
# 配置层 / 密度层
# ---------------------------------------------------------------------------


def test_config_layer_reports_composition():
    rep = _check(CLEAN)
    assert "CFG-COMPOSITION" in codes(rep, "配置")
    assert rep.layer("配置").stats["n_config_kinds"] >= 1


def test_density_total_against_level_004():
    rep = _check(CLEAN, level=13.5)
    d = rep.layer("密度").stats
    assert d["total_notes"] == 24
    assert d["level_bucket"] == 13.5
    # 24 个 note 远低于定数 13.5 档的 p10（492，知识 004）
    assert "DEN-TOTAL-LOW" in codes(rep, "密度", "警告")


def test_density_curve_and_ending_present():
    body = "(160){8}" + "".join("1,2,3,4,5,6,7,8," for _ in range(12)) + "E\n"
    rep = _check(body)
    d = rep.layer("密度").stats
    assert len(d["curve"]) == 12
    assert d["ending"] is not None


# ---------------------------------------------------------------------------
# 采音层
# ---------------------------------------------------------------------------


def test_sampling_skipped_without_analysis():
    rep = _check(CLEAN)
    assert rep.layer("采音").skipped


def test_sampling_reads_grid_patterns():
    """合成一份最小 `song_analysis.json`：骨架轨给 8 个候选，谱面全踩。"""
    analysis = {
        "structure": {"segments": [
            {"start_bar": 1, "end_bar": 1, "function": "chorus",
             "label_ja": "サビ", "skeleton_stem": "drum", "accent_stems": []}]},
        "bars": [{"bar": 1, "start_sec": 0.0, "bpm": 160.0, "division": 8,
                  "patterns": {"drum": "xxxxxxxx", "vocal": "........",
                               "bass": "........", "hook": "........"}}],
    }
    rep = _check("(160){8}1,2,3,4,5,6,7,8,E\n", analysis=analysis)
    smp = rep.layer("采音")
    assert not smp.skipped
    assert smp.stats["bars"][0]["coverage"] == 1.0
    assert smp.stats["bars"][0]["mode"] == "全踩"


def test_sampling_flags_empty_section():
    analysis = {
        "structure": {"segments": [
            {"start_bar": 1, "end_bar": 1, "function": "chorus",
             "label_ja": "サビ", "skeleton_stem": "drum", "accent_stems": []}]},
        "bars": [{"bar": 1, "start_sec": 0.0, "bpm": 160.0, "division": 16,
                  "patterns": {"drum": "xxxxxxxxxxxxxxxx", "vocal": "." * 16,
                               "bass": "." * 16, "hook": "." * 16}}],
    }
    rep = _check("(160){8}1,,,,,,,,E\n", analysis=analysis)
    assert "SMP-EMPTY-SECTION" in codes(rep, "采音", "警告")


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def test_report_renders_markdown_and_json():
    rep = _check(CLEAN, level=13.5)
    md = render_markdown(rep)
    assert "谱面检查报告" in md
    assert "外部三检待接" in md
    data = json.loads(render_json(rep))
    assert data["external_lint"].startswith("未接")
    assert {L["layer"] for L in data["layers"]} == {
        "语法", "手序", "配置", "密度", "采音", "深度", "外部规则"}


# ---------------------------------------------------------------------------
# 密度层：note 种类配比对照（知识 088）——补 `e2e-trial-01` 缺口 G1
# ---------------------------------------------------------------------------

#: 只有 tap 的一张谱：slide / hold 占比都是 0，必然低于任何定数档的 Q1
_TAP_ONLY = "(160){8}" + "".join("1,2,3,4,5,6,7,8," for _ in range(12)) + "E\n"
#: 同样长度，但每拍一条星星 + 一条长条（note 种类配比对照的负例）
_MIXED = ("(160){8}" + "".join("1-5[8:1],2,3h[8:1],4,5-1[8:1],6,7h[8:1],8,"
                               for _ in range(12)) + "E\n")
#: 头尾相连的星星链——配置识别器把它认成**一笔画**（星星族，知识 057）
_ONESTROKE = ("(160){8}" + "".join("1-3[8:1],3-5[8:1],5-7[8:1],7-1[8:1],"
                                   "1-3[8:1],3-5[8:1],5-7[8:1],7-1[8:1],"
                                   for _ in range(12)) + "E\n")


def test_density_reports_note_mix_against_level():
    rep = _check(_TAP_ONLY, level=13.5)
    d = rep.layer("密度").stats
    assert "DEN-NOTEMIX" in codes(rep, "密度")
    assert d["note_mix"]["slide"] == 0.0 and d["note_mix"]["hold"] == 0.0
    assert d["note_mix_official"]["slide"][0] > 0.0


def test_density_flags_slide_and_hold_below_q1():
    rep = _check(_TAP_ONLY, level=13.5)
    c = codes(rep, "密度")
    assert "DEN-NOTEMIX-SLIDE-LOW" in c
    assert "DEN-NOTEMIX-HOLD-LOW" in c
    # 只是提示，**不是错误或警告**
    assert "DEN-NOTEMIX-SLIDE-LOW" not in codes(rep, "密度", "错误")
    assert "DEN-NOTEMIX-SLIDE-LOW" not in codes(rep, "密度", "警告")


def test_density_note_mix_ok_is_not_flagged():
    """写了星星与长条的谱不该被标「低于 Q1」（负例）。"""
    rep = _check(_MIXED, level=13.5)
    c = codes(rep, "密度")
    assert "DEN-NOTEMIX-SLIDE-LOW" not in c
    assert "DEN-NOTEMIX-HOLD-LOW" not in c


def test_density_section_note_mix_needs_analysis():
    """没有 `song_analysis.json` 时不报段落级对照；有了才报。"""
    assert "DEN-NOTEMIX-SECTION" not in codes(_check(_TAP_ONLY, level=13.5))
    analysis = {"structure": {"segments": [
        {"start_bar": 1, "end_bar": 12, "function": "chorus", "label_ja": "サビ",
         "skeleton_stem": "drum", "accent_stems": []}]}, "bars": []}
    rep = _check(_TAP_ONLY, level=13.5, analysis=analysis)
    assert "DEN-NOTEMIX-SECTION" in codes(rep, "密度")
    sec = rep.layer("密度").stats["sections"]
    assert sec and sec[0]["section"] == "サビ" and sec[0]["slide_ratio"] == 0.0


def test_density_section_unknown_label_is_listed_but_not_compared():
    """段落标签不在 160 首的表里时只记数、不对照（负例）。"""
    analysis = {"structure": {"segments": [
        {"start_bar": 1, "end_bar": 12, "function": "??", "label_ja": "Ｘメロ",
         "skeleton_stem": "drum", "accent_stems": []}]}, "bars": []}
    rep = _check(_TAP_ONLY, level=13.5, analysis=analysis)
    assert "DEN-NOTEMIX-SECTION" not in codes(rep, "密度")
    assert rep.layer("密度").stats["sections"][0]["section"] == "Ｘメロ"


# ---------------------------------------------------------------------------
# 深度层：「写得太浅？」复核清单（知识 093）——补 `e2e-trial-01` 缺口 G6
# ---------------------------------------------------------------------------


def test_depth_layer_needs_level():
    rep = _check(_TAP_ONLY)           # &lv_5=13.5 会被粗取到，所以显式清空
    md = parse_maidata(HEAD.replace("&lv_5=13.5", "&lv_5=") + _TAP_ONLY, path="<t>")
    from tools.chart_check import check_chart as _cc
    rep2 = _cc(md)
    assert rep2.layer("深度").skipped
    assert not rep.layer("深度").skipped


def test_depth_flags_shallow_chart():
    """只有 tap、主料里没有星星族 → 至少两条旗标 → DEP-SHALLOW。"""
    rep = _check(_TAP_ONLY, level=13.5)
    dep = rep.layer("深度")
    assert "DEP-SHALLOW" in codes(rep, "深度")
    assert len(dep.stats["flags"]) >= 2
    assert dep.stats["star_main"] == []
    # **只提示，不判错**
    assert dep.count("错误") == 0 and dep.count("警告") == 0


def test_depth_hand_hardness_is_compared_not_scored():
    rep = _check(_TAP_ONLY, level=13.5)
    dep = rep.layer("深度")
    assert "DEP-HAND-HARDNESS" in codes(rep, "深度")
    assert dep.stats["hand_hardness_official"] == (0.114, 0.093, 0.134)
    msg = next(i.message for i in dep.issues if i.code == "DEP-HAND-HARDNESS")
    # 知识 078 / 064 / 066 的免责必须印出来，防止把它当目标优化
    assert "不是难度代理" in msg and "不得当作目标去优化" in msg


def test_depth_star_family_main_is_recognised():
    """成段的一笔画 → 主料里认得出星星族（负例：不该报"一类都没有"）。"""
    rep = _check(_ONESTROKE, level=13.5)
    dep = rep.layer("深度")
    assert "一笔画" in dep.stats["star_main"]
    assert "主料里一类星星族都没有" not in dep.stats["flags"]


def test_depth_single_flag_is_not_shallow():
    """只踩到一条差距时只给 `DEP-SHALLOW-1`，不给「写得太浅」清单（负例）。"""
    rep = _check(_ONESTROKE, level=13.5)
    c = codes(rep, "深度")
    assert "DEP-SHALLOW-1" in c and "DEP-SHALLOW" not in c


# ---------------------------------------------------------------------------
# 外部规则层（MiaCode / MaiMuriDX 无理检测）
#
# ⚠️ **期望值不是我编的**：下面每条的命中数与 gap 都用上游 Starrah/MaiMuriDX
# 的 `cli.py`（静态检查那一段）跑过同一段谱文本对齐；`data/slide_geometry.json`
# 本身也由它的 `SlideInfo` 导出。
# ---------------------------------------------------------------------------

#: MURI_DETECTION_SPEC §7.4 的官方样例：slide 头提前判掉后面的 tap
_SPEC_HEADTAP = "(240){16}\n8>3[4:1],,,,,\n8,\nE\n"
#: 同 §9：`111` 同刻三叠键只留一条
_SPEC_OVERLAP = "(240){16}\n111,\nE\n"


def test_muri_layer_reproduces_spec_headtap_example():
    """MURI_DETECTION_SPEC §7.4 的样例：`8>3[4:1]` 之后的 `8` 被蹭。"""
    rep = _check(_SPEC_HEADTAP)
    lr = rep.layer("外部规则")
    kinds = [h["kind"] for h in lr.stats["hits"]]
    assert kinds == ["SlideHeadTap"], lr.stats["hits"]
    # MaiMuriDX 报 -62 ms（判定区间 200 ms 内）
    assert abs(lr.stats["hits"][0]["gap_ms"] - 62) < 1.5


def test_muri_layer_dedupes_overlap_per_moment():
    """同一时刻的叠键只留一条（MURI_DETECTION_SPEC §9）。"""
    rep = _check(_SPEC_OVERLAP)
    lr = rep.layer("外部规则")
    assert lr.stats["counts"] == {"Overlap": 1}, lr.stats["hits"]
    assert "MURI-OVERLAP" in codes(rep, "外部规则", "错误")


def test_muri_layer_start_tap_on_shoot_is_not_a_hit():
    """拍滑：启动拍上的 tap 恰在启动时刻 → gap 0，**不算外键**（负例）。

    `1-5[8:1],,1,` 在 205 BPM `{8}` 下，`1` 正好落在启动拍（marker + 1 拍）。
    """
    rep = _check("(205)\n{8}1-5[8:1],,1,,,,,,\nE\n")
    lr = rep.layer("外部规则")
    assert lr.stats["hits"] == []
    assert "MURI-OK" in codes(rep, "外部规则")


def test_muri_layer_catches_relay_head_on_previous_tail():
    """205 BPM 两拍接力的一笔画：下一根的头落在上一根尾区 → 撞尾。

    这正是 test-01 修正前 18 处命中里最典型的一族（note 066）。
    """
    rep = _check("(205)\n{8}2^5[8:1],,2,7,5^8[8:1],,5,1,\nE\n")
    lr = rep.layer("外部规则")
    kinds = [h["kind"] for h in lr.stats["hits"]]
    assert "TapOnSlide" in kinds
    hit = next(h for h in lr.stats["hits"] if h["kind"] == "TapOnSlide")
    assert hit["cause"] == "2^5" and abs(hit["gap_ms"] - 161) < 1.5


def test_muri_layer_relay_one_key_short_is_clean():
    """把第一根缩短一键（`2^5` → `2^4`）后同一小节归零（负例，note 066 的改法）。"""
    rep = _check("(205)\n{8}2^4[8:1],,2,7,5^8[8:1],,5,1,\nE\n")
    assert rep.layer("外部规则").stats["hits"] == []


def test_muri_layer_total_line_carries_official_baseline():
    """有命中时要给一条汇总，并把官谱基线印出来（防止被当成"有一条就写坏了"）。"""
    rep = _check(_SPEC_HEADTAP)
    msg = next(i.message for i in rep.issues if i.code == "MURI-TOTAL")
    assert "中位 3" in msg and "验收目标是 0" in msg


def test_muri_layer_geometry_table_covers_all_shapes():
    """几何表要认得 12 种形状 + wifi；认不出的形状必须显式报出来。"""
    from tools.chart_check.murilayer import load_geometry
    geo = load_geometry()
    for key in ("1-4", "2^5", "1v4", "1V37", "2p4", "1qq5", "8<5", "3>7",
                "1s5", "1z5", "1pp5", "6q3"):
        assert key in geo["slides"], key
    assert "1w5" in geo["wifi"]


def test_meta_escape_only_applies_to_text_metas():
    """`&lv_5=13+` 的半角 `+` 合法（simai-syntax §2.2），`&title=` 里的不合法。"""
    ok = parse_maidata("&title=t\n&artist=a\n&des=d\n&first=0\n"
                       "&lv_5=13+\n&inote_5=\n(205)\n{8}1,,,,,,,,\nE\n", path="<t>")
    assert "SYN-META-ESCAPE" not in codes(check_chart(ok))
    bad = parse_maidata("&title=a+b\n&artist=a\n&des=d\n&first=0\n"
                        "&lv_5=13+\n&inote_5=\n(205)\n{8}1,,,,,,,,\nE\n", path="<t>")
    assert "SYN-META-ESCAPE" in codes(check_chart(bad))


def test_sampling_layer_shifts_analysis_clock_by_first():
    """`&first ≠ 0` 时采音层要把 `start_sec` 平移回谱面时钟（note 065 缺口 G16）。

    同一份谱 + 同一份 analysis，只把 `&first` 与 `start_sec` 同步挪 1.349 s，
    coverage 必须一模一样。
    """
    body = "(120)\n{4}1,2,3,4,\nE\n"
    analysis = {"bars": [{"bar": 0, "start_sec": 0.0, "bpm": 120,
                          "patterns": {"drum": "XXXX"}}],
                "structure": {"segments": [{"start_bar": 0, "end_bar": 0,
                                            "skeleton_stem": "drum",
                                            "function": "intro"}]}}
    base = check_chart(parse_maidata(HEAD + body, path="<t>"), analysis=analysis)
    shifted = json.loads(json.dumps(analysis))
    shifted["bars"][0]["start_sec"] = 1.349
    head2 = HEAD.replace("&first=0", "&first=1.349")
    moved = check_chart(parse_maidata(head2 + body, path="<t>"), analysis=shifted)
    assert (moved.layer("采音").stats["bars"][0]["coverage"]
            == base.layer("采音").stats["bars"][0]["coverage"] == 1.0)
    assert moved.layer("采音").stats["first_sec"] == 1.349


# ---------------------------------------------------------------------------
# 交付格式：Visual Maimai 吞分音（2026-09-20 实测，simai-error-checking §9.1）
# ---------------------------------------------------------------------------


def test_vm_comment_is_an_error():
    """`||` 行尾注释 = VM 兼容风险，交付版必须去掉（一条汇总错误）。"""
    rep = _check("(205)\n{8}1,2,3,4,5,6,7,8,  ||m001 测试\n{16}1,,2,,3,,4,,"
                 "5,,6,,7,,8,,\nE\n")
    assert "SYN-VM-COMMENT" in codes(rep, "语法", "错误")
    msg = next(i.message for i in rep.issues if i.code == "SYN-VM-COMMENT")
    assert "1 行 `||` 行尾注释" in msg and "1** 行的下一行以 `{` 开头" in msg


def test_no_comment_body_is_clean():
    """官方格式（一行一小节、行首 `{x}`、无注释）不该报（负例）。"""
    rep = _check("(205)\n{8}1,2,3,4,5,6,7,8,\n{16}1,,2,,3,,4,,5,,6,,7,,8,,\nE\n")
    assert "SYN-VM-COMMENT" not in codes(rep, "语法")


def test_vm_comment_counts_only_lines_before_a_divisor_line():
    """下一行不以 `{` 开头时仍然报（注释本身就是风险），但计数要分开。"""
    rep = _check("(205)\n{8}1,2,3,4,5,6,7,8,  ||a\n1,2,3,4,5,6,7,8,  ||b\nE\n")
    msg = next(i.message for i in rep.issues if i.code == "SYN-VM-COMMENT")
    assert "2 行 `||` 行尾注释" in msg and "0** 行的下一行以 `{` 开头" in msg


def test_sampling_layer_skips_when_the_grid_does_not_match():
    """谱面改成变速、analysis 还是旧的恒定 BPM → **跳过**，不报假的留白/采空音。"""
    analysis = {"bars": [{"bar": i + 1, "start_sec": i * 2.0, "bpm": 120,
                          "patterns": {"drum": "XXXX"}} for i in range(4)],
                "structure": {"segments": [{"start_bar": 0, "end_bar": 3,
                                            "skeleton_stem": "drum", "function": "intro"}]}}
    # 谱面是 240 BPM（每小节 1 s），analysis 是 120 BPM（每小节 2 s）→ 第 2 小节起就差 1 s
    body = "(240){4}1,2,3,4,\n1,2,3,4,\n1,2,3,4,\n1,2,3,4,\nE\n"
    rep = check_chart(parse_maidata(HEAD + body, path="<t>"), analysis=analysis)
    smp = rep.layer("采音")
    assert smp.skipped and "网格对不上" in smp.skipped
    assert smp.stats["grid_drift_sec"] > 0.05
    assert not [i for i in smp.issues if i.level in ("错误", "警告")]


def test_sampling_layer_still_runs_when_the_grid_matches():
    """网格一致时照常跑（负例：别把正常情况也跳掉）。"""
    analysis = {"bars": [{"bar": i + 1, "start_sec": i * 2.0, "bpm": 120,
                          "patterns": {"drum": "XXXX"}} for i in range(4)],
                "structure": {"segments": [{"start_bar": 0, "end_bar": 3,
                                            "skeleton_stem": "drum", "function": "intro"}]}}
    body = "(120){4}1,2,3,4,\n1,2,3,4,\n1,2,3,4,\n1,2,3,4,\nE\n"
    rep = check_chart(parse_maidata(HEAD + body, path="<t>"), analysis=analysis)
    assert not rep.layer("采音").skipped
    assert rep.layer("采音").stats["bars"][0]["coverage"] == 1.0
