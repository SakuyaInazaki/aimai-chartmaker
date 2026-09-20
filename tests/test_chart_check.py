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
    assert {L["layer"] for L in data["layers"]} == {"语法", "手序", "配置", "密度", "采音"}
