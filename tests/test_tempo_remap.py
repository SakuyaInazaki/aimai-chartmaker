#!/usr/bin/env python3
"""`tools/chart_analysis/tempo_remap.py`（变速重排：写 `(bpm)` / 校验小节起点秒）。

**素材全部为自写的合成片段**，不复制官方谱原文。
每条都钉「秒数必须精确相等」——变速重排的验收标准就是误差 0。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_tempo_remap.py -q``
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import pytest  # noqa: E402

from chart_analysis.tempo_remap import (  # noqa: E402
    emit, load_tempo_map, slot_positions, verify,
)

#: 三小节、每小节 8 个八分；第 2 小节整小节换速、并在小节中途再换一次
_TEMPO = {"first": 0.5,
          "bars": [{"bar": 1, "bpm": 120.0, "start_sec": 0.5},
                   {"bar": 2, "bpm": 240.0},
                   {"bar": 2, "beat": 2.0, "bpm": 60.0},
                   {"bar": 3, "bpm": 120.0}]}
_BODY = "{8}1,2,3,4,5,6,7,8,\n{8}1,2,3,4,5,6,7,8,\n{8}1,2,3,4,5,6,7,8,\nE\n"
_HEAD = "&title=t\n&artist=a\n&des=d\n&first=0.5\n&lv_5=13.0\n&inote_5=\n"


@pytest.fixture()
def tm(tmp_path):
    p = tmp_path / "tempo.json"
    p.write_text(json.dumps(_TEMPO), encoding="utf-8")
    return load_tempo_map(p)


def test_tempo_map_normalises_bars_and_midbar_change(tm):
    assert tm.first == 0.5
    assert [(c.beat, c.bpm) for c in tm.changes] == [(0.0, 120.0), (4.0, 240.0),
                                                     (6.0, 60.0), (8.0, 120.0)]
    assert tm.n_bars == 3


def test_tempo_map_bar_start_is_exact(tm):
    assert tm.bar_start(0) == pytest.approx(0.5, abs=1e-12)
    assert tm.bar_start(1) == pytest.approx(0.5 + 4 * 0.5, abs=1e-12)          # 4 拍 @120
    # 第 2 小节：2 拍 @240 (0.5s) + 2 拍 @60 (2.0s)
    assert tm.bar_start(2) == pytest.approx(2.5 + 0.5 + 2.0, abs=1e-12)


def test_slot_anchor_is_after_the_previous_comma_not_the_first_note():
    """锚点取"上一个逗号之后"，`(bpm)` 才落在 `{x}` **前面**（`docs/simai-syntax.md` §6）。"""
    slots = dict(slot_positions(_BODY))
    assert slots[0.0] == 0                      # 正文开头，在 `{8}` 之前
    assert _BODY[slots[4.0]:slots[4.0] + 4].lstrip()[:2] == "{8}"[:2]


def test_emit_writes_bpm_before_the_divisor(tm):
    out = emit(_BODY, tm)
    lines = out.splitlines()
    assert lines[0].startswith("(120){8}")
    assert lines[1].startswith("(240){8}") and "(60)5," in lines[1]
    assert lines[2].startswith("(120){8}")


def test_emit_strips_pre_existing_bpm(tm):
    """正文里原来就有的 `(bpm)` 先剥掉，不会叠加。"""
    out = emit("(999){8}1,2,3,4,5,6,7,8,\n{8}1,2,3,4,5,6,7,8,\n{8}1,2,3,4,5,6,7,8,\nE\n", tm)
    assert "(999)" not in out and out.count("(120)") == 2


def test_emit_rejects_a_change_that_has_no_slot(tmp_path):
    """tempo map 要在某个拍位换速、而正文那一格根本不存在 → 直接报错，不静默取近似。"""
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"first": 0.0, "bars": [{"bar": 1, "bpm": 120.0},
                                                    {"bar": 1, "beat": 0.125, "bpm": 200.0}]}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="没有槽起点"):
        emit(_BODY, load_tempo_map(p))


def test_verify_round_trip_is_exact(tmp_path, tm):
    md = tmp_path / "maidata.txt"
    md.write_text(_HEAD + emit(_BODY, tm), encoding="utf-8")
    rep = verify(md, tm)
    assert rep["ok"], rep["problems"]
    assert rep["worst_err_ms"] == 0.0
    assert rep["n_bars"] == 3
    assert [r["got"] for r in rep["rows"]] == [0.5, 2.5, 5.0]


def test_verify_catches_a_wrong_bpm(tmp_path, tm):
    """把第 2 小节的 `(240)` 手抖写成 `(200)` → 必须被抓出来。"""
    md = tmp_path / "maidata.txt"
    md.write_text(_HEAD + emit(_BODY, tm).replace("(240)", "(200)"), encoding="utf-8")
    rep = verify(md, tm)
    assert not rep["ok"] and rep["worst_err_ms"] > 1.0


def test_verify_catches_a_first_mismatch(tmp_path, tm):
    md = tmp_path / "maidata.txt"
    md.write_text((_HEAD + emit(_BODY, tm)).replace("&first=0.5", "&first=0.7"), encoding="utf-8")
    rep = verify(md, tm)
    assert not rep["ok"]
    assert any("`&first` 不一致" in p for p in rep["problems"])


def test_verify_counts_bars_by_notes_not_by_measure_lines(tmp_path, tm):
    """`measure_starts` 会多带一条"最后一小节之后的小节线"，不能算成一个小节。"""
    md = tmp_path / "maidata.txt"
    md.write_text(_HEAD + emit(_BODY, tm), encoding="utf-8")
    assert verify(md, tm)["n_bars"] == 3
