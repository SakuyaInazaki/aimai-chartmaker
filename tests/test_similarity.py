#!/usr/bin/env python3
"""``tools/chart_analysis/similarity.py`` 的单元测试。

素材全部为自写的小段 simai 片段，不含任何参考谱 / 官方谱原文。

运行：``.venv/bin/python -m pytest tests/test_similarity.py -q``
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis.dx_filter import filter_to_st  # noqa: E402
from chart_analysis.simai_parser import parse_chart  # noqa: E402
from chart_analysis.similarity import (  # noqa: E402
    compare,
    ngram_overlap,
    ngrams,
    note_sequence,
    note_token,
    shuffled_tokens,
)


def _p(src):
    return filter_to_st(parse_chart(src)).result


# ---------------------------------------------------------------------------
# 记号串
# ---------------------------------------------------------------------------


def test_note_token按类型给记号():
    r = _p("(120){4}1,2b,3h[4:1],4-8[4:1],E")
    toks = [note_token(n) for n in r.notes]
    assert toks == ["t1", "b2", "h3", "s4", "--8"]


def test_note_sequence同刻按记号排序():
    r = _p("(120){4}5/1,2,E")
    assert note_sequence(r) == ["t1", "t5", "t2"]


def test_ngrams长度不足给空():
    assert ngrams(["a", "b"], 4) == {}
    assert sum(ngrams(list("abcde"), 4).values()) == 2


# ---------------------------------------------------------------------------
# n-gram 重合
# ---------------------------------------------------------------------------


def test_ngram_overlap相同串给1():
    a = list("abcdefgh")
    o = ngram_overlap(a, a, 4)
    assert o["rate_a_types"] == 1.0
    assert o["rate_a_instances"] == 1.0
    assert o["jaccard"] == 1.0


def test_ngram_overlap无交集给0():
    o = ngram_overlap(list("abcdefgh"), list("ijklmnop"), 4)
    assert o["rate_a_types"] == 0.0 and o["jaccard"] == 0.0
    assert o["shared_types"] == 0


def test_ngram_overlap部分重合():
    a = list("abcdxyzw")     # 5 个 4-gram
    b = list("abcd")         # 1 个 4-gram，等于 a 的第一个
    o = ngram_overlap(a, b, 4)
    assert o["shared_types"] == 1
    assert o["a_types"] == 5
    assert o["rate_a_types"] == 0.2


def test_ngram_overlap不对称():
    a, b = list("abcd"), list("abcdxyzw")
    assert ngram_overlap(a, b, 4)["rate_a_types"] == 1.0
    assert ngram_overlap(b, a, 4)["rate_a_types"] == 0.2


# ---------------------------------------------------------------------------
# 逐小节
# ---------------------------------------------------------------------------


def test_compare同一份谱全部小节相同并按note数分三类():
    src = "(120){4},,,,{4}1,,,,{4}1,2,3,4,E"   # m1 空、m2 单音、m3 四个音
    r = _p(src)
    rep = compare(r, r, ns=(4,))
    assert rep.identical_empty == [1]
    assert rep.identical_single == [2]
    assert rep.identical_multi == [3]           # ← 红线那一类
    assert rep.n_identical == 3
    assert rep.identical_detail[3] == "{4}1,2,3,4,"


def test_compare不同谱面没有相同小节():
    a = _p("(120){4}1,2,3,4,E")
    b = _p("(120){4}5,6,7,8,E")
    rep = compare(a, b, ns=(4,))
    assert rep.n_identical == 0
    assert rep.identical_multi == []


def test_compare空小节相同不算红线():
    a = _p("(120){4},,,,{4}1,2,3,4,E")
    b = _p("(120){4},,,,{4}5,6,7,8,E")
    rep = compare(a, b, ns=(4,))
    assert rep.identical_empty == [1]
    assert rep.identical_multi == []


def test_compare单音小节键位相同也只算单音():
    a = _p("(120){4}3,,,,{4}1,2,3,4,E")
    b = _p("(120){4}3,,,,{4}5,6,7,8,E")
    rep = compare(a, b, ns=(4,))
    assert rep.identical_single == [1] and rep.identical_multi == []


def test_compare同刻多个note的小节相同会被标红():
    a = _p("(120){4}1/8,,,,E")
    rep = compare(a, a, ns=(4,))
    assert rep.identical_multi == [1]           # 一个时间槽两个 note 也算 ≥2 note


def test_report_to_dict字段齐全():
    a = _p("(120){4}1,2,3,4,E")
    d = compare(a, a, ns=(4, 8)).to_dict()
    assert set(d) == {"n_bars", "n_identical", "identical_empty",
                      "identical_single", "identical_multi", "ngram"}
    assert [x["n"] for x in d["ngram"]] == [4, 8]


# ---------------------------------------------------------------------------
# 地板对照
# ---------------------------------------------------------------------------


def test_shuffled_tokens只换顺序不换内容():
    toks = [f"t{i%8+1}" for i in range(50)]
    sh = shuffled_tokens(toks, seed=1)
    assert sorted(sh) == sorted(toks)
    assert sh != toks
    assert shuffled_tokens(toks, seed=1) == sh      # 同 seed 可复现


def test_打乱之后重合率不高于原串():
    a = _p("(120){8}1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,E")
    hi = ngram_overlap(note_sequence(a), note_sequence(a), 4)["rate_a_types"]
    lo = ngram_overlap(shuffled_tokens(note_sequence(a), seed=3),
                       note_sequence(a), 4)["rate_a_types"]
    assert lo <= hi
