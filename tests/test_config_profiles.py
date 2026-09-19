#!/usr/bin/env python3
"""``tools/chart_analysis/config_profiles.py``（配置难度分级与强度对应）的单元测试。

**测试素材全部为自写的合成 simai 片段与合成逐小节行**，不复制官方谱原文、
不读音频。重点钉住三件事：

1. **检测器可替换**——注入 ``bar_configs`` / ``hits`` / ``detector`` 时，
   聚合结果只跟注入的表走，与 `configs.py` 内部实现无关（等手序版
   `hands.py` 落地后要能原样换掉）；
2. 内在难度指标（NPS / 位移 / each / slide / 等效速度 / 去环硬度）算得对；
3. 分档、lift、矩阵、共现的口径。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_config_profiles.py -q``
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis import config_profiles as cpf  # noqa: E402
from chart_analysis.simai_parser import parse_chart  # noqa: E402


def _parse(body: str):
    res = parse_chart(body, name="t")
    assert not res.errors, res.errors
    return res


# 一小节 8 分匀拍单点（120 BPM ⇒ 一小节 2 秒 ⇒ NPS = 4）
BODY_8TH = "(120){8}1,2,3,4,5,6,7,8,E"


# ---------------------------------------------------------------------------
# 分档口径
# ---------------------------------------------------------------------------


def test_事件密度分档():
    assert cpf.density_tier(0) == "稀"
    assert cpf.density_tier(4) == "稀"
    assert cpf.density_tier(5) == "中"
    assert cpf.density_tier(8) == "中"
    assert cpf.density_tier(9) == "密"
    assert cpf.density_tier(None) == ""


def test_BPM分档():
    assert cpf.bpm_bucket(120) == "<150"
    assert cpf.bpm_bucket(150) == "150-180"
    assert cpf.bpm_bucket(179.9) == "150-180"
    assert cpf.bpm_bucket(180) == ">180"
    assert cpf.bpm_bucket(0) == ""


def test_定数分档():
    assert cpf.level_bucket(13.0) == "13.0-13.4"
    assert cpf.level_bucket(13.9) == "13.5-13.9"
    assert cpf.level_bucket(14.5) == "14.5+"
    assert cpf.level_bucket(None) == ""


def test_按切点分档():
    assert cpf.tier_by_cuts(0.1, [0.3, 0.7]) == "低"
    assert cpf.tier_by_cuts(0.3, [0.3, 0.7]) == "低"      # 切点归下一档
    assert cpf.tier_by_cuts(0.5, [0.3, 0.7]) == "中"
    assert cpf.tier_by_cuts(0.9, [0.3, 0.7]) == "高"


# ---------------------------------------------------------------------------
# 检测器注入
# ---------------------------------------------------------------------------


def test_注入bar_configs优先于检测器():
    res = _parse(BODY_8TH)
    t = cpf.bar_config_table(res, bar_configs={0: {"自定义配置"}})
    assert t == {0: {"自定义配置"}}


def test_注入hits按起止小节铺开():
    res = _parse(BODY_8TH)
    t = cpf.bar_config_table(res, hits=[{"config": "X", "bar_start": 0, "bar_end": 2}])
    assert t == {0: {"X"}, 1: {"X"}, 2: {"X"}}


def test_注入detector可调用():
    res = _parse(BODY_8TH)
    called = {}

    def fake(r, s):
        called["hit"] = True
        return [{"config": "假配置", "bar_start": 0, "bar_end": 0}]

    t = cpf.bar_config_table(res, detector=fake)
    assert called.get("hit") and t == {0: {"假配置"}}


def test_默认检测器才用configs模块():
    res = _parse(BODY_8TH)
    t = cpf.bar_config_table(res)
    assert t and "普通交互" in t.get(0, set())      # 8 分匀拍单点 = 知识 018


def test_画像跟着注入的表走():
    """同一份谱，注入不同的配置表 → 聚合出的配置名完全由注入决定。"""
    res = _parse(BODY_8TH)
    prof = cpf.chart_profile(res, name="t", level=13.5,
                             bar_configs={0: {"甲", "乙"}})
    agg = cpf.aggregate_by_config(prof)
    assert set(agg) == {"甲", "乙"}
    assert agg["甲"]["n_bars"] == 1 and agg["乙"]["n_bars"] == 1


# ---------------------------------------------------------------------------
# 内在难度指标
# ---------------------------------------------------------------------------


def test_NPS与等效速度():
    res = _parse(BODY_8TH)
    b = cpf.chart_profile(res, name="t", bar_configs={0: {"X"}})[0]
    assert b.n_notes == 8 and b.n_slots == 8
    assert abs(b.nps - 4.0) < 1e-6                 # 8 槽 / 2 秒
    assert abs(b.finest_divisor - 8.0) < 1e-6
    assert abs(b.equiv_speed - 4.0) < 1e-6         # (8/4) × (120/60)
    assert abs(b.div_share[8.0] - 1.0) < 1e-9


def test_位移均值_相邻键与对位键():
    near = cpf.chart_profile(_parse("(120){8}1,2,1,2,1,2,1,2,E"), name="t",
                             bar_configs={0: {"X"}})[0]
    far = cpf.chart_profile(_parse("(120){8}1,5,1,5,1,5,1,5,E"), name="t",
                            bar_configs={0: {"X"}})[0]
    assert abs(near.move_mean - 1.0) < 1e-9
    assert abs(far.move_mean - 4.0) < 1e-9         # 1↔5 = 环上最大位移


def test_each与slide占比():
    b = cpf.chart_profile(_parse("(120){8}1/5,2/6,3,4,5,6,7,8,E"), name="t",
                          bar_configs={0: {"X"}})[0]
    assert b.n_notes == 10
    assert abs(b.each_ratio - 0.4) < 1e-9          # 前两槽各 2 个 note
    s = cpf.chart_profile(_parse("(120){8}1-5[8:1],2,3,4,5,6,7,8,E"), name="t",
                          bar_configs={0: {"X"}})[0]
    assert s.slide_ratio > 0                        # 滑轨单独计一个 note


def test_去环硬度不含配置项():
    """同一份谱，注入权重完全不同的配置名 → 去环硬度不变，含配置项的硬度会变。"""
    res = _parse("(120){8}1,2,3,4,5,6,7,8,(120){8}1,3,5,7,2,4,6,8,E")
    a = cpf.chart_profile(res, name="t", bar_configs={0: {"大宇宙"}, 1: {"子弹"}})
    b = cpf.chart_profile(res, name="t", bar_configs={0: {"子弹"}, 1: {"大宇宙"}})
    assert [round(x.hardness_noconfig, 9) for x in a] == \
           [round(x.hardness_noconfig, 9) for x in b]
    assert [round(x.hardness, 9) for x in a] != [round(x.hardness, 9) for x in b]


# ---------------------------------------------------------------------------
# 聚合 / CI / 共现 / lift / 矩阵
# ---------------------------------------------------------------------------


def test_均值置信区间():
    m, lo, hi, n = cpf.mean_ci([1.0, 1.0, 1.0, 1.0])
    assert m == 1.0 and lo == 1.0 and hi == 1.0 and n == 4
    m2, lo2, hi2, n2 = cpf.mean_ci([0.0, 1.0, 2.0, 3.0])
    assert abs(m2 - 1.5) < 1e-9 and lo2 < m2 < hi2 and n2 == 4
    assert cpf.mean_ci([])[3] == 0
    assert cpf.mean_ci([1.0, float("nan")])[3] == 1      # NaN 不参与


def test_一小节命中多配置时各计一次():
    res = _parse(BODY_8TH)
    prof = cpf.chart_profile(res, name="t", bar_configs={0: {"甲", "乙"}})
    agg = cpf.aggregate_by_config(prof)
    assert agg["甲"]["n_bars"] == 1 and agg["乙"]["n_bars"] == 1
    assert abs(agg["甲"]["share_of_bars"] - 1.0) < 1e-9


def test_复合内在难度分():
    sc, d, h = cpf.intrinsic_score({"nps": 11.1, "hardness_noconfig": 0.5})
    assert abs(d - 0.5) < 1e-9 and abs(h - 0.5) < 1e-9 and abs(sc - 0.5) < 1e-9
    sc2, d2, _ = cpf.intrinsic_score({"nps": 100.0, "hardness_noconfig": 0.0})
    assert d2 == 1.0 and sc2 == 0.5                  # D 截断在 1


def test_共现只算同正负window小节内():
    p = [cpf.BarProfile(chart="c", measure=0, configs=("甲",)),
         cpf.BarProfile(chart="c", measure=1, configs=("乙",)),
         cpf.BarProfile(chart="c", measure=5, configs=("丙",))]
    co = cpf.cooccurrence(p, window=1)
    assert co["甲"][0][0] == "乙" and abs(co["甲"][0][1] - 1.0) < 1e-9
    assert "丙" not in [x[0] for x in co["甲"]]


def test_共现不跨谱():
    p = [cpf.BarProfile(chart="c1", measure=0, configs=("甲",)),
         cpf.BarProfile(chart="c2", measure=0, configs=("乙",))]
    assert cpf.cooccurrence(p) == {}


def test_lift表():
    rows = [{"t": "高", "configs": ["甲"]}, {"t": "高", "configs": ["甲"]},
            {"t": "低", "configs": ["乙"]}, {"t": "低", "configs": ["乙"]}]
    lf = cpf.lift_table(rows, "t")
    assert abs(lf["甲"]["高"] - 2.0) < 1e-9          # 甲 只出现在"高"
    assert "低" not in lf["甲"]


def test_矩阵两份榜():
    rows = ([{"pool_tier": "密", "I_tier": "高", "configs": ["稀有"], "mode": "半采音",
              "coverage": 0.5}] * 12
            + [{"pool_tier": "密", "I_tier": "高", "configs": ["常见"], "mode": "采全音",
                "coverage": 0.9}] * 20
            + [{"pool_tier": "稀", "I_tier": "低", "configs": ["常见"], "mode": "采全音",
                "coverage": 0.9}] * 200)
    m = cpf.recommend_matrix(rows, min_count=10)
    cell = m["密|高"]
    assert cell["n"] == 32
    assert cell["top_configs"][0]["config"] == "稀有"      # lift 榜
    assert cell["top_common"][0]["config"] == "常见"       # 出现率榜
    assert cell["top_modes"][0]["mode"] == "采全音"
    assert 0.5 < cell["coverage"] < 0.9


def test_矩阵按min_count挡掉小样本():
    rows = ([{"pool_tier": "密", "I_tier": "高", "configs": ["稀有"], "mode": "半采音"}] * 3
            + [{"pool_tier": "密", "I_tier": "高", "configs": ["常见"], "mode": "采全音"}] * 30)
    m = cpf.recommend_matrix(rows, min_count=10)
    assert [c["config"] for c in m["密|高"]["top_configs"]] == ["常见"]


def test_外在放置画像():
    rows = [{"configs": ["甲"], "I_bar": 0.9, "I_tier3": "高", "mode": "半采音",
             "coverage": 0.5, "pos": 0.8},
            {"configs": ["甲"], "I_bar": 0.8, "I_tier3": "高", "mode": "采全音",
             "coverage": 0.9, "pos": 0.6},
            {"configs": ["乙"], "I_bar": 0.1, "I_tier3": "低", "mode": "空音",
             "coverage": 0.2, "pos": 0.1}]
    pl = cpf.placement_by_config(rows)
    assert pl["甲"]["n_bars"] == 2
    assert abs(pl["甲"]["I_mean"] - 0.85) < 1e-9
    assert abs(pl["甲"]["coverage"] - 0.7) < 1e-9
    assert pl["甲"]["lift_I_高"] > 1.0 and pl["乙"]["lift_I_低"] > 1.0
    assert abs(pl["甲"]["mode_半采音"] - 0.5) < 1e-9


def test_空谱不炸():
    res = parse_chart("(120){8},,,,E", name="t")
    assert cpf.chart_profile(res, name="t") == [] or all(
        b.n_notes == 0 for b in cpf.chart_profile(res, name="t"))
    assert cpf.aggregate_by_config([]) == {}
    assert cpf.recommend_matrix([]) == {}
    assert math.isnan(cpf.mean_ci([])[0])
