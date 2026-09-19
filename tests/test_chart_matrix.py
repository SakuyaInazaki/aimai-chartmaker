#!/usr/bin/env python3
"""``tools/chart_analysis/chart_matrix.py``（谱面级 定数 × BPM × 配置 对照）的单元测试。

**测试素材全部为自写的合成 simai 片段与手搭的 ChartRow**，不复制官方谱原文、
不读音频。重点钉住：

1. **检测器可替换**——注入 ``bar_configs`` / ``detector`` 时，一谱一行的配置列
   只跟注入的表走（等手序版 `hands.py` 落地后要能原样换掉）；
2. 主 BPM / 变速多段 / 加权中位 BPM 的口径；
3. 出现率（按谱）与小节占比（只在用了的谱里取中位）**是两个不同的数**；
4. 趋势 / 词汇量 / 共存互斥 / 住址 / 个案序列的算法口径。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_chart_matrix.py -q``
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from chart_analysis import chart_matrix as cmx  # noqa: E402
from chart_analysis.simai_parser import parse_chart  # noqa: E402


def _parse(body: str):
    res = parse_chart(body, name="t")
    assert not res.errors, res.errors
    return res


# 两小节 8 分匀拍单点（120 BPM ⇒ 一小节 2 秒）
BODY_2BARS = "(120){8}1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,E"


def _row(level, bpm, configs=None, vocab_pad=0, **kw):
    """手搭一行（配置给 ``{名: 小节占比}``）。"""
    configs = configs or {}
    cfg = {c: {"n_bars": int(round(s * 100)), "bar_share": s, "n_segments": 1}
           for c, s in configs.items()}
    for i in range(vocab_pad):
        cfg[f"填充{i}"] = {"n_bars": 1, "bar_share": 0.01, "n_segments": 1}
    return cmx.ChartRow(chart=kw.pop("chart", f"c{level}-{bpm}"),
                        level=level, level_band=cmx.level_band(level),
                        bpm_main=bpm, bpm_band=cmx.bpm_band(bpm),
                        n_bars=100, configs=cfg, **kw)


# ---------------------------------------------------------------------------
# 分档
# ---------------------------------------------------------------------------


def test_定数分档边界():
    assert cmx.level_band(13.0) == "13.0-13.2"
    assert cmx.level_band(13.2) == "13.0-13.2"
    assert cmx.level_band(13.3) == "13.3-13.5"
    assert cmx.level_band(13.9) == "13.9-14.1"
    assert cmx.level_band(14.2) == "14.2-14.5"
    assert cmx.level_band(14.5) == "14.2-14.5"
    assert cmx.level_band(12.9) == ""      # 档外不硬塞
    assert cmx.level_band(None) == ""


def test_BPM分档边界():
    assert cmx.bpm_band(139.9) == "<140"
    assert cmx.bpm_band(140) == "140-169"
    assert cmx.bpm_band(169.9) == "140-169"
    assert cmx.bpm_band(170) == "170-199"
    assert cmx.bpm_band(200) == ">=200"
    assert cmx.bpm_band(266) == ">=200"
    assert cmx.bpm_band(0) == ""


# ---------------------------------------------------------------------------
# BPM 画像
# ---------------------------------------------------------------------------


def test_定速谱只有一段BPM():
    segs = cmx.bpm_profile(_parse(BODY_2BARS))
    assert len(segs) == 1
    assert segs[0][0] == 120.0
    assert math.isclose(segs[0][1], 4.0, rel_tol=1e-6)   # 两小节 × 2 秒


def test_变速谱记多段且主BPM取占时最长():
    # 前 2 小节 100 BPM（4.8 秒），后 1 小节 200 BPM（1.2 秒）
    body = "(100){4}1,2,3,4,1,2,3,4,(200){4}1,2,3,4,E"
    segs = cmx.bpm_profile(_parse(body))
    assert [b for b, _ in segs] == [100.0, 200.0]
    assert math.isclose(segs[0][1], 4.8, rel_tol=1e-6)
    assert math.isclose(segs[1][1], 1.2, rel_tol=1e-6)
    row = cmx.chart_row(_parse(body), level=13.4, bar_configs={})
    assert row.bpm_main == 100.0          # 占时最长的那段
    assert row.n_bpm == 2
    assert math.isclose(row.bpm_main_share, 4.8 / 6.0, rel_tol=1e-6)


def test_加权中位BPM在渐变速下比最长段稳健():
    # 一段 150 占 1 份、两段 200/201 各占 3 份 ⇒ 最长段是 200，中位也应落在 200 侧
    segs = [(150.0, 1.0), (200.0, 3.0), (201.0, 3.0)]
    assert cmx.weighted_median_bpm(segs) == 200.0
    # 微调速谱：每段都很短，但中位仍落在密集区
    many = [(160.0 + i * 0.01, 0.5) for i in range(100)] + [(90.0, 1.0)]
    assert 160.0 <= cmx.weighted_median_bpm(many) < 161.0
    assert cmx.weighted_median_bpm([]) == 0.0


# ---------------------------------------------------------------------------
# 一谱一行
# ---------------------------------------------------------------------------


def test_一谱一行基础量():
    row = cmx.chart_row(_parse(BODY_2BARS), chart="t", level=13.4, bar_configs={})
    assert row.n_bars == 2
    assert row.n_notes == 16
    assert row.n_slots == 16
    assert math.isclose(row.seconds, 4.0, rel_tol=1e-6)
    assert math.isclose(row.nps, 4.0, rel_tol=1e-6)
    assert math.isclose(row.notes_per_bar, 8.0, rel_tol=1e-6)
    assert row.div_share == {8.0: 1.0}
    assert row.main_divisor == 8.0
    assert row.level_band == "13.3-13.5"
    assert row.bpm_band == "<140"


def test_一谱一行的种类占比():
    # 1 tap + 1 hold + 1 slide（头 + 轨）+ 1 双押 + 1 break
    body = "(120){4}1,2h[4:1],3-5[4:1],1/5,6b,E"
    row = cmx.chart_row(_parse(body), level=13.4, bar_configs={})
    assert row.hold_ratio > 0
    assert row.slide_ratio > 0
    assert row.break_ratio > 0
    assert row.each_ratio > 0
    assert row.touch_ratio == 0


def test_空配置表时词汇量为零():
    row = cmx.chart_row(_parse(BODY_2BARS), level=13.4, bar_configs={})
    assert row.configs == {}
    assert row.vocab == 0
    assert row.config_set == frozenset()


# ---------------------------------------------------------------------------
# ★ 检测器可替换
# ---------------------------------------------------------------------------


def test_注入bar_configs时配置列只跟注入的表走():
    res = _parse(BODY_2BARS)
    row = cmx.chart_row(res, level=13.4,
                        bar_configs={0: {"甲", "乙"}, 1: {"甲"}})
    assert set(row.configs) == {"甲", "乙"}
    assert row.configs["甲"]["n_bars"] == 2
    assert row.configs["甲"]["bar_share"] == 1.0
    assert row.configs["乙"]["n_bars"] == 1
    assert row.configs["乙"]["bar_share"] == 0.5
    assert row.vocab == 2


def test_注入detector可调用且结果照用():
    res = _parse(BODY_2BARS)

    def fake(_res, _slots):
        return [{"config": "丙", "bar_start": 0, "bar_end": 1}]

    row = cmx.chart_row(res, level=13.4, detector=fake)
    assert set(row.configs) == {"丙"}
    assert row.configs["丙"]["bar_share"] == 1.0
    # 与默认（键位版 configs.detect_all）结果不同 ⇒ 注入确实生效
    assert set(cmx.chart_row(res, level=13.4).configs) != {"丙"}


def test_注入的配置落在没有note的小节时不计入():
    res = _parse(BODY_2BARS)                 # 只有小节 0、1 有 note
    row = cmx.chart_row(res, level=13.4, bar_configs={0: {"甲"}, 9: {"甲"}})
    assert row.configs["甲"]["n_bars"] == 1
    assert row.configs["甲"]["bar_share"] == 0.5


# ---------------------------------------------------------------------------
# 片段数（只依赖小节表）
# ---------------------------------------------------------------------------


def test_片段数等于连续小节段数():
    assert cmx._segments_from_bars([]) == 0
    assert cmx._segments_from_bars([3]) == 1
    assert cmx._segments_from_bars([1, 2, 3]) == 1
    assert cmx._segments_from_bars([1, 2, 5, 6, 9]) == 3
    assert cmx._segments_from_bars([5, 1, 2]) == 2       # 乱序也要先排


def test_片段数走注入的表():
    body = "(120){8}" + ",".join(["1"] * 8) + "," + ",".join(["1"] * 8) + \
           "," + ",".join(["1"] * 8) + "," + ",".join(["1"] * 8) + ",E"
    row = cmx.chart_row(_parse(body), level=13.4,
                        bar_configs={0: {"甲"}, 1: {"甲"}, 3: {"甲"}})
    assert row.configs["甲"]["n_segments"] == 2
    assert row.configs["甲"]["n_bars"] == 3


# ---------------------------------------------------------------------------
# 统计小工具
# ---------------------------------------------------------------------------


def test_中位与分位():
    assert cmx.median([3, 1, 2]) == 2
    assert cmx.median([4, 1, 2, 3]) == 2.5
    assert math.isnan(cmx.median([]))
    assert cmx.quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert cmx.quantile([1, 2, 3, 4, 5], 0.0) == 1


def test_spearman单调与并列():
    assert math.isclose(cmx.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert math.isclose(cmx.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
    assert math.isnan(cmx.spearman([1, 2], [1, 2]))          # n < 3
    assert math.isnan(cmx.spearman([1, 1, 1, 1], [1, 2, 3, 4]))  # 零方差


# ---------------------------------------------------------------------------
# 分格
# ---------------------------------------------------------------------------


def test_出现率与小节占比是两个数():
    """3 张谱里 1 张用了「甲」且占 60% 小节 ⇒ 出现率 1/3，占比中位 0.60（不被 0 拉低）。"""
    rows = [_row(13.4, 150, {"甲": 0.60}), _row(13.4, 150, {}), _row(13.4, 150, {})]
    cell = cmx.cell_summary(rows)
    assert cell["n"] == 3
    assert math.isclose(cell["configs"]["甲"]["rate"], 1 / 3)
    assert math.isclose(cell["configs"]["甲"]["share_median"], 0.60)
    assert math.isclose(cell["configs"]["甲"]["share_mean_all"], 0.20)
    assert math.isclose(cell["configs"]["甲"]["main_rate"], 1 / 3)


def test_主料阈值():
    rows = [_row(13.4, 150, {"甲": 0.30, "乙": 0.02})]
    assert rows[0].main_configs() == frozenset({"甲"})
    assert rows[0].vocab == 2 and rows[0].vocab_main == 1
    assert cmx.cell_summary(rows)["configs"]["乙"]["main_rate"] == 0.0


def test_分格含边际行列():
    rows = [_row(13.0, 120, {"甲": 0.5}), _row(14.3, 210, {"乙": 0.5})]
    g = cmx.grid(rows)
    assert g["13.0-13.2|<140"]["n"] == 1
    assert g["14.2-14.5|>=200"]["n"] == 1
    assert g["13.0-13.2|140-169"]["n"] == 0
    assert g["13.0-13.2|*"]["n"] == 1
    assert g["*|>=200"]["n"] == 1
    assert g["*|*"]["n"] == 2


def test_空格子不报配置():
    g = cmx.grid([_row(13.0, 120, {"甲": 0.5})])
    assert "configs" not in g["14.2-14.5|<140"]


def test_热图形状与取值():
    rows = [_row(13.0, 120, {"甲": 0.5}), _row(13.0, 120, {}),
            _row(14.3, 210, {"甲": 0.2})]
    h = cmx.heatmap(rows, "rate")
    assert math.isclose(h["甲"]["13.0-13.2"]["<140"], 0.5)
    assert math.isclose(h["甲"]["14.2-14.5"][">=200"], 1.0)
    assert math.isnan(h["甲"]["13.6-13.8"]["<140"])         # 空格子
    hs = cmx.heatmap(rows, "share_median")
    assert math.isclose(hs["甲"]["14.2-14.5"][">=200"], 0.2)


def test_最常见组合按主料集计():
    rows = [_row(13.4, 150, {"甲": 0.4, "乙": 0.3, "丙": 0.01}),
            _row(13.4, 150, {"甲": 0.4, "乙": 0.3}),
            _row(13.4, 150, {"甲": 0.4})]
    combos = cmx.cell_summary(rows)["top_combos"]
    assert set(combos[0]["configs"]) == {"甲", "乙"} and combos[0]["n"] == 2
    assert combos[1]["configs"] == ["甲"] and combos[1]["n"] == 1


# ---------------------------------------------------------------------------
# 趋势
# ---------------------------------------------------------------------------


def test_定数趋势为正当出现率随定数上升():
    rows = ([_row(13.0, 150, {}) for _ in range(6)]
            + [_row(13.4, 150, {"甲": 0.2}) for _ in range(6)]
            + [_row(14.3, 150, {"甲": 0.4}) for _ in range(6)])
    t = cmx.trend_by_level(rows)["甲"]
    assert t["rho_pooled"] > 0.5
    assert t["rate_13.0-13.2"] == 0.0
    assert t["rate_14.2-14.5"] == 1.0
    assert t["n_bands_pos"] == 1 and t["n_bands_neg"] == 0


def test_BPM趋势为负当高BPM少用():
    rows = ([_row(13.4, 120, {"甲": 0.3}) for _ in range(6)]
            + [_row(13.4, 150, {"甲": 0.3}) for _ in range(6)]
            + [_row(13.4, 210, {}) for _ in range(6)])
    t = cmx.trend_by_bpm(rows)["甲"]
    assert t["rho_pooled"] < -0.5
    assert t["rate_>=200"] == 0.0


def test_趋势跳过样本不足的档():
    rows = [_row(13.4, 150, {"甲": 0.2}) for _ in range(5)]     # 只有 5 张 < 10
    t = cmx.trend_by_level(rows)["甲"]
    assert t["rho_by_bpm"] == {}
    assert math.isnan(t["rho_pooled"])


def test_趋势可按小节占比而非出现率():
    rows = ([_row(13.0, 150, {"甲": 0.1}) for _ in range(6)]
            + [_row(14.3, 150, {"甲": 0.5}) for _ in range(6)])
    assert math.isnan(cmx.trend_by_level(rows, "rate")["甲"]["rho_pooled"]) or \
        cmx.trend_by_level(rows, "rate")["甲"]["rho_pooled"] == 0.0
    assert cmx.trend_by_level(rows, "share")["甲"]["rho_pooled"] > 0.5


# ---------------------------------------------------------------------------
# 词汇量
# ---------------------------------------------------------------------------


def test_词汇量分布与相关():
    rows = ([_row(13.0, 150, {"甲": 0.2}) for _ in range(6)]
            + [_row(14.3, 150, {"甲": 0.2, "乙": 0.2, "丙": 0.2}) for _ in range(6)])
    v = cmx.vocab_stats(rows)
    assert v["n"] == 12
    assert v["median"] == 2.0
    assert v["hist"] == {1: 6, 3: 6}
    assert v["rho_level"] > 0.9
    assert v["by_level"]["13.0-13.2"]["median"] == 1.0
    assert v["by_level"]["14.2-14.5"]["median"] == 3.0


# ---------------------------------------------------------------------------
# 共存 / 互斥
# ---------------------------------------------------------------------------


def test_共存与互斥的lift方向():
    # 甲乙总一起出现，甲丙从不同谱
    rows = ([_row(13.4, 150, {"甲": 0.3, "乙": 0.3}) for _ in range(20)]
            + [_row(13.4, 150, {"丙": 0.3}) for _ in range(20)])
    ps = {frozenset((p["a"], p["b"])): p
          for p in cmx.pair_stats(rows, min_charts=15)}
    ab = ps[frozenset({"甲", "乙"})]
    assert ab["lift"] > 1.5 and ab["phi"] > 0.9
    ac = ps[frozenset({"甲", "丙"})]
    assert ac["n_both"] == 0 and ac["lift"] == 0.0 and ac["phi"] < -0.9


def test_主料口径比出现口径更能分开互斥():
    """两个配置都"蹭到过"就不算互斥；只有都当主料写才算。"""
    rows = ([_row(13.4, 150, {"甲": 0.40, "乙": 0.01}) for _ in range(20)]
            + [_row(13.4, 150, {"甲": 0.01, "乙": 0.40}) for _ in range(20)])
    loose = {frozenset((p["a"], p["b"])): p
             for p in cmx.pair_stats(rows, min_charts=15)}
    tight = {frozenset((p["a"], p["b"])): p
             for p in cmx.pair_stats(rows, min_charts=15, min_share=0.05)}
    key = frozenset({"甲", "乙"})
    assert loose[key]["lift"] == 1.0          # 全谱都"出现过"，看不出来
    assert tight[key]["n_both"] == 0          # 主料层面互斥
    assert tight[key]["lift"] == 0.0


def test_低频配置被min_charts挡掉():
    rows = ([_row(13.4, 150, {"甲": 0.3}) for _ in range(20)]
            + [_row(13.4, 150, {"甲": 0.3, "稀": 0.3}) for _ in range(3)])
    assert all("稀" not in (p["a"], p["b"])
               for p in cmx.pair_stats(rows, min_charts=15))


# ---------------------------------------------------------------------------
# 住址
# ---------------------------------------------------------------------------


def test_住址只收出现率达标且样本够的格子():
    rows = ([_row(14.3, 210, {"甲": 0.30}) for _ in range(10)]
            + [_row(13.0, 120, {}) for _ in range(10)]
            + [_row(13.0, 210, {"甲": 0.30}) for _ in range(3)])   # n=3 不够
    h = cmx.habitat(rows, thresh=0.5, min_n=8)["甲"]
    assert [(c["level_band"], c["bpm_band"]) for c in h["cells"]] == \
        [("14.2-14.5", ">=200")]
    assert h["peak_cell"]["level_band"] == "14.2-14.5"
    assert h["role"] == "主料"


def test_住址区分主料与点缀():
    rows = [_row(14.3, 210, {"甲": 0.02}) for _ in range(10)]
    assert cmx.habitat(rows, thresh=0.5, min_n=8)["甲"]["role"] == "点缀"


# ---------------------------------------------------------------------------
# 个案序列
# ---------------------------------------------------------------------------


def test_个案序列折叠连续同配置小节():
    body = "(120){8}" + ",".join([",".join(["1"] * 8)] * 4) + ",E"
    row = cmx.chart_row(_parse(body), level=13.4,
                        bar_configs={0: {"甲"}, 1: {"甲"}, 2: {"乙"}, 3: {"甲"}})
    seq = cmx.config_sequence(row)
    assert [(s["bars"], s["configs"]) for s in seq] == \
        [("0-1", ["甲"]), ("2", ["乙"]), ("3", ["甲"])]
    assert seq[0]["n"] == 2


def test_个案序列可不折叠():
    row = cmx.chart_row(_parse(BODY_2BARS), level=13.4,
                        bar_configs={0: {"甲"}, 1: {"甲"}})
    assert len(cmx.config_sequence(row, fold=False)) == 2
    assert cmx.config_sequence(cmx.ChartRow()) == []


# ---------------------------------------------------------------------------
# 语料驱动（合成"语料文件"）
# ---------------------------------------------------------------------------


class _FakeChartFile:
    def __init__(self, name, body, level, raise_on_read=False):
        self.name = name
        self._body = body
        self.internal_level = level
        self.simai_id = name.split("-")[0]
        self.title = name
        self.difficulty = "mas"
        self._boom = raise_on_read

    def read(self):
        if self._boom:
            raise OSError("读不了")
        return self._body


def test_build_rows跳过坏谱并回调():
    files = [_FakeChartFile("1-好谱", BODY_2BARS, 13.4),
             _FakeChartFile("2-坏谱", "", 14.3, raise_on_read=True)]
    bad = []
    rows = cmx.build_rows(files, on_error=lambda cf, e: bad.append(cf.name))
    assert [r.chart for r in rows] == ["1-好谱"]
    assert bad == ["2-坏谱"]
    assert rows[0].level == 13.4 and rows[0].simai_id == "1"


def test_build_rows可整体换检测器():
    files = [_FakeChartFile("1-谱", BODY_2BARS, 13.4)]
    rows = cmx.build_rows(
        files, detector=lambda res, slots: [{"config": "注入",
                                             "bar_start": 0, "bar_end": 1}])
    assert set(rows[0].configs) == {"注入"}


def test_CLI跑得通(tmp_path):
    import csv
    import json

    from chart_analysis import corpus

    d = tmp_path / "charts"
    d.mkdir()
    (d / "1-甲谱-mas.txt").write_text(BODY_2BARS, encoding="utf-8")
    (d / "2-乙谱-remas.txt").write_text(BODY_2BARS, encoding="utf-8")
    out_csv, out_json = tmp_path / "m.csv", tmp_path / "m.json"
    rc = cmx._cli(["--chart-dir", str(d), "--csv", str(out_csv),
                   "--json", str(out_json), "--case", "1-甲谱-mas", "--quiet"])
    assert rc == 0
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    doc = json.loads(out_json.read_text(encoding="utf-8"))
    assert doc["n_charts"] == 2 and doc["n_errors"] == 0
    assert set(doc["grid"]) >= {"13.0-13.2|<140", "*|*"}
    assert "1-甲谱-mas" in doc["cases"]
    assert corpus  # 语料模块可导入（CLI 依赖它）
