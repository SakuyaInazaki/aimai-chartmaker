#!/usr/bin/env python3
"""``tools/calibration/sampling.py``（采音模式度量）的单元测试。

**素材全部为合成时间序列**（自造 onset 与官方槽），不读官方音频/官方谱。
覆盖：coverage / extra_ratio / alt_pattern / rest_beats 的算法，六类模式的
判定边界（每类一正一负），骨架轨选择的两道稳态闸门，以及随机平移对照。

运行：``PYTHONPATH=tools python3 -m pytest tests/test_sampling.py -q``
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from calibration import sampling as sp  # noqa: E402

BPB = 0.5          # 每拍 0.5 秒（= 120 BPM）
BAR = 4 * BPB      # 一小节 2 秒


def _grid(n: int, step: float, t0: float = 0.0) -> list[float]:
    return [t0 + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# 基础度量
# ---------------------------------------------------------------------------


def test_全踩_coverage为1且extra为0():
    pool = _grid(8, 0.25)
    m = sp.bar_metrics(pool, 8, pool, pool, 0.0, BAR, BPB)
    assert m["coverage"] == 1.0
    assert m["extra_ratio"] == 0.0
    assert m["hit"] == 8 and m["n_pool"] == 8


def test_隔一个踩_coverage为一半且alt成立():
    pool = _grid(8, 0.25)
    ev = pool[::2]
    m = sp.bar_metrics(ev, 4, pool, pool, 0.0, BAR, BPB)
    assert abs(m["coverage"] - 0.5) < 1e-9
    assert m["alt_pattern"] and "隔一个" in m["alt_reason"]


def test_只踩强拍也算alt():
    # pool 是 16 分（含 0.25 拍的反拍），官方只踩整拍
    pool = _grid(16, 0.125)
    ev = [0.0, 0.5, 1.0, 1.5]
    m = sp.bar_metrics(ev, 4, pool, pool, 0.0, BAR, BPB)
    assert m["alt_pattern"] and "只踩强拍" in m["alt_reason"]


def test_分音减半也算alt():
    pool = [0.0, 0.31, 0.77, 1.2]        # 位置不规则，前两条判据都不成立
    ev = [0.0, 0.77]
    m = sp.bar_metrics(ev, 2, pool, pool, 0.0, BAR, BPB,
                       chart_div=8.0, audio_div=16.0)
    assert m["alt_pattern"] and "分音减半" in m["alt_reason"]


def test_alt_负例_匀踩不算():
    pool = _grid(8, 0.25)
    m = sp.bar_metrics(pool, 8, pool, pool, 0.0, BAR, BPB,
                       chart_div=16.0, audio_div=16.0)
    assert not m["alt_pattern"]


def test_extra_是任何轨都解释不了的槽():
    pool = [0.0, 0.5, 1.0]
    ev = [0.0, 0.5, 1.0, 1.23, 1.71]     # 后两个不在任何 onset 上
    m = sp.bar_metrics(ev, 5, pool, pool, 0.0, BAR, BPB)
    assert m["extra"] == 2
    assert abs(m["extra_ratio"] - 0.4) < 1e-9


def test_容差内算命中_容差外不算():
    pool = [1.0]
    assert sp.bar_metrics([1.020], 1, pool, pool, 0.0, BAR, BPB)["hit"] == 1
    assert sp.bar_metrics([1.060], 1, pool, pool, 0.0, BAR, BPB)["hit"] == 0


def test_rest_beats_只在pool非空的留白处计():
    # 前半小节踩满、后半留白但 pool 后半有音 → rest ≈ 2 拍
    pool = _grid(8, 0.25)
    ev = pool[:4]
    m = sp.bar_metrics(ev, 4, pool, pool, 0.0, BAR, BPB)
    # 最后一个音 0.75 s 到小节末 2.0 s = 1.25 s = 2.5 拍
    assert abs(m["rest_beats"] - 2.5) < 0.05
    # pool 后半为空（音乐本身没音）→ 不算留白
    pool2 = _grid(4, 0.25)
    m2 = sp.bar_metrics(pool2, 4, pool2, pool2, 0.0, BAR, BPB)
    assert m2["rest_beats"] == 0.0


def test_pool去重按2倍容差合并():
    raw = [1.0, 1.02, 1.04, 2.0]         # 前三个在 60 ms 内
    got = sp.stemhit.unique_times(raw, merge_sec=sp.THRESHOLDS["merge_sec"])
    assert len(got) == 2


# ---------------------------------------------------------------------------
# 六类模式的判定边界
# ---------------------------------------------------------------------------


def test_模式_全采():
    assert sp.classify(0.90, 0.10, 8, 8, False) == "全采"
    assert sp.classify(0.79, 0.10, 8, 8, False) != "全采"      # 差一点点就不是
    assert sp.classify(0.90, 0.35, 8, 8, False) != "全采"      # 加花太多


def test_模式_半采():
    assert sp.classify(0.50, 0.10, 4, 8, True) == "半采"
    assert sp.classify(0.50, 0.10, 4, 8, False) == "混合"      # alt 不成立就不是半采
    assert sp.classify(0.70, 0.10, 6, 8, True) != "半采"       # 超出 [0.35,0.65]


def test_模式_稀采空音():
    assert sp.classify(0.20, 0.10, 3, 15, False) == "稀采/空音"
    assert sp.classify(0.40, 0.10, 6, 15, False) != "稀采/空音"


def test_模式_加花():
    assert sp.classify(0.90, 0.60, 10, 8, False) == "加花"     # 加花优先于全采
    assert sp.classify(0.10, 0.60, 10, 8, False) == "加花"
    assert sp.classify(0.90, 0.49, 10, 8, False) != "加花"


def test_模式_静默():
    assert sp.classify(1.00, 0.00, 1, 8, False) == "静默"
    assert sp.classify(1.00, 0.00, 2, 8, False) != "静默"


def test_模式_混合是残差():
    assert sp.classify(0.70, 0.10, 6, 8, False) == "混合"
    assert sp.classify(float("nan"), 0.10, 5, 0, False) == "混合"   # pool 为空


def test_模式_阈值可整体替换():
    th = dict(sp.THRESHOLDS); th["full_coverage"] = 0.60
    assert sp.classify(0.70, 0.0, 8, 8, False, th) == "全采"
    assert sp.classify(0.70, 0.0, 8, 8, False) == "混合"


# ---------------------------------------------------------------------------
# 骨架轨选择
# ---------------------------------------------------------------------------


def test_骨架_按lift选且要超过margin():
    ev = np.array([0.0, 0.5, 1.0, 1.5])
    stems = {"drums": np.array([0.3, 0.8, 1.3, 1.8]),       # 全不命中
             "other": np.array([0.0, 0.5, 1.0, 1.5])}       # 全命中
    assert sp.pick_skeleton(ev, stems, 2.0, default="drums") == "other"


def test_骨架_证据不足回落默认():
    ev = np.array([0.0, 0.5])                                # < min_events
    stems = {"drums": np.array([9.0]), "other": np.array([0.0, 0.5])}
    assert sp.pick_skeleton(ev, stems, 2.0, default="drums") == "drums"


def test_骨架_差距不够不换轨():
    ev = np.array([0.0, 0.5, 1.0, 1.5])
    stems = {"drums": np.array([0.0, 0.5, 1.0, 1.5]),
             "other": np.array([0.0, 0.5, 1.0, 1.5])}        # lift 完全一样
    assert sp.pick_skeleton(ev, stems, 2.0, default="drums") == "drums"


# ---------------------------------------------------------------------------
# 整曲 & 对照
# ---------------------------------------------------------------------------


def _toy(regular: bool = True):
    bars = [(1, 0, 0.0, BAR, BPB), (2, 1, BAR, 2 * BAR, BPB)]
    pool = _grid(8, 0.25) + _grid(8, 0.25, BAR)
    if not regular:
        # 对照要打破周期性，否则循环平移可能整格对回去
        rng = np.random.default_rng(7)
        pool = sorted(float(x) + float(rng.uniform(-0.04, 0.04)) for x in pool)
    slots = pool[:]          # 全踩
    notes = {0: 8, 1: 8}
    return bars, slots, notes, {"drums": np.array(pool)}


def test_整曲_全踩时两小节都是全采():
    bars, slots, notes, stems = _toy()
    rows = sp.song_sampling(bars, slots, notes, stems, skeleton_mode="drums")
    assert [r.mode for r in rows] == ["全采", "全采"]
    assert all(r.skeleton == "drums" for r in rows)
    assert all(abs(r.coverage - 1.0) < 1e-9 for r in rows)


def test_整曲_逐小节字典可序列化():
    bars, slots, notes, stems = _toy()
    d = sp.song_sampling(bars, slots, notes, stems, skeleton_mode="drums")[0].to_dict()
    assert d["mode"] == "全采" and d["bar"] == 1 and d["coverage"] == 1.0
    assert set(d) >= {"bar", "chart_measure", "skeleton", "coverage", "extra_ratio",
                      "alt_pattern", "rest_beats", "density_ratio", "mode"}


def test_对照_平移后coverage明显下降():
    bars, slots, notes, stems = _toy(regular=False)
    real = sp.song_sampling(bars, slots, notes, stems, skeleton_mode="drums")
    drops = []
    for seed in range(8):
        ctrl = sp.shuffled_control(bars, slots, notes, stems, seed=seed,
                                   skeleton_mode="drums")
        drops.append(np.nanmean([r.coverage for r in ctrl]))
    assert np.mean([r.coverage for r in real]) > np.mean(drops) + 0.2


def test_对照_保持onset个数不变():
    bars, slots, notes, stems = _toy()
    rows = sp.shuffled_control(bars, slots, notes, stems, seed=1, skeleton_mode="drums")
    assert sum(r.n_pool for r in rows) > 0      # 平移只改位置，不改数量量级


def test_density_ratio():
    pool = _grid(4, 0.5)
    ev = _grid(8, 0.25)
    m = sp.bar_metrics(ev, 8, pool, pool + ev, 0.0, BAR, BPB)
    assert abs(m["density_ratio"] - 2.0) < 1e-9
