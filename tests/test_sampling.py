#!/usr/bin/env python3
"""``tools/calibration/sampling.py``（采音模式度量）的单元测试。

**素材全部为合成时间序列**（自造 onset + 强度 + 官方槽），不读官方音频/官方谱。
覆盖：coverage / extra_ratio / rest_beats 的算法，**「挑重音」判据**（两条各一正
一负 + 证据不足），八类模式的判定边界，目标音轨选择的两道稳态闸门与 melodic 口径，
以及随机平移对照（强度跟着 onset 一起搬）。

**2026-09-19**：旧版基于网格/分音的 `alt_pattern` 三条判据已删除，相关单测
（`只踩强拍` / `分音减半` / `隔一个`）一并换成「挑重音」的正负例。

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


def _alt_strength(n: int, strong: float = 0.9, weak: float = 0.2) -> list[float]:
    """偶数位强、奇数位弱。"""
    return [strong if i % 2 == 0 else weak for i in range(n)]


# ---------------------------------------------------------------------------
# 基础度量
# ---------------------------------------------------------------------------


def test_全踩_coverage为1且extra为0():
    pool = _grid(8, 0.25)
    m = sp.bar_metrics(pool, 8, pool, pool, 0.0, BAR, BPB)
    assert m["coverage"] == 1.0
    assert m["extra_ratio"] == 0.0
    assert m["hit"] == 8 and m["n_pool"] == 8


def test_踩一半_coverage为一半():
    pool = _grid(8, 0.25)
    m = sp.bar_metrics(pool[::2], 4, pool, pool, 0.0, BAR, BPB)
    assert abs(m["coverage"] - 0.5) < 1e-9


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
    pool = _grid(8, 0.25)
    ev = pool[:4]
    m = sp.bar_metrics(ev, 4, pool, pool, 0.0, BAR, BPB)
    # 最后一个音 0.75 s 到小节末 2.0 s = 1.25 s = 2.5 拍
    assert abs(m["rest_beats"] - 2.5) < 0.05
    # pool 后半为空（音乐本身没音）→ 不算留白
    pool2 = _grid(4, 0.25)
    m2 = sp.bar_metrics(pool2, 4, pool2, pool2, 0.0, BAR, BPB)
    assert m2["rest_beats"] == 0.0


def test_pool去重取簇内最强的那个事件():
    raw = [1.0, 1.02, 1.04, 2.0]          # 前三个在 60 ms 内
    st = [0.2, 0.9, 0.3, 0.5]
    t, s = sp._unique_with_strength(raw, st, sp.THRESHOLDS["merge_sec"])
    assert len(t) == 2
    assert abs(t[0] - 1.02) < 1e-9 and abs(s[0] - 0.9) < 1e-9


def test_没有网格判据_挑不挑重音都是舍音():
    """判定链上没有任何分音 / BPM / 网格量（2026-09-19 修订）；

    "挑不挑重音"从 2026-09-20 起**不再各自撑起一个类名**——两种都是社区说的
    「舍音」（只踩一部分、舍掉另一部分），是不是踩重音只留 ``accent`` 诊断标记。
    """
    pool = _grid(8, 0.25)                  # 16 分
    ev = pool[::2]                         # 隔一个踩 = 落在 8 分整拍上
    st = [0.5] * 8                         # 强度完全一致 → 挑不出重音
    m = sp.bar_metrics(ev, 4, pool, pool, 0.0, BAR, BPB, st)
    assert abs(m["coverage"] - 0.5) < 1e-9
    assert not m["accent"]
    assert m["mode"] == "舍音"
    # 同样的"隔一个"，踩强的那一半：类名不变，只是 accent 标记翻正（社区叫"主高"）
    m2 = sp.bar_metrics(ev, 4, pool, pool, 0.0, BAR, BPB, _alt_strength(8))
    assert m2["mode"] == "舍音" and m2["accent"]


# ---------------------------------------------------------------------------
# 「挑重音」判据
# ---------------------------------------------------------------------------


def test_挑重音_踩强的那一半成立():
    pool = _grid(8, 0.25)
    st = _alt_strength(8)
    ok, reason, delta = sp.accent_pick(pool, st, pool[::2])   # 偶数位 = 强
    assert ok and "重音优先" in reason and delta > 0.5


def test_挑重音_踩弱的那一半不成立():
    pool = _grid(8, 0.25)
    st = _alt_strength(8)
    ok, reason, delta = sp.accent_pick(pool, st, pool[1::2])  # 奇数位 = 弱
    assert not ok and delta < 0


def test_挑重音_分位单调也算():
    # 均强差为 0（构造成正负抵消），但命中率随强度分位单调上升
    pool = _grid(9, 0.2)
    st = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ev = [pool[0], pool[3], pool[4], pool[6], pool[7], pool[8]]
    ok, reason, _ = sp.accent_pick(pool, st, ev)
    assert ok and ("分位单调" in reason or "重音优先" in reason)
    rates = [1 / 3, 2 / 3, 3 / 3]
    assert rates[0] < rates[-1]           # 文档化：桶命中率 1/3 → 2/3 → 3/3


def test_挑重音_两侧证据不足时不下结论():
    pool = _grid(4, 0.25)
    st = [0.9, 0.2, 0.9, 0.2]
    ok, reason, delta = sp.accent_pick(pool, st, pool[:3])   # 只有 1 个未命中
    assert not ok and reason == "证据不足" and not np.isfinite(delta)


def test_挑重音_强度完全打平不算挑重音():
    pool = _grid(8, 0.25)
    ok, reason, delta = sp.accent_pick(pool, [0.5] * 8, pool[::2])
    assert not ok and delta == 0.0        # 严格大于：打平不算


def test_挑重音_没有强度时判不出():
    pool = _grid(8, 0.25)
    ok, reason, _ = sp.accent_pick(pool, [float("nan")] * 8, pool[::2])
    assert not ok and reason == "无强度"
    ok2, reason2, _ = sp.accent_pick(pool, [], pool[::2])
    assert not ok2 and reason2 == "无强度"


def test_挑重音_margin可整体替换():
    pool = _grid(8, 0.25)
    st = [0.55 if i % 2 == 0 else 0.45 for i in range(8)]     # 均强差只有 0.1
    th = dict(sp.THRESHOLDS)
    th["accent_margin"] = 0.3
    th["accent_bins"] = 8                 # 桶太细 → 分位单调判据自动失效
    ok, _, _ = sp.accent_pick(pool, st, pool[::2], th=th)
    assert not ok
    assert sp.accent_pick(pool, st, pool[::2])[0]             # 默认 margin=0 成立


# ---------------------------------------------------------------------------
# 采音方式：只有社区的三个词（2026-09-20 术语校正，见
# docs/research/sampling-terminology-survey.md）
# ---------------------------------------------------------------------------


def test_采音方式只有社区的三个词():
    """用户 2026-09-20：「这些词是我编的，你自己去社区去论坛搜索自己对应去。」

    调研结论：社区说 **全踩**（MMFC 5.4「全踩人声」）、**舍音**（谱师自述
    「定下了舍采音的基调」）、**留白**（「红谱里留白的例子也不少」）。
    用户自造的 `采全音 / 半采音 / 空音` 与 agent 自造的
    「近全采 / 随机半采 / 加花 / 静默 / 混合」一律不得出现在对外输出里。
    """
    assert sp.MODE_ORDER == ("全踩", "舍音", "留白")
    assert sp.UNKNOWN == "—"
    for bad in ("采全音", "半采音", "空音",
                "近全采", "随机半采", "稀采/空音", "加花", "静默", "混合"):
        assert bad not in sp.MODE_ORDER


def test_空音这个词必须停用_社区里它是反义词():
    """萌娘百科《音乐游戏/用语》「采空音」= 谱面**多**写了音（音乐里没音却塞音符）；

    用户口中的「空音」是谱面**少**写了音。词义相反，所以停用，改叫「留白」。
    旧词只保留在 ``LEGACY_MODE_ALIASES`` 里给读旧报告的人对照。
    """
    assert "空音" not in sp.MODE_ORDER
    assert sp.LEGACY_MODE_ALIASES == {"采全音": "全踩", "半采音": "舍音", "空音": "留白"}
    assert set(sp.LEGACY_MODE_ALIASES.values()) == set(sp.MODE_ORDER)


def test_全踩():
    assert sp.classify(0.90, 0.10, 8, 8, False) == "全踩"
    assert sp.classify(0.66, 0.10, 8, 8, False) == "全踩"     # 不再有"近全采"那一档
    # 谱面另外写了别处的音（extra 多，社区叫"采空音"）也不改采音方式——只是诊断字段
    assert sp.classify(0.90, 0.60, 10, 8, False) == "全踩"


def test_舍音():
    assert sp.classify(0.50, 0.10, 4, 8, True) == "舍音"
    # 挑不挑重音都是舍音（accent 只是诊断标记）
    assert sp.classify(0.50, 0.10, 4, 8, False) == "舍音"
    assert sp.classify(0.40, 0.10, 6, 15, False) == "舍音"


def test_留白要音乐真的在响():
    assert sp.classify(0.20, 0.10, 3, 15, False) == "留白"
    # 谱面这一小节几乎不写音、而音轨在响 → 就是留白（不再叫"静默"/"空音"）
    assert sp.classify(0.00, 0.00, 0, 12, False) == "留白"
    # pool 太小 = 没音可踩，不下结论
    assert sp.classify(0.20, 0.10, 3, 3, False) == sp.UNKNOWN


def test_判不出的小节不下结论():
    assert sp.classify(float("nan"), 0.10, 5, 0, False) == sp.UNKNOWN   # pool 为空
    assert sp.classify(0.30, 0.10, 2, 1, False) == sp.UNKNOWN           # pool 太小


def test_内部切点可整体替换_只是诊断口径():
    th = dict(sp.THRESHOLDS)
    th["full_lo"] = 0.40
    assert sp.classify(0.50, 0.0, 8, 8, False, th) == "全踩"
    assert sp.classify(0.50, 0.0, 8, 8, False) == "舍音"


# ---------------------------------------------------------------------------
# 目标音轨选择
# ---------------------------------------------------------------------------


def test_目标轨_按lift选且要超过margin():
    ev = np.array([0.0, 0.5, 1.0, 1.5])
    stems = {"drums": np.array([0.3, 0.8, 1.3, 1.8]),       # 全不命中
             "other": np.array([0.0, 0.5, 1.0, 1.5])}       # 全命中
    assert sp.pick_skeleton(ev, stems, 2.0, default="drums") == "other"


def test_目标轨_证据不足回落默认():
    ev = np.array([0.0, 0.5])                                # < min_events
    stems = {"drums": np.array([9.0]), "other": np.array([0.0, 0.5])}
    assert sp.pick_skeleton(ev, stems, 2.0, default="drums") == "drums"


def test_目标轨_差距不够不换轨():
    ev = np.array([0.0, 0.5, 1.0, 1.5])
    stems = {"drums": np.array([0.0, 0.5, 1.0, 1.5]),
             "other": np.array([0.0, 0.5, 1.0, 1.5])}        # lift 完全一样
    assert sp.pick_skeleton(ev, stems, 2.0, default="drums") == "drums"


def test_melodic口径只在旋律轨里选():
    bars = [(1, 0, 0.0, BAR, BPB)]
    pool_d = _grid(8, 0.25)
    pool_p = [0.0, 0.5, 1.0, 1.5]
    stems = {"drums": np.array(pool_d), "piano": np.array(pool_p),
             "other": np.array([0.13, 0.62, 1.11, 1.62])}
    rows = sp.song_sampling(bars, pool_p, {0: 4}, stems, skeleton_mode="melodic",
                            melodic_default="other")
    assert rows[0].skeleton == "piano"          # drums 命中更多，但不在 melodic 候选里


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
    st = _alt_strength(16)
    return bars, slots, notes, {"drums": np.array(pool)}, {"drums": np.array(st)}


def test_整曲_全部踩上时两小节都是全踩():
    bars, slots, notes, stems, st = _toy()
    rows = sp.song_sampling(bars, slots, notes, stems, skeleton_mode="drums",
                            stem_strengths=st)
    assert [r.mode for r in rows] == ["全踩", "全踩"]
    assert all(r.skeleton == "drums" for r in rows)
    assert all(abs(r.coverage - 1.0) < 1e-9 for r in rows)


def test_整曲_只踩强音那一半是舍音():
    bars, slots, notes, stems, st = _toy()
    half = [t for t, s in zip(slots, st["drums"]) if s > 0.5]
    rows = sp.song_sampling(bars, half, {0: 4, 1: 4}, stems,
                            skeleton_mode="drums", stem_strengths=st)
    assert [r.mode for r in rows] == ["舍音", "舍音"]
    assert all(r.accent for r in rows)


def test_整曲_踩弱音那一半也是舍音():
    bars, slots, notes, stems, st = _toy()
    weak = [t for t, s in zip(slots, st["drums"]) if s <= 0.5]
    rows = sp.song_sampling(bars, weak, {0: 4, 1: 4}, stems,
                            skeleton_mode="drums", stem_strengths=st)
    assert [r.mode for r in rows] == ["舍音", "舍音"]
    assert not any(r.accent for r in rows)      # 只有诊断标记不同


def test_整曲_逐小节字典可序列化():
    bars, slots, notes, stems, st = _toy()
    d = sp.song_sampling(bars, slots, notes, stems, skeleton_mode="drums",
                         stem_strengths=st)[0].to_dict()
    assert d["mode"] == "全踩" and d["bar"] == 1 and d["coverage"] == 1.0
    assert set(d) >= {"bar", "chart_measure", "skeleton", "coverage", "extra_ratio",
                      "accent", "accent_reason", "accent_delta", "rest_beats",
                      "density_ratio", "mode"}


def test_对照_平移后coverage明显下降():
    bars, slots, notes, stems, st = _toy(regular=False)
    real = sp.song_sampling(bars, slots, notes, stems, skeleton_mode="drums",
                            stem_strengths=st)
    drops = []
    for seed in range(8):
        ctrl = sp.shuffled_control(bars, slots, notes, stems, seed=seed,
                                   stem_strengths=st, skeleton_mode="drums")
        drops.append(np.nanmean([r.coverage for r in ctrl]))
    assert np.mean([r.coverage for r in real]) > np.mean(drops) + 0.2


def test_对照_强度跟着onset一起搬():
    bars, slots, notes, stems, st = _toy(regular=False)
    rows = sp.shuffled_control(bars, slots, notes, stems, seed=1,
                               stem_strengths=st, skeleton_mode="drums")
    assert sum(r.n_pool for r in rows) > 0      # 平移只改位置，不改数量量级
    # 对照里仍然要能算出"挑重音"（强度没有被丢掉），只是方向随机
    assert any(np.isfinite(r.accent_delta) for r in rows)


def test_density_ratio():
    pool = _grid(4, 0.5)
    ev = _grid(8, 0.25)
    m = sp.bar_metrics(ev, 8, pool, pool + ev, 0.0, BAR, BPB)
    assert abs(m["density_ratio"] - 2.0) < 1e-9
