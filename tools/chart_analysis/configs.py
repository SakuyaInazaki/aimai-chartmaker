#!/usr/bin/env python3
"""配置识别器：把官方谱正文切成"配置片段"，并给出逐小节配置硬度分。

⚠️ **定义来源与边界**
---------------------
本模块的**配置定义一律以 `.agent/knowledge/` 017–030 为准**（用户逐条讲授 + 官谱实证），
不自创定义。但"用代码判定一个片段属于某配置"必然需要把文字定义**操作化**成阈值与
几何条件——这一层（窗口长度、匀拍容差、环形跨度上限、权重数值等）是**本模块的操作化**，
不是知识条目本身。所有操作化参数集中在 :data:`PARAMS` / :data:`CONFIG_WEIGHT` 里，
可以整体替换；报告与知识条目引用时必须写明"这是 agent 的操作化"。

覆盖的配置（括号内为知识条目编号）
----------------------------------
========================  ====================================================
`错位` (017)              星星头与启动拍之间塞入其他音符
`普通交互` (018)          匀拍单点、逐音换手的统称（其余交互型配置的母类）
`纵连` (019)              同一键位连续 ≥3 音（短纵连 ≤5 / 长纵连 >5）
`三角交互` (020)          3 tap 一组、每组占 2 或 3 个邻近键
`轴交互` (021)            一手固定在轴键，轴音隔位重复（BABCB）
`楼梯交互` (022)          成对推进、每步只走相邻键、两手**反向**移动
`逆楼梯/方向盘` (023)     成对推进、始终 180° 对位、两手**同向**移动
`大宇宙` (024)            4 键 = 两对相邻键，每手在自己那对上横跳
`散点` (025)              用键分散、步进混合，且不满足任何上述几何条件
`跳拍` (026，粗判)        逗号/空拍不规则（**概念版**，知识 026 本身待补充）
`定拍` (027)              一手固定打拍子，另一手处理别的配置
`子弹` (028)              同一个键在短时间内被打两次（恰好 2 次）
`单双/双单` (029)         双押与单点交替循环
`连续双押` (030)          连续 ≥3 个双押；含 `双押纵` 子型与 **侧边双押 23/67 红线**
========================  ====================================================

硬度分
------
见 :func:`bar_hardness`。公式、权重与归一口径都写在该函数的 docstring 里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from .simai_parser import NoteEvent, ParseResult

# ---------------------------------------------------------------------------
# 操作化参数（**agent 设定**，不是知识条目原文）
# ---------------------------------------------------------------------------

PARAMS: dict = {
    # 匀拍判定：相邻时间槽间隔的相对离散度上限 + 单个间隔的绝对上限（拍）
    "uniform_rel_tol": 0.12,
    "uniform_max_gap_beats": 1.0,
    # 各配置的最短长度（时间槽数）
    "min_len_interaction": 6,     # 普通交互
    "min_len_axis": 5,            # 轴交互 = BABCB 五音小单位
    "min_len_triangle": 6,        # 三角交互 = 至少 2 组
    "min_pairs_stair": 3,         # 楼梯 / 方向盘：至少 3 对推进
    "min_len_macrocosm": 8,       # 大宇宙：至少 2 个完整循环
    "min_len_scatter": 8,         # 散点
    "min_len_vertical": 3,        # 纵连（知识 019 的门槛）
    "long_vertical": 6,           # >5 连算长纵连（知识 019）
    "min_len_double_run": 3,      # 连续双押
    "min_len_sd": 6,              # 单双/双单：至少 2 个完整周期
    "min_slots_beatkeep": 8,      # 定拍：固定键至少出现 8 次
    "bullet_max_gap_beats": 0.5,  # 子弹："短时间内"= 两击间隔 ≤ 半拍
    "scatter_min_keys": 5,        # 散点：窗口内至少 5 个不同键
    "scatter_min_step_kinds": 3,  # 散点：至少 3 种不同的步进幅度
    "side_guard_slots": 4,        # 侧边双押"引导"代理：前后各 N 个槽内出现同键音
    # 无理代理阈值（知识 009/010/011）
    "muri_overlap_sec": 0.0333,   # 叠键：同键 < 2 帧
    "muri_slidehead_sec": 0.200,  # 外键：启动后 200 ms 内同侧 tap/hold
    "muri_tail_pre_sec": 0.050,   # 撞尾危险区起点（结束前）
    "muri_tail_post_sec": 0.200,  # 撞尾危险区终点（结束后）
    # 硬度分
    "nps_redline": 22.2,          # 知识 015：短爆发上限 22.2 键/秒
    "hardness_weights": {"kind": 0.22, "move": 0.22, "speed": 0.26,
                         "muri": 0.10, "config": 0.20},
}

#: 各配置对硬度分的权重（**agent 设定**；知识 003 只给了定性顺序，没有数值）
CONFIG_WEIGHT: dict[str, float] = {
    "大宇宙": 1.00,
    "错位": 0.90,
    "双押纵": 0.90,
    "逆楼梯/方向盘": 0.85,
    "连续双押": 0.80,
    "散点": 0.80,
    "长纵连": 0.75,
    "楼梯交互": 0.70,
    "三角交互": 0.70,
    "定拍": 0.65,
    "轴交互": 0.60,
    "单双/双单": 0.60,
    "跳拍": 0.50,
    "纵连": 0.50,
    "普通交互": 0.45,
    "子弹": 0.40,
}

#: note 种类权重（沿用 `tools/calibration/chartpair.py` 的口径，知识 003-1 的最简量化）
KIND_WEIGHT: dict[str, float] = {
    "tap": 1.0, "hold": 1.2, "slide_star": 1.0, "slide_track": 1.5,
    "touch": 0.8, "touch_hold": 1.0,
}
BREAK_BONUS = 0.3
EACH_BONUS = 0.2

_BEATS_PER_MEASURE = 4.0
_SIDE_DOUBLES = ({2, 3}, {6, 7})     # 知识 030 铁律：侧边双押只指 23 与 67


# ---------------------------------------------------------------------------
# 基础结构
# ---------------------------------------------------------------------------


def cdist(a: int, b: int) -> int:
    """8 键环上的键位距离（0–4）。知识 003-2 的"位移"。"""
    d = abs(int(a) - int(b)) % 8
    return min(d, 8 - d)


def cstep(a: int, b: int) -> int:
    """8 键环上从 a 到 b 的有向步进，取绝对值最小的表示（−3..4）。"""
    d = (int(b) - int(a)) % 8
    return d - 8 if d > 4 else d


@dataclass
class Slot:
    """一个时间槽（同一个逗号之前的全部 note）。"""

    index: int                 # 在 slots 列表中的序号
    group_index: int
    time: float                # 秒（谱面正文起点为 0）
    beat: float
    measure: int
    beat_in_measure: float
    bpm: float
    divisor: float
    keys: tuple[int, ...] = ()        # 外键 1..8（touch 不计入几何判定）
    n_notes: int = 0
    kinds: tuple[str, ...] = ()
    has_slide: bool = False
    has_hold: bool = False
    n_break: int = 0
    n_touch: int = 0

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def is_double(self) -> bool:
        """双押 = 同刻 ≥2 个**外键**。"""
        return len(self.keys) >= 2

    @property
    def key_set(self) -> frozenset[int]:
        return frozenset(self.keys)


@dataclass
class ConfigHit:
    """一个配置片段。一个片段可以同时命中多个配置（各自一条）。"""

    config: str
    bar_start: int             # 小节号（谱面口径，0 起）
    bar_end: int
    slot_start: int
    slot_end: int
    t_start: float
    t_end: float
    n_slots: int
    n_notes: int
    evidence: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"config": self.config, "bar_start": self.bar_start,
                "bar_end": self.bar_end, "slot_start": self.slot_start,
                "slot_end": self.slot_end,
                "t_start": round(self.t_start, 4), "t_end": round(self.t_end, 4),
                "n_slots": self.n_slots, "n_notes": self.n_notes,
                "evidence": self.evidence, "detail": self.detail}


def build_slots(res: ParseResult) -> list[Slot]:
    """把 ``ParseResult.notes`` 归并成时间槽序列（按时间升序）。"""
    buckets: dict[int, list[NoteEvent]] = {}
    for n in res.notes:
        buckets.setdefault(n.group_index, []).append(n)
    slots: list[Slot] = []
    for i, gi in enumerate(sorted(buckets)):
        group = buckets[gi]
        head = group[0]
        keys = sorted({int(n.key) for n in group
                       if n.key.isdigit() and n.kind != "slide_track"})
        slots.append(Slot(
            index=i, group_index=gi, time=head.time, beat=head.beat,
            measure=head.measure, beat_in_measure=head.beat_in_measure,
            bpm=head.bpm, divisor=head.divisor,
            keys=tuple(keys), n_notes=len(group),
            kinds=tuple(n.kind for n in group),
            has_slide=any(n.kind == "slide_track" for n in group),
            has_hold=any(n.kind in ("hold", "touch_hold") for n in group),
            n_break=sum(1 for n in group if n.is_break),
            n_touch=sum(1 for n in group if n.kind in ("touch", "touch_hold")),
        ))
    return slots


def _span(slots: Sequence[Slot], a: int, b: int, config: str, evidence: str,
          detail: dict | None = None) -> ConfigHit:
    sub = slots[a:b + 1]
    return ConfigHit(
        config=config,
        bar_start=sub[0].measure, bar_end=sub[-1].measure,
        slot_start=a, slot_end=b,
        t_start=sub[0].time, t_end=sub[-1].time,
        n_slots=len(sub), n_notes=sum(s.n_notes for s in sub),
        evidence=evidence, detail=detail or {},
    )


# ---------------------------------------------------------------------------
# 匀拍段切分
# ---------------------------------------------------------------------------


def uniform_runs(slots: Sequence[Slot], min_len: int = 4,
                 single_only: bool = True,
                 rel_tol: float | None = None,
                 max_gap_beats: float | None = None) -> list[tuple[int, int]]:
    """切出"匀拍"连续段 ``[(start, end), ...]``（含端点，下标基于 ``slots``）。

    匀拍 = 相邻时间槽的**拍间隔**彼此相等（相对误差 ≤ ``rel_tol``）且 ≤
    ``max_gap_beats``（默认 1 拍，避免把跨小节的稀疏音串成一段）。

    ``single_only=True`` 时只保留全部为单点（1 个外键）的段——交互型配置
    （018/020/021/022/023/024/025）都建立在单点序列上。
    """
    rel_tol = PARAMS["uniform_rel_tol"] if rel_tol is None else rel_tol
    max_gap = PARAMS["uniform_max_gap_beats"] if max_gap_beats is None else max_gap_beats
    runs: list[tuple[int, int]] = []
    n = len(slots)
    i = 0
    while i < n:
        if single_only and slots[i].n_keys != 1:
            i += 1
            continue
        j = i
        gap0: float | None = None
        while j + 1 < n:
            nxt = slots[j + 1]
            if single_only and nxt.n_keys != 1:
                break
            g = nxt.beat - slots[j].beat
            if g <= 0 or g > max_gap:
                break
            if gap0 is None:
                gap0 = g
            elif abs(g - gap0) > rel_tol * gap0:
                break
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j))
        i = max(j, i) + 1
    return runs


# ---------------------------------------------------------------------------
# 各配置检测
# ---------------------------------------------------------------------------


def detect_vertical(slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 019 纵连：同一键位**连续** ≥3 个音符（键位任意，绝赞不打断）。

    另产出 030 的子型 `双押纵`：连续 ≥3 个**键集完全相同**的双押。
    """
    hits: list[ConfigHit] = []
    n = len(slots)
    max_gap = PARAMS["uniform_max_gap_beats"]
    # --- 单点纵连 ---
    i = 0
    while i < n:
        if slots[i].n_keys != 1:
            i += 1
            continue
        k = slots[i].keys[0]
        j = i
        while (j + 1 < n and slots[j + 1].n_keys == 1
               and slots[j + 1].keys[0] == k
               and 0 < slots[j + 1].beat - slots[j].beat <= max_gap):
            j += 1
        ln = j - i + 1
        n_slide = sum(1 for t in range(i, j + 1) if slots[t].has_slide)
        # 星星串（同键连发星星头）不算纵连——纵连的难点在"拆着敲同一个键"
        if ln >= PARAMS["min_len_vertical"] and n_slide * 3 <= ln:
            long_ = ln >= PARAMS["long_vertical"]
            hits.append(_span(slots, i, j, "长纵连" if long_ else "纵连",
                              f"键 {k} 连续 {ln} 音",
                              {"key": k, "length": ln, "long": long_,
                               "n_slide_slots": n_slide}))
        i = j + 1
    # --- 双押纵（知识 030 的"双押纵"类型） ---
    i = 0
    while i < n:
        if not slots[i].is_double:
            i += 1
            continue
        ks = slots[i].key_set
        j = i
        while (j + 1 < n and slots[j + 1].key_set == ks
               and 0 < slots[j + 1].beat - slots[j].beat <= max_gap):
            j += 1
        ln = j - i + 1
        if ln >= PARAMS["min_len_vertical"]:
            hits.append(_span(slots, i, j, "双押纵",
                              f"双押 {'/'.join(str(x) for x in sorted(ks))} 连续 {ln} 次",
                              {"keys": sorted(ks), "length": ln}))
        i = j + 1
    return hits


def detect_axis(slots: Sequence[Slot], runs: Sequence[tuple[int, int]]
                ) -> list[ConfigHit]:
    """知识 021 轴交互：一手固定在轴键，**轴音隔位重复**（B,A,B,C,B…）。

    识别方法直接取自知识 021 的"轴结构的识别方法"：同一键在奇数位或偶数位反复出现。
    另一手的位置必须**有变化**（≥2 个不同键），否则那是纵连/大宇宙而不是轴交互。
    """
    hits: list[ConfigHit] = []
    min_len = PARAMS["min_len_axis"]
    for a, b in runs:
        keys = [s.keys[0] for s in slots[a:b + 1]]
        for phase in (0, 1):
            i = phase
            while i < len(keys):
                axis = keys[i]
                # 从 i 出发，向后找"每隔一位都是 axis"的最长段
                j = i
                while j + 2 < len(keys) and keys[j + 2] == axis and keys[j + 1] != axis:
                    j += 2
                seg_len = j - i + 1
                if seg_len >= min_len and (j + 1) <= len(keys):
                    others = [keys[t] for t in range(i + 1, j, 2)]
                    if len(set(others)) >= 2:
                        hits.append(_span(
                            slots, a + i, a + j, "轴交互",
                            f"轴 {axis}，另一手 {','.join(str(x) for x in others)}",
                            {"axis": axis, "others": others,
                             "length": seg_len,
                             "max_reach": max(cdist(axis, o) for o in others)}))
                        i = j + 1
                        continue
                i += 2
    return _dedup(hits)


def detect_triangle(slots: Sequence[Slot], runs: Sequence[tuple[int, int]]
                    ) -> list[ConfigHit]:
    """知识 020 三角交互：3 tap 一组，每组占 2 或 3 个**邻近**键，组间换位置。

    操作化：把匀拍单点段按 3 个一组切开，一组合法当且仅当
    ①键集大小 ∈ {2,3}；②键集在 8 键环上的跨度 ≤ 3（= 落在 4 个连续位置内，
    覆盖知识 020 的全部引文键集：{3,4}/{5,6}/{2,3,4}/{1,3,4}/{5,6,8}…）；
    ③组内**必须转向**（两步的有向步进不同号），否则 `7,6,5` 这种单向音阶
    会被误判成三角——知识 020 的全部引文三元组都转向
    （`3,4,3` +1/−1、`3,4,2` +1/−2、`2,4,3` +2/−1…）；
    ④相邻两组键集**不相交**（换边）。至少连续 2 组才算命中。
    为避免同一段被三种切分偏移重复命中，**每段只保留链最长的那个偏移**。
    ⚠️ 正三角 / 反三角**不判**——知识 020 明说区分依据目前只有手感。
    """
    hits: list[ConfigHit] = []
    min_len = PARAMS["min_len_triangle"]
    for a, b in runs:
        # 知识 020 原文是"3 个 **tap** 为一组"，星星链不算三角交互
        if any(slots[t].has_slide for t in range(a, b + 1)):
            continue
        keys = [s.keys[0] for s in slots[a:b + 1]]
        best: tuple[int, list] | None = None
        for off in (0, 1, 2):
            chains = _triangle_chains(keys, off)
            total = sum(len(c) for c in chains)
            if best is None or total > best[0]:
                best = (total, chains)
        if not best or not best[1]:
            continue
        for chain in best[1]:
            s0 = a + chain[0][0]
            s1 = a + chain[-1][0] + 2
            if s1 - s0 + 1 < min_len:
                continue
            hits.append(_span(
                slots, s0, s1, "三角交互",
                " | ".join(",".join(str(x) for x in keys[p:p + 3]) for p, _ in chain),
                {"n_groups": len(chain),
                 "key_sets": [sorted(k) for _, k in chain],
                 "occupancy": [len(k) for _, k in chain]}))
    return _dedup(hits)


def _triangle_valid(tri: Sequence[int]) -> bool:
    ks = frozenset(tri)
    if not (2 <= len(ks) <= 3) or _arc_span(ks) > 3:
        return False
    s1, s2 = cstep(tri[0], tri[1]), cstep(tri[1], tri[2])
    return s1 != 0 and s2 != 0 and (s1 > 0) != (s2 > 0)   # 必须转向


def _triangle_chains(keys: Sequence[int], off: int) -> list[list[tuple[int, frozenset]]]:
    """在给定切分偏移下，找出所有"连续 ≥2 个合法三元组"的链。"""
    chains: list[list[tuple[int, frozenset]]] = []
    cur: list[tuple[int, frozenset]] = []
    t = off
    while t + 2 < len(keys) + 0:
        tri = keys[t:t + 3]
        if len(tri) < 3:
            break
        ks = frozenset(tri)
        ok = _triangle_valid(tri) and (not cur or not (ks & cur[-1][1]))
        if ok:
            cur.append((t, ks))
        else:
            if len(cur) >= 2:
                chains.append(cur)
            cur = [(t, ks)] if _triangle_valid(tri) else []
        t += 3
    if len(cur) >= 2:
        chains.append(cur)
    return chains


def _arc_span(ks: Iterable[int]) -> int:
    """键集在 8 键环上的最小跨度（首尾之间的最短弧长）。"""
    s = sorted(set(int(x) for x in ks))
    if len(s) <= 1:
        return 0
    best = 8
    for i in range(len(s)):
        rot = [(x - s[i]) % 8 for x in s]
        best = min(best, max(rot))
    return best


def _pairs_of(keys: Sequence[int]) -> list[tuple[int, int]]:
    """把匀拍单点序列按"两手各一个音为一对"切成 pair 序列（知识 022）。"""
    return [(keys[i], keys[i + 1]) for i in range(0, len(keys) - 1, 2)]


def _dedup_consecutive(pairs: Sequence[tuple[int, int]]
                       ) -> list[tuple[int, int, int, int]]:
    """折叠连续重复的 pair（知识 022"每对重复两次再走"的写法）。

    返回 ``[(a, b, first_pair_index, repeat), ...]``。
    """
    out: list[list] = []
    for i, p in enumerate(pairs):
        if out and (out[-1][0], out[-1][1]) == p:
            out[-1][3] += 1
        else:
            out.append([p[0], p[1], i, 1])
    return [(a, b, i, r) for a, b, i, r in out]


def detect_stair(slots: Sequence[Slot], runs: Sequence[tuple[int, int]]
                 ) -> list[ConfigHit]:
    """知识 022 楼梯交互 / 023 逆楼梯（方向盘）。

    两者共用"成对推进、每步只走相邻键"的骨架，靠**两手方向**与**对位关系**区分：

    - 楼梯（022）：两手**反向**移动（+1 / −1），且不构成 180° 对位；
    - 逆楼梯/方向盘（023）：两手**同向**移动，且每一对**始终 180° 对位**。
    """
    hits: list[ConfigHit] = []
    min_pairs = PARAMS["min_pairs_stair"]
    for a, b in runs:
        keys = [s.keys[0] for s in slots[a:b + 1]]
        for off in (0, 1):
            raw = _pairs_of(keys[off:])
            if len(raw) < min_pairs:
                continue
            folded = _dedup_consecutive(raw)
            i = 0
            while i < len(folded):
                j = i
                kinds: set[str] = set()
                while j + 1 < len(folded):
                    (a0, b0, _, _), (a1, b1, _, _) = folded[j], folded[j + 1]
                    da, db = cstep(a0, a1), cstep(b0, b1)
                    if abs(da) != 1 or abs(db) != 1:
                        break
                    opp0 = cdist(a0, b0) == 4
                    opp1 = cdist(a1, b1) == 4
                    if da == -db:                      # 两手反向 → 楼梯
                        kind = "楼梯交互"
                    elif da == db and opp0 and opp1:   # 同向 + 始终对位 → 方向盘
                        kind = "逆楼梯/方向盘"
                    else:
                        break
                    if kinds and kind not in kinds:
                        break
                    kinds.add(kind)
                    j += 1
                n_pairs = j - i + 1
                if n_pairs >= min_pairs and kinds:
                    kind = kinds.pop()
                    s0 = a + off + folded[i][2] * 2
                    last = folded[j]
                    s1 = min(a + off + (last[2] + last[3]) * 2 - 1, b)
                    seq = " → ".join(f"({x},{y})" for x, y, _, _ in folded[i:j + 1])
                    reps = [r for _, _, _, r in folded[i:j + 1]]
                    hits.append(_span(
                        slots, s0, s1, kind, seq,
                        {"n_pairs": n_pairs, "repeat": reps,
                         "style": "每对重复" if max(reps) > 1 else "不重复直接走",
                         "reversal": _has_reversal(folded[i:j + 1])}))
                    i = j + 1
                else:
                    i += 1
    return _dedup(hits)


def _has_reversal(folded: Sequence[tuple[int, int, int, int]]) -> bool:
    """是否出现折返（知识 023 的固有特征）。"""
    dirs = [cstep(folded[k][0], folded[k + 1][0]) for k in range(len(folded) - 1)]
    return any(dirs[k] * dirs[k + 1] < 0 for k in range(len(dirs) - 1))


def detect_macrocosm(slots: Sequence[Slot], runs: Sequence[tuple[int, int]]
                     ) -> list[ConfigHit]:
    """知识 024 大宇宙：严格 4 键 = 两对**相邻**键，每手在自己那对上横跳。

    操作化：匀拍单点段里，奇数位全落在 pair A、偶数位全落在 pair B，
    A/B 各恰好 2 个**环上相邻**的键且不相交，且两边都真的在横跳（各用满 2 个键）。
    """
    hits: list[ConfigHit] = []
    min_len = PARAMS["min_len_macrocosm"]
    for a, b in runs:
        keys = [s.keys[0] for s in slots[a:b + 1]]
        i = 0
        while i < len(keys):
            j = i
            while j + 1 < len(keys):
                even = set(keys[i:j + 2:2])
                odd = set(keys[i + 1:j + 2:2])
                if (len(even) <= 2 and len(odd) <= 2 and not (even & odd)
                        and all(_arc_span(s) <= 1 for s in (even, odd) if len(s) == 2)):
                    j += 1
                else:
                    break
            even, odd = set(keys[i:j + 1:2]), set(keys[i + 1:j + 1:2])
            ln = j - i + 1
            if (ln >= min_len and len(even) == 2 and len(odd) == 2
                    and _arc_span(even) == 1 and _arc_span(odd) == 1
                    and not (even & odd)):
                hits.append(_span(
                    slots, a + i, a + j, "大宇宙",
                    f"手A {'↔'.join(str(x) for x in sorted(even))}、"
                    f"手B {'↔'.join(str(x) for x in sorted(odd))}",
                    {"pair_a": sorted(even), "pair_b": sorted(odd), "length": ln}))
                i = j + 1
            else:
                i += 1
    return _dedup(hits)


def detect_interaction(slots: Sequence[Slot], runs: Sequence[tuple[int, int]]
                       ) -> list[ConfigHit]:
    """知识 018 普通交互：匀拍单点、逐音换手的**统称**（交互型配置的母类）。

    操作化：匀拍单点段长度 ≥6，且不是纯纵连（不存在同键连续 ≥3）。
    等效速度按知识 015/018 的 ``NPS = (分音分母/4)·BPM/60`` 给出。
    """
    hits: list[ConfigHit] = []
    min_len = PARAMS["min_len_interaction"]
    for a, b in runs:
        if b - a + 1 < min_len:
            continue
        keys = [s.keys[0] for s in slots[a:b + 1]]
        if _max_same_run(keys) >= 3:
            continue
        n_slide = sum(1 for t in range(a, b + 1) if slots[t].has_slide)
        if n_slide * 4 > b - a + 1:
            continue      # 星星链不是"逐音换手的击打交互"（知识 018 的基础形是 tap）
        dt = slots[b].time - slots[a].time
        nps = (b - a) / dt if dt > 0 else 0.0
        hits.append(_span(
            slots, a, b, "普通交互",
            f"{b - a + 1} 连，{','.join(str(x) for x in keys[:12])}"
            + ("…" if len(keys) > 12 else ""),
            {"length": b - a + 1, "nps": round(nps, 2),
             "bpm": slots[a].bpm, "divisor": slots[a].divisor,
             "n_keys": len(set(keys)), "n_slide_slots": n_slide}))
    return hits


def _max_same_run(keys: Sequence[int]) -> int:
    best = cur = 1
    for i in range(1, len(keys)):
        cur = cur + 1 if keys[i] == keys[i - 1] else 1
        best = max(best, cur)
    return best if keys else 0


def detect_scatter(slots: Sequence[Slot], runs: Sequence[tuple[int, int]],
                   others: Sequence[ConfigHit]) -> list[ConfigHit]:
    """知识 025 散点：用键分散、步进混合，且**不满足任何已讲配置的几何条件**。

    操作化（残差定义）：匀拍单点段 ≥8 槽、用键 ≥5 个、步进幅度 ≥3 种，
    且该段与 019–024 的任何命中**不重叠**。
    """
    geo = {"纵连", "长纵连", "三角交互", "轴交互", "楼梯交互",
           "逆楼梯/方向盘", "大宇宙"}
    taken = [(h.slot_start, h.slot_end) for h in others if h.config in geo]
    hits: list[ConfigHit] = []
    for a, b in runs:
        if b - a + 1 < PARAMS["min_len_scatter"]:
            continue
        if any(not (b < s0 or a > s1) for s0, s1 in taken):
            continue
        if any(slots[t].has_slide or slots[t].has_hold for t in range(a, b + 1)):
            continue      # 知识 025：散点的音符形态全为单点（无星星/长条）
        keys = [s.keys[0] for s in slots[a:b + 1]]
        steps = {abs(cstep(keys[i], keys[i + 1])) for i in range(len(keys) - 1)}
        steps.discard(0)          # 同键重复不是"步进"
        if len(set(keys)) < PARAMS["scatter_min_keys"]:
            continue
        if len(steps) < PARAMS["scatter_min_step_kinds"]:
            continue
        hits.append(_span(
            slots, a, b, "散点",
            f"{b - a + 1} 连用 {len(set(keys))} 键，步进 {sorted(steps)}",
            {"length": b - a + 1, "n_keys": len(set(keys)),
             "steps": sorted(steps)}))
    return hits


def detect_double_run(slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 030 连续双押：连续 ≥3 个双押；并检测**侧边双押 23/67 红线**。

    ⚠️ "引导"的判定沿用 `docs/research/config-usage-survey.md` 的代理
    （前后 N 个槽内是否出现同键音），该代理**尚未经用户确认**。
    """
    hits: list[ConfigHit] = []
    n = len(slots)
    max_gap = PARAMS["uniform_max_gap_beats"]
    i = 0
    while i < n:
        if not slots[i].is_double:
            i += 1
            continue
        j = i
        while (j + 1 < n and slots[j + 1].is_double
               and 0 < slots[j + 1].beat - slots[j].beat <= max_gap):
            j += 1
        ln = j - i + 1
        if ln >= PARAMS["min_len_double_run"]:
            sets = [sorted(s.key_set) for s in slots[i:j + 1]]
            shared = sum(1 for t in range(i, j)
                         if slots[t].key_set & slots[t + 1].key_set)
            sub = ("绕圈" if shared >= ln - 1 and len(set(map(tuple, sets))) > 1
                   else "双押纵" if len(set(map(tuple, sets))) == 1 else "一般")
            hits.append(_span(
                slots, i, j, "连续双押",
                " ".join("/".join(str(x) for x in s) for s in sets[:8])
                + ("…" if len(sets) > 8 else ""),
                {"length": ln, "subtype": sub, "shared_steps": shared,
                 "sets": sets[:16]}))
        i = j + 1
    # --- 侧边双押红线 ---
    g = PARAMS["side_guard_slots"]
    for idx, s in enumerate(slots):
        if not s.is_double:
            continue
        ks = set(s.keys)
        if ks not in _SIDE_DOUBLES:
            continue
        neigh: set[int] = set()
        for t in range(max(0, idx - g), min(n, idx + g + 1)):
            if t != idx:
                neigh |= set(slots[t].keys)
        guided = bool(ks & neigh)
        hits.append(_span(
            slots, idx, idx, "侧边双押",
            f"{'/'.join(str(x) for x in sorted(ks))}"
            f"（{'周边有同键音' if guided else '周边无同键音'}）",
            {"keys": sorted(ks), "guided_proxy": guided}))
    return hits


def detect_single_double(slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 029 单双/双单/复合/夹心：双押与单点**交替循环**。

    操作化：连续时间槽的"单(S)/双(D)"串满足周期 2（SDSD/DSDS）或周期 3
    （DSS/SSD/SDS）的完整循环 ≥2 次，且 S、D 都出现。
    """
    hits: list[ConfigHit] = []
    n = len(slots)
    max_gap = PARAMS["uniform_max_gap_beats"]
    min_len = PARAMS["min_len_sd"]
    # 先切成"相邻且间隔不超过一拍"的连续块
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while (j + 1 < n and 0 < slots[j + 1].beat - slots[j].beat <= max_gap
               and slots[j + 1].n_keys >= 1):
            j += 1
        if j > i:
            blocks.append((i, j))
        i = j + 1
    for a, b in blocks:
        sd = "".join("D" if slots[t].is_double else "S" for t in range(a, b + 1))
        for period in (2, 3):
            t = 0
            while t + period * 2 <= len(sd):
                unit = sd[t:t + period]
                if len(set(unit)) < 2:
                    t += 1
                    continue
                reps = 1
                while sd[t + reps * period: t + (reps + 1) * period] == unit:
                    reps += 1
                ln = reps * period
                if reps >= 2 and ln >= min_len:
                    hits.append(_span(
                        slots, a + t, a + t + ln - 1, "单双/双单",
                        f"{unit} × {reps}",
                        {"unit": unit, "period": period, "cycles": reps,
                         "ratio": f"{unit.count('D')}:{unit.count('S')}"}))
                    t += ln
                else:
                    t += 1
    return _dedup(hits)


def detect_beatkeep(slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 027 定拍：一手在同一个键上**稳定打拍子**，另一手处理别的配置。

    操作化：某键 k 在**等间隔**的 ≥8 个时间槽上出现；且这段时间里
    ①有 ≥3 个非 k 的音（另一只手真的在干活）；
    ②不是"每两音敲一次轴"的严格交互（那是知识 021 轴交互）——
      判据取知识 027 的区别点：定拍手可以与另一只手**同刻**（`26,2,26,2`），
      或另一只手**会歇**（非 k 槽数 < k 槽数 − 1）。
    """
    hits: list[ConfigHit] = []
    n = len(slots)
    min_slots = PARAMS["min_slots_beatkeep"]
    by_key: dict[int, list[int]] = {}
    for s in slots:
        for k in s.keys:
            by_key.setdefault(k, []).append(s.index)
    for k, idxs in by_key.items():
        i = 0
        while i < len(idxs):
            j = i
            step: float | None = None
            while j + 1 < len(idxs):
                g = slots[idxs[j + 1]].beat - slots[idxs[j]].beat
                if g <= 0 or g > PARAMS["uniform_max_gap_beats"]:
                    break
                if step is None:
                    step = g
                elif abs(g - step) > PARAMS["uniform_rel_tol"] * step:
                    break
                j += 1
            cnt = j - i + 1
            if cnt >= min_slots:
                s0, s1 = idxs[i], idxs[j]
                n_k = cnt
                n_each = sum(1 for t in idxs[i:j + 1] if slots[t].n_keys >= 2)
                others = [t for t in range(s0, s1 + 1)
                          if k not in slots[t].keys]
                n_other_notes = sum(slots[t].n_keys for t in others) + \
                    sum(slots[t].n_keys - 1 for t in idxs[i:j + 1])
                is_metronome = (n_each >= 0.25 * n_k) or (len(others) < n_k - 1)
                if n_other_notes >= 3 and is_metronome:
                    hits.append(_span(
                        slots, s0, s1, "定拍",
                        f"键 {k} 等间隔 {n_k} 次（步 {step:.3f} 拍），"
                        f"另一手 {n_other_notes} 音",
                        {"key": k, "count": n_k, "step_beats": round(step or 0, 4),
                         "n_each_slots": n_each, "n_other_notes": n_other_notes}))
                    i = j + 1
                    continue
            i = max(j, i) + 1
    return _dedup(hits)


def detect_bullet(slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 028 子弹：同一个键在短时间内被打**恰好两次**。

    两种载体（知识 028）：①同键相邻两音（≥3 就是纵连，排除）；
    ②短双押串 `12,12` / `78,78`（每个键在 2 个槽内各出现两遍；≥3 次重复
    属于知识 030 的"双押纵"，排除）。
    """
    hits: list[ConfigHit] = []
    n = len(slots)
    max_gap = PARAMS["bullet_max_gap_beats"]
    # ① 同键相邻两音
    for i in range(n - 1):
        if slots[i].n_keys != 1 or slots[i + 1].n_keys != 1:
            continue
        if slots[i].has_slide or slots[i + 1].has_slide:
            continue      # 星星头 + 同键 tap 是"错位/星星串"的副产物，不按子弹计
        k = slots[i].keys[0]
        if slots[i + 1].keys[0] != k:
            continue
        if not (0 < slots[i + 1].beat - slots[i].beat <= max_gap):
            continue
        prev_same = i > 0 and slots[i - 1].n_keys == 1 and slots[i - 1].keys[0] == k
        next_same = (i + 2 < n and slots[i + 2].n_keys == 1
                     and slots[i + 2].keys[0] == k)
        if prev_same or next_same:
            continue          # 属于纵连
        hits.append(_span(slots, i, i + 1, "子弹", f"{k},{k}",
                          {"carrier": "同键相邻两音", "key": k}))
    # ② 短双押串
    for i in range(n - 1):
        if not (slots[i].is_double and slots[i + 1].is_double):
            continue
        if slots[i].key_set != slots[i + 1].key_set:
            continue
        if not (0 < slots[i + 1].beat - slots[i].beat <= max_gap):
            continue
        prev_same = i > 0 and slots[i - 1].key_set == slots[i].key_set
        next_same = i + 2 < n and slots[i + 2].key_set == slots[i].key_set
        if prev_same or next_same:
            continue          # 属于双押纵
        ks = "".join(str(x) for x in sorted(slots[i].key_set))
        hits.append(_span(slots, i, i + 1, "子弹", f"{ks},{ks}",
                          {"carrier": "短双押串", "keys": sorted(slots[i].key_set)}))
    return hits


def detect_misalign(res: ParseResult, slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 017 错位：星星头与**启动拍**之间塞入其他音符。

    与双押的界限（知识 017）：
    ①同刻的音是双押不是错位 → 区间取**开区间**；
    ②"双手同刻同任务（都拍星星头 / 都拍错位处 / 都拍启动拍）= 双押，不叫错位" →
    若**星星头同刻有 ≥2 条滑轨**（两手都在划）而中间那个音又是**双押**，
    则两手在同刻做同一件事，判为双押而非错位，跳过。
    同一个时间槽只产出一条命中（两条同刻滑轨不重复计数）。
    """
    hits: list[ConfigHit] = []
    eps = 1e-4
    n_slide_at: dict[int, int] = {}
    for nt in res.notes:
        if nt.kind == "slide_track":
            n_slide_at[nt.group_index] = n_slide_at.get(nt.group_index, 0) + 1
    seen: set[tuple[int, int]] = set()
    for nt in res.notes:
        if nt.kind != "slide_track" or nt.wait <= 0:
            continue
        t0, t1 = nt.time, nt.time + nt.wait
        inner = [s for s in slots if t0 + eps < s.time < t1 - eps]
        if not inner:
            continue
        if (n_slide_at.get(nt.group_index, 1) >= 2
                and all(s.n_keys >= 2 for s in inner)):
            continue          # 双手同刻同任务 → 双押，不叫错位
        if (inner[0].index, inner[-1].index) in seen:
            continue
        seen.add((inner[0].index, inner[-1].index))
        i0 = inner[0].index
        i1 = inner[-1].index
        hits.append(ConfigHit(
            config="错位",
            bar_start=min(nt.measure, inner[0].measure),
            bar_end=max(inner[-1].measure, nt.measure),
            slot_start=i0, slot_end=i1,
            t_start=t0, t_end=t1,
            n_slots=len(inner),
            n_notes=sum(s.n_notes for s in inner),
            evidence=f"{nt.key}{nt.shape}{nt.end_key} 头 → 错位音 "
                     + ",".join("/".join(str(k) for k in s.keys) for s in inner)
                     + " → 启动拍",
            detail={"head_key": int(nt.key), "shape": nt.shape,
                    "end_key": nt.end_key, "wait_sec": round(nt.wait, 4),
                    "n_inner": len(inner),
                    "inner_keys": [list(s.keys) for s in inner]}))
    return hits


def detect_skipbeat(slots: Sequence[Slot]) -> list[ConfigHit]:
    """知识 026 跳拍（**概念版粗判**）：逗号/空拍不规则。

    ⚠️ 知识 026 本身只是概念记录（用户明说待补充），所以这里只做**粗判**：
    以小节为单位，把小节内相邻音的拍间隔量化成"最小间隔的整数倍"，要求
    ①小节内 ≥5 个音；②倍数全部 ≤3（跨度更大的是休息，不是跳拍）；
    ③倍数至少两种取值；④"1 → ≥2" 或 "≥2 → 1" 的切换**出现 ≥2 次**
    （= 知识 026 例子 `A,,B,C,,D,E,,F,` 的反复插空，而不是一处单纯留白）。
    **不得据此下任何强结论。**
    """
    hits: list[ConfigHit] = []
    by_bar: dict[int, list[Slot]] = {}
    for s in slots:
        by_bar.setdefault(s.measure, []).append(s)
    for bar, ss in sorted(by_bar.items()):
        if len(ss) < 5:
            continue
        gaps = [ss[i + 1].beat - ss[i].beat for i in range(len(ss) - 1)]
        gaps = [g for g in gaps if g > 0]
        if len(gaps) < 4:
            continue
        unit = min(gaps)
        mult = [round(g / unit) for g in gaps]
        if max(mult) > 3 or len(set(mult)) < 2:
            continue
        binar = [1 if m == 1 else 2 for m in mult]
        runs = 1 + sum(1 for i in range(len(binar) - 1) if binar[i] != binar[i + 1])
        if runs < 4:
            continue      # 只有一处插空（1…2…1）不算跳拍，要反复交替
        hits.append(_span(slots, ss[0].index, ss[-1].index, "跳拍",
                          f"小节 {bar} 间隔倍数 {mult}",
                          {"multiples": mult, "unit_beats": round(unit, 4),
                           "alt_runs": runs}))
    return hits


def _dedup(hits: Sequence[ConfigHit]) -> list[ConfigHit]:
    """去掉被同名同范围（或被同名更长片段完全包含）的重复命中。"""
    out: list[ConfigHit] = []
    for h in sorted(hits, key=lambda x: (x.config, x.slot_start,
                                         -(x.slot_end - x.slot_start))):
        drop = False
        for k in out:
            if (k.config == h.config and k.slot_start <= h.slot_start
                    and k.slot_end >= h.slot_end):
                drop = True
                break
        if not drop:
            out.append(h)
    return sorted(out, key=lambda x: (x.slot_start, x.config))


# ---------------------------------------------------------------------------
# 无理代理（知识 008/009/010/011）
# ---------------------------------------------------------------------------


@dataclass
class MuriHit:
    kind: str          # 叠键 / 外键 / 撞尾
    measure: int
    time: float
    detail: str = ""


def detect_muri(res: ParseResult) -> list[MuriHit]:
    """无理事件的**代理检测**（知识 009/010/011 的阈值，非完整 MaiMuriDX 实现）。

    - **叠键**（009）：同一键上两个 note 间隔 < 33.3 ms（不含同刻 each）；
    - **外键**（010）：slide 启动后 200 ms 内，同侧（环距 ≤1）出现 tap/hold；
    - **撞尾**（011）：slide 结束时刻前 50 ms ~ 后 200 ms，其**终点键**出现 tap/hold。
    """
    out: list[MuriHit] = []
    hits = [n for n in res.notes
            if n.kind in ("tap", "hold", "slide_star") and n.key.isdigit()]
    by_key: dict[str, list[NoteEvent]] = {}
    for n in hits:
        by_key.setdefault(n.key, []).append(n)
    for k, ns in by_key.items():
        ns.sort(key=lambda x: x.time)
        for i in range(len(ns) - 1):
            dt = ns[i + 1].time - ns[i].time
            if 0 < dt < PARAMS["muri_overlap_sec"]:
                out.append(MuriHit("叠键", ns[i].measure, ns[i].time,
                                   f"键 {k} 间隔 {dt * 1000:.1f} ms"))
    for s in res.notes:
        if s.kind != "slide_track" or not s.key.isdigit():
            continue
        t_move = s.time + s.wait
        t_end = t_move + s.duration
        head = int(s.key)
        for n in hits:
            if n.kind == "slide_star" and abs(n.time - s.time) < 1e-6:
                continue
            if (t_move - 1e-6 < n.time <= t_move + PARAMS["muri_slidehead_sec"]
                    and cdist(int(n.key), head) <= 1 and not n.is_ex):
                out.append(MuriHit("外键", n.measure, n.time,
                                   f"slide {s.key}{s.shape}{s.end_key} 启动后 "
                                   f"{(n.time - t_move) * 1000:.0f} ms 同侧键 {n.key}"))
            if (s.end_key.isdigit() and int(n.key) == int(s.end_key)
                    and t_end - PARAMS["muri_tail_pre_sec"] <= n.time
                    <= t_end + PARAMS["muri_tail_post_sec"] and not n.is_ex):
                out.append(MuriHit("撞尾", n.measure, n.time,
                                   f"slide {s.key}{s.shape}{s.end_key} 尾 "
                                   f"{(n.time - t_end) * 1000:+.0f} ms 键 {n.key}"))
    return out


# ---------------------------------------------------------------------------
# 逐小节配置硬度
# ---------------------------------------------------------------------------


@dataclass
class BarHardness:
    measure: int
    n_notes: int
    n_slots: int
    nps: float
    term_kind: float
    term_move: float
    term_speed: float
    term_muri: float
    term_config: float
    raw: float
    hardness: float = 0.0      # 曲内归一 [0,1]
    hardness_class: str = ""   # 低 / 中 / 高（曲内三分位）
    configs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"measure": self.measure, "n_notes": self.n_notes,
                "n_slots": self.n_slots, "nps": round(self.nps, 3),
                "term_kind": round(self.term_kind, 4),
                "term_move": round(self.term_move, 4),
                "term_speed": round(self.term_speed, 4),
                "term_muri": round(self.term_muri, 4),
                "term_config": round(self.term_config, 4),
                "raw": round(self.raw, 4),
                "hardness_bar": round(self.hardness, 4),
                "hardness_class": self.hardness_class,
                "configs": list(self.configs)}


def bar_hardness(res: ParseResult, slots: Sequence[Slot] | None = None,
                 hits: Sequence[ConfigHit] | None = None,
                 muri: Sequence[MuriHit] | None = None) -> list[BarHardness]:
    """逐小节配置硬度分。

    公式（五项加权和，权重见 ``PARAMS['hardness_weights']``）::

        raw = 0.22·K + 0.22·M + 0.26·V + 0.10·U + 0.20·C

    - **K（note 种类，知识 003-1）**：小节内 note 的平均加权值
      ``(tap 1.0 / hold 1.2 / slide 轨 1.5 / touch 0.8，break +0.3，each +0.2)``，
      再线性压到 [0,1]：``K = clip((w̄ − 1.0) / 0.6, 0, 1)``；
      若该小节命中 `错位`（知识 003 说错位是同踩音同键位下最强的星星用法），
      ``K`` 额外 ``+0.25`` 后再截断。
    - **M（位移，知识 003-2）**：取两个口径的均值再除以环上最大位移 4——
      ``move_adj`` = 相邻时间槽之间的键位距离均值（跨槽所有键对取均值）；
      ``move_hand`` = 隔一个槽（同一只手）之间的距离均值。
      ⚠️ "同一只手 = 隔一个音"是**交替手序的代理**，不是真实手序。
    - **V（速度，知识 015/018）**：``NPS = 时间槽数 / 小节时长``，
      除以短爆发红线 22.2 键/秒后截断到 [0,1]（BPM 与分音的联动已经含在 NPS 里，
      所以知识 003-3 的 BPM 因素不再单列）。
    - **U（无理，知识 008/009/010/011）**：小节内叠键/外键/撞尾代理事件数，
      ``U = clip(n / 3, 0, 1)``。
    - **C（配置，知识 017–030）**：覆盖该小节的配置里 ``CONFIG_WEIGHT`` 的最大值。

    归一：``hardness_bar`` = 该曲**有 note 的小节**上对 ``raw`` 做 min-max；
    ``hardness_class`` = 曲内三分位（低 / 中 / 高）。
    """
    slots = list(build_slots(res)) if slots is None else list(slots)
    hits = list(detect_all(res, slots)) if hits is None else list(hits)
    muri = list(detect_muri(res)) if muri is None else list(muri)
    w = PARAMS["hardness_weights"]

    by_bar_notes: dict[int, list[NoteEvent]] = {}
    for n in res.notes:
        by_bar_notes.setdefault(n.measure, []).append(n)
    by_bar_slots: dict[int, list[Slot]] = {}
    for s in slots:
        by_bar_slots.setdefault(s.measure, []).append(s)
    by_bar_muri: dict[int, int] = {}
    for m in muri:
        by_bar_muri[m.measure] = by_bar_muri.get(m.measure, 0) + 1
    by_bar_cfg: dict[int, set[str]] = {}
    for h in hits:
        for b in range(h.bar_start, h.bar_end + 1):
            by_bar_cfg.setdefault(b, set()).add(h.config)

    if not by_bar_notes:
        return []
    lo, hi = min(by_bar_notes), max(by_bar_notes)
    out: list[BarHardness] = []
    for bar in range(lo, hi + 1):
        ns = by_bar_notes.get(bar, [])
        ss = by_bar_slots.get(bar, [])
        cfgs = by_bar_cfg.get(bar, set())
        if not ns:
            out.append(BarHardness(bar, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0,
                                   configs=tuple(sorted(cfgs))))
            continue
        # K
        tot = 0.0
        for n in ns:
            v = KIND_WEIGHT.get(n.kind, 1.0)
            if n.is_break:
                v += BREAK_BONUS
            if n.is_each:
                v += EACH_BONUS
            tot += v
        kbar = tot / len(ns)
        K = min(max((kbar - 1.0) / 0.6, 0.0), 1.0)
        if "错位" in cfgs:
            K = min(K + 0.25, 1.0)
        # M
        M = _move_term(ss)
        # V
        bpm = ns[0].bpm or 0.0
        bar_sec = (240.0 / bpm) if bpm > 0 else 0.0
        nps = (len(ss) / bar_sec) if bar_sec > 0 else 0.0
        V = min(nps / PARAMS["nps_redline"], 1.0)
        # U
        U = min(by_bar_muri.get(bar, 0) / 3.0, 1.0)
        # C
        C = max((CONFIG_WEIGHT.get(c, 0.0) for c in cfgs), default=0.0)
        raw = (w["kind"] * K + w["move"] * M + w["speed"] * V
               + w["muri"] * U + w["config"] * C)
        out.append(BarHardness(bar, len(ns), len(ss), nps, K, M, V, U, C, raw,
                               configs=tuple(sorted(cfgs))))

    vals = [b.raw for b in out if b.n_notes > 0]
    if vals:
        mn, mx = min(vals), max(vals)
        rng = (mx - mn) or 1.0
        for b in out:
            b.hardness = (b.raw - mn) / rng if b.n_notes > 0 else 0.0
        live = sorted(vals)
        q1 = live[int(len(live) / 3)] if len(live) >= 3 else mn
        q2 = live[int(2 * len(live) / 3)] if len(live) >= 3 else mx
        for b in out:
            if b.n_notes == 0:
                b.hardness_class = ""
            elif b.raw <= q1:
                b.hardness_class = "低"
            elif b.raw <= q2:
                b.hardness_class = "中"
            else:
                b.hardness_class = "高"
    return out


def _move_term(ss: Sequence[Slot]) -> float:
    """位移项（知识 003-2）：相邻槽位移与"同手"（隔一槽）位移的均值 / 4。"""
    if len(ss) < 2:
        return 0.0

    def mean_dist(step: int) -> float:
        vals = []
        for i in range(len(ss) - step):
            a, b = ss[i].keys, ss[i + step].keys
            if not a or not b:
                continue
            vals.append(sum(cdist(x, y) for x in a for y in b) / (len(a) * len(b)))
        return sum(vals) / len(vals) if vals else 0.0

    adj = mean_dist(1)
    hand = mean_dist(2) if len(ss) >= 3 else adj
    return min((0.5 * (adj + hand)) / 4.0, 1.0)


# ---------------------------------------------------------------------------
# 总入口
# ---------------------------------------------------------------------------


def detect_all(res: ParseResult, slots: Sequence[Slot] | None = None
               ) -> list[ConfigHit]:
    """跑全部配置检测器，返回按时间排序的命中列表（一个片段可命中多个配置）。"""
    slots = list(build_slots(res)) if slots is None else list(slots)
    if not slots:
        return []
    runs = uniform_runs(slots, min_len=4, single_only=True)
    hits: list[ConfigHit] = []
    hits += detect_vertical(slots)
    hits += detect_axis(slots, runs)
    hits += detect_triangle(slots, runs)
    hits += detect_stair(slots, runs)
    hits += detect_macrocosm(slots, runs)
    hits += detect_interaction(slots, runs)
    hits += detect_double_run(slots)
    hits += detect_single_double(slots)
    hits += detect_beatkeep(slots)
    hits += detect_bullet(slots)
    hits += detect_misalign(res, slots)
    hits += detect_skipbeat(slots)
    hits += detect_scatter(slots, runs, hits)
    return sorted(hits, key=lambda h: (h.slot_start, h.config))


def summarize(hits: Sequence[ConfigHit]) -> dict[str, dict]:
    """按配置名汇总片段数 / 槽数 / note 数。"""
    out: dict[str, dict] = {}
    for h in hits:
        d = out.setdefault(h.config, {"n_segments": 0, "n_slots": 0, "n_notes": 0})
        d["n_segments"] += 1
        d["n_slots"] += h.n_slots
        d["n_notes"] += h.n_notes
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n_segments"]))


def iter_bar_configs(hits: Sequence[ConfigHit]) -> Iterator[tuple[int, set[str]]]:
    """按小节产出命中的配置集合。"""
    by_bar: dict[int, set[str]] = {}
    for h in hits:
        for b in range(h.bar_start, h.bar_end + 1):
            by_bar.setdefault(b, set()).add(h.config)
    for b in sorted(by_bar):
        yield b, by_bar[b]
