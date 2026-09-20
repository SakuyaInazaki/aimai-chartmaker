#!/usr/bin/env python3
"""配置识别器（**手序版**）：从 `hands.assign()` 的左右手分配结果里认配置。

为什么有这个模块
----------------
用户 2026-09-19：

> 「不可能只有定拍、错位、大宇宙是按手定义的啊，**任何配置都应该按手定义**吧。」

`configs.py`（键位版）只看键位序列与时间几何，"手"是隐含的代理（"同一只手 = 隔一个音"）。
本模块**不看原始键位串**，只消费 :func:`chart_analysis.hands.assign` 的输出：
逐任务 `(t, bar, key, kind, hand, occupancy)`、`left_seq` / `right_seq` / `pairs`、
逐小节手序指标。每个配置的判据写成"左手做什么 / 右手做什么"。

规格来源
--------
- **`docs/hand-sequencing.md` §5**（17–30 的手级定义表）与 **§5.1**（星星篇 047–063）；
- 知识条目原文 **017–030**（用户讲授）、**033–046**（第二期）、**047–063**（第三期）；
- 机检判据草案：`docs/research/config-terms-ep2.md` §6、`config-terms-ep3.md` §6.2。

每个检测函数的 docstring 首行必须写出**依据的条目号**；凡超出条目原文的判据一律标
`agent 推断`。操作化阈值集中在 :data:`PARAMS`，**可整体替换**。

两个分数并存（不合成）
----------------------
`docs/hand-sequencing.md` §5.2 的结论：`CONFIG_WEIGHT` 量的是**体力/精度负担**，
手序难度量的是**手的调度余量**，二者在 388 谱上几乎相反。因此 :func:`bar_scores`
逐小节同时输出：

- ``physical_hardness``：沿用键位版 `configs.bar_hardness` 的五项加权和口径
  （配置项 C 改用**手序版命中**的 :data:`CONFIG_WEIGHT_PHYS`）；
- ``hand_difficulty``：直接取 `hands.py` 的逐小节 `hand_hardness`。

**两者不相加、不平均**（用户已撤回 `(D+H)/2` 那类合成排序）。

⚠️ 操作化参数的定位（2026-09-20 术语修正）
-----------------------------------------
:data:`PARAMS` 里的窗口长度、最短串长、容差**全是 agent 为了能跑代码而定的内部口径**，
**不是术语、不是分级、也不拿去问用户**——用户 2026-09-20 明确：制谱判断是谱师的定性
判断（「正常写谱谁去特意算这些事情」「根本就没有这些词吧」）。这些数字只在工具内部
当诊断用；要问用户的只有**必须用谱例问的定性问题**。
:data:`PENDING_USER_PARAMS` 因此已清空（见该处说明）。

CLI::

    PYTHONPATH=tools python3 -m chart_analysis.configs_hand <谱面关键字> [--bars a-b]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from . import hands as H
from .hands import (HandAssignment, HandTask, cdist, cstep, home_side,
                    is_chuzhang, page_depth, side_double_events)
from .simai_parser import NoteEvent, ParseResult

_BEATS_PER_MEASURE = 4.0
_RIGHT_HOME = frozenset({1, 2, 3, 4})     # 知识 006 分页：右手主场
_LEFT_HOME = frozenset({5, 6, 7, 8})
_SIDE_DOUBLES = ({2, 3}, {6, 7})          # 知识 030 铁律：侧边双押只指 23 与 67

# ---------------------------------------------------------------------------
# 操作化参数（**agent 设定**，不是知识条目原文）
# ---------------------------------------------------------------------------

PARAMS: dict = {
    # --- 匀拍 / 窗口 ---
    "uniform_rel_tol": 0.12,          # 相邻槽拍间隔的相对离散度上限
    "uniform_max_gap_beats": 1.0,     # 单个间隔的绝对上限（拍）
    "min_len_interaction": 6,         # 018 普通交互
    "min_len_axis": 5,                # 021 轴交互 = BABCB 五音
    "min_len_triangle": 6,            # 020 三角 = 至少 2 组
    "min_pairs_stair": 3,             # 022/023 至少 3 对推进
    "min_len_macrocosm": 8,           # 024 大宇宙
    #: 024「每只手在自己那对相邻键之间**来回横跳**」的 trill 判据（agent 操作化）
    "macrocosm_min_key_alt": 0.6,
    "min_len_scatter": 8,             # 025 散点
    "min_len_vertical": 3,            # 019 纵连门槛
    "long_vertical": 6,               # >5 连 = 长纵连
    "min_len_double_run": 3,          # 030 连续双押
    "min_len_sd": 6,                  # 029 单双/双单
    "min_slots_beatkeep": 8,          # 027 定拍：固定手至少 8 槽
    "bullet_max_gap_beats": 0.5,      # 028 子弹
    "scatter_min_keys": 4,            # 025 每手键集合 ≥4（§5 规格）
    "scatter_min_step_kinds": 3,      # 025 同手步进种类 ≥3
    "require_uniform_interaction": True,   # 018/020–025 是否要求匀拍（知识 018「匀拍节奏下」）
    # --- 手序阈值 ---
    #: 019 纵连"一手连打 vs 两手拆"的分界：hands.t_required(0) = 83.3 ms（知识 015 反推）。
    #: **只对长度 >`vertical_single_hand_max` 的纵连生效**（知识 019 用户补充：长度优先）
    "vertical_split_ms": 83.3,
    #: 019 用户补充：「3 个以内」的纵连稍快也可单手（用户 2026-09-20：**只要需要击打
    #: 按键的 note 都算** —— tap / hold 头 / 星星头，见 `hands._HIT_KINDS`）
    "vertical_single_hand_max": 3,
    #: 043 出张的**段级**成段口径：一段 slide/hold 钉住一只手的窗口里，出现几个出张落点
    #: 才把这一段记成"出张"配置。出张本身是布尔（`hands.is_chuzhang`，知识 064），
    #: **没有分档**；这个 2 只是"几个落点才值得单独记一段"的**内部计数口径**（agent 操作化）。
    "chuzhang_min_cross": 2,
    #: hands.assign 的段首起手；None = 交给 hands 自己选（知识 060 的逐段枚举未实现）
    "start_hand": None,
    #: 是否允许"一只手一次吃两个对象"（知识 042/055b/060②）——默认关，ep3 §5.4-3 要求实测前不放宽
    "one_hand_two_objects": False,
    #: 划轨手的归属规则：'assigned' = 直接用 hands.assign 的结果（知识 054 允许换手）
    "slide_owner_rule": "assigned",
    # --- 星星族 ---
    "tapslide_min_units": 3,          # 049/053 连续拍滑最少单元数
    "tapslide_rel_tol": 0.15,         # 拍滑单元间隔的相对离散度
    #: 拍滑段内允许"拍头手 ≠ 划轨手"的比例上限（知识 054：换手是自由度，agent 操作化）
    "tapslide_switch_tol": 0.34,
    "double_tapslide_min": 3,         # 050 双压连续拍滑
    "onestroke_min_len": 2,           # 057 一笔画链长 ≥2（知识 057「至少 2 条星星」）
    "onestroke_gap_beats": 1.25,      # 链内"时间首尾相接"的容差（拍）
    "wave_min_roots": 3,              # 058 挥手段：同头 ≥3 根
    "dealing_min_roots": 4,           # 060 同起点拍滑：同头 ≥4 根
    #: 061 三叉戟：教程"从一个点放射三根"是**形状**说法；ep3 §5.1 把 Party 4U
    #: m051（同头 6 根全给一只手）也列作三叉戟谱例 → 实现成"≥3 根且全给一只手"
    "trident_roots": 3,
    #: 058/061「段内不换手」的宽容度：主手占比下限（agent 操作化）
    "same_head_one_hand_share": 0.8,
    "radiate_window_beats": 8.0,      # 同头放射的时间窗（拍）
    "drumbeat_min_units": 6,          # 062 鼓动段（教程口径 ≥6 根）
    "pierce_min_roots": 3,            # 055 穿心 ≥3 根
    #: 056 绕圈星星：ep3 §6.2 草案写 ≥3 根，官谱 `112-Mysterious Destiny`
    #: m020–m023 的弧线接力是**两根一组**（8^3→3^6）→ 按 ground truth 改 2
    "circle_min_roots": 2,
    "defuse_min_pending": 3,          # 063 拆弹：同时"已落头未划完" ≥3
    "scythe_min_walk": 8,             # 033 死镰段：走圈 tap ≥8
    #: 033 走圈 tap 里"主手"的占比下限（agent 操作化；知识 033 只说"一手走圈"）
    "scythe_walk_share": 0.55,
    #: 035 如龙段：ep2 §6 草案写 ≥3 个细胞，但官谱 Outlaw m058–m061 每组只有 **2**
    #: 个完整细胞（第三个被小节边界截断）——按 ground truth 下调到 2（agent 操作化）
    "dragon_min_cells": 2,
    "sweep_min_len": 6,               # 036 二连扫
    "backhand_min_len": 4,            # 037 反手：≥4 个连续音两手同半圈
    "nplus1_min_cycles": 3,           # 039/041
    "nplus1_min_cycles_3": 2,         # 040
    # --- 硬度分（沿用键位版口径；权重见 configs.bar_hardness 的 docstring）---
    "nps_redline": 22.2,
    "hardness_weights": {"kind": 0.22, "move": 0.22, "speed": 0.26,
                         "muri": 0.10, "config": 0.20},
}

#: **已清空（2026-09-20）**。
#:
#: 原 v1（提交 4942759）列过 6 个"待用户对齐"口径，2026-09-19 的用户讲授结掉 5 个，
#: 留下 ``chuzhang_min_cross``（"几个落点才算一段出张"）。用户 2026-09-20 指出这类
#: 量化问题根本不该抛给谱师（「出张本来是一件很小的事啊，怎么就分档了？」
#: 「怎么能量化这些事情呢？」），因此**最后一项也撤出**：它是 agent 的内部计数口径，
#: 不是待裁定项。
#:
#: 规矩：**agent 自造的阈值/分档/合成分不得作为问题抛给用户**；要问用户的只能是
#: 必须用谱例问的**定性**问题（例如某一段到底算不算出张/反手）。
PENDING_USER_PARAMS: tuple[str, ...] = ()

#: 已按用户口径结案的口径（保留名字，供报告对照）
RESOLVED_USER_PARAMS: dict[str, str] = {
    "vertical_split_ms": "019 补充：长度优先，≤3 个击打音可单手",
    "side_double_guard_slots": "030：改用 `hands.side_double_events` 的**引导八型**（定性）",
    "start_hand": "067：段首起手 = 离上一段结束近的手",
    "one_hand_two_objects": "066：保持关闭（写谱不以手法为导向）",
    "slide_owner_rule": "065：头与条可分属两手、无偏好",
    "chuzhang_min_cross": "064：出张是布尔判定 + 计数，不分档；成段口径属内部操作化",
}

#: 体力/精度硬度的配置权重（沿用键位版 `configs.CONFIG_WEIGHT`，星星族为本轮新增，
#: **全部是 agent 设定**——知识 003 只给了定性顺序）
CONFIG_WEIGHT_PHYS: dict[str, float] = {
    # —— 17–30（与 configs.CONFIG_WEIGHT 逐条一致，不改动 ——
    "大宇宙": 1.00, "错位": 0.90, "双押纵": 0.90, "逆楼梯/方向盘": 0.85,
    "连续双押": 0.80, "散点": 0.80, "长纵连": 0.75, "楼梯交互": 0.70,
    "三角交互": 0.70, "定拍": 0.65, "轴交互": 0.60, "单双/双单": 0.60,
    "跳拍": 0.50, "纵连": 0.50, "普通交互": 0.45, "子弹": 0.40,
    "侧边双押": 0.95,
    # —— 第二期（033–045）——
    "死镰段": 0.90, "如龙段": 0.90, "反手": 0.85, "出张": 0.75,
    "二连扫": 0.65, "2+1": 0.45, "3+1": 0.50, "N+1": 0.70,
    # —— 第三期星星篇（047–063）——
    "一笔画": 0.55, "连续拍滑": 0.60, "夹键拍滑": 0.70, "双压连续拍滑": 0.75,
    "同起点拍滑": 0.75, "三叉戟": 0.70, "挥手段": 0.70, "鼓动段": 0.85,
    "穿心": 0.75, "大风车": 0.85, "绕圈星星": 0.55, "井字星星": 0.80,
    "CYCLES型星星": 0.80, "拆弹": 0.75,
}

#: note 种类权重（与键位版 `configs.KIND_WEIGHT` 同口径）
KIND_WEIGHT: dict[str, float] = {
    "tap": 1.0, "hold": 1.2, "slide_star": 1.0, "slide_track": 1.5,
    "touch": 0.8, "touch_hold": 1.0,
}
BREAK_BONUS = 0.3
EACH_BONUS = 0.2


# ---------------------------------------------------------------------------
# 基础结构
# ---------------------------------------------------------------------------


@dataclass
class HSlot:
    """一个"手级时间槽"：同刻（`simul_tol` 内）的全部任务 + 各自的手。"""

    index: int
    t: float
    beat: float
    measure: int
    bpm: float
    tasks: tuple[int, ...] = ()          # 任务下标（进 HandAssignment.tasks）
    hands: tuple[str, ...] = ()          # 与 tasks 同序
    kinds: tuple[str, ...] = ()
    keys: tuple[int | None, ...] = ()

    # ---- 视图 ----
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def _hit_idx(self) -> tuple[int, ...]:
        """"击打任务"的位置下标：排除**正在划的轨道**。

        与键位版 `configs.Slot.keys`（不含 ``slide_track``）同口径——
        这样两版的"单点槽 / 双押槽"切分可比；轨道占手的信息仍在 ``tasks`` 里。
        """
        return tuple(i for i, k in enumerate(self.kinds)
                     if k not in ("slide", "wifi"))

    @property
    def n_hits(self) -> int:
        return len(self._hit_idx)

    @property
    def hit_hands(self) -> tuple[str, ...]:
        return tuple(self.hands[i] for i in self._hit_idx)

    @property
    def hit_keys(self) -> tuple[int | None, ...]:
        return tuple(self.keys[i] for i in self._hit_idx)

    def hit_key(self, hand: str) -> int | None:
        for i in self._hit_idx:
            if self.hands[i] == hand or (self.hands[i] == "LR" and hand in ("L", "R")):
                return self.keys[i]
        return None

    def hand_task(self, hand: str) -> int | None:
        """该槽里归 ``hand`` 的任务下标（多于一个时取第一个）。"""
        for ti, h in zip(self.tasks, self.hands):
            if h == hand or (h == "LR" and hand in ("L", "R")):
                return ti
        return None

    def hand_key(self, hand: str) -> int | None:
        ti = self.hand_task(hand)
        if ti is None:
            return None
        i = self.tasks.index(ti)
        return self.keys[i]

    @property
    def is_pair(self) -> bool:
        """双押 = 同刻 ≥2 个**击打**任务（划到一半的轨道不算双押的一半）。

        ⚠️ **不要求两只手**：知识 036 明说相邻两键可以由一只手扫过去，
        `hands.py` 也为此开了 ``w_sweep_pair`` 的口子；谱面上写的仍然是双押。
        两键是不是真的落在两只手上由 :attr:`two_hands` 单独报。
        """
        return self.n_hits >= 2

    @property
    def two_hands(self) -> bool:
        return self.hit_key("L") is not None and self.hit_key("R") is not None

    @property
    def single_hand(self) -> str:
        """单任务槽的那只手；不是单任务槽时返回 ``''``。"""
        if self.n_hits != 1:
            return ""
        return self.hit_hands[0]

    @property
    def has_slide(self) -> bool:
        return any(k in ("slide", "wifi") for k in self.kinds)

    @property
    def has_star(self) -> bool:
        return "star" in self.kinds

    @property
    def hit_key_set(self) -> frozenset[int]:
        return frozenset(k for k in self.hit_keys if k is not None)

    @property
    def key_set(self) -> frozenset[int]:
        return frozenset(k for k in self.keys if k is not None)


@dataclass
class ConfigHitH:
    """一个**手级**配置片段。一个片段可同时命中多个配置（各自一条）。"""

    config: str
    bar_start: int
    bar_end: int
    slot_start: int
    slot_end: int
    t_start: float
    t_end: float
    n_slots: int
    n_tasks: int
    hands_pattern: str = ""      # 左右手角色的一行速记，如 "L:定拍(5) R:变化(4,6)"
    evidence: str = ""           # 手级证据（说明这一段为什么命中）
    source: str = ""             # 依据的知识条目号，如 "024" / "057"
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"config": self.config, "bar_start": self.bar_start,
                "bar_end": self.bar_end, "slot_start": self.slot_start,
                "slot_end": self.slot_end,
                "t_start": round(self.t_start, 4), "t_end": round(self.t_end, 4),
                "n_slots": self.n_slots, "n_tasks": self.n_tasks,
                "hands_pattern": self.hands_pattern,
                "evidence": self.evidence, "source": self.source,
                "detail": self.detail}


@dataclass
class HandCtx:
    """一次检测所需的全部上下文（只读）。"""

    res: ParseResult
    ha: HandAssignment
    slots: list[HSlot]
    p: dict

    # ---- 便捷访问 ----
    def task(self, ti: int) -> HandTask:
        return self.ha.tasks[ti]

    def hand(self, ti: int) -> str:
        return self.ha.task_hand[ti]

    def head_task(self, ti: int) -> int:
        """轨道任务的星星头任务下标（被拍划吞掉时 = 吞掉它的那个任务）。"""
        return self.ha.tasks[ti].star_idx

    def slot_of_task(self) -> dict[int, int]:
        return self._t2s

    #: 只含**划到一半的轨道**、没有任何击打的槽（启动拍那一刻）——
    #: 序列型检测器不看它们，否则会把一串交互/双押从中间切断
    all_slots: list["HSlot"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.all_slots:
            self.all_slots = list(self.slots)
        # 序列层只保留"有击打"的槽，并重排下标（与键位版 `configs.Slot` 同口径）
        hit = [s for s in self.all_slots if s.n_hits >= 1]
        for i, s in enumerate(hit):
            s.index = i
        self.slots = hit
        self._t2s: dict[int, int] = {}
        times = [s.t for s in hit]
        import bisect
        for s in self.all_slots:
            if s.n_hits >= 1:
                idx = s.index
            else:
                j = bisect.bisect_right(times, s.t) - 1
                idx = max(0, min(len(hit) - 1, j)) if hit else 0
            for ti in s.tasks:
                self._t2s[ti] = idx


def build_hslots(ha: HandAssignment, p: dict | None = None) -> list[HSlot]:
    """把 `HandAssignment` 的任务按同刻归并成手级时间槽。"""
    pp = dict(PARAMS)
    if p:
        pp.update(p)
    tol = ha.params.get("simul_tol", H.PARAMS["simul_tol"])
    out: list[HSlot] = []
    cur: list[int] = []
    tasks = ha.tasks
    for i, t in enumerate(tasks):
        if cur and t.t - tasks[cur[0]].t > tol:
            out.append(_mk_slot(len(out), ha, cur))
            cur = []
        cur.append(i)
    if cur:
        out.append(_mk_slot(len(out), ha, cur))
    _fix_slot_beats(out, ha)
    return out


def _fix_slot_beats(slots: list["HSlot"], ha: HandAssignment) -> None:
    """修正只含 slide 轨道的槽的拍位置。

    ``HandTask.beat`` 对轨道任务记的是**星星头**的拍，而 ``t`` 已经是启动拍——
    两者差一个四分音符（知识 014）。用所有**非轨道**任务的 ``(t, beat)`` 做分段
    线性插值把轨道槽的拍补正，否则 `uniform_runs_h` 会在星星段上算出负间隔。
    """
    ref = sorted({(round(t.t, 6), t.beat) for t in ha.tasks
                  if t.kind not in ("slide", "wifi")})
    for s in slots:
        if any(k not in ("slide", "wifi") for k in s.kinds):
            hits = [ha.tasks[ti].beat for ti, k in zip(s.tasks, s.kinds)
                    if k not in ("slide", "wifi")]
            s.beat = min(hits)
            continue
        if not ref:
            t0 = ha.tasks[s.tasks[0]]
            spb = 60.0 / t0.bpm if t0.bpm else 0.5
            s.beat = t0.beat + 1.0
            continue
        lo = None
        hi = None
        for t, b in ref:
            if t <= s.t + 1e-9:
                lo = (t, b)
            elif hi is None:
                hi = (t, b)
                break
        if lo and hi and hi[0] > lo[0]:
            s.beat = lo[1] + (s.t - lo[0]) * (hi[1] - lo[1]) / (hi[0] - lo[0])
        else:
            t0 = ha.tasks[s.tasks[0]]
            s.beat = t0.beat + 1.0


def _mk_slot(idx: int, ha: HandAssignment, tis: Sequence[int]) -> HSlot:
    head = ha.tasks[tis[0]]
    return HSlot(
        index=idx, t=head.t, beat=head.beat, measure=head.measure, bpm=head.bpm,
        tasks=tuple(tis),
        hands=tuple(ha.task_hand[i] for i in tis),
        kinds=tuple(ha.tasks[i].kind for i in tis),
        keys=tuple(ha.tasks[i].key for i in tis),
    )


def make_ctx(res: ParseResult, ha: HandAssignment | None = None,
             p: dict | None = None) -> HandCtx:
    """``parse_chart`` 的结果 → 手序上下文（顺带跑 `hands.assign`）。"""
    pp = dict(PARAMS)
    if p:
        pp.update(p)
    if ha is None:
        hp = {}
        if pp.get("start_hand") in ("L", "R"):
            hp["force_first_hand"] = pp["start_hand"]
        ha = H.assign(res, hp or None)
    allslots = build_hslots(ha, pp)
    return HandCtx(res=res, ha=ha, slots=list(allslots), p=pp,
                   all_slots=allslots)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _span(ctx: HandCtx, a: int, b: int, config: str, source: str,
          evidence: str, hands_pattern: str = "",
          detail: dict | None = None) -> ConfigHitH:
    sub = ctx.slots[a:b + 1]
    return ConfigHitH(
        config=config, bar_start=sub[0].measure, bar_end=sub[-1].measure,
        slot_start=a, slot_end=b, t_start=sub[0].t, t_end=sub[-1].t,
        n_slots=len(sub), n_tasks=sum(s.n_tasks for s in sub),
        hands_pattern=hands_pattern, evidence=evidence, source=source,
        detail=detail or {})


def _span_tasks(ctx: HandCtx, tis: Sequence[int], config: str, source: str,
                evidence: str, hands_pattern: str = "",
                detail: dict | None = None) -> ConfigHitH:
    """按任务下标集合造片段（星星族用：任务未必落在连续槽上）。"""
    t2s = ctx.slot_of_task()
    si = sorted({t2s[i] for i in tis if i in t2s})
    if not si:
        ts = [ctx.task(i) for i in tis]
        return ConfigHitH(config=config, bar_start=min(t.measure for t in ts),
                          bar_end=max(t.measure for t in ts),
                          slot_start=-1, slot_end=-1,
                          t_start=min(t.t for t in ts), t_end=max(t.t for t in ts),
                          n_slots=0, n_tasks=len(tis),
                          hands_pattern=hands_pattern, evidence=evidence,
                          source=source, detail=detail or {})
    return _span(ctx, si[0], si[-1], config, source, evidence, hands_pattern,
                 detail)


def uniform_runs_h(ctx: HandCtx, min_len: int = 4, single_only: bool = True,
                   rel_tol: float | None = None,
                   max_gap_beats: float | None = None) -> list[tuple[int, int]]:
    """匀拍连续段（手级槽口径，与 `configs.uniform_runs` 同算法）。"""
    p = ctx.p
    rel_tol = p["uniform_rel_tol"] if rel_tol is None else rel_tol
    max_gap = p["uniform_max_gap_beats"] if max_gap_beats is None else max_gap_beats
    slots = ctx.slots
    runs: list[tuple[int, int]] = []
    n = len(slots)
    i = 0
    while i < n:
        if single_only and slots[i].n_hits != 1:
            i += 1
            continue
        j = i
        gap0: float | None = None
        while j + 1 < n:
            nxt = slots[j + 1]
            if single_only and nxt.n_hits != 1:
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


def _hand_runs(ctx: HandCtx, a: int, b: int) -> tuple[list[str], list[int]]:
    """段内单任务槽的 (手序列, 键序列)。"""
    hs, ks = [], []
    for s in ctx.slots[a:b + 1]:
        if s.n_hits != 1:
            continue
        hs.append(s.hit_hands[0])
        ks.append(s.hit_keys[0] if s.hit_keys[0] is not None else -1)
    return hs, ks


def _alt_rate(hs: Sequence[str]) -> float:
    if len(hs) < 2:
        return 0.0
    ch = sum(1 for i in range(1, len(hs)) if hs[i] != hs[i - 1])
    return ch / (len(hs) - 1)


def _max_same_run(hs: Sequence[str]) -> int:
    best = cur = 0
    prev = None
    for h in hs:
        cur = cur + 1 if h == prev else 1
        prev = h
        best = max(best, cur)
    return best


def _cross_count(hs: Sequence[str], ks: Sequence[int]) -> int:
    """跨半圈任务数（知识 006 分页：右手 1234 / 左手 5678）。"""
    return sum(1 for h, k in zip(hs, ks) if k > 0 and page_depth(h, k) > 0)


def _by_hand(hs: Sequence[str], ks: Sequence[int], hand: str) -> list[int]:
    return [k for h, k in zip(hs, ks) if h == hand and k > 0]


def _key_alt_rate(keys: Sequence[int]) -> float:
    """一只手自己的键序列里"换键"的比例——知识 024 的"横跳"（2 键 trill）判据。"""
    if len(keys) < 2:
        return 0.0
    return sum(1 for i in range(1, len(keys)) if keys[i] != keys[i - 1]) / (len(keys) - 1)


def _arc_span(ks) -> int:
    """键集合在 8 键环上的最小弧跨度。"""
    s = sorted(set(int(k) for k in ks))
    if len(s) <= 1:
        return 0
    best = 8
    for i in range(len(s)):
        rot = [(x - s[i]) % 8 for x in s]
        best = min(best, max(rot))
    return best


# ---------------------------------------------------------------------------
# 一、17–30 的手级检测（规格 = docs/hand-sequencing.md §5）
# ---------------------------------------------------------------------------


def detect_misalign(ctx: HandCtx) -> list[ConfigHitH]:
    """**017 错位**（手级）。

    规格（§5）：存在 slide 任务 T，其 ``wait_of`` 窗口内有任务 X，且
    ``hand(X) == hand(head(T)) == hand(T)``，且 X 的键 ≠ T 的头键。

    知识 017 原文「错位音固定由**滑动的那只手自己**拍——头 → 错位音 → 启动拍 →
    划」。键位版只判"启动窗口里有音"，因此**不区分是谁拍的**；手级版把
    "另一只手拍错位音 = 玩家打错，不是合法配置"这一句真正编成了判据。

    知识 017 的界限「双手同刻同任务 = 双押，不叫错位」由 ``hand(X)`` 唯一性天然满足。
    """
    out: list[ConfigHitH] = []
    ha = ctx.ha
    for ti, task in enumerate(ha.tasks):
        if task.kind not in ("slide", "wifi"):
            continue
        h_track = ctx.hand(ti)
        if h_track not in ("L", "R", "LR"):
            continue
        hd = task.star_idx
        h_head = ctx.hand(hd) if hd >= 0 else h_track
        if h_head != h_track and h_track != "LR":
            continue          # 换手了 → 那是 054 换手，不是 017 错位
        inner = [j for j, x in enumerate(ha.tasks)
                 if ti in x.wait_of and x.key is not None and x.key != task.key]
        own = [j for j in inner if ctx.hand(j) == h_track]
        if not own:
            continue
        keys = [ha.tasks[j].key for j in own]
        out.append(_span_tasks(
            ctx, [ti, hd] + own if hd >= 0 else [ti] + own, "错位", "017",
            f"{task.key}{task.shape}{task.end_key} 头→错位音 "
            f"{','.join(str(k) for k in keys)}→启动拍（同一只手 {h_track}）",
            f"{h_track}:头→错位音→启动拍→划轨  {_other(h_track)}:另一条星星线",
            {"head_key": task.key, "end_key": task.end_key,
             "inner_keys": keys, "n_inner": len(own),
             "n_inner_other_hand": len(inner) - len(own),
             "hand": h_track}))
    return _dedup(out)


def _other(h: str) -> str:
    return "R" if h == "L" else "L" if h == "R" else "-"


def detect_interaction(ctx: HandCtx, runs: Sequence[tuple[int, int]]
                       ) -> list[ConfigHitH]:
    """**018 普通交互**（手级）。

    规格（§5）：连续 ≥6 个单任务槽，``alt_rate == 1.0`` 且 ``max_same_run == 1``；
    跨半圈数 = 0。

    键位版只能数"匀拍单点 ≥6 连"，无法验证"每击打一次就换手"——那正是知识 018
    的**定义本身**（"每击打一次音符就进行左右手交替"）。星星链排除沿用键位版第 7 类
    假阳性的修法（星星槽占比 > 1/4 即跳过）。
    """
    out: list[ConfigHitH] = []
    min_len = ctx.p["min_len_interaction"]
    for a, b in runs:
        if b - a + 1 < min_len:
            continue
        hs, ks = _hand_runs(ctx, a, b)
        if len(hs) < min_len:
            continue
        n_slide = sum(1 for s in ctx.slots[a:b + 1] if s.has_slide)
        n_star = sum(1 for s in ctx.slots[a:b + 1] if s.has_star)
        if (n_slide + n_star) * 4 > (b - a + 1):
            continue
        if _alt_rate(hs) < 1.0 or _max_same_run(hs) > 1:
            continue
        if _cross_count(hs, ks) > 0:
            continue
        dt = ctx.slots[b].t - ctx.slots[a].t
        nps = (b - a) / dt if dt > 0 else 0.0
        out.append(_span(
            ctx, a, b, "普通交互", "018",
            f"{len(hs)} 连逐音换手（交替率 1.00、同手最长连打 1、跨半圈 0）",
            f"L:{_fmt_keys(_by_hand(hs, ks, 'L'))}  R:{_fmt_keys(_by_hand(hs, ks, 'R'))}",
            {"length": len(hs), "nps": round(nps, 2), "bpm": ctx.slots[a].bpm,
             "alt_rate": 1.0, "cross": 0}))
    return out


def _fmt_keys(ks: Sequence[int], n: int = 8) -> str:
    s = ",".join(str(k) for k in ks[:n])
    return s + ("…" if len(ks) > n else "")


def detect_vertical(ctx: HandCtx) -> list[ConfigHitH]:
    """**019 纵连**（手级）+ 030 子型 `双押纵`。

    规格（§5，**v0.2 按知识 019 的用户补充改**）：同一键连续 ≥3 个任务；
    **长度优先于速度**——长度 ≤ ``vertical_single_hand_max``（3 个 tap）时
    「稍快也可以单手处理」，不因为过了 ``t_req(0) = 83.3 ms`` 就算该拆；
    长度 >3 才按 ``vertical_split_ms`` 判"一手连打 vs 两手拆"。长纵连（>5）看
    ``max_same_run``。

    手级增量：键位版只报"同键 N 连"，手级版还报**这 N 连是被一只手连打还是两手拆的**
    —— 知识 019 的难点（个人差、内屏手）正挂在这个区分上。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    n = len(slots)
    max_gap = ctx.p["uniform_max_gap_beats"]
    split = ctx.p["vertical_split_ms"] / 1000.0
    i = 0
    while i < n:
        if slots[i].n_hits != 1 or slots[i].hit_keys[0] is None:
            i += 1
            continue
        k = slots[i].hit_keys[0]
        j = i
        while (j + 1 < n and slots[j + 1].n_hits == 1
               and slots[j + 1].hit_keys[0] == k
               and 0 < slots[j + 1].beat - slots[j].beat <= max_gap):
            j += 1
        ln = j - i + 1
        n_slide = sum(1 for t in range(i, j + 1) if slots[t].has_slide)
        if ln >= ctx.p["min_len_vertical"] and n_slide * 3 <= ln:
            hs = [slots[t].hit_hands[0] for t in range(i, j + 1)]
            gaps = [slots[t + 1].t - slots[t].t for t in range(i, j)]
            gmin = min(gaps) if gaps else 0.0
            msr = _max_same_run(hs)
            split_play = _alt_rate(hs) >= 0.5      # 两手交替 ≥ 一半 = "拆"
            long_ = ln >= ctx.p["long_vertical"]
            out.append(_span(
                ctx, i, j, "长纵连" if long_ else "纵连", "019",
                f"键 {k} 连续 {ln} 音，手序 {''.join(hs)}"
                f"（{'两手拆' if split_play else '一手连打'}，最小间隔 {gmin*1000:.0f} ms"
                f"{'，已过单手下界 83.3 ms' if gmin < split else ''}）",
                f"{'L/R 交替敲同一键（知识 019「拆」）' if split_play else hs[0] + ' 单手连打'}",
                {"key": k, "length": ln, "long": long_,
                 "hand_seq": "".join(hs), "max_same_run": msr,
                 "split": split_play, "min_gap_ms": round(gmin * 1000, 1),
                 "below_single_hand_limit": gmin < split,
                 # 知识 019 用户补充：≤3 个 tap 的纵连即使稍快也可单手
                 "single_hand_ok": ln <= ctx.p["vertical_single_hand_max"]}))
        i = j + 1
    # --- 双押纵（知识 030 的子型：两手键都常量）---
    i = 0
    while i < n:
        if not slots[i].is_pair:
            i += 1
            continue
        ks = slots[i].hit_key_set
        j = i
        while (j + 1 < n and slots[j + 1].is_pair
               and slots[j + 1].hit_key_set == ks
               and 0 < slots[j + 1].beat - slots[j].beat <= max_gap):
            j += 1
        ln = j - i + 1
        if ln >= ctx.p["min_len_vertical"]:
            kl = slots[i].hit_key("L")
            kr = slots[i].hit_key("R")
            fixed = all(slots[t].hit_key("L") == kl and slots[t].hit_key("R") == kr
                        for t in range(i, j + 1))
            out.append(_span(
                ctx, i, j, "双押纵", "030",
                f"键集 {sorted(ks)} 连续 {ln} 次"
                + ("（两手各守一键不动）" if fixed else "（段内两手换过键）"),
                f"L:{kl} R:{kr}" + ("（不动）" if fixed else "（换过）"),
                {"keys": sorted(ks), "length": ln, "hands_fixed": fixed}))
        i = j + 1
    return out


def detect_axis(ctx: HandCtx, runs: Sequence[tuple[int, int]]) -> list[ConfigHitH]:
    """**021 轴交互**（手级）。

    规格（§5）：某手的键序列常量（``len(set)==1``）而另一手在 ≥2 个键上摆动，
    交替率 = 1。

    与键位版的差别：键位版靠"同一键在奇/偶位反复出现"猜轴（隔位重复），
    手级版直接问"**哪只手一直没动**"——知识 021 的定义原文就是"一只手固定在轴键上"。
    """
    out: list[ConfigHitH] = []
    min_len = ctx.p["min_len_axis"]
    for a, b in runs:
        hs, ks = _hand_runs(ctx, a, b)
        if len(hs) < min_len:
            continue
        # 在 run 内找最长的"某手常量 + 另一手 ≥2 键 + 严格交替"子段
        i = 0
        while i < len(hs) - min_len + 1:
            best_j = -1
            best_axis: tuple[str, int] | None = None
            for j in range(i + min_len - 1, len(hs)):
                sub_h, sub_k = hs[i:j + 1], ks[i:j + 1]
                if _alt_rate(sub_h) < 1.0:
                    break
                for hand in ("L", "R"):
                    mine = _by_hand(sub_h, sub_k, hand)
                    other = _by_hand(sub_h, sub_k, _other(hand))
                    # 知识 021：BABCB = 轴音 3 次 + 邻音 2 次；另一手**不得**也落在轴键上
                    # （那是 019 纵连被两手拆开，知识 021 明说"与纵连无关"）
                    if (len(mine) >= 3 and len(set(mine)) == 1
                            and len(set(other)) >= 2 and mine[0] not in set(other)):
                        best_j, best_axis = j, (hand, mine[0])
            if best_j >= 0 and best_axis:
                hand, axis = best_axis
                sub_h, sub_k = hs[i:best_j + 1], ks[i:best_j + 1]
                others = _by_hand(sub_h, sub_k, _other(hand))
                out.append(_span(
                    ctx, a + i, a + best_j, "轴交互", "021",
                    f"{hand} 手钉在轴键 {axis}（{sub_h.count(hand)} 次），"
                    f"{_other(hand)} 手走 {_fmt_keys(sorted(set(others)))}，交替率 1.00",
                    f"{hand}:轴 {axis} 不动  {_other(hand)}:{_fmt_keys(others)}",
                    {"axis_hand": hand, "axis_key": axis,
                     "other_keys": sorted(set(others)), "length": best_j - i + 1}))
                i = best_j + 1
            else:
                i += 1
    return _dedup(out)


def detect_beatkeep(ctx: HandCtx) -> list[ConfigHitH]:
    """**027 定拍**（手级）。

    规格（§5）：某手键序列常量且**槽槽都有**，另一手同窗内有空槽；与 021 的区别 =
    定拍手**不参与交替结构**（相邻两次之间**不一定**夹着另一手的音）。

    ⚠️ **对规格的一处修正（有官谱依据）**：§5 写的"槽槽都有"按字面读会漏掉知识 027
    自己点名的谱例——`141-カゲロウデイズ Re:MASTER` 后半是 `{16}5,4,5,4`，定拍手的 5
    落在**每隔一个十六分**上（仍是稳定的八分）。知识 027 的原话是"**从不歇**"而不是
    "每个槽都有"，所以这里实现为：**定拍手在一个恒定的拍间隔上连续出现、期间不碰别的键**
    （agent 操作化）。另一只手必须在这些定拍位里**至少歇一次**，且存在"相邻两次定拍之间
    没有另一只手的音"——后者是与 021 轴交互的分界（知识 027 明说的区别点）。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    n = len(slots)
    min_slots = ctx.p["min_slots_beatkeep"]
    max_gap = ctx.p["uniform_max_gap_beats"]
    rel = ctx.p["uniform_rel_tol"]
    for hand in ("L", "R"):
        # 该手的击打序列（槽下标 + 键）
        seq = [(s.index, s.hit_key(hand)) for s in slots if s.hit_key(hand) is not None]
        i = 0
        while i < len(seq):
            k = seq[i][1]
            j, step = i, None
            while j + 1 < len(seq):
                if seq[j + 1][1] != k:
                    break                      # 定拍手碰了别的键 → 断
                g = slots[seq[j + 1][0]].beat - slots[seq[j][0]].beat
                if g <= 0 or g > max_gap:
                    break                      # 歇了 → 断（知识 027「从不歇」）
                if step is None:
                    step = g
                elif abs(g - step) > rel * step:
                    break                      # 不稳定 → 不是节拍器
                j += 1
            cnt = j - i + 1
            if cnt >= min_slots:
                s0, s1 = seq[i][0], seq[j][0]
                oth = _other(hand)
                # 另一只手"在干活"要把**划轨**也算进去（知识 027 原文：另一只手处理
                # 滑条 / 双押的另一半 / 变化音）——只数击打会漏掉 141 カゲロウデイズ
                # 那种"另一手整段在划 4>1"的写法
                t_a, t_b = slots[s0].t, slots[s1].t
                n_other = sum(1 for tk, hh in zip(ctx.ha.tasks, ctx.ha.task_hand)
                              if hh == oth and t_a - 1e-6 <= tk.t <= t_b + 1e-6)
                idle = sum(1 for x in seq[i:j + 1]
                           if slots[x[0]].hit_key(oth) is None)
                # 与 021 的分界：存在"相邻两次定拍之间没有另一只手的音"
                not_alt = False
                for t in range(i, j):
                    a, b = seq[t][0], seq[t + 1][0]
                    if not any(slots[u].hit_key(oth) is not None
                               for u in range(a + 1, b)):
                        not_alt = True
                        break
                if n_other >= 3 and idle >= 1 and not_alt:
                    okeys = [slots[t].hit_key(oth) for t in range(s0, s1 + 1)
                             if slots[t].hit_key(oth) is not None]
                    out.append(_span(
                        ctx, s0, s1, "定拍", "027",
                        f"{hand} 手在键 {k} 上稳定敲 {cnt} 次（步 {step:.3f} 拍、从不歇），"
                        f"{oth} 手 {n_other} 音、在定拍位上歇 {idle} 次",
                        f"{hand}:定拍 {k}（节拍器）  {oth}:变化 {_fmt_keys(sorted(set(okeys)))}",
                        {"hand": hand, "key": k, "count": cnt,
                         "step_beats": round(step or 0, 4),
                         "other_notes": n_other, "other_idle": idle}))
                    i = j + 1
                    continue
            i = max(j, i) + 1
    return _dedup(out)


def detect_triangle(ctx: HandCtx, runs: Sequence[tuple[int, int]]
                    ) -> list[ConfigHitH]:
    """**020 三角交互**（手级）。

    规格（§5）：按 3 音切组；组内手序为 ``XYX``（占 2 键）或 ``XYX``（占 3 键）；
    正/反三角的手级差别**未找到**（`docs/hand-sequencing.md` §7-7）。

    几何条件（键集 2–3 键、弧跨 ≤3、组内转向、相邻组不相交、只取最长偏移、
    带星星的段跳过）与键位版一致——那是第一轮 7 类假阳性里第 1、2 条的修法。
    手级增量是"**组内手序必须是 XYX**"（一只手打两次、另一只手夹在中间）。
    """
    out: list[ConfigHitH] = []
    min_len = ctx.p["min_len_triangle"]
    for a, b in runs:
        if any(ctx.slots[t].has_slide or ctx.slots[t].has_star
               for t in range(a, b + 1)):
            continue
        hs, ks = _hand_runs(ctx, a, b)
        if len(ks) < min_len:
            continue
        best: tuple[int, list] | None = None
        for off in (0, 1, 2):
            chains = _tri_chains(hs, ks, off)
            total = sum(len(c) for c in chains)
            if best is None or total > best[0]:
                best = (total, chains)
        if not best or not best[1]:
            continue
        for chain in best[1]:
            s0, s1 = a + chain[0][0], a + chain[-1][0] + 2
            if s1 - s0 + 1 < min_len:
                continue
            groups = [",".join(str(x) for x in ks[p:p + 3]) for p, _ in chain]
            hpat = ["".join(hs[p:p + 3]) for p, _ in chain]
            out.append(_span(
                ctx, s0, s1, "三角交互", "020",
                f"{len(chain)} 组：{' | '.join(groups[:4])}"
                + ("…" if len(groups) > 4 else "")
                + f"；组内手序 {'/'.join(sorted(set(hpat)))}",
                f"每组 XYX：一手打两次、另一手夹中间（占 "
                f"{'/'.join(str(len(kk)) for _, kk in chain[:4])} 键）",
                {"n_groups": len(chain), "hand_patterns": hpat[:8],
                 "key_sets": [sorted(kk) for _, kk in chain][:8],
                 "occupancy": [len(kk) for _, kk in chain][:8]}))
    return _dedup(out)


def _tri_valid(hs: Sequence[str], ks: Sequence[int]) -> bool:
    if len(ks) < 3:
        return False
    kk = frozenset(ks)
    if not (2 <= len(kk) <= 3) or _arc_span(kk) > 3:
        return False
    s1, s2 = cstep(ks[0], ks[1]), cstep(ks[1], ks[2])
    if s1 == 0 or s2 == 0 or (s1 > 0) == (s2 > 0):
        return False
    return hs[0] == hs[2] and hs[1] != hs[0]      # 手级：XYX


def _tri_chains(hs: Sequence[str], ks: Sequence[int], off: int):
    chains: list[list[tuple[int, frozenset]]] = []
    cur: list[tuple[int, frozenset]] = []
    t = off
    while t + 2 < len(ks) + 0:
        sub_h, sub_k = hs[t:t + 3], ks[t:t + 3]
        if len(sub_k) < 3:
            break
        kk = frozenset(sub_k)
        valid = _tri_valid(sub_h, sub_k)
        ok = valid and (not cur or not (kk & cur[-1][1]))
        if ok:
            cur.append((t, kk))
        else:
            if len(cur) >= 2:
                chains.append(cur)
            cur = [(t, kk)] if valid else []
        t += 3
    if len(cur) >= 2:
        chains.append(cur)
    return chains


def detect_stair(ctx: HandCtx, runs: Sequence[tuple[int, int]]) -> list[ConfigHitH]:
    """**022 楼梯交互 / 023 逆楼梯（方向盘）**（手级）。

    规格（§5）：
    - 022：相邻**同手**步进 ``cstep`` 恒为 +1 或 −1，且两手符号**相反**；
    - 023：每个同刻/相邻对满足 ``cdist(L,R) == 4``，且两手 ``cstep`` 符号**相同**，必有折返。

    键位版把"成对"定义成"相邻两个槽"（隐含严格交替手序）；手级版直接按**真实的左右手
    序列**算各自的步进，"哪两个音属于同一对"不再是假设。知识 022 的"每对重复两次再走"
    写法由"同手原地重复不算推进"吸收；**走向**（monotone run）是 022 的原文分组单位，
    所以本实现先把每只手的推进切成走向，再在走向内比对两手的方向关系——
    这同时挡掉了"大宇宙式横跳"（步进 +1/−1 反复交替）被误判成楼梯（agent 操作化）。
    """
    out: list[ConfigHitH] = []
    min_pairs = ctx.p["min_pairs_stair"]
    for a, b in runs:
        hs, ks = _hand_runs(ctx, a, b)
        if len(hs) < min_pairs * 2:
            continue
        seq: dict[str, list[tuple[int, int]]] = {"L": [], "R": []}
        for idx, (h, k) in enumerate(zip(hs, ks)):
            if h not in ("L", "R") or k <= 0:
                continue
            if seq[h] and seq[h][-1][1] == k:
                continue          # 原地重复（知识 022"每对重复两次再走"）不算推进
            seq[h].append((idx, k))
        if len(seq["L"]) < min_pairs or len(seq["R"]) < min_pairs:
            continue
        runsL = _monotone_runs([k for _, k in seq["L"]])
        runsR = _monotone_runs([k for _, k in seq["R"]])
        for (i0, i1, dL) in runsL:
            for (j0, j1, dR) in runsR:
                # 两只手的走向必须在时间上重叠
                lo_i, hi_i = seq["L"][i0][0], seq["L"][i1][0]
                lo_j, hi_j = seq["R"][j0][0], seq["R"][j1][0]
                lo, hi = max(lo_i, lo_j), min(hi_i, hi_j)
                if hi - lo + 1 < min_pairs * 2:
                    continue
                lk = [k for idx, k in seq["L"][i0:i1 + 1] if lo <= idx <= hi]
                rk = [k for idx, k in seq["R"][j0:j1 + 1] if lo <= idx <= hi]
                n_pairs = min(len(lk), len(rk))
                if n_pairs < min_pairs:
                    continue
                # 对位关系（知识 023）：**成对**地看（两手各一个音为一对），
                # 两种切分偏移试一遍，任一偏移下每对都 180° 即判对位
                opposite = _paired_opposition(hs, ks, lo, hi)
                s0, s1 = a + lo, a + hi
                if dL == dR and opposite:
                    rev = _has_reversal_seq([k for _, k in seq["L"]])
                    out.append(_span(
                        ctx, s0, s1, "逆楼梯/方向盘", "023",
                        f"两手始终 180° 对位、同向转动（L {dL:+d} / R {dR:+d}），"
                        f"{n_pairs} 对，{'有折返' if rev else '无折返'}",
                        f"L:{_fmt_keys(lk)}  R:{_fmt_keys(rk)}",
                        {"n_pairs": n_pairs, "opposite": True, "reversal": rev,
                         "dir_l": dL, "dir_r": dR}))
                elif dL == -dR and not opposite:
                    out.append(_span(
                        ctx, s0, s1, "楼梯交互", "022",
                        f"两手反向成对推进（L {dL:+d} / R {dR:+d}，每步只走相邻键），"
                        f"{n_pairs} 对",
                        f"L:{_fmt_keys(lk)}  R:{_fmt_keys(rk)}",
                        {"n_pairs": n_pairs, "opposite": False, "dir_l": dL,
                         "dir_r": dR}))
    return _dedup(out)


def _paired_opposition(hs: Sequence[str], ks: Sequence[int],
                       lo: int, hi: int) -> bool:
    """段 ``[lo, hi]`` 内"两手各一个音为一对"时，每一对是否恒 180° 对位（知识 023）。

    两种切分偏移各试一遍（成对的起点可能在偶数位也可能在奇数位）。
    """
    for off in (0, 1):
        pairs = []
        t = lo + off
        while t + 1 <= hi and t + 1 < len(ks):
            if hs[t] != hs[t + 1] and ks[t] > 0 and ks[t + 1] > 0:
                pairs.append(cdist(ks[t], ks[t + 1]) == 4)
            else:
                pairs.append(False)
            t += 2
        if len(pairs) >= 2 and all(pairs):
            return True
    return False


def _monotone_runs(keys: Sequence[int]) -> list[tuple[int, int, int]]:
    """把一只手的键序列切成**走向**（知识 022）：每步 ``cstep`` 恒为 +1 或 −1。

    返回 ``[(起点下标, 终点下标, 方向), ...]``，只保留长度 ≥3 个键（=2 步）的走向。
    """
    out: list[tuple[int, int, int]] = []
    i = 0
    n = len(keys)
    while i < n - 1:
        d = cstep(keys[i], keys[i + 1])
        if abs(d) != 1:
            i += 1
            continue
        j = i + 1
        while j + 1 < n and cstep(keys[j], keys[j + 1]) == d:
            j += 1
        if j - i >= 2:
            out.append((i, j, d))
        i = j
    return out


def _has_reversal_seq(keys: Sequence[int]) -> bool:
    """是否出现折返（知识 023 的固有特征）。"""
    d = [cstep(keys[i], keys[i + 1]) for i in range(len(keys) - 1)]
    return any(d[i] * d[i + 1] < 0 for i in range(len(d) - 1))


def detect_macrocosm(ctx: HandCtx, runs: Sequence[tuple[int, int]]
                     ) -> list[ConfigHitH]:
    """**024 大宇宙**（手级）。

    规格（§5）：各手键集合大小 = 2 且组内 ``cdist == 1``；两手键集合不交；
    两集合并起来恰 4 键。

    与键位版的关键差别：键位版按"奇数位 / 偶数位"分两组——这是**严格交替手序的假设**。
    `docs/research/config-usage-survey.md` 已指出该口径把 159 张谱判成大宇宙（膨胀）。
    手级版按**真实分手**取两手的键集合，"每只手在自己那对相邻键上横跳"是知识 024
    的原文硬约束，因此本判据是 024 的直译。
    """
    out: list[ConfigHitH] = []
    min_len = ctx.p["min_len_macrocosm"]
    for a, b in runs:
        hs, ks = _hand_runs(ctx, a, b)
        n = len(hs)
        if n < min_len:
            continue
        i = 0
        while i <= n - min_len:
            j = i
            while j + 1 < n:
                sl = set(_by_hand(hs[i:j + 2], ks[i:j + 2], "L"))
                sr = set(_by_hand(hs[i:j + 2], ks[i:j + 2], "R"))
                if (len(sl) <= 2 and len(sr) <= 2 and not (sl & sr)
                        and all(_arc_span(s) <= 1 for s in (sl, sr) if len(s) == 2)):
                    j += 1
                else:
                    break
            sl = set(_by_hand(hs[i:j + 1], ks[i:j + 1], "L"))
            sr = set(_by_hand(hs[i:j + 1], ks[i:j + 1], "R"))
            ln = j - i + 1
            altl = _key_alt_rate(_by_hand(hs[i:j + 1], ks[i:j + 1], "L"))
            altr = _key_alt_rate(_by_hand(hs[i:j + 1], ks[i:j + 1], "R"))
            min_alt = ctx.p["macrocosm_min_key_alt"]
            if (ln >= min_len and len(sl) == 2 and len(sr) == 2
                    and _arc_span(sl) == 1 and _arc_span(sr) == 1 and not (sl & sr)
                    and len(sl | sr) == 4
                    and altl >= min_alt and altr >= min_alt):
                out.append(_span(
                    ctx, a + i, a + j, "大宇宙", "024",
                    f"L 手在 {'↔'.join(str(x) for x in sorted(sl))} 横跳、"
                    f"R 手在 {'↔'.join(str(x) for x in sorted(sr))} 横跳"
                    f"（两对相邻键不交、并集 4 键，{ln} 音）",
                    f"L:{sorted(sl)} 横跳  R:{sorted(sr)} 横跳",
                    {"pair_l": sorted(sl), "pair_r": sorted(sr), "length": ln,
                     "key_alt_l": round(altl, 3), "key_alt_r": round(altr, 3)}))
                i = j + 1
            else:
                i += 1
    return _dedup(out)


def detect_scatter(ctx: HandCtx, runs: Sequence[tuple[int, int]],
                   others: Sequence[ConfigHitH]) -> list[ConfigHitH]:
    """**025 散点**（手级）。

    规格（§5）：交替率高但**每手键集合 ≥4**、**同手步进种类 ≥3**，且不满足
    022/023/024 的手级条件。

    知识 025 原文「不收敛于固定的小键位集合」「无统一方向、无固定对位关系」——
    手级版把"收敛"落在**每只手自己**的键集上（键位版是整段的键集，会把
    "两手各守一小块"的规整配置也算成分散）。星星/长条段排除沿用键位版第 4 类假阳性修法。
    """
    geo = {"纵连", "长纵连", "三角交互", "轴交互", "楼梯交互",
           "逆楼梯/方向盘", "大宇宙", "定拍"}
    taken = [(h.slot_start, h.slot_end) for h in others if h.config in geo]
    out: list[ConfigHitH] = []
    for a, b in runs:
        if b - a + 1 < ctx.p["min_len_scatter"]:
            continue
        if any(not (b < s0 or a > s1) for s0, s1 in taken):
            continue
        if any(s.has_slide or s.has_star or "hold" in s.kinds
               for s in ctx.slots[a:b + 1]):
            continue
        hs, ks = _hand_runs(ctx, a, b)
        kl, kr = _by_hand(hs, ks, "L"), _by_hand(hs, ks, "R")
        if len(set(kl)) < ctx.p["scatter_min_keys"] or \
           len(set(kr)) < ctx.p["scatter_min_keys"]:
            continue
        steps = set()
        for arr in (kl, kr):
            for i in range(len(arr) - 1):
                s = cstep(arr[i], arr[i + 1])
                if s:
                    steps.add(abs(s))
        if len(steps) < ctx.p["scatter_min_step_kinds"]:
            continue
        out.append(_span(
            ctx, a, b, "散点", "025",
            f"L 手用 {len(set(kl))} 键、R 手用 {len(set(kr))} 键，"
            f"同手步进 {sorted(steps)}（两手都不收敛）",
            f"L:{_fmt_keys(kl)}  R:{_fmt_keys(kr)}",
            {"length": b - a + 1, "n_keys_l": len(set(kl)),
             "n_keys_r": len(set(kr)), "steps": sorted(steps)}))
    return out


def detect_double_run(ctx: HandCtx) -> list[ConfigHitH]:
    """**030 连续双押**（手级）+ `侧边双押` 复核清单。

    规格（§5）：``pairs`` 连续 ≥3；**双押纵** = 两手键都常量；**绕圈** = 两手同向逐格推进；
    **侧边双押** = ``{L,R} ∈ {{2,3},{6,7}}``，标**是哪一种引导**（判据来自
    `hands.side_double_events`，**八型形状判定 + 两级阅读窗口**，
    **"突然"不用数字定义**）。

    ⚠️ 八型（A/B·C·D 来自用户三个例子，G·H·F1·F2·E 来自 388 官谱普查）仍是 agent 的
    归纳，**覆盖率高不等于归纳正确**。所以无引导的那些是**供人复核的清单**，
    不记无理、不下判决。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    n = len(slots)
    max_gap = ctx.p["uniform_max_gap_beats"]
    i = 0
    while i < n:
        if not slots[i].is_pair:
            i += 1
            continue
        j = i
        while (j + 1 < n and slots[j + 1].is_pair
               and 0 < slots[j + 1].beat - slots[j].beat <= max_gap):
            j += 1
        ln = j - i + 1
        if ln >= ctx.p["min_len_double_run"]:
            lk = [slots[t].hit_key("L") for t in range(i, j + 1)]
            rk = [slots[t].hit_key("R") for t in range(i, j + 1)]
            sets = [tuple(sorted(slots[t].hit_key_set)) for t in range(i, j + 1)]
            const = len(set(sets)) == 1
            n_one_hand = sum(1 for t in range(i, j + 1) if not slots[t].two_hands)
            dl = [cstep(lk[t], lk[t + 1]) for t in range(ln - 1) if lk[t] and lk[t + 1]]
            dr = [cstep(rk[t], rk[t + 1]) for t in range(ln - 1) if rk[t] and rk[t + 1]]
            same_dir = (bool(dl) and bool(dr)
                        and all(x == dl[0] and x != 0 for x in dl)
                        and all(x == dr[0] and x != 0 for x in dr)
                        and (dl[0] > 0) == (dr[0] > 0))
            shared = sum(1 for t in range(i, j)
                         if slots[t].hit_key_set & slots[t + 1].hit_key_set)
            varied = len({tuple(sorted(slots[t].hit_key_set))
                          for t in range(i, j + 1)}) > 1
            # 知识 030「DRAGOON 绕圈串」：共享键、逐步绕环行走 → 有引导
            sub = ("双押纵" if const
                   else "绕圈" if (same_dir or (shared >= ln - 1 and varied))
                   else "一般")
            out.append(_span(
                ctx, i, j, "连续双押", "030",
                f"连续 {ln} 个双押（{sub}）：L {_fmt_keys([k for k in lk if k])} / "
                f"R {_fmt_keys([k for k in rk if k])}"
                + (f"；其中 {n_one_hand} 个由一只手扫（知识 036）" if n_one_hand else ""),
                f"L:{_fmt_keys([k for k in lk if k])}  R:{_fmt_keys([k for k in rk if k])}",
                {"length": ln, "subtype": sub, "shared_steps": shared,
                 "n_one_hand_sweep": n_one_hand,
                 "keys_l": [k for k in lk if k][:16],
                 "keys_r": [k for k in rk if k][:16]}))
        i = j + 1
    # --- 侧边双押（知识 030 铁律）---
    # 判据整体来自 `hands.side_double_events()`：铁律读作「**禁止突然（＝没有引导）的**
    # 侧边双押」，标**是哪一种引导**（八型形状判定 + 拍级→乐句级两级阅读窗口），
    # **不用数字定义"突然"**（用户 2026-09-20）。
    # 无引导（＝**乐句级窗口仍判不出**）的那些是**复核清单**，不是判决、不记无理。
    for ev in ctx.ha.side_doubles:
        idx = _slot_at(slots, ev["time"])
        if idx is None:
            continue
        types = ev["guide_types"] or ev["guide_types_phrase"]
        tag = (f"有引导（{'+'.join(types)}；{ev['guide_level']}级窗口）" if ev["guided"]
               else "无引导 —— 请人复核（知识 030 铁律）")
        out.append(_span(
            ctx, idx, idx, "侧边双押", "030",
            f"{ev['keys'][0]}/{ev['keys'][1]}（{tag}）",
            f"L:{ev['L']}  R:{ev['R']}（同半圈相邻，必有一手出张）",
            {"keys": ev["keys"], "guided": ev["guided"],
             "guide_type": ev["guide_type"], "guide_types": list(types),
             "guide_level": ev["guide_level"],
             # ↓ 纯诊断字段（定位用），不参与判定
             "near_dist": ev["near_dist"],
             "gap_ms": (None if ev["gap"] is None
                        else round(ev["gap"] * 1000, 1)),
             "run_len": ev["run_len"]}))
    return out


def _slot_at(slots, t: float) -> int | None:
    """把时间对回槽下标（侧边双押事件来自 `hands`，需要回挂到槽上）。"""
    best, bd = None, 1e9
    for i, s in enumerate(slots):
        d = abs(s.t - t)
        if d < bd:
            best, bd = i, d
    return best if bd <= 0.01 else None


def s_keys(s: HSlot) -> set[int]:
    return {k for k in s.keys if k is not None}


def detect_single_double(ctx: HandCtx) -> list[ConfigHitH]:
    """**029 单双 / 双单 / 夹心**（手级）。

    规格（§5）：``pairs`` 与单任务槽交替出现；"骨架" = 在双押与单点中**键位不变**的那只手。

    键位版只能报"SD 串的周期"；手级版能直接回答知识 029 的第一个变化维度
    「**谁是骨架**」——在整段里键位不变的那只手。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    n = len(slots)
    max_gap = ctx.p["uniform_max_gap_beats"]
    min_len = ctx.p["min_len_sd"]
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while (j + 1 < n and 0 < slots[j + 1].beat - slots[j].beat <= max_gap
               and slots[j + 1].n_hits >= 1):
            j += 1
        if j > i:
            blocks.append((i, j))
        i = j + 1
    for a, b in blocks:
        sd = "".join("D" if slots[t].is_pair else "S" for t in range(a, b + 1))
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
                    s0, s1 = a + t, a + t + ln - 1
                    skel = _skeleton_hand(ctx, s0, s1)
                    out.append(_span(
                        ctx, s0, s1, "单双/双单", "029",
                        f"{unit} × {reps}；骨架 = "
                        + (f"{skel[0]}（{skel[1]} 不变）" if skel else "无（都在变）"),
                        (f"{skel[0]}:骨架 {skel[1]}"
                         + (f"  {_other(skel[0])}:变化" if skel[0] in ("L", "R") else "")
                         if skel else "两手都在变"),
                        {"unit": unit, "period": period, "cycles": reps,
                         "ratio": f"{unit.count('D')}:{unit.count('S')}",
                         "skeleton_hand": skel[0] if skel else "",
                         "skeleton_key": skel[1] if skel else None}))
                    t += ln
                else:
                    t += 1
    return _dedup(out)


def _skeleton_hand(ctx: HandCtx, a: int, b: int) -> tuple[str, object] | None:
    """段内的"骨架"（知识 029 的第一个变化维度「谁是骨架」）。

    三种形态（知识 029 的实例表里都出现过）：

    - ``("L"/"R", 键)``：某只手在整段里**键位不变**（Bloody Trail 的单点固定 3）；
    - ``("双押", [键,键])``：双押的键集恒定（キミノヨゾラ 的 4/5）；
    - ``("单点", 键)``：全部单点槽落在同一个键上。
    """
    for hand in ("L", "R"):
        ks = [ctx.slots[t].hit_key(hand) for t in range(a, b + 1)]
        ks = [k for k in ks if k is not None]
        if len(ks) >= 3 and len(set(ks)) == 1:
            return hand, ks[0]
    pairs = [tuple(sorted(ctx.slots[t].hit_key_set))
             for t in range(a, b + 1) if ctx.slots[t].is_pair]
    if len(pairs) >= 2 and len(set(pairs)) == 1:
        return "双押", list(pairs[0])
    singles = [ctx.slots[t].hit_keys[0] for t in range(a, b + 1)
               if ctx.slots[t].n_hits == 1]
    if len(singles) >= 2 and len(set(singles)) == 1:
        return "单点", singles[0]
    return None


def detect_bullet(ctx: HandCtx) -> list[ConfigHitH]:
    """**028 子弹**（手级）。

    规格（§5）：同键恰 2 次且间隔 ≤ ``bullet_gap``；**记录是否同手**
    （同手 = 真连打，异手 = 分摊）。

    键位版无法区分这两种，而知识 028 的手感差异（"塞进错位间隙手感差"）正来自
    "同一只手在极短时间里回到同一个键"。星星载体排除沿用键位版第 5 类假阳性修法。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    n = len(slots)
    max_gap = ctx.p["bullet_max_gap_beats"]
    for i in range(n - 1):
        if slots[i].n_hits != 1 or slots[i + 1].n_hits != 1:
            continue
        if slots[i].has_slide or slots[i + 1].has_slide or \
           slots[i].has_star or slots[i + 1].has_star:
            continue
        k = slots[i].hit_keys[0]
        if k is None or slots[i + 1].hit_keys[0] != k:
            continue
        if not (0 < slots[i + 1].beat - slots[i].beat <= max_gap):
            continue
        prev_same = i > 0 and slots[i - 1].n_hits == 1 and slots[i - 1].hit_keys[0] == k
        next_same = (i + 2 < n and slots[i + 2].n_hits == 1
                     and slots[i + 2].hit_keys[0] == k)
        if prev_same or next_same:
            continue
        same_hand = slots[i].hit_hands[0] == slots[i + 1].hit_hands[0]
        out.append(_span(
            ctx, i, i + 1, "子弹", "028",
            f"{k},{k}（{'同手连打' if same_hand else '两手分摊'}："
            f"{slots[i].hit_hands[0]}→{slots[i+1].hit_hands[0]}）",
            f"{slots[i].hit_hands[0]}→{slots[i+1].hit_hands[0]} 同键两次",
            {"carrier": "同键相邻两音", "key": k, "same_hand": same_hand,
             "hands": slots[i].hit_hands[0] + slots[i + 1].hit_hands[0]}))
    for i in range(n - 1):
        if not (slots[i].is_pair and slots[i + 1].is_pair):
            continue
        if slots[i].hit_key_set != slots[i + 1].hit_key_set:
            continue
        if not (0 < slots[i + 1].beat - slots[i].beat <= max_gap):
            continue
        prev_same = i > 0 and slots[i - 1].hit_key_set == slots[i].hit_key_set
        next_same = i + 2 < n and slots[i + 2].hit_key_set == slots[i].hit_key_set
        if prev_same or next_same:
            continue
        ks = "".join(str(x) for x in sorted(slots[i].hit_key_set))
        same_hand = (slots[i].hit_key("L") == slots[i + 1].hit_key("L")
                     and slots[i].hit_key("L") is not None)
        out.append(_span(
            ctx, i, i + 1, "子弹", "028",
            f"{ks},{ks}（短双押串，{'两手各守一键' if same_hand else '两手换键'}）",
            f"L:{slots[i].hit_key('L')}→{slots[i+1].hit_key('L')}  "
            f"R:{slots[i].hit_key('R')}→{slots[i+1].hit_key('R')}",
            {"carrier": "短双押串", "keys": sorted(slots[i].hit_key_set),
             "same_hand": same_hand}))
    return out


def detect_skipbeat(ctx: HandCtx) -> list[ConfigHitH]:
    """**026 跳拍**（**时间层**，非手级；规格 §5 明确"跳拍不是手级配置"）。

    ``hands_pattern`` 恒为 ``—``：分配器给出的两手序列在跳拍与匀拍之间看不出区别，
    所以本检测与键位版**完全同算法**（逐小节间隔倍数的二值化交替段数 ≥4）。
    保留它只为让三张表的配置维度与键位版对齐。
    """
    out: list[ConfigHitH] = []
    by_bar: dict[int, list[HSlot]] = {}
    for s in ctx.slots:
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
            continue
        out.append(_span(ctx, ss[0].index, ss[-1].index, "跳拍", "026",
                         f"小节 {bar} 间隔倍数 {mult}（时间层判据）", "—",
                         {"multiples": mult, "unit_beats": round(unit, 4),
                          "alt_runs": runs, "layer": "time"}))
    return out


def _dedup(hits: Sequence[ConfigHitH]) -> list[ConfigHitH]:
    """去掉同名同范围 / 被同名更长片段完全包含的重复命中。"""
    out: list[ConfigHitH] = []
    for h in sorted(hits, key=lambda x: (x.config, x.slot_start,
                                         -(x.slot_end - x.slot_start))):
        if any(k.config == h.config and k.slot_start <= h.slot_start
               and k.slot_end >= h.slot_end for k in out):
            continue
        out.append(h)
    return sorted(out, key=lambda x: (x.slot_start, x.config))


# ---------------------------------------------------------------------------
# 二、星星族（047–063）与第二期（033–045）的手级检测
# ---------------------------------------------------------------------------


@dataclass
class SlideUnit:
    """一根星星的"手级单元"：头 + 轨道（知识 054：二者可以不同手）。"""

    ti: int                 # 轨道任务下标
    head_ti: int            # 星星头任务下标（被拍划吞掉时 = 吞掉它的那个任务）
    key: int                # 头键
    end: int | None         # 终点键
    shape: str
    t_head: float           # 星星头时刻
    t_start: float          # 启动拍（知识 014：头 + 60/BPM）
    t_end: float            # 名义到达终点时刻
    beat_head: float
    measure: int
    bpm: float
    hand: str               # 划轨手
    head_hand: str          # 拍头手

    @property
    def switched(self) -> bool:
        """知识 054 换手：拍头手 ≠ 划轨手。"""
        return self.head_hand in ("L", "R") and self.hand in ("L", "R") \
            and self.head_hand != self.hand

    @property
    def chuzhang(self) -> bool:
        """**出张（知识 064 用户口径）**：划轨手落进出张区（右手 6/7、左手 2/3）。

        头键或尾键任一落进去就算——这只手整条轨都要走一遍。
        v0.1 的旧口径（"划轨手不在 slide 末尾所在半边"，知识 007）已被 064/065
        作废，保留在 :attr:`chuzhang_end_side` 里供对照。
        """
        if self.hand not in ("L", "R"):
            return False
        return is_chuzhang(self.hand, self.key) or is_chuzhang(self.hand, self.end)

    @property
    def chuzhang_end_side(self) -> bool:
        """**v0.1 旧口径**：划轨手不在 slide 末尾所在半边（知识 007，已作废）。"""
        if self.end is None or self.hand not in ("L", "R"):
            return False
        return home_side(self.end) != self.hand

    @property
    def is_straight(self) -> bool:
        return self.shape in ("-", "")

    @property
    def is_arc(self) -> bool:
        return self.shape.startswith("^") or self.shape.startswith(">") \
            or self.shape.startswith("<")


def slide_units(ctx: HandCtx) -> list[SlideUnit]:
    """把全谱的 slide 轨道任务整理成 :class:`SlideUnit` 列表（按头时刻排序）。"""
    out: list[SlideUnit] = []
    ha = ctx.ha
    for ti, t in enumerate(ha.tasks):
        if t.kind not in ("slide", "wifi") or t.key is None:
            continue
        hd = t.star_idx
        if hd >= 0:
            t_head = ha.tasks[hd].t
            head_hand = ctx.hand(hd)
        else:
            spb = 60.0 / t.bpm if t.bpm else 0.0
            t_head = t.t - spb
            head_hand = ctx.hand(ti)
        out.append(SlideUnit(
            ti=ti, head_ti=hd, key=int(t.key), end=t.end_key, shape=t.shape,
            t_head=t_head, t_start=t.t, t_end=t.t_end_nominal or t.t_end,
            beat_head=t.beat, measure=t.measure, bpm=t.bpm,
            hand=ctx.hand(ti), head_hand=head_hand))
    out.sort(key=lambda u: (u.t_head, u.key))
    return out


def _u_desc(u: SlideUnit) -> str:
    return f"{u.key}{u.shape}{u.end}"


# ---------- 057 一笔画 ----------


def detect_onestroke(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**057 一笔画**（手级；知识 057 + 007）。

    规格（§5.1）：``end(track_i) == head(track_{i+1})`` 且时间首尾相接 → ``chain``，
    **同一只手**；另一只手专拍头。

    知识 057 的打法原文：「一手专拍头、一手一路不抬划完」。形状判据（头尾相连）
    是 A 档；**"同一只手划完整条链"是手级增量**——键位版根本看不到这一层。
    ``closed`` = 链尾回到首个头键（BREAK YOU!! m066 那种闭环）。
    """
    out: list[ConfigHitH] = []
    n = len(units)
    used = set()
    for i in range(n):
        if i in used:
            continue
        chain = [units[i]]
        j = i
        while True:
            cur = chain[-1]
            nxt = None
            for k in range(j + 1, n):
                u = units[k]
                if u.t_head < cur.t_head - 1e-6:
                    continue
                if u.key != cur.end:
                    continue
                spb = 60.0 / u.bpm if u.bpm else 0.5
                if u.t_start - cur.t_end > ctx.p["onestroke_gap_beats"] * spb:
                    continue
                nxt, j = u, k
                break
            if nxt is None:
                break
            chain.append(nxt)
        if len(chain) < ctx.p["onestroke_min_len"]:
            continue
        for u in chain:
            used.add(units.index(u))
        hands = [u.hand for u in chain]
        heads = [u.head_hand for u in chain]
        one_hand = len(set(hands)) == 1
        closed = chain[-1].end == chain[0].key
        out.append(_span_tasks(
            ctx, [u.ti for u in chain] + [u.head_ti for u in chain if u.head_ti >= 0],
            "一笔画", "057",
            f"{len(chain)} 根首尾相连 "
            + " → ".join(_u_desc(u) for u in chain[:5])
            + ("…" if len(chain) > 5 else "")
            + f"；划轨手 {''.join(hands)}"
            + ("（一只手一路划完）" if one_hand else "（中途换手）")
            + ("，闭环" if closed else ""),
            (f"{hands[0]}:一路不抬划完 {len(chain)} 根  "
             f"{_other(hands[0])}:专拍头（领先一拍）" if one_hand
             else f"划轨手 {''.join(hands)} / 拍头手 {''.join(heads)}（链内换手）"),
            {"length": len(chain), "closed": closed, "one_hand": one_hand,
             "track_hands": "".join(hands), "head_hands": "".join(heads),
             "keys": [u.key for u in chain] + [chain[-1].end]}))
    return out


# ---------- 049 / 053 / 062 拍滑家族 ----------


def _tapslide_runs(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[list[SlideUnit]]:
    """拍滑骨架的连续段：等间隔 + 两手交替 + **拍头手 == 划轨手**（知识 049）。

    ``lag = (60/BPM) / 头间距`` —— 1 = 四分音拍滑，2 = 八分音拍滑（知识 049b）。
    """
    rel = ctx.p["tapslide_rel_tol"]
    singles = [u for u in units if u.hand in ("L", "R")]
    # 同刻只有一个头的单元才进异相位骨架（同刻两个头 → 050/051，见下）
    by_head: dict[int, list[SlideUnit]] = {}
    for u in singles:
        by_head.setdefault(int(round(u.t_head * 1000)), []).append(u)
    seq = [v[0] for k, v in sorted(by_head.items()) if len(v) == 1]
    runs: list[list[SlideUnit]] = []
    i = 0
    while i < len(seq):
        j = i
        dt0: float | None = None
        while j + 1 < len(seq):
            a, b = seq[j], seq[j + 1]
            dt = b.t_head - a.t_head
            if dt <= 1e-6:
                break
            if a.hand == b.hand:
                break                      # 拍滑家族 = 两手交替（知识 049）
            # 知识 049 的窍门是"打○那只手去滑"，但知识 054 明说换手是**自由度**——
            # 所以这里不一刀切，段末统一用 ``tapslide_switch_tol`` 兜底
            pass
            if dt0 is None:
                spb = 60.0 / a.bpm if a.bpm else 0.5
                lag = spb / dt
                if abs(lag - round(lag)) > 0.2 or not (1 <= round(lag) <= 2):
                    break
                dt0 = dt
            elif abs(dt - dt0) > rel * dt0:
                break
            j += 1
        g = list(seq[i:j + 1])
        if len(g) >= ctx.p["tapslide_min_units"]:
            n_sw = sum(1 for u in g if u.switched)
            if n_sw <= ctx.p["tapslide_switch_tol"] * len(g):
                runs.append(g)
        i = max(j, i) + 1
    return runs


def detect_tapslide(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**049 连续拍滑**（+ 049b 八分音拍滑）、**053 夹键拍滑**、**062 鼓动段**。

    规格（§5.1）：049 = ``hand(star_n) == hand(track_n)``、两手相位差一拍；
    053 = 049 且同一键连续 3 个八分（首=头、尾=启动拍）；
    062 = 拍滑单元等间隔左右交替 ≥6 次 + 统计 ``scrape_pressure``。

    **手级增量**：教程明说的识别窍门「都是打 ○ 那个手去滑星星」在键位版里根本
    表达不了（键位版只能看到"有 tap 落在上一根头键上"）。
    """
    out: list[ConfigHitH] = []
    for run in _tapslide_runs(ctx, units):
        dt = run[1].t_head - run[0].t_head
        spb = 60.0 / run[0].bpm if run[0].bpm else 0.5
        lag = int(round(spb / dt)) if dt > 0 else 1
        hands = "".join(u.hand for u in run)
        sub = "八分音拍滑" if lag >= 2 else "四分音拍滑"
        n_clamp = sum(1 for u in run if _clamp_keys(ctx, u) == 1)
        out.append(_span_tasks(
            ctx, [u.ti for u in run] + [u.head_ti for u in run if u.head_ti >= 0],
            "连续拍滑", "049",
            f"{len(run)} 根等间隔（{sub}，lag={lag}），拍头手==划轨手、两手交替 {hands}；"
            f"头键 {','.join(str(u.key) for u in run[:6])}"
            + ("…" if len(run) > 6 else ""),
            f"每根：本格打★的手一拍后打○并划出；两手相位差一拍（{hands}）",
            {"n_units": len(run), "subtype": sub, "lag": lag,
             "hands": hands, "keys": [u.key for u in run][:16],
             "n_clamp_units": n_clamp}))
        # 053 夹键拍滑：每根的"头→启动拍"之间，同一个键上恰好再多一下
        if n_clamp >= max(2, int(0.6 * len(run))):
            out.append(_span_tasks(
                ctx, [u.ti for u in run] + [u.head_ti for u in run if u.head_ti >= 0],
                "夹键拍滑", "053",
                f"{n_clamp}/{len(run)} 根在头与启动拍之间于**同一键**上多打一下"
                f"（同键连吃三个八分：头→夹键→启动拍）",
                f"每只手在自己的键上连打三个八分、第三下划出；每四分音换一次手（{hands}）",
                {"n_units": len(run), "n_clamp_units": n_clamp, "hands": hands}))
        # 062 鼓动段 / 防蹭段：≥6 根 + 蹭的压力
        if len(run) >= ctx.p["drumbeat_min_units"]:
            bars = {u.measure for u in run}
            sp = sum(1 for m in ctx.ha.scrape if m.measure in bars)
            out.append(_span_tasks(
                ctx, [u.ti for u in run] + [u.head_ti for u in run if u.head_ti >= 0],
                "鼓动段", "062",
                f"拍滑单元等间隔左右交替 {len(run)} 次（≥6 = 教程口径的鼓动段），"
                f"段内蹭的压力 {sp} 次",
                f"两手严格交替、每根一换（{hands}）；摆位互相挡路",
                {"n_units": len(run), "scrape_pressure": sp, "hands": hands}))
    return out


def _clamp_keys(ctx: HandCtx, u: SlideUnit) -> int:
    """知识 053：头与启动拍之间落在**同一个键**上的任务数（夹键）。"""
    cnt = 0
    for t in ctx.ha.tasks:
        if t.kind in ("slide", "wifi"):
            continue
        if t.key == u.key and u.t_head + 1e-4 < t.t < u.t_start - 1e-4:
            cnt += 1
    return cnt


def detect_double_tapslide(ctx: HandCtx, units: Sequence[SlideUnit]
                           ) -> list[ConfigHitH]:
    """**050 双压连续拍滑** 与 **051 CYCLES 型星星**（同相位家族）。

    规格（§5.1）：050 = 两手同刻同相位、**段内不换手**（每手键序列常量、`pairs` 连续）；
    051 = 成对 slide 共享两端点、方向相反。
    """
    out: list[ConfigHitH] = []
    by_head: dict[int, list[SlideUnit]] = {}
    for u in units:
        by_head.setdefault(int(round(u.t_head * 1000)), []).append(u)
    slots = [(k, v) for k, v in sorted(by_head.items()) if len(v) == 2]
    # --- 050：连续 ≥3 个"两头同刻"的格，且两手各自键位常量 ---
    i = 0
    while i < len(slots):
        j = i
        while j + 1 < len(slots):
            dt = (slots[j + 1][0] - slots[j][0]) / 1000.0
            a = slots[j][1][0]
            spb = 60.0 / a.bpm if a.bpm else 0.5
            if dt <= 1e-6 or dt > 2.2 * spb:
                break
            j += 1
        grp = slots[i:j + 1]
        if len(grp) >= ctx.p["double_tapslide_min"]:
            kl = [next((u.key for u in g if u.hand == "L"), None) for _, g in grp]
            kr = [next((u.key for u in g if u.hand == "R"), None) for _, g in grp]
            hands_fixed = (len([k for k in kl if k]) == len(grp)
                           and len([k for k in kr if k]) == len(grp)
                           and len(set(kl)) == 1 and len(set(kr)) == 1)
            sets = {frozenset(u.key for u in g) for _, g in grp}
            const = len(sets) == 1 and len(next(iter(sets))) == 2
            if const:
                allu = [u for _, g in grp for u in g]
                out.append(_span_tasks(
                    ctx, [u.ti for u in allu] + [u.head_ti for u in allu if u.head_ti >= 0],
                    "双压连续拍滑", "050",
                    f"{len(grp)} 个四分音每格两头同刻，键对恒定 "
                    f"{sorted(next(iter(sets)))}"
                    + ("、段内不换手" if hands_fixed else "（中途两手互换了守的键）"),
                    (f"L:守键 {kl[0]} 拍头+划轨  R:守键 {kr[0]} 拍头+划轨（同相位）"
                     if hands_fixed else
                     f"两手同相位各守一键，段内换过手（L {kl} / R {kr}）"),
                    {"n_units": len(grp), "key_pair": sorted(next(iter(sets))),
                     "hands_fixed": hands_fixed,
                     "key_l": kl[0], "key_r": kr[0]}))
        i = max(j, i) + 1
    # --- 051：同刻两根共享端点、方向相反 ---
    for _, g in slots:
        a, b = g[0], g[1]
        if a.end == b.key and b.end == a.key and a.key != b.key:
            out.append(_span_tasks(
                ctx, [a.ti, b.ti], "CYCLES型星星", "051",
                f"同刻两根共享端点、方向相反（{_u_desc(a)} + {_u_desc(b)}）",
                f"L/R 各守一条镜像曲线、段内不换手（{a.hand}/{b.hand}）",
                {"pair": [_u_desc(a), _u_desc(b)]}))
    return _dedup(out)


# ---------- 058 / 060 / 061 同头星星家族 ----------


def _radiate_groups(ctx: HandCtx, units: Sequence[SlideUnit]
                    ) -> list[list[SlideUnit]]:
    """同一头键在一个时间窗内连续放射的星星组（知识 058/060/061 的共同骨架）。"""
    groups: list[list[SlideUnit]] = []
    by_key: dict[int, list[SlideUnit]] = {}
    for u in units:
        by_key.setdefault(u.key, []).append(u)
    for key, us in by_key.items():
        us = sorted(us, key=lambda x: x.t_head)
        i = 0
        while i < len(us):
            j = i
            while j + 1 < len(us):
                spb = 60.0 / us[j].bpm if us[j].bpm else 0.5
                if us[j + 1].t_head - us[j].t_head > ctx.p["radiate_window_beats"] / 4 * spb:
                    break
                j += 1
            if j - i + 1 >= 2:
                groups.append(us[i:j + 1])
            i = j + 1
    return groups


def detect_same_head(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**060 同起点拍滑 / 061 三叉戟 / 058 挥手段**（同头星星三分）。

    规格（§5.1）：
    - 060：同一头键放射 ≥4 根、**两手交替各领一根**；``出张`` = 与 007 末尾半边冲突的根数；
    - 061：同一头键**恰 3 根全给一只手**，另一手独立线；
    - 058：某手的 slide 头键常量且 ≥3 根、终点交替落在两侧，另一手做独立线、段内不换。

    **手级增量**：060 与 061 在键位上完全一样（都是"同一个键放射 N 根"），
    键位版分不开；分界线**只在手上**——两手轮流 vs 一只手包下。
    """
    out: list[ConfigHitH] = []
    for g in _radiate_groups(ctx, units):
        key = g[0].key
        hands = "".join(u.hand for u in g)
        n = len(g)
        alt = _alt_rate([u.hand for u in g])
        hs_list = [u.hand for u in g]
        main_hand = max(set(hs_list), key=hs_list.count)
        share = hs_list.count(main_hand) / len(hs_list)
        # 知识 058/061 都说"段内不换手"；允许少量离群根（agent 操作化）
        one_hand = share >= ctx.p["same_head_one_hand_share"]
        exact_one = len(set(hs_list)) == 1
        chz = sum(1 for u in g if u.chuzhang)
        ends = [u.end for u in g]
        tis = [u.ti for u in g] + [u.head_ti for u in g if u.head_ti >= 0]
        # 另一只手在同窗里是否另有一条线
        t0, t1 = g[0].t_head, g[-1].t_end
        oth = _other(main_hand) if one_hand else ""
        own = {u.ti for u in g} | {u.head_ti for u in g}
        n_other = sum(1 for t, h in zip(ctx.ha.tasks, ctx.ha.task_hand)
                      if h == oth and t0 - 1e-6 <= t.t <= t1 + 1e-6
                      and t.index not in own) if oth else 0
        if n >= ctx.p["dealing_min_roots"] and alt >= 0.75 and not one_hand:
            out.append(_span_tasks(
                ctx, tis, "同起点拍滑", "060",
                f"头键 {key} 放射 {n} 根（终点 {ends}），两手交替各领一根 {hands}；"
                f"出张 {chz} 根",
                f"两手交替：拍谁谁划、头领先轨道一拍（{hands}）",
                {"head_key": key, "n_roots": n, "ends": ends, "hands": hands,
                 "chuzhang": chz, "alt_rate": round(alt, 3)}))
        if n >= ctx.p["trident_roots"] and exact_one:
            out.append(_span_tasks(
                ctx, tis, "三叉戟", "061",
                f"头键 {key} 放射 {n} 根**全给 {main_hand} 一只手**（终点 {ends}），"
                f"{oth} 手同窗另有 {n_other} 个任务"
                + ("；canonical（头在 1/8 分页缝上）" if key in (1, 8) else ""),
                f"{main_hand}:包下 {n} 根（拍头+划轨）  {oth}:完全独立的另一条线",
                {"head_key": key, "n_roots": n, "ends": ends,
                 "hand": main_hand, "n_other_tasks": n_other,
                 "canonical": key in (1, 8)}))
        if (n >= ctx.p["wave_min_roots"] and one_hand
                and n_other >= max(2, n // 2)
                and _ends_alternate_sides(key, ends)):
            out.append(_span_tasks(
                ctx, tis, "挥手段", "058",
                f"{main_hand} 手钉在头键 {key} 甩出 {n} 根（主手占比 {share:.0%}）、"
                f"终点交替落在两侧 {ends}；"
                f"{oth} 手同窗 {n_other} 个任务（独立线）",
                f"{main_hand}:钉在 {key} 拍头+划轨（从中间向两边）  {oth}:自由手另一条线",
                {"head_key": key, "n_roots": n, "ends": ends,
                 "hand": main_hand, "hand_share": round(share, 3),
                 "n_other_tasks": n_other}))
    return _dedup(out)


def _ends_alternate_sides(key: int, ends: Sequence[int | None]) -> bool:
    """终点是否在头键的两侧交替（知识 058「从中间向两边」）。"""
    sides = [1 if 1 <= ((e - key) % 8) <= 3 else -1
             for e in ends if e is not None and (e - key) % 8 != 0]
    if len(sides) < 3 or len(set(sides)) < 2:
        return False
    return sum(1 for i in range(1, len(sides)) if sides[i] != sides[i - 1]) \
        >= len(sides) // 2


# ---------- 055 / 056 / 059 / 063 ----------


def detect_pierce(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**055 穿心 / 大风车**（知识 055）。

    判据（ep3 §6.2 A 档）：``(end−head) mod 8 == 4`` 的直线 slide，相邻两根 head
    ``Δ=±1`` 同向、连续 ≥3 根 → 穿心；head 走满 8 根 → 大风车。
    手级增量：记录"每根一换"是否成立（正攻），以及出张根数。
    """
    out: list[ConfigHitH] = []
    cand = [u for u in units if u.end is not None and (u.end - u.key) % 8 == 4]
    i = 0
    while i < len(cand):
        j = i
        d0: int | None = None
        while j + 1 < len(cand):
            a, b = cand[j], cand[j + 1]
            d = cstep(a.key, b.key)
            spb = 60.0 / a.bpm if a.bpm else 0.5
            if abs(d) != 1 or b.t_head - a.t_head > 2.2 * spb:
                break
            if d0 is None:
                d0 = d
            elif d != d0:
                break
            j += 1
        g = cand[i:j + 1]
        if len(g) >= ctx.p["pierce_min_roots"]:
            hands = "".join(u.hand for u in g)
            chz = sum(1 for u in g if u.chuzhang)
            windmill = len(g) >= 7
            out.append(_span_tasks(
                ctx, [u.ti for u in g] + [u.head_ti for u in g if u.head_ti >= 0],
                "大风车" if windmill else "穿心", "055",
                f"{len(g)} 根过心直星沿圈同向推进（头 "
                f"{','.join(str(u.key) for u in g[:8])}），划轨手 {hands}，出张 {chz}",
                f"每根一换（{hands}）：拍头手就近、划轨手同一只",
                {"n_roots": len(g), "dir": d0, "hands": hands, "chuzhang": chz,
                 "windmill": windmill}))
        i = max(j, i) + 1
    return out


def detect_circling(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**056 绕圈星星**（知识 056，agent 推断为主——教程原文只有一句）。

    判据：弧型 slide（`^`/`<`/`>`）首尾相连、沿圆周同向接力 ≥3 根。
    手级增量：报出"是不是一只手一路划到底"（知识 056 推断的最经济打法 = 057 一笔画）。
    """
    out: list[ConfigHitH] = []
    arcs = [u for u in units if u.is_arc]
    i = 0
    while i < len(arcs):
        chain = [arcs[i]]
        j = i
        while j + 1 < len(arcs):
            a, b = chain[-1], arcs[j + 1]
            spb = 60.0 / a.bpm if a.bpm else 0.5
            if b.key != a.end or b.t_head - a.t_head > 4 * spb:
                break
            chain.append(b)
            j += 1
        if len(chain) >= ctx.p["circle_min_roots"]:
            hands = "".join(u.hand for u in chain)
            out.append(_span_tasks(
                ctx, [u.ti for u in chain] + [u.head_ti for u in chain if u.head_ti >= 0],
                "绕圈星星", "056",
                f"{len(chain)} 根弧线首尾相连沿圆周接力 "
                + " → ".join(_u_desc(u) for u in chain[:5])
                + f"；划轨手 {hands}",
                (f"{hands[0]}:一路划到底（= 057 一笔画）" if len(set(hands)) == 1
                 else f"跨中线换手（{hands}）"),
                {"n_roots": len(chain), "hands": hands,
                 "one_hand": len(set(hands)) == 1}))
        i = max(j, i) + 1
    return out


def detect_well(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**059 井字星星 / 四合院**（知识 059，**B 档**：教程只给"45° 交叉"四个字）。

    判据：≥2 条 ``|head−end| mod 8 ∈ {2,3}`` 的直线 slide 在时间上重叠、
    两手各领一条（agent 操作化：把"45° 交叉"读成"两条弦的跨度方向不同"）。
    """
    out: list[ConfigHitH] = []
    chords = [u for u in units if u.end is not None and u.is_straight
              and cdist(u.key, u.end) in (2, 3)]
    for i in range(len(chords)):
        for j in range(i + 1, len(chords)):
            a, b = chords[i], chords[j]
            if b.t_head > a.t_end:
                break
            if a.hand == b.hand or a.hand not in ("L", "R"):
                continue
            if {a.key, a.end} & {b.key, b.end}:
                continue           # 共端点的是一笔画/CYCLES，不是井字
            if not _chords_cross(a.key, a.end, b.key, b.end):
                continue           # 必须真的交叉（教程的"45° 交叉"）
            out.append(_span_tasks(
                ctx, [a.ti, b.ti], "井字星星", "059",
                f"两条不过圆心的弦在时间上交叠（{_u_desc(a)} × {_u_desc(b)}），"
                f"两手各领一条（{a.hand}/{b.hand}）",
                f"{a.hand}:{_u_desc(a)}  {b.hand}:{_u_desc(b)}（手走外圈）",
                {"chords": [_u_desc(a), _u_desc(b)]}))
    return _dedup(out)


def _chords_cross(a1: int, a2: int, b1: int, b2: int) -> bool:
    """两条弦在圆上是否相交：端点沿环**交错**即相交（agent 操作化"45° 交叉"）。"""
    def inside(x: int) -> bool:
        lo, hi = a1 % 8, a2 % 8
        return ((x - lo) % 8) < ((hi - lo) % 8) and ((x - lo) % 8) > 0
    return inside(b1 % 8) != inside(b2 % 8)


def detect_defuse(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**063 拆弹**（知识 063，**B 档**：教程原文只有"不按顺序打会爆炸的星星"）。

    判据：任一时刻"**已落头但尚未划完**"的 slide 数 ≥3 且持续 > 一个四分音。
    手级增量：报出这些同时挂屏的星星分给了几只手（同一只手挂 ≥2 根 = 真的要排顺序）。
    """
    out: list[ConfigHitH] = []
    events: list[tuple[float, int, SlideUnit]] = []
    for u in units:
        events.append((u.t_head, 1, u))
        events.append((u.t_end, -1, u))
    events.sort(key=lambda e: (e[0], e[1]))
    live: list[SlideUnit] = []
    run_start: float | None = None
    run_units: set[int] = set()
    for t, d, u in events:
        if d > 0:
            live.append(u)
        elif u in live:
            live.remove(u)
        if len(live) >= ctx.p["defuse_min_pending"]:
            if run_start is None:
                run_start = t
            run_units |= {x.ti for x in live}
        else:
            if run_start is not None:
                spb = 60.0 / u.bpm if u.bpm else 0.5
                if t - run_start > spb and run_units:
                    us = [x for x in units if x.ti in run_units]
                    hs = "".join(x.hand for x in us)
                    same = max(hs.count("L"), hs.count("R"))
                    out.append(_span_tasks(
                        ctx, sorted(run_units), "拆弹", "063",
                        f"同屏 ≥{ctx.p['defuse_min_pending']} 根"
                        f"「已落头未划完」持续 {t - run_start:.2f}s，涉及 {len(us)} 根；"
                        f"单手最多挂 {same} 根",
                        f"划轨手 {hs}（顺序错一根就连锁）",
                        {"n_slides": len(us), "dur_sec": round(t - run_start, 3),
                         "hands": hs, "max_one_hand": same}))
            run_start, run_units = None, set()
    return _dedup(out)


# ---------- 033 / 034 / 035 / 036 / 037 / 043 / 039–041 ----------


def detect_scythe(ctx: HandCtx, units: Sequence[SlideUnit]) -> list[ConfigHitH]:
    """**033 死镰段** 与 **034 水神段**（知识 033/034）。

    规格：一手走圈 tap 链（每步 ±1 同向、≥8 音）+ 另一手划**反向贴边弧**；
    034 = 同骨架但把贴边弧换成穿心直星。

    手级增量：知识 033 的核心就是"双手长期分任务"——走圈的手与划星的手必须是
    **不同的两只**；键位版只能看到"圆周走位里夹着星星头"。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    n = len(slots)
    walk = [s for s in slots if s.n_hits == 1 and s.hit_keys[0] is not None]
    i = 0
    while i < len(walk):
        j, d0 = i, None
        while j + 1 < len(walk):
            a, b = walk[j], walk[j + 1]
            d = cstep(a.hit_keys[0], b.hit_keys[0])
            spb = 60.0 / a.bpm if a.bpm else 0.5
            if abs(d) != 1 or b.t - a.t > 1.1 * spb:
                break
            if d0 is None:
                d0 = d
            elif d != d0:
                break
            j += 1
        g = walk[i:j + 1]
        if len(g) >= ctx.p["scythe_min_walk"]:
            t0, t1 = g[0].t, g[-1].t
            keyset = {s.hit_keys[0] for s in g}
            inner = [u for u in units if t0 - 1e-6 <= u.t_head <= t1 + 1e-6
                     and u.key in keyset]
            arcs = [u for u in inner if u.is_arc]
            straights = [u for u in inner if u.is_straight and u.end is not None
                         and abs(((u.end - u.key + 4) % 8) - 4) <= 1]
            hands = [s.hit_hands[0] for s in g]
            main = max(set(hands), key=hands.count)
            share = hands.count(main) / len(hands)
            for pool, name, src in ((arcs, "死镰段", "033"),
                                    (straights, "水神段", "034")):
                cross = [u for u in pool if u.hand != main]
                if (len(pool) >= 2 and len(cross) >= 2
                        and share >= ctx.p["scythe_walk_share"]):
                    out.append(_span(
                        ctx, g[0].index, g[-1].index, name, src,
                        f"{main} 手沿圈走 {len(g)} 个 tap（每步 {d0:+d}，占比 {share:.0%}），"
                        f"{_other(main)} 手划 {len(cross)} 根"
                        + ("贴边弧" if name == "死镰段" else "穿心直星"),
                        f"{main}:走圈 tap（定位）  {_other(main)}:按时划星（防蹭）",
                        {"walk_len": len(g), "walk_dir": d0, "walk_hand": main,
                         "walk_share": round(share, 3), "n_slides": len(cross),
                         "slides": [_u_desc(u) for u in cross][:8]}))
        i = max(j, i) + 1
    return _dedup(out)


def detect_dragon(ctx: HandCtx) -> list[ConfigHitH]:
    """**035 如龙段**（知识 035）。

    判据（ep2 §6 A 档）：细胞 = 「双押 (n, n±2) → 单点 n±1 → 单点 n±2」，
    相邻细胞逐格平移、连续 ≥3 个。
    手级增量：报出"扫过三键的是哪只手"——知识 035 的手序推断是"一手扫三键、
    另一手补双押远端"（agent 推断，存疑）。
    """
    out: list[ConfigHitH] = []
    slots = ctx.slots
    cells: list[tuple[int, int, int, str]] = []     # (起槽, 止槽, n, 扫键手)
    for i in range(len(slots) - 2):
        a, b, c = slots[i], slots[i + 1], slots[i + 2]
        if not a.is_pair or b.n_hits != 1 or c.n_hits != 1:
            continue
        ks = sorted(a.hit_key_set)
        if len(ks) != 2 or ((ks[1] - ks[0]) % 8) not in (2, 6):
            continue
        # ⚠️ 知识 035 的手序（"一手扫三键、另一手补双押远端"）在条目里就标了
        # **agent 推断**；官谱 `312-Outlaw's Lullaby` m058–m061 上分配器给出的是
        # **两手交替扫**（＝教程说的"全换手法"）。因此这里**不**把"同手扫"写成
        # 必要条件，只把实际手序记进证据。
        # 双押的两个键相差 2 格 → 细胞可以从任一端起、向另一端同向扫
        for n0, step in ((ks[0], 1), (ks[1], -1), (ks[1], 1), (ks[0], -1)):
            if ((n0 + 2 * step - 1) % 8 + 1) not in ks:
                continue
            if (b.hit_keys[0] == (n0 + step - 1) % 8 + 1
                    and c.hit_keys[0] == (n0 + 2 * step - 1) % 8 + 1):
                cells.append((i, i + 2, n0, b.hit_hands[0] + c.hit_hands[0]))
                break
    i = 0
    while i < len(cells):
        j = i
        while j + 1 < len(cells):
            # 允许细胞之间夹 ≤1 个被截断的双押（官谱 Outlaw m058 每小节末尾就是这样）
            if not (cells[j][1] + 1 <= cells[j + 1][0] <= cells[j][1] + 2):
                break
            if abs(cstep(cells[j][2], cells[j + 1][2])) != 1:
                break
            j += 1
        if j - i + 1 >= ctx.p["dragon_min_cells"]:
            g = cells[i:j + 1]
            hs = "/".join(x[3] for x in g)
            same = all(x[3][0] == x[3][1] for x in g)
            out.append(_span(
                ctx, g[0][0], g[-1][1], "如龙段", "035",
                f"{len(g)} 个细胞「双押(n,n±2)→单点 n±1→单点 n±2」逐格平移"
                f"（n = {','.join(str(x[2]) for x in g)}），扫键手序 {hs}"
                f"（{'同手扫三键' if same else '两手交替扫＝全换手法'}）",
                (f"一手把三键同向扫过去、另一手补双押远端（{hs}）" if same
                 else f"两手交替扫（全换手法，{hs}）"),
                {"n_cells": len(g), "n_seq": [x[2] for x in g],
                 "sweep_hands": hs, "same_hand_sweep": same}))
            i = j + 1
        else:
            i += 1
    return out


def detect_sweep(ctx: HandCtx, runs: Sequence[tuple[int, int]]) -> list[ConfigHitH]:
    """**036 二连扫**（知识 036）。

    判据：连续同向相邻走位 ≥6，且**键位集合不封闭**（>4 键）——封闭的 4 键是
    024 大宇宙。手级增量：报出两手是否真的在交替扫（教程原文"两只手交替进行扫"）。
    """
    out: list[ConfigHitH] = []
    for a, b in runs:
        hs, ks = _hand_runs(ctx, a, b)
        i = 0
        while i < len(ks) - 1:
            d0 = cstep(ks[i], ks[i + 1])
            if abs(d0) != 1:
                i += 1
                continue
            j = i + 1
            while j + 1 < len(ks) and cstep(ks[j], ks[j + 1]) == d0:
                j += 1
            ln = j - i + 1
            if ln >= ctx.p["sweep_min_len"] and len(set(ks[i:j + 1])) > 4:
                sub_h = hs[i:j + 1]
                out.append(_span(
                    ctx, a + i, a + j, "二连扫", "036",
                    f"{ln} 音同向相邻推进（{d0:+d}）、用键 {len(set(ks[i:j+1]))} 个"
                    f"（不封闭 → 不是大宇宙），手序 {''.join(sub_h)}",
                    f"两手交替扫（交替率 {_alt_rate(sub_h):.2f}）",
                    {"length": ln, "dir": d0, "n_keys": len(set(ks[i:j + 1])),
                     "alt_rate": round(_alt_rate(sub_h), 3)}))
            i = j
    return _dedup(out)


def detect_backhand(ctx: HandCtx) -> list[ConfigHitH]:
    """**037 反手 / 反手交互**（知识 037）。

    判据（ep2 §6，**必须有手序模型**）：存在连续 ≥4 个音，其手序分配使两手
    **真正交叉**（左手落在右半圈且右手落在左半圈），或**两手被挤进同一半圈且其中
    至少一只手出张**。

    这是键位版**根本无法实现**的一条——同一串音符换个手序就从普通交互变成反手。

    ⚠️ **v0.2 收紧了"同半圈"这一支**（知识 064）：v0.1 只要求"两手同在一个半圈"，
    那是建立在分页二分上的——舒适区模型下左手打 1/4、右手打 5/8 本来就零代价，
    "两手同在右半圈（如 L=1、R=4）"根本不难，再叫反手与知识 037「**出张的极端形**」
    的定义冲突。改为**必须有一只手真的出张**（L→2/3 或 R→6/7）。
    全库效果：5770 → 见报告 §"v0.2 重跑"。

    ⚠️ **v0.3（2026-09-20）把 ``crossed`` 这一支也按 064 收紧**
    （`docs/research/side-double-guidance-and-chuzhang.md` §5.5 / §6.2 建议 1）：
    v0.2 只收紧了 ``same_half``，``crossed = ka in 1234 and kb in 5678`` 仍是 **006 的
    中线**——结果 388 官谱上报出的 **266 段反手里 217 段（81.6%）一个出张落点都没有**，
    全是 ``4,5,4,5`` / ``1,8,1,8`` 这种两手在 8–4 / 8–1 轴两侧交替敲
    （`61-天国と地獄` m018/m052、`491-極圏` m097、`345-Oshama Scramble!` m006…）。
    按知识 064，``L→1/4`` 与 ``R→5/8`` 都在共享区，那既不是出张、更不是"出张的极端形"。
    **现在两支同一条件：至少一只手真出张。**
    普查量级：官谱里"跨共享键的交叉"4 167 个交叉步中 **94.8% 两手都站在共享键
    ``{1,4,5,8}`` 上**（``(L4,R5)`` 1 929、``(L1,R8)`` 1 473），出现在 374/388 张谱——
    它就是扫键 / 楼梯在 8–4 轴上的正常换手，**不需要引导也不吃协调力**。
    教程口径（037 的"跨半圈"）只作教学语汇保留，机检以 064 为准。
    """
    out: list[ConfigHitH] = []
    slots = [s for s in ctx.slots if s.n_hits == 1 and s.hit_keys[0] is not None]
    n = len(slots)
    i = 0
    while i < n - 1:
        j = i
        while j + 1 < n:
            a, b = slots[j], slots[j + 1]
            spb = 60.0 / a.bpm if a.bpm else 0.5
            if a.hit_hands[0] == b.hit_hands[0] or b.t - a.t > 1.1 * spb:
                break
            ka = a.hit_keys[0] if a.hit_hands[0] == "L" else b.hit_keys[0]
            kb = b.hit_keys[0] if a.hit_hands[0] == "L" else a.hit_keys[0]
            if ka == kb:
                break            # 两手敲同一个键是 019 纵连的"拆"，不是 037 反手
            # ka = 左手键、kb = 右手键
            same_half = ((ka in _RIGHT_HOME and kb in _RIGHT_HOME)
                         or (ka in _LEFT_HOME and kb in _LEFT_HOME))
            crossed = ka in _RIGHT_HOME and kb in _LEFT_HOME
            if not (same_half or crossed):
                break
            # 知识 064：**两支都一样**——只有在真的出张时才是 037 说的"出张的极端形"。
            # （v0.3 把 `crossed` 支也收进来；v0.2 只收紧了 `same_half`，见 docstring。）
            if not (is_chuzhang("L", ka) or is_chuzhang("R", kb)):
                break
            j += 1
        ln = j - i + 1
        if ln >= ctx.p["backhand_min_len"]:
            ks = [s.hit_keys[0] for s in slots[i:j + 1]]
            hs = "".join(s.hit_hands[0] for s in slots[i:j + 1])
            out.append(_span(
                ctx, slots[i].index, slots[j].index, "反手", "037",
                f"{ln} 音里两手始终被挤进同一半圈/交叉，且始终有一只手出张"
                f"（键 {_fmt_keys(ks)}，手序 {hs}）",
                f"L/R 交叉或同半圈 + 出张（协调力，非手速）",
                {"length": ln, "keys": ks[:12], "hands": hs}))
            i = j + 1
        else:
            i += 1
    return out


def detect_chuzhang(ctx: HandCtx) -> list[ConfigHitH]:
    """**043 出张**（知识 043）。

    **v0.2 改按知识 064（用户 2026-09-19）的键位定义**：出张 = **右手落 6/7 或
    左手落 2/3**，只有这四个「手 × 键」组合；共享区 ``{1,4,5,8}`` 两手都不算。
    知识 043 的教程侧粗口径（"左手打右边"）与 ep2 §6 的 ``page_depth ≥ 1`` 代理
    **一并作废**——后者会把右手打 5/8、左手打 1/4 也算成跨界，与用户口径冲突。

    **出张不分档**（用户 2026-09-20：「出张本来是一件很小的事啊，怎么就分档了？」）——
    它就是 ``is_chuzhang(hand, key)`` 这一个**布尔判定**加一个**计数**；跨到 6 还是 7、
    跨多久、跨多频繁都不构成档位，本模块不得自造分级。

    **两种成段方式**（v0.3，2026-09-20 起并列；依据
    `docs/research/side-double-guidance-and-chuzhang.md` §4.2 / §6.3 建议 1）：

    1. **被占手逼出**：以每条 slide/hold 的 **[头, 名义尾]** 为窗口（这段时间那只手被
       钉住），统计窗口内**落进出张区**的任务数；≥ ``chuzhang_min_cross`` 且其中至少
       有一个是**击打**（不是纯轨道）时记成一段。
    2. **纯单点出张**（v0.3 新增）：一串**连续的单任务槽**里落了 ≥ ``chuzhang_min_cross``
       个出张点，而这段时间**两只手都没有被 slide / hold 钉住**——
       既不是被谁逼出来的，也不是双押/纵连的副产品，就是"这一下由那只手更顺"。

    加第 2 种的理由是普查证据：388 官谱 5 739 个出张落点里，
    **「另一只手被星星钉住 20.8% + 被长条钉住 15.4%」合计只占 36.2%**，
    而**纯单点出张 30.2% 是最大的一类**（出张的是星星自己 22.3%、同刻双押的另一半
    9.1%、落在纵连串里 2.1%）。只留第 1 种会让"出张段数"系统性地少掉最大的那一类，
    读者容易把它误当成"出张总量"——**全谱出张落点计数一律看 `BarHand.chuzhang`
    / `chart_summary` 的 ``chuzhang``，本检测器给的是"段"。**

    ``chuzhang_min_cross`` 只是"几个落点才值得单独记一段"的**内部计数口径**
    （agent 操作化），不是难度分级，也不向用户索取。
    """
    out: list[ConfigHitH] = []
    ha = ctx.ha
    for ti, t in enumerate(ha.tasks):
        if t.kind not in ("slide", "wifi", "hold"):
            continue
        h = ctx.hand(ti)
        if h not in ("L", "R"):
            continue
        t0 = t.t
        if t.kind in ("slide", "wifi") and t.star_idx >= 0:
            t0 = ha.tasks[t.star_idx].t
        t1 = t.t_end_nominal or t.t_end
        cross = [j for j, x in enumerate(ha.tasks)
                 if x.key is not None and t0 - 1e-6 <= x.t <= t1 + 1e-6
                 and ctx.hand(j) in ("L", "R")
                 and is_chuzhang(ctx.hand(j), x.key)]
        hits_cross = [j for j in cross
                      if ha.tasks[j].kind not in ("slide", "wifi")]
        if len(cross) >= ctx.p["chuzhang_min_cross"] and hits_cross:
            desc = ", ".join(f"{ctx.hand(j)}→{ha.tasks[j].key}" for j in cross[:6])
            out.append(_span_tasks(
                ctx, [ti] + cross, "出张", "043",
                f"{h} 手被 {t.kind}（{t.key}→{t.end_key or ''}）钉住 {t1 - t0:.2f}s，"
                f"窗口内 {len(cross)} 个任务落进出张区（{desc}）",
                f"{h}:被钉住  出张落点 {len(cross)} 个（知识 064：R→6/7、L→2/3）",
                {"mode": "被占手逼出", "pin_hand": h, "pin_kind": t.kind,
                 "n_cross": len(cross),
                 "cross": [[ctx.hand(j), ha.tasks[j].key] for j in cross][:8],
                 "dur_sec": round(t1 - t0, 3)}))
    out.extend(_chuzhang_pure_singles(ctx))
    return _dedup(out)


def _chuzhang_pure_singles(ctx: HandCtx) -> list[ConfigHitH]:
    """**纯单点出张**（报告 §4.2 类 6，官谱里最大的一类，30.2%）。

    形态：一串**连续的单任务槽**（相邻槽间隔 ≤1 拍，与 `detect_backhand` 同口径）里
    出现 ≥ ``chuzhang_min_cross`` 个出张落点，且这段时间**没有任何一只手被
    slide / wifi / hold 钉住**——没有被占手，也不是双押的另一半、不是纵连，
    就是把一个 tap/hold 交给了对侧的手。

    "没有手被钉住"这一条是为了**与上面那支互斥**（那支专管"被占手逼出"的出张），
    不是难度判据。整段判定里没有任何阈值：``chuzhang_min_cross`` 是成段计数口径，
    "≤1 拍"是"算不算同一串"的连读口径（同 037 反手）。
    """
    out: list[ConfigHitH] = []
    ha = ctx.ha
    pinned = [(t.t if t.kind == "hold" else
               (ha.tasks[t.star_idx].t if t.star_idx >= 0 else t.t),
               t.t_end_nominal or t.t_end)
              for t in ha.tasks if t.kind in ("slide", "wifi", "hold")]
    slots = [s for s in ctx.slots if s.n_hits == 1 and s.hit_keys[0] is not None]
    n = len(slots)
    i = 0
    while i < n:
        j = i
        while j + 1 < n:
            a, b = slots[j], slots[j + 1]
            spb = 60.0 / a.bpm if a.bpm else 0.5
            if b.t - a.t > 1.1 * spb:
                break
            j += 1
        seg = slots[i:j + 1]
        cross = [s for s in seg
                 if is_chuzhang(s.hit_hands[0], s.hit_keys[0])]
        if len(cross) >= ctx.p["chuzhang_min_cross"]:
            t0, t1 = seg[0].t, seg[-1].t
            if not any(a < t1 + 1e-6 and b > t0 - 1e-6 for a, b in pinned):
                desc = ", ".join(f"{s.hit_hands[0]}→{s.hit_keys[0]}" for s in cross[:6])
                out.append(_span(
                    ctx, seg[0].index, seg[-1].index, "出张", "043",
                    f"{len(seg)} 个单点里 {len(cross)} 个落进出张区（{desc}）；"
                    f"这段两只手都没有被长条/星星钉住",
                    f"纯单点出张 {len(cross)} 个（知识 064：R→6/7、L→2/3）",
                    {"mode": "纯单点出张", "n_cross": len(cross),
                     "cross": [[s.hit_hands[0], s.hit_keys[0]] for s in cross][:8],
                     "n_slots": len(seg), "dur_sec": round(t1 - t0, 3)}))
        i = j + 1
    return out


def detect_nplus1(ctx: HandCtx) -> list[ConfigHitH]:
    """**039 2+1 / 040 3+1 / 041 N+1**（知识 039–041）。

    判据（ep2 §6 A 档）：单任务槽做游程编码，找 ``(K^n, X)`` 循环——
    039 n=2 且重复 ≥3；040 n=3 且重复 ≥2；041 n 在段内取 ≥2 个不同值且重复 ≥3。
    手级增量：报出"第 N+1 个音落到哪只手"的**奇偶翻转**（知识 041 的难点）。
    """
    out: list[ConfigHitH] = []
    slots = [s for s in ctx.slots if s.n_hits == 1 and s.hit_keys[0] is not None]
    if len(slots) < 6:
        return out
    # 游程编码（同键连打），要求相邻槽间隔 ≤1 拍
    runs: list[tuple[int, int, int]] = []        # (起, 止, 键)
    i = 0
    while i < len(slots):
        j = i
        while (j + 1 < len(slots) and slots[j + 1].hit_keys[0] == slots[i].hit_keys[0]
               and 0 < slots[j + 1].beat - slots[j].beat <= 1.0):
            j += 1
        runs.append((i, j, slots[i].hit_keys[0]))
        i = j + 1
    # 找 (K^n, X^1) 的循环
    i = 0
    while i + 3 < len(runs):
        cyc: list[int] = []
        j = i
        while j + 1 < len(runs):
            a, b = runs[j], runs[j + 1]
            na, nb = a[1] - a[0] + 1, b[1] - b[0] + 1
            if nb != 1 or na < 2 or na > 6:
                break
            if slots[b[0]].beat - slots[a[1]].beat > 1.0:
                break
            cyc.append(na)
            j += 2
        if len(cyc) >= 2:
            s0, s1 = runs[i][0], runs[min(j, len(runs) - 1)][1]
            ns = sorted(set(cyc))
            hs = "".join(s.hit_hands[0] for s in slots[s0:s1 + 1])
            if len(ns) >= 2 and len(cyc) >= ctx.p["nplus1_min_cycles"]:
                name, src = "N+1", "041"
            elif ns == [2] and len(cyc) >= ctx.p["nplus1_min_cycles"]:
                name, src = "2+1", "039"
            elif ns == [3] and len(cyc) >= ctx.p["nplus1_min_cycles_3"]:
                name, src = "3+1", "040"
            else:
                i = max(j, i + 1)
                continue
            flips = sum(1 for k in range(len(cyc)) if cyc[k] % 2 == 1)
            out.append(_span(
                ctx, slots[s0].index, slots[s1].index, name, src,
                f"(K^n, X) 循环 {len(cyc)} 次，n = {cyc}；手序 {hs}"
                + (f"；奇数 n 出现 {flips} 次 → 换手奇偶翻转" if flips else ""),
                f"同键 n 连（{'两手拆' if _alt_rate(hs) > 0.4 else '一手连打'}）+ 单点换手",
                {"n_seq": cyc, "cycles": len(cyc), "hands": hs,
                 "parity_flips": flips}))
            i = j
        else:
            i += 1
    return _dedup(out)


def detect_polyrhythm(ctx: HandCtx) -> list[ConfigHitH]:
    """**045 混 fr**（知识 045，**B 档 / 存疑**）：两手各跑一条**不同分音**的节奏线。

    判据（agent 操作化）：窗口内两手各 ≥4 个任务、各自间隔匀齐（相对误差 ≤0.15），
    且两手的间隔之比**不是整数也不是整数倒数**（差 >0.15）→ 打 `polyrhythm`。
    """
    out: list[ConfigHitH] = []
    bars: dict[int, list[HSlot]] = {}
    for s in ctx.slots:
        bars.setdefault(s.measure, []).append(s)
    for m, ss in sorted(bars.items()):
        seq = {"L": [], "R": []}
        for s in ss:
            for hand in ("L", "R"):
                if s.hand_task(hand) is not None:
                    seq[hand].append(s.beat)
        if len(seq["L"]) < 4 or len(seq["R"]) < 4:
            continue
        info = {}
        ok = True
        for hand in ("L", "R"):
            g = [seq[hand][i + 1] - seq[hand][i] for i in range(len(seq[hand]) - 1)]
            g = [x for x in g if x > 0]
            if not g:
                ok = False
                break
            mu = sum(g) / len(g)
            if any(abs(x - mu) > 0.15 * mu for x in g):
                ok = False
                break
            info[hand] = mu
        if not ok:
            continue
        r = info["L"] / info["R"]
        if min(abs(r - round(r)), abs(1 / r - round(1 / r))) <= 0.15:
            continue
        out.append(_span(ctx, ss[0].index, ss[-1].index, "混fr", "045",
                         f"L 手间隔 {info['L']:.3f} 拍 / R 手间隔 {info['R']:.3f} 拍"
                         f"（比 {r:.3f}，非整数比）",
                         "两手各跑一条不同分音的节奏线",
                         {"gap_l": round(info["L"], 4), "gap_r": round(info["R"], 4),
                          "ratio": round(r, 4)}))
    return out


# ---------------------------------------------------------------------------
# 三、总入口 + 两个并存的分数
# ---------------------------------------------------------------------------

#: 本模块实现的配置 → 依据条目号
IMPLEMENTED: dict[str, str] = {
    "错位": "017", "普通交互": "018", "纵连": "019", "长纵连": "019",
    "三角交互": "020", "轴交互": "021", "楼梯交互": "022", "逆楼梯/方向盘": "023",
    "大宇宙": "024", "散点": "025", "跳拍": "026（时间层）", "定拍": "027",
    "子弹": "028", "单双/双单": "029", "连续双押": "030", "双押纵": "030",
    "侧边双押": "030",
    "死镰段": "033", "水神段": "034", "如龙段": "035", "二连扫": "036",
    "反手": "037", "2+1": "039", "3+1": "040", "N+1": "041", "出张": "043",
    "混fr": "045",
    "连续拍滑": "049", "双压连续拍滑": "050", "CYCLES型星星": "051",
    "夹键拍滑": "053", "穿心": "055", "大风车": "055", "绕圈星星": "056",
    "一笔画": "057", "挥手段": "058", "井字星星": "059", "同起点拍滑": "060",
    "三叉戟": "061", "鼓动段": "062", "拆弹": "063",
}

#: **未实现**的术语 → 原因（报告"未实现清单"直接引用）
NOT_IMPLEMENTED: dict[str, str] = {
    "菱形星星（052）": "ep3 §6.2 判为 C 档——教程原文只有「一个字：蹭」，形状判据无背书",
    "绕手交互（038）": "教程只给手感，阈值必须先跑 388 谱分布；ep2 §6 判 B 档且无数值",
    "狗刨（042）": "手法不是配置；且 ST 谱无 touch（知识 016），本轮语料恒为 0",
    "分页（044）": "手法（动态操作），不是可命中的谱面片段",
    "闪光弹/光污染（046）": "纯 touch 规则，ST 谱恒为 0（知识 016/046）",
    "换手（054）": "不是配置而是**自由度**；本模块按 §5.1 的退化口径统计"
                   "（hand(star)≠hand(track) 的根数），见 chart_summary 的 n_switch",
    "正/反三角（020 子型）": "知识 020 明说区分依据目前只有手感；"
                            "hand-sequencing.md §7-7 也说分配器看不出差别",
}


def detect_all(res: ParseResult, slots=None, ha: HandAssignment | None = None,
               params: dict | None = None) -> list[ConfigHitH]:
    """跑全部**手序版**检测器。

    签名与 `configs.detect_all(res, slots)` 兼容（第二个位置参数被忽略），
    因此可以直接注入 `config_profiles.bar_config_table(detector=...)` 与
    `chart_matrix.build_rows(detector=...)`。
    """
    ctx = make_ctx(res, ha, params)
    if not ctx.slots:
        return []
    runs = uniform_runs_h(ctx, min_len=4, single_only=True)
    units = slide_units(ctx)
    hits: list[ConfigHitH] = []
    # —— 17–30 ——
    hits += detect_vertical(ctx)
    hits += detect_axis(ctx, runs)
    hits += detect_triangle(ctx, runs)
    hits += detect_stair(ctx, runs)
    hits += detect_macrocosm(ctx, runs)
    hits += detect_interaction(ctx, runs)
    hits += detect_double_run(ctx)
    hits += detect_single_double(ctx)
    hits += detect_beatkeep(ctx)
    hits += detect_bullet(ctx)
    hits += detect_misalign(ctx)
    hits += detect_skipbeat(ctx)
    # —— 第二期 ——
    hits += detect_scythe(ctx, units)
    hits += detect_dragon(ctx)
    hits += detect_sweep(ctx, runs)
    hits += detect_backhand(ctx)
    hits += detect_chuzhang(ctx)
    hits += detect_nplus1(ctx)
    hits += detect_polyrhythm(ctx)
    # —— 星星篇 ——
    hits += detect_onestroke(ctx, units)
    hits += detect_tapslide(ctx, units)
    hits += detect_double_tapslide(ctx, units)
    hits += detect_same_head(ctx, units)
    hits += detect_pierce(ctx, units)
    hits += detect_circling(ctx, units)
    hits += detect_well(ctx, units)
    hits += detect_defuse(ctx, units)
    # —— 散点是"残差"，必须最后跑 ——
    hits += detect_scatter(ctx, runs, hits)
    hits = _merge_same_config(hits)
    hits = _drop_wave_inside_double_tapslide(hits)
    return sorted(hits, key=lambda h: (h.slot_start, h.config))


def _drop_wave_inside_double_tapslide(hits: list[ConfigHitH]) -> list[ConfigHitH]:
    """058 挥手段 与 050 双压连续拍滑 的分界（ep3 §3.2）。

    050 的两只手是**同相位地各守一个键放射**，所以每只手都满足"同头 ≥3 根"；
    但它不是 058（058 的另一只手跑的是**另一种**线）。凡被 050 片段覆盖的挥手段
    片段一律丢弃——同一格里有 2 个星星头就是 050，1 个才是 058/060 族。
    """
    dbl = [(h.slot_start, h.slot_end) for h in hits if h.config == "双压连续拍滑"]
    if not dbl:
        return hits
    return [h for h in hits
            if h.config != "挥手段"
            or not any(a <= h.slot_start and h.slot_end <= b for a, b in dbl)]


#: 这些配置的片段天然会被切碎（走向 / 走圈 / 放射各产出一条），合并重叠片段
_MERGE_CONFIGS = frozenset({"楼梯交互", "逆楼梯/方向盘", "大宇宙", "反手",
                            "二连扫", "井字星星", "出张", "CYCLES型星星"})


def _merge_same_config(hits: Sequence[ConfigHitH]) -> list[ConfigHitH]:
    out: list[ConfigHitH] = []
    byc: dict[str, list[ConfigHitH]] = {}
    for h in hits:
        (byc.setdefault(h.config, []) if h.config in _MERGE_CONFIGS
         else out).append(h) if h.config in _MERGE_CONFIGS else out.append(h)
    for cfg, arr in byc.items():
        arr = sorted(arr, key=lambda x: (x.slot_start, x.slot_end))
        cur = None
        for h in arr:
            if cur is None:
                cur = h
            elif h.slot_start <= cur.slot_end + 1:
                if h.slot_end > cur.slot_end:
                    cur.slot_end = h.slot_end
                    cur.bar_end = max(cur.bar_end, h.bar_end)
                    cur.t_end = max(cur.t_end, h.t_end)
                    cur.n_slots = cur.slot_end - cur.slot_start + 1
                    cur.detail.setdefault("merged", 0)
                    cur.detail["merged"] += 1
            else:
                out.append(cur)
                cur = h
        if cur is not None:
            out.append(cur)
    return _dedup(out)


# ---------- 逐小节两个分数 ----------


@dataclass
class BarScores:
    """逐小节两个**并存**的分数（§5.2：量的是两个不同的轴，**不合成**）。"""

    measure: int
    n_notes: int = 0
    n_tasks: int = 0
    nps: float = 0.0
    # 体力/精度（键位版口径，配置项换成手序版命中）
    term_kind: float = 0.0
    term_move: float = 0.0
    term_speed: float = 0.0
    term_muri: float = 0.0
    term_config: float = 0.0
    physical_raw: float = 0.0
    physical_hardness: float = 0.0      # 曲内 min-max
    # 手的调度余量（来自 hands.py，原样透传）
    hand_cost: float = 0.0
    hand_difficulty: float = 0.0        # = BarHand.hardness
    alt_rate: float = 0.0
    max_same_run: int = 0
    cross_count: int = 0
    chuzhang: int = 0
    scrape: int = 0
    muri: int = 0
    n_switch: int = 0                   # 知识 054 换手根数（退化统计）
    configs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"measure": self.measure, "n_notes": self.n_notes,
                "n_tasks": self.n_tasks, "nps": round(self.nps, 3),
                "term_kind": round(self.term_kind, 4),
                "term_move": round(self.term_move, 4),
                "term_speed": round(self.term_speed, 4),
                "term_muri": round(self.term_muri, 4),
                "term_config": round(self.term_config, 4),
                "physical_raw": round(self.physical_raw, 4),
                "physical_hardness": round(self.physical_hardness, 4),
                "hand_cost": round(self.hand_cost, 4),
                "hand_difficulty": round(self.hand_difficulty, 4),
                "alt_rate": round(self.alt_rate, 4),
                "max_same_run": self.max_same_run,
                "cross_count": self.cross_count, "chuzhang": self.chuzhang,
                "scrape": self.scrape, "muri": self.muri,
                "n_switch": self.n_switch,
                "configs": list(self.configs)}


def bar_scores(res: ParseResult, ha: HandAssignment | None = None,
               hits: Sequence[ConfigHitH] | None = None,
               params: dict | None = None) -> list[BarScores]:
    """逐小节输出 **两个并存的分数**（``physical_hardness`` 与 ``hand_difficulty``）。

    - ``physical_hardness``：`configs.bar_hardness` 的五项加权和口径
      ``0.22·K + 0.22·M + 0.26·V + 0.10·U + 0.20·C``，其中 C 取**手序版命中**的
      :data:`CONFIG_WEIGHT_PHYS` 最大值，U 取 `hands.py` 的绝对/硬无理数；
    - ``hand_difficulty``：直接取 `hands.BarHand.hardness`（代价/任务数的曲内归一）。

    ⚠️ **两者不合成**（`docs/hand-sequencing.md` §5.2；用户已撤回 `(D+H)/2` 排序）。
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    ctx = make_ctx(res, ha, p)
    ha = ctx.ha
    hits = list(detect_all(res, None, ha, p)) if hits is None else list(hits)
    w = p["hardness_weights"]
    by_bar_notes: dict[int, list[NoteEvent]] = {}
    for n in res.notes:
        by_bar_notes.setdefault(n.measure, []).append(n)
    by_bar_slots: dict[int, list[HSlot]] = {}
    for s in ctx.slots:
        by_bar_slots.setdefault(s.measure, []).append(s)
    by_bar_cfg: dict[int, set[str]] = {}
    for h in hits:
        for b in range(h.bar_start, h.bar_end + 1):
            by_bar_cfg.setdefault(b, set()).add(h.config)
    bh = {b.measure: b for b in ha.bars}
    n_switch: dict[int, int] = {}
    for u in slide_units(ctx):
        if u.switched:
            n_switch[u.measure] = n_switch.get(u.measure, 0) + 1
    by_bar_scrape: dict[int, int] = {}
    for m in ha.scrape:
        by_bar_scrape[m.measure] = by_bar_scrape.get(m.measure, 0) + 1

    if not by_bar_notes:
        return []
    lo, hi = min(by_bar_notes), max(by_bar_notes)
    out: list[BarScores] = []
    for bar in range(lo, hi + 1):
        ns = by_bar_notes.get(bar, [])
        ss = by_bar_slots.get(bar, [])
        cfgs = by_bar_cfg.get(bar, set())
        b = bh.get(bar)
        row = BarScores(measure=bar, n_notes=len(ns), n_tasks=b.n_tasks if b else 0,
                        configs=tuple(sorted(cfgs)))
        if b:
            row.hand_cost = b.hardness_raw
            row.hand_difficulty = b.hardness
            row.alt_rate = b.alt_rate
            row.max_same_run = b.max_same_run
            row.cross_count = b.cross_count
            row.chuzhang = b.chuzhang
            row.muri = sum(b.muri.values()) if b.muri else 0
        row.scrape = by_bar_scrape.get(bar, 0)
        row.n_switch = n_switch.get(bar, 0)
        if not ns:
            out.append(row)
            continue
        tot = 0.0
        for n in ns:
            v = KIND_WEIGHT.get(n.kind, 1.0)
            if n.is_break:
                v += BREAK_BONUS
            if n.is_each:
                v += EACH_BONUS
            tot += v
        K = min(max((tot / len(ns) - 1.0) / 0.6, 0.0), 1.0)
        if "错位" in cfgs:
            K = min(K + 0.25, 1.0)
        M = _move_term_hand(ctx, ss)
        bpm = ns[0].bpm or 0.0
        bar_sec = (240.0 / bpm) if bpm > 0 else 0.0
        nps = (len(ss) / bar_sec) if bar_sec > 0 else 0.0
        V = min(nps / p["nps_redline"], 1.0)
        U = min(row.muri / 3.0, 1.0)
        C = max((CONFIG_WEIGHT_PHYS.get(c, 0.0) for c in cfgs), default=0.0)
        row.nps = nps
        row.term_kind, row.term_move, row.term_speed = K, M, V
        row.term_muri, row.term_config = U, C
        row.physical_raw = (w["kind"] * K + w["move"] * M + w["speed"] * V
                            + w["muri"] * U + w["config"] * C)
        out.append(row)
    vals = [r.physical_raw for r in out if r.n_notes > 0]
    if vals:
        mn, mx = min(vals), max(vals)
        rng = (mx - mn) or 1.0
        for r in out:
            r.physical_hardness = (r.physical_raw - mn) / rng if r.n_notes else 0.0
    return out


def _move_term_hand(ctx: HandCtx, ss: Sequence[HSlot]) -> float:
    """位移项（知识 003-2）——**手级**：按每只手自己的相邻任务算，不再用"隔一个音"代理。"""
    if len(ss) < 2:
        return 0.0
    vals: list[float] = []
    for hand in ("L", "R"):
        ks = [s.hand_key(hand) for s in ss]
        ks = [k for k in ks if k is not None]
        for i in range(len(ks) - 1):
            vals.append(cdist(ks[i], ks[i + 1]))
    # 相邻槽之间的键位位移（与键位版同口径，保留可比性）
    adj: list[float] = []
    for i in range(len(ss) - 1):
        a, b = s_keys(ss[i]), s_keys(ss[i + 1])
        if a and b:
            adj.append(sum(cdist(x, y) for x in a for y in b) / (len(a) * len(b)))
    hand_mean = sum(vals) / len(vals) if vals else 0.0
    adj_mean = sum(adj) / len(adj) if adj else 0.0
    return min((0.5 * (hand_mean + adj_mean)) / 4.0, 1.0)


# ---------- 汇总 ----------


def summarize(hits: Sequence[ConfigHitH]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for h in hits:
        d = out.setdefault(h.config, {"n_segments": 0, "n_slots": 0, "n_tasks": 0,
                                      "source": h.source})
        d["n_segments"] += 1
        d["n_slots"] += h.n_slots
        d["n_tasks"] += h.n_tasks
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["n_segments"]))


def iter_bar_configs(hits: Sequence[ConfigHitH]) -> Iterator[tuple[int, set[str]]]:
    by_bar: dict[int, set[str]] = {}
    for h in hits:
        for b in range(h.bar_start, h.bar_end + 1):
            by_bar.setdefault(b, set()).add(h.config)
    for b in sorted(by_bar):
        yield b, by_bar[b]


def chart_summary(res: ParseResult, ha: HandAssignment | None = None,
                  params: dict | None = None) -> dict:
    """全谱一行汇总（CSV 口径）：两个分数 + 手法统计。"""
    ctx = make_ctx(res, ha, params)
    hits = detect_all(res, None, ctx.ha, params)
    rows = bar_scores(res, ctx.ha, hits, params)
    units = slide_units(ctx)
    live = [r for r in rows if r.n_notes > 0]
    return {
        "n_bars": len(live),
        "n_tasks": ctx.ha.n_tasks,
        "n_configs": len(summarize(hits)),
        "n_segments": len(hits),
        "physical_hardness_mean": round(
            sum(r.physical_raw for r in live) / len(live), 5) if live else 0.0,
        "hand_difficulty_mean": round(ctx.ha.hand_hardness, 5),
        "n_switch": sum(1 for u in units if u.switched),
        "n_chuzhang": sum(1 for u in units if u.chuzhang),
        "n_slides": len(units),
        "muri": len(ctx.ha.muri),
        "scrape": len(ctx.ha.scrape),
        "configs": summarize(hits),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    from . import corpus
    from .simai_parser import parse_chart

    ap = argparse.ArgumentParser(
        prog="python -m chart_analysis.configs_hand",
        description="配置识别（手序版）：从 hands.assign() 的左右手分配里认配置")
    ap.add_argument("chart", help="谱面文件路径或关键字")
    ap.add_argument("--bars", default=None, help="只看 a-b 小节")
    ap.add_argument("--config", default=None, help="只看某个配置")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    path = Path(a.chart)
    if path.exists():
        res = parse_chart(path.read_text(encoding="utf-8"), name=path.stem)
        name = path.stem
    else:
        cands = [c for c in corpus.discover() if a.chart in c.name]
        if not cands:
            print(f"没找到谱面：{a.chart}")
            return 2
        res = parse_chart(cands[0].read(), name=cands[0].name)
        name = cands[0].name
    hits = detect_all(res)
    lo, hi = (-1, 10 ** 9)
    if a.bars:
        lo, hi = (int(x) for x in a.bars.split("-"))
    print(f"# {name}  片段 {len(hits)}  配置 {len(summarize(hits))}")
    for h in hits:
        if h.bar_end < lo or h.bar_start > hi:
            continue
        if a.config and h.config != a.config:
            continue
        print(f"m{h.bar_start:03d}-{h.bar_end:03d} [{h.source}] {h.config}")
        print(f"    证据: {h.evidence}")
        if h.hands_pattern:
            print(f"    手: {h.hands_pattern}")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"chart": name, "hits": [h.to_dict() for h in hits],
             "summary": summarize(hits)}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"写出 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
