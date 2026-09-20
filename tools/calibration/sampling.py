#!/usr/bin/env python3
"""采音方式度量：官方谱在每个小节**怎么采**目标音轨发出的音。

问题
----
`stemhit.py` 回答的是"官方踩的是哪条轨"（recall / precision / lift）。
本模块回答的是**同一条轨上、采音的方式**。

**说法只用社区谱师实际在用的词**（2026-09-20 社区调研，来源见
`docs/research/sampling-terminology-survey.md`）：

- **全踩**：把这段音轨的音基本都踩了。
  出处：MMFC《谱面创作基础学》5.4「第一副歌中采用了写 vocal 曲最常见的做法——全踩人声」；
  社区里它是**动词短语**（全踩 + 音轨名），不是名词。
- **舍音**：只踩其中一部分，舍掉另一部分。
  出处：maimai 创作谱面自述「定下了舍采音的基调」（白明星琉姬）、
  Arcaea《谱面的个人心得 其一 采音和舍音》「舍弃掉一部分没有采的音节」。
- **留白**：音乐里有音，谱面故意不踩。
  出处：maimai 自制谱自述「这段不好采音干脆不采了，红谱里留白的例子也不少」；
  失误版社区叫**漏音 / 漏采**（「是对音错误，漏音（没写音乐中存在的音）」）。
- **似踩非踩**（**第四个社区说法，2026-09-20 补**）：谱面按匀速分音把小节铺满，
  而这些音并不对应音轨实际发出的音。
  出处：MMFC《谱面创作基础学》**5.3 踩音与配置**——
  「海底谭的副歌采用了一种似踩非踩的写法，全程铺满 8 分音符」。
  本地形态锚点：`resource/official-chart/336-ウミユリ海底譚-mas.txt` m030–m031。

  ⚠️ **它不是第四个互斥档位**。在本工具的三词口径里它**藏在「全踩」里**——
  铺满会把音轨那点音全都顺手盖住（所以 ``coverage`` 反而高），露馅的是 ``extra``。
  指纹 = **匀速铺满 + coverage 高 + extra 高**，诊断字段 :attr:`BarSampling.pseudo_sample`。
  不给它切 ``extra_ratio`` 的档：社区没有把这四个词当一组互斥分类用，
  而"铺满得多满算铺满"一旦定阈值就又回到用户 2026-09-20 批评过的那条路上去了。
  官谱用法（160 首实测，见 `docs/research/config-gaps-survey.md` §4）：84 首用过，
  多数是**一两小节的过渡填充**；段落上偏好落ちサビ / Bメロ / イントロ / 間奏，
  **最强的 ラスサビ 与 ドロップ 最不用**；与段落强度无关；8 / 12 / 16 分都写。

判不出来的小节（这一小节音轨本来就没发出几个音）落 :data:`UNKNOWN`（``"—"``），
它**不是一种采音方式**，只表示"没有依据、不下结论"。

⚠️ **「空音」这个词已经停用，因为它在社区里是反义词。**
萌娘百科《音乐游戏/用语》的 **采空音** 词条：「原本音乐中并没有音的地方却在谱面的
对应位置塞入了音符」——即谱面**多**写了音；用户口中的「空音」是谱面**少**写了音。
本模块原来叫 ``空音`` 的那一档改叫 ``留白``；而"谱面写了哪条轨都没有的音"这件事
（诊断字段 ``extra`` / ``extra_ratio``）才是社区的**采空音 / 插空音**（4K 圈叫 dump / 塞）。

**2026-09-20 术语校正（用户原话）**：
「什么是空音？什么是半采音？什么是采全音？**这些词是我编的，你自己去社区去论坛搜索
自己对应去。**」——于是 ``采全音 / 半采音 / 空音`` 三个词整体换成上面的社区用词。

> ⚠️ 诚实边界：社区**没有**把「全踩 / 舍音 / 留白」当成一组互斥的三分类在用，
> 它们只是谱师描述采音疏密时各自会说的话。把连续的 coverage 切成三档**是本工具的
> 内部口径**（:data:`THRESHOLDS`），不是社区术语定义，也**不拿去问用户**
> （"踩七成算不算全踩"这种问题不该抛给谱师）。
> ``coverage`` / ``extra_ratio`` / ``accent`` / ``pseudo_sample`` 等保留为**诊断字段**，
> CLI 默认不打印。

**2026-09-20 之前的一轮修正（用户五条批评之一）**——原话：
「什么叫采七成算不算全采？根本就没有这些词吧，这些词哪来的臆造出来的吗？
怎么能量化这些事情呢？」

本模块此前对外输出过八类：``全采 / 近全采 / 半采 / 随机半采 / 稀采·空音 / 加花 /
静默 / 混合``，**全是 agent 臆造的类名**，已从对外输出、报告结论与知识草案里整体移除。

**2026-09-19 修订（用户原话：「采音肯定是根据音乐来踩的啊，不能光根据 bpm 来。」）**

旧版判"只踩一部分"用的是三条判据：①被踩中的 pool 下标构成等差为 2 的序列（"隔一个"）；
②官方槽全落在 0.5 拍整数倍上（"只踩强拍"）；③谱面最细分音 = 音频量化分音的一半
（"分音减半"）。**②③ 是网格/分音/BPM 判据，与"音乐里哪个音更重"无关，已删除**；
①（下标等差 2）也只是网格现象，同样删除。整条采音判定链上不再有任何分音 / BPM / 网格量。

"挑重音"（被踩中的音平均 onset 强度更高）现在只是 ``舍音`` 小节的一个**诊断标记**
（``accent`` / ``accent_reason``），不再把"挑重音的半采"和"不挑重音的半采"分成两类。
（社区对"以采明显的音为主、采不明显的音为少数"另有一个词叫 **主高**，
Arcaea 圈用语，见调研报告 §2；本模块只把它当诊断，不当类名。）

口径
----
- **时间基准**：官方 note 时间 = ``&first + note.time + φ``（φ = 全局相位补偿，
  与 `tools/calibration/cli.py` 的 `best_shift` 同口径，|φ*| ≤ 10 ms 时不补偿）；
- **官方时间槽**：同刻的 each / 双押算**一个采音事件**（`stemhit.unique_times`）；
- **pool（音乐事件池）**：目标轨在该小节的 onset，按 ``2τ = 60 ms`` 合并去重
  （两个挨得比容差还近的 onset，谱面上只可能对应一个音）；每个事件带一个
  **onset 强度**（`onsets.detect_onsets` 的 `strengths`，曲内已按最大值归一）；
- **hit**：pool 中被官方时间槽 ±τ 覆盖的个数；
- **extra**：官方时间槽中，**任何一条 stem** 都覆盖不到的个数——社区叫
  **采空音 / 插空音**（4K 圈：dump / 塞），这里含装饰音与 onset 漏检，三者无法区分。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import stemhit

#: 参与 "extra" 判定的候选轨（与 n=160 标定的候选池口径一致：四路 + piano）
POOL_STEMS = ("drums", "bass", "other", "vocals", "piano")
#: 可以当骨架的轨
SKELETON_STEMS = ("drums", "bass", "other", "vocals", "piano")
#: 旋律类轨（`skeleton_mode="melodic"` 口径：三者取 lift 最高）
MELODIC_STEMS = ("other", "piano", "vocals")

#: ⚠️ **工具内部的诊断口径，不是术语定义**（用户 2026-09-20）。
#: 把连续的 coverage 落到"全踩 / 舍音 / 留白"三个词上总得有个切点，这些数字就是
#: 那个切点——它们**不向用户索取、不作为待裁定项、不写进知识库**。
THRESHOLDS: dict = {
    "full_lo": 0.65,            # 全踩：coverage > 此值（"基本都踩了"）
    "empty_hi": 0.35,           # 留白：coverage < 此值（"音乐有音而谱面不踩"）
    #                             两者之间 = 舍音（"只踩一部分，舍掉另一部分"）
    "empty_min_pool": 4,        # 留白还要求 pool ≥（音乐真的在响，否则是没音可踩）
    "min_pool": 2,              # pool 少于这么多就不下结论（落 UNKNOWN，"没音可踩"）
    "merge_sec": 0.060,         # pool 去重 granularity（= 2τ）
    # ↓ 只影响 `accent` 这个**诊断标记**，不影响三个词的归类
    "accent_margin": 0.0,       # 挑重音：均强差 > 此值才算"踩的是重音"（打平不算）
    "accent_bins": 3,           # 强度分位桶数（"命中率随强度单调上升"判据）
    "accent_min_side": 2,       # 两侧各至少这么多个有强度的事件才判挑重音
    # ↓ 只影响 `pseudo_sample`（似踩非踩）这个**诊断标记**，**不新增档位**、不影响三个词。
    #   三个数就是 `docs/research/config-gaps-survey.md` §4.2 把候选捞出来用的口径，
    #   照搬过来是为了让报告与工具对得上；它们**不是术语定义、不是判据门槛**。
    "even_tol": 0.02,           # 匀速：相邻间隔的离散系数（std/mean）小于此值
    "even_min_slots": 8,        # 铺满：一小节至少这么多个官方时间槽
    "pseudo_extra_lo": 1.0 / 3, # 似踩非踩：`extra_ratio` 不低于此值（"加进来的格子"够多）
}

#: 判不出来的小节（音轨这一小节本来就没发出几个音）。**不是一种采音方式**。
UNKNOWN = "—"

#: 对外只有这三个社区用词（2026-09-20 调研；原 ``采全音 / 半采音 / 空音`` 是用户
#: 自述"我编的"，已停用。映射：采全音→全踩、半采音→舍音、空音→留白）。
MODE_ORDER = ("全踩", "舍音", "留白")

#: 已停用的旧词 → 现用社区词。只给读旧报告 / 旧 JSON 的人对照用，
#: **不在任何对外输出里出现**。
LEGACY_MODE_ALIASES: dict = {"采全音": "全踩", "半采音": "舍音", "空音": "留白"}


# ---------------------------------------------------------------------------
# 「挑重音」判据（替代旧的网格/分音 alt_pattern）
# ---------------------------------------------------------------------------


def accent_pick(pool: Sequence[float], strengths: Sequence[float],
                slot_times: Sequence[float],
                tol: float = stemhit.DEFAULT_TOL_SEC,
                th: dict | None = None) -> tuple[bool, str, float]:
    """判断"被踩的是更响/更重的那一半"。

    两条判据任一成立即算（各自写进 ``accent_reason``）：

    1. **重音优先**：命中事件的平均 onset 强度 − 未命中事件的平均 onset 强度
       **严格大于** ``accent_margin``（默认 0 ⇒ "踩中的平均更响"；强度完全打平
       的小节**不算**挑重音）；
    2. **分位单调**：把 pool 按 onset 强度切成 ``accent_bins`` 个等频桶，
       各桶命中率**不下降**且最高桶严格高于最低桶。

    强度缺失（NaN）的事件两条判据都不参与；两侧有效事件各少于
    ``accent_min_side`` 个时返回 ``(False, "证据不足", nan)``。

    返回 ``(是否挑重音, 理由, 均强差)``。
    """
    th = THRESHOLDS if th is None else th
    p = np.atleast_1d(np.asarray(pool, dtype=float))
    s = np.atleast_1d(np.asarray(strengths, dtype=float))
    ev = np.atleast_1d(np.asarray(slot_times, dtype=float))
    if p.size == 0 or s.size != p.size:
        return False, "无强度", float("nan")
    ok = np.isfinite(s)
    if not np.any(ok):
        return False, "无强度", float("nan")
    hit = stemhit.covered_mask(p, ev, tol)
    hs, ms = s[ok & hit], s[ok & ~hit]
    if hs.size < th["accent_min_side"] or ms.size < th["accent_min_side"]:
        return False, "证据不足", float("nan")
    delta = float(np.mean(hs) - np.mean(ms))
    reasons: list[str] = []
    if delta > th["accent_margin"]:
        reasons.append("重音优先")
    nb = int(th["accent_bins"])
    sv, hv = s[ok], hit[ok]
    if sv.size >= nb * th["accent_min_side"]:
        order = np.argsort(sv, kind="stable")
        bins = np.array_split(order, nb)
        rates = [float(np.mean(hv[b])) if len(b) else float("nan") for b in bins]
        if all(np.isfinite(r) for r in rates):
            if all(rates[i] <= rates[i + 1] + 1e-12 for i in range(nb - 1)) \
                    and rates[-1] > rates[0] + 1e-12:
                reasons.append("分位单调")
    return bool(reasons), "+".join(reasons), delta


# ---------------------------------------------------------------------------
# 逐小节记录
# ---------------------------------------------------------------------------


@dataclass
class BarSampling:
    """一小节的采音度量。"""

    bar: int                       # 音频小节号（1 起）
    chart_measure: int             # 谱面小节号（0 起）
    t0: float
    t1: float
    skeleton: str = "drums"
    n_pool: int = 0
    n_slots: int = 0               # 官方时间槽数（同刻算一个）
    n_notes: int = 0               # 官方 note 数
    hit: int = 0
    extra: int = 0
    coverage: float = float("nan")
    extra_ratio: float = float("nan")
    accent: bool = False
    accent_reason: str = ""
    accent_delta: float = float("nan")
    #: **似踩非踩**（MMFC 5.3）的诊断标记：匀速铺满 ∧ coverage 高 ∧ extra 高。
    #: **不是第四个档位**——这些小节的 :attr:`mode` 绝大多数仍是「全踩」。
    pseudo_sample: bool = False
    rest_beats: float = 0.0
    density_ratio: float = float("nan")   # 官方槽数 / pool 大小
    mode: str = UNKNOWN
    per_stem_pool: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        def r(v, n=4):
            return None if not np.isfinite(v) else round(float(v), n)
        return {"bar": self.bar, "chart_measure": self.chart_measure,
                "skeleton": self.skeleton, "n_pool": self.n_pool,
                "n_slots": self.n_slots, "n_notes": self.n_notes,
                "hit": self.hit, "extra": self.extra,
                "coverage": r(self.coverage), "extra_ratio": r(self.extra_ratio),
                "accent": self.accent, "accent_reason": self.accent_reason,
                "accent_delta": r(self.accent_delta),
                "pseudo_sample": self.pseudo_sample,
                "rest_beats": r(self.rest_beats, 3),
                "density_ratio": r(self.density_ratio), "mode": self.mode}


def classify(coverage: float, extra_ratio: float, n_slots: int,
             n_pool: int, accent: bool, th: dict | None = None) -> str:
    """把一小节的度量落成**社区的三个词**之一（或 :data:`UNKNOWN`）。

    - ``全踩``：``coverage > full_lo``——把这段音轨的音基本都踩了；
    - ``留白``：``coverage < empty_hi`` 且 ``n_pool ≥ empty_min_pool``——
      音轨在响，谱面却不踩；
    - ``舍音``：两者之间——只踩一部分，舍掉另一部分；
    - ``—``（:data:`UNKNOWN`）：音轨这一小节发出的音太少（``n_pool < min_pool``），
      **没有依据、不下结论**。谱面这一小节几乎不写音、而音轨在响的情形**不是**
      判不出——它 coverage ≈ 0，就是 ``留白``。

    ⚠️ 切点在 :data:`THRESHOLDS` 里，是**工具内部的诊断口径**，不是术语定义
    （用户 2026-09-20：「根本就没有这些词吧……怎么能量化这些事情呢？」）。
    社区也**不把这三个词当互斥分类用**——三分类这件事本身是本工具的操作化。
    ``extra_ratio``（谱面写了哪条轨都没有的音，社区叫**采空音 / 插空音**）与
    ``accent``（踩的是不是重音，社区叫**主高**）只作诊断字段，
    **不再各自撑起一个类名**（原来的"加花""静默""随机半采"等 agent 自造类名已整体删除）。

    ⚠️ **「全踩」这一档里混着一部分其实是「似踩非踩」**（MMFC 5.3 的第四个社区说法）：
    谱面按匀速分音把小节铺满，铺满把音轨的音全盖住了，所以 ``coverage`` 高、
    落进「全踩」；要把它认出来得看 ``extra_ratio``（谱面格子里对不上任何 stem 的比例），
    诊断标记是 ``pseudo_sample``。160 首实测：314 个候选里 **266 个被本函数判成「全踩」**
    （另有 30「舍音」、14「—」、4「留白」）。**本函数不为它新增档位**——
    社区没有把这四个词当一组互斥分类用。
    """
    th = THRESHOLDS if th is None else th
    if n_pool < th["min_pool"] or not np.isfinite(coverage):
        return UNKNOWN
    if coverage > th["full_lo"]:
        return "全踩"
    if coverage < th["empty_hi"]:
        return "留白" if n_pool >= th["empty_min_pool"] else UNKNOWN
    return "舍音"


def pseudo_sample_flag(slot_times: Sequence[float], coverage: float,
                       extra_ratio: float, th: dict | None = None) -> bool:
    """**似踩非踩**（MMFC 5.3）的诊断标记——**布尔诊断，不是第四个档位**。

    三件事同时成立才置位（口径照搬 `docs/research/config-gaps-survey.md` §4.2
    把候选捞出来用的那一套，**不是判据门槛、不是术语定义**）：

    1. **匀速铺满**：官方时间槽 ≥ ``even_min_slots`` 个，且相邻间隔的离散系数
       （std/mean）< ``even_tol``——整小节一个格子不空地匀速排着；
    2. **coverage 高**：``coverage > full_lo``（铺满顺手把音轨那点音全盖住了，
       所以它落在「全踩」那一档里）；
    3. **extra 高**：``extra_ratio ≥ pseudo_extra_lo``（铺出来的格子里有一大批
       **任何一条 stem 都对不上**——社区叫采空音 / 插空音）。

    ⚠️ 第 3 条的 ``extra`` 是**代理**：它含谱师自由发挥、装饰音、**以及 onset 漏检**，
    三者分不开（知识 032 的老问题）。所以这只是"请人看一眼"的标记，
    **不能当成"这一小节确实是 MMFC 说的那回事"的判决**。

    ⚠️ ``extra`` 高**不等于谱写坏了**——成段的高 ``extra`` 恰恰是似踩非踩这种正规写法。
    """
    th = THRESHOLDS if th is None else th
    ev = np.sort(np.atleast_1d(np.asarray(slot_times, dtype=float)))
    if ev.size < th["even_min_slots"]:
        return False
    d = np.diff(ev)
    if d.size == 0 or not np.all(np.isfinite(d)) or d.mean() <= 0:
        return False
    if float(d.std() / d.mean()) >= th["even_tol"]:
        return False
    if not np.isfinite(coverage) or coverage <= th["full_lo"]:
        return False
    return bool(np.isfinite(extra_ratio) and extra_ratio >= th["pseudo_extra_lo"])


# ---------------------------------------------------------------------------
# 单小节度量
# ---------------------------------------------------------------------------


def bar_metrics(slot_times: Sequence[float], note_count: int,
                skeleton_pool: Sequence[float], any_onsets: Sequence[float],
                t0: float, t1: float, sec_per_beat: float,
                pool_strengths: Sequence[float] | None = None,
                tol: float = stemhit.DEFAULT_TOL_SEC,
                th: dict | None = None) -> dict:
    """一小节的全部采音度量（纯函数，方便单测）。

    ``sec_per_beat`` 只用来把 ``rest_beats`` 换算成拍（单位换算，不参与任何判据）。
    """
    th = THRESHOLDS if th is None else th
    ev = np.atleast_1d(np.asarray(slot_times, dtype=float))
    pool = np.atleast_1d(np.asarray(skeleton_pool, dtype=float))
    allon = np.atleast_1d(np.asarray(any_onsets, dtype=float))
    strg = (np.full(pool.size, np.nan) if pool_strengths is None
            else np.atleast_1d(np.asarray(pool_strengths, dtype=float)))
    n_slots = int(ev.size)
    n_pool = int(pool.size)
    hit = int(np.sum(stemhit.covered_mask(pool, ev, tol))) if n_pool else 0
    extra = int(np.sum(~stemhit.covered_mask(ev, allon, tol))) if n_slots else 0
    coverage = hit / n_pool if n_pool else float("nan")
    extra_ratio = extra / n_slots if n_slots else float("nan")
    acc, reason, delta = accent_pick(pool, strg, ev, tol, th)
    # 最长留白：相邻官方槽之间（含小节两端）的最大间隔，且该区间内 pool 非空
    rest = 0.0
    edges = np.concatenate([[t0], np.sort(ev), [t1]])
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 0:
            continue
        if np.any((pool > a + tol) & (pool < b - tol)):
            rest = max(rest, (b - a) / sec_per_beat if sec_per_beat > 0 else 0.0)
    mode = classify(coverage, extra_ratio, n_slots, n_pool, acc, th)
    return {"n_slots": n_slots, "n_notes": int(note_count), "n_pool": n_pool,
            "hit": hit, "extra": extra, "coverage": coverage,
            "extra_ratio": extra_ratio, "accent": acc, "accent_reason": reason,
            "accent_delta": delta, "rest_beats": rest, "mode": mode,
            # 似踩非踩（MMFC 5.3）：**诊断标记**，与 `mode` 并行、不改 `mode`
            "pseudo_sample": pseudo_sample_flag(ev, coverage, extra_ratio, th),
            "density_ratio": (n_slots / n_pool) if n_pool else float("nan")}


# ---------------------------------------------------------------------------
# 整曲
# ---------------------------------------------------------------------------


def pick_skeleton(ev: np.ndarray, stem_onsets: dict, span: float,
                  tol: float = stemhit.DEFAULT_TOL_SEC,
                  min_events: int = 3, min_onsets: int = 3,
                  default: str = "drums", margin: float = 1.25) -> str:
    """按 `lift` 最高选目标音轨（与标定报告 §5 的归因口径一致）。

    两道稳态闸门（**没有这两道，逐小节 argmax 会一节一换、纯噪声**）：

    1. 证据不足（官方槽 < ``min_events``，或该轨 onset < ``min_onsets``）→ 回落 ``default``；
    2. 只有当最高 lift **超过 ``default`` 的 lift ``margin`` 倍**才换轨，否则留在
       ``default``（``default`` 一般是该段的段落骨架）。
    """
    if ev.size < min_events:
        return default
    lifts: dict[str, float] = {}
    for k, v in stem_onsets.items():
        on = np.atleast_1d(np.asarray(v, dtype=float))
        if on.size < min_onsets:
            continue
        lf = stemhit.hit_stat(ev, on, tol, span_sec=span).lift
        if np.isfinite(lf):
            lifts[k] = float(lf)
    if not lifts:
        return default
    best = max(lifts, key=lambda k: lifts[k])
    base = lifts.get(default, 0.0)
    if best != default and lifts[best] < margin * max(base, 1e-9):
        return default
    return best


def song_sampling(bars: Sequence[tuple[int, int, float, float, float]],
                  slot_times: Sequence[float], notes_per_bar: dict,
                  stem_onsets: dict, skeleton_mode: str = "bar",
                  segment_skeleton: dict | None = None,
                  stem_strengths: dict | None = None,
                  melodic_default: dict | str = "other",
                  tol: float = stemhit.DEFAULT_TOL_SEC,
                  th: dict | None = None) -> list[BarSampling]:
    """逐小节跑采音度量。

    参数
    ----
    bars
        ``[(音频小节号, 谱面小节号, t0, t1, 每拍秒数), ...]``
    slot_times
        全曲官方时间槽（已做 φ 补偿、同刻已去重），绝对秒。
    notes_per_bar
        ``{谱面小节号: note 数}``。
    stem_onsets
        ``{轨名: np.ndarray(秒)}``。
    stem_strengths
        ``{轨名: np.ndarray}``，与 ``stem_onsets`` 等长的 onset 强度；缺了就按
        NaN 处理（该小节判不出"挑重音"，`accent` 这个诊断标记为 False，
        **不影响**落哪个词）。
    skeleton_mode
        - ``bar``：逐小节在五条轨里按 lift 取（证据不足回落段落/drums）；
        - ``melodic``：逐小节只在 ``MELODIC_STEMS`` 里按 lift 取；
        - ``segment``：用 ``segment_skeleton`` 给的段落骨架；
        - 其余值 = 固定用该轨（如 ``drums``）。
    """
    th = THRESHOLDS if th is None else th
    strengths = stem_strengths or {}
    ev_all = np.atleast_1d(np.asarray(slot_times, dtype=float))
    allon = np.sort(np.concatenate(
        [np.atleast_1d(np.asarray(stem_onsets.get(s, np.zeros(0)), dtype=float))
         for s in POOL_STEMS])) if stem_onsets else np.zeros(0)
    out: list[BarSampling] = []
    for bar, measure, t0, t1, spb in bars:
        ev = ev_all[(ev_all >= t0) & (ev_all < t1)]
        seg_sk = (segment_skeleton or {}).get(bar, "drums")
        if skeleton_mode in ("bar", "melodic"):
            cands = SKELETON_STEMS if skeleton_mode == "bar" else MELODIC_STEMS
            dflt = seg_sk if skeleton_mode == "bar" else (
                melodic_default.get(bar, "other")
                if isinstance(melodic_default, dict) else melodic_default)
            per = {k: np.asarray(v, dtype=float) for k, v in stem_onsets.items()
                   if k in cands}
            per = {k: v[(v >= t0) & (v < t1)] for k, v in per.items()}
            sk = pick_skeleton(ev, per, max(t1 - t0, 1e-6), tol, default=dflt)
        elif skeleton_mode == "segment":
            sk = seg_sk
        else:
            sk = skeleton_mode
        raw = np.atleast_1d(np.asarray(stem_onsets.get(sk, np.zeros(0)), dtype=float))
        rst = np.atleast_1d(np.asarray(strengths.get(sk, np.zeros(0)), dtype=float))
        if rst.size != raw.size:
            rst = np.full(raw.size, np.nan)
        sel = (raw >= t0) & (raw < t1)
        pool, pstr = _unique_with_strength(raw[sel], rst[sel], th["merge_sec"])
        m = bar_metrics(ev, notes_per_bar.get(measure, 0), pool,
                        allon[(allon >= t0 - 0.2) & (allon < t1 + 0.2)],
                        t0, t1, spb, pstr, tol, th)
        bs = BarSampling(bar=bar, chart_measure=measure, t0=t0, t1=t1, skeleton=sk,
                         **{k: m[k] for k in ("n_pool", "n_slots", "n_notes", "hit",
                                              "extra", "coverage", "extra_ratio",
                                              "accent", "accent_reason",
                                              "accent_delta", "rest_beats",
                                              "density_ratio", "mode",
                                              "pseudo_sample")})
        out.append(bs)
    return out


def _unique_with_strength(times, strengths, merge_sec: float
                          ) -> tuple[np.ndarray, np.ndarray]:
    """按 ``merge_sec`` 合并去重，合并簇取**最大强度**那个事件的时间与强度。

    取最大而不是取第一个：一簇 60 ms 内的 onset 在谱面上只能对应一个音，
    谱师听到的是这一簇里最响的那下。
    """
    t = np.atleast_1d(np.asarray(times, dtype=float))
    s = np.atleast_1d(np.asarray(strengths, dtype=float))
    if t.size == 0:
        return np.zeros(0), np.zeros(0)
    if s.size != t.size:
        s = np.full(t.size, np.nan)
    order = np.argsort(t, kind="stable")
    t, s = t[order], s[order]
    keep_t: list[float] = [float(t[0])]
    keep_s: list[float] = [float(s[0])]
    for x, v in zip(t[1:], s[1:]):
        if float(x) - keep_t[-1] > merge_sec:
            keep_t.append(float(x))
            keep_s.append(float(v))
        elif np.isfinite(v) and (not np.isfinite(keep_s[-1]) or v > keep_s[-1]):
            keep_t[-1] = float(x)
            keep_s[-1] = float(v)
    return np.asarray(keep_t, dtype=float), np.asarray(keep_s, dtype=float)


def shuffled_control(bars: Sequence[tuple[int, int, float, float, float]],
                     slot_times: Sequence[float], notes_per_bar: dict,
                     stem_onsets: dict, seed: int = 0,
                     stem_strengths: dict | None = None, **kw) -> list[BarSampling]:
    """对照：把**每条 stem 的 onset 整体循环平移**一个随机量后重跑。

    平移量取 ``[0.1·曲长, 0.9·曲长]`` 上的均匀分布（各轨独立），落在曲长上循环回绕
    （按比例取而不是固定秒数，短曲/长曲都不会退化成常数平移）。
    这样 onset 的**密度、节奏纹理与强度分布完全保留**（强度跟着 onset 一起搬），
    只有"和谱面的对齐关系"被打断——如果三个词的分布（尤其是舍音里"挑重音"的比例）
    在对照上也一样，说明它只是密度的伪影。
    """
    rng = np.random.default_rng(seed)
    t_end = max(b[3] for b in bars) if bars else 0.0
    fake: dict = {}
    fake_s: dict = {}
    src_s = stem_strengths or {}
    for k, v in stem_onsets.items():
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        st = np.atleast_1d(np.asarray(src_s.get(k, np.zeros(0)), dtype=float))
        if st.size != arr.size:
            st = np.full(arr.size, np.nan)
        if arr.size == 0 or t_end <= 0.0:
            fake[k], fake_s[k] = arr, st
            continue
        shift = float(rng.uniform(0.1 * t_end, 0.9 * t_end))
        moved = (arr + shift) % t_end
        order = np.argsort(moved, kind="stable")
        fake[k] = moved[order]
        fake_s[k] = st[order]
    return song_sampling(bars, slot_times, notes_per_bar, fake,
                         stem_strengths=fake_s, **kw)


# ---------------------------------------------------------------------------
# 装载（轻量路径：不解码混音、不重算强度）
# ---------------------------------------------------------------------------


SHIFT_SCAN = np.arange(-0.045, 0.04501, 0.0025)
APPLY_SHIFT_THRESHOLD = 0.010


def load_pair(song_dir: Path, onset_cache: Path | None = None,
              tol: float = stemhit.DEFAULT_TOL_SEC,
              pitch_conf: float | None = None) -> dict:
    """装载一首曲子的"谱面 × 音频"配对，只取采音分析需要的东西。

    与 `loader.load_song` 的差别：**不解码 track.mp3、不重算强度曲线**
    （这两件事占了装载时间的九成以上，而采音分析用不到）。
    onset 与 onset 强度从 ``onset_cache/<曲名>.npz`` 读（键 ``<轨>`` 与 ``<轨>__str``）；
    没有缓存就现场从 `stems/*.wav` 跑（参数与管线完全一致）。

    ``pitch_conf`` 不为 None 时，把 ``stems_htdemucs_6s/pitch_notes.json`` 里
    **置信度 ≥ pitch_conf** 的有音高 note onset 并进对应 melodic 轨的池
    （⚠️ 默认关闭：`docs/research/stem-refinement-n40.md` §13 实测 confidence
    不是好的筛选量，melody 轨整体是候选池灌水；这里只留作敏感性开关）。
    """
    import sys

    repo = Path(__file__).resolve().parents[2]
    for p in (str(repo), str(repo / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from chart_analysis import configs as cfg_mod
    from chart_analysis.density import chart_density
    from chart_analysis.simai_parser import parse_chart

    from . import chartpair as cp
    from . import loader as loader_mod

    song_dir = Path(song_dir)
    meta = loader_mod.read_maidata_header(song_dir / "maidata.txt")
    analysis = json.loads((song_dir / "song_analysis.json").read_text(encoding="utf-8"))
    res = parse_chart(loader_mod.extract_inote(song_dir / "maidata.txt"),
                      name=song_dir.name)
    dens = chart_density(res, song_dir.name)

    # ---- onset + 强度 ----
    onsets: dict[str, np.ndarray] = {}
    strengths: dict[str, np.ndarray] = {}
    npz = (Path(onset_cache) / f"{song_dir.name}.npz") if onset_cache else None
    if npz is not None and npz.exists():
        with np.load(npz) as z:
            for k in z.files:
                if k.endswith("__str"):
                    strengths[k[:-5]] = np.asarray(z[k], dtype=float)
                else:
                    onsets[k] = np.asarray(z[k], dtype=float)
    else:
        from tools.audio_analysis import onsets as onsets_mod
        from tools.audio_analysis import stems as stems_mod
        todo = [(s, song_dir / "stems" / f"{s}.wav")
                for s in ("drums", "bass", "other", "vocals")]
        todo.append(("piano", song_dir / "stems_htdemucs_6s" / "piano.wav"))
        for s, p in todo:
            if not p.exists():
                continue
            y, _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
            tr = onsets_mod.detect_onsets(y, s, sr=onsets_mod.ANALYSIS_SR,
                                          hop=onsets_mod.HOP)
            onsets[s], strengths[s] = tr.times, tr.strengths

    if pitch_conf is not None:
        _merge_pitch_notes(song_dir, onsets, strengths, float(pitch_conf),
                           THRESHOLDS["merge_sec"])

    # ---- 小节对齐（与 cli.analyze_song 同口径）----
    g = analysis["grid"]
    first = float(g["first"])
    stats = dens.measures
    measure_starts = [first + s.start_time for s in stats]
    bar_starts = _bar_starts(analysis)
    m2b = cp.map_measures_to_bars(measure_starts, bar_starts, tol_sec=0.05)

    note_times = np.array([first + n.time for n in res.notes], dtype=float)
    slot_times = stemhit.unique_times(note_times, merge_sec=1e-3)
    allon = np.sort(np.concatenate(
        [np.atleast_1d(onsets.get(s, np.zeros(0))) for s in POOL_STEMS])) \
        if onsets else np.zeros(0)
    shift = stemhit.best_shift(slot_times, allon, SHIFT_SCAN, tol=tol)
    phi = float(shift["best_shift_sec"])
    applied = phi if abs(phi) > APPLY_SHIFT_THRESHOLD else 0.0

    bars: list[tuple[int, int, float, float, float]] = []
    for i in sorted(m2b):
        bar = m2b[i]
        bpm = float(stats[i].bpm) or float(g["bpm"])
        spb = 60.0 / bpm
        t0 = measure_starts[i] + applied
        bars.append((bar, stats[i].measure, t0, t0 + 4.0 * spb, spb))

    notes_per_bar = {s.measure: s.notes for s in stats}
    seg_of_bar, seg_rows = _segments_by_bar(analysis)

    # 段落骨架：段内按 lift 最高的轨（证据比逐小节稳）；melodic 口径另算一份
    seg_sk: dict[int, str] = {}
    seg_mel: dict[int, str] = {}
    for row in seg_rows:
        a, b = row["start_bar"], row["end_bar"]
        t_a = next((x[2] for x in bars if x[0] >= a), None)
        t_b = next((x[3] for x in reversed(bars) if x[0] <= b), None)
        if t_a is None or t_b is None or t_b <= t_a:
            continue
        ev = slot_times + applied
        ev = ev[(ev >= t_a) & (ev < t_b)]
        per = {k: v[(v >= t_a) & (v < t_b)] for k, v in onsets.items()
               if k in SKELETON_STEMS}
        sk = pick_skeleton(ev, per, t_b - t_a, tol, default="drums")
        mel = {k: v for k, v in per.items() if k in MELODIC_STEMS}
        mk = pick_skeleton(ev, mel, t_b - t_a, tol, default="other", margin=1.0)
        for bb in range(a, b + 1):
            seg_sk[bb] = sk
            seg_mel[bb] = mk

    return {
        "name": song_dir.name,
        "level": float(meta.get("lv_5", 0.0) or 0.0),
        "genre": str(meta.get("genre", "") or ""),
        "bpm": float(g["bpm"]), "first": first,
        "analysis": analysis, "parse": res, "density": dens,
        "onsets": onsets, "strengths": strengths, "bars": bars,
        "slot_times": slot_times + applied,
        "notes_per_bar": notes_per_bar,
        "seg_of_bar": seg_of_bar, "segments": seg_rows,
        "segment_skeleton": seg_sk, "melodic_default": seg_mel,
        "phi_ms": round(phi * 1000.0, 2), "phi_applied_ms": round(applied * 1000.0, 2),
        "configs": cfg_mod,
    }


def _merge_pitch_notes(song_dir: Path, onsets: dict, strengths: dict,
                       conf: float, merge_sec: float) -> None:
    """把高置信度的 basic-pitch note onset 并进对应 melodic 轨（原地改）。"""
    p = song_dir / "stems_htdemucs_6s" / "pitch_notes.json"
    if not p.exists():
        return
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                    # noqa: BLE001
        return
    for row in doc.get("results", []):
        stem = str(row.get("stem", ""))
        if stem not in MELODIC_STEMS or not row.get("available"):
            continue
        t = np.asarray(row.get("onsets", []), dtype=float)
        c = np.asarray(row.get("confidences", []), dtype=float)
        if t.size == 0 or c.size != t.size:
            continue
        t = t[c >= conf]
        if t.size == 0:
            continue
        base = np.atleast_1d(np.asarray(onsets.get(stem, np.zeros(0)), dtype=float))
        bs = np.atleast_1d(np.asarray(strengths.get(stem, np.zeros(0)), dtype=float))
        if bs.size != base.size:
            bs = np.full(base.size, np.nan)
        allt = np.concatenate([base, t])
        alls = np.concatenate([bs, np.full(t.size, np.nan)])   # 有音高 note 无强度
        onsets[stem], strengths[stem] = _unique_with_strength(allt, alls, merge_sec)


def _bar_starts(analysis: dict) -> list[float]:
    """音频小节起始秒（1 起）。优先用 `bars[].start_sec`，缺了就按网格推。"""
    rows = analysis.get("bars", [])
    if rows and "start_sec" in rows[0]:
        return [float(r["start_sec"]) for r in rows]
    g = analysis["grid"]
    spb = 60.0 / float(g["bpm"])
    return [float(g["first"]) + i * 4.0 * spb for i in range(int(g["n_bars"]))]


def _segments_by_bar(analysis: dict) -> tuple[dict[int, dict], list[dict]]:
    rows = analysis.get("structure", {}).get("segments", [])
    by_bar: dict[int, dict] = {}
    for r in rows:
        for b in range(int(r["start_bar"]), int(r["end_bar"]) + 1):
            by_bar[b] = r
    return by_bar, list(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m tools.calibration.sampling",
        description="采音方式度量（全踩 / 舍音 / 留白，社区用词），逐小节"
                    "；--diagnostics 另出「似踩非踩」（MMFC 5.3 的第四个说法）标记")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("song", help="单曲逐小节表")
    s1.add_argument("song_dir")
    s1.add_argument("--onset-cache", default=None)
    s1.add_argument("--skeleton", default="bar",
                    help="bar / melodic / segment / drums / vocals / …")
    s1.add_argument("--pitch-conf", type=float, default=None,
                    help="并入置信度 ≥ 此值的 basic-pitch note（默认关闭）")
    s1.add_argument("--json", default=None, help="逐小节 JSON 输出路径")
    s1.add_argument("--diagnostics", action="store_true",
                    help="额外打印 coverage / extra / pool / 挑重音 / 似踩非踩等"
                         "**诊断数字与标记**（默认不打印：它们是工具内部口径，不是术语）。"
                         "extra = 谱面写了、哪条 stem 都对不上的格子数占比，社区叫"
                         "**采空音 / 插空音**——它高**不等于**谱写坏了：成段的高 extra "
                         "恰恰是 MMFC 5.3 说的**似踩非踩**（匀速铺满但不跟音轨）这种正规写法，"
                         "「似踩非踩」列就是「匀速铺满 + coverage 高 + extra 高」的诊断标记，"
                         "**不是第四个档位**（这些小节的采音方式多数仍报「全踩」）")

    s2 = sub.add_parser("corpus", help="全库汇总（三词分布 + 对照）")
    s2.add_argument("--calib-dir", default="out/calib")
    s2.add_argument("--onset-cache", default=None)
    s2.add_argument("--skeleton", default="bar")
    s2.add_argument("--pitch-conf", type=float, default=None)
    s2.add_argument("--json", required=True)
    s2.add_argument("--limit", type=int, default=0)

    a = p.parse_args(argv)
    if a.cmd == "song":
        d = load_pair(Path(a.song_dir),
                      Path(a.onset_cache) if a.onset_cache else None,
                      pitch_conf=a.pitch_conf)
        rows = song_sampling(d["bars"], d["slot_times"], d["notes_per_bar"],
                             d["onsets"], skeleton_mode=a.skeleton,
                             segment_skeleton=d["segment_skeleton"],
                             stem_strengths=d["strengths"],
                             melodic_default=d["melodic_default"])
        print(f"# {d['name']}  φ*={d['phi_ms']:+.1f}ms  小节 {len(rows)}")
        if a.diagnostics:
            print("bar  seg              采音方式  目标轨   [诊断] cov  extra pool slots "
                  "似踩非踩 挑重音")
        else:
            print("bar  seg              采音方式  目标轨")
        for r in rows:
            seg = d["seg_of_bar"].get(r.bar, {}).get("label_ja", "")
            line = f"{r.bar:>3}  {seg:<14} {r.mode:<8} {r.skeleton:<7}"
            if a.diagnostics:
                cov = "  n/a" if not np.isfinite(r.coverage) else f"{r.coverage:5.2f}"
                line += (f"  {cov} {r.extra_ratio:5.2f} {r.n_pool:>4} "
                         f"{r.n_slots:>5} {'似踩非踩' if r.pseudo_sample else '    ':<8} "
                         f"{r.accent_reason}")
            print(line)
        if a.json:
            Path(a.json).write_text(json.dumps([r.to_dict() for r in rows],
                                               ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        return 0

    from . import loader as loader_mod
    dirs = loader_mod.discover(Path(a.calib_dir))
    if a.limit:
        dirs = dirs[:a.limit]
    out = []
    for i, sd in enumerate(dirs, 1):
        try:
            d = load_pair(sd, Path(a.onset_cache) if a.onset_cache else None,
                          pitch_conf=a.pitch_conf)
            rows = song_sampling(d["bars"], d["slot_times"], d["notes_per_bar"],
                                 d["onsets"], skeleton_mode=a.skeleton,
                                 segment_skeleton=d["segment_skeleton"],
                                 stem_strengths=d["strengths"],
                                 melodic_default=d["melodic_default"])
            out.append({"name": d["name"], "genre": d["genre"], "level": d["level"],
                        "bpm": d["bpm"], "bars": [r.to_dict() for r in rows]})
        except Exception as exc:                     # noqa: BLE001
            out.append({"name": sd.name, "error": repr(exc)})
        if i % 20 == 0:
            print(f"  {i}/{len(dirs)}", flush=True)
    Path(a.json).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"写出 {a.json}（{len(out)} 曲）")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
