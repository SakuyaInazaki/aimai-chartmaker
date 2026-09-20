#!/usr/bin/env python3
"""左右手分配器：把一张谱的 note 事件分配给左右手，并给出手序合理性判定。

⚠️ **定义来源与边界**
---------------------
"手"的规则一律以 `.agent/knowledge/` 为准：

**v0.2（2026-09-19，用户讲授对齐）**：知识 064–067 与 019/030 的用户补充落地——
舒适区模型取代分页二分、出张改按键位定义、撤销星头换手与末尾半边、纵连长度优先、
段首按位置连续起手。逐项见 `docs/hand-sequencing.md` v0.2
与 `docs/research/hand-alignment-user-2026-09-19.md`；v0.1 参数集保留在 :data:`PARAMS_LEGACY`。

**v0.3（2026-09-20，去量化）**：用户指出 agent 不能拿阈值去量化制谱判断——
① **出张不分档**（一个布尔判定 + 计数，`w_chuzhang` 只是搜索用的内部代价）；
② **侧边双押去掉"位移/间隔"两个阈值门**，只留知识 030 的原话逻辑「没有引导就是突然」，
   且结果只作**复核清单**、不记无理；
③ **纵连长度按"需要击打按键的 note"数**（tap/hold 头/星星头都算）。

======  ==========================================================
064     **舒适区与出张**（用户）：左手 8765+1/4、右手 1234+8/5，共享区 {1,4,5,8}；
        **右手落 6/7、左手落 2/3 = 出张**，轻易不要写、写了要做好引导
065     **星星头与星星条可分属两手**（用户）：A 拍头可 A 划条也可 B 划条，
        **没有严格规定**，无偏好 → 撤销 `w_star_switch`、`w_slide_end_side` 降为 tie-breaker
066     **写谱不以特殊处理的手法为导向**（用户）：同时处理/狗刨是玩家解法不是谱师许可
        → "一手吃两个对象"保持关闭
067     **段首先出手 = 离上一段结束近的那只手**（用户）：段边界不重置两手状态
006     分页处理（右手 1234 / 左手 5678，**已被 064 的舒适区模型取代**）；正规手法：
        Tap/Touch 按 1 帧 (16.67 ms) 松手、Hold 按到尾 +1 帧、slide 星星头按 Tap、
        轨道匀速、**启动拍上的 Tap（拍划）并入划动手法不另占手**、Wifi 视作双手
007     划 slide 的手 = slide 末尾所在半边（**已被 065 降级为打分项/tie-breaker**）；
        四种换手定式；一笔画 = 一手拍一手划
019     纵连：**长度 ≤3 个音即使稍快也可单手**（用户补充，长度优先于速度）；
        "3 个以内"数的是**需要击打按键的 note**（tap / hold 头 / 星星头，用户 2026-09-20）
030     连续双押：**禁止突然的**侧边双押（23/67）——**突然 = 没有引导**，
        引导 = 双押带过来 / 18·87·76·65·54 / 18·23·45·67（用户补充）；不用数字定义
008     无理五类（多押 / 内屏 / 叠键 / 外键 / 撞尾）与软/硬/绝对三级
009     叠键红线：同判定区 < 2 帧 (33.3 ms) = 绝对无理
010     外键：slide 启动后 200 ms 内同侧 Tap/Hold
011     撞尾：末端 A 区 −50 ms ~ +200 ms
012     Hold 尾留空；短 Hold (<18 帧) 手感等同 Tap
014     slide 启动拍 = 60/BPM 固定一拍（与 `{x}` 分音无关）
015     速度红线（官谱实测：同键纵连 42.2 ms、扫键 18.75 ms、22.2 键/秒）
023     可及范围：右手够到 {8,1,2,3,4,5}、左手够到 {4,5,6,7,8,1}
======  ==========================================================

但"用代码判定一个 note 该给哪只手"必然要把文字规则**操作化**成代价函数与阈值。
这一层（权重数值、Fitts 式移动时间模型、束宽、路径近似）是**本模块的操作化**，
集中在 :data:`PARAMS` 里，可整体替换；引用时必须写明"这是 agent 的操作化"。

对外稳定 API
------------
::

    from chart_analysis.hands import assign, bar_hand_sequences

    ha = assign(parse_chart(text))        # 也接受 list[NoteEvent]
    ha.rows                               # [(t, bar, key, note_type, hand, occupancy_until), ...]
    ha.left_seq / ha.right_seq            # 每手的 (t, bar, key, kind) 序列
    ha.pairs                              # 同刻双手对 [(t, bar, Lkey, Rkey)]
    ha.bars                               # 逐小节手序指标（BarHand）
    ha.muri                               # 无理/不合理标记
    ha.infeasible                         # 不可行事件（带原因，绝不静默）
    bar_hand_sequences(ha)                # {bar: {"L": [...], "R": [...], "pairs": [...]}}

这套输出是**按手识别配置**（知识 017–030 的手级定义，见
`docs/hand-sequencing.md` §5）的输入口径——配置识别应当消费左右手序列，
而不是再去看原始键位串。

命令行
------
::

    PYTHONPATH=tools python3 -m chart_analysis.hands <chart.txt|关键字> [--bars a-b] [--json]
    python3 tools/chart_analysis/hands.py <chart.txt> --bars 20-24
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

try:  # 允许以脚本方式直接运行
    from .simai_parser import NoteEvent, ParseResult, parse_chart
except ImportError:  # pragma: no cover - 脚本入口
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from chart_analysis.simai_parser import NoteEvent, ParseResult, parse_chart

FRAME = 1.0 / 60.0  # 知识 006：一帧 16.67 ms

#: **需要击打按键的任务**（用户 2026-09-20：「只要需要击打按键的不就算吗？」）——
#: tap、hold 的头、星星头都要按下一个键，纵连长度与双押配对都按这个口径数。
_HIT_KINDS = ("tap", "hold", "star")

# ---------------------------------------------------------------------------
# 操作化参数（**agent 设定**，不是知识条目原文；默认值的来源写在注释里）
# ---------------------------------------------------------------------------

PARAMS: dict = {
    # ---- 手的可及模型（知识 064，用户 2026-09-19 讲授）----
    # "comfort" = 舒适区模型（**当前口径**）：共享区 {1,4,5,8} 两手零代价；
    #             **出张** = 左手落 {2,3} / 右手落 {6,7}。
    #             出张就是这一个**布尔判定**（`is_chuzhang`）+ 落点计数，没有档位/深度之分
    #             （知识 064 只区分"是不是出张"；"跨到 6 还是 7、跨多久、跨多频繁"
    #             用户没讲过，**本模块不得自造分档**）。
    # "page"    = v0.1 的分页二分 + 越界深度（知识 006/023），只在 PARAMS_LEGACY 里用。
    "hand_model": "comfort",
    # 搜索时给出张记的代价（**agent 操作化的内部数字，不是术语、不是给用户看的分级**；
    # 用户只说"轻易不要写"）。它的唯一作用是让搜索在"能不出张"时不要出张。
    "w_chuzhang": 0.90,
    # 共享区 {1,4,5,8} 上两手都是 0 代价（知识 064），于是"1 该谁打"在模型里成了**完全平局**，
    # 解由束序偶然决定，还会给出两手交叉（左手 1 / 右手 8）这种物理上说不通的解。
    # 故保留一个**极小的几何 tie-breaker**：落在知识 006 分页的对侧 +0.01。
    # 量级只有 `w_move`(0.12) 的 1/12、`w_chuzhang` 的 1/90 —— 它只打破平局，
    # **不表达 064 之外的任何偏好**（agent 操作化，v0.2 新增）。
    "w_home_tiebreak": 0.01,
    # ---- 时间常数 ----
    "frame": FRAME,                 # 知识 006
    "simul_tol": 0.004,             # 视作"同刻"的时间容差（伪 each 相差 1 ms）
    "hold_tail_frames": 1.0,        # 知识 006：Hold 按到尾 + 1 帧
    "hold_release_slack": 0.0167,   # Hold 尾放宽 1 帧（知识 012"结尾 12 帧不检查按压"的保守取值）
    "short_hold_sec": 0.300,        # 知识 012：<18 帧的 Hold 手感等同 Tap
    # 知识 048：slide 本体的判定容错 ≈ 终点前后 15 帧（≈250 ms）。于是占手区间是
    # **可伸缩**的 [头 + 一个四分音（014 启动拍）, 引导星到终点 ± 15 帧（048）]，
    # 不是固定时长。`slide_release_slack` = 允许"提前快滑"多久放手。
    "slide_tolerance_sec": 0.250,
    "slide_release_slack": 0.250,
    # ---- 单手移动时间模型（Fitts 式；t_req = t0 + t1·log2(1+d)） ----
    # t0 由知识 015 反推：官谱最快同键纵连 177.6BPM 32 分 = 42.2 ms，
    # 两手拆打则单手 84.4 ms —— 取 1/12 s = 83.3 ms 为单手连续击打的下界。
    "t_hit_min": 0.0833,
    "t_per_key": 0.030,             # agent 推断（用 LAMIA "飞 3 格" 150 ms 校准）
    # 硬下界（越过即判"超速不可行"）
    "t_sweep_req": 0.0333,          # 扫键（相邻键）的舒适线：2 帧（知识 015/036）
    "w_sweep_step": 0.35,           # 走扫键模式的额外代价（agent 推断）
    "t_hard_same": 0.0333,          # 知识 009：同键 2 帧
    "t_hard_floor": 0.0187,         # 知识 015：怒锤 200BPM 64 分扫键 18.75 ms
    "t_hard_per_key": 0.0,
    # ---- 代价权重 ----
    "w_page": 0.30,                 # 分页先验：每"深一格"的代价（知识 006/023）
    "w_move": 0.12,                 # 位移代价（知识 003-2），按 d/4 缩放
    "w_speed": 0.90,                # 逼近速度上界的代价
    "speed_exponent": 0.5,          # 速度代价对"贴线程度"取凸函数（越贴线越陡）
    "w_same_hand": 0.20,            # 连续两个单点给同一只手（违反交替先验，知识 018）
    # 同键连打：手已经在那个键上，续敲比换手省事（知识 027 定拍）——给**负**代价；
    # 知识 019 的"拆"由速度项负责（同键连打贴近单手上限时速度代价迅速上升）。
    # **v0.2 重标定**：v0.1 取 −0.12，靠 `w_page` 把共享键钉在某只手上才守得住知识 027
    # 的定拍；舒适区模型下 {1,4,5,8} 两手同价，−0.12 会让定拍段被拆成左右交替。
    # 取 −0.22 使"舒适速度内同键续敲"净代价 −0.02（略优于换手），
    # 贴近单手上限时优惠不给、回到知识 019 的"拆"。
    "w_same_key_extra": -0.22,
    # 知识 019 补充（用户 2026-09-19）：「理论上稍快且长度较小的纵连也可以单手处理
    # （长度较小指的是 3 个 tap 或以内）」→ **长度优先于速度**：
    # 同键连续串长度 ≤ `vertical_single_hand_max` 时允许单手——不加 `w_same_hand`、
    # `w_same_key_extra` 全额给出、**免除速度项**（硬下界 t_hard_same 仍生效，知识 009）。
    # **计数口径（用户 2026-09-20 原话：「只要需要击打按键的不就算吗？」）**：
    # 同键连续串里 **tap / hold 头 / 星星头** 都算一个音（`_HIT_KINDS`），不再只数 tap。
    "vertical_length_first": True,
    "vertical_single_hand_max": 3,
    # 知识 067（用户 2026-09-19）：「先出手最好是离上一段结束的手，因为上一个配置把手
    # 带到这里了」→ 段/配置片段的**首个任务**：① 不吃交替先验（隔了休止不存在"该换手了"）；
    # ② 额外按位移计价，让**离得近的那只手**起手。二者都不是硬约束——
    # `w_chuzhang`(0.90) 明显大于 `w_seg_start_far`(0.30)+`w_move`(0.12)，
    # 所以"就近起手"若会造成出张，**避免出张优先**（重调清单 §4-6 的保守取法）。
    "seg_gap_beats": 2.0,           # 距上一任务组 ≥ 这么多拍 = 新片段起手（agent 操作化）
    "seg_gap_min_sec": 0.40,
    "w_seg_start_far": 0.30,
    "seg_reset_alternation": True,
    # 知识 065（用户 2026-09-19）：「没有严格规定说拍星星头的手就一定能或者不能划星星条，
    # 也就是 A 拍头可以 A 划条也可以 B 划条」→ **撤销**星头换手代价（置 0）。
    "w_star_switch": 0.0,
    # 同上：「末尾半边」（知识 007）失去用户侧依据 → 降为**极小 tie-breaker**，
    # 只在两解代价几乎相等时打破平局（量级 ≤ 0.02，重调清单 §1 的保守取法）。
    "w_slide_end_side": 0.02,
    "w_cross": 0.40,                # 两手同时越到对侧（拓扑/撞手风险）
    # 一手扫掉**同刻**相邻两键（知识 036 的"扫"）。**v0.2 上调 0.90 → 1.20**：
    # 舒适区模型下 `2/3`、`6/7` 的两手解要付一次出张（0.90），旧值会让分配器改用
    # "一只手扫掉整组"来绕开——那正是知识 066 禁止的"因为有手法所以可以塞难配置"。
    # 取 1.20 > `w_chuzhang` + tie-breaker，使同刻扫只在两手解不可行时才被选中。
    "w_sweep_pair": 1.20,
    "w_unassigned": 3.0,            # 无手可用（多押）
    # 一笔画续划（知识 007）：上一条轨道正好停在本条轨道的头键上，手不松开直接接着划
    "chain_tol": 0.020,
    "w_chain_bonus": -0.30,         # 一笔画续划的奖励（知识 007）
    # 知识 017：星星头与启动拍之间的"错位音"**固定由划动的那只手自己拍**
    "w_misalign_other": 0.45,
    "misalign_ring": 1,             # 错位先验的作用半径（与星星头的环距）
    #: 知识 017：这条轨道的等待期里有错位音时，拍头手 ≠ 划轨手的代价（v0.2 新增，见上）
    "w_misalign_switch": 0.45,
    "allow_sweep_pair": True,
    # 知识 066（用户 2026-09-19）：「同时处理 / 狗刨这两个词语是手法的意思，使用手法会
    # 变得更好打，**不是说因为可以用这个手法就塞难打配置**……写谱也不是以特殊处理的手法
    # 为导向这么写的」→ "一手吃两个对象"（042 狗刨 / 055 穿心同时处理 / 060② 同起点同时
    # 处理）**保持关闭**，且理由从"实测前不得放宽 008/009"升级为**规划期口径**。
    # `allow_sweep_pair` 只为知识 036 的"同刻相邻两键一次刷过"开口，不属于本条范围。
    "one_hand_two_objects": False,
    "merge_multislide": True,       # 同头多 slide（`*`）视作一次连续划动（agent 推断，存疑）
    # ---- 无理阈值（知识 009/010/011） ----
    "muri_overlap_sec": 0.0333,
    "muri_slidehead_sec": 0.200,
    "muri_slidehead_warn_sec": 0.150,
    "muri_slidehead_near_sec": 0.050,
    "muri_slidehead_ring": 0,       # 知识 010 的"同头"= **同一个键**（388 谱校准，报告 §3）
    "slidehead_same_page": True,    # 知识 010 的"同侧"按**同半边**读（见报告 §3 校准）
    "slidehead_anchor": "move",     # 外键窗口锚点：move=启动拍（388 谱校准）/ star=星星头
    "muri_tail_pre_sec": 0.050,
    "muri_tail_post_sec": 0.200,
    "muri_tail_hard_sec": 0.150,
    # 知识 011："旧框 210BPM 以下八分撞尾可容忍" → 正好一个八分、且 BPM ≤ 210 的降为软
    "tail_eighth_tol": 0.012,
    "tail_bpm_tolerant": 210.0,
    "cross_recent_slide_sec": 1.0,  # 知识 007 定式 4（双手换位划）后的交叉不算拧巴
    "hold_tail_win": 0.050,         # Hold 尾多押的判定窗口（知识 012）
    "path_brush": True,             # 轨道途经 A 区的"蹭键"检查（agent 推断，存疑）
    "path_brush_win": 0.060,        # 途经判定区的前后窗口（agent 推断）
    # ---- 侧边双押（知识 030 铁律 + 2026-09-19 用户补充 + 2026-09-20 官谱普查）----
    # 知识 030 的原话逻辑只有一句：**禁止突然（＝没有引导）的侧边双押**。
    #   用户给的引导 = A 双押带过来 / B 18·87·76·65·54（共享键环上步进）
    #                / C 18·23·45·67（相邻键对等距轮转）
    #                （+ D 临近位置已经落在这两个键上——A 的单点版，知识 030 原表的
    #                  "单点铺垫 → 双押串"）
    # 388 官谱普查（`docs/research/side-double-guidance-and-chuzhang.md` §3）又归纳出
    #   五型：G 同对回访 / H 星星·长条落点送手 / F1 镜像对 / F2 同型相邻对 / E 相邻键铺垫。
    #   八型见 :data:`SIDE_GUIDE_TYPES`，**全部是结构关系，没有任何数值门**。
    # ⚠️ **"突然"不用数字定义**（用户 2026-09-20：「什么是侧边双押位移大的阈值……
    #    正常写谱谁去特意算这些事情」）——v0.2 曾有的"位移 ≥4 格 / 间隔 ≤140 ms"两个
    #    阈值门已整体删除，不再是规则的一部分。
    # ⚠️ 判定结果**只作复核清单**（`HandAssignment.side_doubles`），**不记无理**：
    #    八型仍是 agent 的归纳（用户只给过三个例子），覆盖率高不等于归纳正确。
    #    人看清单、人裁定。
    "side_read_slots": 4,           # **拍级阅读窗口** = 事件之前几个任务组（报告 §1.2 的
                                    #   阅读粒度，agent 操作化——是"人读上下文会看的那几拍"，
                                    #   不是判据门槛）。第二级窗口是**乐句级**（本小节此前 +
                                    #   前一小节），那是结构单位、没有参数。
    "side_near_slots": 2,           # **"临近位置"**（另一个结构粒度，不是阈值）：D 型问的是
                                    #   "手**还在不在**这两个键上"，那是紧挨着的一两组的事，
                                    #   不是"最近几拍见过"——所以 D 与 `near_dist` 诊断用它，
                                    #   其余七型用上面的阅读窗口。
    "side_guide_gap_beats": 1.0,    # 只供 `run_len` / `pos_in_run` 两个**诊断**字段：相邻两组
                                    #   双押算"连续双押串"的间隔上限（拍；同 configs 口径）。
                                    #   **A/B 与 C 的判定不再依赖它**（报告 §6.1 建议 3/4）。
    "side_rotation_min_run": 3,     # C 型等距轮转**子段**所需的最短长度（双押个数）
    # ---- 搜索 ----
    "beam": 48,
    "force_first_hand": "",         # "L"/"R" 时强制首个任务的手（起手奇偶实验用）
}

#: **v0.1 旧口径**（提交 4942759 的参数集），保留供新旧对照用。
#: 差别：分页二分 + 越界深度（006/023）、`w_star_switch` 与 `w_slide_end_side` 在位、
#: 纵连按速度线拆手、段首不作特殊处理。
#: （侧边双押的复核清单与它无关——那是纯诊断输出，两套参数下都会算。）
PARAMS_LEGACY: dict = dict(PARAMS)
PARAMS_LEGACY.update({
    "hand_model": "page",
    "w_chuzhang": 0.0,
    "w_home_tiebreak": 0.0,
    "w_same_key_extra": -0.12,
    "w_sweep_pair": 0.90,
    "w_misalign_switch": 0.0,
    "w_page": 0.30,
    "w_star_switch": 0.25,
    "w_slide_end_side": 0.35,
    "vertical_length_first": False,
    "w_seg_start_far": 0.0,
    "seg_reset_alternation": False,
})

_RIGHT_HOME = frozenset({1, 2, 3, 4})   # 知识 006：右手分页
_LEFT_HOME = frozenset({5, 6, 7, 8})
#: 知识 023：右手最多够到 5、左手最多够到 4（深度 ≤1 为可及，=2 为越界）
HOME: dict[str, frozenset[int]] = {"R": _RIGHT_HOME, "L": _LEFT_HOME}

#: **知识 064（用户 2026-09-19 讲授）的舒适区**——原话：
#: 「左手最好处理左半屏（8765）以及向右方延伸的上方 1 和下方 4、右手最好处理右半屏
#: （1234）以及 85，如果写出来涉及到右手需要拍 6 或者 7，左手拍 2 或者 3
#: 那你就要好好考虑难度了。」
COMFORT: dict[str, frozenset[int]] = {
    "L": frozenset({8, 7, 6, 5, 1, 4}),
    "R": frozenset({1, 2, 3, 4, 8, 5}),
}
#: 两手都"最好处理"的共享区（知识 064）——**两手落这里都零代价**
SHARED_KEYS = frozenset({1, 4, 5, 8})
#: 出张的四个「手 × 键」组合（知识 064）：右手落 6/7、左手落 2/3
CHUZHANG_KEYS: dict[str, frozenset[int]] = {"L": frozenset({2, 3}),
                                            "R": frozenset({6, 7})}

_TASK_KINDS = ("tap", "hold", "star", "slide", "wifi", "touch", "touch_hold")


# ---------------------------------------------------------------------------
# 键位几何
# ---------------------------------------------------------------------------


def cdist(a: int, b: int) -> int:
    """8 键环上的键位距离（0–4）。"""
    d = abs(int(a) - int(b)) % 8
    return min(d, 8 - d)


def cstep(a: int, b: int) -> int:
    """8 键环上从 a 到 b 的有向步进，取绝对值最小的表示（−3..4）。"""
    d = (int(b) - int(a)) % 8
    return d - 8 if d > 4 else d


def home_side(key: int) -> str:
    """键位的分页归属（知识 006）：1–4 属右手，5–8 属左手。"""
    return "R" if int(key) in _RIGHT_HOME else "L"


def page_depth(hand: str, key: int | None) -> int:
    """某手落在某键的"越界深度"。

    0 = 本方分页内；1 = 越到对侧一格（知识 023 的可及边界 5/4 与 8/1）；
    2 = 越过可及边界（右手的 6/7、左手的 2/3）。
    """
    if key is None:
        return 0
    k = int(key)
    if k in HOME[hand]:
        return 0
    return min(cdist(k, h) for h in HOME[hand])


def is_chuzhang(hand: str, key: int | None) -> bool:
    """**出张**（知识 064，用户口径）：右手落 6/7，或左手落 2/3。

    只有这四个「手 × 键」组合算出张；共享区 ``{1,4,5,8}`` 两手都不算。
    出张**不是无理**（用户给的是"要好好考虑难度 / 做好引导"），属于风格预算。
    """
    if key is None or hand not in ("L", "R"):
        return False
    return int(key) in CHUZHANG_KEYS[hand]


def reach_cost(hand: str, key: int | None, p: dict) -> float:
    """一只手落在某键的**可及性代价**。

    ``hand_model == "comfort"``（当前口径，知识 064）：出张 → ``w_chuzhang``，其余 0。
    ``hand_model == "page"``（v0.1 旧口径，知识 006/023）：``w_page × 越界深度``。
    """
    if p.get("hand_model", "comfort") == "page":
        return p["w_page"] * page_depth(hand, key)
    if key is None:
        return 0.0
    c = p["w_chuzhang"] if is_chuzhang(hand, key) else 0.0
    if home_side(key) != hand:
        c += p["w_home_tiebreak"]
    return c


def _other(hand: str) -> str:
    return "L" if hand == "R" else "R"


# ---------------------------------------------------------------------------
# slide 轨道几何（近似；**agent 推断，存疑**）
# ---------------------------------------------------------------------------


def slide_path_areas(shape: str, head: int, end: int) -> list[tuple[float, int]]:
    """轨道**途经**的外圈 A 区（不含起点与终点），返回 ``[(时间比例, 区号), ...]``。

    只覆盖沿判定圈走的弧线形状（``^`` / ``>`` / ``<`` / ``qq`` / ``pp``）与
    ``V`` 的折点；直线 ``-``、经中心的 ``v`` / ``q`` / ``p`` / ``s`` / ``z``
    不经过途中 A 区。``w``（Wifi）另行处理。

    ⚠️ 这是按 `docs/simai-syntax.md` §4.6 的形状描述做的**几何近似**，
    官方"始点-终点关系表"未逐格转录（见该文档 §8）——**agent 推断，存疑**。
    """
    head, end = int(head), int(end)
    sh = shape[:2] if shape[:2] in ("pp", "qq") else (shape[:1] if shape else "")
    if sh in ("-", "v", "q", "p", "s", "z", ""):
        return []
    if sh == "V":  # 三键记法：折点在 shape 之外，调用方另行给出
        return []
    if sh == "^":
        step = cstep(head, end)
        if step == 0 or abs(step) == 4:
            return []
        direction = 1 if step > 0 else -1
        span = abs(step)
    elif sh in (">", "<"):
        # 语法文档 §4.6：上半屏键 (1,2,7,8) 的 '>' 为顺时针（键号递增）
        base = 1 if head in (1, 2, 7, 8) else -1
        direction = base if sh == ">" else -base
        span = (end - head) % 8 if direction > 0 else (head - end) % 8
        if span == 0:
            span = 8
    else:  # qq / pp：绕外圈切线的大弧，近似为沿圈走
        direction = 1 if sh == "qq" else -1
        span = (end - head) % 8 if direction > 0 else (head - end) % 8
        if span == 0:
            span = 8
    out: list[tuple[float, int]] = []
    for i in range(1, span):
        area = (head - 1 + direction * i) % 8 + 1
        out.append((i / span, area))
    return out


def wifi_ends(end: int) -> tuple[int, int, int]:
    """Wifi 扇形的三个终点 A 区（知识 006：视作双手张开）。"""
    e = int(end)
    return ((e - 2) % 8 + 1, e, e % 8 + 1)


# ---------------------------------------------------------------------------
# 任务（需要一只手去做的最小单位）
# ---------------------------------------------------------------------------


@dataclass
class HandTask:
    """一个需要占用手的原子任务。"""

    index: int
    kind: str                 # tap / hold / star / slide / wifi / touch / touch_hold
    key: int | None           # 起始键位（touch 用伪键，中央为 None）
    t: float                  # 开始时刻（slide 轨道 = 启动拍，知识 014）
    t_end: float              # 动作结束时刻（不含 1 帧松手）
    measure: int
    beat: float
    end_key: int | None = None       # slide 轨道终点
    note_idx: int = -1               # 对应 ParseResult.notes 的下标
    slide_idx: int = -1              # 同一条 slide 的轨道任务下标（星头用）
    star_idx: int = -1               # 同一条 slide 的星头任务下标（轨道用）
    merged_taps: tuple[int, ...] = ()  # 并入本任务的拍划 tap（note 下标）
    merged_stars: tuple[int, ...] = ()  # 并入本任务的拍划**星星头**（note 下标）
    feeds: tuple[int, ...] = ()        # 本任务顺带打下的星星头所属的轨道任务下标
    wait_of: tuple[int, ...] = ()      # 落在哪些 slide 的"启动拍等待期"里（错位窗口）
    misalign_taps: tuple[int, ...] = ()  # （轨道任务用）落在本条轨道错位窗口里的任务
    #: 本任务所在的**同键连续串**长度（知识 019 用户补充：≤3 可单手）；不在串里时 = 1。
    #: 计数口径 = **所有需要击打按键的 note**（tap / hold 头 / 星星头，见 :data:`_HIT_KINDS`）
    #: ——用户 2026-09-20 原话：「只要需要击打按键的不就算吗？」
    vrun_len: int = 1
    t_end_nominal: float = 0.0         # 引导星按匀速到达终点的名义时刻
    t_end_max: float = 0.0             # 最晚放手时刻（名义尾 + 知识 048 容错）
    is_break: bool = False
    is_ex: bool = False
    shape: str = ""
    bpm: float = 0.0

    @property
    def free(self) -> float:
        """手被释放的时刻（知识 006：所有按压/划动带 1 帧松手延迟）。"""
        return self.t_end + PARAMS["frame"]


@dataclass
class HandNote:
    """逐 note 的分配结果。"""

    t: float
    measure: int
    beat: float
    key: str
    note_type: str            # tap / hold / slide_star / slide_track / touch / ...
    hand: str                 # 'L' / 'R' / 'LR'（Wifi）/ '-'（未分配=多押）
    occupancy_until: float
    t_hand: float = 0.0       # 手真正开始这个任务的时刻（slide 轨道 = 启动拍）
    task_kind: str = ""
    end_key: str = ""
    merged: bool = False      # 拍划：并入划动手法，不另占手（知识 006）
    reason: str = ""

    @property
    def row(self) -> tuple:
        return (self.t, self.measure, self.key, self.note_type,
                self.hand, self.occupancy_until)


@dataclass
class HandMuri:
    """无理 / 不合理标记。"""

    kind: str                 # 多押 / 叠键 / 外键 / 撞尾 / 路径蹭键 / 超速 / 占用冲突 / 换手拧巴
    level: str                # 绝对 / 硬 / 软
    measure: int
    time: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "level": self.level, "measure": self.measure,
                "time": round(self.time, 4), "detail": self.detail}


@dataclass
class BarHand:
    """逐小节手序指标。"""

    measure: int
    n_tasks: int = 0
    n_left: int = 0
    n_right: int = 0
    alt_rate: float = 0.0          # 交替率：相邻任务换手的比例
    max_same_run: int = 0          # 同手最长连打
    cross_count: int = 0           # 跨半圈（落在对侧分页）的任务数（v0.1 旧口径诊断量）
    #: **出张**（知识 064 用户口径）：右手落 6/7 或左手落 2/3 的**落点数**
    chuzhang: int = 0
    n_slides: int = 0              # 本小节的 slide/wifi 轨道任务数
    chuzhang_slide: int = 0        # 其中头键或尾键落进出张区的根数（知识 064）
    chuzhang_end_side: int = 0     # **v0.1 旧口径**：划轨手不在 slide 末尾半边的根数（007，已作废）
    max_speed_l: float = 0.0       # 左手最大等效速度（键/秒）
    max_speed_r: float = 0.0
    conflicts: int = 0             # 占用冲突 / 无手可用
    muri: dict = field(default_factory=dict)
    cost: float = 0.0              # 本小节累计代价
    hardness_raw: float = 0.0      # 代价 / 任务数
    hardness: float = 0.0          # 曲内 min-max 归一 [0,1]

    def to_dict(self) -> dict:
        return {"measure": self.measure, "n_tasks": self.n_tasks,
                "n_left": self.n_left, "n_right": self.n_right,
                "alt_rate": round(self.alt_rate, 4),
                "max_same_run": self.max_same_run,
                "cross_count": self.cross_count, "chuzhang": self.chuzhang,
                "n_slides": self.n_slides,
                "chuzhang_slide": self.chuzhang_slide,
                "chuzhang_end_side": self.chuzhang_end_side,
                "max_speed_l": round(self.max_speed_l, 3),
                "max_speed_r": round(self.max_speed_r, 3),
                "conflicts": self.conflicts, "muri": dict(self.muri),
                "cost": round(self.cost, 4),
                "hardness_raw": round(self.hardness_raw, 4),
                "hand_hardness": round(self.hardness, 4)}


@dataclass
class HandAssignment:
    """一张谱的双手分配结果（对外稳定 API）。"""

    notes: list[HandNote] = field(default_factory=list)
    tasks: list[HandTask] = field(default_factory=list)
    task_hand: list[str] = field(default_factory=list)   # 与 tasks 同序
    bars: list[BarHand] = field(default_factory=list)
    muri: list[HandMuri] = field(default_factory=list)
    #: "蹭的压力"——星星篇（知识 048/052/062）里大量的"蹭"是**设计手段**不是无理，
    #: 因此与 :attr:`muri` 分开：软级的撞尾/外键/路径蹭键归这里，硬/绝对仍算无理。
    scrape: list[HandMuri] = field(default_factory=list)
    infeasible: list[HandMuri] = field(default_factory=list)
    #: 全谱侧边双押事件（知识 030：每一处**是哪一种引导**，多标签 ``guide_types``，
    #: 值域见 :data:`SIDE_GUIDE_TYPES`）。``guided=False`` 的那些（**乐句级窗口仍判不出**）
    #: 就是**供人复核的清单**——不记无理、不下判决，见 :func:`side_double_events`。
    side_doubles: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    n_tasks: int = 0
    params: dict = field(default_factory=dict)

    # ---- 便捷视图 ----
    @property
    def rows(self) -> list[tuple]:
        """``[(t, bar, key, note_type, hand, occupancy_until), ...]``（按时间排序）。"""
        return [n.row for n in self.notes]

    def __iter__(self):
        return iter(self.rows)

    def _seq(self, hand: str) -> list[tuple[float, int, str, str]]:
        out = []
        for t, h in zip(self.tasks, self.task_hand):
            if h == hand or (h == "LR" and hand in ("L", "R")):
                out.append((t.t, t.measure, str(t.key) if t.key else "C", t.kind))
        return out

    @property
    def left_seq(self) -> list[tuple[float, int, str, str]]:
        """左手序列 ``[(时间, 小节, 键位, 任务类型), ...]``。"""
        return self._seq("L")

    @property
    def right_seq(self) -> list[tuple[float, int, str, str]]:
        return self._seq("R")

    @property
    def pairs(self) -> list[tuple[float, int, str, str]]:
        """同刻双手对 ``[(时间, 小节, 左手键, 右手键), ...]``（双押）。"""
        tol = PARAMS["simul_tol"]
        out = []
        byt: dict[int, dict[str, HandTask]] = {}
        for t, h in zip(self.tasks, self.task_hand):
            if h not in ("L", "R"):
                continue
            slot = int(round(t.t / tol))
            byt.setdefault(slot, {})[h] = t
        for slot in sorted(byt):
            d = byt[slot]
            if "L" in d and "R" in d:
                out.append((d["L"].t, d["L"].measure,
                            str(d["L"].key or "C"), str(d["R"].key or "C")))
        return out

    @property
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for m in self.muri:
            c[m.kind] = c.get(m.kind, 0) + 1
        return c

    @property
    def scrape_counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for m in self.scrape:
            c[m.kind] = c.get(m.kind, 0) + 1
        return c

    @property
    def hand_hardness(self) -> float:
        """全谱手序难度分 = 总代价 / 任务数。"""
        return self.total_cost / self.n_tasks if self.n_tasks else 0.0

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "total_cost": round(self.total_cost, 4),
            "hand_hardness": round(self.hand_hardness, 5),
            "n_infeasible": len(self.infeasible),
            "muri_counts": self.counts,
            "bars": [b.to_dict() for b in self.bars],
        }


# ---------------------------------------------------------------------------
# 1. 事件 → 任务
# ---------------------------------------------------------------------------


def build_tasks(res: ParseResult | Sequence[NoteEvent],
                params: dict | None = None) -> tuple[list[HandTask], list[NoteEvent], dict[int, int]]:
    """把 note 事件序列展开成手的任务序列。

    规则（知识 006/012/014）：

    - ``tap`` / ``slide_star`` → 占手 1 帧；
    - ``hold`` → 占手到尾 + 1 帧（短 Hold 见 ``short_hold_sec``）；
    - ``slide_track`` → **从启动拍（头 + 60/BPM）到轨道结束**占手（知识 014）；
      ``w`` 形 Wifi 视作双手（知识 006）；
    - **拍划**：落在启动拍上、与星星头同键的 Tap 并入划动手法，不另占手（知识 006）。

    返回 ``(tasks, notes, note_idx → task_idx)``；被并入的拍划 tap 映射到轨道任务。
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    notes = list(res.notes) if isinstance(res, ParseResult) else list(res)
    notes = sorted(notes, key=lambda n: (n.time, n.kind, n.key))

    tasks: list[HandTask] = []
    n2t: dict[int, int] = {}
    star_of: dict[tuple[float, str], int] = {}   # (时间, 键) → 星头任务下标

    # 先建非轨道任务，再建轨道任务（轨道需要知道星头）
    for i, n in enumerate(notes):
        if n.kind == "slide_track":
            continue
        key = int(n.key) if n.key.isdigit() else None
        if n.kind in ("tap", "slide_star"):
            kind = "star" if n.kind == "slide_star" else "tap"
            t_end = n.time
        elif n.kind == "hold":
            kind = "hold"
            # 知识 006：Hold 按到尾（+1 帧松手由 HandTask.free 加）；
            # 知识 012 的"结尾若干帧不检查按压"用 hold_release_slack 放宽
            t_end = max(n.time, n.time + n.duration - p["hold_release_slack"])
        elif n.kind in ("touch", "touch_hold"):
            kind = n.kind
            t_end = n.time + (n.duration if n.kind == "touch_hold" else 0.0)
            key = int(n.key[1]) if len(n.key) > 1 and n.key[1].isdigit() else None
        else:
            continue
        idx = len(tasks)
        tasks.append(HandTask(index=idx, kind=kind, key=key, t=n.time, t_end=t_end,
                              measure=n.measure, beat=n.beat, note_idx=i,
                              is_break=n.is_break, is_ex=n.is_ex, bpm=n.bpm))
        n2t[i] = idx
        if kind == "star" and key is not None:
            star_of[(round(n.time, 4), n.key)] = idx

    # 轨道任务
    track_tasks: list[tuple[int, NoteEvent]] = []
    for i, n in enumerate(notes):
        if n.kind != "slide_track" or not n.key.isdigit():
            continue
        t_move = n.time + n.wait
        kind = "wifi" if n.shape.startswith("w") else "slide"
        idx = len(tasks)
        st = star_of.get((round(n.time, 4), n.key), -1)
        nominal_end = t_move + n.duration
        task = HandTask(index=idx, kind=kind, key=int(n.key), t=t_move,
                        t_end=nominal_end,
                        t_end_nominal=nominal_end,
                        t_end_max=nominal_end + p["slide_tolerance_sec"],
                        measure=n.measure, beat=n.beat,
                        end_key=int(n.end_key) if n.end_key.isdigit() else None,
                        note_idx=i, star_idx=st, is_break=n.is_break,
                        shape=n.shape, bpm=n.bpm)
        tasks.append(task)
        n2t[i] = idx
        track_tasks.append((idx, n))
        if st >= 0:
            tasks[st].slide_idx = idx

    # 拍划合并（知识 006）：落在启动拍上、与星星头**同键**的 Tap 并入划动手法，
    # 不单独占用一只手。知识 006 同时规定"slide 星星头按 Tap 处理"，所以
    # **落在启动拍上的另一条 slide 的星星头**同样并入（星链/一圈星星头连飞）——
    # 划动的手在那一刻正好按在同一个键上，一次按压兼两件事。
    tol = p["simul_tol"]
    merged_tap: set[int] = set()
    merged_star: set[int] = set()
    by_note = {t.note_idx: t for t in tasks}
    for idx, n in track_tasks:
        task = tasks[idx]
        for j, m in enumerate(notes):
            if m.key != n.key or abs(m.time - task.t) > max(tol, p["frame"]):
                continue
            if m.kind == "tap" and j not in merged_tap:
                merged_tap.add(j)
                task.merged_taps = task.merged_taps + (j,)
            elif m.kind == "slide_star" and j not in merged_star:
                st = by_note.get(j)
                if st is None or st.kind != "star" or st.index == idx:
                    continue
                merged_star.add(j)
                task.merged_stars = task.merged_stars + (j,)

    # 过滤掉被并入的任务，按时间重排，再用对象身份重建全部下标引用
    dropped = {t.index for t in tasks
               if (t.kind == "tap" and t.note_idx in merged_tap)
               or (t.kind == "star" and t.note_idx in merged_star)}
    star_task_by_note = {t.note_idx: t for t in tasks if t.kind == "star"}
    kept = [t for t in tasks if t.index not in dropped]
    kept.sort(key=lambda t: (t.t, t.index))
    new_index = {id(t): i for i, t in enumerate(kept)}
    by_old = {t.index: t for t in tasks}

    # feeds：本任务顺带打下的星星头，对应哪些轨道任务
    feeds: dict[int, list[int]] = {}
    for t in tasks:
        if t.index in dropped:
            continue
        acc: list = []
        if t.kind == "star" and t.slide_idx >= 0:
            acc.append(by_old.get(t.slide_idx))
        for j in t.merged_stars:
            st = star_task_by_note.get(j)
            if st is not None and st.slide_idx >= 0:
                acc.append(by_old.get(st.slide_idx))
        feeds[t.index] = [a for a in acc if a is not None]

    ref_star = {t.index: by_old.get(t.star_idx) for t in kept}
    ref_slide = {t.index: by_old.get(t.slide_idx) for t in kept}
    ref_feed = {t.index: feeds.get(t.index, []) for t in kept}
    # 星星头被并入时，"打下我星星头的那只手" = 吞掉它的那个轨道任务
    absorbed_by: dict[int, HandTask] = {}
    for t in tasks:
        for j in t.merged_stars:
            st = star_task_by_note.get(j)
            if st is not None and st.slide_idx >= 0:
                tr = by_old.get(st.slide_idx)
                if tr is not None:
                    absorbed_by[tr.index] = t
    for t in kept:
        sr = ref_star[t.index] or absorbed_by.get(t.index)
        sl = ref_slide[t.index]
        t.star_idx = new_index.get(id(sr), -1) if sr is not None else -1
        t.slide_idx = new_index.get(id(sl), -1) if sl is not None else -1
        t.feeds = tuple(sorted({new_index[id(a)] for a in ref_feed[t.index]
                                if id(a) in new_index}))
    for i, t in enumerate(kept):
        t.index = i
    n2t_extra: dict[int, int] = {}
    # 同头多 slide（`*`）：几条轨道同头同刻出发，视作**一次连续划动**（agent 推断，存疑）
    if p["merge_multislide"]:
        groups: dict[tuple, list[HandTask]] = {}
        for t in kept:
            if t.kind == "slide":
                groups.setdefault((round(t.t, 4), t.key), []).append(t)
        drop2 = set()
        for g in groups.values():
            if len(g) < 2:
                continue
            main = max(g, key=lambda x: x.t_end)
            for x in g:
                if x is main:
                    continue
                main.t_end = max(main.t_end, x.t_end)
                main.t_end_nominal = max(main.t_end_nominal, x.t_end_nominal)
                main.t_end_max = max(main.t_end_max, x.t_end_max)
                main.merged_taps += x.merged_taps
                main.merged_stars += x.merged_stars
                main.feeds = tuple(sorted(set(main.feeds) | set(x.feeds)))
                drop2.add(x.index)
                for j in [x.note_idx] + list(x.merged_taps) + list(x.merged_stars):
                    if j >= 0:
                        n2t_extra.setdefault(j, main.index)
        if drop2:
            surv = [t for t in kept if t.index not in drop2]
            remap2 = {t.index: i for i, t in enumerate(surv)}
            for t in surv:
                t.star_idx = remap2.get(t.star_idx, -1)
                t.slide_idx = remap2.get(t.slide_idx, -1)
                t.feeds = tuple(sorted({remap2[f] for f in t.feeds if f in remap2}))
            n2t_extra = {k: remap2.get(v, -1) for k, v in n2t_extra.items()}
            for i, t in enumerate(surv):
                t.index = i
            kept = surv

    n2t = {}
    for t in kept:
        if t.note_idx >= 0:
            n2t[t.note_idx] = t.index
        for j in t.merged_taps + t.merged_stars:
            n2t[j] = t.index
    for j, ti in n2t_extra.items():
        if ti >= 0:
            n2t.setdefault(j, ti)

    # 同键连续串的长度（知识 019 的 2026-09-19 用户补充：长度优先于速度）。
    # "连续"= 在任务时间序里**紧挨着**且键位相同的一段（中间夹了别的键就断开）。
    # **计数口径（用户 2026-09-20）**：「只要需要击打按键的不就算吗？」——
    # tap / hold 头 / 星星头都要按下这个键，一律计入（`_HIT_KINDS`）。
    # v0.2 只数 tap 是照原话"3 个 tap 或以内"的字面取的，已被用户这句话否掉。
    i = 0
    while i < len(kept):
        if kept[i].kind not in _HIT_KINDS or kept[i].key is None:
            i += 1
            continue
        j = i
        while (j + 1 < len(kept) and kept[j + 1].kind in _HIT_KINDS
               and kept[j + 1].key == kept[i].key
               and kept[j + 1].t > kept[j].t + 1e-9):
            j += 1
        ln = j - i + 1
        for k in range(i, j + 1):
            kept[k].vrun_len = ln
        i = j + 1

    # 错位窗口（知识 017）：标注每个任务落在哪些 slide 的"星星头 → 启动拍"等待期里
    tracks = [t for t in kept if t.kind in ("slide", "wifi")]
    for t in kept:
        if t.kind in ("slide", "wifi"):
            continue
        acc = []
        for tr in tracks:
            st = (kept[tr.star_idx].t if 0 <= tr.star_idx < len(kept)
                  else notes[tr.note_idx].time if tr.note_idx >= 0 else tr.t)
            if (st - 1e-6 < t.t < tr.t - 1e-6 and t.key is not None
                    and tr.key is not None
                    and cdist(t.key, tr.key) <= p["misalign_ring"]):
                acc.append(tr.index)
        t.wait_of = tuple(acc)
    # 反向索引：每条轨道的错位窗口里有哪些任务（知识 017 的"错位音"）
    back: dict[int, list[int]] = {}
    for t in kept:
        for w in t.wait_of:
            back.setdefault(w, []).append(t.index)
    for tr in tracks:
        tr.misalign_taps = tuple(back.get(tr.index, ()))
    return kept, notes, n2t


# ---------------------------------------------------------------------------
# 2. 代价模型
# ---------------------------------------------------------------------------


def t_required(d: int, p: dict) -> float:
    """单手从当前键移动 ``d`` 格并完成下一次击打所需的**舒适**时间（秒）。"""
    return p["t_hit_min"] + p["t_per_key"] * math.log2(1 + d)


def t_hard_limit(d: int, p: dict) -> float:
    """单手移动 ``d`` 格的**物理硬下界**（秒）；低于此判"超速不可行"。"""
    if d == 0:
        return p["t_hard_same"]
    return max(p["t_hard_floor"], p["t_hard_floor"] + p["t_hard_per_key"] * math.log2(1 + d))


@dataclass(frozen=True)
class _HState:
    key: int | None
    ready: float          # 上一个动作**名义结束**的时刻（此后可移动）
    elastic: float = 0.0  # 可提前放手的余量（知识 048：slide 终点 ±15 帧容错）

    @property
    def free(self) -> float:
        return self.ready + PARAMS["frame"]

    @property
    def free_elastic(self) -> float:
        """最早可以脱手去做下一件事的时刻（"提前快滑"用掉容错窗口）。"""
        return self.ready - self.elastic + PARAMS["frame"]


@dataclass
class _Node:
    cost: float
    L: _HState
    R: _HState
    last_hand: str
    last_t: float
    pending: tuple            # ((track_task_index, star_hand), ...)
    parent: "_Node | None"
    decisions: tuple          # ((task_index, hand), ...)
    marks: tuple              # ((kind, level, task_index, detail), ...)


def _hit_cost(hand: str, st: _HState, key: int | None, t: float,
              p: dict, seg_start: bool = False,
              vrun_single: bool = False) -> tuple[float, list[tuple[str, str, str]]]:
    """一只手从状态 ``st`` 去 ``key`` 在 ``t`` 出手的代价；返回 (cost, marks) 或 (inf, ...)。

    ``seg_start``：本任务是新段/新配置片段的首个任务（知识 067）——额外按位移计价，
    让"离上一段结束近的那只手"起手。
    ``vrun_single``：本任务属于**长度 ≤3 的同键 tap 串**（知识 019 用户补充）——
    免除速度项（硬下界仍生效），长度优先于速度。
    """
    marks: list[tuple[str, str, str]] = []
    cost = reach_cost(hand, key, p)
    if st.key is None:
        return cost, marks
    if key is None:
        return cost, marks
    d = cdist(st.key, key)
    cost += p["w_move"] * d / 4.0
    if seg_start:
        # 知识 067：段首先出手 = 离上一段结束位置近的那只手
        cost += p["w_seg_start_far"] * d / 4.0
    dt = t - st.ready
    if vrun_single and d == 0:
        # 知识 019 用户补充：「稍快且长度较小的纵连也可以单手处理（3 个 tap 或以内）」
        # —— 长度优先于速度；只有知识 009 的叠键红线（33.3 ms）仍是硬下界
        if dt <= p["t_hard_same"]:
            return math.inf, [("超速", "绝对", f"{hand} 手同键连打仅 {dt * 1000:.1f} ms "
                                              f"< 叠键红线 {p['t_hard_same'] * 1000:.1f} ms")]
        return cost, marks
    treq = t_required(d, p)
    thard = t_hard_limit(d, p)
    if d == 1 and dt < treq and dt >= p["t_sweep_req"]:
        # 扫键：一只手刷过相邻键（知识 015 的官谱实测下界 18.75 ms、知识 036 的"扫"）
        return cost + p["w_sweep_step"], marks
    if d == 1 and dt < p["t_sweep_req"]:
        treq = p["t_sweep_req"]
        cost += p["w_sweep_step"]
    if dt >= treq:
        return cost, marks
    if dt <= thard:
        return math.inf, [("超速", "绝对", f"{hand} 手 {st.key}→{key}（{d} 格）"
                                          f"仅 {dt * 1000:.1f} ms < 硬下界 {thard * 1000:.1f} ms")]
    frac = (treq - dt) / (treq - thard)
    cost += p["w_speed"] * frac ** p["speed_exponent"]
    if frac > 0.75:
        marks.append(("超速", "硬", f"{hand} 手 {st.key}→{key}（{d} 格）"
                                   f"{dt * 1000:.1f} ms（舒适线 {treq * 1000:.0f} ms）"))
    return cost, marks


def _apply_one(node: _Node, hand: str, task: HandTask, p: dict,
               pending: dict) -> tuple[float, _HState, list]:
    """把单个任务交给 ``hand``；返回 (增量代价, 新手状态, marks)。"""
    st = node.L if hand == "L" else node.R
    marks: list = []
    # 一笔画续划（知识 007）：上一个动作正好停在本条轨道的头键上 → 手不松开接着划，
    # 不吃 1 帧松手延迟、也不算"移动"。官方谱的星链（slide 首尾相接）全靠这条。
    chain = (task.kind in ("slide", "wifi") and st.key is not None
             and task.key is not None and int(st.key) == int(task.key)
             and abs(task.t - st.ready) <= p["chain_tol"])
    avail = st.ready if chain else st.free_elastic
    # 占用冲突（知识 006/012）
    if task.t < avail - 1e-6:
        return math.inf, st, [("占用冲突", "绝对",
                               f"{hand} 手被占到 {avail:.3f}s，"
                               f"但 {task.t:.3f}s 又来 {task.kind}")]
    seg_start = _is_seg_start(node, task, p)
    vrun_single = bool(p["vertical_length_first"]
                       and 2 <= task.vrun_len <= p["vertical_single_hand_max"])
    if chain:
        cost, m = reach_cost(hand, task.key, p) + p["w_chain_bonus"], []
    else:
        cost, m = _hit_cost(hand, st, task.key, task.t, p, seg_start, vrun_single)
    marks.extend(m)
    if cost == math.inf:
        return math.inf, st, marks

    if task.kind in ("slide", "wifi"):
        # 划轨手要把整条轨道走完，可及性代价按"头 + 尾"的平均算
        # （_hit_cost 里只按头键算了一次，这里补上尾键的一半差额）。
        # 知识 064 + 065：这一项**只表达"轨道尾落不落在这只手的舒适区"这层可及性成本**，
        # 不再表达"尾在哪边就该谁划"（那是 007 的末尾半边，已被 065 降级）。
        if task.end_key is not None:
            cost += (reach_cost(hand, task.end_key, p)
                     - reach_cost(hand, task.key, p)) / 2.0
        # 知识 065（用户）：星星头与星星条**可分属两手，没有严格规定** →
        # `w_star_switch` 已置 0；`w_slide_end_side` 只剩极小 tie-breaker。
        if not chain:
            if task.end_key is not None and home_side(task.end_key) != hand:
                cost += p["w_slide_end_side"]
            sh = pending.get(task.index)
            if sh is not None and sh != hand:
                cost += p["w_star_switch"]
                # 知识 017（用户原文）：「错位音**固定由滑动的那只手自己拍**——
                # 头 → 错位音 → 启动拍 → 划轨」。知识 065 撤销的是"头与条同手"的
                # **普遍**偏好，**不是 017 这一条配置定义**：一旦这条轨道的启动拍
                # 等待期里真的有错位音，头 / 错位音 / 轨道就该是同一只手。
                # 不加这一项，w_star_switch=0 后模型会把"头+错位音"给 A、轨道给 B，
                # 绕开 `w_misalign_other` 而实际并不满足 017。
                if task.misalign_taps:
                    cost += p["w_misalign_switch"]
        end_key = task.end_key if task.end_key is not None else task.key
        new = _HState(end_key, task.t_end, p["slide_release_slack"])
    else:
        new = _HState(task.key, task.t_end)

    # 错位先验（知识 017）：落在某条 slide 启动拍等待期、且贴近其星星头的音符，
    # 应当由**将要划那条 slide 的那只手**自己拍（"头 → 错位音 → 启动拍"连续作业）
    for w in task.wait_of:
        sh = pending.get(w)
        if sh is not None and sh != hand:
            cost += p["w_misalign_other"]

    # 交替先验（知识 018）：上一个单手任务也是这只手
    same_key = (st.key is not None and task.key is not None and st.key == task.key)
    if node.last_hand == hand and node.last_t < task.t - 1e-9:
        if vrun_single and same_key:
            # 知识 019 用户补充：≤3 个 tap 的短纵连"其实没什么要求"，单手是合法解 →
            # 不加交替代价，同键续敲的优惠**全额**给出（不再要求落在舒适速度内）
            cost += p["w_same_key_extra"]
        elif seg_start and p["seg_reset_alternation"]:
            # 知识 067：隔了一个休止的新片段，不存在"上一音该换手了"这回事
            pass
        else:
            cost += p["w_same_hand"]
            # 同键续敲的优惠只在**舒适速度内**给（知识 027 定拍）；
            # 一旦贴近单手上限就是知识 019 的"拆"，不再优惠
            if same_key and task.t - st.ready >= t_required(0, p) - 1e-9:
                cost += p["w_same_key_extra"]
    return cost, new, marks


def _is_seg_start(node: _Node, task: HandTask, p: dict) -> bool:
    """本任务是否为一个新段 / 新配置片段的首个任务（知识 067 的作用点）。

    判据（**agent 操作化**，用户未给量化）：距上一个任务组 ≥ ``seg_gap_beats`` 拍
    （下界 ``seg_gap_min_sec``）。用户原话讲的是"上一个配置把手带到这里"，
    因此只要**中间有一段空档**就按"新配置起手"处理。
    """
    if p["w_seg_start_far"] <= 0 and not p["seg_reset_alternation"]:
        return False
    gap = task.t - node.last_t
    if gap <= 0:
        return False
    thr = p["seg_gap_min_sec"]
    if task.bpm:
        thr = max(thr, p["seg_gap_beats"] * 60.0 / task.bpm)
    return gap >= thr


def _enumerate_vectors(tasks: Sequence[HandTask], p: dict) -> list[tuple]:
    """枚举一个同刻任务组的手分配向量。

    向量元素 ∈ ``'L' / 'R' / 'LR'（Wifi 占双手）/ None（无手可用）``。
    """
    n = len(tasks)
    wifi = [i for i, t in enumerate(tasks) if t.kind == "wifi"]
    if wifi:
        if n == 1:
            return [("LR",)]
        # Wifi 与别的 note 同刻 → 其余无手可用
        vec = [None] * n
        vec[wifi[0]] = "LR"
        return [tuple(vec)]
    if n == 1:
        return [("L",), ("R",)]
    if n == 2:
        out = [("L", "R"), ("R", "L")]
        if (p["allow_sweep_pair"] and tasks[0].key and tasks[1].key
                and cdist(tasks[0].key, tasks[1].key) == 1
                and all(t.kind in ("tap", "star") for t in tasks)):
            out.extend([("L", "L"), ("R", "R")])
        return out
    # n >= 3 → 多押：只能挑两个交给两只手，其余无手可用
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vec: list = [None] * n
            vec[i] = "L"
            vec[j] = "R"
            out.append(tuple(vec))
    return out


def _apply_vector(node: _Node, tasks: Sequence[HandTask], vec: tuple,
                  p: dict) -> _Node | None:
    """在 ``node`` 上施加一个分配向量，返回新节点（不可行返回 None）。"""
    pending = dict(node.pending)
    cost = 0.0
    marks: list = []
    decisions: list = []
    L, R = node.L, node.R
    cur = _Node(node.cost, L, R, node.last_hand, node.last_t, node.pending,
                node, (), ())

    # Wifi：双手同用
    if any(v == "LR" for v in vec):
        wi = vec.index("LR")
        task = tasks[wi]
        for hand in ("L", "R"):
            st = cur.L if hand == "L" else cur.R
            chain = (st.key is not None and task.key is not None
                     and int(st.key) == int(task.key)
                     and abs(task.t - st.ready) <= p["chain_tol"])
            if task.t < (st.ready if chain else st.free_elastic) - 1e-6:
                return None
        cc = []
        for hand, st in (("L", cur.L), ("R", cur.R)):
            chained = (st.key is not None and task.key is not None
                       and int(st.key) == int(task.key)
                       and abs(task.t - st.ready) <= p["chain_tol"])
            if chained:
                cc.append(reach_cost(hand, task.key, p))
            else:
                c, _ = _hit_cost(hand, st, task.key, task.t, p,
                                 _is_seg_start(node, task, p))
                cc.append(c)
        if any(c == math.inf for c in cc):
            return None
        cost += sum(cc)
        ek = task.end_key if task.end_key is not None else task.key
        L = _HState(ek, task.t_end, p["slide_release_slack"])
        R = _HState(ek, task.t_end, p["slide_release_slack"])
        decisions.append((task.index, "LR"))
        for i, v in enumerate(vec):
            if i != wi:
                cost += p["w_unassigned"]
                marks.append(("多押", "绝对", tasks[i].index,
                              f"Wifi 同刻另有 {tasks[i].kind}，无手可用"))
                decisions.append((tasks[i].index, "-"))
        last_hand, last_t = "", task.t
        return _Node(node.cost + cost, L, R, last_hand, last_t,
                     tuple(sorted(pending.items())), node,
                     tuple(decisions), tuple(marks))

    # 一手扫同刻相邻两键（知识 036：一只手一次刷过两个相邻键）
    hands_used = [v for v in vec if v in ("L", "R")]
    if len(hands_used) == 2 and hands_used[0] == hands_used[1]:
        hand = hands_used[0]
        st = cur.L if hand == "L" else cur.R
        pair = [t for t, v in zip(tasks, vec) if v == hand]
        if tasks[0].t < st.free - 1e-6:
            return None
        if st.key is not None:
            pair.sort(key=lambda t: cdist(st.key, t.key or 0))
        # 只对"先碰到的那个键"计移动代价；第二个键由扫的动作顺带吃掉
        c, mk = _hit_cost(hand, st, pair[0].key, pair[0].t, p,
                          _is_seg_start(node, pair[0], p))
        if c == math.inf:
            return None
        # 两个键都要落进这只手（知识 064：出张按落点算），第二个键补一次可及性代价
        cost += c + p["w_sweep_pair"] + reach_cost(hand, pair[-1].key, p)
        for kind, level, detail in mk:
            marks.append((kind, level, pair[0].index, detail))
        end_st = _HState(pair[-1].key, max(t.t_end for t in pair))
        for t in pair:
            decisions.append((t.index, hand))
        if hand == "L":
            L = end_st
        else:
            R = end_st
        return _Node(node.cost + cost, L, R, hand, tasks[0].t,
                     tuple(sorted(pending.items())), node,
                     tuple(decisions), tuple(marks))

    # 常规：每手至多一个任务
    for i, v in enumerate(vec):
        task = tasks[i]
        if v is None:
            cost += p["w_unassigned"]
            marks.append(("多押", "绝对", task.index,
                          f"同刻 {len(tasks)} 个需处理对象，无手可用"))
            decisions.append((task.index, "-"))
            continue
        probe = _Node(0.0, L, R, node.last_hand, node.last_t, (), None, (), ())
        c, new, mk = _apply_one(probe, v, task, p, pending)
        if c == math.inf:
            return None
        cost += c
        for kind, level, detail in mk:
            marks.append((kind, level, task.index, detail))
        if v == "L":
            L = new
        else:
            R = new
        decisions.append((task.index, v))
        for f in task.feeds:
            pending[f] = v
        if task.kind in ("slide", "wifi"):
            pending.pop(task.index, None)

    assigned = [v for v in vec if v in ("L", "R")]
    if len(assigned) == 1:
        last_hand, last_t = assigned[0], tasks[0].t
    else:
        last_hand, last_t = "", tasks[0].t
    return _Node(node.cost + cost, L, R, last_hand, last_t,
                 tuple(sorted(pending.items())), node, tuple(decisions), tuple(marks))


# ---------------------------------------------------------------------------
# 3. 主分配（束搜索 Viterbi）
# ---------------------------------------------------------------------------


def _group_tasks(tasks: Sequence[HandTask], tol: float) -> list[list[HandTask]]:
    """把任务按"同刻"分组（时间差 ≤ tol）。"""
    groups: list[list[HandTask]] = []
    for t in tasks:
        if groups and abs(t.t - groups[-1][0].t) <= tol:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def assign(chart_events: ParseResult | Sequence[NoteEvent],
           params: dict | None = None) -> HandAssignment:
    """把一张谱的 note 事件分配给左右手。

    ``chart_events`` 可以是 :class:`ParseResult`，也可以是 ``list[NoteEvent]``。

    返回 :class:`HandAssignment`；其 ``.rows`` 即
    ``[(t, bar, key, note_type, hand, occupancy_until), ...]``。
    """
    p = dict(PARAMS)
    if params:
        p.update(params)
    tasks, notes, n2t = build_tasks(chart_events, p)
    out = HandAssignment(tasks=tasks, n_tasks=len(tasks), params=p)
    if not tasks:
        return out

    groups = _group_tasks(tasks, p["simul_tol"])
    init = _Node(0.0, _HState(None, -1e9), _HState(None, -1e9), "", -1e9, (),
                 None, (), ())
    beam = [init]
    for gi, grp in enumerate(groups):
        nxt: list[_Node] = []
        for node in beam:
            vecs = _enumerate_vectors(grp, p)
            if gi == 0 and p["force_first_hand"] in ("L", "R"):
                vecs = [v for v in vecs if v[0] == p["force_first_hand"]] or vecs
            for vec in vecs:
                nn = _apply_vector(node, grp, vec, p)
                if nn is not None:
                    nxt.append(nn)
        if not nxt:
            # 全部不可行 → 强行推进：记录冲突并重置两手
            best = min(beam, key=lambda n: n.cost)
            marks = (("占用冲突", "绝对", grp[0].index,
                      f"{grp[0].t:.3f}s 同刻 {len(grp)} 个任务"
                      f"（{'/'.join(t.kind for t in grp)}）在任何手法下都不可行"),)
            dec = tuple((t.index, "-") for t in grp)
            nxt = [_Node(best.cost + p["w_unassigned"] * len(grp),
                         _HState(grp[0].key, grp[-1].t_end),
                         _HState(grp[-1].key, grp[-1].t_end),
                         "", grp[0].t, best.pending, best, dec, marks)]
        # 去重 + 剪枝
        seen: dict[tuple, _Node] = {}
        for nn in nxt:
            k = (nn.L.key, round(nn.L.ready, 4), round(nn.L.elastic, 3),
                 nn.R.key, round(nn.R.ready, 4), round(nn.R.elastic, 3),
                 nn.last_hand, nn.pending)
            old = seen.get(k)
            if old is None or nn.cost < old.cost:
                seen[k] = nn
        beam = sorted(seen.values(), key=lambda n: n.cost)[: p["beam"]]

    best = min(beam, key=lambda n: n.cost)
    out.total_cost = best.cost

    # 回溯
    hand_of: dict[int, str] = {}
    all_marks: list[tuple] = []
    step_cost: dict[int, float] = {}
    node = best
    chain: list[_Node] = []
    while node is not None and node.parent is not None:
        chain.append(node)
        node = node.parent
    chain.reverse()
    for nd in chain:
        inc = nd.cost - (nd.parent.cost if nd.parent else 0.0)
        for ti, h in nd.decisions:
            hand_of[ti] = h
            step_cost[ti] = inc / max(1, len(nd.decisions))
        all_marks.extend(nd.marks)

    out.task_hand = [hand_of.get(t.index, "-") for t in tasks]

    # note 级输出（含被并入的拍划）
    rows: list[HandNote] = []
    for i, n in enumerate(notes):
        ti = n2t.get(i)
        if ti is None:
            rows.append(HandNote(n.time, n.measure, n.beat, n.key, n.kind, "-",
                                 n.time, reason="未建模（无键位）"))
            continue
        task = tasks[ti]
        h = hand_of.get(ti, "-")
        merged = ((n.kind == "tap" and i in task.merged_taps)
                  or (n.kind == "slide_star" and i in task.merged_stars))
        rows.append(HandNote(
            t=n.time, measure=n.measure, beat=n.beat, key=n.key, note_type=n.kind,
            hand=h, occupancy_until=task.free if not merged else n.time + p["frame"],
            t_hand=task.t,
            task_kind=task.kind, end_key=task.end_key and str(task.end_key) or "",
            merged=merged,
            reason="拍划并入划动手法（知识 006）" if merged else ""))
    rows.sort(key=lambda r: (r.t, r.key))
    out.notes = rows

    # 标记
    for kind, level, ti, detail in all_marks:
        task = tasks[ti] if 0 <= ti < len(tasks) else None
        hm = HandMuri(kind, level, task.measure if task else 0,
                      task.t if task else 0.0, detail)
        out.muri.append(hm)
        if level == "绝对" and kind in ("多押", "占用冲突", "超速"):
            out.infeasible.append(hm)
    # 侧边双押复核清单（知识 030）：标**是哪一种引导**（八型多标签），不记无理
    # （见 side_double_events）
    out.side_doubles = side_double_events(tasks, out.task_hand, p)
    out.muri.extend(detect_hand_muri(notes, tasks, out.task_hand, p))
    out.muri.sort(key=lambda m: m.time)
    # 无理 vs 蹭的压力（知识 048：软级的蹭多为手法/设计手段，不是无理）
    soft_scrape = {"撞尾", "外键", "路径蹭键", "换手拧巴"}
    out.scrape = [m for m in out.muri if m.level == "软" and m.kind in soft_scrape]
    out.muri = [m for m in out.muri if m not in out.scrape]
    out.bars = bar_metrics(tasks, out.task_hand, out.muri, step_cost, p)
    return out


# ---------------------------------------------------------------------------
# 4. 无理检测（手感知版；知识 008–011）
# ---------------------------------------------------------------------------


_SIDE_DOUBLE_SETS = (frozenset({2, 3}), frozenset({6, 7}))


#: 侧边双押的**引导形态**（知识 030 的用户三型 + 388 官谱普查归纳的五型；
#: 见 `docs/research/side-double-guidance-and-chuzhang.md` §3.1）。
#: 元组顺序 = **主形态**的优先级：用户明说过的三型（A/B、C、D）在前，
#: 普查归纳的五型（G、H、F1、F2、E）在后。
#: 八个标签**全部是结构关系**（共享键 / 相邻键 / 同一对 / 镜像对 / slide 终点 /
#: 被长条压住），**没有任何数值门槛、不分档、不是术语**。
SIDE_GUIDE_TYPES: tuple[str, ...] = ("A/B", "C", "D", "G", "H", "F1", "F2", "E")

#: 八型的一句话形状说明（报告 §3.1 的"形状定义"列）。前三条的出处是用户讲授
#: （知识 030 的 2026-09-19 补充），后五条是 388 官谱普查的 agent 归纳。
SIDE_GUIDE_DESC: dict[str, str] = {
    "A/B": "共享键步进：与阅读窗口里的**上一组双押共享一个键**"
           "（用户例「双押带过来」/ 18·87·76·65·54）",
    "C": "等距轮转：窗口里的**双押子序列**都是相邻键对、锚沿环**等距轮转**"
         "（用户例 18·23·45·67）",
    "D": "同键已在手下：**临近位置**（紧挨着的那一两组）已经落在这两个键之一上"
         "——手还在那儿（用户例「单点铺垫 → 双押串」）",
    "G": "同对回访：窗口里**整对**已经打过一次（这一对刚打过，手是回来不是过去）",
    "H": "星星/长条落点送手：窗口里有 **slide 终点**落在该对键上，或该对键被 **hold 压过**",
    "F1": "镜像对带过来：窗口里出现过它的**镜像对**（2/3 ↔ 6/7）",
    "F2": "同型相邻对带过来：窗口里出现过**别的相邻两键双押**（12/34/45/56/78/18…），"
          "手型是现成的",
    "E": "相邻键铺垫：窗口里有与该对**环距 1** 的键——手就在隔壁",
}

_SIDE_READ_LEVELS = ("拍", "乐句")


def _pair_anchor(keys: frozenset[int]) -> int | None:
    """相邻键对 ``{k, k+1}`` 的锚（逆时针在前的那个键）；非相邻对返回 None。"""
    if len(keys) != 2:
        return None
    a, b = sorted(keys)
    if b - a == 1:
        return a
    if a == 1 and b == 8:
        return 8
    return None


def mirror_pair(keys: frozenset[int]) -> frozenset[int]:
    """键对关于 8–1 / 4–5 轴的**镜像对**：``{2,3} ↔ {6,7}``、``{1,2} ↔ {7,8}``…

    知识 030 的侧边双押只有 ``{2,3}`` 与 ``{6,7}`` 两对，它们互为镜像——
    官谱 F1 型「先写一边、再写另一边」靠的就是这个对称（报告 §3.2 F1）。
    """
    return frozenset(((9 - int(k)) % 8) or 8 for k in keys)


def side_double_events(tasks: Sequence[HandTask], task_hand: Sequence[str],
                       p: dict) -> list[dict]:
    """列出全谱的**侧边双押**（``{2,3}`` / ``{6,7}``）并标注**是哪一种引导**。

    知识 030 的铁律原话只有一句：**禁止突然（＝没有引导）的侧边双押**——
    「除非前期做好了充分多的引导，不然禁止随便写侧边双押」。用户 2026-09-19 给的
    「引导」有三个例子：A「双押带过来」/ B ``18·87·76·65·54``/ C ``18·23·45·67``。

    **v0.4（2026-09-20，388 官谱普查后）改为多标签**：输出 ``guide_types``
    （:data:`SIDE_GUIDE_TYPES` 的子集，一次事件可以同时是好几种），
    ``guided = bool(guide_types or guide_types_phrase)``；**``redline`` 字段整体删除**
    ——那个名字本身在误导（报告 §6.1 建议 1：判据一放宽，"无引导"就从 307 掉到个位数，
    说明问题在 agent 的归纳，不在官谱）。

    八型（形状定义见 :data:`SIDE_GUIDE_DESC`，全部是**结构关系**、零阈值）：

    ====  ==========================  ===========================================
    标签  形态                        出处
    ====  ==========================  ===========================================
    A/B   共享键步进                  用户 2026-09-19（知识 030）
    C     等距轮转                    用户 2026-09-19（知识 030）
    D     同键已在手下                知识 030 原表「单点铺垫 → 双押串」（A 的单点版）
    G     同对回访                    388 官谱普查（报告 §3.2 G）
    H     星星 / 长条落点送手         388 官谱普查（报告 §3.2 H）
    F1    镜像对带过来                388 官谱普查（报告 §3.2 F1）
    F2    同型相邻对带过来            388 官谱普查（报告 §3.2 F2）
    E     相邻键铺垫                  388 官谱普查（报告 §3.2 E）
    ====  ==========================  ===========================================

    **两级阅读窗口**（报告 §6.1 建议 5；两级都是**结构单位**，不是可调门槛）：

    1. **拍级** = 事件之前 ``side_read_slots`` 个任务组（"人读上下文会看的那几拍"）；
       其中 **D 型另看一段更短的"临近位置"**（``side_near_slots`` 个任务组）——
       D 问的是"手**还在不在**这两个键上"，那是紧挨着的一两组的事，不是"最近几拍见过"；
    2. 拍级八型全不命中时再看**乐句级** = **本小节此前 + 前一小节**（D 的"临近"随之放宽）。

    ``guide_types`` 是拍级窗口的命中集；拍级为空时 ``guide_types_phrase`` 给出乐句级
    的命中集。``guide_level`` 记录是在哪一级看出来的（``"拍"`` / ``"乐句"`` / ``""``）。
    `chart_summary` 的 ``side_double_unguided`` **只统计乐句级仍判不出的那些**。

    **A/B 与 C 的"串"按报告 §6.1 建议 3/4 放宽**：

    - **A/B 允许插单点**：看的是**阅读窗口里的上一组双押**，不要求它与本组在同一个
      连续双押串里（`72-みんなのマイマイマー` m042：``1b/2b`` 与 ``2/3`` 共享 2，
      旧实现只因中间隔了 ``8,7,6,5`` 扫键就判不出）；
    - **C 串中一个非相邻对不整串作废**：在双押串里找**最长的等距轮转子段**
      （`141-カゲロウデイズ` Re:MAS m051：``2/3→4/5→6/7→8/1`` 锚等距，
      旧实现只因串尾第 5 组是 ``{5,7}`` 就 ``any(a is None) → continue`` 整串判不出）；
      串的"连续"也改按阅读窗口判定（上一组双押落在本组的拍级窗口内），
      不再用 ``side_guide_gap_beats`` 那个拍数——**双押之间插多少单点都不断串**。
      C 仍然看**整串**（含事件之后的那几组）：用户给的 ``18·23·45·67`` 里，
      ``2/3`` 的引导正是"这一串在等距轮转"这件事本身。

    ⚠️ **"突然"不用数字定义**（用户 2026-09-20：「什么是侧边双押位移大的阈值，不是，
    你怎么能量化来分析这些事情呢？正常写谱谁去特意算这些事情。」）——v0.2 曾用
    "位移 ≥4 格 ∨ 间隔 ≤140 ms" 当"突然"的门，两个阈值已整体删除。

    ⚠️ **输出是复核清单，不是判决**：八型仍是 agent 的归纳（用户只给过三个例子），
    388 官谱上覆盖率高**不等于归纳正确**（报告 §9 局限 2），`assign()` **不**据此记无理。

    ⚠️ **一条普查事实**（报告 §2，记在这里免得后人再去找"长条逼出侧边双押"那条路）：
    388 官谱 685 次侧边双押里，**同刻另有一只手被长条/星星占住的是 0 次**——
    侧边双押本身就把两只手同时占满，任何长条/星星只能在它之前结束或在它之后开始。
    **所以它的引导只能来自前面**，不存在"用长条钉住一只手逼出侧边双押"这种写法。

    事件里的 ``near_dist`` / ``gap`` / ``hand_dist`` / ``run_len`` 是**诊断字段**
    （离临近位置多少格、距上一个任务组多少秒、两手各要飞多远、所在连续双押串多长），
    只为人看清单时定位方便，**不参与任何判定**。
    """
    tol = p["simul_tol"]
    read_slots = max(1, int(p["side_read_slots"]))
    # 1) 同刻双手对（只看击打：tap / hold 头 / 星星头）
    slots: dict[int, dict[str, HandTask]] = {}
    for t, h in zip(tasks, task_hand):
        if h in ("L", "R") and t.key is not None and t.kind in _HIT_KINDS:
            slots.setdefault(int(round(t.t / tol)), {})[h] = t
    pair_slots = [(sorted(d.values(), key=lambda x: x.t)[0].t, d)
                  for _, d in sorted(slots.items()) if "L" in d and "R" in d]
    if not pair_slots:
        return []
    pairs = [(t, frozenset({d["L"].key, d["R"].key}), d) for t, d in pair_slots]
    pair_times = [t for t, _ks, _d in pairs]

    # 2) 连续双押串（按 `side_guide_gap_beats` 的拍数切）——**只作 `run_len` /
    #    `pos_in_run` 两个诊断字段**。A/B 与 C 的判定不再依赖它（报告 §6.1 建议 3/4）。
    runs: list[list[int]] = []
    for i, (t, _ks, d) in enumerate(pairs):
        bpm = d["L"].bpm or d["R"].bpm or 0.0
        lim = p["side_guide_gap_beats"] * 60.0 / bpm if bpm else 0.5
        if runs and 0 < t - pairs[runs[-1][-1]][0] <= lim + 1e-6:
            runs[-1].append(i)
        else:
            runs.append([i])
    run_of = {i: (ri, pos) for ri, r in enumerate(runs) for pos, i in enumerate(r)}

    # 3) 任务组（同刻合并）= **阅读窗口的结构单位**。
    #    每组记：时刻 / 小节 / 组内出现的全部键 / 该组之前两手各自的上一落键。
    g_t: list[float] = []
    g_measure: list[int] = []
    g_keys: list[set[int]] = []
    g_prev: list[tuple[int | None, int | None]] = []
    prev_key: dict[str, int | None] = {"L": None, "R": None}
    cur_t = -1e9
    pend: list[tuple[str, int]] = []
    for i in sorted(range(len(tasks)), key=lambda x: tasks[x].t):
        t, h = tasks[i], task_hand[i]
        if t.t > cur_t + tol:
            for hh, k in pend:
                prev_key[hh] = k
            pend = []
            cur_t = t.t
            g_t.append(t.t)
            g_measure.append(t.measure)
            g_keys.append(set())
            g_prev.append((prev_key["L"], prev_key["R"]))
        for hh in (("L", "R") if h == "LR" else ([h] if h in ("L", "R") else [])):
            if t.key is not None:
                pend.append((hh, int(t.key)))
        if t.key is not None:
            g_keys[-1].add(int(t.key))

    #: 每组双押所在的任务组下标（C 型的"串"按阅读窗口接续用）
    pair_group = [min(max(bisect.bisect_left(g_t, t - tol), 0), len(g_t) - 1)
                  for t in pair_times]

    # 4) C 型：**双押子序列**上的等距轮转子段（串中一个非相邻对不整串作废）。
    #    串的接续判据 = 上一组双押落在本组的**拍级阅读窗口**内（中间插多少单点都不断串）。
    rotation: set[int] = set()
    chain: list[int] = []
    for j in range(len(pairs)):
        if chain and pair_group[j] - pair_group[chain[-1]] <= read_slots:
            chain.append(j)
        else:
            _mark_rotation(chain, pairs, rotation, p["side_rotation_min_run"])
            chain = [j]
    _mark_rotation(chain, pairs, rotation, p["side_rotation_min_run"])

    # 5) H 型的两份素材：slide **终点**到达时刻 + hold **压住**的区间。
    #    （知识 014：轨道任务的 `t` 是启动拍，终点用名义到达时刻。）
    slide_ends: list[tuple[float, frozenset[int]]] = []
    hold_spans: list[tuple[float, float, int]] = []
    for t in tasks:
        if t.kind in ("slide", "wifi") and t.end_key is not None:
            ke = (frozenset(wifi_ends(t.end_key)) if t.kind == "wifi"
                  else frozenset({int(t.end_key)}))
            slide_ends.append((t.t_end_nominal or t.t_end, ke))
        elif t.kind == "hold" and t.key is not None:
            hold_spans.append((t.t, t.t_end, int(t.key)))

    def _window_facts(lo: int, hi: int, near_lo: int, ev_t: float) -> tuple:
        """把窗口 ``groups[lo:hi]`` 摊成判型要用的四件事实。

        ``near_lo`` 划出**"临近位置"**那一小段（``groups[near_lo:hi]``）——只有 D 型
        与 ``near_dist`` 诊断看它，其余七型看整个阅读窗口。
        """
        t_lo = g_t[lo] if hi > lo else ev_t
        keys: set[int] = set()
        near: set[int] = set()
        for gi in range(lo, hi):
            keys |= g_keys[gi]
            if gi >= near_lo:
                near |= g_keys[gi]
        j0 = bisect.bisect_left(pair_times, t_lo - tol)
        j1 = bisect.bisect_left(pair_times, ev_t - tol)
        wpairs = [pairs[j][1] for j in range(j0, j1)]
        sent: set[int] = set()
        for te, ke in slide_ends:
            if t_lo - tol <= te <= ev_t + tol:
                sent |= ke
        for th, te, k in hold_spans:
            if th < ev_t + tol and te >= t_lo - tol:
                sent.add(k)
        return keys, wpairs, sent, near

    def _guides(ks: frozenset[int], facts: tuple, is_rot: bool) -> list[str]:
        """八型判定：全部是形状关系，**没有任何数值比较**。"""
        keys, wpairs, sent, near = facts
        mir = mirror_pair(ks)
        hit: list[str] = []
        # A/B 共享键步进：窗口里的**上一组双押**与本组共享一个键
        if wpairs and (wpairs[-1] & ks):
            hit.append("A/B")
        # C 等距轮转：本组落在一段等距轮转子段里（且不是子段的头）
        if is_rot:
            hit.append("C")
        # D 同键已在手下：**临近位置**已经落在这两个键之一上（手还在那儿）
        if ks & near:
            hit.append("D")
        # G 同对回访：窗口里**整对**打过
        if any(w == ks for w in wpairs):
            hit.append("G")
        # H 星星/长条落点送手：slide 终点落在该对键上，或该对键被 hold 压过
        if sent & ks:
            hit.append("H")
        # F1 镜像对带过来
        if any(w == mir for w in wpairs):
            hit.append("F1")
        # F2 同型相邻对带过来：别的**相邻两键**双押（手型是现成的）
        if any(_pair_anchor(w) is not None and w != ks and w != mir for w in wpairs):
            hit.append("F2")
        # E 相邻键铺垫：手就在隔壁（环距 1）
        if any(cdist(k, m) == 1 for k in ks for m in keys):
            hit.append("E")
        return hit

    out: list[dict] = []
    for i, (t, ks, d) in enumerate(pairs):
        if ks not in _SIDE_DOUBLE_SETS:
            continue
        ri, pos = run_of[i]
        hi = pair_group[i]                            # 本事件所在的任务组
        lo_beat = max(0, hi - read_slots)             # 拍级窗口
        m_ev = g_measure[hi]                          # 乐句级窗口
        lo_phrase = hi
        while lo_phrase > 0 and g_measure[lo_phrase - 1] >= m_ev - 1:
            lo_phrase -= 1

        near_lo = max(lo_beat, hi - max(1, int(p["side_near_slots"])))
        f_beat = _window_facts(lo_beat, hi, near_lo, t)
        types = _guides(ks, f_beat, i in rotation)
        types_phrase: list[str] = []
        level = _SIDE_READ_LEVELS[0] if types else ""
        if not types:
            # 乐句级：窗口整体放宽到"本小节此前 + 前一小节"，"临近"也随之放宽
            types_phrase = _guides(
                ks, _window_facts(lo_phrase, hi, lo_phrase, t), False)
            level = _SIDE_READ_LEVELS[1] if types_phrase else ""
        main = (types or types_phrase or [""])[0]

        # ---- 以下全是诊断字段，不参与判定 ----
        pl, pr = g_prev[hi]
        dl = cdist(pl, d["L"].key) if pl is not None else 4
        dr = cdist(pr, d["R"].key) if pr is not None else 4
        nk = f_beat[3]          # 诊断：离"临近位置"多少格（同 D 型的那一小段窗口）
        near_dist = (min(cdist(k, m) for k in ks for m in nk) if nk else 0)
        gap = t - g_t[hi - 1] if hi > 0 else 1e9
        out.append({
            "measure": d["L"].measure, "time": t,
            "keys": sorted(ks), "L": d["L"].key, "R": d["R"].key,
            # ↓ 纯诊断字段，不参与判定
            "hand_dist": max(dl, dr), "disp_l": dl, "disp_r": dr,
            "near_dist": near_dist,
            "gap": round(gap, 4) if gap < 1e8 else None,
            "run_len": len(runs[ri]), "pos_in_run": pos,
            # ↓ 判定：是哪一种（或哪几种）引导
            "guided": bool(types or types_phrase),
            "guide_types": types,
            "guide_types_phrase": types_phrase,
            "guide_type": main,
            "guide_level": level,
        })
    return out


def _mark_rotation(chain: Sequence[int], pairs: Sequence[tuple],
                   rotation: set[int], min_run: int) -> None:
    """在一条双押串里找出**所有等距轮转子段**，把子段里除串头外的成员记进 ``rotation``。

    报告 §6.1 建议 4②：旧实现 ``any(anchor is None) → continue`` 会让
    "串尾多出一个非相邻对"作废整串（`141-カゲロウデイズ` Re:MAS m051）。
    这里改成：先按"是不是相邻键对"把串切成连续块，再在块内切成**等步长**的子段，
    子段长度 ≥ ``min_run``（"几个双押才算一段轮转"的计数口径，不是难度阈值）时
    把它的第 2 个及以后的成员记为有 C 型引导——**串首没有"前面的引导"，不算**。
    """
    if len(chain) < min_run:
        return
    anchors = [_pair_anchor(pairs[j][1]) for j in chain]
    b = 0
    while b < len(chain):
        if anchors[b] is None:
            b += 1
            continue
        e = b
        while e + 1 < len(chain) and anchors[e + 1] is not None:
            e += 1
        # 块 [b, e] 全是相邻键对；在块内按"步长是否改变"切子段
        s0 = b
        while s0 < e:
            step = cstep(anchors[s0], anchors[s0 + 1])
            s1 = s0 + 1
            if step != 0:
                while (s1 + 1 <= e
                       and cstep(anchors[s1], anchors[s1 + 1]) == step):
                    s1 += 1
                if s1 - s0 + 1 >= min_run:
                    rotation.update(chain[s0 + 1:s1 + 1])
            s0 = s1
        b = e + 1


def detect_hand_muri(notes: Sequence[NoteEvent], tasks: Sequence[HandTask],
                     task_hand: Sequence[str], p: dict) -> list[HandMuri]:
    """叠键 / 外键 / 撞尾 / 路径蹭键 / 两手交叉。

    与 ``configs.detect_muri`` 的区别：本函数**知道每个 note 归哪只手**，
    所以能排除"由划轨手自己顺手处理"的假阳性，并按知识 010/011 的分级给
    软 / 硬 / 绝对三档。
    """
    out: list[HandMuri] = []
    hits = [n for n in notes if n.kind in ("tap", "hold", "slide_star") and n.key.isdigit()]

    # 叠键（知识 009）：同判定区 < 2 帧 → 绝对无理
    by_key: dict[str, list[NoteEvent]] = {}
    for n in hits:
        by_key.setdefault(n.key, []).append(n)
    for k, ns in by_key.items():
        ns.sort(key=lambda x: x.time)
        for a, b in zip(ns, ns[1:]):
            dt = b.time - a.time
            if 0 < dt < p["muri_overlap_sec"]:
                out.append(HandMuri("叠键", "绝对", a.measure, a.time,
                                    f"键 {k} 间隔 {dt * 1000:.1f} ms"))

    hand_of_note: dict[int, str] = {}
    for t, h in zip(tasks, task_hand):
        if t.note_idx >= 0:
            hand_of_note[t.note_idx] = h
        for j in t.merged_taps:
            hand_of_note[j] = h
    idx_of = {id(n): i for i, n in enumerate(notes)}

    for task in tasks:
        if task.kind not in ("slide", "wifi") or task.key is None:
            continue
        swiper = task_hand[task.index] if task.index < len(task_hand) else "-"
        t_move = task.t
        t_end = task.t_end_nominal or task.t_end   # 撞尾按名义到达时刻算
        t_star = (notes[tasks[task.star_idx].note_idx].time
                  if task.star_idx >= 0 and tasks[task.star_idx].note_idx >= 0
                  else (notes[task.note_idx].time if task.note_idx >= 0 else t_move))
        head = task.key
        ends = [task.end_key] if task.end_key else []
        if task.kind == "wifi" and task.end_key:
            ends = list(wifi_ends(task.end_key))
        own_star = tasks[task.star_idx].note_idx if task.star_idx >= 0 else -1
        for n in hits:
            ni = idx_of.get(id(n))
            if ni is not None and ni == own_star:
                continue  # 本条 slide 自己的星星头不算
            nh = hand_of_note.get(ni, "-")
            nk = int(n.key)
            # ---- 外键（知识 010）----
            # 锚点：知识 010 原文里"slide 启动后"与"slide 头后"两种说法并存，
            # 物理上划动手从**击打星星头**起就守在头键附近（启动拍等待期），
            # 故默认锚在星星头 (`slidehead_anchor="star"`)，见报告 §3 校准。
            anchor = t_star if p["slidehead_anchor"] == "star" else t_move
            gap = n.time - anchor
            same_side = (cdist(nk, head) <= p["muri_slidehead_ring"]
                         and (not p["slidehead_same_page"]
                              or home_side(nk) == home_side(head)))
            if (-1e-6 < gap <= p["muri_slidehead_sec"] and same_side
                    and nh != swiper and nh != "-"):
                if n.is_ex:
                    level = "软"
                elif gap >= p["muri_slidehead_warn_sec"]:
                    level = "软"
                elif gap <= p["muri_slidehead_near_sec"]:
                    level = "软"
                else:
                    level = "硬"
                out.append(HandMuri("外键", level, n.measure, n.time,
                                    f"slide {head}{task.shape}{task.end_key} 启动后 "
                                    f"{gap * 1000:.0f} ms，{nh} 手在同侧键 {nk}"))
            # ---- 撞尾（知识 011）----
            for ek in ends:
                if nk != ek:
                    continue
                d = n.time - t_end
                if not (-p["muri_tail_pre_sec"] <= d <= p["muri_tail_post_sec"]):
                    continue
                if nh == swiper:
                    continue
                eighth = (30.0 / task.bpm) if task.bpm else 0.0
                tolerant = (eighth > 0 and abs(d - eighth) <= p["tail_eighth_tol"]
                            and task.bpm <= p["tail_bpm_tolerant"])
                if abs(d) < 1e-6:
                    level = "绝对"
                elif 0 < d < p["muri_tail_hard_sec"] and not tolerant:
                    level = "软" if n.is_ex else "硬"
                else:
                    level = "软"
                out.append(HandMuri("撞尾", level, n.measure, n.time,
                                    f"slide {head}{task.shape}{ek} 尾 {d * 1000:+.0f} ms "
                                    f"落 {nh} 手键 {nk}"))
            # ---- 路径蹭键（agent 推断，存疑）----
            if not p["path_brush"] or task.kind == "wifi" or not task.end_key:
                continue
            dur = max(1e-6, t_end - t_move)
            for frac, area in slide_path_areas(task.shape, head, task.end_key):
                if nk != area or nh == swiper or nh == "-":
                    continue
                tc = t_move + frac * dur
                if abs(n.time - tc) <= p["path_brush_win"]:
                    out.append(HandMuri("路径蹭键", "软", n.measure, n.time,
                                        f"slide {head}{task.shape}{task.end_key} 途经 A{area} "
                                        f"（{(n.time - tc) * 1000:+.0f} ms），{nh} 手键 {nk}"))

    # Hold 尾多押（知识 012："Hold 尾判 + 同时双押 = 标准多押"）
    tol = p["simul_tol"]
    slot_n: dict[float, int] = {}
    for t in tasks:
        if t.kind in ("tap", "hold", "star", "slide", "wifi"):
            slot_n[round(t.t, 4)] = slot_n.get(round(t.t, 4), 0) + 1
    for n in notes:
        if n.kind != "hold" or not n.key.isdigit():
            continue
        nominal = n.time + n.duration          # 名义尾（不含 hold_release_slack）
        for st, cnt in slot_n.items():
            if cnt >= 2 and abs(st - nominal) <= p["hold_tail_win"]:
                out.append(HandMuri("Hold尾多押", "硬", n.measure, nominal,
                                    f"键 {n.key} 的 Hold 尾（{nominal:.3f}s）撞上同刻 "
                                    f"{cnt} 个 note，需要三只手（知识 012）"))
                break

    # 两手交叉（拓扑；agent 推断）
    slotmap: dict[int, dict[str, HandTask]] = {}
    for t, h in zip(tasks, task_hand):
        if h in ("L", "R") and t.key is not None:
            slotmap.setdefault(int(round(t.t / tol)), {})[h] = t
    slide_ts = sorted(t.t_end for t in tasks if t.kind in ("slide", "wifi"))
    import bisect
    for slot in sorted(slotmap):
        d = slotmap[slot]
        if "L" not in d or "R" not in d:
            continue
        if any(x.kind in ("slide", "wifi", "star") for x in d.values()):
            continue
        lk, rk = d["L"].key, d["R"].key
        if home_side(lk) != "R" or home_side(rk) != "L":
            continue
        # 知识 007 定式 4："一对贴边对称双押 slide 打完立即双手换位划" → 刚划完的交叉不算拧巴
        i = bisect.bisect_left(slide_ts, d["L"].t - p["cross_recent_slide_sec"])
        if i < len(slide_ts) and slide_ts[i] <= d["L"].t + 1e-6:
            continue
        dep = page_depth("L", lk) + page_depth("R", rk)
        if dep >= 3:
            out.append(HandMuri("换手拧巴", "软", d["L"].measure, d["L"].t,
                                f"左手在 {lk}、右手在 {rk}（交叉深度 {dep}）"))

    # ⚠️ 知识 030 的「突然（＝无引导）的侧边双押」**不在这里记无理**：
    #    "引导"的八型（A/B·C·D 来自用户三个例子，G·H·F1·F2·E 来自 388 官谱普查）
    #    仍是 agent 的归纳——覆盖率高不等于归纳正确，把它当自动判决等于替用户下结论。
    #    无引导（＝**乐句级窗口仍判不出**）的侧边双押以**复核清单**的形式出在
    #    `HandAssignment.side_doubles`（`side_double_events()`），由人裁定。
    return out


# ---------------------------------------------------------------------------
# 5. 逐小节指标
# ---------------------------------------------------------------------------


def bar_metrics(tasks: Sequence[HandTask], task_hand: Sequence[str],
                muri: Sequence[HandMuri], step_cost: dict[int, float],
                p: dict) -> list[BarHand]:
    """逐小节手序指标与手序难度分。"""
    if not tasks:
        return []
    lo = min(t.measure for t in tasks)
    hi = max(t.measure for t in tasks)
    bars = {m: BarHand(measure=m) for m in range(lo, hi + 1)}
    prev: dict[str, HandTask] = {}
    prev_hand = ""
    run = 0
    for t, h in zip(tasks, task_hand):
        b = bars[t.measure]
        b.n_tasks += 1
        b.cost += step_cost.get(t.index, 0.0)
        if h == "-":
            b.conflicts += 1
            continue
        for hh in (("L", "R") if h == "LR" else (h,)):
            if hh == "L":
                b.n_left += 1
            else:
                b.n_right += 1
            if page_depth(hh, t.key) >= 1:
                b.cross_count += 1
            # ★ 出张（知识 064 用户口径）：**右手落 6/7 或左手落 2/3**，只有这四个组合。
            #   旧口径（"划轨手不在 slide 末尾半边"，知识 007）已被 064/065 作废，
            #   仍并列输出在 `chuzhang_end_side` 里供新旧对照。
            if is_chuzhang(hh, t.key):
                b.chuzhang += 1
            if t.kind in ("slide", "wifi"):
                b.n_slides += 1
                if is_chuzhang(hh, t.key) or is_chuzhang(hh, t.end_key):
                    b.chuzhang_slide += 1
                if t.end_key is not None and home_side(t.end_key) != hh:
                    b.chuzhang_end_side += 1
            pv = prev.get(hh)
            if pv is not None and pv.key is not None and t.key is not None:
                dt = t.t - pv.t_end
                if dt > 1e-6:
                    v = cdist(pv.key, t.key) / dt
                    if hh == "L":
                        b.max_speed_l = max(b.max_speed_l, v)
                    else:
                        b.max_speed_r = max(b.max_speed_r, v)
            prev[hh] = t
        if h == prev_hand:
            run += 1
        else:
            run = 1
            if prev_hand:
                b.alt_rate += 1  # 临时计数：换手次数
        prev_hand = h if h in ("L", "R") else ""
        b.max_same_run = max(b.max_same_run, run)
    for b in bars.values():
        n_seq = max(1, b.n_tasks - 1)
        b.alt_rate = min(1.0, b.alt_rate / n_seq)
    for m in muri:
        b = bars.get(m.measure)
        if b is not None:
            key = f"{m.kind}:{m.level}"
            b.muri[key] = b.muri.get(key, 0) + 1
    vals = []
    for b in bars.values():
        b.hardness_raw = b.cost / b.n_tasks if b.n_tasks else 0.0
        if b.n_tasks:
            vals.append(b.hardness_raw)
    if vals:
        mn, mx = min(vals), max(vals)
        rng = (mx - mn) or 1.0
        for b in bars.values():
            b.hardness = (b.hardness_raw - mn) / rng if b.n_tasks else 0.0
    return [bars[m] for m in range(lo, hi + 1)]


def bar_hand_sequences(ha: HandAssignment) -> dict[int, dict[str, list]]:
    """逐小节的左右手键位/时间序列与同刻双手对（给按手识别配置用）。

    ``{小节号: {"L": [(t, key, kind), ...], "R": [...], "pairs": [(t, Lkey, Rkey)]}}``
    """
    out: dict[int, dict[str, list]] = {}
    for t, h in zip(ha.tasks, ha.task_hand):
        d = out.setdefault(t.measure, {"L": [], "R": [], "pairs": []})
        for hh in (("L", "R") if h == "LR" else ([h] if h in ("L", "R") else [])):
            d[hh].append((t.t, t.key, t.kind))
    for t, m, lk, rk in ha.pairs:
        out.setdefault(m, {"L": [], "R": [], "pairs": []})["pairs"].append((t, lk, rk))
    return out


def chart_summary(ha: HandAssignment) -> dict:
    """全谱汇总（给 CSV / 批量分析用）。"""
    bars = [b for b in ha.bars if b.n_tasks]
    n = len(bars) or 1
    counts = ha.counts
    lvl: dict[str, int] = {}
    for m in ha.muri:
        lvl[m.level] = lvl.get(m.level, 0) + 1
    return {
        "n_tasks": ha.n_tasks,
        "n_bars": len(bars),
        "total_cost": round(ha.total_cost, 3),
        "hand_hardness": round(ha.hand_hardness, 5),
        "alt_rate": round(sum(b.alt_rate for b in bars) / n, 4),
        "max_same_run": max((b.max_same_run for b in bars), default=0),
        "cross_rate": round(sum(b.cross_count for b in bars)
                            / max(1, sum(b.n_tasks for b in bars)), 4),
        "max_speed": round(max((max(b.max_speed_l, b.max_speed_r) for b in bars),
                               default=0.0), 3),
        # ★ 出张（知识 064 用户口径）：右手落 6/7 / 左手落 2/3 的落点数
        "chuzhang": sum(b.chuzhang for b in bars),
        "chuzhang_per_1k": round(1000 * sum(b.chuzhang for b in bars)
                                 / max(1, ha.n_tasks), 3),
        "chuzhang_slide": sum(b.chuzhang_slide for b in bars),
        "chuzhang_per_slide": round(
            sum(b.chuzhang_slide for b in bars)
            / max(1, sum(1 for t in ha.tasks if t.kind in ("slide", "wifi"))), 4),
        # v0.1 旧口径（划轨手不在 slide 末尾半边，知识 007；已被 064/065 作废）
        "chuzhang_end_side": sum(b.chuzhang_end_side for b in bars),
        "chuzhang_end_side_per_slide": round(
            sum(b.chuzhang_end_side for b in bars)
            / max(1, sum(1 for t in ha.tasks if t.kind in ("slide", "wifi"))), 4),
        # 侧边双押复核清单（知识 030）：总数与"没有引导、请人看一眼"的数量。
        # `side_double_unguided` 只数**乐句级窗口仍判不出**的那些（拍级判不出、
        # 乐句级判得出的不计）。**不是判决**——见 `side_double_events()` 的说明。
        "side_double": len(ha.side_doubles),
        "side_double_unguided": sum(1 for e in ha.side_doubles if not e["guided"]),
        # 八型引导的分布（主形态计数；一次事件可命中多型，这里只数主形态）
        **{f"side_guide_{g}": sum(1 for e in ha.side_doubles
                                  if e["guide_type"] == g)
           for g in SIDE_GUIDE_TYPES},
        "scrape_pressure": len(ha.scrape),
        "scrape_per_1k": round(1000 * len(ha.scrape) / max(1, ha.n_tasks), 3),
        "n_infeasible": len(ha.infeasible),
        "n_muri": len(ha.muri),
        "muri_绝对": lvl.get("绝对", 0),
        "muri_硬": lvl.get("硬", 0),
        "muri_软": lvl.get("软", 0),
        **{f"muri_{k}": v for k, v in counts.items()},
        **{f"scrape_{k}": v for k, v in ha.scrape_counts.items()},
    }


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def _find_chart(pattern: str):
    from pathlib import Path

    pth = Path(pattern)
    if pth.exists() and pth.is_file():
        return pth.stem, pth.read_text(encoding="utf-8", errors="replace")
    try:
        from . import corpus
    except ImportError:  # pragma: no cover
        from chart_analysis import corpus  # type: ignore
    hits = [c for c in corpus.discover() if pattern.lower() in c.name.lower()]
    if not hits:
        raise SystemExit(f"未找到匹配 {pattern!r} 的谱面")
    if len(hits) > 1:
        print(f"[多个匹配，取第一个] " + " / ".join(h.name for h in hits[:5]))
    return hits[0].name, hits[0].read()


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="左右手分配器（手序合理性）")
    ap.add_argument("chart", help="谱面文件路径，或官方谱文件名关键字")
    ap.add_argument("--bars", default="", help="只看某段小节，如 20-24")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--muri-only", action="store_true", help="只列无理标记")
    args = ap.parse_args(argv)

    name, text = _find_chart(args.chart)
    res = parse_chart(text, name=name)
    ha = assign(res)

    lo, hi = -1, 10 ** 9
    if args.bars:
        a, _, b = args.bars.partition("-")
        lo = int(a)
        hi = int(b) if b else lo

    if args.json:
        data = chart_summary(ha)
        data["name"] = name
        data["bars"] = [b.to_dict() for b in ha.bars if lo <= b.measure <= hi]
        data["notes"] = [
            {"t": round(n.t, 4), "bar": n.measure, "key": n.key,
             "type": n.note_type, "hand": n.hand,
             "until": round(n.occupancy_until, 4), "merged": n.merged}
            for n in ha.notes if lo <= n.measure <= hi]
        data["muri"] = [m.to_dict() for m in ha.muri if lo <= m.measure <= hi]
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return 0

    s = chart_summary(ha)
    print(f"谱面：{name}")
    print(f"任务数 {s['n_tasks']}｜小节 {s['n_bars']}｜总代价 {s['total_cost']}"
          f"｜手序难度 {s['hand_hardness']}")
    print(f"交替率 {s['alt_rate']}｜同手最长连打 {s['max_same_run']}"
          f"｜跨半圈比例 {s['cross_rate']}｜最大等效速度 {s['max_speed']} 键/秒")
    print(f"不可行 {s['n_infeasible']}｜无理标记 {s['n_muri']}"
          f"（绝对 {s['muri_绝对']} / 硬 {s['muri_硬']} / 软 {s['muri_软']}）")
    kinds = {k[5:]: v for k, v in s.items()
             if k.startswith("muri_") and k[5:] not in ("绝对", "硬", "软")}
    if kinds:
        print("  分类：" + "  ".join(f"{k} {v}" for k, v in sorted(kinds.items())))

    if not args.muri_only:
        print("\n时间(s) | 小节 | note        | 手")
        for n in ha.notes:
            if not (lo <= n.measure <= hi):
                continue
            tag = f"{n.key}{'→' + n.end_key if n.end_key else ''}"
            mk = "（拍划并入）" if n.merged else ""
            ts = f"{n.t:8.3f}" if abs(n.t_hand - n.t) < 1e-6 else f"{n.t:8.3f}→{n.t_hand:.3f}"
            print(f"{ts:>17} | {n.measure:4d} | {n.note_type:<11} {tag:<6}"
                  f"| {n.hand}{mk}")
        print("\n小节 | 任务 | L/R  | 交替率 | 同手连打 | 跨半圈 | 冲突 | 手序难度")
        for b in ha.bars:
            if not (lo <= b.measure <= hi) or not b.n_tasks:
                continue
            print(f"{b.measure:4d} | {b.n_tasks:4d} | {b.n_left:2d}/{b.n_right:<2d} "
                  f"| {b.alt_rate:6.2f} | {b.max_same_run:8d} | {b.cross_count:6d} "
                  f"| {b.conflicts:4d} | {b.hardness:.3f}")

    ms = [m for m in ha.muri if lo <= m.measure <= hi]
    if ms:
        print(f"\n无理标记（{len(ms)}）")
        for m in ms[:200]:
            print(f"  m{m.measure:04d} {m.time:8.3f}s [{m.kind}/{m.level}] {m.detail}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
