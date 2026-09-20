#!/usr/bin/env python3
"""深度层：「**这张谱写得对不对得起它的定数**」的复核清单（知识 093）。

为什么要有这一层
----------------
`docs/research/e2e-trial-01.md` 的缺口 **G6**：那一轮试写终稿 **0 错误 0 警告**、
note 总数与 NPS 都落在定数区间内，但手序难度只有官谱的 41%、slide 只有 28%、
hold 只有 8%——**五层里没有任何一条提示指向"写浅了"**。

本层不新增任何判据，只做一件事：**把已经算好的三样东西放到同定数档官谱的分布上比一比**——

1. **手序难度**（`hands.py` 的内部诊断量）对照 388 谱同定数档的中位与四分位；
2. **note 种类占比**（slide / hold / each）对照知识 088 的谱级分布（密度层已逐项报过，
   这里只取"落在 Q1 以下"的那几项合成一句结论）；
3. **主料构成**（知识 073）——主料里有没有**星星族**、有没有把**几何族**写成主料。

⚠️ 三条铁律（AGENT.md 准则 13 / 知识 064 / 066 / 078）：

- 本层**全部是提示，不判对错**，也**不给合成分**；
- **手序难度不是难度代理**（知识 078：它与配置硬度正交，最强共变量是出张数）——
  它只说明"这张谱是不是异常地省手"，**不能当目标去优化**，优化它会直接撞上
  知识 064「出张轻易不要写」与 066「不以特殊处理的手法为导向」；
- 官谱自己也有四分之一落在各项的 Q1 以下，**落在低位完全可以有正当理由**，写明理由即可。
"""

from __future__ import annotations

from .model import LayerResult

#: 知识 093：388 官谱 `hand_hardness`（`hands.py` 口径）按定数档的中位 / Q1 / Q3。
#: 档 = `floor(定数×2)/2`，与密度层同口径。**只作对照，不作判据。**
LEVEL_HAND_HARDNESS = {
    13.0: {"n": 168, "q": (0.110, 0.095, 0.128)},
    13.5: {"n": 135, "q": (0.114, 0.093, 0.134)},
    14.0: {"n": 73, "q": (0.136, 0.110, 0.155)},
    14.5: {"n": 12, "q": (0.148, 0.120, 0.162)},
}

#: 知识 073/093：388 官谱里**主料中含星星族**的谱占比（定数档 → 比例）。
LEVEL_STAR_MAIN_RATE = {13.0: 0.61, 13.5: 0.68, 14.0: 0.67, 14.5: 0.75}
#: 知识 073/093：**主料中含几何族**的谱占比——官谱里几何只作点缀。
LEVEL_GEO_MAIN_RATE = {13.0: 0.09, 13.5: 0.13, 14.0: 0.19, 14.5: 0.42}
#: ⚠️ 14.5 档只有 12 张谱，比例只当量级看。

#: 星星族（知识 049–063 + 017 错位 + 033 死镰段），与知识 090 同一份名单
STAR_FAMILY = (
    "错位", "连续拍滑", "双压连续拍滑", "CYCLES型星星", "夹键拍滑", "穿心", "大风车",
    "绕圈星星", "一笔画", "挥手段", "井字星星", "同起点拍滑", "三叉戟", "鼓动段",
    "拆弹", "死镰段",
)
#: 几何族（与 `configlayer.GEOMETRIC` 同一份名单）
GEOMETRIC = ("三角交互", "楼梯交互", "大宇宙", "轴交互", "散点", "方向盘", "逆楼梯")


def _nearest(bucket: float | None, table: dict):
    if bucket is None:
        return None
    return min(table, key=lambda k: abs(k - float(bucket)))


def check_depth(report) -> LayerResult:
    """跑在其余五层之后，只消费它们的 `stats`。"""
    out = LayerResult(layer="深度")
    play = report.layer("手序")
    dens = report.layer("密度")
    cfg = report.layer("配置")
    if dens is None or dens.skipped:
        out.skipped = "密度层没跑 → 深度对照跳过"
        return out
    bucket = dens.stats.get("level_bucket")
    if bucket is None:
        out.skipped = "没有定数（--level 或 &lv_N）→ 没法对照同定数档官谱"
        return out

    flags: list[str] = []
    stats: dict = {"level_bucket": bucket}

    # ---- ① 手序难度 ----
    key = _nearest(bucket, LEVEL_HAND_HARDNESS)
    hh = (play.stats.get("hand_hardness") if play and not play.skipped else None)
    if hh is not None and key is not None:
        med, q1, q3 = LEVEL_HAND_HARDNESS[key]["q"]
        where = ("低于 Q1" if hh < q1 else "高于 Q3" if hh > q3 else "在 Q1–Q3 内")
        stats["hand_hardness"] = hh
        stats["hand_hardness_official"] = LEVEL_HAND_HARDNESS[key]["q"]
        out.add("提示", "DEP-HAND-HARDNESS",
                f"手序难度 {hh:.3f}（官谱定数 {key} 档 {med:.3f} "
                f"[{q1:.3f}–{q3:.3f}]，n={LEVEL_HAND_HARDNESS[key]['n']}，{where}）。"
                "⚠️ 知识 078：**手序难度不是难度代理**，它与配置硬度正交、"
                "最强共变量是出张数——只能读作「这张谱省不省手」，"
                "**不得当作目标去优化**（知识 064：出张轻易不要写 / 066：不以手法为导向）。",
                source="知识 078 / 093")
        if hh < q1:
            flags.append(f"手序难度 {hh:.3f} < 官谱 Q1 {q1:.3f}")

    # ---- ② note 种类占比 ----
    mix = dens.stats.get("note_mix") or {}
    ref = dens.stats.get("note_mix_official") or {}
    low = []
    for k, cn in (("slide", "slide 轨"), ("hold", "hold"), ("each", "each note")):
        if k in mix and k in ref:
            if mix[k] < ref[k][1]:
                low.append(f"{cn} {mix[k]:.1%} < Q1 {ref[k][1]:.1%}")
    stats["note_mix_below_q1"] = low
    if low:
        flags.extend(low)

    # ---- ③ 主料构成 ----
    main = list((cfg.stats.get("main") or {}) ) if cfg and not cfg.skipped else []
    star_main = [m for m in main if m in STAR_FAMILY]
    geo_main = [m for m in main if any(g in m for g in GEOMETRIC)]
    kstar = _nearest(bucket, LEVEL_STAR_MAIN_RATE)
    stats["main"] = main
    stats["star_main"] = star_main
    stats["geometric_main"] = geo_main
    if main:
        out.add("提示", "DEP-MAIN-MIX",
                f"主料 {len(main)} 类：{'、'.join(main[:8])}"
                + (f"；其中星星族 {('、'.join(star_main))}" if star_main
                   else "；**一类星星族都没有**")
                + (f"；几何族被写成主料：{('、'.join(geo_main))}" if geo_main else "")
                + f"。官谱定数 {kstar} 档里 **{LEVEL_STAR_MAIN_RATE[kstar]:.0%} 的谱**"
                  f"有星星族主料、只有 {LEVEL_GEO_MAIN_RATE[kstar]:.0%} 的谱"
                  "把几何族写成主料（知识 073 / 093）。",
                source="知识 073 / 093")
        if not star_main:
            flags.append("主料里一类星星族都没有")
        if geo_main:
            flags.append("几何族被写成主料：" + "、".join(geo_main))

    # ---- 合成一句「写得太浅」的复核提示（**只提示，不判错**） ----
    if len(flags) >= 2:
        out.add("提示", "DEP-SHALLOW",
                "**写得太浅？复核清单**（同时踩到 "
                f"{len(flags)} 条与同定数档官谱的差距）：" + "；".join(flags)
                + "。⚠️ 这**不是错误**——官谱自己也有四分之一落在各项的 Q1 以下"
                "（知识 093）；但 `e2e-trial-01` 的教训是"
                "「0 错误 0 警告 + note 总数进区间」完全拦不住写浅，"
                "所以这几条一起出现时请回头看一眼，**偏离要能说出理由**。",
                source="知识 093")
    elif flags:
        out.add("提示", "DEP-SHALLOW-1",
                "与同定数档官谱的差距（只有 1 条，通常不必动）：" + "；".join(flags),
                source="知识 093")
    else:
        out.add("提示", "DEP-OK",
                "手序难度 / note 种类占比 / 主料构成 三项都没有明显低于同定数档官谱的分布"
                "——**这只说明没有触发复核清单，不说明谱写得好**（知识 093）。",
                source="知识 093")

    stats["flags"] = flags
    out.stats = stats
    return out
