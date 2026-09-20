#!/usr/bin/env python3
"""配置层：这张谱**用了哪些配置、按什么顺序用、谁是主料谁是点缀**。

判定全部来自 `chart_analysis.configs_hand`（手序版，AGENT.md 准则 12：
任何配置都按左右手定义）。本层**只做对照与提示，不判对错**——
知识 073/074 是 agent 观察条目（未经用户确认），拿它当判据是越界。

对照的两条：

- **知识 073**：一张谱只用一部分配置；40 类透镜下官谱配置种类中位 **15**（Q1 13 / Q3 18），
  写到 ≥5% 小节的**主料中位只有 4 类**；几何族只作点缀（方向盘在 388 张谱里没有一张
  把它写成主料）。⚠️「主料 ≥5% 小节」是**本项目的内部计数口径**，不是社区说法。
- **知识 074**：配置的段落分区（イントロ 纵连族 / サビ 双押系与接得顺的星星 /
  ドロップ・間奏 几何 / Bメロ 侧边双押与要分手的星星 / 落ちサビ 出张最集中）。
  有 `song_analysis.json` 时才对得上段落。
"""

from __future__ import annotations

from ._deps import configs_hand as CH
from .model import LayerResult

#: 知识 073 的官谱量级（40 类手序版透镜）。只作对照，不作判据。
OFFICIAL_N_CONFIG_MEDIAN = 15
OFFICIAL_N_CONFIG_Q = (13, 18)
OFFICIAL_N_MAIN_MEDIAN = 4
MAIN_SHARE = 0.05          # 「主料」= 写到 ≥5% 小节（**内部计数口径**）

#: 知识 073 §4：「几乎每张谱都有」的三件（出现率 93–95%）
UBIQUITOUS = ("子弹", "连续双押", "跳拍")

#: 知识 071/073：几何族（随定数才被启用、只作点缀的一族）
GEOMETRIC = ("三角交互", "楼梯交互", "大宇宙", "轴交互", "散点", "方向盘", "逆楼梯")

#: 知识 074 的段落地盘（日式标签 → 该段富集的配置族，lift 见条目正文）
SECTION_HOME = {
    "イントロ": ("纵连", "长纵连"),
    "ドロップ": ("普通交互", "三角交互", "楼梯交互", "散点"),
    "間奏": ("轴交互", "三角交互", "楼梯交互"),
    "サビ": ("单双", "连续双押", "双押纵", "错位", "一笔画", "连续拍滑", "绕圈星星", "穿心"),
    "ラスサビ": ("一笔画", "连续拍滑", "错位", "连续双押"),
    "落ちサビ": ("出张", "绕圈星星", "穿心", "方向盘"),
    "Bメロ": ("侧边双押", "鼓动段", "二连扫", "CYCLES型星星", "反手", "三叉戟"),
}


def check_configs(res, ha, segments: list[dict] | None,
                  measure_texts: dict[int, str]) -> LayerResult:
    out = LayerResult(layer="配置")
    hits = CH.detect_all(res, None, ha)
    if not hits:
        out.skipped = "配置识别器一条都没认出来（谱太短或全是未建模形态）"
        out.stats = {"n_hits": 0}
        return out

    n_measures = max((n.measure for n in res.notes), default=0) + 1
    # 每类配置覆盖了多少小节
    bars_of: dict[str, set[int]] = {}
    for h in hits:
        bars_of.setdefault(h.config, set()).update(range(h.bar_start, h.bar_end + 1))
    share = {k: len(v) / max(1, n_measures) for k, v in bars_of.items()}
    main = sorted([k for k, v in share.items() if v >= MAIN_SHARE],
                  key=lambda k: -share[k])
    garnish = sorted([k for k in share if k not in main], key=lambda k: -share[k])

    # ---- 按小节顺序的片段摘要 ----
    seq = []
    for h in sorted(hits, key=lambda x: (x.bar_start, x.config)):
        seq.append({"bars": f"m{h.bar_start:03d}–m{h.bar_end:03d}", "config": h.config,
                    "n_tasks": h.n_tasks, "hands": h.hands_pattern,
                    "source": h.source})

    out.add("提示", "CFG-COMPOSITION",
            f"配置构成：共 **{len(share)} 类**（官谱 40 类透镜下中位 "
            f"{OFFICIAL_N_CONFIG_MEDIAN}，Q1 {OFFICIAL_N_CONFIG_Q[0]} / "
            f"Q3 {OFFICIAL_N_CONFIG_Q[1]}，知识 073）；"
            f"主料（≥{MAIN_SHARE:.0%} 小节，**内部计数口径**）{len(main)} 类："
            + "、".join(f"{k}({share[k]:.0%})" for k in main[:8]),
            source="知识 073")
    if garnish:
        out.add("提示", "CFG-GARNISH",
                "点缀：" + "、".join(f"{k}({share[k]:.0%})" for k in garnish[:14])
                + ("…" if len(garnish) > 14 else ""),
                source="知识 073")

    # ---- 对照 073 ----
    if len(share) < OFFICIAL_N_CONFIG_Q[0]:
        out.add("提示", "CFG-VARIETY-LOW",
                f"配置种类 {len(share)} 类低于官谱 Q1（{OFFICIAL_N_CONFIG_Q[0]}）——"
                "知识 071：定数升档靠「加种类、加每类时长」，种类偏少通常意味着"
                "这张谱的词汇量还没铺开。**这不是错误**，只是对照。",
                source="知识 071 / 073")
    elif len(share) > OFFICIAL_N_CONFIG_Q[1]:
        out.add("提示", "CFG-VARIETY-HIGH",
                f"配置种类 {len(share)} 类高于官谱 Q3（{OFFICIAL_N_CONFIG_Q[1]}）——"
                "知识 073：官谱**从来不把配置用满**，选配置是「挑一小撮来用」"
                "而不是清单式铺开。",
                source="知识 073")
    if len(main) > OFFICIAL_N_MAIN_MEDIAN + 2:
        out.add("提示", "CFG-MAIN-MANY",
                f"主料 {len(main)} 类，官谱中位只有 {OFFICIAL_N_MAIN_MEDIAN} 类"
                "（知识 073：每张谱只主打几类）",
                source="知识 073")
    geo_main = [k for k in main if any(g in k for g in GEOMETRIC)]
    if geo_main:
        out.add("提示", "CFG-GEOMETRIC-MAIN",
                "几何族被写成了主料：" + "、".join(geo_main)
                + "——知识 073：官谱里几何配置**只作点缀**"
                "（方向盘在 388 张谱里没有一张写成主料）",
                source="知识 073")
    missing_ubi = [k for k in UBIQUITOUS if k not in share]
    if missing_ubi:
        out.add("提示", "CFG-UBIQUITOUS-MISSING",
                "官谱里「几乎每张谱都有」的三件里缺了：" + "、".join(missing_ubi)
                + "（子弹 94.8% / 连续双押 94.3% / 跳拍 93.8%，知识 073）"
                "——**不是必须有**，只是对照。",
                source="知识 073")

    # ---- 对照 074（需要段落标签） ----
    seg_note = ""
    if segments:
        seg_of_bar: dict[int, str] = {}
        for s in segments:
            ja = s.get("label_ja") or s.get("function") or ""
            for b in range(int(s["start_bar"]), int(s["end_bar"]) + 1):
                seg_of_bar[b - 1] = ja       # song sheet 小节 1 起，谱面 0 起
        for ja, family in SECTION_HOME.items():
            bars = {b for b, lab in seg_of_bar.items() if lab == ja}
            if not bars:
                continue
            used = sorted({c for c, bs in bars_of.items() if bs & bars})
            hit_family = [f for f in family if any(f in u for u in used)]
            out.add("提示", "CFG-SECTION",
                    f"{ja}（{len(bars)} 小节）用了：{('、'.join(used[:10]) or '—')}"
                    + ("…" if len(used) > 10 else "")
                    + f"；知识 074 说这一段的地盘是 {('、'.join(family))}，"
                    + (f"命中 {('、'.join(hit_family))}" if hit_family else "**一个都没命中**")
                    + "。⚠️ 074 是 agent 观察（未经用户确认），只作对照。",
                    source="知识 074")
    else:
        seg_note = "没有 song_analysis.json → 段落分区（知识 074）无法对照"
        out.add("提示", "CFG-NO-SECTIONS", seg_note, source="知识 074")

    out.stats = {
        "n_config_kinds": len(share),
        "n_main": len(main),
        "main": {k: round(share[k], 4) for k in main},
        "garnish": {k: round(share[k], 4) for k in garnish},
        "sequence": seq,
        "official_median_kinds": OFFICIAL_N_CONFIG_MEDIAN,
        "official_median_main": OFFICIAL_N_MAIN_MEDIAN,
    }
    return out
