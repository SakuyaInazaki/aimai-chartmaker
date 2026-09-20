#!/usr/bin/env python3
"""手序 / 可玩性层：`hands.assign()` 的结果翻成**可核对的清单**。

对应 AGENT.md 验收标准 2（人类可玩）与 3（手序合理）。分级沿用
`docs/hand-sequencing.md` §4：

- **不可行**（`infeasible`）与**绝对级无理** → 错误，必须为 0；
- **硬级无理** → 警告。`docs/hand-sequencing.md` §6.2 要求 `budget.muri`（含硬级）为 0，
  但 388 官谱自己也有几十次硬级（知识 077：Hold 尾多押 74 / 超速 24 / 外键 13 / 撞尾 7）
  ——记成错误等于宣布官谱不合格。写谱时**硬级尽量清零**，清不掉要写明理由；
- **软级**（`scrape`，撞尾 / 外键 / 路径蹭键 / 换手拧巴）→ 提示 ——
  知识 048/052/062 明说"蹭"是设计手段，**不是无理**；
- **出张落点**（知识 064）→ 提示清单，附每一处的小节与原文；
- **侧边双押**（知识 030 + 082 八型）→ 提示清单；**乐句级仍判不出引导**的单列为警告。

⚠️ 这里的无理计数依赖 `hands.py` 选出的**一套**手序，而合法手序不止一套
（知识 065）。所以"0 绝对无理"读作"**存在一套合法手序**"，不读作"任何手序都合法"。
"""

from __future__ import annotations

from ._deps import hands_mod as H
from .excerpt import excerpt_of
from .model import LayerResult

#: 官谱基线（知识 079）：388 谱 5 739 个出张落点 / 272 207 个手任务 ≈ 21.1 ‰，
#: `出张 / slide` ≈ 0.232。只作**对照**，不作判据。
OFFICIAL_CHUZHANG_PER_1000 = 21.1
OFFICIAL_CHUZHANG_PER_SLIDE = 0.232


def _measure_locator(res):
    """``时间(秒) → 小节号``。`hands.py` 的部分标记没带小节号（`measure=0`），
    直接拿 0 去取原文会指错地方。

    优先用 `simai_parser` 给的 `measure_starts`（解析时逐槽记下来的精确小节线）。
    ⚠️ 旧写法 ``note.time - beat_in_measure * 60/note.bpm`` 在**小节中途换速**的那一小节
    会算错（实测 `1789-咲キ誇レ常世ノ華` m013 差 **619 ms**），只留作没有 `measure_starts`
    时的兜底。
    """
    by_m = dict(getattr(res, "measure_starts", None) or {})
    if not by_m:
        starts = sorted({(n.measure, n.time - n.beat_in_measure * 60.0 / (n.bpm or 1.0))
                         for n in res.notes})
        for m, t0 in starts:
            by_m.setdefault(m, t0)
    items = sorted(by_m.items())

    def at(t: float, fallback: int) -> int:
        best = fallback
        for m, t0 in items:
            if t0 <= t + 1e-6:
                best = m
            else:
                break
        return best
    return at


def check_play(res, ha, measure_texts: dict[int, str]) -> LayerResult:
    out = LayerResult(layer="手序")
    at_measure = _measure_locator(res)

    # ---- ① 不可行 ----
    for ev in ha.infeasible:
        meas = at_measure(ev.time, ev.measure)
        out.add("错误", "PLAY-INFEASIBLE",
                f"不可行：{ev.kind}——{ev.detail}",
                measure=meas, excerpt=excerpt_of(measure_texts, meas),
                source="hand-sequencing §4")

    # ---- ② 无理 ----
    # **绝对级 = 错误**（AGENT.md 验收标准 2/3：人类可玩 / 手序合理）；
    # **硬级 = 警告**——`docs/hand-sequencing.md` §6.2 要求 `budget.muri`（含硬级）为 0，
    # 但 388 官谱自己也踩到几十次硬级（知识 077：Hold 尾多押 74 / 超速 24 / 外键 13 /
    # 撞尾 7），把它记成错误等于宣布官谱不合格。写谱时**硬级尽量清零**，清不掉要说明理由。
    for m in ha.muri:
        code = {"叠键": "PLAY-OVERLAP", "多押": "PLAY-MULTI",
                "Hold尾多押": "PLAY-HOLDTAIL", "Hold 尾多押": "PLAY-HOLDTAIL",
                "占用冲突": "PLAY-CONFLICT",
                "超速": "PLAY-SPEED", "外键": "PLAY-OUTER",
                "撞尾": "PLAY-TAILHIT"}.get(m.kind, "PLAY-MURI")
        src = {"叠键": "知识 009", "多押": "知识 008", "Hold尾多押": "知识 012",
               "Hold 尾多押": "知识 012",
               "外键": "知识 010", "撞尾": "知识 011", "超速": "知识 015"}.get(
                   m.kind, "hand-sequencing §4")
        meas = at_measure(m.time, m.measure)
        out.add("错误" if m.level == "绝对" else "警告", code,
                f"[{m.level}无理] {m.kind}：{m.detail}"
                + ("" if m.level == "绝对" else
                   "（硬级：388 官谱也有几十次，见知识 077；写谱时尽量清零）"),
                measure=meas, excerpt=excerpt_of(measure_texts, meas),
                source=src)

    # ---- ③ 软级：蹭的压力（不是无理） ----
    scrape_counts: dict[str, int] = {}
    for m in ha.scrape:
        scrape_counts[m.kind] = scrape_counts.get(m.kind, 0) + 1
    for kind, cnt in sorted(scrape_counts.items(), key=lambda kv: -kv[1]):
        sample = next(m for m in ha.scrape if m.kind == kind)
        sm = at_measure(sample.time, sample.measure)
        out.add("提示", "PLAY-SCRAPE",
                f"软级「{kind}」×{cnt}（**不是无理**，知识 048/052/062：蹭是设计手段）"
                f"；首处 m{sm:03d}：{sample.detail}",
                measure=sm,
                excerpt=excerpt_of(measure_texts, sm),
                source="hand-sequencing §4 / 知识 077")

    # ---- ④ 出张落点清单（知识 064） ----
    # 口径与 `hands.chart_summary` 完全一致（`BarHand.chuzhang`：只数**落键**，
    # slide 的头/尾另记在 `chuzhang_slide` 里），这样才能和知识 079 的官谱数直接比。
    summary = H.chart_summary(ha)
    chuzhang: list[tuple[int, float, str, int]] = []
    for t, h in zip(ha.tasks, ha.task_hand):
        for hh in (("L", "R") if h == "LR" else (h,)):
            if hh in ("L", "R") and H.is_chuzhang(hh, t.key):
                chuzhang.append((t.measure, t.t, hh, int(t.key)))
    n_slide = sum(1 for t in ha.tasks if t.kind in ("slide", "wifi"))
    per_bar: dict[int, list[str]] = {}
    for meas, _t, h, k in chuzhang:
        per_bar.setdefault(meas, []).append(f"{h}→{k}")
    for meas in sorted(per_bar):
        out.add("提示", "PLAY-CHUZHANG",
                f"出张 {len(per_bar[meas])} 处：{'、'.join(per_bar[meas])}"
                "（知识 064：轻易不要写；写了要判断是容易游玩的还是故意设置的难点，"
                "就算是难点也要做好引导）",
                measure=meas, excerpt=excerpt_of(measure_texts, meas),
                source="知识 064 / 079 / 087")

    # ---- ⑤ 侧边双押与引导类型（知识 030 / 082） ----
    unguided = 0
    for ev in ha.side_doubles:
        meas = int(ev.get("measure", -1))
        types = (ev.get("guide_types") or []) + (ev.get("guide_types_phrase") or [])
        desc = "、".join(str(t) for t in types) if types else "（判不出）"
        level = ev.get("guide_level") or ""
        if ev.get("guided"):
            out.add("提示", "PLAY-SIDEDOUBLE",
                    f"侧边双押 {ev.get('keys', '')}：引导型 {desc}"
                    + (f"（{level}窗口）" if level else "") +
                    "（知识 082 官谱八型；铁律的主语是「突然」，不是侧边双押本身）",
                    measure=meas, excerpt=excerpt_of(measure_texts, meas),
                    source="知识 030 / 082")
        else:
            unguided += 1
            out.add("警告", "PLAY-SIDEDOUBLE-UNGUIDED",
                    f"侧边双押 {ev.get('keys', '')}：**乐句级窗口仍判不出引导**——"
                    "按知识 080/082 这只进**复核清单**、不记无理，请人看一眼"
                    "（388 官谱里这种只剩 1 次，且那是全谱第一组 note）",
                    measure=meas, excerpt=excerpt_of(measure_texts, meas),
                    source="知识 030 / 080 / 082")

    out.stats = {
        "n_tasks": ha.n_tasks,
        "n_slide_tasks": n_slide,
        "hand_hardness": round(ha.hand_hardness, 5),
        "muri_counts": ha.counts,
        "scrape_counts": scrape_counts,
        "n_infeasible": len(ha.infeasible),
        "n_absolute": sum(1 for m in ha.muri if m.level == "绝对"),
        "n_hard": sum(1 for m in ha.muri if m.level == "硬"),
        "n_chuzhang": summary["chuzhang"],
        "chuzhang_per_1000_tasks": summary["chuzhang_per_1k"],
        "chuzhang_slide": summary["chuzhang_slide"],
        "chuzhang_per_slide": summary["chuzhang_per_slide"],
        "official_chuzhang_per_1000_tasks": OFFICIAL_CHUZHANG_PER_1000,
        "official_chuzhang_per_slide": OFFICIAL_CHUZHANG_PER_SLIDE,
        "n_side_double": len(ha.side_doubles),
        "n_side_double_unguided": unguided,
    }
    return out
