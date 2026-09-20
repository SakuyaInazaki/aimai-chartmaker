#!/usr/bin/env python3
"""报告渲染：中文 Markdown + JSON。两边同源，禁止各写各的。"""

from __future__ import annotations

import json

from .model import LAYERS

_EXTERNAL = (
    "> ⚠️ **外部三检待接。** `docs/simai-error-checking.md` §10-4 要求 "
    "**SimaiSharp lint + MajdataEdit SyntaxCheck + MiaCode strict** 三方实跑，"
    "本机没有装这三个工具——本报告的语法层是**按它们的规则交集实现的内部校验**，"
    "**不能代替真机跑谱**。"
)

_CAVEAT = (
    "> ⚠️ **读法**：`错误` 必须修；`警告` 要看一眼；`提示` 是**复核清单**，"
    "不判对错。手序层的无理计数依赖 `hands.py` 选出的**一套**手序，"
    "而合法手序不止一套（知识 065）——「绝对无理 0」读作"
    "「**存在一套合法手序**」，不读作「任何手序都合法」。"
    "配置层与采音层的对照基准（知识 068–087）是 **agent 观察条目、未经用户确认**，"
    "只作对照，不作判据。"
)


def render_markdown(report) -> str:
    L: list[str] = []
    L.append(f"# 谱面检查报告 — {report.name}")
    L.append("")
    L.append("```")
    L.append(f"文件: {report.path}")
    L.append(f"难度: &inote_{report.inote}" +
             (f"  &lv_{report.inote}={report.level_text}" if report.level_text else ""))
    if report.level is not None:
        L.append(f"对照定数: {report.level}")
    L.append(f"结论: {report.verdict}")
    L.append("错误 {e} / 警告 {w} / 提示 {h}".format(
        e=report.count("错误"), w=report.count("警告"), h=report.count("提示")))
    L.append("```")
    L.append("")
    L.append(_EXTERNAL)
    L.append(">")
    L.append(_CAVEAT)
    L.append("")

    # 总览表
    L.append("## 0. 总览")
    L.append("")
    L.append("| 层 | 错误 | 警告 | 提示 | 状态 |")
    L.append("|----|-----:|-----:|-----:|------|")
    for layer in LAYERS:
        lr = report.layer(layer)
        if lr is None:
            L.append(f"| {layer} | — | — | — | 没跑 |")
            continue
        state = lr.skipped or ("通过" if lr.count("错误") == 0 else "**有错误**")
        L.append(f"| {layer} | {lr.count('错误')} | {lr.count('警告')} | "
                 f"{lr.count('提示')} | {state} |")
    L.append("")

    for i, layer in enumerate(LAYERS, 1):
        lr = report.layer(layer)
        if lr is None:
            continue
        L.append(f"## {i}. {layer}层")
        L.append("")
        if lr.skipped:
            L.append(f"**没跑**：{lr.skipped}")
            L.append("")
            continue
        for level in ("错误", "警告", "提示"):
            items = [it for it in lr.issues if it.level == level]
            if not items:
                continue
            L.append(f"### {level}（{len(items)}）")
            L.append("")
            for it in items:
                loc = it.where()
                head = f"- **[{it.code}]** " + (f"`{loc}` " if loc else "") + it.message
                if it.source:
                    head += f"　_依据：{it.source}_"
                L.append(head)
                if it.excerpt:
                    L.append(f"  - 原文：`{it.excerpt}`")
            L.append("")
        if lr.stats:
            L.append("<details><summary>统计块</summary>")
            L.append("")
            L.append("```json")
            L.append(json.dumps(_trim(lr.stats), ensure_ascii=False, indent=2))
            L.append("```")
            L.append("")
            L.append("</details>")
            L.append("")
    return "\n".join(L)


def _trim(stats: dict) -> dict:
    """统计块里的长数组在 md 里截断（JSON 输出仍是全量）。"""
    out = {}
    for k, v in stats.items():
        if isinstance(v, list) and len(v) > 24:
            out[k] = v[:24] + [f"…（共 {len(v)} 项）"]
        else:
            out[k] = v
    return out


def render_json(report) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
