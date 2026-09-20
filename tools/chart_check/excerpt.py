#!/usr/bin/env python3
"""逐小节原文片段：报告里每条问题都要能指回**谱面原文的那一小节**。

`simai_parser` 只给 note 事件，不保留原文位置；这里按同一套时间推进规则
（`docs/simai-syntax.md` §1.2 槽长公式）再走一遍规范化后的正文，把每个逗号槽
归到小节上，拼回该小节的原文。

⚠️ 引用约定（AGENT.md）：每处谱面引用 ≤ 6 行 —— :func:`excerpt_of` 默认只给一行，
`chart_check` 的报告也只在问题条目上给一行。
"""

from __future__ import annotations

from .syntax import normalize_with_map

_BEATS_PER_MEASURE = 4.0


def measure_texts(body: str) -> dict[int, str]:
    """``{小节号(0 起): 该小节的正文（含 `,`）}``。"""
    s, _ = normalize_with_map(body)
    out: dict[int, list[str]] = {}
    bpm: float | None = None
    div: float | None = None
    beat = 0.0
    buf: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "(":
            j = s.find(")", i)
            if j < 0:
                break
            try:
                bpm = float(s[i + 1:j])
            except ValueError:
                pass
            buf.append(s[i:j + 1])
            i = j + 1
            continue
        if c == "{":
            j = s.find("}", i)
            if j < 0:
                break
            token = s[i + 1:j]
            if not token.startswith("#"):
                try:
                    div = float(token)
                except ValueError:
                    pass
            buf.append(s[i:j + 1])
            i = j + 1
            continue
        if c == ",":
            buf.append(",")
            m = int(beat // _BEATS_PER_MEASURE)
            out.setdefault(m, []).append("".join(buf))
            buf = []
            if bpm and div:
                beat += 4.0 / div
            i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        m = int(beat // _BEATS_PER_MEASURE)
        out.setdefault(m, []).append("".join(buf))
    return {k: "".join(v) for k, v in out.items()}


def excerpt_of(texts: dict[int, str], measure: int, limit: int = 110) -> str:
    """取一小节原文，超长截断（报告里只给一行）。"""
    t = texts.get(measure, "")
    if not t:
        return ""
    return t if len(t) <= limit else t[:limit - 1] + "…"
