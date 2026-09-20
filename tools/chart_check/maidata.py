#!/usr/bin/env python3
"""`maidata.txt` 的切分：meta 头 + 各难度谱面正文，且**保留行号**。

`chart_analysis.simai_parser` 只吃正文（模块注释明说「若传入带 meta 的 maidata.txt，
请先自行切出正文」）。检查器要报行列，所以这里切的时候把每个 `&inote_N=` 正文
在原文件里的**起始行号与起始偏移**一起带出来。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_META_LINE = re.compile(r"^&([A-Za-z_0-9]+)=(.*)$")
_INOTE = re.compile(r"^inote_([1-7])$")


@dataclass
class Meta:
    """一条 `&key=value`（value 可跨多行）。"""

    key: str
    value: str
    line: int          # 1-based，`&key=` 所在行
    end_line: int      # 1-based，value 的最后一行


@dataclass
class MaiData:
    path: str = ""
    text: str = ""
    metas: list[Meta] = field(default_factory=list)
    #: ``{难度号: (正文, 起始行号(1-based), 起始字符偏移)}``
    inotes: dict[int, tuple[str, int, int]] = field(default_factory=dict)

    def meta(self, key: str) -> str | None:
        for m in self.metas:
            if m.key == key:
                return m.value
        return None

    def levels(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for m in self.metas:
            mm = re.match(r"^lv_([1-7])$", m.key)
            if mm:
                out[int(mm.group(1))] = m.value.strip()
        return out

    def pick_inote(self, want: int | None = None) -> tuple[int, str, int, int]:
        """选一份要检查的正文：指定 `want` 就用它，否则取**难度号最大的非空**那份。"""
        if not self.inotes:
            raise ValueError("文件里没有任何 &inote_N=")
        if want is not None:
            if want not in self.inotes:
                raise ValueError(f"文件里没有 &inote_{want}=")
            body, ln, off = self.inotes[want]
            return want, body, ln, off
        for d in sorted(self.inotes, reverse=True):
            body, ln, off = self.inotes[d]
            if body.strip():
                return d, body, ln, off
        d = max(self.inotes)
        body, ln, off = self.inotes[d]
        return d, body, ln, off


def parse_maidata(text: str, path: str = "") -> MaiData:
    """切 meta 与正文。**不做任何校验**，校验在 `syntax.py`。"""
    md = MaiData(path=path, text=text)
    lines = text.split("\n")
    # 每行起始的字符偏移
    offs: list[int] = []
    pos = 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1

    i = 0
    n = len(lines)
    while i < n:
        m = _META_LINE.match(lines[i])
        if not m:
            i += 1
            continue
        key, first = m.group(1), m.group(2)
        start_line = i + 1
        body_lines = [first]
        j = i + 1
        while j < n and not _META_LINE.match(lines[j]):
            body_lines.append(lines[j])
            j += 1
        value = "\n".join(body_lines)
        md.metas.append(Meta(key=key, value=value, line=start_line, end_line=j))
        mi = _INOTE.match(key)
        if mi:
            # 正文的字符偏移 = 该行起点 + "&inote_N=" 的长度
            off = offs[i] + len(key) + 2
            md.inotes[int(mi.group(1))] = (value, start_line, off)
        i = j
    return md


def read_maidata(path: str | Path) -> MaiData:
    p = Path(path)
    return parse_maidata(p.read_text(encoding="utf-8", errors="replace"), str(p))


def line_col(text: str, offset: int) -> tuple[int, int]:
    """字符偏移 → ``(1-based 行, 1-based 列)``。"""
    if offset < 0:
        return 0, 0
    head = text[:offset]
    line = head.count("\n") + 1
    col = offset - (head.rfind("\n") + 1) + 1
    return line, col
