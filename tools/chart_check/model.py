#!/usr/bin/env python3
"""检查器的公共数据结构：一条 :class:`Issue` = 报告里的一行。"""

from __future__ import annotations

from dataclasses import dataclass, field

#: 六层的固定顺序（报告与 JSON 都按这个顺序）。
#: 前五层各自独立跑；**「深度」层跑在最后、只消费前五层的 `stats`**（见 `depthlayer.py`）。
LAYERS = ("语法", "手序", "配置", "密度", "采音", "深度")

#: 三个严重级别。`错误` 必须修；`警告` 要看一眼；`提示` 只是复核清单，不判对错。
LEVELS = ("错误", "警告", "提示")

_LEVEL_RANK = {lv: i for i, lv in enumerate(LEVELS)}


@dataclass
class Issue:
    """一条检查结果。

    `code` 是稳定标识（单测钉它，不钉中文措辞）；`message` 是给人看的中文。
    `source` 写依据——文档小节号或知识条目号，**没有依据的判断不写进 Issue**。
    """

    layer: str
    level: str
    code: str
    message: str
    line: int = 0          # 1-based；0 = 不适用
    col: int = 0           # 1-based；0 = 不适用
    measure: int = -1      # 谱面小节号（0 起，与 simai_parser 同口径）；-1 = 不适用
    excerpt: str = ""      # 原文片段（**最多 6 行**，见 AGENT.md 引用约定）
    source: str = ""

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"未知的层：{self.layer}")
        if self.level not in LEVELS:
            raise ValueError(f"未知的级别：{self.level}")

    @property
    def rank(self) -> int:
        return _LEVEL_RANK[self.level]

    def where(self) -> str:
        bits = []
        if self.measure >= 0:
            bits.append(f"m{self.measure:03d}")
        if self.line:
            bits.append(f"{self.line}行" + (f"{self.col}列" if self.col else ""))
        return " ".join(bits)

    def to_dict(self) -> dict:
        return {"layer": self.layer, "level": self.level, "code": self.code,
                "message": self.message, "line": self.line, "col": self.col,
                "measure": self.measure, "excerpt": self.excerpt,
                "source": self.source}


@dataclass
class LayerResult:
    """一层的产出：若干 Issue + 该层自己的统计块。"""

    layer: str
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    #: 该层没能跑（缺输入 / 上一层已失败）时写原因，报告里显式说明**没跑**
    skipped: str = ""

    def add(self, level: str, code: str, message: str, **kw) -> Issue:
        it = Issue(layer=self.layer, level=level, code=code, message=message, **kw)
        self.issues.append(it)
        return it

    def count(self, level: str) -> int:
        return sum(1 for i in self.issues if i.level == level)

    def to_dict(self) -> dict:
        return {"layer": self.layer, "skipped": self.skipped,
                "n_error": self.count("错误"), "n_warn": self.count("警告"),
                "n_hint": self.count("提示"),
                "issues": [i.to_dict() for i in self.issues],
                "stats": self.stats}
