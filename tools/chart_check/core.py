#!/usr/bin/env python3
"""检查器主流程：把五层串起来。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ._deps import hands_mod as H
from ._deps import parse_chart
from .configlayer import check_configs
from .densitylayer import check_density
from .excerpt import measure_texts as _measure_texts
from .maidata import MaiData, read_maidata
from .model import LAYERS, Issue, LayerResult
from .play import check_play
from .samplinglayer import check_sampling
from .syntax import check_syntax

_LEVEL_NUM = re.compile(r"^(\d+(?:\.\d+)?)")


@dataclass
class CheckReport:
    name: str = ""
    path: str = ""
    inote: int = 5
    level: float | None = None
    level_text: str = ""
    layers: list[LayerResult] = field(default_factory=list)

    def layer(self, name: str) -> LayerResult | None:
        for lr in self.layers:
            if lr.layer == name:
                return lr
        return None

    @property
    def issues(self) -> list[Issue]:
        return [it for lr in self.layers for it in lr.issues]

    def count(self, level: str) -> int:
        return sum(1 for it in self.issues if it.level == level)

    @property
    def verdict(self) -> str:
        """一句话结论。**只看错误**——警告与提示交给人。"""
        n = self.count("错误")
        if n == 0:
            return "零错误（内部校验口径；外部三检待接）"
        return f"**{n} 条错误**，未通过"

    @property
    def ok(self) -> bool:
        return self.count("错误") == 0

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path, "inote": self.inote,
                "level": self.level, "level_text": self.level_text,
                "verdict": self.verdict, "ok": self.ok,
                "n_error": self.count("错误"), "n_warn": self.count("警告"),
                "n_hint": self.count("提示"),
                "external_lint": "未接（本机无 SimaiSharp / MajdataEdit / MiaCode）",
                "layers": [lr.to_dict() for lr in self.layers]}


def _level_from_text(text: str) -> float | None:
    """`&lv_5=13+` / `13.6` → 数字。`13+` 取 13.7（官方 `+` 档的常见定数下界不确定，
    这里只做**粗取**，给 `--level` 覆盖）。"""
    if not text:
        return None
    m = _LEVEL_NUM.match(text.strip())
    if not m:
        return None
    v = float(m.group(1))
    if text.strip().endswith("+") and v == int(v):
        return v + 0.7
    return v


def check_chart(md: MaiData, *, inote: int | None = None,
                level: float | None = None,
                analysis: dict | None = None) -> CheckReport:
    """对一份已切好的 `maidata` 跑完五层。"""
    diff, body, _line, body_off = md.pick_inote(inote)
    rep = CheckReport(name=md.meta("title") or Path(md.path).stem or "<chart>",
                      path=md.path, inote=diff)
    rep.level_text = md.levels().get(diff, "")
    rep.level = level if level is not None else _level_from_text(rep.level_text)

    res = parse_chart(body, name=rep.name)
    texts = _measure_texts(body)

    # ① 语法
    rep.layers.append(check_syntax(md, body, body_off, res))

    # ② 手序（解析不出 note 就没法跑）
    if res.notes:
        ha = H.assign(res)
        rep.layers.append(check_play(res, ha, texts))
        rep.layers.append(check_configs(res, ha,
                                        ((analysis or {}).get("structure") or {}).get("segments"),
                                        texts))
        rep.layers.append(check_density(res, rep.level, texts))
        rep.layers.append(check_sampling(res, analysis, texts))
    else:
        for layer in LAYERS[1:]:
            lr = LayerResult(layer=layer)
            lr.skipped = "解析不出任何 note，上游语法层先修"
            rep.layers.append(lr)
    return rep


def check_file(path: str | Path, *, inote: int | None = None,
               level: float | None = None,
               analysis_path: str | Path | None = None) -> CheckReport:
    md = read_maidata(path)
    analysis = None
    p = Path(path)
    cand = Path(analysis_path) if analysis_path else (p.parent / "song_analysis.json")
    if cand.exists():
        analysis = json.loads(cand.read_text(encoding="utf-8"))
    return check_chart(md, inote=inote, level=level, analysis=analysis)
