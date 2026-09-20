#!/usr/bin/env python3
"""DX 谱 → ST 可比视图过滤器。

**为什么有这个模块**：对照学习时拿到的参考谱是 **DX 谱**（`&lv_5=14`，
HachimiDX 抄谱），里面混着本项目不写、也不学的 DX/FESTiVAL 要素
（Touch / Touch HOLD、EX `x`、滑条绝赞 `]b`、Break HOLD、拼接（连锁）星星）。
直接统计会把这些算进去，得到的"note 配比 / 配置构成"跟 ST 谱不可比。

本模块**只做减法**：在 :class:`~tools.chart_analysis.simai_parser.ParseResult`
上把非 ST 要素摘掉，**时间轴（`tempo_segments` / `measure_starts` / 每个 note 的
`time`/`beat`/`measure`）一个字节都不动**，于是过滤后的结果可以直接喂给
``hands.assign()`` 与 ``configs_hand.detect_all()``，与我们自己的 ST 谱并排比。

ST 白/黑名单依据 `docs/st-chart-elements.md`（知识 016）：

========================  ========================================  ==========
要素                       处置                                       记号
========================  ========================================  ==========
Touch / Touch HOLD         **整个丢掉**                                ``touch``
EX 修饰符 ``x``            **摘掉修饰符**，note 本体保留                ``ex``
Break HOLD（``1bxh``）     **降级成普通 HOLD**（ST 无 Break HOLD）      ``break_hold``
滑条绝赞（``1-4[8:1]b``）  **降级成普通 slide**（ST 无 Break Slide）    ``break_slide``
拼接（连锁）星星           **丢掉滑轨**，星星头按下面的开关处置          ``chain``
非 ST 形状的星星           同上（当前 12 形状全部 ST 合法，故通常为 0）  ``shape``
========================  ========================================  ==========

**星星头的处置（二选一，本模块默认 `"tap"`）**：滑轨被丢掉之后，那个
星星头在 ST 视角下就是"一个要打的按键"，所以默认 **`head_as="tap"`**——
把它记成 TAP，进 `taps` 计数。另一种口径 `head_as="star"` 把它留在
`slide_star`（"星星头"）里，好处是能看出"这里原本是星星"，坏处是
`counts["taps"]` 会把一个没有轨道的头当成星星头算进去、与 ST 谱不可比。
**做谱面对照时用默认的 `"tap"`。**

命令行::

    python -m tools.chart_analysis.dx_filter <maidata.txt> [--inote 5] [--bars 34-49]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace as _dc_replace

from .simai_parser import NoteEvent, ParseResult, parse_chart

#: ST 合法的 12 种 slide 形状（`docs/st-chart-elements.md` §二；PiNK 之后不再新增）
ST_SLIDE_SHAPES: frozenset[str] = frozenset({"-", "^", "<", ">", "v", "V",
                                             "p", "q", "pp", "qq", "s", "z", "w"})

#: 丢弃/降级的原因记号
REASONS = ("touch", "ex", "break_hold", "break_slide", "chain", "shape")


@dataclass
class DropRecord:
    """一条被丢掉或被降级的记录（保留时间信息，便于定位到小节）。"""

    bar: int          # 1 起的小节号（= note.measure + 1）
    beat_in_bar: float
    time: float
    reason: str       # REASONS 之一
    kind: str         # 原 note 的 kind
    key: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {"bar": self.bar, "beat_in_bar": round(self.beat_in_bar, 4),
                "time": round(self.time, 4), "reason": self.reason,
                "kind": self.kind, "key": self.key, "detail": self.detail}


@dataclass
class FilterResult:
    """过滤结果。``result`` 是一份新的 :class:`ParseResult`（时间轴与原件一致）。"""

    result: ParseResult
    dropped: list[DropRecord] = field(default_factory=list)   # 整个消失的 note
    demoted: list[DropRecord] = field(default_factory=list)   # 保留但改了属性的 note
    head_as: str = "tap"
    counts_before: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        """丢弃/降级统计（给报告用）。"""
        d = Counter(r.reason for r in self.dropped)
        m = Counter(r.reason for r in self.demoted)
        return {
            "head_as": self.head_as,
            "dropped": dict(sorted(d.items())),
            "demoted": dict(sorted(m.items())),
            "n_dropped": len(self.dropped),
            "n_demoted": len(self.demoted),
            "counts_before": self.counts_before,
            "counts_after": self.result.counts,
        }


def _record(note: NoteEvent, reason: str, detail: str = "") -> DropRecord:
    return DropRecord(bar=note.measure + 1, beat_in_bar=note.beat_in_measure,
                      time=note.time, reason=reason, kind=note.kind,
                      key=note.key, detail=detail)


def filter_to_st(res: ParseResult, *, head_as: str = "tap") -> FilterResult:
    """把一份（可能是 DX 的）解析结果削成 ST 可比视图。

    ``head_as``：滑轨被丢掉后星星头怎么记——``"tap"``（默认）或 ``"star"``。
    """
    if head_as not in ("tap", "star"):
        raise ValueError(f"head_as 只能是 'tap' / 'star'，收到 {head_as!r}")

    out = ParseResult(
        errors=list(res.errors),
        warnings=list(res.warnings),
        bpm_events=list(res.bpm_events),
        tempo_segments=list(res.tempo_segments),
        total_beats=res.total_beats,
        total_seconds=res.total_seconds,
        measure_starts=dict(res.measure_starts),
        has_end_marker=res.has_end_marker,
    )
    dropped: list[DropRecord] = []
    demoted: list[DropRecord] = []

    # 第一轮：先判定每条滑轨的去留，好让同组的星星头知道自己还有没有轨道
    drop_track: list[bool] = []
    for n in res.notes:
        if n.kind != "slide_track":
            drop_track.append(False)
            continue
        if len(n.segments) > 1:
            drop_track.append(True)
        elif n.shape and n.shape not in ST_SLIDE_SHAPES:
            drop_track.append(True)
        else:
            drop_track.append(False)

    # 每个 (group_index, 星头键) 还剩几条轨道
    alive: Counter = Counter()
    for n, dt in zip(res.notes, drop_track):
        if n.kind == "slide_track" and not dt:
            alive[(n.group_index, n.key)] += 1

    kept: list[NoteEvent] = []
    for n, dt in zip(res.notes, drop_track):
        if n.kind in ("touch", "touch_hold"):
            dropped.append(_record(n, "touch"))
            continue
        if n.kind == "slide_track" and dt:
            reason = "chain" if len(n.segments) > 1 else "shape"
            dropped.append(_record(n, reason,
                                   detail=f"{n.key}{n.shape}{n.end_key}"))
            continue

        m = _dc_replace(n)
        if n.is_ex:
            demoted.append(_record(n, "ex"))
            m.is_ex = False
        if n.kind == "hold" and n.is_break:
            demoted.append(_record(n, "break_hold"))
            m.is_break = False
        if n.kind == "slide_track" and n.is_break:
            demoted.append(_record(n, "break_slide",
                                   detail=f"{n.key}{n.shape}{n.end_key}"))
            m.is_break = False
        if n.kind == "slide_star" and alive[(n.group_index, n.key)] == 0:
            # 这个头的轨道全被丢了
            if head_as == "tap":
                demoted.append(_record(n, "chain", detail="星头按 tap 计"))
                m.kind = "tap"
            else:
                demoted.append(_record(n, "chain", detail="星头按星星头计（无轨道）"))
        kept.append(m)

    # 同刻 note 数变了，`is_each` 要重算；`slide_chain_segments`/`each_groups` 同理
    by_group: Counter = Counter(n.group_index for n in kept)
    for n in kept:
        n.is_each = by_group[n.group_index] >= 2
    out.notes = kept
    out.each_groups = sum(1 for v in by_group.values() if v >= 2)
    out.slide_chain_segments = sum(len(n.segments) for n in kept
                                   if n.kind == "slide_track")

    return FilterResult(result=out, dropped=dropped, demoted=demoted,
                        head_as=head_as, counts_before=res.counts)


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------

_INOTE_RE = re.compile(r"&inote_(\d+)\s*=", re.M)


def split_inote(text: str, inote: int | None = None) -> str:
    """从完整 maidata.txt 里切出谱面正文（`&inote_N=` 之后到下一个 `&` 开头行）。"""
    text = text.lstrip("﻿")
    hits = list(_INOTE_RE.finditer(text))
    if not hits:
        return text
    chosen = None
    if inote is not None:
        for h in hits:
            if int(h.group(1)) == inote:
                chosen = h
                break
    if chosen is None:
        chosen = hits[-1]
    start = chosen.end()
    tail = text[start:]
    nxt = re.search(r"^&\w+\s*=", tail, re.M)
    return tail[: nxt.start()] if nxt else tail


def load_st_view(path: str, *, inote: int | None = None,
                 head_as: str = "tap") -> FilterResult:
    """读一个 maidata.txt，解析并削成 ST 可比视图。"""
    with open(path, encoding="utf-8-sig") as fh:
        text = fh.read()
    res = parse_chart(split_inote(text, inote), name=path)
    return filter_to_st(res, head_as=head_as)


# ---------------------------------------------------------------------------
# 逐小节回写（"ST 可比视图"的可读形态）
# ---------------------------------------------------------------------------

#: 回写时允许的分音（384 的约数，`docs/simai-syntax.md` §3.3）
_DIVISORS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384)
_DUR_X = (1, 2, 4, 8, 16, 32, 64)
_EPS = 1e-6


def _fmt_duration(seconds: float, bpm: float) -> str:
    """把秒数写回 ``[x:y]``；凑不出整数比就退回 ``[#秒]``。"""
    if bpm <= 0 or seconds <= 0:
        return ""
    beats = seconds * bpm / 60.0
    for x in _DUR_X:
        y = beats * x / 4.0
        if abs(y - round(y)) < 1e-4 and round(y) >= 1:
            return f"[{x}:{round(y)}]"
    return f"[#{seconds:.4f}]"


def _fmt_group(notes: list[NoteEvent]) -> str:
    """把同一个时间槽里的 note 写成 simai 文本（`/` 分隔）。"""
    # 星头与它的滑轨要拼在一起；同头多轨用 `*`
    tracks: dict[str, list[NoteEvent]] = {}
    for n in notes:
        if n.kind == "slide_track":
            tracks.setdefault(n.key, []).append(n)
    parts: list[str] = []
    for n in notes:
        if n.kind == "slide_track":
            continue
        b = "b" if n.is_break else ""
        if n.kind == "tap":
            parts.append(f"{n.key}{b}")
        elif n.kind == "hold":
            parts.append(f"{n.key}{b}h{_fmt_duration(n.duration, n.bpm)}")
        elif n.kind == "slide_star":
            segs = tracks.get(n.key, [])
            if not segs:
                parts.append(f"{n.key}{b}")
                continue
            body = "*".join(
                f"{t.shape}{t.end_key}{_fmt_duration(t.duration, t.bpm)}"
                f"{'b' if t.is_break else ''}" for t in segs)
            parts.append(f"{n.key}{b}{body}")
    return "/".join(parts)


def render_bars(res: ParseResult, lo: int = 1, hi: int | None = None) -> dict[int, str]:
    """把过滤后的谱面逐小节写回 simai 文本（1 起的小节号 → 一行）。

    **这是"回写"不是原文**：分音按该小节实际落点取最小可行值、时长按 ``[x:y]`` 重写，
    所以文本形态可能与原谱不同（原谱一行未必是一小节）；**note 与时刻一一对应**。
    """
    by_bar: dict[int, dict[int, list[NoteEvent]]] = {}
    for n in res.notes:
        by_bar.setdefault(n.measure + 1, {}).setdefault(n.group_index, []).append(n)
    if hi is None:
        hi = max(by_bar) if by_bar else lo
    out: dict[int, str] = {}
    for bar in range(lo, hi + 1):
        groups = by_bar.get(bar, {})
        if not groups:
            out[bar] = ""
            continue
        offs = sorted({round(ns[0].beat_in_measure, 6) for ns in groups.values()})
        div = None
        for d in _DIVISORS:
            if all(abs(o * d / 4.0 - round(o * d / 4.0)) < 1e-4 for o in offs):
                div = d
                break
        if div is None:                       # 落点对不上任何分音（`[#秒]` 槽等）
            out[bar] = "  ".join(
                f"@{ns[0].beat_in_measure:.3f}:{_fmt_group(ns)}"
                for _, ns in sorted(groups.items(), key=lambda kv: kv[1][0].beat_in_measure))
            continue
        slot_txt = [""] * div
        for ns in groups.values():
            idx = int(round(ns[0].beat_in_measure * div / 4.0))
            if 0 <= idx < div:
                slot_txt[idx] = _fmt_group(sorted(ns, key=lambda n: (n.kind != "tap", n.key)))
        out[bar] = "{%d}" % div + ",".join(slot_txt) + ","
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DX 谱 → ST 可比视图")
    ap.add_argument("path")
    ap.add_argument("--inote", type=int, default=None)
    ap.add_argument("--head-as", choices=("tap", "star"), default="tap")
    ap.add_argument("--bars", default="", help="只列这些小节的丢弃明细，如 34-49")
    args = ap.parse_args(argv)

    fr = load_st_view(args.path, inote=args.inote, head_as=args.head_as)
    s = fr.summary()
    print(f"== {args.path}")
    print(f"过滤前 {s['counts_before']}")
    print(f"过滤后 {s['counts_after']}")
    print(f"丢弃 {s['n_dropped']}：{s['dropped']}")
    print(f"降级 {s['n_demoted']}：{s['demoted']}")
    if args.bars:
        lo, _, hi = args.bars.partition("-")
        lo, hi = int(lo), int(hi or lo)
        print(f"\n-- 小节 {lo}-{hi} 的丢弃/降级明细 --")
        for r in fr.dropped + fr.demoted:
            if lo <= r.bar <= hi:
                print(f"  m{r.bar:03d} b{r.beat_in_bar:5.2f} {r.reason:12s}"
                      f" {r.kind:12s} {r.key:3s} {r.detail}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
