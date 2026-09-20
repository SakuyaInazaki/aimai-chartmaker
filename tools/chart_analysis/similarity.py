#!/usr/bin/env python3
"""谱面相似度自检：**我们写的谱 vs 参考谱，逐小节 + n-gram**。

**为什么有这个模块**：对照学习（note 069）之后要按学到的**写法**重写 test-01，
但用户的硬约束是「**不得抄它的 note 序列**」。这件事不能靠自觉，要能量出来。

本模块量两件事，都**只报数字、不判对错**：

1. **逐小节完全相同**：两份谱的同一个小节，`dx_filter.render_bars()` 回写出来的
   文本一字不差。按该小节的 note 数分三类——
   **空小节**（两边都没音）、**单音小节**（≤1 个 note）、**≥2 note 的小节**。
   前两类撞上是正常的（空就是空、一个音摆在同一个键上很常见），
   **第三类就是"抄了"的直接证据**，逐个列出来。
2. **n-gram 重合率**：把谱面摊成一串「note 记号」（时间序，同刻按键位排），
   取长度 n 的滑窗，算两份谱共享了多少种 n-gram。

   ⚠️ **这个数字单看没有意义**——只有 8 个键，随便写两张谱都会共享一堆 4-gram。
   **必须和基线一起读**：`--baseline <旧谱>` 给出"**没看过参考谱时写出来的谱**"的重合率
   （test-01 v1 就是这样的谱），`--control` 再给一个把小节顺序打乱的对照。
   **v2 的重合率不高于 v1 基线**，才说明没有抄。

命令行::

    python -m tools.chart_analysis.similarity <我们的 maidata> <参考 maidata> \\
        [--baseline <旧版 maidata>] [--control] [--inote 5]
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field

from .dx_filter import filter_to_st, load_st_view, render_bars
from .simai_parser import NoteEvent, ParseResult

#: n-gram 的长度，默认报这两档
DEFAULT_NS = (4, 8)


# ---------------------------------------------------------------------------
# note 记号串
# ---------------------------------------------------------------------------


def note_token(n: NoteEvent) -> str:
    """一个 note 的记号。**只记"玩家要做什么"**，不记时刻（时刻由逐小节那一层管）。"""
    if n.kind == "tap":
        return ("b" if n.is_break else "t") + n.key
    if n.kind == "hold":
        return ("H" if n.is_break else "h") + n.key
    if n.kind == "slide_star":
        return ("S" if n.is_break else "s") + n.key
    if n.kind == "slide_track":
        return "-" + n.shape + n.end_key
    return "?" + n.key


def note_sequence(res: ParseResult) -> list[str]:
    """整张谱的 note 记号串（时间序；同刻按记号排序，保证可复现）。"""
    by_slot: dict[int, list[NoteEvent]] = {}
    for n in res.notes:
        by_slot.setdefault(n.group_index, []).append(n)
    out: list[str] = []
    for gi in sorted(by_slot):
        out.extend(sorted(note_token(n) for n in by_slot[gi]))
    return out


def ngrams(tokens: list[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def ngram_overlap(a: list[str], b: list[str], n: int) -> dict:
    """``a`` 相对 ``b`` 的 n-gram 重合。``rate_a`` = a 的 n-gram 里有多少**出现在** b。"""
    ca, cb = ngrams(a, n), ngrams(b, n)
    sa, sb = set(ca), set(cb)
    shared = sa & sb
    inst = sum(ca[g] for g in shared)          # 按出现次数算的 a 侧命中
    return {
        "n": n,
        "a_types": len(sa), "b_types": len(sb),
        "shared_types": len(shared),
        "rate_a_types": round(len(shared) / len(sa), 4) if sa else 0.0,
        "rate_a_instances": round(inst / sum(ca.values()), 4) if ca else 0.0,
        "jaccard": round(len(shared) / len(sa | sb), 4) if (sa | sb) else 0.0,
    }


# ---------------------------------------------------------------------------
# 逐小节
# ---------------------------------------------------------------------------


def _bar_note_counts(res: ParseResult) -> dict[int, int]:
    out: Counter = Counter()
    for n in res.notes:
        out[n.measure + 1] += 1
    return dict(out)


@dataclass
class SimilarityReport:
    n_bars: int = 0
    identical_empty: list[int] = field(default_factory=list)
    identical_single: list[int] = field(default_factory=list)
    identical_multi: list[int] = field(default_factory=list)   # ← 红线
    identical_detail: dict[int, str] = field(default_factory=dict)
    ngram: list[dict] = field(default_factory=list)

    @property
    def n_identical(self) -> int:
        return (len(self.identical_empty) + len(self.identical_single)
                + len(self.identical_multi))

    def to_dict(self) -> dict:
        return {"n_bars": self.n_bars, "n_identical": self.n_identical,
                "identical_empty": self.identical_empty,
                "identical_single": self.identical_single,
                "identical_multi": self.identical_multi,
                "ngram": self.ngram}


def compare(ours: ParseResult, ref: ParseResult, ns=DEFAULT_NS) -> SimilarityReport:
    """``ref`` 应当已经过 `dx_filter.filter_to_st()`（ST 可比视图）。"""
    rep = SimilarityReport()
    last = max([n.measure + 1 for n in ours.notes] + [n.measure + 1 for n in ref.notes] + [0])
    rep.n_bars = last
    a_bars = render_bars(ours, 1, last)
    b_bars = render_bars(ref, 1, last)
    a_cnt, b_cnt = _bar_note_counts(ours), _bar_note_counts(ref)
    for bar in range(1, last + 1):
        ta, tb = a_bars.get(bar, ""), b_bars.get(bar, "")
        if ta != tb:
            continue
        k = max(a_cnt.get(bar, 0), b_cnt.get(bar, 0))
        if k == 0:
            rep.identical_empty.append(bar)
        elif k == 1:
            rep.identical_single.append(bar)
        else:
            rep.identical_multi.append(bar)
            rep.identical_detail[bar] = ta
    sa, sb = note_sequence(ours), note_sequence(ref)
    rep.ngram = [ngram_overlap(sa, sb, n) for n in ns]
    return rep


def shuffled_tokens(tokens: list[str], seed: int = 0) -> list[str]:
    """把记号串整体打乱——给 n-gram 一个**"只剩字母表、没有任何结构"**的地板值。

    ⚠️ 打乱**小节顺序**没用（4-gram 几乎全都保留下来了，实测两边数字一模一样），
    所以这里打乱的是记号本身。
    """
    out = list(tokens)
    random.Random(seed).shuffle(out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt_ngram(rows: list[dict], label: str) -> str:
    bits = [f"{label:<22s}"]
    for r in rows:
        bits.append(f"{r['n']}-gram 种类重合 {r['rate_a_types']:.3f} / "
                    f"出现次数重合 {r['rate_a_instances']:.3f} / Jaccard {r['jaccard']:.3f}")
    return "\n".join([bits[0]] + ["    " + b for b in bits[1:]])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="谱面相似度自检（逐小节 + n-gram）")
    ap.add_argument("ours")
    ap.add_argument("reference")
    ap.add_argument("--baseline", default=None,
                    help="没看过参考谱时写的旧谱，给 n-gram 一个基线")
    ap.add_argument("--control", action="store_true", help="再给一个打乱小节顺序的对照值")
    ap.add_argument("--inote", type=int, default=5)
    ap.add_argument("--ns", default="4,8")
    args = ap.parse_args(argv)
    ns = tuple(int(x) for x in args.ns.split(",") if x.strip())

    ref = load_st_view(args.reference, inote=args.inote).result
    ours = load_st_view(args.ours, inote=args.inote).result
    rep = compare(ours, ref, ns)

    print(f"== 逐小节（共 {rep.n_bars} 小节）==")
    print(f"  完全相同 {rep.n_identical} 个：空小节 {len(rep.identical_empty)}、"
          f"单音小节 {len(rep.identical_single)}、**≥2 note 的小节 {len(rep.identical_multi)}**")
    if rep.identical_empty:
        print(f"    空:   {rep.identical_empty}")
    if rep.identical_single:
        print(f"    单音: {rep.identical_single}")
    for bar in rep.identical_multi:
        print(f"    🚩 m{bar:03d} {rep.identical_detail[bar]}")
    print()
    print("== n-gram 重合（数字必须和基线一起读）==")
    print(_fmt_ngram(rep.ngram, "本谱 vs 参考谱"))
    if args.baseline:
        base = load_st_view(args.baseline, inote=args.inote).result
        brep = compare(base, ref, ns)
        print(_fmt_ngram(brep.ngram, "基线谱 vs 参考谱"))
        print(f"    （基线谱逐小节完全相同 {brep.n_identical} 个，"
              f"其中 ≥2 note 的 {len(brep.identical_multi)} 个）")
    if args.control:
        sa2 = shuffled_tokens(note_sequence(ours))
        sb2 = note_sequence(ref)
        print(_fmt_ngram([ngram_overlap(sa2, sb2, n) for n in ns],
                         "地板：本谱记号打乱后"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
