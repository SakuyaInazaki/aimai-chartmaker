#!/usr/bin/env python3
"""变速重排的两件事：**把 tempo map 写进谱面头部**、**回头逐小节校验误差为 0**。

起因（note 068）：test-01 的前奏与尾奏是变速段，205 只在中段成立。
重排的时候人手去摆 `(bpm)` 很容易摆错一格，而错一格在听感上就是整段偏。
所以把两件事都做成可复跑的：

```bash
# ① 把 tempo map 的 (bpm) 序列写进正文（正文一行一小节、不带 (bpm)）
python3 -m tools.chart_analysis.tempo_remap emit \\
    --tempo out/test-01/tempo_map.json --body body.txt --out body_with_bpm.txt

# ② 校验：逐小节起点秒 vs tempo map 的秒表，误差必须为 0
python3 -m tools.chart_analysis.tempo_remap verify \\
    --tempo out/test-01/tempo_map.json --maidata samples/test-01/maidata.txt
```

**口径**（与 `simai_parser` 同源，见 `tests/test_chart_analysis.py` 的变速三条）：

- 一个逗号槽 = `4/分音` 拍 = `(240/BPM)/分音` 秒，BPM 取**当前段**；
- `(bpm)` 只能出现在逗号之间 ⇒ **一个槽内 BPM 恒定**，小节线落在槽中间时线性插值是精确的；
- 小节 = 4 拍（`&whole_time_signature` 非 4/4 的谱本模块不支持，会直接报错）；
- 谱面时间从**正文起点**算起，音频绝对秒 = 谱面时间 + `&first`。

## tempo map 的两种写法都收

```json
// 本项目 tempo_map v1（out/test-01/tempo_map.json）
{"first_sec": 0.4279, "n_bars": 131,
 "segments": [{"start_bar": 1, "start_beat_in_bar": 0, "bpm": 195.0, "start_sec": 0.4279}, ...],
 "bars": [{"bar": 1, "start_sec": 0.4279, "bpm_at_bar_start": 195.0}, ...]}
// 也收这两种简写
{"first": 1.349, "bars": [{"bar": 1, "bpm": 195.0, "start_sec": 1.349}, ...]}
{"first": 1.349, "segments": [{"start_bar": 1, "end_bar": 9, "bpm": 195.0}, ...]}
```

`bar` 一律 **1 起**（与设计说明的 `mNNN` 同口径），内部转成 0 起。
**换速点只从 `segments[]` 取**（半小节换速只写在那里）；`bars[]` 只提供
`start_sec`（校验基准）与小节数。`start_sec` 缺了就按 BPM 自己累出来。
半小节换速写成 `{"start_bar": 12, "start_beat_in_bar": 2.0, "bpm": 155.0}`
（拍位 0 起；`beat` / `beat_in_bar` 同义）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_BEATS_PER_MEASURE = 4.0
#: 校验容差。变速谱的秒数是浮点累加出来的，1 µs 只用来吃掉累加误差，不是"允许偏差"。
EPS = 1e-6
#: 对**秒表**（tempo map 的 `start_sec`）另给的容差：那一列通常只写到小数点后 4 位，
#: 本身就带 ±0.05 ms 的四舍五入。0.2 ms 只覆盖这个取整，不覆盖任何真实偏差。
SHEET_EPS = 2e-4


@dataclass(frozen=True)
class TempoChange:
    """一次换速。`beat` 是**从正文起点算起的绝对拍位**（4 拍一小节）。"""

    beat: float
    bpm: float

    @property
    def measure(self) -> int:
        return int(self.beat // _BEATS_PER_MEASURE)


@dataclass
class TempoMap:
    first: float
    changes: tuple[TempoChange, ...]
    #: ``{小节号(0 起): 音频绝对秒}``——外部给的基准；没给就是空的
    bar_seconds: dict[int, float]
    n_bars: int

    def bpm_at(self, beat: float) -> float:
        bpm = self.changes[0].bpm
        for c in self.changes:
            if c.beat <= beat + 1e-9:
                bpm = c.bpm
            else:
                break
        return bpm

    def bar_start(self, measure: int) -> float:
        """该小节第 0 拍的**音频绝对秒**（按 BPM 自己累）。"""
        sec = self.first
        beat = 0.0
        target = measure * _BEATS_PER_MEASURE
        edges = sorted({c.beat for c in self.changes} | {target})
        for i, b in enumerate(edges):
            if b >= target - 1e-9:
                break
            nxt = min(edges[i + 1], target) if i + 1 < len(edges) else target
            sec += (nxt - b) * 60.0 / self.bpm_at(b)
            beat = nxt
        if beat < target - 1e-9:
            sec += (target - beat) * 60.0 / self.bpm_at(beat)
        return sec


def load_tempo_map(path: str | Path) -> TempoMap:
    """读 tempo map。三种写法都收，见模块 docstring。

    **换速点只从 `segments[]` 取**（半小节换速只写在那里）；`bars[]` 只用来拿
    `start_sec` 当校验基准与小节数。只有在没有 `segments[]` 时才拿 `bars[]` 当换速点。
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    first = float(d.get("first", d.get("first_sec", 0.0)))
    bar_seconds: dict[int, float] = {}
    n_bars = int(d.get("n_bars", 0) or 0)

    def _bpm(row: dict) -> float:
        for k in ("bpm", "bpm_at_bar_start"):
            if k in row:
                return float(row[k])
        raise KeyError(f"{path}: 这一行既没有 bpm 也没有 bpm_at_bar_start：{row}")

    def _beat_in_bar(row: dict) -> float:
        for k in ("beat", "start_beat_in_bar", "beat_in_bar"):
            if k in row:
                return float(row[k])
        return 0.0

    for row in d.get("bars", []):
        bar = int(row["bar"]) - 1                      # 1 起 → 0 起
        n_bars = max(n_bars, bar + 1)
        if "start_sec" in row:
            bar_seconds[bar] = float(row["start_sec"])

    raw: list[tuple[float, float]] = []
    segs = d.get("segments") or []
    for row in segs:
        a = int(row.get("start_bar", row.get("bar", 1))) - 1
        n_bars = max(n_bars, a + 1)
        raw.append((a * _BEATS_PER_MEASURE + _beat_in_bar(row), _bpm(row)))
        if "start_sec" in row and _beat_in_bar(row) == 0.0:
            bar_seconds.setdefault(a, float(row["start_sec"]))
        if "end_bar" in row:
            n_bars = max(n_bars, int(row["end_bar"]))
    if not raw:                                        # 没有 segments 就退回 bars
        for row in d.get("bars", []):
            bar = int(row["bar"]) - 1
            raw.append((bar * _BEATS_PER_MEASURE + _beat_in_bar(row), _bpm(row)))
    if not raw:
        raise ValueError(f"{path}: 既没有 segments[] 也没有 bars[]")

    raw.sort()
    # 同一拍位重复给 BPM 时取最后一条；BPM 与上一段相同的不重复写
    changes: list[TempoChange] = []
    for beat, bpm in raw:
        if changes and abs(changes[-1].beat - beat) < 1e-9:
            changes[-1] = TempoChange(beat, bpm)
            continue
        if changes and abs(changes[-1].bpm - bpm) < 1e-9:
            continue
        changes.append(TempoChange(beat, bpm))
    if abs(changes[0].beat) > 1e-9:
        raise ValueError(f"{path}: 第一条换速不在第 1 小节第 0 拍（beat={changes[0].beat}）")
    return TempoMap(first=first, changes=tuple(changes),
                    bar_seconds=bar_seconds, n_bars=n_bars or (changes[-1].measure + 1))


# ---------------------------------------------------------------------------
# 正文扫描：逐槽的 (拍位, 字符下标)
# ---------------------------------------------------------------------------


def slot_positions(body: str) -> list[tuple[float, int]]:
    """``[(该槽起点的绝对拍位, 该槽第一个字符在 body 里的下标)]``。

    只认 `{x}` 分音（`{#秒}` 与 `<HS*>` 跳过不计拍）；`||` 注释先剥掉。
    """
    out: list[tuple[float, int]] = []
    div: float | None = None
    beat = 0.0
    #: 本槽的插入锚点 = 上一个逗号之后的位置（正文开头就是 0）。
    #: 取"逗号之后"而不是"第一个 note"，`(bpm)` 才会插在 `{x}` **前面** → `(bpm){x}`。
    slot_head = 0
    i, n = 0, len(body)
    in_comment = False
    while i < n:
        c = body[i]
        if c == "\n":
            in_comment = False
            i += 1
            continue
        if in_comment:
            i += 1
            continue
        if body[i:i + 2] == "||":
            in_comment = True
            i += 2
            continue
        if c in "({":
            close = ")" if c == "(" else "}"
            j = body.find(close, i)
            if j < 0:
                break
            if c == "{":
                tok = body[i + 1:j]
                if tok and not tok.startswith("#"):
                    try:
                        div = float(tok)
                    except ValueError:
                        pass
            i = j + 1
            continue
        if body[i:i + 3] == "<HS":
            j = body.find(">", i)
            i = (j + 1) if j >= 0 else n
            continue
        if c == ",":
            out.append((beat, slot_head))
            if div:
                beat += _BEATS_PER_MEASURE / div
            slot_head = i + 1
            i += 1
            continue
        i += 1
    return out


def emit(body: str, tm: TempoMap) -> str:
    """把 tempo map 的 `(bpm)` 序列插进正文。正文里原有的 `(bpm)` 会**先被剥掉**。"""
    clean = re.sub(r"\((\d+(?:\.\d+)?)\)", "", body)
    slots = slot_positions(clean)
    by_beat = {round(b, 9): pos for b, pos in slots}
    ins: list[tuple[int, str]] = []
    for c in tm.changes:
        pos = by_beat.get(round(c.beat, 9))
        if pos is None:
            raise ValueError(
                f"tempo map 要在第 {c.beat / 4 + 1:.3f} 小节位置换速到 {c.bpm}，"
                f"但正文在拍位 {c.beat} 上没有槽起点——先把那一小节的分音切出这一格")
        ins.append((pos, f"({c.bpm:g})"))
    out = clean
    for pos, txt in sorted(ins, reverse=True):
        out = out[:_anchor(out, pos)] + txt + out[_anchor(out, pos):]
    return out


def _anchor(text: str, pos: int) -> int:
    """把插入点推到该槽的第一个**有效**字符前。

    跳过空白**与 `||` 行尾注释**——不然 `(bpm)` 会卡在上一行的注释前面
    （形如 `…,1,,   (185)||m009`），落到错误的行上。跳过之后拿到的是
    下一行的第一个非空白字符，于是写成官方顺序 `(bpm){x}`
    （`docs/simai-syntax.md` §6 禁 `{x}(bpm)`）。
    """
    n = len(text)
    while pos < n:
        if text[pos] in " \t\r\n　":
            pos += 1
            continue
        if text[pos:pos + 2] == "||":            # 行尾注释：整段跳到下一行
            nl = text.find("\n", pos)
            if nl < 0:
                return n
            pos = nl + 1
            continue
        break
    return pos


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------


def verify(maidata_path: str | Path, tm: TempoMap, inote: int | None = None,
           eps: float = EPS) -> dict:
    """逐小节起点秒 vs tempo map，返回报告 dict（`ok` 为 True 才算过）。"""
    try:                                     # 允许从仓库根或 tools/ 两种路径跑
        from .simai_parser import parse_chart
        from .hands import strip_meta
    except ImportError:                      # pragma: no cover
        from chart_analysis.simai_parser import parse_chart   # type: ignore
        from chart_analysis.hands import strip_meta           # type: ignore

    text = Path(maidata_path).read_text(encoding="utf-8")
    m = re.search(r"^&first=(.*)$", text, re.M)
    first = float(m.group(1).strip()) if (m and m.group(1).strip()) else 0.0
    body = strip_meta(text, inote)
    res = parse_chart("\n".join(l.split("||")[0] for l in body.splitlines()), name="remap")

    # 谱面小节数 = 有 note 的最大小节号 + 1。`measure_starts` 会多带一条"最后一小节之后
    # 的小节线"（128 小节的谱有 0..128 共 129 条），那不是一个小节。
    last = max((n.measure for n in res.notes), default=-1)
    rows = []
    worst = 0.0        # 对"按 BPM 累算"的偏差——这一项必须是 0
    worst_sheet = 0.0  # 对 tempo map 秒表的偏差——只受那一列的取整限制
    for measure, t_chart in sorted(res.measure_starts.items()):
        if measure > last:
            continue
        got = t_chart + first
        want = tm.bar_start(measure)               # 同一套 BPM 累算 ⇒ 可以要求精确
        err = got - want
        worst = max(worst, abs(err))
        sheet = tm.bar_seconds.get(measure)
        derr = (got - sheet) if sheet is not None else None
        if derr is not None:
            worst_sheet = max(worst_sheet, abs(derr))
        rows.append({"bar": measure + 1, "got": round(got, 6),
                     "want": round(want, 6), "err_ms": round(err * 1000, 4),
                     "sheet": None if sheet is None else round(sheet, 6),
                     "sheet_ms": None if derr is None else round(derr * 1000, 4)})
    problems = []
    if abs(first - tm.first) > eps:
        problems.append(f"`&first` 不一致：谱面 {first} / tempo map {tm.first}")
    n_bars = last + 1
    notes = []
    if tm.n_bars > 1 and n_bars > tm.n_bars:
        problems.append(f"谱面比 tempo map **长**：谱面 {n_bars} 小节 / tempo map {tm.n_bars}")
    elif tm.n_bars > 1 and n_bars < tm.n_bars:
        # 谱面收在音乐还没走完的地方是正常的（尾奏留白 / E 提前收），只提示
        notes.append(f"谱面 {n_bars} 小节 < tempo map {tm.n_bars} 小节"
                     f"（末 {tm.n_bars - n_bars} 小节没写 note，`E` 收在 m{n_bars:03d}）")
    over = [r for r in rows if abs(r["err_ms"]) > eps * 1000]
    if over:
        problems.append(f"{len(over)} 个小节起点与**按 BPM 累算**的值对不上，"
                        f"最差 {max(abs(r['err_ms']) for r in over):.4f} ms")
    sheet_over = [r for r in rows
                  if r["sheet_ms"] is not None and abs(r["sheet_ms"]) > SHEET_EPS * 1000]
    if sheet_over:
        problems.append(f"{len(sheet_over)} 个小节与 tempo map 的**秒表**对不上，"
                        f"最差 {max(abs(r['sheet_ms']) for r in sheet_over):.4f} ms"
                        f"（秒表只写到 4 位小数，容差 {SHEET_EPS * 1000:.1f} ms）")
    return {"ok": not problems, "problems": problems, "notes": notes, "n_bars": n_bars,
            "worst_err_ms": round(worst * 1000, 6),
            "worst_sheet_ms": round(worst_sheet * 1000, 6),
            "first": first, "rows": rows}


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="tools.chart_analysis.tempo_remap",
                                 description="变速重排：写 (bpm) / 校验小节起点秒")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="把 tempo map 的 (bpm) 序列插进正文")
    e.add_argument("--tempo", required=True)
    e.add_argument("--body", required=True, help="一行一小节的正文（不带 meta 头）")
    e.add_argument("--out", default=None)
    v = sub.add_parser("verify", help="逐小节起点秒 vs tempo map")
    v.add_argument("--tempo", required=True)
    v.add_argument("--maidata", required=True)
    v.add_argument("--inote", type=int, default=None)
    v.add_argument("--show", type=int, default=12, help="打印前 N 个小节")
    a = ap.parse_args(argv)
    tm = load_tempo_map(a.tempo)

    if a.cmd == "emit":
        out = emit(Path(a.body).read_text(encoding="utf-8"), tm)
        if a.out:
            Path(a.out).write_text(out, encoding="utf-8")
            print(f"写出 {a.out}；&first={tm.first}；插了 {len(tm.changes)} 条 (bpm)")
        else:
            print(f"&first={tm.first}")
            print(out, end="")
        return 0

    rep = verify(a.maidata, tm, a.inote)
    print(f"小节 {rep['n_bars']}｜&first={rep['first']}｜"
          f"对累算 最大误差 {rep['worst_err_ms']:.6f} ms｜"
          f"对秒表 最大误差 {rep['worst_sheet_ms']:.4f} ms")
    for r in rep["rows"][:a.show]:
        sheet = "—" if r["sheet_ms"] is None else f"{r['sheet_ms']:+.4f}"
        print(f"  m{r['bar']:03d} 谱面 {r['got']:9.4f}s  累算 {r['want']:9.4f}s"
              f" 误差 {r['err_ms']:+.4f} ms｜秒表 {sheet} ms")
    for n in rep.get("notes", []):
        print("  ·", n)
    if rep["problems"]:
        for p in rep["problems"]:
            print("  ✗", p)
    print("结论：" + ("**通过**（误差 0）" if rep["ok"] else "**未通过**"))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
