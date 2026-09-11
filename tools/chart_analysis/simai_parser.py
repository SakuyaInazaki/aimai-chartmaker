#!/usr/bin/env python3
"""simai 谱面正文解析器（时间轴 + note 事件）。

依据：`docs/simai-syntax.md`（v1.0 定稿）。本模块只解析**谱面正文**
（`&inote_N=` 之后的部分，即 `resource/official-chart/*.txt` 的文件内容），
不处理 meta 头；若传入带 meta 的 maidata.txt，请先自行切出正文。

设计要点
--------
1. **时间轴以"拍"（四分音符）为主轴**：一个逗号推进 ``4 / 分音`` 拍，
   秒数推进 ``(240 / BPM) / 分音``。这样 BPM 变速不会破坏小节划分。
2. **小节 = 4 拍**（simai 无小节线、无拍号指令，隐含 4/4，见语法文档 §1.3）。
   小节号 = ``floor(拍位置 / 4)``，从 0 起。
3. **不静默跳过**：无法识别的字符会记入 ``ParseResult.errors``，
   同时尽力继续解析，便于批量扫描时给出覆盖率与失败原因。

口径（note 计数）
----------------
参照 mai-notes / 官方查分器的分项口径（见 ``docs/research/`` 与知识 004）：

- **TAP**：单点；**slide 的星星头也计入 TAP**（一条多头 slide ``*`` 只有一个星头）。
- **HOLD**：``h`` 系列。
- **SLIDE**：每条滑轨算 1；``*`` 分隔的同头多 slide 每条各算 1；
  ``1-4q7-2[1:2]`` 这类首尾相接的**连锁 slide 整体算 1**
  （分段口径的总数另存于 ``ParseResult.slide_chain_segments``，供对照）。
- **BREAK**：带 ``b`` 的 note **从其本类中移出**、单独计入 BREAK
  （即 ``notes = taps + hold + slide + touch + breaks``，与 manifest 字段一致）。
- **TOUCH**：ST 谱理论上不含 touch；若遇到照常统计并单列。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# 滑条形状字符（单字符）；``pp`` / ``qq`` 为双字符形状，另行判定
_SHAPE_SINGLE = set("-^<>vVpqszw")
_MODIFIERS = set("bx!?$@")
_DIGITS = set("12345678")
_TOUCH_AREAS = set("ABCDE")
_BEATS_PER_MEASURE = 4.0


class SimaiParseError(Exception):
    """解析过程中的致命错误（无法继续推进时间轴）。"""


@dataclass
class NoteEvent:
    """单个 note 事件。"""

    time: float  # 绝对时间（秒），谱面正文起点为 0
    beat: float  # 绝对拍位置（四分音符为 1 拍）
    measure: int  # 小节号，从 0 起
    beat_in_measure: float  # 小节内拍位置 [0, 4)
    kind: str  # tap / hold / slide_star / slide_track / touch / touch_hold
    key: str  # 键位 '1'..'8'，或 touch 区（如 'C', 'A3'）
    is_break: bool = False
    is_ex: bool = False
    is_each: bool = False  # 所属时间槽内同刻 note 数 >= 2
    group_index: int = 0  # 时间槽序号（同一个逗号前的所有 note 共享）
    duration: float = 0.0  # hold 时长 / slide 移动时长（秒）
    wait: float = 0.0  # slide 启动拍等待（秒）
    shape: str = ""  # slide 形状串，如 '-' / 'V' / 'pp'
    end_key: str = ""  # slide 终点键
    bpm: float = 0.0  # 该 note 所处 BPM
    divisor: float = 0.0  # 该 note 所处分音


@dataclass
class ParseResult:
    """一份谱面的解析结果。"""

    notes: list[NoteEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bpm_events: list[tuple[float, float]] = field(default_factory=list)  # (beat, bpm)
    # 速度分段 (起始拍, 起始秒, 每拍秒数)，用于把任意拍位置换算成秒
    tempo_segments: list[tuple[float, float, float]] = field(default_factory=list)
    total_beats: float = 0.0
    total_seconds: float = 0.0
    has_end_marker: bool = False
    slide_chain_segments: int = 0  # 连锁 slide 的**分段**总数（对照口径用）
    each_groups: int = 0  # 含 >= 2 个 note 的时间槽个数

    # ---- 计数口径 ----
    @property
    def counts(self) -> dict[str, int]:
        """返回官方分项口径计数：taps / hold / slide / touch / breaks / notes。"""
        taps = hold = slide = touch = breaks = 0
        for n in self.notes:
            if n.is_break:
                breaks += 1
                continue
            if n.kind in ("tap", "slide_star"):
                taps += 1
            elif n.kind == "hold":
                hold += 1
            elif n.kind == "slide_track":
                slide += 1
            elif n.kind == "touch":
                touch += 1
            elif n.kind == "touch_hold":
                touch += 1
        return {
            "taps": taps,
            "hold": hold,
            "slide": slide,
            "touch": touch,
            "breaks": breaks,
            "notes": taps + hold + slide + touch + breaks,
        }

    @property
    def measure_count(self) -> int:
        if not self.notes:
            return 0
        return max(n.measure for n in self.notes) + 1


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    """去掉 ``||`` 行注释（到行尾）。"""
    out = []
    for line in text.splitlines():
        idx = line.find("||")
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


_WS_RE = re.compile(r"[\s　]+")


def _normalize(text: str) -> str:
    text = _strip_comments(text)
    return _WS_RE.sub("", text)


# ---------------------------------------------------------------------------
# 时长括号 [ ... ]
# ---------------------------------------------------------------------------


def _read_bracket(s: str, i: int) -> tuple[str, int]:
    """读取 ``[...]``，返回（括号内文本, 新下标）。``i`` 指向 ``[``。"""
    j = s.find("]", i)
    if j < 0:
        raise SimaiParseError(f"位置 {i}: '[' 未闭合")
    return s[i + 1 : j], j + 1


def parse_duration(body: str, bpm: float) -> tuple[float, float | None]:
    """解析时长括号内容，返回 ``(移动/持续秒数, 等待秒数或 None)``。

    支持语法文档 §4.4 / §4.6 的全部形式：

    ====================  ==========================================
    写法                   含义
    ====================  ==========================================
    ``x:y``               当前 BPM 的 x 分音 × y
    ``#秒``               直接秒数
    ``BPM#x:y``           指定 BPM 的 x 分音 × y
    ``BPM#秒``            等待用指定 BPM 一拍，移动为直接秒数
    ``秒##秒``            等待秒 + 移动秒
    ``秒##x:y``           等待秒 + 当前 BPM x 分音 × y
    ``秒##BPM#x:y``       等待秒 + 指定 BPM x 分音 × y
    ====================  ==========================================
    """
    body = body.strip()
    if not body:
        raise SimaiParseError("空时长括号 []")

    def ratio(expr: str, use_bpm: float) -> float:
        x_s, _, y_s = expr.partition(":")
        x, y = float(x_s), float(y_s)
        if x == 0:
            raise SimaiParseError(f"时长分音为 0: [{body}]")
        return (240.0 / use_bpm / x) * y

    if "##" in body:
        head, _, tail = body.partition("##")
        wait = float(head)
        if "#" in tail:  # 秒##BPM#x:y
            b_s, _, rest = tail.partition("#")
            return ratio(rest, float(b_s)), wait
        if ":" in tail:  # 秒##x:y
            return ratio(tail, bpm), wait
        return float(tail), wait  # 秒##秒

    if body.startswith("#"):  # #秒
        return float(body[1:]), None

    if "#" in body:  # BPM#x:y 或 BPM#秒
        b_s, _, rest = body.partition("#")
        use_bpm = float(b_s)
        if ":" in rest:
            return ratio(rest, use_bpm), 60.0 / use_bpm
        return float(rest), 60.0 / use_bpm

    if ":" in body:  # x:y
        return ratio(body, bpm), None

    # ``[秒]`` 无冒号单值：官方未定义、MajdataEdit 不认，但容错解析为秒
    return float(body), None


# ---------------------------------------------------------------------------
# 时间槽（逗号之间的 note 组）解析
# ---------------------------------------------------------------------------


def _is_shape_start(s: str, i: int) -> bool:
    return i < len(s) and s[i] in _SHAPE_SINGLE


def _read_shape(s: str, i: int) -> tuple[str, int]:
    """读取形状串，处理 ``pp`` / ``qq`` 双字符形状。"""
    c = s[i]
    if c in "pq" and i + 1 < len(s) and s[i + 1] == c:
        return c * 2, i + 2
    return c, i + 1


@dataclass
class _RawNote:
    kind: str
    key: str
    is_break: bool = False
    is_ex: bool = False
    duration: float = 0.0
    wait: float = 0.0
    shape: str = ""
    end_key: str = ""


def _parse_member(s: str, bpm: float, res: ParseResult, where: str) -> list[_RawNote]:
    """解析一个 each 成员（``/`` 分隔的一段），可能产出多个 note。"""
    notes: list[_RawNote] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]

        # ---- Touch / Touch-Hold ----
        if c in _TOUCH_AREAS:
            key = c
            i += 1
            if i < n and s[i].isdigit():
                key += s[i]
                i += 1
            flags = set()
            while i < n and s[i] in "fhbx":
                flags.add(s[i])
                i += 1
            dur = 0.0
            if i < n and s[i] == "[":
                body, i = _read_bracket(s, i)
                dur, _ = parse_duration(body, bpm)
            while i < n and s[i] in "fhbx":
                flags.add(s[i])
                i += 1
            notes.append(
                _RawNote(kind="touch_hold" if "h" in flags else "touch", key=key, duration=dur)
            )
            continue

        if c not in _DIGITS:
            res.errors.append(f"{where}: 无法识别的字符 {c!r}（片段 {s!r}）")
            i += 1
            continue

        key = c
        i += 1
        flags: set[str] = set()
        while i < n and s[i] in _MODIFIERS:
            flags.add(s[i])
            i += 1

        # ---- HOLD ----
        if i < n and s[i] == "h":
            i += 1
            while i < n and s[i] in _MODIFIERS:
                flags.add(s[i])
                i += 1
            dur = 0.0
            if i < n and s[i] == "[":
                body, i = _read_bracket(s, i)
                dur, _ = parse_duration(body, bpm)
                while i < n and s[i] in "bx":
                    flags.add(s[i])
                    i += 1
            else:
                # ``3h`` 省略时长 = 伪 TAP，内部等价 [1280:1]
                dur = 240.0 / bpm / 1280.0
            notes.append(
                _RawNote(
                    kind="hold", key=key, is_break="b" in flags, is_ex="x" in flags, duration=dur
                )
            )
            continue

        # ---- 带时长括号但无 h 的 TAP ----
        # 官方语法不允许；SimaiSharp 将其视为 Hold（语法文档 §4.4 末注），此处从之并告警
        if i < n and s[i] == "[":
            body, i = _read_bracket(s, i)
            dur, _ = parse_duration(body, bpm)
            while i < n and s[i] in "bx":
                flags.add(s[i])
                i += 1
            res.warnings.append(f"{where}: {key}[{body}] 缺少 'h'，按 HOLD 处理")
            notes.append(
                _RawNote(
                    kind="hold", key=key, is_break="b" in flags, is_ex="x" in flags, duration=dur
                )
            )
            continue

        # ---- SLIDE ----
        if _is_shape_start(s, i):
            star_break = "b" in flags
            star_ex = "x" in flags
            chains, i = _parse_slide_chains(s, i, key, bpm, res, where)
            # 星头：``*`` 多头 slide 共用一个星头
            notes.append(
                _RawNote(kind="slide_star", key=key, is_break=star_break, is_ex=star_ex)
            )
            for chain in chains:
                res.slide_chain_segments += len(chain["segments"])
                notes.append(
                    _RawNote(
                        kind="slide_track",
                        key=key,
                        is_break=chain["is_break"],
                        duration=chain["duration"],
                        wait=chain["wait"] if chain["wait"] is not None else 60.0 / bpm,
                        shape="".join(seg[0] for seg in chain["segments"]),
                        end_key=chain["segments"][-1][1] if chain["segments"] else "",
                    )
                )
            continue

        # ---- TAP ----
        notes.append(
            _RawNote(kind="tap", key=key, is_break="b" in flags, is_ex="x" in flags)
        )
    return notes


def _parse_slide_chains(
    s: str, i: int, head_key: str, bpm: float, res: ParseResult, where: str
) -> tuple[list[dict], int]:
    """解析从 ``i`` 开始的 slide 部分，返回（chain 列表, 新下标）。

    一个 chain = 一条滑轨（可能由多段首尾相接组成）；``*`` 开启新的 chain。
    """
    chains: list[dict] = []
    cur: dict = {"segments": [], "duration": 0.0, "wait": None, "is_break": False}
    n = len(s)
    while True:
        if not _is_shape_start(s, i):
            break
        shape, i = _read_shape(s, i)
        # 容错：``6>b3[8:1]`` 这类把 b/x 写在形状与终点键之间的野写法（官方谱中出现过 1 例）
        while i < n and s[i] in "bx":
            if s[i] == "b":
                cur["is_break"] = True
            res.warnings.append(f"{where}: slide 修饰符 {s[i]!r} 写在终点键之前，已容错")
            i += 1
        if shape == "V":  # 三键记法 1V36
            if i + 1 >= n:
                raise SimaiParseError(f"{where}: V 形 slide 缺少键位")
            end_key = s[i + 1]
            i += 2
        else:
            if i >= n:
                raise SimaiParseError(f"{where}: slide 缺少终点键")
            end_key = s[i]
            i += 1
        # 时长括号与 b/x 修饰符的先后顺序两种写法都接受
        # （官方形式 ``1-4[8:3]b``；MajdataView 形式 ``1-4b[4:1]``，语法文档 §4.6）
        while i < n and s[i] in "bx[":
            if s[i] == "[":
                body, i = _read_bracket(s, i)
                move, wait = parse_duration(body, bpm)
                cur["duration"] += move
                if wait is not None:
                    cur["wait"] = wait
            else:
                if s[i] == "b":
                    cur["is_break"] = True
                i += 1
        cur["segments"].append((shape, end_key))

        if i < n and s[i] == "*":  # 同头多 slide
            i += 1
            chains.append(cur)
            cur = {"segments": [], "duration": 0.0, "wait": None, "is_break": False}
            continue
        if _is_shape_start(s, i):  # 连锁 slide，继续下一段
            continue
        break
    if cur["segments"]:
        chains.append(cur)
    return chains, i


# ---------------------------------------------------------------------------
# 主解析
# ---------------------------------------------------------------------------


def parse_chart(text: str, *, name: str = "<chart>") -> ParseResult:
    """解析 simai 谱面正文，返回 :class:`ParseResult`。"""
    res = ParseResult()
    src = _normalize(text)

    # 末尾 E 结束标记（区别于 Touch 的 E 区：E 区必带数字）
    if src.endswith("E"):
        res.has_end_marker = True
        src = src[:-1]
    else:
        res.warnings.append(f"{name}: 正文未以大写 E 结尾")

    bpm: float | None = None
    divisor: float | None = None
    beat = 0.0
    seconds = 0.0
    group_index = 0

    i = 0
    n = len(src)
    buf: list[str] = []  # 当前时间槽累积的 note 文本

    def flush(where: str) -> None:
        nonlocal group_index
        raw = "".join(buf)
        buf.clear()
        if not raw:
            return
        if bpm is None:
            res.errors.append(f"{where}: 出现 note 但尚未指定 BPM")
            return
        members = [m for m in re.split(r"[/`]", raw) if m]
        raws: list[_RawNote] = []
        for m in members:
            try:
                raws.extend(_parse_member(m, bpm, res, where))
            except (SimaiParseError, ValueError, IndexError) as exc:
                res.errors.append(f"{where}: 成员 {m!r} 解析失败: {exc}")
        if not raws:
            return
        # 同刻 >= 2 个 note 即构成 EACH（星头与其滑轨同刻，也计入）
        is_each = len(raws) >= 2
        if is_each:
            res.each_groups += 1
        measure = int(beat // _BEATS_PER_MEASURE)
        bim = beat - measure * _BEATS_PER_MEASURE
        for r in raws:
            res.notes.append(
                NoteEvent(
                    time=seconds,
                    beat=beat,
                    measure=measure,
                    beat_in_measure=bim,
                    kind=r.kind,
                    key=r.key,
                    is_break=r.is_break,
                    is_ex=r.is_ex,
                    is_each=is_each,
                    group_index=group_index,
                    duration=r.duration,
                    wait=r.wait,
                    shape=r.shape,
                    end_key=r.end_key,
                    bpm=bpm,
                    divisor=divisor if divisor else 0.0,
                )
            )
        group_index += 1

    seconds_per_slot_override: float | None = None  # ``{#秒}``

    while i < n:
        c = src[i]
        if c == "(":  # BPM
            j = src.find(")", i)
            if j < 0:
                res.errors.append(f"{name}@{i}: '(' 未闭合")
                break
            try:
                bpm = float(src[i + 1 : j])
            except ValueError:
                res.errors.append(f"{name}@{i}: 非法 BPM {src[i + 1 : j]!r}")
            else:
                if bpm <= 0:
                    res.errors.append(f"{name}@{i}: BPM 必须为正数")
                    bpm = None
                else:
                    res.bpm_events.append((beat, bpm))
                    res.tempo_segments.append((beat, seconds, 60.0 / bpm))
            i = j + 1
            continue
        if c == "<":
            # ``<HS*x>`` 速度标记（Majdata 扩展）；``<`` 也是 slide 形状，需区分
            if src[i : i + 3] == "<HS":
                j = src.find(">", i)
                if j < 0:
                    res.errors.append(f"{name}@{i}: '<HS' 未闭合")
                    break
                i = j + 1
                continue
            buf.append(c)
            i += 1
            continue
        if c == "{":  # 分音 / 秒数槽
            j = src.find("}", i)
            if j < 0:
                res.errors.append(f"{name}@{i}: '{{' 未闭合")
                break
            body = src[i + 1 : j]
            if not body:
                res.warnings.append(f"{name}@{i}: 空分音标记 '{{}}'，沿用上一分音")
                i = j + 1
                continue
            if body.startswith("#"):
                try:
                    seconds_per_slot_override = float(body[1:])
                except ValueError:
                    res.errors.append(f"{name}@{i}: 非法秒数槽 {body!r}")
                divisor = None
            else:
                try:
                    d = float(body)
                except ValueError:
                    res.errors.append(f"{name}@{i}: 非法分音 {body!r}")
                else:
                    if d <= 0:
                        res.errors.append(f"{name}@{i}: 分音必须为正数")
                    else:
                        divisor = d
                        seconds_per_slot_override = None
            i = j + 1
            continue
        if c == ",":  # 推进一个时间槽
            where = f"{name}@{i}"
            flush(where)
            if seconds_per_slot_override is not None:
                if bpm is None:
                    res.errors.append(f"{where}: '{{#秒}}' 槽推进但尚未指定 BPM，无法折算拍位")
                    dt = seconds_per_slot_override
                    db = 0.0
                else:
                    dt = seconds_per_slot_override
                    db = dt * bpm / 60.0
            else:
                if bpm is None or divisor is None:
                    res.errors.append(f"{where}: 逗号推进但 BPM/分音未就绪")
                    i += 1
                    continue
                db = 4.0 / divisor
                dt = (240.0 / bpm) / divisor
            beat += db
            seconds += dt
            i += 1
            continue
        buf.append(c)
        i += 1

    flush(f"{name}@EOF")
    res.total_beats = beat
    res.total_seconds = seconds
    return res


def beat_to_time(res: ParseResult, target_beat: float) -> float:
    """把拍位置换算成秒（按 BPM 分段线性推算）。"""
    if not res.tempo_segments:
        return 0.0
    seg = res.tempo_segments[0]
    for s in res.tempo_segments:
        if s[0] <= target_beat:
            seg = s
        else:
            break
    start_beat, start_time, spb = seg
    return start_time + (target_beat - start_beat) * spb


def iter_measures(res: ParseResult) -> Iterable[tuple[int, list[NoteEvent]]]:
    """按小节号分组产出 ``(小节号, note 列表)``，含空小节。"""
    if not res.notes:
        return
    last = max(n.measure for n in res.notes)
    buckets: dict[int, list[NoteEvent]] = {m: [] for m in range(last + 1)}
    for note in res.notes:
        buckets[note.measure].append(note)
    for m in range(last + 1):
        yield m, buckets[m]
