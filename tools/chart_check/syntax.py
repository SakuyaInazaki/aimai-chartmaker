#!/usr/bin/env python3
"""语法层：零报错安全子集 + ST 要素黑名单，逐条报错/警告（带行列）。

规则来源（一条不自造）：

- `docs/simai-syntax.md` **§6 零报错安全子集**（禁用清单 7 组）与 §4 记法定义；
- `docs/simai-error-checking.md` **§2 MajdataEdit SyntaxCheck**（15 组判定，含 slide
  端点几何表）、**§8.2 MiaCode 34 类规则**（全角字符 7 类、分隔符、分段时值）、
  §4 SimaiSharp 异常体系；
- `docs/st-chart-elements.md` **§三 ST 谱面要素黑名单**（DX / FESTiVAL 特性）。

⚠️ **这一层只是内部校验。** `docs/simai-error-checking.md` §10-4 要求的
**SimaiSharp lint + MajdataEdit SyntaxCheck + MiaCode strict 三检**需要实跑那三个工具，
本机没有装 —— 本模块按它们的**规则交集**实现，**不能代替真机跑谱**。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .maidata import MaiData, line_col
from .model import LayerResult

# --- 记法常量（与 chart_analysis.simai_parser 同源） ---
SHAPES_TWO = ("qq", "pp")
SHAPES_ONE = set("-^<>vVpqszw")
DIGITS = set("12345678")
TOUCH_AREAS = set("ABCDE")
#: 384 的全部约数（`docs/simai-syntax.md` §1.4）
DIV_384 = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384}

_WS_RE = re.compile(r"[\s　]+")


def cdist(a: int, b: int) -> int:
    """环距（1..8 的键位圈上两键的最短间隔）。"""
    d = abs(a - b) % 8
    return min(d, 8 - d)


# ---------------------------------------------------------------------------
# 规范化（复刻 simai_parser 的口径，但带「规范化下标 → 原文下标」映射）
# ---------------------------------------------------------------------------


def normalize_with_map(body: str) -> tuple[str, list[int]]:
    """去 `||` 注释、去空白，返回 ``(规范化串, 每个字符在 body 里的下标)``。"""
    out: list[str] = []
    idx: list[int] = []
    pos = 0
    for line in body.split("\n"):
        cut = line.find("||")
        keep = line if cut < 0 else line[:cut]
        for k, ch in enumerate(keep):
            if _WS_RE.match(ch):
                continue
            out.append(ch)
            idx.append(pos + k)
        pos += len(line) + 1
    return "".join(out), idx


@dataclass
class _Ctx:
    """一次语法检查的位置上下文。"""

    text: str          # 整个 maidata.txt
    body_off: int      # 正文在文件里的字符偏移
    idx: list[int]     # 规范化下标 → body 内下标

    def at(self, k: int) -> tuple[int, int]:
        if not self.idx:
            return 0, 0
        k = max(0, min(k, len(self.idx) - 1))
        return line_col(self.text, self.body_off + self.idx[k])


# ---------------------------------------------------------------------------
# 全角字符（MiaCode 7 类；LLM 生成文本高危项）
# ---------------------------------------------------------------------------

_FULLWIDTH_HINT = {
    "０１２３４５６７８９": "全角数字",
    "ＡＢＣＤＥ": "全角触摸字母",
    "（）｛｝［］": "全角括号",
    "，、／｀": "全角分隔符",
    "－＾＜＞ｖＶｐｑｓｚｗ": "全角 slide 符号",
    "ｂｘｈｆ": "全角修饰符",
}


def _fullwidth_kind(ch: str) -> str:
    for chars, name in _FULLWIDTH_HINT.items():
        if ch in chars:
            return name
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return "全角字符"
    return ""


# ---------------------------------------------------------------------------
# meta 层
# ---------------------------------------------------------------------------

_NEED_META = ("title", "artist", "des", "first")
#: `docs/simai-syntax.md` §2.2：meta 文本里的半角 `&` `+` `%` `\` 必须转义
_META_RAW = "&+%\\"

#: 只有**文本类** meta 受转义规则约束（`docs/simai-syntax.md` §2.2：
#: 「meta **文本**中的半角 `&` `+` `%` `\` 必须写成 `\＆` `\＋` `\％` `\￥`」）。
#: `&lv_N=` 是**等级值**不是文本，同节明写「可写 `13+`（+半角）」——
#: 早先版本把 `&lv_5=13+` 报成未转义是**误报**（note 065 记录的工具缺口）。
_TEXT_META_KEYS = ("title", "artist", "des")


def _is_text_meta(key: str) -> bool:
    """`title` / `artist` / `des` / `des_N` 才走转义检查。"""
    return key in _TEXT_META_KEYS or key.startswith("des_")


def check_meta(md: MaiData, out: LayerResult) -> None:
    keys = {m.key for m in md.metas}
    for k in _NEED_META:
        if k not in keys:
            out.add("警告", "SYN-META-MISSING",
                    f"缺少 meta `&{k}=`（生成器最低输出：title/artist/des/first）",
                    source="simai-syntax §2.2")
    first = md.meta("first")
    if first is not None and not first.strip():
        out.add("错误", "SYN-META-FIRST-EMPTY",
                "`&first=` 为空——MiaCode 记录空 first 会让 MajdataPlay 的 double.Parse 崩溃，"
                "至少写 `&first=0`",
                line=[m.line for m in md.metas if m.key == "first"][0],
                source="simai-error-checking §8.3")
    for m in md.metas:
        if not _is_text_meta(m.key):
            continue
        for k, ch in enumerate(m.value):
            if ch in _META_RAW:
                # `\＆` 这种转义形式：半角 \ 后跟全角字符
                if ch == "\\" and k + 1 < len(m.value) and m.value[k + 1] in "＆＋％￥":
                    continue
                out.add("错误", "SYN-META-ESCAPE",
                        f"meta `&{m.key}=` 的值里有未转义的半角 {ch!r}"
                        "（应写成 `\\＆` `\\＋` `\\％` `\\￥`）",
                        line=m.line, excerpt=m.value.splitlines()[0][:80],
                        source="simai-syntax §2.2")
                break


# ---------------------------------------------------------------------------
# 正文：结构级扫描（注释 / 括号 / BPM / 分音 / E）
# ---------------------------------------------------------------------------


def check_comments_and_brackets(body: str, ctx: _Ctx, out: LayerResult) -> None:
    lines = body.split("\n")
    pos = 0
    #: `(行, 列, 原文, 下一行是否以 `{` 开头)`——VM 兼容风险汇总用
    comment_lines: list[tuple[int, int, str, bool]] = []
    for li, line in enumerate(lines):
        # 单个 `|`（SimaiSharp 直接抛 UnexpectedCharacterException）
        j = 0
        while j < len(line):
            if line[j] == "|":
                if line[j:j + 2] == "||":
                    break  # 到行尾都是注释
                ln, col = line_col(ctx.text, ctx.body_off + pos + j)
                out.add("错误", "SYN-PIPE",
                        "出现单个 `|`——SimaiSharp 抛 UnexpectedCharacterException；"
                        "注释要写 `||` 且以换行结尾",
                        line=ln, col=col, excerpt=line.strip()[:80],
                        source="simai-syntax §3.6")
                break
            j += 1
        cut = line.find("||")
        if cut >= 0 and li == len(lines) - 1 and not body.endswith("\n"):
            ln, col = line_col(ctx.text, ctx.body_off + pos + cut)
            out.add("警告", "SYN-COMMENT-EOF",
                    "`||` 注释在文件末尾且没有换行结尾——不换行时注释不生效",
                    line=ln, col=col, source="simai-syntax §3.6")
        if cut >= 0:
            nxt = lines[li + 1].lstrip() if li + 1 < len(lines) else ""
            ln, col = line_col(ctx.text, ctx.body_off + pos + cut)
            comment_lines.append((ln, col, line.strip()[:80], nxt.startswith("{")))
        pos += len(line) + 1

    if comment_lines:
        ln, col, excerpt, _ = comment_lines[0]
        n_swallow = sum(1 for *_, hit in comment_lines if hit)
        out.add("错误", "SYN-VM-COMMENT",
                f"**Visual Maimai 兼容风险**：正文里有 **{len(comment_lines)} 行 `||` 行尾注释**"
                f"（其中 **{n_swallow}** 行的下一行以 `{{` 开头）。"
                "VM 会**吞掉 `||` 注释行之后那一行行首的 `{x}` 分音标记**，整段按上一个分音重算"
                "——test-01 实测：m004–m058 的 `{8}` 被当成 `{16}` 走，"
                "谱面从 128 小节 / 149.9 s 缩成 108.5 小节，note 位置全错位。"
                "官方谱 414 份里 **405 份有行首 `{x}`、0 份含 `||`**，"
                "所以触发条件是注释而不是行首分音。"
                "**交付版必须是官方格式：一行一小节、行首 `{x}` 可留、`||` 一个不留**；"
                "设计意图注释另存（`out/<曲名>/maidata-annotated.txt` 或 writeup 小节表）。",
                line=ln, col=col, excerpt=excerpt,
                source="simai-error-checking §9.1")

    s, _ = normalize_with_map(body)
    for op, cl, code in (("(", ")", "SYN-PAREN"), ("{", "}", "SYN-BRACE"),
                         ("[", "]", "SYN-SQUARE")):
        if s.count(op) != s.count(cl):
            out.add("错误", code,
                    f"`{op}` 与 `{cl}` 数量不匹配（{s.count(op)} vs {s.count(cl)}）",
                    source="simai-error-checking §2 判定 2/3/7")


def check_directives(s: str, ctx: _Ctx, out: LayerResult) -> None:
    """BPM / 分音标记：顺序、连写、取值、384 约数。"""
    toks: list[tuple[str, int, str]] = []   # (kind, 起始下标, 内容)
    i, n = 0, len(s)
    first_note_at = -1
    while i < n:
        c = s[i]
        if c == "(":
            j = s.find(")", i)
            if j < 0:
                break
            toks.append(("bpm", i, s[i + 1:j]))
            i = j + 1
            continue
        if c == "{":
            j = s.find("}", i)
            if j < 0:
                break
            toks.append(("div", i, s[i + 1:j]))
            i = j + 1
            continue
        if c in DIGITS or c in TOUCH_AREAS:
            if first_note_at < 0:
                first_note_at = i
        i += 1

    prev_kind = ""
    prev_end = -1
    seen_bpm = False
    seen_div = False
    for kind, pos, bodytxt in toks:
        ln, col = ctx.at(pos)
        end_pos = pos + len(bodytxt) + 2      # 含两个括号
        # ⚠️ 只在**文本上真正紧挨着**时才算"连写"：`{8}1,2,{16}` 这种中间隔着 note
        # 的当然不是连写（官谱每首都这么写）
        if kind == prev_kind and pos == prev_end:
            out.add("错误", "SYN-DOUBLE-DIRECTIVE",
                    f"连续两个{'BPM' if kind == 'bpm' else '分音'}标记"
                    "（`docs/simai-syntax.md` §3.1/§3.2：不要连写）",
                    line=ln, col=col, source="simai-syntax §3.1/§3.2")
        if kind == "bpm":
            seen_bpm = True
            try:
                v = float(bodytxt)
            except ValueError:
                out.add("错误", "SYN-BPM-VALUE", f"非法 BPM `({bodytxt})`",
                        line=ln, col=col, source="simai-error-checking §2 判定 5")
            else:
                if v <= 0:
                    out.add("错误", "SYN-BPM-VALUE", f"BPM 必须为正数（读到 {v}）",
                            line=ln, col=col, source="simai-error-checking §8.2")
            if prev_kind == "div" and pos == prev_end:
                out.add("错误", "SYN-ORDER",
                        "`{分音}` 紧接着写在 `(BPM)` 之前——官方要求 BPM 在前"
                        "（`(120){4}` ✓ / `{4}(120)` ✗）",
                        line=ln, col=col, source="simai-syntax §3.1")
        else:
            seen_div = True
            if bodytxt.startswith("#"):
                out.add("错误", "SYN-SECONDS-SLOT",
                        f"秒数槽 `{{{bodytxt}}}`——MajdataEdit 校验器要求 `{{}}` 内为整数，"
                        "安全子集禁用，改用 `(bpm){x}`",
                        line=ln, col=col, source="simai-syntax §6-3")
            elif not bodytxt:
                out.add("警告", "SYN-DIV-EMPTY", "空分音标记 `{}`", line=ln, col=col,
                        source="simai-error-checking §2 判定 5")
            else:
                try:
                    v = float(bodytxt)
                except ValueError:
                    out.add("错误", "SYN-DIV-VALUE", f"非法分音 `{{{bodytxt}}}`",
                            line=ln, col=col, source="simai-error-checking §2 判定 5")
                else:
                    if v <= 0 or v != int(v):
                        out.add("错误", "SYN-DIV-VALUE",
                                f"分音必须是正整数（读到 {bodytxt}）",
                                line=ln, col=col, source="simai-error-checking §8.2")
                    elif int(v) not in DIV_384:
                        out.add("警告", "SYN-DIV-384",
                                f"分音 `{{{int(v)}}}` 不是 384 的约数"
                                "（MiaCode strict 报 Warning；安全子集要求是约数）",
                                line=ln, col=col, source="simai-syntax §6-5")
        prev_kind, prev_end = kind, end_pos

    if first_note_at >= 0:
        pre = [t for t in toks if t[1] < first_note_at]
        if not any(t[0] == "bpm" for t in pre):
            ln, col = ctx.at(first_note_at)
            out.add("错误", "SYN-NO-BPM", "第一个 note 之前没有 `(BPM)` 标记",
                    line=ln, col=col, source="simai-syntax §6-1")
        elif not any(t[0] == "div" for t in pre):
            ln, col = ctx.at(first_note_at)
            out.add("错误", "SYN-NO-DIV", "第一个 note 之前没有 `{分音}` 标记",
                    line=ln, col=col, source="simai-syntax §6-1")
    if not seen_bpm and toks:
        out.add("错误", "SYN-NO-BPM", "正文里没有任何 `(BPM)` 标记",
                source="simai-syntax §6-1")


def check_end_marker(s: str, ctx: _Ctx, out: LayerResult) -> None:
    if s.endswith("E"):
        return
    if s.endswith("e"):
        ln, col = ctx.at(len(s) - 1)
        out.add("错误", "SYN-LOWER-E", "结束标记写成了小写 `e`（必须大写 `E`）",
                line=ln, col=col, source="simai-syntax §3.5")
        return
    out.add("错误", "SYN-NO-E",
            "正文没有以大写 `E` 结尾——MajdataEdit 报「[警告] 谱面应当以\"E\"结尾」，"
            "安全子集要求必须写",
            source="simai-syntax §6-3")


# ---------------------------------------------------------------------------
# 正文：成员级扫描（note 记法 / ST 黑名单 / slide 几何）
# ---------------------------------------------------------------------------

#: DX / FESTiVAL 要素 → (code, 中文说明)。`docs/st-chart-elements.md` §三
ST_BLACKLIST = {
    "x": ("ST-EX", "EX（保护）`x` 是 DX 要素，ST 谱禁用；SimaiSharp 对 slide 上的 `x` 直接抛异常"),
    "!": ("ST-NOHEAD", "无头星星 `!` 是 3simai/DX 工具扩展，ST 谱 0 使用"),
    "?": ("ST-NOHEAD", "无头星星 `?` 是 3simai/DX 工具扩展，ST 谱 0 使用"),
    "$": ("ST-STARTAP", "星形 tap `$`/`$$` 是工具扩展，ST 谱 0 使用；安全子集默认不用"),
    "@": ("ST-RESTORE", "还原符 `@` MajdataEdit 不支持，安全子集禁用"),
}


def _read_shape(s: str, i: int) -> tuple[str, int]:
    c = s[i]
    if c in "pq" and i + 1 < len(s) and s[i + 1] == c:
        return c * 2, i + 2
    return c, i + 1


def _slide_geometry(shape: str, head: int, pivot: int | None,
                   end: int) -> tuple[str, str]:
    """返回 ``(级别, 说明)``；合法返回 ``("", "")``。

    依据 `docs/simai-error-checking.md` §2 的 MajdataEdit / MajdataView 端点几何表；
    `v` 取**官方口径**（禁自身 + 对向），比 MajdataView 严。
    ``<`` / ``>`` 官方**没有硬限制**，同键 = 绕一整圈，官谱写过（`3<3` / `7>7`），
    所以只给提示。
    """
    d = cdist(head, end)
    if shape == "-":
        if head == end or d < 2:
            return "错误", "`-` 直线要求起点≠终点且至少隔一键（相对终点须 3–7）"
    elif shape == "^":
        if d == 0 or d == 4:
            return "错误", "`^` 弧线不能同键、也不能走到正对面（对向请用 `-`/`v`）"
    elif shape == "v":
        if head == end or d == 4:
            return "错误", "`v` 折线终点不能是自身、也不能是正对面（官方口径）"
    elif shape in ("s", "z", "w"):
        if d != 4:
            return "错误", f"`{shape}` 的终点必须是正对面键（相隔 4 键）"
    elif shape == "V":
        if pivot is None:
            return "错误", "`V` 缺少拐点键（三键记法 `1V36`）"
        if cdist(head, pivot) != 2:
            return "错误", "`V` 的拐点只能隔开一键（相对位 3 或 7）"
        if cdist(pivot, end) < 2:
            return "错误", "`V` 的终点到拐点至少隔一键"
        if head == end:
            return "错误", "`V` 的起点与终点不能相同"
    elif shape in ("<", ">"):
        if d == 0:
            return "提示", (f"`{shape}` 的起点与终点相同＝绕一整圈；官方无硬限制，"
                            "官谱写过（`3<3` / `7>7`），确认是有意为之即可")
    return "", ""


def scan_members(s: str, ctx: _Ctx, out: LayerResult) -> dict:
    """逐 note 成员扫描。返回统计块（slide 形状用量、note 记法用量等）。"""
    stats = {"n_slots": 0, "shapes": {}, "n_slide": 0, "n_chain": 0,
             "n_samehead": 0, "n_touch": 0, "n_break": 0}
    n = len(s)
    i = 0
    slot_start = 0
    members: list[tuple[int, str]] = []   # (起始下标, 文本)
    cur: list[str] = []
    cur_start = 0

    def end_member() -> None:
        nonlocal cur, cur_start
        members.append((cur_start, "".join(cur)))
        cur = []

    while i < n:
        c = s[i]
        if c == "(":
            j = s.find(")", i)
            i = (j + 1) if j >= 0 else n
            continue
        if c == "{":
            j = s.find("}", i)
            i = (j + 1) if j >= 0 else n
            continue
        if s[i:i + 3] == "<HS":
            j = s.find(">", i)
            i = (j + 1) if j >= 0 else n
            continue
        if c == "[":
            j = s.find("]", i)
            if j < 0:
                break
            if not cur:
                cur_start = i
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if c in "/`":
            if c == "`":
                ln, col = ctx.at(i)
                out.add("错误", "ST-PSEUDOEACH",
                        "伪双押 `` ` `` 是工具扩展，ST 谱 0 使用",
                        line=ln, col=col, source="st-chart-elements §三")
            end_member()
            cur_start = i + 1
            i += 1
            continue
        if c == ",":
            end_member()
            _check_slot(members, slot_start, ctx, out, stats)
            stats["n_slots"] += 1
            members = []
            slot_start = i + 1
            cur_start = i + 1
            i += 1
            continue
        if c == "E" and i == n - 1:
            i += 1
            continue
        if not cur:
            cur_start = i
        cur.append(c)
        i += 1
    if cur:
        end_member()
    if members:
        _check_slot(members, slot_start, ctx, out, stats)
    return stats


def _check_slot(members: list[tuple[int, str]], slot_start: int, ctx: _Ctx,
                out: LayerResult, stats: dict) -> None:
    real = [(p, m) for p, m in members if m]
    if len(members) > 1 and len(real) != len(members):
        ln, col = ctx.at(slot_start)
        out.add("错误", "SYN-EMPTY-MEMBER",
                "`/` 两侧出现空成员（如 `1/` 或 `1//2`）——MajdataEdit 判定 1 报语法错误",
                line=ln, col=col, source="simai-error-checking §2 判定 1")
    heads: list[int] = []
    for pos, m in real:
        info = _check_member(m, pos, ctx, out, stats)
        if info.get("slide_head") is not None:
            heads.append(info["slide_head"])
    if len(heads) != len(set(heads)):
        ln, col = ctx.at(slot_start)
        out.add("错误", "SYN-SAMEHEAD-SLASH",
                "同一槽里用 `/` 写了同起点的多条 slide——会制造两个重叠星星头，"
                "无法游玩；同头星星必须用 `*`",
                line=ln, col=col, source="simai-syntax §6-3")


def _check_member(m: str, base: int, ctx: _Ctx, out: LayerResult,
                  stats: dict) -> dict:
    """扫一个 each 成员，返回 ``{"slide_head": int|None}``。"""
    info: dict = {"slide_head": None}
    i, n = 0, len(m)
    while i < n:
        c = m[i]
        ln, col = ctx.at(base + i)
        # 全角字符
        kind = _fullwidth_kind(c)
        if kind:
            out.add("错误", "SYN-FULLWIDTH",
                    f"出现{kind} {c!r}——MiaCode 把全角字符列为 Error（LLM 生成文本高危项）",
                    line=ln, col=col, excerpt=m[:40], source="simai-error-checking §8.2")
            i += 1
            continue
        if c in TOUCH_AREAS:
            stats["n_touch"] += 1
            out.add("错误", "ST-TOUCH",
                    f"Touch / Touch-Hold（`{m[i:i + 2]}`）是 DX 要素，ST 谱禁用",
                    line=ln, col=col, excerpt=m[:40], source="st-chart-elements §三")
            i += 1
            if i < n and m[i].isdigit():
                i += 1
            while i < n and m[i] in "fhbxm":
                i += 1
            if i < n and m[i] == "[":
                j = m.find("]", i)
                i = (j + 1) if j >= 0 else n
            continue
        if c == "m":
            out.add("错误", "ST-MINE", "地雷 `m` 是 SimaiSharp 内部扩展，安全子集禁用",
                    line=ln, col=col, source="simai-syntax §4.7")
            i += 1
            continue
        if c in ST_BLACKLIST:
            code, msg = ST_BLACKLIST[c]
            out.add("错误", code, msg, line=ln, col=col, excerpt=m[:40],
                    source="st-chart-elements §三")
            i += 1
            continue
        if c not in DIGITS:
            out.add("错误", "SYN-UNKNOWN-CHAR",
                    f"无法识别的字符 {c!r}（成员 `{m}`）",
                    line=ln, col=col, source="simai-error-checking §4")
            i += 1
            continue

        head = int(c)
        i += 1
        mods: set[str] = set()
        while i < n and m[i] in "bx!?$@":
            if m[i] in ST_BLACKLIST:
                code, msg = ST_BLACKLIST[m[i]]
                lnk, colk = ctx.at(base + i)
                out.add("错误", code, msg, line=lnk, col=colk, excerpt=m[:40],
                        source="st-chart-elements §三")
            if m[i] == "b":
                stats["n_break"] += 1
            mods.add(m[i])
            i += 1

        # HOLD
        if i < n and m[i] == "h":
            i += 1
            while i < n and m[i] in "bx!?$@":
                if m[i] in ST_BLACKLIST:
                    code, msg = ST_BLACKLIST[m[i]]
                    lnk, colk = ctx.at(base + i)
                    out.add("错误", code, msg, line=lnk, col=colk, source="st-chart-elements §三")
                mods.add(m[i])
                i += 1
            if "$" in mods:
                out.add("错误", "SYN-HOLD-DOLLAR", "Hold 上的 `$` 被 MajdataEdit 拒绝",
                        line=ln, col=col, source="simai-syntax §6-3")
            if i < n and m[i] == "[":
                j = m.find("]", i)
                dur = m[i + 1:j] if j >= 0 else m[i + 1:]
                _check_hold_duration(dur, base + i, ctx, out)
                i = (j + 1) if j >= 0 else n
                while i < n and m[i] in "bx":
                    if m[i] == "x":
                        code, msg = ST_BLACKLIST["x"]
                        lnk, colk = ctx.at(base + i)
                        out.add("错误", code, msg, line=lnk, col=colk,
                                source="st-chart-elements §三")
                    i += 1
            continue

        # 带 [...] 但没有 h → SimaiSharp 会当 Hold 读
        if i < n and m[i] == "[":
            out.add("错误", "SYN-TAP-DURATION",
                    f"`{m}`：TAP 带了时长括号但没有 `h`——SimaiSharp 会把它当 Hold，"
                    "MiaCode 直接报「有 [] 无 h」",
                    line=ln, col=col, source="simai-syntax §4.4")
            j = m.find("]", i)
            i = (j + 1) if j >= 0 else n
            continue

        # SLIDE
        if i < n and (m[i] in SHAPES_ONE or m[i:i + 2] in SHAPES_TWO):
            info["slide_head"] = head
            n_seg = 0
            n_dur_blocks = 0
            while i < n and (m[i] in SHAPES_ONE or m[i:i + 2] in SHAPES_TWO):
                shp_pos = i
                shape, i = _read_shape(m, i)
                while i < n and m[i] in "bx":
                    lnk, colk = ctx.at(base + i)
                    out.add("错误", "SYN-SLIDE-MOD-POS",
                            f"slide 修饰符 `{m[i]}` 写在终点键之前（野写法，解析器只是容错）",
                            line=lnk, col=colk, source="simai-syntax §4.6")
                    i += 1
                pivot = None
                if shape == "V":
                    if i + 1 >= n or m[i] not in DIGITS or m[i + 1] not in DIGITS:
                        lnk, colk = ctx.at(base + shp_pos)
                        out.add("错误", "SYN-SLIDE-V", "`V` 形缺少拐点或终点键（三键记法 `1V36`）",
                                line=lnk, col=colk, source="simai-syntax §4.6")
                        break
                    pivot, end = int(m[i]), int(m[i + 1])
                    i += 2
                else:
                    if i >= n or m[i] not in DIGITS:
                        lnk, colk = ctx.at(base + shp_pos)
                        out.add("错误", "SYN-SLIDE-END", f"`{shape}` 形 slide 缺少终点键",
                                line=lnk, col=colk, source="simai-syntax §4.6")
                        break
                    end = int(m[i])
                    i += 1
                n_seg += 1
                stats["n_slide"] += 1 if n_seg == 1 else 0
                stats["shapes"][shape] = stats["shapes"].get(shape, 0) + 1
                lvl, bad = _slide_geometry(shape, head, pivot, end)
                if bad:
                    lnk, colk = ctx.at(base + shp_pos)
                    out.add(lvl, "SYN-SLIDE-GEOM", f"{bad}（读到 `{head}{shape}"
                            + (str(pivot) if pivot else "") + f"{end}`）",
                            line=lnk, col=colk, excerpt=m[:40],
                            source="simai-error-checking §2 slide 端点几何")
                while i < n and m[i] in "bx[":
                    if m[i] == "[":
                        j = m.find("]", i)
                        dur = m[i + 1:j] if j >= 0 else m[i + 1:]
                        n_dur_blocks += 1
                        _check_slide_duration(dur, base + i, ctx, out)
                        i = (j + 1) if j >= 0 else n
                    else:
                        lnk, colk = ctx.at(base + i)
                        if m[i] == "b":
                            out.add("错误", "ST-BREAKSLIDE",
                                    "滑条绝赞（`]b` / slide 上的 `b`）是 FESTiVAL 要素，"
                                    "ST 谱禁用；星星头绝赞要写成 `1b-4[8:1]`",
                                    line=lnk, col=colk, excerpt=m[:40],
                                    source="st-chart-elements §三")
                        else:
                            code, msg = ST_BLACKLIST["x"]
                            out.add("错误", code, msg, line=lnk, col=colk,
                                    source="st-chart-elements §三")
                        i += 1
                head = end
                if i < n and m[i] == "*":
                    stats["n_samehead"] += 1
                    i += 1
                    head = info["slide_head"]
                    n_seg = 0
                    n_dur_blocks = 0
                    continue
            if n_seg > 1:
                stats["n_chain"] += 1
                lnk, colk = ctx.at(base)
                out.add("错误", "ST-CHAIN",
                        "连锁（组合）星星是 FESTiVAL 要素，ST 谱 0 使用；"
                        "且分段时值在 MiaCode strict 会报错",
                        line=lnk, col=colk, excerpt=m[:40], source="st-chart-elements §三")
            if n_dur_blocks == 0:
                lnk, colk = ctx.at(base)
                out.add("错误", "SYN-SLIDE-NO-DURATION", f"slide `{m}` 没有时长括号",
                        line=lnk, col=colk, source="simai-syntax §4.6")
            continue

        # 纯 TAP
        if "$" in mods:
            pass  # 已在黑名单里报过
    return info


_DUR_XY = re.compile(r"^\d+(\.\d+)?:\d+(\.\d+)?$")
_DUR_SEC = re.compile(r"^#\d+(\.\d+)?$")
_DUR_BPM_XY = re.compile(r"^\d+(\.\d+)?#\d+(\.\d+)?:\d+(\.\d+)?$")
_DUR_BPM_SEC = re.compile(r"^\d+(\.\d+)?#\d+(\.\d+)?$")
_DUR_SS = re.compile(r"^\d+(\.\d+)?##.+$")


def _check_hold_duration(body: str, pos: int, ctx: _Ctx, out: LayerResult) -> None:
    ln, col = ctx.at(pos)
    if body.startswith("##"):
        out.add("错误", "SYN-HOLD-DUR",
                f"hold 时长 `[{body}]` 用了 `[##秒]`——MajdataView wiki 笔误，SimaiSharp 拒绝",
                line=ln, col=col, source="simai-syntax §6-3")
        return
    if _DUR_XY.match(body) or _DUR_SEC.match(body) or _DUR_BPM_XY.match(body):
        return
    out.add("错误", "SYN-HOLD-DUR",
            f"hold 时长 `[{body}]` 不在官方三形式里（`[x:y]` / `[#秒]` / `[BPM#x:y]`）",
            line=ln, col=col, source="simai-syntax §4.4")


def _check_slide_duration(body: str, pos: int, ctx: _Ctx, out: LayerResult) -> None:
    ln, col = ctx.at(pos)
    if _DUR_XY.match(body) or _DUR_BPM_XY.match(body) or _DUR_BPM_SEC.match(body) \
            or _DUR_SS.match(body) or _DUR_SEC.match(body):
        if _DUR_XY.match(body):
            x, y = body.split(":")
            if float(x) <= 0 or float(y) <= 0:
                out.add("错误", "SYN-SLIDE-DUR",
                        f"slide 时长 `[{body}]` 必须严格为正（MiaCode：零时长报错）",
                        line=ln, col=col, source="simai-error-checking §8.2")
        return
    if ":" not in body and "#" not in body:
        out.add("错误", "SYN-SLIDE-DUR",
                f"slide 时长 `[{body}]` 是无冒号单值形式——MajdataEdit 校验器不认，"
                "改用 `[x:y]` 或 `[秒##秒]`",
                line=ln, col=col, source="simai-syntax §6-3")
        return
    out.add("错误", "SYN-SLIDE-DUR",
            f"slide 时长 `[{body}]` 不在官方 6 形式里",
            line=ln, col=col, source="simai-syntax §4.6")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def check_syntax(md: MaiData, body: str, body_off: int, parse_res) -> LayerResult:
    """跑完语法层。`parse_res` 是 `simai_parser.parse_chart` 的结果。"""
    out = LayerResult(layer="语法")
    s, idx = normalize_with_map(body)
    ctx = _Ctx(text=md.text, body_off=body_off, idx=idx)

    check_meta(md, out)
    check_comments_and_brackets(body, ctx, out)
    check_directives(s, ctx, out)
    check_end_marker(s, ctx, out)
    stats = scan_members(s.rstrip("E"), ctx, out)

    for e in parse_res.errors:
        out.add("错误", "SYN-PARSE", f"解析器报错：{e}", source="simai_parser")
    for w in parse_res.warnings:
        if "正文未以大写 E 结尾" in w:
            continue  # 已由 SYN-NO-E 报过
        out.add("警告", "SYN-PARSE-WARN", f"解析器警告：{w}", source="simai_parser")

    stats.update({
        "n_notes_parsed": len(parse_res.notes),
        "has_end_marker": parse_res.has_end_marker,
        "slide_chain_segments": parse_res.slide_chain_segments,
        "each_groups": parse_res.each_groups,
        "外部三检": "未接（本机无 SimaiSharp / MajdataEdit / MiaCode）",
    })
    out.stats = stats
    return out
