#!/usr/bin/env python3
"""外部规则层：**MiaCode / MaiMuriDX 的无理检测**（内无 / 外无 / 撞尾 / 叠键 / 多押）。

前六层是本项目自己的口径（`hands.py` 的无理按 388 官谱校准，宽）。这一层换一套口径：
**社区工具真正会在编辑器里弹出来的那些"无理"**——Visual Maimai 的无理面板、
MiaCode 的谱面检查、MaiMuriDX 集成 Mod，三者规则同源。生成端只有这一层归零，
用户把谱导进 Visual Maimai 才不会满屏提示。

## 规则来源（逐条可查）

- 规格：MiaCode `docs/specs/muri/MURI_DETECTION_SPEC.md`（336 行，§3 类型矩阵 /
  §4 时间常量 / §5 判定规则 / §6 告警级别），摘要见 `docs/research/miacode-vm-research.md` §A.6
  与 `docs/simai-error-checking.md` §8.5。
- 实现：上游 **Starrah/MaiMuriDX**（Python）——MiaCode 的 5 类 `MuriKind` 与它同源
  （spec §11 自述"`MaiMuriDX` 的 slide 判定里也有同款双分支实现"）。
  本层是 `judge.py::StaticMuriChecker.check`（46–197 行）的逐条翻译，
  常量取 `core.py` 38–98 行 + `config.json` 默认值。
- 几何：`data/slide_geometry.json` 由 MaiMuriDX `slides.py::SlideInfo`（`init()` 后的
  `_entries`）导出——600 个 slide 形状 + 8 个 wifi 的 **A 区进入时刻**（归一化 [0,1]）、
  `critical_proportion` 与路径长度。**是数据复用，不是重新推导几何。**

## 覆盖范围与边界（必须写明）

MURI_DETECTION_SPEC §3 把 5 类分成两路：

| 类型 | 中文 | 运行时分析 | 静态参考 | 本层 |
|---|---|---|---|---|
| `SlideHeadTap` | 外键（外无） | 是 | 是 | **实现（静态）** |
| `TapOnSlide` | 撞尾 | 是 | 是 | **实现（静态）** |
| `Overlap` | 叠键 | 是 | 是 | **实现（静态）** |
| `SlideTooFast` | 内屏（内无） | 是 | **否** | ✗ 不覆盖 |
| `MultiTouch` | 多押 | 是 | **否** | ✗ 不覆盖 |

后两类**只来自 180 TPS 的运行时手势模拟**，没有静态参考，本层不复现那套模拟。
⚠️ 所以「本层 0 命中」**不等于**「MiaCode 面板 0 条」——发布前仍应以真机（MiaCode /
Visual Maimai）跑一遍为准，同 `docs/simai-error-checking.md` §10-4 的"外部三检待接"。
多押另由第二层 `hands.py` 用本项目口径覆盖（知识 008/077）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from .model import LayerResult

_DATA = Path(__file__).resolve().parent / "data" / "slide_geometry.json"

# ---------------------------------------------------------------------------
# 时间常量（MURI_DETECTION_SPEC §4 / MaiMuriDX core.py 38–98 + config.json）
# 判定时基 180 TPS；MaiMuriDX 内部以 tick 计，这里一律换算成**秒**。
# ---------------------------------------------------------------------------
JUDGE_TPS = 180.0

#: tap 还能被判定的最晚可用窗口（27 tick）
TAP_AVAILABLE = 27.0 / JUDGE_TPS              # 150.0 ms
#: 过滤"贴着 slide 起点的极小时间差"，避免把启动拍 tap 算成外键（config 默认 0.3333 帧）
TAP_ON_SLIDE_THRESHOLD = 3.0 * 0.3333 / JUDGE_TPS   # ≈ 5.6 ms
#: 同 pad 叠键的最大时间差（6 tick）
OVERLAY_THRESHOLD = 6.0 / JUDGE_TPS           # 33.3 ms
#: 撞尾 / 外键的静态阈值（36 tick；MiaCode 默认 200 ms，可调 150–250）
COLLIDE_THRESHOLD = 36.0 / JUDGE_TPS          # 200.0 ms
#: 撞尾 / 外键区间左端的额外冗余（9 tick）
COLLIDE_EXTRA_DELTA = 9.0 / JUDGE_TPS         # 50.0 ms

#: 边界容差。MaiMuriDX 在 **tick** 域比较，本层在**秒**域比较，
#: 分音累加出来的浮点误差会让"恰好等于阈值"（如八分音 @150BPM 正好 200 ms）
#: 掉到窗口外。1 µs 远小于任何判定量级，只用来吃掉这个误差。
_EPS = 1e-6

#: SlideHeadTap 降 Warning：没有启动拍 tap 时的小 gap 上限（spec §6.2）
HEADTAP_WARN_GAP = 0.050
#: SlideHeadTap / TapOnSlide 降 Warning：晚窗口下限（spec §6.2 / §6.3）
LATE_WARN_GAP = 0.150

#: 5 类 MuriKind 的中文名（MURI_DETECTION_SPEC §3；括号里是社区叫法）
KIND_CN = {"SlideHeadTap": "外键无理", "TapOnSlide": "撞尾无理",
           "Overlap": "叠键无理", "SlideTooFast": "内屏无理",
           "MultiTouch": "多押无理"}

_SIMPLE_KINDS = ("tap", "hold", "slide_star", "touch", "touch_hold")
_TAPLIKE = ("tap", "slide_star")          # slide 碰撞参考只认 tap / hold / head star
_HOLDLIKE = ("hold", "touch_hold")


@lru_cache(maxsize=1)
def load_geometry() -> dict:
    """读 `data/slide_geometry.json`（MaiMuriDX `SlideInfo` 导出）。"""
    return json.loads(_DATA.read_text(encoding="utf-8"))


def pad_of(key: str) -> str:
    """note 的键位 → 判定区名。`'1'..'8'` → `'A1'..'A8'`；touch 区原样（`C` / `B3` / `E5`）。"""
    k = key.strip()
    if k.isdigit() and 1 <= int(k) <= 8:
        return f"A{int(k)}"
    return k.upper() if k else ""


@dataclass
class _Simple:
    """普通物件（tap / hold / touch / touch_hold / slide 头星）。"""

    kind: str
    key: str
    pad: str
    t: float
    end_t: float
    measure: int
    text: str
    is_ex: bool = False

    @property
    def is_taplike(self) -> bool:
        return self.kind in _TAPLIKE

    @property
    def is_holdlike(self) -> bool:
        return self.kind in _HOLDLIKE


@dataclass
class _Slide:
    """一条滑轨（含连锁多段）。时间全部是秒。"""

    key: str                  # 起点键 '1'..'8'
    start: int
    end: int
    shoot: float              # 启动拍（marker + wait）
    end_moment: float
    critical: float
    is_wifi: bool
    measure: int
    text: str
    #: `(判定区, 进入时刻, 区间左端, 区间右端)`
    entries: list[tuple[str, float, float, float]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def _build_slide(n, geo: dict) -> _Slide | None:
    """把一个 `slide_track` NoteEvent 折算成判定用的 `_Slide`。"""
    segs = n.segments or ((n.key, n.shape, n.end_key),)
    keys = ["".join(s) for s in segs]
    tbl = geo["slides"]
    wtbl = geo["wifi"]
    is_wifi = any(k in wtbl for k in keys)
    infos = [tbl.get(k) or wtbl.get(k) for k in keys]
    unknown = [k for k, info in zip(keys, infos) if info is None]
    infos = [i for i in infos if i is not None]
    shoot0 = n.time + n.wait
    if not infos:
        # 形状全都不在几何表里：只挂一条提示，不做轨道检查
        return _Slide(key=n.key, start=0, end=0, shoot=shoot0,
                      end_moment=shoot0 + n.duration, critical=shoot0 + n.duration,
                      is_wifi=is_wifi, measure=n.measure, text="", unknown=unknown)

    # 连锁 slide 只给总时长（MiaCode strict 禁止分段时值），按路径长度分配
    lengths = [float(i["len"]) or 1.0 for i in infos]
    total_len = sum(lengths) or 1.0
    durations = [n.duration * L / total_len for L in lengths]

    shoot = n.time + n.wait
    sl = _Slide(key=n.key, start=int(infos[0]["start"]), end=int(infos[-1]["end"]),
                shoot=shoot, end_moment=shoot + sum(durations),
                critical=0.0, is_wifi=is_wifi, measure=n.measure, text="",
                unknown=unknown)
    # 正解时刻 = 引导星进入最后一个区的时刻（MaiMuriDX SimaiSlideChain.critical_moment）
    last_area = (1.0 - float(infos[-1]["crit"])) * durations[-1]
    sl.critical = sl.end_moment - last_area

    seg_start = shoot
    for info, dur in zip(infos, durations):
        for pad, frac in info.get("pads", ()):
            enter = seg_start + float(frac) * dur
            lo = max(enter - COLLIDE_EXTRA_DELTA, shoot + TAP_ON_SLIDE_THRESHOLD) - _EPS
            hi = enter + COLLIDE_THRESHOLD + _EPS
            sl.entries.append((pad, enter, lo, hi))
        seg_start += dur
    return sl


def _headtap_level(gap: float, protected: bool, has_start_tap: bool) -> str:
    """SlideHeadTap 的告警级别（MURI_DETECTION_SPEC §6.2）。"""
    if protected:
        return "警告"
    if 0.0 < gap <= HEADTAP_WARN_GAP and not has_start_tap:
        return "警告"
    if gap >= LATE_WARN_GAP:
        return "警告"
    return "错误"


def _taponslide_level(gap: float, protected: bool) -> str:
    """TapOnSlide 的告警级别（MURI_DETECTION_SPEC §6.3）。

    静态参考产出的 gap 天然 ≤ `COLLIDE_THRESHOLD`（默认 200 ms），
    按 spec 的三个条件（`gap > 150 ms` / `gap ≤ 静态阈值` / 带保护）**全部落 Warning**；
    真正判成 Muri 的撞尾只来自运行时分析（本层不覆盖，见模块 docstring）。
    """
    if protected or gap > LATE_WARN_GAP or gap <= COLLIDE_THRESHOLD:
        return "警告"
    return "错误"  # pragma: no cover - 静态路径够不到


def check_muri(res, measure_texts=None) -> LayerResult:
    """跑 MiaCode 静态无理检测。`res` 是 `simai_parser.ParseResult`。"""
    out = LayerResult(layer="外部规则")
    texts = measure_texts or {}
    try:
        geo = load_geometry()
    except Exception as exc:  # pragma: no cover
        out.skipped = f"slide 几何表没读上（{exc}）"
        return out

    simples: list[_Simple] = []
    slides: list[_Slide] = []
    for n in res.notes:
        if n.kind in _SIMPLE_KINDS:
            simples.append(_Simple(
                kind=n.kind, key=n.key, pad=pad_of(n.key), t=n.time,
                end_t=n.time + (n.duration if n.kind in _HOLDLIKE else 0.0),
                measure=n.measure,
                text=f"{n.key}{'h' if n.kind in _HOLDLIKE else ''}",
                is_ex=bool(n.is_ex)))
        elif n.kind == "slide_track":
            sl = _build_slide(n, geo)
            if sl is not None:
                sl.text = f"{n.key}{n.shape}{n.end_key}"
                slides.append(sl)

    unknown_shapes = sorted({u for s in slides for u in s.unknown})
    if unknown_shapes:
        out.add("提示", "MURI-SHAPE-UNKNOWN",
                "这些 slide 形状不在 MaiMuriDX 几何表里，本层跳过了它们的轨道检查："
                + "、".join(unknown_shapes),
                source="data/slide_geometry.json")

    # 每条 slide 的启动拍 tap（同键、恰在启动时刻）——SlideHeadTap 降级条件要用
    start_taps = set()
    for sl in slides:
        for s in simples:
            if (s.is_taplike and s.pad == f"A{sl.start}"
                    and abs(s.t - sl.shoot) <= TAP_ON_SLIDE_THRESHOLD + _EPS):
                start_taps.add(id(sl))
                break

    hits: list[dict] = []

    # ---- 外键 / 撞尾（MaiMuriDX judge.py:101–154）----
    for sl in slides:
        for s in simples:
            if not (s.is_taplike or s.kind == "hold"):
                continue                       # 筛掉 touch：静态参考不覆盖
            if s.t < sl.shoot - _EPS or s.t > sl.end_moment + COLLIDE_THRESHOLD + _EPS:
                continue
            if (s.pad == f"A{sl.start}"
                    and TAP_ON_SLIDE_THRESHOLD - _EPS
                    <= s.t - sl.shoot <= COLLIDE_THRESHOLD + _EPS):
                hits.append({"kind": "SlideHeadTap", "note": s, "slide": sl,
                             "delta": sl.shoot - s.t})
            if sl.is_wifi:
                # wifi 只查头尾（judge.py:146–154）：终点键与它左右各一键
                ends = {(sl.end - 1) % 8 + 1, sl.end % 8 + 1, (sl.end - 2) % 8 + 1}
                if s.key.isdigit() and int(s.key) in ends:
                    lo = max(sl.critical - COLLIDE_EXTRA_DELTA,
                             sl.shoot + TAP_ON_SLIDE_THRESHOLD) - _EPS
                    hi = max(sl.critical + COLLIDE_THRESHOLD,
                             sl.end_moment + COLLIDE_EXTRA_DELTA) + _EPS
                    if lo <= s.t <= hi:
                        hits.append({"kind": "TapOnSlide", "note": s, "slide": sl,
                                     "delta": sl.critical - s.t})
                continue
            for pad, enter, lo, hi in sl.entries:
                if pad == s.pad and lo <= s.t <= hi:
                    hits.append({"kind": "TapOnSlide", "note": s, "slide": sl,
                                 "delta": enter - s.t})
            # 最后一个区的区间尾再延长到"星星结束 + 50ms"（judge.py:133–136）
            if (s.pad == f"A{sl.end}"
                    and sl.critical + COLLIDE_THRESHOLD + _EPS < s.t
                    <= sl.end_moment + COLLIDE_EXTRA_DELTA + _EPS):
                hits.append({"kind": "TapOnSlide", "note": s, "slide": sl,
                             "delta": sl.critical - s.t})

    # ---- 叠键（judge.py:156–178）----
    for i, a in enumerate(simples):
        for j, b in enumerate(simples):
            if a is b:
                continue
            if not a.is_holdlike:
                if not b.is_holdlike:
                    if (i < j and a.pad == b.pad
                            and abs(a.t - b.t) <= OVERLAY_THRESHOLD + _EPS):
                        hits.append({"kind": "Overlap", "note": a, "other": b})
                elif (a.pad == b.pad
                      and b.t - OVERLAY_THRESHOLD - _EPS <= a.t
                      <= b.end_t + OVERLAY_THRESHOLD + _EPS):
                    hits.append({"kind": "Overlap", "note": a, "other": b})
            elif b.is_holdlike:
                if i < j and a.pad == b.pad and (
                        b.t - OVERLAY_THRESHOLD - _EPS <= a.t
                        <= b.end_t + OVERLAY_THRESHOLD + _EPS
                        or a.t - OVERLAY_THRESHOLD - _EPS <= b.t
                        <= a.end_t + OVERLAY_THRESHOLD + _EPS):
                    hits.append({"kind": "Overlap", "note": a, "other": b})

    # ---- 叠键去重：同一发生时刻只留一条（MURI_DETECTION_SPEC §9）----
    seen_overlap: set[float] = set()
    kept: list[dict] = []
    for h in hits:
        if h["kind"] == "Overlap":
            stamp = round(h["note"].t, 4)
            if stamp in seen_overlap:
                continue
            seen_overlap.add(stamp)
        kept.append(h)
    hits = kept
    hits.sort(key=lambda h: (h["note"].t, h["note"].measure, h["kind"]))

    counts: dict[str, int] = {}
    rows = []
    for h in hits:
        s = h["note"]
        kind = h["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "Overlap":
            level = "错误"                     # spec §6.1：Overlap 恒为 Muri
            other = h["other"]
            msg = (f"{KIND_CN[kind]}：`{s.text}` 与 `{other.text}` 同判定区 {s.pad} 重叠"
                   f"（时差 {abs(s.t - other.t) * 1000:.0f} ms ≤ "
                   f"{OVERLAY_THRESHOLD * 1000:.1f} ms）")
            code = "MURI-OVERLAP"
        else:
            sl = h["slide"]
            gap = abs(h["delta"])
            if kind == "SlideHeadTap":
                level = _headtap_level(gap, s.is_ex, id(sl) in start_taps)
                code = "MURI-SLIDEHEADTAP"
                msg = (f"{KIND_CN[kind]}：`{s.text}` 会被 `{sl.text}` 的星星头蹭到"
                       f"（启动拍后 {gap * 1000:.0f} ms，红线 "
                       f"{COLLIDE_THRESHOLD * 1000:.0f} ms）")
            else:
                level = _taponslide_level(gap, s.is_ex)
                code = "MURI-TAPONSLIDE"
                msg = (f"{KIND_CN[kind]}：`{s.text}` 落在 `{sl.text}` 的轨道判定区 "
                       f"{s.pad} 上（轨道进区后 {gap * 1000:.0f} ms，红线 "
                       f"{COLLIDE_THRESHOLD * 1000:.0f} ms）")
        if s.is_ex:
            msg += "（带 Ex 保护，MiaCode 降 Warning）"
        out.add(level, code, msg, measure=s.measure,
                excerpt=(texts.get(s.measure, "") or "")[:200],
                source="MiaCode MURI_DETECTION_SPEC §5/§6")
        rows.append({"kind": kind, "level": level, "measure": s.measure,
                     "time": round(s.t, 4), "note": s.text,
                     "cause": h["slide"].text if kind != "Overlap" else h["other"].text,
                     "gap_ms": None if kind == "Overlap" else round(abs(h["delta"]) * 1000, 1)})

    if hits:
        out.add("警告", "MURI-TOTAL",
                f"本层共 **{len(hits)} 条**命中（"
                + "、".join(f"{KIND_CN[k]} {v}" for k, v in sorted(counts.items()))
                + "）。**这些就是导入 Visual Maimai / MiaCode 后会在无理面板里逐条弹出来的东西。**"
                f"官谱基线（`out/calib` 40 首 ST Re:MASTER 13.0–14.5，同一套检测）："
                f"中位 3 / 均值 3.8 / p90 10 / 最大 21，13 首为 0——"
                "**官谱自己也命中，所以不是「有一条就写坏了」；但生成端的验收目标是 0**，"
                "留下来的每一条都要写明理由。",
                source="MiaCode MURI_DETECTION_SPEC §3；官谱基线 note 066")
    else:
        out.add("提示", "MURI-OK",
                "MiaCode 静态无理检测 0 命中（外键 / 撞尾 / 叠键三类）。"
                "⚠️ 内屏与多押只有运行时分析、本层不覆盖，发布前仍要真机跑一遍。",
                source="MiaCode MURI_DETECTION_SPEC §3")

    out.stats = {"hits": rows, "counts": counts, "n_slides": len(slides),
                 "n_simple": len(simples),
                 "覆盖": "静态三类（SlideHeadTap / TapOnSlide / Overlap）；"
                         "SlideTooFast 与 MultiTouch 只有运行时分析，未覆盖",
                 "阈值": {"collide_ms": COLLIDE_THRESHOLD * 1000,
                          "overlay_ms": OVERLAY_THRESHOLD * 1000,
                          "extra_delta_ms": COLLIDE_EXTRA_DELTA * 1000,
                          "tap_on_slide_ms": TAP_ON_SLIDE_THRESHOLD * 1000}}
    return out
