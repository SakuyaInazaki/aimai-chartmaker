#!/usr/bin/env python3
"""配置难度/密度分级与强度对应表。

回答用户 2026-09-19 的第二条要求——
「你就不能稍微确定一下什么样的配置比较密比较难，适合放在强度高的地方；
什么样的配置比较简单适合放在强度低的地方吗？」

两层指标
--------
- **内在难度**（只看谱面，不需要音频；388 官方谱都能算）：知识 003 四因素 +
  知识 015 红线的可计算代理——note/小节、NPS、分音占比、each 占比、相邻 note
  键位位移、slide 占比、等效速度（分音 × BPM）、无理代理命中率，以及
  **去掉配置项的硬度分**（破循环论证）。
- **外在放置**（要音频；160 首标定曲）：所在小节的曲内强度分位、段落分布、
  曲内相对位置、定数分层，以及三口径下的采音方式（全踩 / 舍音 / 留白，社区用词；
  2026-09-20 术语校正前叫 采全音 / 半采音 / 空音）。

⚠️ **检测器可替换**
------------------
本模块**不硬绑** `configs.py` 的内部函数。"这一小节命中了哪些配置"是一个
**可注入的输入**：

- 传 ``bar_configs={小节号: {配置名, ...}}`` —— 直接给一张表（任何来源）；
- 传 ``detector=fn``，``fn(res, slots)`` 返回任何带 ``config`` /
  ``bar_start`` / ``bar_end``（属性或字典键均可）的片段对象；
- 都不传时才回落到 :func:`default_detector`（= `configs.detect_all`）。

这样等 `tools/chart_analysis/hands.py`（双手分配器）落地、`configs.py` 按手序
重写之后，本模块与报告口径可以**原样重跑**，只换注入的检测器。

⚠️ 本轮报告用的是**键位版**检测器（`configs.py` 现状：按键位与时间几何判定，
不含左右手分配），结论标注为"待手序版重跑"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .simai_parser import NoteEvent, ParseResult

# ---------------------------------------------------------------------------
# 分档口径（**agent 设定**）
# ---------------------------------------------------------------------------

#: 目标音轨每小节事件数档（= 采音 pool 大小；用户要的矩阵行）
DENSITY_TIERS: tuple[tuple[str, int, int], ...] = (
    ("稀", 0, 4),        # ≤4
    ("中", 5, 8),        # 5–8
    ("密", 9, 10 ** 9),  # ≥9
)

#: BPM 分档（知识 003 第 3 条：同配置 BPM 越高越强）
BPM_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<150", 0.0, 150.0),
    ("150-180", 150.0, 180.0),
    (">180", 180.0, 1e9),
)

#: 定数分层
LEVEL_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("13.0-13.4", 13.0, 13.5),
    ("13.5-13.9", 13.5, 14.0),
    ("14.0-14.4", 14.0, 14.5),
    ("14.5+", 14.5, 99.0),
)

#: 报告里固定的分音列
DIVISORS_REPORTED: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def density_tier(n_events: float | None) -> str:
    """目标音轨每小节事件数 → 稀 / 中 / 密。"""
    if n_events is None or not _finite(n_events):
        return ""
    for name, lo, hi in DENSITY_TIERS:
        if lo <= n_events <= hi:
            return name
    return ""


def bpm_bucket(bpm: float | None) -> str:
    if bpm is None or not _finite(bpm) or bpm <= 0:
        return ""
    for name, lo, hi in BPM_BUCKETS:
        if lo <= bpm < hi:
            return name
    return BPM_BUCKETS[-1][0]


def level_bucket(level: float | None) -> str:
    if level is None or not _finite(level) or level <= 0:
        return ""
    for name, lo, hi in LEVEL_BUCKETS:
        if lo <= level < hi:
            return name
    return ""


def tier_by_cuts(value: float | None, cuts: Sequence[float],
                 names: Sequence[str] = ("低", "中", "高")) -> str:
    """按给定切点把连续量分档（切点数 = 档数 − 1）。"""
    if value is None or not _finite(value):
        return ""
    for i, c in enumerate(cuts):
        if value <= c:
            return names[i]
    return names[len(cuts)]


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 检测器注入层
# ---------------------------------------------------------------------------

#: 一个"配置命中片段"只需要三样东西：配置名 + 起止小节
HitLike = object


def _hit_field(h: HitLike, name: str):
    if isinstance(h, dict):
        return h.get(name)
    return getattr(h, name, None)


def default_detector(res: ParseResult, slots: Sequence | None = None):
    """默认检测器 = `configs.detect_all`（**键位版**，不含手序）。"""
    from . import configs as _cfg
    return _cfg.detect_all(res, slots)


def hand_detector(res: ParseResult, slots: Sequence | None = None):
    """**手序版**检测器 = `configs_hand.detect_all`（消费 `hands.assign` 的分手结果）。"""
    from . import configs_hand as _ch
    return _ch.detect_all(res, slots)


#: CLI 的 ``--detector`` 取值
DETECTORS: dict[str, Callable] = {"key": default_detector, "hand": hand_detector}


def bar_config_table(res: ParseResult, slots: Sequence | None = None,
                     detector: Callable | None = None,
                     bar_configs: dict[int, set[str]] | None = None,
                     hits: Iterable[HitLike] | None = None
                     ) -> dict[int, set[str]]:
    """``{小节号: {配置名, ...}}``。三种注入方式，优先级 bar_configs > hits > detector。"""
    if bar_configs is not None:
        return {int(k): set(v) for k, v in bar_configs.items()}
    if hits is None:
        hits = (detector or default_detector)(res, slots)
    table: dict[int, set[str]] = {}
    for h in hits:
        cfg = _hit_field(h, "config")
        a, b = _hit_field(h, "bar_start"), _hit_field(h, "bar_end")
        if cfg is None or a is None:
            continue
        b = a if b is None else b
        for m in range(int(a), int(b) + 1):
            table.setdefault(m, set()).add(str(cfg))
    return table


# ---------------------------------------------------------------------------
# 逐小节内在难度画像
# ---------------------------------------------------------------------------


@dataclass
class BarProfile:
    """一张谱、一小节的内在难度画像（不需要音频）。"""

    chart: str
    measure: int
    bpm: float = 0.0
    level: float = 0.0
    pos: float = 0.0              # 曲内相对位置 [0,1]
    n_notes: int = 0              # 官方口径 note 数
    n_slots: int = 0              # 时间槽数（同刻算一个）
    nps: float = 0.0              # 时间槽数 / 小节秒长
    finest_divisor: float = 0.0
    div_share: dict = field(default_factory=dict)   # {分音: 该小节 note 占比}
    each_ratio: float = 0.0       # 参与 each 的 note 占比
    slide_ratio: float = 0.0      # slide 轨占 note 比
    hold_ratio: float = 0.0
    break_ratio: float = 0.0
    touch_ratio: float = 0.0
    move_mean: float = 0.0        # 相邻时间槽键位位移均值（0–4）
    equiv_speed: float = 0.0      # 等效速度 = 最细分音/4 × BPM/60（键/秒）
    muri_count: int = 0           # 无理代理事件数（叠键/外键/撞尾）
    hardness: float = 0.0         # 曲内归一硬度（含配置项）
    hardness_noconfig: float = 0.0   # 去掉配置项后重新归一的硬度
    h_class: str = ""
    h_class_noconfig: str = ""
    configs: tuple[str, ...] = ()

    def to_row(self) -> dict:
        d = {"chart": self.chart, "measure": self.measure, "bpm": self.bpm,
             "level": self.level, "pos": round(self.pos, 4),
             "n_notes": self.n_notes, "n_slots": self.n_slots,
             "nps": round(self.nps, 3), "finest_divisor": self.finest_divisor,
             "each_ratio": round(self.each_ratio, 4),
             "slide_ratio": round(self.slide_ratio, 4),
             "hold_ratio": round(self.hold_ratio, 4),
             "break_ratio": round(self.break_ratio, 4),
             "touch_ratio": round(self.touch_ratio, 4),
             "move_mean": round(self.move_mean, 4),
             "equiv_speed": round(self.equiv_speed, 3),
             "muri_count": self.muri_count,
             "hardness": round(self.hardness, 4),
             "hardness_noconfig": round(self.hardness_noconfig, 4),
             "h_class": self.h_class, "h_class_noconfig": self.h_class_noconfig,
             "configs": list(self.configs)}
        d["div_share"] = {str(k): round(v, 4) for k, v in self.div_share.items()}
        return d


def chart_profile(res: ParseResult, name: str = "", level: float = 0.0,
                  detector: Callable | None = None,
                  bar_configs: dict[int, set[str]] | None = None,
                  slots: Sequence | None = None) -> list[BarProfile]:
    """把一份谱面变成逐小节内在难度画像。

    硬度分沿用 `configs.bar_hardness`（五项加权和），并额外算一份**去掉配置项**、
    权重重新归一的版本——凡是"配置 × 硬度"的结论都以去环版本为准。
    """
    from . import configs as _cfg

    slots = list(_cfg.build_slots(res)) if slots is None else list(slots)
    table = bar_config_table(res, slots, detector=detector, bar_configs=bar_configs)
    hits = [_FakeHit(c, m) for m, cs in table.items() for c in cs]
    muri = _cfg.detect_muri(res)
    hard = _cfg.bar_hardness(res, slots, hits, muri)
    w = _cfg.PARAMS["hardness_weights"]
    w_nc = w["kind"] + w["move"] + w["speed"] + w["muri"]

    by_bar_notes: dict[int, list[NoteEvent]] = {}
    for n in res.notes:
        by_bar_notes.setdefault(n.measure, []).append(n)
    by_bar_slots: dict[int, list] = {}
    for s in slots:
        by_bar_slots.setdefault(s.measure, []).append(s)

    measures = [h.measure for h in hard]
    lo, hi = (min(measures), max(measures)) if measures else (0, 0)
    span = max(hi - lo, 1)

    raws_nc: dict[int, float] = {}
    out: list[BarProfile] = []
    for h in hard:
        m = h.measure
        ns = by_bar_notes.get(m, [])
        ss = by_bar_slots.get(m, [])
        raw_nc = (w["kind"] * h.term_kind + w["move"] * h.term_move
                  + w["speed"] * h.term_speed + w["muri"] * h.term_muri) / w_nc
        raws_nc[m] = raw_nc
        bpm = ns[0].bpm if ns else (ss[0].bpm if ss else 0.0)
        n_notes = len(ns)
        div_share: dict[float, float] = {}
        if n_notes:
            for n in ns:
                div_share[float(n.divisor)] = div_share.get(float(n.divisor), 0.0) + 1.0
            div_share = {k: v / n_notes for k, v in div_share.items()}
        finest = max((float(n.divisor) for n in ns), default=0.0)
        out.append(BarProfile(
            chart=name, measure=m, bpm=float(bpm), level=float(level),
            pos=(m - lo) / span,
            n_notes=n_notes, n_slots=len(ss), nps=h.nps,
            finest_divisor=finest, div_share=div_share,
            each_ratio=(sum(1 for n in ns if n.is_each) / n_notes) if n_notes else 0.0,
            slide_ratio=(sum(1 for n in ns if n.kind == "slide_track") / n_notes)
            if n_notes else 0.0,
            hold_ratio=(sum(1 for n in ns if n.kind in ("hold", "touch_hold"))
                        / n_notes) if n_notes else 0.0,
            break_ratio=(sum(1 for n in ns if n.is_break) / n_notes) if n_notes else 0.0,
            touch_ratio=(sum(1 for n in ns if n.kind in ("touch", "touch_hold"))
                         / n_notes) if n_notes else 0.0,
            move_mean=_move_mean(ss),
            equiv_speed=(finest / 4.0) * (float(bpm) / 60.0) if bpm else 0.0,
            muri_count=int(round(h.term_muri * 3.0)),
            hardness=h.hardness, h_class=h.hardness_class,
            configs=h.configs,
        ))

    live = [raws_nc[b.measure] for b in out if b.n_notes > 0]
    if live:
        mn, mx = min(live), max(live)
        rng = (mx - mn) or 1.0
        srt = sorted(live)
        q1 = srt[len(srt) // 3]
        q2 = srt[2 * len(srt) // 3]
        for b in out:
            r = raws_nc[b.measure]
            if b.n_notes == 0:
                b.hardness_noconfig, b.h_class_noconfig = 0.0, ""
            else:
                b.hardness_noconfig = (r - mn) / rng
                b.h_class_noconfig = "低" if r <= q1 else ("中" if r <= q2 else "高")
    return out


@dataclass
class _FakeHit:
    """把 ``{小节: 配置集}`` 还原成 `bar_hardness` 需要的片段形状。"""

    config: str
    bar_start: int

    @property
    def bar_end(self) -> int:
        return self.bar_start


def _move_mean(ss: Sequence) -> float:
    """相邻时间槽的键位位移均值（环上 0–4），与 `configs._move_term` 的相邻项同口径。"""
    from . import configs as _cfg

    vals: list[float] = []
    for a, b in zip(ss[:-1], ss[1:]):
        if not a.keys or not b.keys:
            continue
        vals.append(sum(_cfg.cdist(x, y) for x in a.keys for y in b.keys)
                    / (len(a.keys) * len(b.keys)))
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def mean_ci(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float, int]:
    """均值 + 正态近似 95% 置信区间 ``(mean, lo, hi, n)``。"""
    v = [float(x) for x in values if _finite(x)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = sum(v) / n
    if n < 2:
        return m, m, m, n
    var = sum((x - m) ** 2 for x in v) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - z * se, m + z * se, n


#: 排序表里的内在难度列（列名 → 取值函数）
INTRINSIC_FIELDS: dict[str, Callable[[BarProfile], float]] = {
    "hardness_noconfig": lambda b: b.hardness_noconfig,
    "hardness": lambda b: b.hardness,
    "n_notes": lambda b: float(b.n_notes),
    "n_slots": lambda b: float(b.n_slots),
    "nps": lambda b: b.nps,
    "equiv_speed": lambda b: b.equiv_speed,
    "move_mean": lambda b: b.move_mean,
    "each_ratio": lambda b: b.each_ratio,
    "slide_ratio": lambda b: b.slide_ratio,
    "hold_ratio": lambda b: b.hold_ratio,
    "break_ratio": lambda b: b.break_ratio,
    "muri_rate": lambda b: 1.0 if b.muri_count > 0 else 0.0,
    "finest_divisor": lambda b: b.finest_divisor,
    "level": lambda b: b.level,
    "bpm": lambda b: b.bpm,
}


def aggregate_by_config(profiles: Iterable[BarProfile],
                        fields: dict | None = None) -> dict[str, dict]:
    """按配置聚合内在难度指标（每配置一行，含 95% CI 与 n）。

    一小节可以同时命中多个配置 → 该小节在每个命中的配置下各计一次
    （所以各配置 n 的总和 > 小节数，这是有意的："这个配置出现时，
    周围长什么样"）。
    """
    fields = INTRINSIC_FIELDS if fields is None else fields
    bucket: dict[str, list[BarProfile]] = {}
    allbars: list[BarProfile] = []
    for b in profiles:
        allbars.append(b)
        for c in b.configs:
            bucket.setdefault(c, []).append(b)
    out: dict[str, dict] = {}
    for cfg, rows in bucket.items():
        rec: dict = {"n_bars": len(rows),
                     "share_of_bars": len(rows) / max(1, len(allbars))}
        for k, fn in fields.items():
            m, lo, hi, n = mean_ci([fn(b) for b in rows])
            rec[k] = m
            rec[k + "_lo"] = lo
            rec[k + "_hi"] = hi
            rec[k + "_n"] = n
        for d in DIVISORS_REPORTED:
            rec[f"div{int(d)}"] = sum(b.div_share.get(d, 0.0) for b in rows) / len(rows)
        for name, _, _ in LEVEL_BUCKETS:
            sub = [b for b in rows if level_bucket(b.level) == name]
            rec[f"lv_{name}"] = len(sub) / len(rows)
        for name, _, _ in BPM_BUCKETS:
            sub = [b for b in rows if bpm_bucket(b.bpm) == name]
            rec[f"bpm_{name}"] = len(sub) / len(rows)
        out[cfg] = rec
    return out


def intrinsic_score(rec: dict, redline: float = 22.2) -> tuple[float, float, float]:
    """复合内在难度分 ``(score, D, H)``。

    - ``D``（密）= 该配置所在小节的平均 NPS ÷ ``redline``（知识 015 的短爆发上限
      22.2 键/秒），截断到 [0,1]；
    - ``H``（难）= 去掉配置项后的硬度分（note 种类 / 位移 / 速度 / 无理四项）；
    - ``score = (D + H) / 2``。

    两项分开给是因为它们**预测的东西不一样**（见报告 §3.3）：``D`` 预测
    "这个配置出现在多高定数的谱里"，``H`` 预测"放在曲内多强的位置"。
    """
    d = min(max(float(rec.get("nps", 0.0)) / redline, 0.0), 1.0)
    h = float(rec.get("hardness_noconfig", 0.0))
    return (d + h) / 2.0, d, h


def placement_by_config(rows: Iterable[dict], config_key: str = "configs",
                        intensity_key: str = "I_bar", mode_key: str = "mode",
                        tier_key: str = "I_tier3", modes: Sequence[str] = ()
                        ) -> dict[str, dict]:
    """每配置的**外在放置**画像（需要音频配对的逐小节行）。

    强度分位（均值/中位/四分位）、高/低强度档的富集 lift、曲内相对位置、
    coverage 与各**采音方式**占比（三个词见 `tools/calibration/sampling.py`）。
    """
    rows = list(rows)
    base: dict[str, int] = {}
    for r in rows:
        t = str(r.get(tier_key, "") or "")
        if t:
            base[t] = base.get(t, 0) + 1
    n_base = sum(base.values()) or 1
    bucket: dict[str, list[dict]] = {}
    for r in rows:
        for c in r.get(config_key, ()) or ():
            bucket.setdefault(c, []).append(r)
    out: dict[str, dict] = {}
    for cfg, sub in bucket.items():
        iv = sorted(float(r[intensity_key]) for r in sub
                    if _finite(r.get(intensity_key)))
        cov = [float(r["coverage"]) for r in sub if _finite(r.get("coverage"))]
        rec: dict = {"n_bars": len(sub)}
        if iv:
            rec.update({"I_mean": sum(iv) / len(iv), "I_median": _pct(iv, 0.50),
                        "I_q1": _pct(iv, 0.25), "I_q3": _pct(iv, 0.75)})
            for t, k in base.items():
                share = sum(1 for r in sub
                            if str(r.get(tier_key, "")) == t) / len(sub)
                rec[f"lift_I_{t}"] = share / (k / n_base) if k else float("nan")
        rec["pos_mean"] = (sum(float(r.get("pos", 0.0)) for r in sub) / len(sub))
        rec["coverage"] = (sum(cov) / len(cov)) if cov else float("nan")
        cnt: dict[str, int] = {}
        for r in sub:
            m = str(r.get(mode_key, ""))
            cnt[m] = cnt.get(m, 0) + 1
        for m in (modes or sorted(cnt)):
            rec[f"mode_{m}"] = cnt.get(m, 0) / len(sub)
        out[cfg] = rec
    return out


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(int(q * (len(sorted_vals) - 1)), len(sorted_vals) - 1)
    return float(sorted_vals[i])


def cooccurrence(profiles: Iterable[BarProfile], window: int = 1,
                 top: int = 3) -> dict[str, list[tuple[str, float, int]]]:
    """配置共现：同小节 ± ``window`` 小节内还出现了哪些配置（按条件概率排序）。

    返回 ``{配置: [(共现配置, P(另一个|本配置), 次数), ...]}``，每个配置取前 ``top``。
    """
    by_chart: dict[str, dict[int, set[str]]] = {}
    for b in profiles:
        by_chart.setdefault(b.chart, {})[b.measure] = set(b.configs)
    counts: dict[str, dict[str, int]] = {}
    base: dict[str, int] = {}
    for _, bars in by_chart.items():
        for m, cfgs in bars.items():
            near: set[str] = set()
            for d in range(-window, window + 1):
                near |= bars.get(m + d, set())
            for c in cfgs:
                base[c] = base.get(c, 0) + 1
                for o in near:
                    if o == c:
                        continue
                    counts.setdefault(c, {})[o] = counts.setdefault(c, {}).get(o, 0) + 1
    out: dict[str, list[tuple[str, float, int]]] = {}
    for c, row in counts.items():
        items = sorted(row.items(), key=lambda kv: -kv[1])[:top]
        out[c] = [(o, n / max(1, base.get(c, 1)), n) for o, n in items]
    return out


def lift_table(rows: Iterable[dict], key: str, config_key: str = "configs"
               ) -> dict[str, dict[str, float]]:
    """``P(配置 | key 取某值) / P(配置)`` 的富集倍数表。

    ``rows`` 是任意带 ``key`` 与配置集合的字典行（可以是加了音频列的逐小节行）。
    """
    rows = list(rows)
    total = len(rows)
    by_val: dict[str, int] = {}
    cfg_total: dict[str, int] = {}
    cfg_by_val: dict[str, dict[str, int]] = {}
    for r in rows:
        v = str(r.get(key, "") or "")
        by_val[v] = by_val.get(v, 0) + 1
        for c in r.get(config_key, ()) or ():
            cfg_total[c] = cfg_total.get(c, 0) + 1
            cfg_by_val.setdefault(c, {})[v] = cfg_by_val.setdefault(c, {}).get(v, 0) + 1
    out: dict[str, dict[str, float]] = {}
    for c, per in cfg_by_val.items():
        p_c = cfg_total[c] / total if total else 0.0
        out[c] = {}
        for v, n in per.items():
            denom = by_val.get(v, 0)
            out[c][v] = ((n / denom) / p_c) if denom and p_c else float("nan")
    return out


def recommend_matrix(rows: Iterable[dict], row_key: str = "pool_tier",
                     col_key: str = "I_tier", config_key: str = "configs",
                     mode_key: str = "mode", top: int = 3,
                     min_count: int = 10) -> dict:
    """「音乐事件密度 × 强度档 → 推荐配置」矩阵。

    每格给出：n、该格里 **lift 最高**的前 ``top`` 个配置（``top_configs``，
    要求该格出现次数 ≥ ``min_count``）、**出现率最高**的前 ``top`` 个配置
    （``top_common``）、最常见的**采音方式**前 2、以及该格的平均 coverage。

    两份榜都给：lift 榜回答"这一格比别处更爱用什么"，出现率榜回答
    "这一格实际最常写的是什么"——只看 lift 会选出小样本配置，
    只看出现率则每格都是全库高频配置（错位 / 连续双押 / 子弹）。
    """
    rows = list(rows)
    total = len(rows)
    cfg_total: dict[str, int] = {}
    for r in rows:
        for c in r.get(config_key, ()) or ():
            cfg_total[c] = cfg_total.get(c, 0) + 1
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        cells.setdefault((str(r.get(row_key, "")), str(r.get(col_key, ""))), []).append(r)
    out: dict = {}
    for (rk, ck), sub in cells.items():
        if not rk or not ck:
            continue
        n = len(sub)
        cnt: dict[str, int] = {}
        modes: dict[str, int] = {}
        covs: list[float] = []
        for r in sub:
            for c in r.get(config_key, ()) or ():
                cnt[c] = cnt.get(c, 0) + 1
            modes[str(r.get(mode_key, ""))] = modes.get(str(r.get(mode_key, "")), 0) + 1
            if _finite(r.get("coverage")):
                covs.append(float(r["coverage"]))
        ranked = []
        for c, k in cnt.items():
            p_cell = k / n
            p_all = cfg_total.get(c, 0) / total if total else 0.0
            ranked.append((c, p_cell, (p_cell / p_all) if p_all else float("nan"), k))
        by_lift = sorted([x for x in ranked if x[3] >= min_count],
                         key=lambda x: (-x[2], -x[1]))
        by_common = sorted(ranked, key=lambda x: -x[1])
        out[f"{rk}|{ck}"] = {
            "row": rk, "col": ck, "n": n,
            "top_configs": [{"config": c, "p": p, "lift": lf, "n": k}
                            for c, p, lf, k in by_lift[:top]],
            "top_common": [{"config": c, "p": p, "lift": lf, "n": k}
                           for c, p, lf, k in by_common[:top]],
            "top_modes": [{"mode": m, "p": k / n}
                          for m, k in sorted(modes.items(), key=lambda kv: -kv[1])[:2]],
            "coverage": (sum(covs) / len(covs)) if covs else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _csv_rows_config(agg: dict[str, dict], co: dict,
                     place: dict | None = None) -> list[dict]:
    place = place or {}
    scored = {c: intrinsic_score(r) for c, r in agg.items()}
    rows = []
    for cfg, rec in sorted(agg.items(), key=lambda kv: -scored[kv[0]][0]):
        sc, dv, hv = scored[cfg]
        r = {"scope": "config", "config": cfg,
             "intrinsic_score": round(sc, 5), "term_D": round(dv, 5),
             "term_H": round(hv, 5)}
        for k, v in rec.items():
            r[k] = round(v, 5) if isinstance(v, float) else v
        for k, v in (place.get(cfg) or {}).items():
            r["place_" + k] = round(v, 5) if isinstance(v, float) else v
        r["cooccur_top3"] = "; ".join(f"{o}:{p:.2f}" for o, p, _ in co.get(cfg, []))
        rows.append(r)
    return rows


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import csv
    import json
    import sys
    from pathlib import Path

    from . import corpus
    from .simai_parser import parse_chart

    p = argparse.ArgumentParser(
        prog="python -m tools.chart_analysis.config_profiles",
        description="配置难度/密度分级与强度对应表（388 官方谱 + 160 首音频配对）")
    p.add_argument("--chart-dir", default=None, help="官方谱目录（默认 resource/official-chart）")
    p.add_argument("--bars-json", default=None,
                   help="音频配对逐小节数据（JSON：{'bars':[...]}），给外在放置与矩阵用")
    p.add_argument("--json", default=None, help="完整结果 JSON 输出")
    p.add_argument("--csv", default=None, help="汇总 CSV 输出（每配置一行 + 矩阵长表）")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--detector", choices=sorted(DETECTORS), default="key",
                   help="key = 键位版 configs.py（默认）；hand = 手序版 configs_hand.py")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    det = DETECTORS[a.detector]

    files = corpus.discover(Path(a.chart_dir)) if a.chart_dir else corpus.discover()
    if a.limit:
        files = files[:a.limit]
    profiles: list[BarProfile] = []
    n_err = 0
    for i, cf in enumerate(files, 1):
        try:
            res = parse_chart(cf.read(), name=cf.name)
            profiles.extend(chart_profile(res, name=cf.name,
                                          level=cf.internal_level or 0.0,
                                          detector=det))
        except Exception as exc:                       # noqa: BLE001
            n_err += 1
            print(f"[ERR] {cf.name}: {exc!r}", file=sys.stderr)
        if not a.quiet and i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)
    agg = aggregate_by_config(profiles)
    co = cooccurrence(profiles)
    result: dict = {"n_charts": len(files), "n_errors": n_err,
                    "detector": a.detector,
                    "n_bars": len(profiles), "configs": agg,
                    "cooccurrence": {k: [list(x) for x in v] for k, v in co.items()}}

    if a.bars_json:
        doc = json.loads(Path(a.bars_json).read_text(encoding="utf-8"))
        bars = doc.get("bars", doc if isinstance(doc, list) else [])
        iv = sorted(float(r["I_bar"]) for r in bars if _finite(r.get("I_bar")))
        cuts = ([iv[len(iv) // 3], iv[2 * len(iv) // 3]] if len(iv) >= 3 else [0.0, 1.0])
        for r in bars:
            r.setdefault("pool_tier", density_tier(r.get("n_pool")))
            r["I_tier3"] = tier_by_cuts(r.get("I_bar"), cuts)
        result["placement"] = {
            "n_bars": len(bars), "intensity_cuts": cuts,
            "by_segment": lift_table(bars, "label_ja"),
            "by_intensity": lift_table(bars, "I_tier3"),
            "by_config": placement_by_config(bars),
            "matrix": recommend_matrix(bars, row_key="pool_tier", col_key="I_tier3"),
        }

    if a.json:
        Path(a.json).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"写出 {a.json}")
    if a.csv:
        rows = _csv_rows_config(
            agg, co, (result.get("placement") or {}).get("by_config"))
        mat = (result.get("placement") or {}).get("matrix", {})
        for k, cell in sorted(mat.items()):
            rows.append({"scope": "matrix", "config": "",
                         "matrix_row": cell["row"], "matrix_col": cell["col"],
                         "n_bars": cell["n"],
                         "coverage": round(cell["coverage"], 4)
                         if _finite(cell["coverage"]) else "",
                         "top_configs": "; ".join(
                             f"{c['config']}({c['lift']:.2f}/{c['p']:.2f})"
                             for c in cell["top_configs"]),
                         "top_common": "; ".join(
                             f"{c['config']}({c['p']:.2f})"
                             for c in cell.get("top_common", [])),
                         "top_modes": "; ".join(
                             f"{m['mode']}:{m['p']:.2f}" for m in cell["top_modes"])})
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print(f"写出 {a.csv}（{len(rows)} 行）")
    if not a.quiet:
        print(f"谱面 {len(files)}（错 {n_err}）  小节 {len(profiles)}  配置 {len(agg)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
