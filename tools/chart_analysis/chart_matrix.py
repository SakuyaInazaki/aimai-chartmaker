#!/usr/bin/env python3
"""谱面级「定数 × BPM × 配置」对照矩阵。

用户 2026-09-19 的裁定
--------------------
    「大体上肯定是按照每张谱的定数和它对应的 bpm 和它的配置这样对照着来看的啊，
      没有说一定要有什么什么配置，一个谱也不可能把所有的配置都放进去。」

所以本模块**不给配置打合成难度分、不排难度序**（`config_profiles.intrinsic_score`
的 (D+H)/2 排序已撤回）。它做的唯一一件事是**把每张官方谱摊成一行**——
定数、BPM、它实际用了哪些配置、各占多少小节——然后按 **定数档 × BPM 档**
分格，看"同一格里的谱长什么样、换一格又变成什么样"。

配置不是必选项：一张谱只会用到全部配置里的一小撮，本模块的"出现率"
（多少张谱用了它）与"小节占比"（用了的谱里它占多少小节）分开报，
就是为了不把"没用某配置"读成"缺了什么"。

⚠️ 检测器可替换
--------------
"这一小节命中了哪些配置"一律经 :func:`config_profiles.bar_config_table`
取得（可注入 ``bar_configs`` / ``hits`` / ``detector``）。本模块**不直接调用**
`configs.py` 的任何检测函数，等 `hands.py` 手序版落地、`configs.py` 按手重写后，
只要换注入的检测器即可原样重跑：

    python -m tools.chart_analysis.chart_matrix --csv ... --json ...

⚠️ 本轮跑的是**键位版**检测器（`configs.detect_all`：按键位与时间几何判定，
不含左右手分配），所有结论标注为"待手序版重跑"。

片段数口径
---------
因为命中只经"小节表"这一层，**片段数 = 该配置在小节表上的连续小节段数**
（不是检测器内部的 ConfigHit 条数）。这个口径只依赖注入的表，换检测器不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from .config_profiles import bar_config_table
from .simai_parser import ParseResult

# ---------------------------------------------------------------------------
# 分档（**agent 设定**，按用户给的档口；样本量见报告 §2）
# ---------------------------------------------------------------------------

#: 定数档（左闭右开）
LEVEL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("13.0-13.2", 13.0, 13.3),
    ("13.3-13.5", 13.3, 13.6),
    ("13.6-13.8", 13.6, 13.9),
    ("13.9-14.1", 13.9, 14.2),
    ("14.2-14.5", 14.2, 14.6),
)

#: BPM 档（左闭右开；取"占时最长的那一段 BPM"）
BPM_BANDS: tuple[tuple[str, float, float], ...] = (
    ("<140", 0.0, 140.0),
    ("140-169", 140.0, 170.0),
    ("170-199", 170.0, 200.0),
    (">=200", 200.0, 1e9),
)

#: 报告固定列出的分音
DIVISORS_REPORTED: tuple[float, ...] = (4.0, 8.0, 12.0, 16.0, 24.0, 32.0)

#: 判定"这张谱把某配置当主料"的小节占比阈值（**agent 设定**）
MAIN_SHARE = 0.10

#: 一个格子至少几张谱才把它的数字当话讲（**agent 设定**）
MIN_CELL_N = 8


def level_band(level: float | None) -> str:
    if level is None or not _finite(level):
        return ""
    for name, lo, hi in LEVEL_BANDS:
        if lo <= float(level) < hi:
            return name
    return ""


def bpm_band(bpm: float | None) -> str:
    if bpm is None or not _finite(bpm) or float(bpm) <= 0:
        return ""
    for name, lo, hi in BPM_BANDS:
        if lo <= float(bpm) < hi:
            return name
    return BPM_BANDS[-1][0]


def _finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# BPM 画像
# ---------------------------------------------------------------------------


def bpm_profile(res: ParseResult) -> list[tuple[float, float]]:
    """``[(BPM, 占时秒), ...]``，按占时降序；同一 BPM 的多段合并。

    变速谱据此记多段；主 BPM = 占时最长的一段。
    """
    ev = sorted(res.bpm_events)
    if not ev:
        return []
    agg: dict[float, float] = {}
    for i, (beat, bpm) in enumerate(ev):
        nxt = ev[i + 1][0] if i + 1 < len(ev) else res.total_beats
        if not _finite(bpm) or float(bpm) <= 0:
            continue
        dur = max(0.0, float(nxt) - float(beat)) * 60.0 / float(bpm)
        agg[float(bpm)] = agg.get(float(bpm), 0.0) + dur
    return sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))


def weighted_median_bpm(segments: Sequence[tuple[float, float]]) -> float:
    """占时加权中位 BPM。

    渐变速谱（``916-ってゐ！``、``917-東方妖々夢`` 各有 184 / 311 段微调速）里
    "占时最长的一段"只占 3–4%，主 BPM 不稳；加权中位是稳健的交叉验证值。
    """
    segs = sorted((float(b), float(s)) for b, s in segments if float(s) > 0)
    tot = sum(s for _, s in segs)
    if not segs or tot <= 0:
        return 0.0
    acc = 0.0
    for b, s in segs:
        acc += s
        if acc >= tot / 2.0:
            return b
    return segs[-1][0]


# ---------------------------------------------------------------------------
# 一谱一行
# ---------------------------------------------------------------------------


@dataclass
class ChartRow:
    """一张官方谱摊成的一行（定数 × BPM × 配置）。"""

    chart: str = ""
    simai_id: str = ""
    title: str = ""
    difficulty: str = ""          # mas / remas
    level: float = 0.0
    level_band: str = ""
    bpm_main: float = 0.0
    bpm_band: str = ""
    bpm_segments: tuple[tuple[float, float], ...] = ()   # (BPM, 秒)
    bpm_main_share: float = 1.0   # 主 BPM 占时比（<1 即变速谱）
    bpm_median_time: float = 0.0  # 占时加权中位 BPM（渐变速谱的稳健代表值）
    n_bpm: int = 1

    n_bars: int = 0               # 有 note 的小节数
    bar_lo: int = 0
    bar_hi: int = 0
    seconds: float = 0.0          # 有 note 的小节合计时长
    n_notes: int = 0              # 官方口径 note 数
    n_slots: int = 0              # 时间槽数
    nps: float = 0.0              # note 数 / 有 note 小节时长
    notes_per_bar: float = 0.0

    each_ratio: float = 0.0
    slide_ratio: float = 0.0
    hold_ratio: float = 0.0
    break_ratio: float = 0.0
    touch_ratio: float = 0.0
    div_share: dict = field(default_factory=dict)   # {分音: note 占比}
    main_divisor: float = 0.0

    #: {配置: {"n_bars", "bar_share", "n_segments"}}
    configs: dict = field(default_factory=dict)
    #: 逐小节配置表（个案用）
    bar_configs: dict = field(default_factory=dict)

    @property
    def config_set(self) -> frozenset:
        return frozenset(self.configs)

    @property
    def vocab(self) -> int:
        """配置词汇量 = 这张谱用到的配置种类数。"""
        return len(self.configs)

    def main_configs(self, thresh: float = MAIN_SHARE) -> frozenset:
        """占小节 ≥ ``thresh`` 的配置集合（这张谱的"主料"）。"""
        return frozenset(c for c, d in self.configs.items()
                         if d["bar_share"] >= thresh)

    @property
    def vocab_main(self) -> int:
        return len(self.main_configs())

    def share(self, config: str) -> float:
        """小节占比。

        谱内每小节都是 4 拍、同一 BPM 下等长，所以这同时就是**时长占比**
        （变速谱除外，它按小节数近似）。
        """
        d = self.configs.get(config)
        return float(d["bar_share"]) if d else 0.0

    def seg_per_min(self, config: str) -> float:
        """每分钟的配置片段数。

        小节占比会被**小节粒度**放大：低 BPM 谱一小节长（<140 档中位 1.89 秒 vs
        ≥200 档 1.14 秒），一次命中就整小节算进去。每分钟片段数不吃这个偏差，
        是跨 BPM 档比较时的稳健口径。
        """
        d = self.configs.get(config)
        if not d or self.seconds <= 0:
            return 0.0
        return float(d["n_segments"]) / (self.seconds / 60.0)

    def seg_bars(self, config: str) -> float:
        """配置片段的平均长度（小节）。"""
        d = self.configs.get(config)
        if not d or not d["n_segments"]:
            return 0.0
        return float(d["n_bars"]) / float(d["n_segments"])

    def to_row(self) -> dict:
        d = {
            "chart": self.chart, "simai_id": self.simai_id, "title": self.title,
            "difficulty": self.difficulty, "level": self.level,
            "level_band": self.level_band,
            "bpm_main": round(self.bpm_main, 3), "bpm_band": self.bpm_band,
            "n_bpm": self.n_bpm,
            "bpm_main_share": round(self.bpm_main_share, 4),
            "bpm_median_time": round(self.bpm_median_time, 3),
            "bpm_band_median": bpm_band(self.bpm_median_time),
            "bpm_segments": "; ".join(f"{b:g}@{s:.1f}s" for b, s in self.bpm_segments),
            "n_bars": self.n_bars, "seconds": round(self.seconds, 2),
            "n_notes": self.n_notes, "n_slots": self.n_slots,
            "nps": round(self.nps, 3),
            "notes_per_bar": round(self.notes_per_bar, 3),
            "each_ratio": round(self.each_ratio, 4),
            "slide_ratio": round(self.slide_ratio, 4),
            "hold_ratio": round(self.hold_ratio, 4),
            "break_ratio": round(self.break_ratio, 4),
            "touch_ratio": round(self.touch_ratio, 4),
            "main_divisor": self.main_divisor,
            "vocab": self.vocab, "vocab_main": self.vocab_main,
        }
        for dv in DIVISORS_REPORTED:
            d[f"div{int(dv)}"] = round(self.div_share.get(dv, 0.0), 4)
        d["config_list"] = "; ".join(
            f"{c}:{v['bar_share']:.2f}/{v['n_segments']}"
            for c, v in sorted(self.configs.items(),
                               key=lambda kv: -kv[1]["bar_share"]))
        d["main_config_list"] = "; ".join(sorted(self.main_configs()))
        return d


def _segments_from_bars(bars: Sequence[int]) -> int:
    """连续小节段数（片段数口径：只依赖小节表）。"""
    if not bars:
        return 0
    s = sorted(set(int(b) for b in bars))
    n = 1
    for a, b in zip(s[:-1], s[1:]):
        if b != a + 1:
            n += 1
    return n


def chart_row(res: ParseResult, *, chart: str = "", simai_id: str = "",
              title: str = "", difficulty: str = "", level: float = 0.0,
              detector: Callable | None = None,
              bar_configs: dict[int, set[str]] | None = None,
              slots: Sequence | None = None) -> ChartRow:
    """把一份解析好的谱面摊成一行。

    配置命中经 :func:`config_profiles.bar_config_table` 取得——传
    ``bar_configs`` / ``detector`` 即可整体换检测器。
    """
    table = bar_config_table(res, slots, detector=detector, bar_configs=bar_configs)

    bars_with_notes: dict[int, list] = {}
    for n in res.notes:
        bars_with_notes.setdefault(int(n.measure), []).append(n)
    played = sorted(bars_with_notes)
    n_bars = len(played)

    segs = bpm_profile(res)
    total_sec = sum(s for _, s in segs) or res.total_seconds
    bpm_main = segs[0][0] if segs else 0.0
    main_share = (segs[0][1] / total_sec) if (segs and total_sec > 0) else 1.0

    # 有 note 的小节合计时长：逐小节按该小节首 note 的 BPM 折算 4 拍
    seconds = 0.0
    for m in played:
        b = float(bars_with_notes[m][0].bpm or 0.0)
        if b > 0:
            seconds += 4.0 * 60.0 / b

    n_notes = res.counts["notes"]
    n_slots = len({n.group_index for n in res.notes})

    div_cnt: dict[float, float] = {}
    for n in res.notes:
        div_cnt[float(n.divisor)] = div_cnt.get(float(n.divisor), 0.0) + 1.0
    tot = sum(div_cnt.values()) or 1.0
    div_share = {k: v / tot for k, v in div_cnt.items()}
    main_div = max(div_share.items(), key=lambda kv: kv[1])[0] if div_share else 0.0

    allnotes = res.notes
    nn = len(allnotes) or 1
    cfg_bars: dict[str, list[int]] = {}
    for m, cs in table.items():
        if m not in bars_with_notes:      # 只统计有 note 的小节
            continue
        for c in cs:
            cfg_bars.setdefault(c, []).append(m)
    configs = {
        c: {"n_bars": len(bs), "bar_share": len(bs) / max(1, n_bars),
            "n_segments": _segments_from_bars(bs)}
        for c, bs in cfg_bars.items()
    }

    return ChartRow(
        chart=chart, simai_id=simai_id, title=title, difficulty=difficulty,
        level=float(level), level_band=level_band(level),
        bpm_main=bpm_main, bpm_band=bpm_band(bpm_main),
        bpm_segments=tuple(segs), bpm_main_share=main_share, n_bpm=len(segs),
        bpm_median_time=weighted_median_bpm(segs),
        n_bars=n_bars, bar_lo=(played[0] if played else 0),
        bar_hi=(played[-1] if played else 0), seconds=seconds,
        n_notes=n_notes, n_slots=n_slots,
        nps=(n_notes / seconds) if seconds > 0 else 0.0,
        notes_per_bar=(n_notes / n_bars) if n_bars else 0.0,
        each_ratio=sum(1 for n in allnotes if n.is_each) / nn,
        slide_ratio=sum(1 for n in allnotes if n.kind == "slide_track") / nn,
        hold_ratio=sum(1 for n in allnotes if n.kind in ("hold", "touch_hold")) / nn,
        break_ratio=sum(1 for n in allnotes if n.is_break) / nn,
        touch_ratio=sum(1 for n in allnotes if n.kind in ("touch", "touch_hold")) / nn,
        div_share=div_share, main_divisor=main_div,
        configs=configs,
        bar_configs={m: sorted(cs) for m, cs in sorted(table.items())
                     if m in bars_with_notes},
    )


# ---------------------------------------------------------------------------
# 统计小工具
# ---------------------------------------------------------------------------


def median(vals: Sequence[float]) -> float:
    v = sorted(float(x) for x in vals if _finite(x))
    if not v:
        return float("nan")
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def quantile(vals: Sequence[float], q: float) -> float:
    v = sorted(float(x) for x in vals if _finite(x))
    if not v:
        return float("nan")
    i = min(int(q * (len(v) - 1) + 0.5), len(v) - 1)
    return v[i]


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman 秩相关（平均秩处理并列）。"""
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys)
             if _finite(a) and _finite(b)]
    n = len(pairs)
    if n < 3:
        return float("nan")
    rx = _ranks([p[0] for p in pairs])
    ry = _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else float("nan")


def _ranks(v: Sequence[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


# ---------------------------------------------------------------------------
# 定数档 × BPM 档 分格
# ---------------------------------------------------------------------------


def all_configs(rows: Iterable[ChartRow]) -> list[str]:
    s: set[str] = set()
    for r in rows:
        s |= set(r.configs)
    return sorted(s)


def cell_summary(rows: Sequence[ChartRow], configs: Sequence[str] | None = None,
                 top: int = 5, main_share: float = MAIN_SHARE) -> dict:
    """一个格子的汇总。

    每个配置两个数分开给：

    - ``rate``  = **谱面出现率**（这一格里有多少比例的谱用到它，哪怕只有 1 小节）；
    - ``share`` = **小节占比中位**（只在用到它的谱里算，不把没用的 0 拉进中位）。
    """
    rows = list(rows)
    configs = list(configs) if configs is not None else all_configs(rows)
    n = len(rows)
    rec: dict = {"n": n}
    if n == 0:
        return rec
    rec["level_median"] = median([r.level for r in rows])
    rec["bpm_median"] = median([r.bpm_main for r in rows])
    rec["nps_median"] = median([r.nps for r in rows])
    rec["notes_median"] = median([float(r.n_notes) for r in rows])
    rec["bars_median"] = median([float(r.n_bars) for r in rows])
    rec["bar_seconds_median"] = median([r.seconds / r.n_bars for r in rows if r.n_bars])
    rec["seconds_median"] = median([r.seconds for r in rows])
    rec["vocab_median"] = median([float(r.vocab) for r in rows])
    rec["vocab_main_median"] = median([float(r.vocab_main) for r in rows])
    rec["each_median"] = median([r.each_ratio for r in rows])
    rec["slide_median"] = median([r.slide_ratio for r in rows])
    rec["hold_median"] = median([r.hold_ratio for r in rows])
    for dv in DIVISORS_REPORTED:
        rec[f"div{int(dv)}_median"] = median([r.div_share.get(dv, 0.0) for r in rows])
    per: dict[str, dict] = {}
    for c in configs:
        users = [r for r in rows if c in r.configs]
        per[c] = {
            "rate": len(users) / n,
            "n_charts": len(users),
            "share_median": median([r.share(c) for r in users]) if users else 0.0,
            "share_mean_all": sum(r.share(c) for r in rows) / n,
            "seg_median": median([float(r.configs[c]["n_segments"]) for r in users])
            if users else 0.0,
            "seg_per_min_median": median([r.seg_per_min(c) for r in users])
            if users else 0.0,
            "seg_bars_median": median([r.seg_bars(c) for r in users]) if users else 0.0,
            "main_rate": sum(1 for r in rows if r.share(c) >= main_share) / n,
        }
    rec["configs"] = per
    combos: dict[frozenset, int] = {}
    for r in rows:
        combos[r.main_configs(main_share)] = combos.get(r.main_configs(main_share), 0) + 1
    rec["top_combos"] = [
        {"configs": sorted(k), "n": v, "p": v / n}
        for k, v in sorted(combos.items(), key=lambda kv: (-kv[1], -len(kv[0])))[:top]
    ]
    return rec


def grid(rows: Iterable[ChartRow], top: int = 5,
         main_share: float = MAIN_SHARE) -> dict[str, dict]:
    """``{"定数档|BPM档": cell_summary}``，另含 ``"*|BPM档"`` / ``"定数档|*"`` 边际。"""
    rows = list(rows)
    cfgs = all_configs(rows)
    out: dict[str, dict] = {}
    for lb, _, _ in LEVEL_BANDS:
        for bb, _, _ in BPM_BANDS:
            sub = [r for r in rows if r.level_band == lb and r.bpm_band == bb]
            out[f"{lb}|{bb}"] = cell_summary(sub, cfgs, top, main_share)
    for lb, _, _ in LEVEL_BANDS:
        out[f"{lb}|*"] = cell_summary([r for r in rows if r.level_band == lb],
                                      cfgs, top, main_share)
    for bb, _, _ in BPM_BANDS:
        out[f"*|{bb}"] = cell_summary([r for r in rows if r.bpm_band == bb],
                                      cfgs, top, main_share)
    out["*|*"] = cell_summary(rows, cfgs, top, main_share)
    return out


def heatmap(rows: Iterable[ChartRow], metric: str = "rate"
            ) -> dict[str, dict[str, dict[str, float]]]:
    """``{配置: {定数档: {BPM档: 值}}}``；``metric`` ∈ rate / share_median / main_rate。"""
    g = grid(rows)
    cfgs = all_configs(rows)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for c in cfgs:
        out[c] = {}
        for lb, _, _ in LEVEL_BANDS:
            out[c][lb] = {}
            for bb, _, _ in BPM_BANDS:
                cell = g[f"{lb}|{bb}"]
                v = (cell.get("configs", {}).get(c, {}) or {}).get(metric)
                out[c][lb][bb] = float(v) if v is not None else float("nan")
    return out


# ---------------------------------------------------------------------------
# 四个对照问题
# ---------------------------------------------------------------------------


def metric_value(row: ChartRow, config: str, metric: str = "rate") -> float:
    """趋势用的逐谱取值：``rate`` 用没用过（0/1）、``share`` 小节占比、
    ``seg_per_min`` 每分钟片段数（跨 BPM 档比较用这个，见 :meth:`ChartRow.seg_per_min`）。"""
    if metric == "rate":
        return 1.0 if config in row.configs else 0.0
    if metric == "seg_per_min":
        return row.seg_per_min(config)
    if metric == "share":
        return row.share(config)
    raise ValueError(f"未知口径：{metric}")


def trend_by_level(rows: Iterable[ChartRow], metric: str = "rate"
                   ) -> dict[str, dict]:
    """同 BPM 档内、定数升档时每个配置怎么走。

    每个配置给：全库（控 BPM：各 BPM 档内先算 Spearman 再平均）的定数趋势
    ``rho_pooled``、按谱的 Spearman ``rho_raw``、以及各定数档的出现率。
    """
    rows = [r for r in rows if r.level_band and r.bpm_band]
    cfgs = all_configs(rows)
    out: dict[str, dict] = {}
    for c in cfgs:
        per_band: dict[str, float] = {}
        rhos: list[float] = []
        weights: list[float] = []
        for bb, _, _ in BPM_BANDS:
            sub = [r for r in rows if r.bpm_band == bb]
            if len(sub) < 10:
                continue
            ys = [metric_value(r, c, metric) for r in sub]
            rho = spearman([r.level for r in sub], ys)
            if _finite(rho):
                per_band[bb] = rho
                rhos.append(rho)
                weights.append(len(sub))
        rec: dict = {"rho_by_bpm": per_band}
        rec["rho_pooled"] = (sum(r * w for r, w in zip(rhos, weights)) / sum(weights)
                             if rhos else float("nan"))
        ys_all = [metric_value(r, c, metric) for r in rows]
        rec["rho_raw"] = spearman([r.level for r in rows], ys_all)
        rec["n_bands_pos"] = sum(1 for v in per_band.values() if v > 0)
        rec["n_bands_neg"] = sum(1 for v in per_band.values() if v < 0)
        for lb, _, _ in LEVEL_BANDS:
            sub = [r for r in rows if r.level_band == lb]
            rec[f"rate_{lb}"] = (sum(1 for r in sub if c in r.configs) / len(sub)
                                 if sub else float("nan"))
            users = [r for r in sub if c in r.configs]
            rec[f"share_{lb}"] = median([r.share(c) for r in users]) if users else 0.0
            rec[f"spm_{lb}"] = (median([r.seg_per_min(c) for r in users])
                                if users else 0.0)
        out[c] = rec
    return out


def trend_by_bpm(rows: Iterable[ChartRow], metric: str = "rate") -> dict[str, dict]:
    """同定数档内、BPM 升高时每个配置怎么走（控定数的 BPM 趋势）。"""
    rows = [r for r in rows if r.level_band and r.bpm_band]
    cfgs = all_configs(rows)
    out: dict[str, dict] = {}
    for c in cfgs:
        per_band: dict[str, float] = {}
        rhos: list[float] = []
        weights: list[float] = []
        for lb, _, _ in LEVEL_BANDS:
            sub = [r for r in rows if r.level_band == lb]
            if len(sub) < 10:
                continue
            ys = [metric_value(r, c, metric) for r in sub]
            rho = spearman([r.bpm_main for r in sub], ys)
            if _finite(rho):
                per_band[lb] = rho
                rhos.append(rho)
                weights.append(len(sub))
        rec: dict = {"rho_by_level": per_band}
        rec["rho_pooled"] = (sum(r * w for r, w in zip(rhos, weights)) / sum(weights)
                             if rhos else float("nan"))
        ys_all = [metric_value(r, c, metric) for r in rows]
        rec["rho_raw"] = spearman([r.bpm_main for r in rows], ys_all)
        rec["n_bands_pos"] = sum(1 for v in per_band.values() if v > 0)
        rec["n_bands_neg"] = sum(1 for v in per_band.values() if v < 0)
        for bb, _, _ in BPM_BANDS:
            sub = [r for r in rows if r.bpm_band == bb]
            rec[f"rate_{bb}"] = (sum(1 for r in sub if c in r.configs) / len(sub)
                                 if sub else float("nan"))
            users = [r for r in sub if c in r.configs]
            rec[f"share_{bb}"] = median([r.share(c) for r in users]) if users else 0.0
            rec[f"spm_{bb}"] = (median([r.seg_per_min(c) for r in users])
                                if users else 0.0)
        out[c] = rec
    return out


def vocab_stats(rows: Iterable[ChartRow]) -> dict:
    """配置词汇量分布 + 与定数/BPM 的关系。"""
    rows = list(rows)
    v = [float(r.vocab) for r in rows]
    vm = [float(r.vocab_main) for r in rows]
    hist: dict[int, int] = {}
    for r in rows:
        hist[r.vocab] = hist.get(r.vocab, 0) + 1
    rec = {
        "n": len(rows),
        "median": median(v), "q1": quantile(v, 0.25), "q3": quantile(v, 0.75),
        "min": min(v) if v else float("nan"), "max": max(v) if v else float("nan"),
        "mean": (sum(v) / len(v)) if v else float("nan"),
        "main_median": median(vm), "main_q1": quantile(vm, 0.25),
        "main_q3": quantile(vm, 0.75),
        "hist": dict(sorted(hist.items())),
        "rho_level": spearman([r.level for r in rows], v),
        "rho_bpm": spearman([r.bpm_main for r in rows], v),
        "rho_nps": spearman([r.nps for r in rows], v),
        "rho_bars": spearman([float(r.n_bars) for r in rows], v),
        "main_rho_level": spearman([r.level for r in rows], vm),
        "main_rho_bpm": spearman([r.bpm_main for r in rows], vm),
        "by_level": {}, "by_bpm": {}, "by_cell": {},
    }
    for lb, _, _ in LEVEL_BANDS:
        sub = [r for r in rows if r.level_band == lb]
        rec["by_level"][lb] = {"n": len(sub), "median": median([float(r.vocab) for r in sub]),
                               "q1": quantile([float(r.vocab) for r in sub], 0.25),
                               "q3": quantile([float(r.vocab) for r in sub], 0.75),
                               "main_median": median([float(r.vocab_main) for r in sub])}
    for bb, _, _ in BPM_BANDS:
        sub = [r for r in rows if r.bpm_band == bb]
        rec["by_bpm"][bb] = {"n": len(sub), "median": median([float(r.vocab) for r in sub]),
                             "q1": quantile([float(r.vocab) for r in sub], 0.25),
                             "q3": quantile([float(r.vocab) for r in sub], 0.75),
                             "main_median": median([float(r.vocab_main) for r in sub])}
    # 控 BPM 的定数偏相关（各 BPM 档内先算再按 n 加权平均）
    rhos, ws = [], []
    for bb, _, _ in BPM_BANDS:
        sub = [r for r in rows if r.bpm_band == bb]
        if len(sub) >= 10:
            rho = spearman([r.level for r in sub], [float(r.vocab) for r in sub])
            if _finite(rho):
                rhos.append(rho)
                ws.append(len(sub))
    rec["rho_level_ctrl_bpm"] = (sum(a * b for a, b in zip(rhos, ws)) / sum(ws)
                                 if rhos else float("nan"))
    rhos, ws = [], []
    for lb, _, _ in LEVEL_BANDS:
        sub = [r for r in rows if r.level_band == lb]
        if len(sub) >= 10:
            rho = spearman([r.bpm_main for r in sub], [float(r.vocab) for r in sub])
            if _finite(rho):
                rhos.append(rho)
                ws.append(len(sub))
    rec["rho_bpm_ctrl_level"] = (sum(a * b for a, b in zip(rhos, ws)) / sum(ws)
                                 if rhos else float("nan"))
    return rec


def pair_stats(rows: Iterable[ChartRow], min_charts: int = 15,
               min_share: float = 0.0) -> list[dict]:
    """配置两两共存 / 互斥（按谱计）。

    ``lift = P(A∧B) / (P(A)·P(B))``；``phi`` 为 2×2 相关系数；
    ``p_b_given_a`` / ``p_a_given_b`` 给条件概率。lift ≫1 = 总一起出现，
    lift ≪1 = 几乎不同谱共存。

    ``min_share > 0`` 时只看**主料集**（小节占比 ≥ 该值的配置）——"整张谱里
    有没有出现过"太宽松（长谱什么都蹭得到），"有没有当主料写"才分得开。
    """
    rows = list(rows)
    n = len(rows)
    sets = [(r.main_configs(min_share) if min_share > 0 else r.config_set)
            for r in rows]
    cfgs = [c for c in all_configs(rows)
            if sum(1 for s in sets if c in s) >= min_charts]
    out: list[dict] = []
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            na = sum(1 for s in sets if a in s)
            nb = sum(1 for s in sets if b in s)
            nab = sum(1 for s in sets if a in s and b in s)
            pa, pb, pab = na / n, nb / n, nab / n
            exp = pa * pb
            lift = (pab / exp) if exp else float("nan")
            n11, n10, n01 = nab, na - nab, nb - nab
            n00 = n - n11 - n10 - n01
            den = math.sqrt(max(1e-12, (n11 + n10) * (n01 + n00) *
                                (n11 + n01) * (n10 + n00)))
            phi = (n11 * n00 - n10 * n01) / den
            out.append({"a": a, "b": b, "n_a": na, "n_b": nb, "n_both": nab,
                        "expected": exp * n, "lift": lift, "phi": phi,
                        "p_b_given_a": (nab / na) if na else float("nan"),
                        "p_a_given_b": (nab / nb) if nb else float("nan")})
    return sorted(out, key=lambda d: d["lift"])


def habitat(rows: Iterable[ChartRow], thresh: float = 0.5,
            min_n: int = MIN_CELL_N) -> dict[str, dict]:
    """每类配置"主要住在哪个定数×BPM 区间"。

    住址 = 出现率 ≥ ``thresh`` 且样本量 ≥ ``min_n`` 的格子；并在这些格子里给
    小节占比中位，用来分"主料"（占比高）还是"点缀"（占比低）。
    """
    rows = list(rows)
    g = grid(rows)
    out: dict[str, dict] = {}
    overall = g["*|*"]["configs"]
    for c in all_configs(rows):
        cells = []
        for lb, _, _ in LEVEL_BANDS:
            for bb, _, _ in BPM_BANDS:
                cell = g[f"{lb}|{bb}"]
                if cell["n"] < min_n:
                    continue
                rec = cell["configs"].get(c) or {}
                if rec.get("rate", 0.0) >= thresh:
                    cells.append({"level_band": lb, "bpm_band": bb,
                                  "n": cell["n"], "rate": rec["rate"],
                                  "share_median": rec["share_median"],
                                  "main_rate": rec["main_rate"]})
        best_cell = None
        best = -1.0
        for lb, _, _ in LEVEL_BANDS:
            for bb, _, _ in BPM_BANDS:
                cell = g[f"{lb}|{bb}"]
                if cell["n"] < min_n:
                    continue
                rec = cell["configs"].get(c) or {}
                if rec.get("rate", 0.0) > best:
                    best = rec.get("rate", 0.0)
                    best_cell = {"level_band": lb, "bpm_band": bb, "n": cell["n"],
                                 "rate": rec.get("rate", 0.0),
                                 "share_median": rec.get("share_median", 0.0)}
        ov = overall.get(c, {})
        out[c] = {
            "overall_rate": ov.get("rate", 0.0),
            "overall_share_median": ov.get("share_median", 0.0),
            "overall_main_rate": ov.get("main_rate", 0.0),
            "cells": sorted(cells, key=lambda d: -d["rate"]),
            "peak_cell": best_cell,
            "role": ("主料" if ov.get("share_median", 0.0) >= MAIN_SHARE else "点缀"),
        }
    return out


def config_sequence(row: ChartRow, fold: bool = True) -> list[dict]:
    """个案读法：按小节顺序的配置序列摘要。

    ``fold=True`` 时把"连续小节、配置集合相同"的折成一段，
    输出 ``[{bars: "12-19", n: 8, configs: [...]}, ...]``。
    """
    items = sorted(row.bar_configs.items())
    if not items:
        return []
    out: list[dict] = []
    cur_lo, cur_hi, cur = items[0][0], items[0][0], tuple(items[0][1])
    for m, cs in items[1:]:
        t = tuple(cs)
        if fold and t == cur and m == cur_hi + 1:
            cur_hi = m
            continue
        out.append({"bars": f"{cur_lo}-{cur_hi}" if cur_hi > cur_lo else str(cur_lo),
                    "n": cur_hi - cur_lo + 1, "configs": list(cur)})
        cur_lo, cur_hi, cur = m, m, t
    out.append({"bars": f"{cur_lo}-{cur_hi}" if cur_hi > cur_lo else str(cur_lo),
                "n": cur_hi - cur_lo + 1, "configs": list(cur)})
    return out


# ---------------------------------------------------------------------------
# 语料驱动
# ---------------------------------------------------------------------------


def build_rows(files: Iterable, detector: Callable | None = None,
               on_error: Callable | None = None) -> list[ChartRow]:
    """从 `corpus.ChartFile` 列表构造每谱一行。"""
    from .simai_parser import parse_chart

    out: list[ChartRow] = []
    for cf in files:
        try:
            res = parse_chart(cf.read(), name=cf.name)
            out.append(chart_row(
                res, chart=cf.name, simai_id=getattr(cf, "simai_id", ""),
                title=getattr(cf, "title", ""),
                difficulty=getattr(cf, "difficulty", ""),
                level=getattr(cf, "internal_level", None) or 0.0,
                detector=detector))
        except Exception as exc:                        # noqa: BLE001
            if on_error:
                on_error(cf, exc)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import csv
    import json
    import sys
    from pathlib import Path

    from . import corpus

    p = argparse.ArgumentParser(
        prog="python -m tools.chart_analysis.chart_matrix",
        description="谱面级「定数 × BPM × 配置」对照矩阵（388 官方 ST 谱）")
    p.add_argument("--chart-dir", default=None)
    p.add_argument("--csv", default=None, help="每谱一行 CSV")
    p.add_argument("--json", default=None, help="分格 / 热图 / 趋势 / 共存 完整结果 JSON")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min-cell", type=int, default=MIN_CELL_N)
    p.add_argument("--habitat-thresh", type=float, default=0.5)
    p.add_argument("--case", action="append", default=[],
                   help="个案谱名（可多次），输出配置序列")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    files = corpus.discover(Path(a.chart_dir)) if a.chart_dir else corpus.discover()
    if a.limit:
        files = files[:a.limit]
    errs: list[str] = []
    rows = build_rows(files, on_error=lambda cf, e: errs.append(f"{cf.name}: {e!r}"))
    for e in errs:
        print(f"[ERR] {e}", file=sys.stderr)
    if not a.quiet:
        print(f"谱面 {len(rows)}/{len(files)}（错 {len(errs)}）  "
              f"配置 {len(all_configs(rows))}")

    if a.csv:
        out = [r.to_row() for r in rows]
        keys: list[str] = []
        for r in out:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(out)
        print(f"写出 {a.csv}（{len(out)} 行）")

    if a.json:
        doc = {
            "n_charts": len(rows), "n_errors": len(errs), "errors": errs,
            "configs": all_configs(rows),
            "level_bands": [b[0] for b in LEVEL_BANDS],
            "bpm_bands": [b[0] for b in BPM_BANDS],
            "grid": grid(rows),
            "heat_rate": heatmap(rows, "rate"),
            "heat_share": heatmap(rows, "share_median"),
            "trend_level": trend_by_level(rows),
            "trend_bpm": trend_by_bpm(rows),
            "trend_level_share": trend_by_level(rows, "share"),
            "trend_bpm_share": trend_by_bpm(rows, "share"),
            "trend_bpm_spm": trend_by_bpm(rows, "seg_per_min"),
            "trend_level_spm": trend_by_level(rows, "seg_per_min"),
            "vocab": vocab_stats(rows),
            "pairs": pair_stats(rows),
            "pairs_main": pair_stats(rows, min_share=0.05),
            "habitat": habitat(rows, a.habitat_thresh, a.min_cell),
        }
        if a.case:
            by_name = {r.chart: r for r in rows}
            doc["cases"] = {
                nm: {"row": by_name[nm].to_row(),
                     "sequence": config_sequence(by_name[nm])}
                for nm in a.case if nm in by_name}
        Path(a.json).write_text(json.dumps(doc, ensure_ascii=False,
                                           default=_json_default), encoding="utf-8")
        print(f"写出 {a.json}")
    return 0


def _json_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(repr(o))


if __name__ == "__main__":
    raise SystemExit(_cli())
