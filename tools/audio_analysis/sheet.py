"""输出层：`song_sheet.md`（给 LLM）/ `song_analysis.json`（给程序）/ `plot.png`（给人）。

**双格式同源**（v2 §5.3）：md 与 json 由同一份 payload 渲染，禁止两边手改。

song sheet 的四条硬约束（v2 §5.2）：
- (a) 网格分音**跟随该小节的量化结论**，每行显式写 `div=`；
- (b) **每行自包含**：`小节 | 段落 | 强度 | div | 4 轨串`，不靠"上一行说过"；
- (c) 字符集与制谱直觉一致：`X` 重音 / `x` onset / `-` 延音 / `.` 空；
- (d) **轨数上限 4**（drum / vocal / bass / hook）——轨越多越诱导采密；
- (e) Header 显式声明"网格是候选池不是谱面"，并给目标 note 总数区间（知识 004）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .structure import LABELS_JA
from .tracks import TRACK_ORDER

TRACK_HEADER = {"drum": "drum ", "vocal": "vocal", "bass": "bass ", "hook": "hook "}

# 旧文件名（v0.1）→ 新文件名，保留兼容
LEGACY_NAMES = {"song_sheet.md": "song-sheet.md", "song_analysis.json": "analysis.json"}


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化：{type(o)}")


def write_analysis_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")
    return path


# ---------------- song sheet ----------------


def _display_width(text: str) -> int:
    """等宽终端里的显示宽度（CJK / 全角字符算 2 列）。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
               for ch in str(text))


def _pad_display(text: str, width: int) -> str:
    """按显示宽度右侧补空格——逐小节表的四条轨必须能纵向对齐。"""
    return str(text) + " " * max(0, width - _display_width(text))


def _header(payload: dict) -> list[str]:
    g, oc, song = payload["grid"], payload["offset_check"], payload["song"]
    inten, q = payload["intensity"], payload["quantize"]
    tgt = payload["target"]
    L: list[str] = []
    L.append(f"# SONG SHEET — {song['name']}")
    L.append("")
    L.append("## 0. Header")
    L.append("")
    L.append("```")
    L.append(f"bpm: {g['bpm']}" + ("（含变速点，实验性）" if g["bpm_changes"] else "（恒定）")
             + f"   offset(&first): {g['first']} s   bars: 1..{g['n_bars']}"
             + f"   time-sig: {g['beats_per_bar']}/4")
    L.append(f"duration: {song['duration_sec']:.2f} s"
             + (f"   container start_time: {song['container_start_time_sec']*1000:.1f} ms"
                if song.get("container_start_time_sec") else ""))
    if oc.get("available"):
        if oc.get("offbeat_ambiguous"):
            state = "反拍歧义→不可判定"
        elif abs(oc["delta_ms"]) <= 15:
            state = "OK"
        elif abs(oc["delta_ms"]) <= 30:
            state = "疑似 MP3 解码延迟"
        else:
            state = "需复核"
        L.append(f"offset_check: φ* = {oc['delta_ms']:+.1f} ms "
                 f"（{state}，方式 {oc.get('method')}，z={oc['confidence_z']:.2f}，"
                 f"搜索窗 ±{oc.get('search_beats', 0.4):.2f} 拍）")
    L.append(f"structure: {payload['structure']['method']}   "
             f"**音频能量高潮** bar: {inten.get('audio_climax_bar', inten['climax_bar'])}"
             f"（次峰 {'、'.join(str(x) for x in inten['climax_peaks'])}）"
             f"　⚠️ 这是**音乐最激烈处，不是谱面最密处**（8 首实测中位误差 33 小节）")
    L.append("分音使用统计: " + ("  ".join(
        f"{{{k}}}={v:.0%}" for k, v in q["division_share"].items()) or "—"))
    res_bars, all_bars = q["resolved_bars"], q["bars_with_onsets"]
    L.append(f"  其中**真的被选中**（τ 内解释全部 onset）: {res_bars}/{all_bars} 小节 → "
             + ("  ".join(f"{{{k}}}×{v}" for k, v in
                          q["division_histogram_resolved"].items()) or "—"))
    L.append(f"  其余 {all_bars - res_bars} 小节是「找不到合法分音、被压到上限」，"
             f"网格串只是近似，不是分音结论")
    L.append(f"未能落格的 onset: {q['unquantized_onsets']}/{q['onsets_considered']}"
             f" = {q['unquantized_ratio']:.1%}"
             f"（{{32}} 红线拦下 {q['fine_blocked_bars']} 小节；三连小节 {q['triplet_bars']}）")
    if tgt.get("level") is not None:
        L.append(f"target: 定数 {tgt['level']} → note 总数 {tgt['total_p10']:.0f}–"
                 f"{tgt['total_p90']:.0f}（均值 {tgt['total_mean']:.0f}，知识 004）")
        L.append(f"密度锚（主）: NPS {tgt.get('nps')} × 每小节秒数 → "
                 f"{tgt['notes_per_bar']:.2f} note/小节（{tgt.get('nps_source', '')}）")
        L.append(f"密度锚（对照）: 定数→note/小节 "
                 f"{tgt.get('notes_per_bar_level_anchor', float('nan')):.2f}"
                 f"　⚠️ 该口径在 BPM ≥ 200 的曲子上实测系统性高估 3.1–3.7 note/小节，"
                 f"故 v0.3 改以 NPS 为主锚")
    else:
        L.append(f"target: 未给 --level → 不给 note 总数区间；"
                 f"密度用全库量级 {tgt['notes_per_bar']:.2f} note/小节（无 NPS 锚）")
    fl = tgt.get("density_floor", {})
    if fl:
        L.append(f"密度地板: 段落 {fl.get('section')} / 小节 {fl.get('bar')}"
                 f"（n=8 配对标定的段落最优是 {fl.get('section_calibrated_optimum')}，"
                 f"样本太小未采纳为默认；可用 --section-floor 自试）")
    cap = tgt.get("structural_cap", {})
    if cap.get("applied"):
        L.append(f"结构封顶: intro/outro 建议密度 ≤ {cap.get('cap_notes_per_bar')} note/小节"
                 f"（= 全曲建议均值 × {cap.get('cap_ratio')}），已封顶 "
                 f"{cap.get('bars_capped')} 小节 —— 前奏/尾奏的密度由**结构**决定，"
                 f"不跟能量曲线走")
    L.append("```")
    L.append("")
    L.append("> ⚠️ **下面的网格是「候选池」，不是谱面。** 按知识 005「采音要简——删到不能"
             "再删」从候选中**筛选**：能听到的音全踩上是新人谱师最典型的通病，留白本身"
             "就是表达。目标 note 总数区间见上方 `target`。")
    L.append(">")
    L.append("> 📏 **候选池参考量级（8 首官方音频×官方谱实测，n=8）**：官方谱只采用候选池的"
             "**约 55%**（precision 0.546；候选池对官方 note 的 recall 0.821）——"
             "**删到不能再删**。逐曲 precision 0.234（最稀的谱）–0.695（最密的谱），"
             "谱越密用掉的候选越多。")
    L.append(">")
    L.append("> 字符集（仅这四个）：`X` 重音（drum=kick / vocal=有音高且强 / 其他=强 onset）、"
             "`x` 普通 onset、`-` 延音持续（人声 VAD 有声但无新 onset）、`.` 空。")
    L.append(">")
    L.append("> 四条轨：`drum`（鼓，X=kick）/ `vocal`（人声）/ `bass` / `hook`（other stem："
             "吉他/合成器/riff）。hihat 不给网格串，只在逐小节表给计数列"
             "（密集段里它与 kick 的串常常完全一样，铺出来是噪声）。")
    L.append("")
    return L


def _sections(payload: dict, segments) -> list[str]:
    L: list[str] = []
    L.append("## 1. Sections")
    L.append("")
    L.append("| 小节范围 | 段落（日式(英文)） | 强度 | 建议密度档 | 建议 note/小节 | "
             "**骨架** | **点缀** | 估计占比 | chorus# | repeat_of | upgrade | rest |")
    L.append("|----------|-------------------|------|-----------|----------------|"
             "------|------|---------|---------|-----------|---------|------|")
    for s in segments:
        rep = f"{s.repeat_of[0]}–{s.repeat_of[1]}" if s.repeat_of else "—"
        acc = "＞".join(s.accent_stems) if s.accent_stems else "—"
        share = "、".join(f"{k} {v:.2f}" for k, v in (s.accent_share or {}).items())
        L.append(
            f"| {s.start_bar}–{s.end_bar} | {s.label_ja} | {s.intensity:.2f}"
            f"（{s.intensity_tier}） | {s.suggested_division} | "
            f"{s.suggested_notes_per_bar:.1f} | {s.skeleton_stem or '—'} | "
            f"{acc} | {share or '—'} | {s.chorus_index or '—'} | {rep} | "
            f"{'✅' if s.upgrade else '—'} | {'✅' if s.rest else '—'} |")
    L.append("")
    L.append("> **骨架 / 点缀 / 依据**（v0.3 模型，取代 v0.2 的单值「主踩音轨」）："
             "官方谱的实测形态是 **鼓骨架 + 大量非鼓填充**——8 首配对标定里 drums 命中率"
             "0.607、**36.2% 的官方 note 不落在鼓上**、只落鼓的仅 28.1%、只落人声的仅 3.2%，"
             "另有 **18.6% 什么 stem 都不落**（谱师自由发挥/装饰，占比里记作 `free`）。"
             "**没有任何一段是「只踩一条轨」**，所以别把「骨架」读成「整段只踩这条」。")
    L.append("")
    L.append("**逐段备注与依据**")
    L.append("")
    for s in segments:
        bits = [b for b in (s.notes or []) if b]
        ev = "；".join(s.evidence) if s.evidence else ""
        L.append(f"- **{s.start_bar}–{s.end_bar} {s.label_ja}**"
                 + (f"　段落判定证据：{ev}" if ev else "")
                 + ("　" + "　".join(bits) if bits else ""))
        for line in (s.plan_evidence or []):
            L.append(f"    - 踩音依据（纯音频特征）：{line}")
    L.append("")
    L.append("> 「建议 note/小节」= `(floor + (1−floor)·强度) × 锚点`，锚点 v0.3 起走 "
             "**NPS**（定数 → NPS → × 每小节秒数），段落尺度 floor 见 Header。"
             "**精度只到档位**：8 首配对实测逐小节 MAE 2.07–4.69（均值 3.4 note/小节，"
             "相对误差 30–45%）—— 它是 **±3 note 的粗档**，不是逐小节目标值。")
    L.append("")
    return L


def _bars(payload: dict, bar_rows: list[dict], segments) -> list[str]:
    seg_of_bar: dict[int, str] = {}
    for s in segments:
        for b in range(s.start_bar, s.end_bar + 1):
            seg_of_bar[b] = s.label_ja
    L: list[str] = []
    L.append("## 2. Bars（逐小节；每行自包含）")
    L.append("")
    L.append("> 每行格式：`小节 | 段落 | I=强度 | div=该小节分音 | 四轨网格串`。"
             "串长 = `div`（每小节等分数，对应 simai 的 `{div}`）。"
             "**同一小节四条轨共用一个 div**，可直接纵向对齐读。")
    L.append("")
    L.append("```")
    for r in bar_rows:
        bar = r["bar"]
        label = _pad_display(seg_of_bar.get(bar, "—"), 22)
        head = (f"bar {bar:>3} | {label} | I={r['intensity']:.2f}"
                f" | div={r['division']:<2} | ")
        pad = " " * _display_width(head)
        for i, t in enumerate(TRACK_ORDER):
            prefix = head if i == 0 else pad
            L.append(f"{prefix}{TRACK_HEADER[t]} {r['patterns'].get(t, '')}")
        extra_pad = pad
        extra = []
        if r.get("n_onset_hihat"):
            extra.append(f"hihat×{int(r['n_onset_hihat'])}")
        if r.get("suggested_notes"):
            extra.append(f"建议≈{r['suggested_notes']:.1f} note")
        if not r.get("resolved", True):
            extra.append(f"⚠️未落格 {r.get('n_unquantized', 0)} 个 onset")
        if extra:
            L.append(f"{extra_pad}note: " + "; ".join(extra))
    L.append("```")
    L.append("")
    return L


def _limits(payload: dict) -> list[str]:
    q = payload["quantize"]
    L: list[str] = []
    L.append("## 3. 量化与已知限制")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|----|----|")
    L.append(f"| 分音直方图（全部有 onset 的小节） | {q['division_histogram']} |")
    L.append(f"| 分音直方图（**真的解释得通**的小节） | "
             f"{q['division_histogram_resolved']}（{q['resolved_bars']}/"
             f"{q['bars_with_onsets']}） |")
    L.append(f"| 未落格 onset 比例 | {q['unquantized_ratio']:.1%}"
             f"（{q['unquantized_onsets']}/{q['onsets_considered']}） |")
    L.append(f"| 未能解释的小节 | {q['unresolved_bars']} |")
    L.append(f"| {{32}} 红线拦下的小节 | {q['fine_blocked_bars']} |")
    L.append(f"| 三连（{{12}}/{{24}}）小节 | {q['triplet_bars']} |")
    for name, s in q["per_track"].items():
        L.append(f"| {name} 量化误差（平均/P95/最大） | {s['mean_abs_err_ms']} / "
                 f"{s['p95_abs_err_ms']} / {s['max_abs_err_ms']} ms |")
    L.append("")
    for w in payload.get("warnings", []):
        L.append(f"- {w}")
    L.append("- 鼓件（kick/snare/hihat）是**频带能量启发式**，不是鼓转录：底鼓 vs 低音 tom、"
             "军鼓 vs 拍手、hihat vs 镲片/齿音泄漏都会混淆，只当倾向性提示。")
    L.append("- 人声 VAD 基于 vocals stem 的 RMS + 谐波性，合成器/和声泄漏会误判为有人声。"
             "**「副歌全踩人声」的实测结论就建立在这条不可靠的人声 onset 上**，"
             "故规则 4 已降级为条件触发 + 存疑（待人工听审）。")
    L.append("- 强度融合权重与高潮票权重仍是调研 v2 的**初值**：2026-09-11 用 8 首官方音频"
             "×官方谱做过配对标定，LOSO 交叉验证提升仅 +0.027（5/8 折、p≈0.36），"
             "**不满足「明显优于」→ 维持初值**。两条待验假设：`voiced`→0、`flux` 0.10→0.25"
             "（拿到 ≥20 首配对数据再验）。")
    L.append("- 音频强度与官方谱密度只有**中等相关**（8 首实测：逐小节 Spearman 0.454、"
             "逐段 0.523）。够定「这一段大概多密」，**不够定「高潮在哪一小节」**。")
    L.append("")
    return L


def build_song_sheet_md(payload: dict, segments, bar_rows: list[dict]) -> str:
    L = _header(payload)
    L += _sections(payload, segments)
    L += _bars(payload, bar_rows, segments)
    L += _limits(payload)
    return "\n".join(L)


# ---------------- plot ----------------


def plot_overview(path: Path, grid, intensity_res, segments,
                  bar_features: dict[str, np.ndarray],
                  bar_division: dict[int, int] | None = None,
                  title: str = "") -> Path:
    """强度曲线 + 段落边界 + stem 活动度 + 逐小节 div（人工复核用）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in ("Hiragino Sans", "Hiragino Sans GB", "PingFang SC",
                       "Arial Unicode MS", "Heiti TC", "STHeiti",
                       "Noto Sans CJK SC", "Songti SC") if n in available]
    plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = grid.n_bars
    x = np.arange(1, n + 1)
    fig, axes = plt.subplots(4, 1, figsize=(max(12, n * 0.09), 11), sharex=True)

    ax = axes[0]
    # v0.2：raw 与 smoothed **同基准**（都已归一到 [0,1]，平滑不改变电平）
    ax.plot(x, intensity_res.bar_raw, lw=0.7, color="#7f8c8d", alpha=0.65,
            label="I raw (normalized, unsmoothed)")
    ax.plot(x, intensity_res.bar_intensity, lw=1.8, color="#c0392b",
            label="I smoothed")
    shade = ("#ffffff", "#eef2f5")
    for i, s in enumerate(segments):
        ax.axvspan(s.start_bar, s.end_bar + 1, color=shade[i % 2], zorder=0)
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.8, alpha=0.6)
        ja = LABELS_JA.get(s.function, s.function)
        ax.text(s.start_bar + 0.4, 1.005 + 0.06 * (i % 2),
                f"{ja}({s.function})\n{s.start_bar}-{s.end_bar}",
                fontsize=6, va="bottom", color="#2c3e50")
    if intensity_res.climax_bar:
        ax.axvline(intensity_res.climax_bar, color="#e67e22", lw=2.0, alpha=0.9,
                   label=f"climax bar {intensity_res.climax_bar}")
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("intensity")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title or "intensity / structure / stem activity / division")

    ax = axes[1]
    colors = {"drums": "#2980b9", "bass": "#8e44ad", "other": "#27ae60",
              "vocals": "#d35400"}
    for stem in ("drums", "bass", "other", "vocals"):
        c = bar_features.get(f"n_onset_{stem}")
        if c is None:
            continue
        m = float(np.max(c)) or 1.0
        ax.plot(x, np.asarray(c, dtype=float) / m, lw=1.1, color=colors[stem],
                label=f"{stem} onsets/bar (norm)")
    ax.plot(x, bar_features.get("voiced_ratio", np.zeros(n)), lw=1.5,
            color="#000000", alpha=0.55, label="voiced_ratio")
    for s in segments:
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.6, alpha=0.4)
    ax.set_ylabel("stem activity")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = axes[2]
    for name, color in (("kick", "#16a085"), ("snare", "#c0392b"),
                        ("hihat", "#f39c12")):
        c = bar_features.get(f"n_onset_{name}")
        if c is None:
            continue
        ax.plot(x, np.asarray(c, dtype=float), lw=1.0, color=color, label=f"{name}/bar")
    for s in segments:
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.6, alpha=0.4)
    ax.set_ylabel("drum parts (heuristic)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax = axes[3]
    if bar_division:
        divs = np.array([bar_division.get(b, 0) for b in range(1, n + 1)], dtype=float)
        ax.step(x, divs, where="mid", lw=1.2, color="#34495e", label="bar division")
        ax.set_yticks([4, 8, 12, 16, 24, 32])
    for s in segments:
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.6, alpha=0.4)
    ax.set_ylabel("division {d}")
    ax.set_xlabel("bar")
    ax.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
