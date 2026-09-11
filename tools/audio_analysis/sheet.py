"""输出层：analysis.json / song-sheet.md / plot.png。

`song-sheet.md` 是给 LLM（与人）直接消费的"歌曲分析单"——谱师拿到 mp3 后
第一步要知道的全部事实：BPM/first 是否对得上、段落怎么分、哪里是高潮、
每小节该踩哪条轨、每小节的最小分音与网格串。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .structure import tier_of

# song-sheet 逐小节表展示的网格串顺序
PATTERN_COLUMNS = ("kick", "snare", "hihat", "vocals", "bass", "other")
PATTERN_HEADER = ("kick", "snare", "hat", "vocal", "bass", "other")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_json_default(x) if isinstance(x, (np.floating, np.integer)) else x
                for x in o.tolist()]
    raise TypeError(f"不可序列化：{type(o)}")


def write_analysis_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               default=_json_default), encoding="utf-8")
    return path


def _primary_stem_of(segment, bar_onsets: dict[str, np.ndarray],
                     vocal_act: np.ndarray) -> str:
    """段落主活动 stem：onset 密度最高者；人声活动 >50% 时优先记人声。"""
    s, e = segment.start_bar - 1, segment.end_bar
    scores = {}
    for name, counts in bar_onsets.items():
        if name in ("kick", "snare", "hihat"):
            continue
        scores[name] = float(np.mean(counts[s:e])) if e <= len(counts) else 0.0
    if not scores:
        return ""
    # 各轨 onset 数量级不同 → 用各自全曲均值归一
    norm = {}
    for name, counts in bar_onsets.items():
        if name in ("kick", "snare", "hihat"):
            continue
        m = float(np.mean(counts)) or 1.0
        norm[name] = scores[name] / m
    best = max(norm, key=norm.get)
    va = float(np.mean(vocal_act[s:e])) if e <= len(vocal_act) else 0.0
    if va > 0.5 and norm.get("vocals", 0) > 0.6:
        return "vocals"
    return best


def build_song_sheet_md(payload: dict, segments, bar_rows: list[dict]) -> str:
    """生成中文分析单 Markdown。"""
    g = payload["grid"]
    oc = payload["offset_check"]
    song = payload["song"]
    inten = payload["intensity"]
    st = payload["structure"]

    L: list[str] = []
    L.append(f"# 歌曲分析单 — {song['name']}")
    L.append("")
    L.append("> 由 `tools/audio_analysis` 自动生成。BPM 与 first 为**用户给定值**，"
             "管线不做节拍追踪，只对 offset 做一致性校验（不覆盖）。")
    L.append("")
    L.append("## 1. 基本信息")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|----|----|")
    L.append(f"| 音频 | `{song['audio']}` |")
    L.append(f"| 时长 | {song['duration_sec']:.2f} s |")
    L.append(f"| BPM | {g['bpm']}" + (f"（变速点：{g['bpm_changes']}）|" if g['bpm_changes'] else " |"))
    L.append(f"| `&first` | {g['first']} s |")
    L.append(f"| 每小节拍数 | {g['beats_per_bar']} |")
    L.append(f"| 小节数 | {g['n_bars']} |")
    L.append(f"| offset 校验 | {payload['offset_verdict']} |")
    if oc.get("available"):
        L.append(f"| 自动最佳 first | {oc['best_first']:.4f} s（差 {oc['delta_ms']:+.1f} ms，"
                 f"置信 z={oc['confidence_z']:.2f}）|")
    L.append(f"| 分离模型 | {payload['stems']['model']}（设备 {payload['stems']['device']}，"
             f"{payload['stems']['elapsed_sec']:.1f}s）|")
    L.append(f"| 结构分段路径 | {st['method']} |")
    L.append(f"| 高潮小节 | 第 {inten['climax_bar']} 小节"
             f"（次峰：{'、'.join(str(x) for x in inten['climax_peaks'])}）|")
    L.append("")
    if st.get("notes"):
        L.append("**结构分段备注**：" + "；".join(st["notes"]))
        L.append("")

    L.append("## 2. 段落表")
    L.append("")
    L.append("| 段 | 小节范围 | 小节数 | 标签 | 功能 | 强度 | 档位 | 主要活动 stem | 重复关系 |")
    L.append("|----|----------|--------|------|------|------|------|----------------|----------|")
    for i, s in enumerate(segments, 1):
        rep = "—"
        if s.is_repeat and s.repeat_of:
            rep = f"重复于 {s.repeat_of[0]}–{s.repeat_of[1]} 小节"
        L.append(f"| {i} | {s.start_bar}–{s.end_bar} | {s.n_bars} | {s.label} | "
                 f"{s.function} | {s.intensity:.2f} | {s.intensity_tier} | "
                 f"{s.primary_stem or '—'} | {rep} |")
    L.append("")

    L.append("## 3. 逐小节表")
    L.append("")
    L.append("> 网格串：`.` 空、`x` 有 onset、`X` 强 onset；串长 = 该轨该小节选中的分音"
             "（每小节等分数）。「最小分音」= 该小节所有轨中最细的那个。"
             "kick/snare/hat 为**频带启发式**分件，仅供参考。")
    L.append("")
    head = "| 小节 | 强度 | 人声活动 | 鼓onset | 最小分音 | " + " | ".join(PATTERN_HEADER) + " |"
    sep = "|------|------|----------|---------|----------|" + "|".join(["------"] * len(PATTERN_HEADER)) + "|"
    L.append(head)
    L.append(sep)
    for r in bar_rows:
        pats = " | ".join(f"`{r['patterns'].get(k, '')}`" for k in PATTERN_COLUMNS)
        L.append(f"| {r['bar']} | {r['intensity']:.2f} | {r['vocal_activity']:.2f} | "
                 f"{r['drum_onsets']} | {r['finest_division']} | {pats} |")
    L.append("")

    L.append("## 4. 量化误差与已知限制")
    L.append("")
    L.append("| stem | onset 数 | 平均误差 | P95 误差 | 无法解释的小节 | 分音分布 |")
    L.append("|------|----------|----------|----------|----------------|----------|")
    for name, s in payload["quantize_stats"].items():
        L.append(f"| {name} | {s['n_onsets']} | {s['mean_abs_err_ms']} ms | "
                 f"{s['p95_abs_err_ms']} ms | {s['unresolved_bars']} | "
                 f"{s['division_histogram']} |")
    L.append("")
    L.append("- 「无法解释的小节」= 在 {4,8,12,16,24,32} 里找不到能在容差内覆盖全部 onset 的分音"
             "（多为装饰音、鼓刷、人声连奏或分离残留）；这些小节的网格串按 32 分强行拟合，不可当真。")
    L.append("- 人声活动度基于 Demucs vocals 轨的 RMS 阈值，合成器/和声泄漏会导致误判为有人声。")
    L.append("")
    return "\n".join(L)


def plot_overview(path: Path, grid, intensity_res, segments,
                  bar_onsets: dict[str, np.ndarray], vocal_act: np.ndarray,
                  title: str = "") -> Path:
    """强度曲线 + 段落边界 + 各 stem 逐小节活动度叠加图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 曲名常含日文/中文，给 matplotlib 配一组 CJK 字体回退，否则标题全是豆腐块
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [n for n in ("Hiragino Sans", "Hiragino Sans GB", "PingFang SC",
                       "Arial Unicode MS", "Heiti TC", "STHeiti",
                       "Noto Sans CJK SC", "Songti SC") if n in available]
    plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = grid.n_bars
    x = np.arange(1, n + 1)
    fig, axes = plt.subplots(3, 1, figsize=(max(12, n * 0.09), 9), sharex=True)

    ax = axes[0]
    ax.plot(x, intensity_res.bar_intensity, lw=1.8, color="#c0392b", label="intensity (smoothed)")
    ax.plot(x, intensity_res.bar_raw / (intensity_res.bar_raw.max() + 1e-12),
            lw=0.7, color="#7f8c8d", alpha=0.6, label="raw")
    shade = ("#ffffff", "#eef2f5")
    for i, s in enumerate(segments):
        ax.axvspan(s.start_bar, s.end_bar + 1, color=shade[i % 2], zorder=0)
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.8, alpha=0.6)
        # 交错两行放标签，避免短段落的文字互相压住
        ax.text(s.start_bar + 0.4, 1.005 + 0.055 * (i % 2),
                f"{s.label}/{s.function}\n{s.start_bar}-{s.end_bar}",
                fontsize=6, va="bottom", color="#2c3e50")
    if intensity_res.climax_bar:
        ax.axvline(intensity_res.climax_bar, color="#e67e22", lw=2.0, alpha=0.9,
                   label=f"climax bar {intensity_res.climax_bar}")
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("intensity")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title or "intensity / structure / stem activity")

    ax = axes[1]
    colors = {"drums": "#2980b9", "bass": "#8e44ad", "other": "#27ae60", "vocals": "#d35400"}
    for name in ("drums", "bass", "other", "vocals"):
        c = bar_onsets.get(name)
        if c is None:
            continue
        m = float(np.max(c)) or 1.0
        ax.plot(x, np.asarray(c, dtype=float) / m, lw=1.1,
                color=colors.get(name, "#555"), label=f"{name} onsets/bar (norm)")
    ax.plot(x, vocal_act, lw=1.4, color="#000000", alpha=0.5, label="vocal activity")
    for s in segments:
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.6, alpha=0.4)
    ax.set_ylabel("stem activity")
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    ax = axes[2]
    for name, color in (("kick", "#16a085"), ("snare", "#c0392b"), ("hihat", "#f39c12")):
        c = bar_onsets.get(name)
        if c is None:
            continue
        ax.plot(x, np.asarray(c, dtype=float), lw=1.0, color=color, label=f"{name}/bar")
    for s in segments:
        ax.axvline(s.start_bar, color="#2c3e50", ls="--", lw=0.6, alpha=0.4)
    ax.set_ylabel("drum parts (heuristic)")
    ax.set_xlabel("bar")
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
