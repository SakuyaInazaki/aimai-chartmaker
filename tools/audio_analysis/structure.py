"""结构分段（v0.2：三路边界投票 + 日式标签词表 + 派生规则）。

## 相对 v0.1 的改动（主会话验收的 D 项）

v0.1 走"all-in-one 主线 / SSM 退路"二选一，TransientTears 上把 13–16 标成
chorus（4 小节）、17–40 标成 pre_chorus/build（24 小节）——**明显错**：
build 不可能占全曲四分之一，4 小节也撑不起一段副歌。

v0.2 改成：

1. **边界三路投票**（v2 §2.5「边界：三者取交集 + 吸附到最近下拍」）：
   all-in-one-infer / SSM checkerboard novelty / 重复段变化点，各自提名边界，
   吸附到最近小节线（优先 4 小节乐句线），**≥2 路同意才采纳**；
   路数不够时按 novelty 强度补齐到最少段数。
2. **标签用 v2 §2.4 的日式词表双写**（`サビ(chorus)`），而不是 all-in-one 的
   8 类原始标签——制谱经验知识库与 MMFC 教程都用中/日语境，`落ちサビ` 这类词
   本身就携带"休息段"的制谱含义。
3. **`pre_chorus` 按 v2 §2.5 派生**（all-in-one 根本没有这个标签）：
   位于 chorus 前 4–8 小节 ∧ 强度单调上升 ∧（kick↓ ∨ hihat↑ ∨ 高频↑）。
   前一段过长时**只切出最后 4–8 小节**当 Bメロ，剩下的还给 Aメロ。
4. **chorus 需 ≥2 路证据**（重复段 / 强度高 / 人声活动），器乐曲改走 `drop` 能量票。
5. 输出 `chorus_index` / `repeat_of` / `upgrade` / `rest`（MMFC 5.4-6/5.4-7 需要）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIN_SEGMENT_BARS = 4
PHRASE_BARS = 4          # 乐句长度：边界优先吸附到 4 小节线
SNAP_TOLERANCE = 1       # 吸附容差（小节）
BOUNDARY_MATCH = 1       # 两路边界相差 ≤ 此值视为同一个边界

# v2 §2.4 标签词表（日式 + 英文双写）
LABELS_JA = {
    "intro": "イントロ",
    "verse": "Aメロ",
    "pre_chorus": "Bメロ",
    "chorus": "サビ",
    "quiet_chorus": "落ちサビ",
    "final_chorus": "ラスサビ",
    "bridge": "Cメロ",
    "interlude": "間奏",
    "drop": "ドロップ",
    "outro": "アウトロ",
}
CHORUS_FAMILY = ("chorus", "final_chorus", "quiet_chorus")


def label_ja(function: str) -> str:
    """功能标签 → `日式(英文)` 双写。"""
    ja = LABELS_JA.get(function)
    return f"{ja}({function})" if ja else str(function)


@dataclass
class Segment:
    """一个段落。"""

    start_bar: int              # 含
    end_bar: int                # 含
    label: str = ""             # 重复分组标签 A/B/C…
    function: str = ""          # 本项目标准标签（intro/verse/pre_chorus/…）
    is_repeat: bool = False
    repeat_of: list[int] | None = None
    intensity: float = 0.0
    intensity_tier: str = ""
    primary_stem: str = ""
    secondary_stem: str = ""
    chorus_index: int | None = None
    upgrade: bool = False
    rest: bool = False
    voiced_ratio: float = 0.0
    density_norm: float = 0.0
    suggested_notes_per_bar: float = 0.0
    suggested_division: str = ""
    evidence: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_bars(self) -> int:
        return self.end_bar - self.start_bar + 1

    @property
    def label_ja(self) -> str:
        return label_ja(self.function)

    def to_dict(self) -> dict:
        return {
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "n_bars": self.n_bars,
            "cluster": self.label,
            "function": self.function,
            "label_ja": LABELS_JA.get(self.function, ""),
            "label_display": self.label_ja,
            "is_repeat": self.is_repeat,
            "repeat_of": self.repeat_of,
            "chorus_index": self.chorus_index,
            "upgrade": self.upgrade,
            "rest": self.rest,
            "intensity": round(float(self.intensity), 4),
            "intensity_tier": self.intensity_tier,
            "voiced_ratio": round(float(self.voiced_ratio), 4),
            "density_norm": round(float(self.density_norm), 4),
            "suggested_notes_per_bar": round(float(self.suggested_notes_per_bar), 2),
            "suggested_division": self.suggested_division,
            "primary_stem": self.primary_stem,
            "secondary_stem": self.secondary_stem,
            "evidence": self.evidence,
            "notes": self.notes,
        }


# ---------------- 主线：all-in-one ----------------


def _extract_raw_segments(result) -> list[dict]:
    return [{"start": float(s.start), "end": float(s.end), "label": str(s.label)}
            for s in (getattr(result, "segments", []) or [])]


def _try_allin1(audio_path: Path, stems_dir: Path | None = None,
                work_dir: Path | None = None, device: str = "cpu") -> dict:
    """尝试 all-in-one（优先 `allin1_infer`，可复用已跑好的 Demucs stems）。"""
    reasons: list[str] = []
    try:
        import allin1_infer  # type: ignore

        t0 = time.time()
        kwargs: dict = dict(device=device, keep_byproducts=False, multiprocess=False)
        if work_dir is not None:
            kwargs["demix_dir"] = str(Path(work_dir) / "demix")
            kwargs["spec_dir"] = str(Path(work_dir) / "spec")
        if stems_dir is not None and all((Path(stems_dir) / f"{n}.wav").exists()
                                         for n in ("bass", "drums", "other", "vocals")):
            si = allin1_infer.create_stems_input_from_directory(
                str(stems_dir), identifier=Path(audio_path).stem)
            result = allin1_infer.analyze(stems_input=si, **kwargs)
        else:
            result = allin1_infer.analyze(str(audio_path), **kwargs)
        if isinstance(result, list):
            result = result[0]
        raw = _extract_raw_segments(result)
        if raw:
            return {"ok": True, "raw_segments": raw, "elapsed_sec": time.time() - t0,
                    "impl": "allin1_infer"}
        reasons.append("allin1_infer 未返回段落")
    except Exception as exc:
        reasons.append(f"allin1_infer 不可用：{type(exc).__name__}: {exc}")

    try:
        import allin1  # type: ignore

        t0 = time.time()
        result = allin1.analyze(str(audio_path), keep_byproducts=False, device=device)
        if isinstance(result, list):
            result = result[0]
        raw = _extract_raw_segments(result)
        if raw:
            return {"ok": True, "raw_segments": raw, "elapsed_sec": time.time() - t0,
                    "impl": "allin1"}
        reasons.append("allin1 未返回段落")
    except Exception as exc:
        reasons.append(f"allin1 不可用：{type(exc).__name__}: {exc}")
    return {"ok": False, "reason": "；".join(reasons)}


# ---------------- SSM 特征 ----------------


def _bar_sync_features(y: np.ndarray, sr: int, grid) -> tuple[np.ndarray, np.ndarray]:
    """逐小节聚合 chroma 与 MFCC。"""
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=20)[1:]
    ft = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    n = grid.n_bars
    C = np.zeros((n, chroma.shape[0]))
    M = np.zeros((n, mfcc.shape[0]))
    for bar in range(1, n + 1):
        t0 = grid.bar_start(bar)
        sel = (ft >= t0) & (ft < t0 + grid.bar_duration(bar))
        if not sel.any():
            sel = np.zeros_like(ft, dtype=bool)
            sel[min(len(ft) - 1, max(0, int(np.searchsorted(ft, t0))))] = True
        C[bar - 1] = np.median(chroma[:, sel], axis=1)
        M[bar - 1] = np.median(mfcc[:, sel], axis=1)
    return C, M


def _cosine_ssm(F: np.ndarray) -> np.ndarray:
    X = F - F.mean(axis=0, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return (Xn @ Xn.T + 1.0) / 2.0


def _checkerboard_kernel(half: int) -> np.ndarray:
    k = np.arange(-half, half + 1)
    X, Y = np.meshgrid(k, k)
    taper = np.exp(-0.5 * (X ** 2 + Y ** 2) / (max(half, 1) / 2.0) ** 2)
    sign = np.sign(X) * np.sign(Y)
    sign[sign == 0] = 1.0
    return taper * sign


def novelty_curve(S: np.ndarray, half: int = 8) -> np.ndarray:
    """checkerboard novelty（越大越像段落边界）。"""
    n = S.shape[0]
    K = _checkerboard_kernel(half)
    Sp = np.pad(S, half, mode="edge")
    nov = np.array([float((Sp[i:i + 2 * half + 1, i:i + 2 * half + 1] * K).sum())
                    for i in range(n)])
    nov = np.maximum(nov, 0.0)
    return nov / nov.max() if nov.max() > 0 else nov


def repeat_change_points(S: np.ndarray, min_lag: int = PHRASE_BARS,
                         sim_threshold: float = 0.55) -> list[int]:
    """重复段变化点：**最佳匹配 lag 发生跳变**的小节（第三路边界提名）。

    RefraiD（Goto 2006）在 RWC-Pop（80 首日语 J-pop）上 100 首中 80 首完全正确，
    F=0.938——纯重复性方法在日语流行乐上有一手实证支持（v2 §2.3）。这里用它的
    现代最小等价物：对每个小节找"与之最相似的历史小节"，该 lag 跳变即结构变化。
    """
    n = S.shape[0]
    best_lag = np.zeros(n, dtype=int)
    best_sim = np.zeros(n, dtype=float)
    for b in range(n):
        hi = b - min_lag
        if hi < 0:
            continue
        sims = S[b, :hi + 1]
        j = int(np.argmax(sims))
        best_lag[b] = b - j
        best_sim[b] = float(sims[j])
    out: list[int] = []
    for b in range(1, n):
        if best_sim[b] < sim_threshold and best_sim[b - 1] < sim_threshold:
            continue
        if abs(int(best_lag[b]) - int(best_lag[b - 1])) > 2:
            out.append(b)
    return out


def _snap_bar(idx: int, n_bars: int) -> int:
    """吸附到最近的 4 小节乐句线（容差 1 小节），否则保持原位。"""
    r = idx % PHRASE_BARS
    if r <= SNAP_TOLERANCE:
        cand = idx - r
    elif PHRASE_BARS - r <= SNAP_TOLERANCE:
        cand = idx + (PHRASE_BARS - r)
    else:
        cand = idx
    return int(max(0, min(n_bars - 1, cand)))


def vote_boundaries(sources: dict[str, list[int]], n_bars: int,
                    novelty: np.ndarray | None = None,
                    min_segments: int = 5,
                    min_gap: int = MIN_SEGMENT_BARS) -> tuple[list[int], dict]:
    """三路边界投票（0 起小节索引）。

    1. 每路提名先吸附到最近 4 小节乐句线；
    2. 相差 ≤1 小节的提名归为同一候选，票数 = **不同来源**的个数；
    3. ≥2 票直接采纳；不足 `min_segments` 段时按 novelty 强度补齐单票候选；
    4. 强制最小段长。
    """
    snapped: dict[str, list[int]] = {
        k: sorted({_snap_bar(int(b), n_bars) for b in v if 0 <= int(b) < n_bars})
        for k, v in sources.items()
    }
    cands: list[dict] = []
    for src, bars in snapped.items():
        for b in bars:
            hit = next((c for c in cands if abs(c["bar"] - b) <= BOUNDARY_MATCH), None)
            if hit is None:
                cands.append({"bar": b, "sources": {src}})
            else:
                hit["sources"].add(src)
                hit["bar"] = min(hit["bar"], b) if b % PHRASE_BARS == 0 else hit["bar"]
    for c in cands:
        c["votes"] = len(c["sources"])
        c["novelty"] = float(novelty[c["bar"]]) if novelty is not None and \
            c["bar"] < len(novelty) else 0.0
    cands.sort(key=lambda c: c["bar"])

    def _accept(pool: list[dict], chosen: list[int]) -> list[int]:
        for c in pool:
            if all(abs(c["bar"] - x) >= min_gap for x in chosen):
                chosen.append(c["bar"])
        return sorted(chosen)

    chosen = _accept(sorted([c for c in cands if c["votes"] >= 2],
                            key=lambda c: (-c["votes"], -c["novelty"])), [0])
    detail = {"multi_vote": len(chosen) - 1, "topped_up": 0}
    if len(chosen) < min_segments:
        singles = sorted([c for c in cands if c["votes"] < 2],
                         key=lambda c: -c["novelty"])
        before = len(chosen)
        for c in singles:
            if len(chosen) >= min_segments:
                break
            if all(abs(c["bar"] - x) >= min_gap for x in chosen):
                chosen.append(c["bar"])
        chosen = sorted(chosen)
        detail["topped_up"] = len(chosen) - before
    detail["candidates"] = [{"bar": int(c["bar"] + 1), "votes": int(c["votes"]),
                             "sources": sorted(c["sources"])} for c in cands]
    return sorted(set(chosen)), detail


def cluster_spans(F: np.ndarray, bounds: list[int], n_bars: int,
                  n_clusters: int | None = None) -> list[tuple[int, int, str]]:
    """段落聚类打重复分组标签 A/B/C…（按首次出现顺序）。"""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    spans = []
    for i, b in enumerate(bounds):
        e = (bounds[i + 1] - 1) if i + 1 < len(bounds) else (n_bars - 1)
        if e >= b:
            spans.append((b, e))
    if len(spans) <= 1:
        return [(0, n_bars - 1, "A")]
    V = np.vstack([F[b:e + 1].mean(axis=0) for b, e in spans])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    if n_clusters is None:
        n_clusters = int(min(max(3, round(len(spans) / 2.0)), 6, len(spans)))
    Z = linkage(pdist(V, metric="cosine"), method="average")
    cl = fcluster(Z, t=n_clusters, criterion="maxclust")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    order: dict[int, str] = {}
    out = []
    for (b, e), c in zip(spans, cl):
        if c not in order:
            order[c] = alphabet[len(order) % 26]
        out.append((b, e, order[c]))
    return out


def spans_to_segments(spans: list[tuple[int, int, str]]) -> list[Segment]:
    segs: list[Segment] = []
    first: dict[str, list[int]] = {}
    for b, e, lab in spans:
        sb, eb = b + 1, e + 1
        s = Segment(sb, eb, label=lab)
        if lab in first:
            s.is_repeat = True
            s.repeat_of = list(first[lab])
        else:
            first[lab] = [sb, eb]
        segs.append(s)
    return segs


# ---------------- 统一入口 ----------------


def analyze_structure(audio_path: Path, y: np.ndarray, sr: int, grid,
                      prefer_allin1: bool = True,
                      stems_dir: Path | None = None,
                      work_dir: Path | None = None,
                      device: str = "cpu") -> dict:
    """三路边界投票 → 段落列表（功能标签由 `assign_functions` 再补）。"""
    t0 = time.time()
    notes: list[str] = []
    C, M = _bar_sync_features(y, sr, grid)
    F = np.hstack([
        C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12),
        (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-12) * 0.25,
    ])
    S = 0.5 * _cosine_ssm(C) + 0.5 * _cosine_ssm(M)
    half = max(3, min(12, grid.n_bars // 12))
    nov = novelty_curve(S, half=half)

    sources: dict[str, list[int]] = {}
    allin1_elapsed = 0.0
    allin1_raw = None
    if prefer_allin1:
        res = _try_allin1(Path(audio_path), stems_dir=stems_dir, work_dir=work_dir,
                          device=device)
        if res.get("ok"):
            allin1_elapsed = res.get("elapsed_sec", 0.0)
            allin1_raw = res["raw_segments"]
            bars = []
            for item in allin1_raw:
                b = grid.bar_of(item["start"] + 1e-6)
                if b >= 1:
                    bars.append(b - 1)
            sources["allin1"] = bars
            notes.append(f"边界来源 1/3：{res['impl']}（{len(set(bars))} 个提名）")
        else:
            notes.append("all-in-one 不可用：" + str(res.get("reason", "")))

    # 第二路：SSM novelty 峰
    nov_bounds = _novelty_peaks(nov, min_gap=MIN_SEGMENT_BARS,
                                target_n=max(6, grid.n_bars // 8))
    sources["novelty"] = nov_bounds
    notes.append(f"边界来源 2/3：SSM checkerboard novelty（{len(nov_bounds)} 个提名）")

    # 第三路：重复段变化点
    rep_bounds = repeat_change_points(S)
    sources["repeat"] = rep_bounds
    notes.append(f"边界来源 3/3：重复段变化点（{len(rep_bounds)} 个提名）")

    bounds, detail = vote_boundaries(sources, grid.n_bars, novelty=nov)
    notes.append(f"投票结果：{detail['multi_vote']} 个边界获 ≥2 路同意"
                 + (f"，另按 novelty 补 {detail['topped_up']} 个（否则段数不足）"
                    if detail["topped_up"] else ""))

    spans = cluster_spans(F, bounds, grid.n_bars)
    segments = spans_to_segments(spans)

    return {
        "segments": segments,
        "method": "vote(allin1+novelty+repeat)" if "allin1" in sources
        else "vote(novelty+repeat)",
        "elapsed_sec": time.time() - t0,
        "allin1_elapsed_sec": allin1_elapsed,
        "notes": notes,
        "boundary_vote": detail,
        "allin1_raw": allin1_raw,
        "ssm": S,
        "novelty": nov,
        "features": F,
    }


def _novelty_peaks(nov: np.ndarray, min_gap: int = MIN_SEGMENT_BARS,
                   target_n: int | None = None, floor: float = 0.12) -> list[int]:
    n = len(nov)
    if target_n is None:
        target_n = max(4, min(16, n // 8))
    chosen: list[int] = []
    for i in np.argsort(nov)[::-1]:
        if nov[i] <= floor or len(chosen) >= target_n:
            break
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
    return sorted(chosen)


# ---------------- 功能标签派生 ----------------


def _seg_mean(arr, s: Segment) -> float:
    a = np.asarray(arr, dtype=float)
    lo, hi = s.start_bar - 1, min(s.end_bar, len(a))
    return float(np.mean(a[lo:hi])) if hi > lo else 0.0


def _rising(arr: np.ndarray, s: Segment) -> bool:
    """段内强度是否单调上升（线性斜率 > 0 且 ≥60% 的相邻差非负）。"""
    a = np.asarray(arr, dtype=float)[s.start_bar - 1: s.end_bar]
    if a.size < 3:
        return False
    slope = float(np.polyfit(np.arange(a.size), a, 1)[0])
    if slope <= 0:
        return False
    d = np.diff(a)
    return float(np.mean(d >= -1e-9)) >= 0.6


def _build_spectral_evidence(bar_features: dict[str, np.ndarray], s: Segment
                             ) -> tuple[bool, str]:
    """build 的频谱证据：kick 密度下降 ∨ hihat 密度上升 ∨ 高频占比上升。"""
    half = max(1, s.n_bars // 2)
    lo, hi = s.start_bar - 1, s.end_bar

    def halves(key):
        a = np.asarray(bar_features.get(key, np.zeros(0)), dtype=float)
        if a.size < hi:
            return None, None
        seg = a[lo:hi]
        return float(np.mean(seg[:half])), float(np.mean(seg[-half:]))

    k0, k1 = halves("n_onset_kick")
    if k0 is not None and k1 is not None and k1 < k0 * 0.75:
        return True, "kick 密度下降"
    h0, h1 = halves("n_onset_hihat")
    if h0 is not None and h1 is not None and h1 > h0 * 1.25:
        return True, "hihat 密度上升"
    f0, f1 = halves("hf_ratio")
    if f0 is not None and f1 is not None and f1 > f0 * 1.15:
        return True, "高频能量上升(riser)"
    return False, ""


def assign_functions(segments: list[Segment], bar_intensity: np.ndarray,
                     bar_features: dict[str, np.ndarray], grid,
                     instrumental: bool = False) -> tuple[list[Segment], list[str]]:
    """把重复分组 + 强度 + 人声活动翻成 v2 §2.4 的标准标签。"""
    notes: list[str] = []
    if not segments:
        return segments, notes
    voiced = np.asarray(bar_features.get("voiced_ratio", np.zeros(grid.n_bars)),
                        dtype=float)
    for s in segments:
        s.intensity = _seg_mean(bar_intensity, s)
        s.voiced_ratio = _seg_mean(voiced, s)

    # 分位数取**逐小节**强度而不是段落均值：段落只有 5–12 个，分位数会被
    # 段数整除关系锁死在某个具体段上（例如 6 段时 P40 恰好等于第 3 小的那段）。
    V = np.array([s.voiced_ratio for s in segments])
    p40, p60, p70, p75 = (float(np.percentile(np.asarray(bar_intensity, dtype=float), q))
                          for q in (40, 60, 70, 75))
    group_count: dict[str, int] = {}
    for s in segments:
        group_count[s.label] = group_count.get(s.label, 0) + 1

    # --- 首尾 ---
    segments[0].function = "intro"
    segments[0].evidence.append("全曲第一段")
    segments[-1].function = "outro"
    segments[-1].evidence.append("全曲最后一段")

    if instrumental:
        # 器乐曲（BOF/东方/hardcore/EDM 系）没有"副歌"概念，只有 build → drop
        # （v2 §2.3 保留意见 b）。此时人声票失效，标签只能沿强度轴分档。
        notes.append(f"全曲 voiced_ratio 低（器乐向）→ 不判 chorus，改走能量票："
                     f"I≥P75({p75:.2f}) = ドロップ、≥P60({p60:.2f}) = Aメロ(主 riff)、"
                     f"其余 = 間奏")
        for s in segments[1:-1]:
            if s.intensity >= p75:
                s.function = "drop"
                s.evidence.append(f"器乐曲能量票：强度 {s.intensity:.2f} ≥ P75 {p75:.2f}")
            elif s.intensity >= p60:
                s.function = "verse"
                s.evidence.append(f"器乐曲：强度 {s.intensity:.2f} ∈ [P60,P75) → 主 riff 段")
            else:
                s.function = "interlude"
                s.evidence.append(f"器乐曲：强度 {s.intensity:.2f} < P60 {p60:.2f}")
    else:
        # --- chorus：需 ≥2 路证据（重复段 / 强度高 / 人声活动）---
        # 阈值都取**相对分位**：人声曲整曲 voiced 都高时，绝对阈值 0.40 会让
        # 几乎每一段都拿到"人声活动"票，副歌就泛滥了（实测 金魚鉢 9 段里 5 段被判副歌）。
        v_hi = max(0.45, float(np.percentile(V, 60)))
        i_hi = max(p70, 0.45)
        cands: list[tuple[Segment, list[str]]] = []
        for s in segments[1:-1]:
            ev: list[str] = []
            if group_count.get(s.label, 0) >= 2:
                ev.append(f"重复段（分组 {s.label} 出现 {group_count[s.label]} 次）")
            if s.intensity >= i_hi:
                ev.append(f"强度高（{s.intensity:.2f} ≥ P70 {p70:.2f}）")
            if s.voiced_ratio >= v_hi:
                ev.append(f"人声活动（voiced {s.voiced_ratio:.2f} ≥ {v_hi:.2f}）")
            if len(ev) >= 2:
                cands.append((s, ev))
            else:
                s.evidence.extend(ev)
                s.notes.append(f"chorus 证据只有 {len(ev)} 路（需 ≥2）")
        # 全局护栏：副歌不可能占掉全曲一半以上的段落
        cap = max(1, int(round(0.45 * len(segments))))
        if len(cands) > cap:
            cands.sort(key=lambda kv: kv[0].intensity, reverse=True)
            for s, ev in cands[cap:]:
                s.evidence.extend(ev)
                s.notes.append(f"chorus 证据够，但副歌候选 {len(cands)} 段 > 全曲段数的 45%"
                               f"（上限 {cap}）→ 按强度排序落选")
            cands = cands[:cap]
        for s, ev in cands:
            s.function = "chorus"
            s.evidence.extend(ev)

    # --- pre_chorus 派生（v2 §2.5）---
    _derive_pre_chorus(segments, bar_intensity, bar_features, notes)

    # --- 其余段落 ---
    for s in segments:
        if s.function:
            continue
        # 顺序要紧：**有人声就是 Aメロ**（v2 §4.4 规则 2：verse 的判据是
        # voiced_ratio ∈ [0.3,0.7] 且人声长音为主），不能因为强度偏低就先判成间奏。
        if s.voiced_ratio >= 0.30:
            s.function = "verse"
        elif s.voiced_ratio < 0.25 or s.intensity <= p40:
            s.function = "interlude"
        else:
            s.function = "bridge"

    # --- chorus_index / repeat_of / upgrade / final_chorus / quiet_chorus ---
    chorus_like = [s for s in segments if s.function in ("chorus", "drop")]
    if chorus_like:
        ci = [s.intensity for s in chorus_like]
        # 先挑落ちサビ（副歌家族里明显更弱的那个 = 减压段），再编号
        if len(ci) >= 3:
            p40c = float(np.percentile(ci, 40))
            for s in chorus_like:
                if s.function == "chorus" and s.intensity <= p40c:
                    s.function = "quiet_chorus"
                    s.rest = True
                    s.evidence.append(
                        f"副歌家族但强度落在副歌内部 P40 以下（{s.intensity:.2f} ≤ "
                        f"{p40c:.2f}）→ 落ちサビ（减压段）")
        # 编号与 upgrade 只给"正经副歌"，落ちサビ 不算"第 N 次副歌升级"
        core = [s for s in segments if s.function in ("chorus", "drop")]
        for i, s in enumerate(core, 1):
            s.chorus_index = i
            if i >= 2:
                s.upgrade = True
                s.notes.append("第 ≥2 次副歌：踩音与首次相同，强度靠**配置升级**"
                               "（知识 002 / MMFC 5.4-7），不靠加密")
        if core:
            last = core[-1]
            if last.function == "chorus" and last.intensity >= max(
                    s.intensity for s in core) - 1e-9:
                last.function = "final_chorus"
                last.evidence.append("最后一个 chorus 且强度最高 → ラスサビ")
            elif last.function == "chorus":
                notes.append(f"最后一个 chorus（{last.start_bar}–{last.end_bar}）"
                             f"不是强度最高的 chorus → 不标 ラスサビ（v2 §3.2 口径）")

    # repeat_of 指向同分组首次出现
    first_seen: dict[str, list[int]] = {}
    for s in segments:
        key = s.label
        if key in first_seen:
            s.is_repeat = True
            s.repeat_of = list(first_seen[key])
        else:
            first_seen[key] = [s.start_bar, s.end_bar]
            s.is_repeat = False
            s.repeat_of = None

    # --- rest：两个高强度段之间且 I < P40 ---
    for i in range(1, len(segments) - 1):
        s = segments[i]
        if s.intensity < p40 and segments[i - 1].intensity >= p60 \
                and segments[i + 1].intensity >= p60:
            s.rest = True
            s.notes.append("休息段：夹在两个高强度段之间且强度 < P40（MMFC 5.4-6），"
                           "密度降一档")
    return segments, notes


def _derive_pre_chorus(segments: list[Segment], bar_intensity: np.ndarray,
                       bar_features: dict[str, np.ndarray],
                       notes: list[str]) -> None:
    """`pre_chorus` 派生规则（v2 §2.5）——all-in-one 根本没有这个标签。

    条件：位于 chorus 起点前 4–8 小节 ∧ 强度单调上升 ∧
    （kick 密度↓ ∨ hihat 密度↑ ∨ 高频能量↑）。
    前一段长于 8 小节时**只切出最后 8 小节**当 Bメロ，其余还给前一段——
    这是 v0.1「pre_chorus/build 占 24 小节」的直接修法。
    """
    i = 0
    while i < len(segments):
        s = segments[i]
        if s.function not in ("chorus", "final_chorus", "drop") or i == 0:
            i += 1
            continue
        prev = segments[i - 1]
        if prev.function in ("chorus", "final_chorus", "quiet_chorus", "drop",
                             "pre_chorus", "intro"):
            i += 1
            continue
        take = min(8, prev.n_bars)
        if take < 4:
            i += 1
            continue
        cand = Segment(prev.end_bar - take + 1, prev.end_bar, label=prev.label)
        cand.intensity = _seg_mean(bar_intensity, cand)
        cand.voiced_ratio = _seg_mean(bar_features.get("voiced_ratio",
                                                       np.zeros(len(bar_intensity))), cand)
        if not _rising(bar_intensity, cand):
            i += 1
            continue
        ok, why = _build_spectral_evidence(bar_features, cand)
        if not ok:
            i += 1
            continue
        cand.function = "pre_chorus"
        cand.evidence = [f"位于 chorus 前 {take} 小节", "强度单调上升", why]
        cand.notes.append("MMFC 5.4-3：build 末尾 1–2 小节切回人声给副歌铺垫")
        if cand.start_bar <= prev.start_bar:
            # 整段都是 Bメロ
            prev.function = "pre_chorus"
            prev.evidence = cand.evidence
            prev.notes.extend(cand.notes)
        else:
            prev.end_bar = cand.start_bar - 1
            segments.insert(i, cand)
            i += 1
        i += 1
    return None


def tier_of(value: float) -> str:
    """强度 → 档位（谱面端按此映射密度，知识 001/003）。"""
    if value >= 0.80:
        return "peak"
    if value >= 0.60:
        return "high"
    if value >= 0.35:
        return "mid"
    return "low"


def suggest_division(tier: str) -> str:
    """强度档 → 建议密度档（simai 分音口径，给 LLM 的粗基调）。"""
    return {"peak": "{16}", "high": "{16}", "mid": "{8}", "low": "{8}"}.get(tier, "{8}")
