"""结构分段（段落 / 重复段 / 副歌候选）。

对应设计文档 §4 L3 结构层。两条路径：

1. **主线**：`allin1`（all-in-one, ISMIR 2023，MIT）或 `all-in-one-infer`——
   直接输出 10 类功能标签（intro/verse/chorus/bridge/outro…）；
2. **退路**：自研 SSM 方案——bar-synchronous chroma+MFCC → 自相似矩阵 →
   checkerboard novelty 边界 → 段落聚类标签 A/B/C… → 重复最多/最长的段 = 副歌候选。

无论走哪条路，输出都归一成同一份段落列表（起止小节 + 标签 + 重复关系），
并在 `method` 字段记录实际走的路径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 段落最短长度（小节）——音游曲段落一般 4/8 小节起
MIN_SEGMENT_BARS = 4
# 边界吸附：novelty 峰落在 4 小节整倍数附近时吸附过去
SNAP_TO_BARS = 4
SNAP_TOLERANCE = 1


@dataclass
class Segment:
    """一个段落。"""

    start_bar: int              # 含
    end_bar: int                # 含
    label: str                  # SSM 路径为 A/B/C…；allin1 路径为功能标签
    function: str = ""          # 功能猜测（intro/verse/chorus/…）
    is_repeat: bool = False
    repeat_of: list[int] | None = None   # [start_bar, end_bar] of first occurrence
    intensity: float = 0.0
    intensity_tier: str = ""
    primary_stem: str = ""

    @property
    def n_bars(self) -> int:
        return self.end_bar - self.start_bar + 1

    def to_dict(self) -> dict:
        return {
            "start_bar": self.start_bar,
            "end_bar": self.end_bar,
            "n_bars": self.n_bars,
            "label": self.label,
            "function": self.function,
            "is_repeat": self.is_repeat,
            "repeat_of": self.repeat_of,
            "intensity": round(float(self.intensity), 4),
            "intensity_tier": self.intensity_tier,
            "primary_stem": self.primary_stem,
        }


# ---------------- 主线：all-in-one ----------------


def _extract_raw_segments(result) -> list[dict]:
    raw = []
    for seg in getattr(result, "segments", []) or []:
        raw.append({"start": float(seg.start), "end": float(seg.end),
                    "label": str(seg.label)})
    return raw


def _try_allin1(audio_path: Path, grid, stems_dir: Path | None = None,
                work_dir: Path | None = None, device: str = "cpu") -> dict | None:
    """尝试用 all-in-one 做功能标签分段。失败返回 {"ok": False, "reason": ...}。

    优先 `allin1_infer`（openmirlab 的纯 PyTorch 版，可直接吃我们已经跑好的
    Demucs stems，省掉重复分离）；不可用时退回官方 `allin1`。
    """
    reasons: list[str] = []

    # 路径 A：allin1_infer（可复用已有 stems）
    try:
        import allin1_infer  # type: ignore

        t0 = time.time()
        kwargs: dict = dict(device=device, keep_byproducts=False, multiprocess=False)
        if work_dir is not None:
            kwargs["demix_dir"] = str(Path(work_dir) / "demix")
            kwargs["spec_dir"] = str(Path(work_dir) / "spec")
        if stems_dir is not None and all(
            (Path(stems_dir) / f"{n}.wav").exists()
            for n in ("bass", "drums", "other", "vocals")
        ):
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
                    "bpm": getattr(result, "bpm", None), "impl": "allin1_infer"}
        reasons.append("allin1_infer 未返回段落")
    except Exception as exc:
        reasons.append(f"allin1_infer 不可用：{type(exc).__name__}: {exc}")

    # 路径 B：官方 allin1
    try:
        import allin1  # type: ignore

        t0 = time.time()
        result = allin1.analyze(str(audio_path), keep_byproducts=False, device=device)
        if isinstance(result, list):
            result = result[0]
        raw = _extract_raw_segments(result)
        if raw:
            return {"ok": True, "raw_segments": raw, "elapsed_sec": time.time() - t0,
                    "bpm": getattr(result, "bpm", None), "impl": "allin1"}
        reasons.append("allin1 未返回段落")
    except Exception as exc:
        reasons.append(f"allin1 不可用：{type(exc).__name__}: {exc}")

    return {"ok": False, "reason": "；".join(reasons)}


def _segments_from_seconds(raw: list[dict], grid) -> list[Segment]:
    """把秒级段落边界对齐到小节网格。"""
    segs: list[Segment] = []
    for item in raw:
        sb = max(1, grid.bar_of(item["start"] + 1e-6) or 1)
        eb = grid.bar_of(item["end"] - 1e-6)
        eb = min(grid.n_bars, eb if eb >= sb else sb)
        if eb < sb:
            continue
        segs.append(Segment(start_bar=sb, end_bar=eb, label=str(item["label"]),
                            function=str(item["label"])))
    # 合并同标签相邻段 / 去重
    merged: list[Segment] = []
    for s in sorted(segs, key=lambda x: x.start_bar):
        if merged and s.start_bar <= merged[-1].end_bar:
            s.start_bar = merged[-1].end_bar + 1
            if s.start_bar > s.end_bar:
                continue
        if merged and merged[-1].label == s.label and s.start_bar == merged[-1].end_bar + 1:
            merged[-1].end_bar = s.end_bar
            continue
        merged.append(s)
    if merged:
        merged[0].start_bar = 1
        merged[-1].end_bar = grid.n_bars
    return _absorb_tiny(merged, min_bars=2)


def _absorb_tiny(segs: list[Segment], min_bars: int = 2) -> list[Segment]:
    """把过短的碎段（all-in-one 常在首尾吐出 1 小节的 start/end）并进相邻段。"""
    if len(segs) <= 1:
        return segs
    out = list(segs)
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, s in enumerate(out):
            if s.n_bars >= min_bars:
                continue
            if i == 0:
                out[1].start_bar = s.start_bar
            else:
                out[i - 1].end_bar = s.end_bar
            out.pop(i)
            changed = True
            break
    return out


# ---------------- 退路：自研 SSM ----------------


def _bar_sync_features(y: np.ndarray, sr: int, grid) -> tuple[np.ndarray, np.ndarray]:
    """逐小节聚合 chroma 与 MFCC 特征。返回 (chroma[bars,12], mfcc[bars,n])。"""
    import librosa

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=20)[1:]  # 丢掉能量项
    ft = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    n = grid.n_bars
    C = np.zeros((n, chroma.shape[0]))
    M = np.zeros((n, mfcc.shape[0]))
    for bar in range(1, n + 1):
        t0 = grid.bar_start(bar)
        t1 = t0 + grid.bar_duration(bar)
        sel = (ft >= t0) & (ft < t1)
        if not sel.any():
            sel = np.zeros_like(ft, dtype=bool)
            sel[min(len(ft) - 1, max(0, int(np.searchsorted(ft, t0))))] = True
        C[bar - 1] = np.median(chroma[:, sel], axis=1)
        M[bar - 1] = np.median(mfcc[:, sel], axis=1)
    return C, M


def _cosine_ssm(F: np.ndarray) -> np.ndarray:
    X = F - F.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norm
    S = Xn @ Xn.T
    return (S + 1.0) / 2.0


def _checkerboard_kernel(half: int) -> np.ndarray:
    k = np.arange(-half, half + 1)
    X, Y = np.meshgrid(k, k)
    taper = np.exp(-0.5 * (X ** 2 + Y ** 2) / (max(half, 1) / 2.0) ** 2)
    sign = np.sign(X) * np.sign(Y)
    sign[sign == 0] = 1.0
    return taper * sign


def _novelty(S: np.ndarray, half: int = 8) -> np.ndarray:
    n = S.shape[0]
    K = _checkerboard_kernel(half)
    pad = half
    Sp = np.pad(S, pad, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        block = Sp[i: i + 2 * half + 1, i: i + 2 * half + 1]
        nov[i] = float((block * K).sum())
    nov = np.maximum(nov, 0.0)
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def _pick_boundaries(nov: np.ndarray, min_gap: int = MIN_SEGMENT_BARS,
                     target_n: int | None = None) -> list[int]:
    """从 novelty 曲线挑边界（小节索引，0 起），带最小间隔与 4 小节吸附。"""
    n = len(nov)
    order = np.argsort(nov)[::-1]
    chosen: list[int] = []
    if target_n is None:
        target_n = max(4, min(16, n // 8))
    for i in order:
        if nov[i] <= 0.12:
            break
        if all(abs(int(i) - c) >= min_gap for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= target_n:
            break
    # 4 小节吸附
    snapped = []
    for c in chosen:
        r = c % SNAP_TO_BARS
        cand = c - r if r <= SNAP_TOLERANCE else (c + (SNAP_TO_BARS - r)
                                                  if SNAP_TO_BARS - r <= SNAP_TOLERANCE else c)
        cand = max(0, min(n - 1, cand))
        if all(abs(cand - s) >= min_gap for s in snapped):
            snapped.append(cand)
    snapped = sorted(set(snapped))
    if not snapped or snapped[0] != 0:
        snapped = [0] + snapped
    return snapped


def _label_segments(F: np.ndarray, bounds: list[int], n_bars: int,
                    n_clusters: int | None = None) -> list[tuple[int, int, str]]:
    """段落聚类打标签 A/B/C…（按首次出现顺序）。

    用段落平均特征做层次聚类（average linkage + cosine），簇数按段落数自适应，
    再把相邻同标签段合并——避免"整首歌都是 B"这种退化标注。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    spans = []
    for i, b in enumerate(bounds):
        e = (bounds[i + 1] - 1) if i + 1 < len(bounds) else (n_bars - 1)
        if e >= b:
            spans.append((b, e))
    if not spans:
        return [(0, n_bars - 1, "A")]
    if len(spans) == 1:
        return [(spans[0][0], spans[0][1], "A")]

    V = np.vstack([F[b:e + 1].mean(axis=0) for b, e in spans])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    if n_clusters is None:
        n_clusters = int(min(max(3, round(len(spans) / 2.5)), 6, len(spans)))
    Z = linkage(pdist(V, metric="cosine"), method="average")
    cl = fcluster(Z, t=n_clusters, criterion="maxclust")

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    order: dict[int, str] = {}
    labeled: list[tuple[int, int, str]] = []
    for (b, e), c in zip(spans, cl):
        if c not in order:
            order[c] = alphabet[len(order) % 26]
        labeled.append((b, e, order[c]))

    # 合并相邻同标签段
    merged: list[list] = []
    for b, e, lab in labeled:
        if merged and merged[-1][2] == lab and b == merged[-1][1] + 1:
            merged[-1][1] = e
        else:
            merged.append([b, e, lab])
    return [(b, e, lab) for b, e, lab in merged]


def _spans_to_segments(spans: list[tuple[int, int, str]]) -> list[Segment]:
    segs: list[Segment] = []
    first_seen: dict[str, list[int]] = {}
    for b, e, lab in spans:
        sb, eb = b + 1, e + 1
        if lab in first_seen:
            segs.append(Segment(sb, eb, lab, is_repeat=True, repeat_of=first_seen[lab]))
        else:
            first_seen[lab] = [sb, eb]
            segs.append(Segment(sb, eb, lab))
    return segs


def _ssm_structure(y: np.ndarray, sr: int, grid) -> dict:
    """自研 SSM 分段（退路）。

    边界数与簇数都做**自适应重试**：先按默认参数分一版，若体检不通过
    （段数太少 / 某段独占全曲 >35%），逐步加密边界与簇数重试，
    最后取"最长段占比最小"的那一版。
    """
    t0 = time.time()
    C, M = _bar_sync_features(y, sr, grid)
    F = np.hstack([
        C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12),
        (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-12) * 0.25,
    ])
    S = 0.5 * _cosine_ssm(C) + 0.5 * _cosine_ssm(M)
    half = max(3, min(12, grid.n_bars // 12))
    nov = _novelty(S, half=half)

    attempts: list[tuple[float, int, int, list[Segment]]] = []
    chosen: list[Segment] | None = None
    tried: list[str] = []
    for target_n in (max(6, grid.n_bars // 8), max(8, grid.n_bars // 6),
                     max(12, grid.n_bars // 4)):
        bounds = _pick_boundaries(nov, min_gap=MIN_SEGMENT_BARS, target_n=target_n)
        for k in (None, 5, 6, 8):
            spans = _label_segments(F, bounds, grid.n_bars, n_clusters=k)
            segs = _spans_to_segments(spans)
            share = max(s.n_bars for s in segs) / max(grid.n_bars, 1)
            attempts.append((share, target_n, k or 0, segs))
            ok, why = _segmentation_is_usable(segs, grid.n_bars)
            tried.append(f"target_n={target_n},k={k}:{'ok' if ok else why}")
            if ok:
                chosen = segs
                break
        if chosen:
            break
    if chosen is None:
        attempts.sort(key=lambda a: a[0])
        chosen = attempts[0][3]

    return {
        "segments": chosen,
        "method": "ssm-fallback",
        "elapsed_sec": time.time() - t0,
        "ssm": S,
        "novelty": nov,
        "ssm_attempts": tried,
    }


# ---------------- 统一入口 ----------------


def analyze_structure(audio_path: Path, y: np.ndarray, sr: int, grid,
                      prefer_allin1: bool = True,
                      stems_dir: Path | None = None,
                      work_dir: Path | None = None,
                      device: str = "cpu") -> dict:
    """结构分段主入口。返回 dict（segments / method / notes / elapsed_sec）。"""
    notes: list[str] = []
    allin1_segs: list[Segment] | None = None
    allin1_elapsed = 0.0
    allin1_impl = ""
    if prefer_allin1:
        res = _try_allin1(Path(audio_path), grid, stems_dir=stems_dir,
                          work_dir=work_dir, device=device)
        if res and res.get("ok"):
            allin1_segs = _mark_repeats_by_label(
                _segments_from_seconds(res["raw_segments"], grid))
            allin1_elapsed = res.get("elapsed_sec", 0.0)
            allin1_impl = res.get("impl", "allin1")
            ok, why = _segmentation_is_usable(allin1_segs, grid.n_bars)
            if ok:
                return {
                    "segments": allin1_segs,
                    "method": allin1_impl,
                    "elapsed_sec": allin1_elapsed,
                    "notes": notes,
                    "ssm": None,
                    "novelty": None,
                    "alt_segments": None,
                }
            notes.append(f"{allin1_impl} 分段退化（{why}）→ 改用 SSM 退路")
        elif res:
            notes.append(str(res.get("reason", "allin1 不可用")))

    out = _ssm_structure(y, sr, grid)
    out["elapsed_sec"] = out.get("elapsed_sec", 0.0) + allin1_elapsed
    notes.append("SSM 退路：chroma+MFCC 自相似矩阵 → checkerboard novelty 边界 → 层次聚类标签")
    ok, why = _segmentation_is_usable(out["segments"], grid.n_bars)
    if not ok:
        notes.append(f"⚠️ SSM 分段体检也未通过（{why}），段落划分仅供参考")
    out["notes"] = notes
    out["alt_segments"] = [s.to_dict() for s in allin1_segs] if allin1_segs else None
    return out


def _segmentation_is_usable(segs: list[Segment], n_bars: int,
                            min_segments: int = 5,
                            max_share: float = 0.35) -> tuple[bool, str]:
    """分段可用性体检：段数太少、或某一段独占全曲太大比例，都视为退化。

    对制谱没用的分段（例如把 63/100 小节标成一个 `solo`）必须被拦下来，
    否则 L4 规划层拿到的"段落"毫无信息量。
    """
    if len(segs) < min_segments:
        return False, f"只有 {len(segs)} 段（<{min_segments}）"
    longest = max(s.n_bars for s in segs)
    if longest > max_share * n_bars:
        return False, f"最长段占 {longest}/{n_bars} 小节（>{max_share:.0%}）"
    return True, ""


def _mark_repeats_by_label(segs: list[Segment]) -> list[Segment]:
    seen: dict[str, list[int]] = {}
    for s in segs:
        if s.label in seen:
            s.is_repeat = True
            s.repeat_of = seen[s.label]
        else:
            seen[s.label] = [s.start_bar, s.end_bar]
    return segs


# ---------------- 副歌 / 功能猜测 ----------------

_FUNCTION_KEYWORDS = {
    "intro", "verse", "chorus", "bridge", "outro", "inst", "solo",
    "break", "start", "end", "pre-chorus",
}


def guess_functions(segments: list[Segment], bar_intensity: np.ndarray,
                    method: str) -> list[Segment]:
    """给段落补功能猜测。

    - allin1 路径：标签本身就是功能，直接沿用；
    - SSM 路径：首段=intro、末段=outro；重复次数最多且平均强度最高的标签=chorus；
      其余按强度分 verse / bridge。
    """
    if not segments:
        return segments
    for s in segments:
        s.intensity = float(np.mean(bar_intensity[s.start_bar - 1: s.end_bar])) \
            if s.end_bar <= len(bar_intensity) else 0.0

    if str(method).startswith("allin1"):
        for s in segments:
            if not s.function:
                s.function = s.label
        return segments

    # 各标签的总小节数、出现次数、平均强度
    stats: dict[str, dict] = {}
    for s in segments:
        d = stats.setdefault(s.label, {"bars": 0, "count": 0, "inten": []})
        d["bars"] += s.n_bars
        d["count"] += 1
        d["inten"].append(s.intensity)
    for lab, d in stats.items():
        d["mean_inten"] = float(np.mean(d["inten"]))

    # 副歌候选：出现次数 ≥2 优先，再按 (平均强度, 总小节数) 排序
    ranked = sorted(
        stats.items(),
        key=lambda kv: (kv[1]["count"] >= 2, kv[1]["mean_inten"], kv[1]["bars"]),
        reverse=True,
    )
    chorus_label = ranked[0][0] if ranked else None

    for i, s in enumerate(segments):
        if i == 0:
            s.function = "intro"
        elif i == len(segments) - 1:
            s.function = "outro"
        elif s.label == chorus_label:
            s.function = "chorus"
        elif s.intensity >= 0.55:
            s.function = "pre_chorus/build"
        elif s.intensity <= 0.30:
            s.function = "interlude/rest"
        else:
            s.function = "verse"
    return segments


def tier_of(value: float) -> str:
    """强度 → 档位（谱面端按此映射密度，知识 001/003）。"""
    if value >= 0.80:
        return "peak"
    if value >= 0.60:
        return "high"
    if value >= 0.35:
        return "mid"
    return "low"
