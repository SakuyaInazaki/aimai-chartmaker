"""切轨建议：段落类型 × 逐小节特征 → 主踩/副踩音轨（调研 v2 §4.4 规则表）。

v2 §4.4 把 MMFC《谱面创作基础学》5.4 的 8 条谱师判断逐条翻成可计算判据；
本模块就是那张表的直译实现。

| # | 段落 | 主踩 | 副踩 |
|---|------|------|------|
| 1 | `intro` | `argmax_s share_s`，**逐 2–4 小节重算** | — |
| 2 | `verse` | `other` 或 `drums` | 小节尾混 `vocal` |
| 3 | `pre_chorus` | 段首 `drums`/`other` | **最后 1–2 小节切 `vocal`** |
| 4 | `chorus` | `vocal` | kick 处叠 each/双押 |
| 5 | `interlude` | vocal 采样（有人声采样时）否则 `other` | 强度靠位移不靠密度 |
| 6 | `rest` | **次响音轨**（share 第二名） | 密度降一档 |
| 7 | `final_chorus` / 第 N 次副歌 | **与 `repeat_of` 同轨** | `upgrade` 置位 |
| 8 | `outro` | **与 `intro` 同轨** | 收尾可摆 `{24}` 位移 |

通用规则：
- **降级**：主轨 `n_onset_s < 2` 或 `grid_fit_s < 0.5` → 退回 `drums` 骨架；
- **反新手护栏**（知识 002/005）：全曲 `primary_stem` 只有 1 个取值且曲长 > 90 s → 警告
  （这正是 MMFC 批评的"哪个最响就一直踩哪个"；MMFC 5.6 的《金星》是已知例外，
  所以是**警告不是硬错误**）。
"""

from __future__ import annotations

import numpy as np

# 逻辑轨名（与 tracks.TRACK_ORDER 一致）→ stem
TRACK_OF_STEM = {"drums": "drum", "vocals": "vocal", "bass": "bass", "other": "hook"}
STEM_OF_TRACK = {v: k for k, v in TRACK_OF_STEM.items()}

MIN_ONSETS = 2
MIN_GRID_FIT = 0.5
SINGLE_TRACK_WARN_SEC = 90.0


def _seg_slice(arr, s) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[s.start_bar - 1: min(s.end_bar, len(a))]


def _loudest(bar_features: dict[str, np.ndarray], s, exclude=()) -> list[str]:
    """段内按能量占比排序的 stem 列表（降序）。"""
    scores = {}
    for stem in ("drums", "bass", "other", "vocals"):
        if stem in exclude:
            continue
        arr = bar_features.get(f"share_{stem}")
        if arr is None:
            continue
        seg = _seg_slice(arr, s)
        scores[stem] = float(np.mean(seg)) if seg.size else 0.0
    return sorted(scores, key=scores.get, reverse=True)


def _usable(bar_features: dict[str, np.ndarray], s, stem: str) -> tuple[bool, str]:
    """降级判据：onset 太少或不落格的轨不能当骨架。"""
    n = _seg_slice(bar_features.get(f"n_onset_{stem}", np.zeros(0)), s)
    # 优先用"对齐到该小节实际分音"的落格率；没有时退回 v2 原始的 {8} 口径
    fit = _seg_slice(bar_features.get(f"grid_fit_bar_{stem}",
                                      bar_features.get(f"grid_fit_{stem}", np.zeros(0))), s)
    if n.size and float(np.mean(n)) < MIN_ONSETS:
        return False, f"{stem} 段内 onset 均值 {float(np.mean(n)):.1f} < {MIN_ONSETS}"
    if fit.size and float(np.mean(fit)) < MIN_GRID_FIT:
        return False, f"{stem} 的落格率 {float(np.mean(fit)):.2f} < {MIN_GRID_FIT}"
    return True, ""


def _intro_windows(bar_features: dict[str, np.ndarray], s, win: int = 4) -> list[dict]:
    """intro/outro 的"哪个响踩哪个"：逐 2–4 小节重算最响轨。"""
    out = []
    b = s.start_bar
    while b <= s.end_bar:
        e = min(s.end_bar, b + win - 1)
        sub = type(s)(start_bar=b, end_bar=e)
        ranked = _loudest(bar_features, sub)
        out.append({"bars": [b, e], "stem": ranked[0] if ranked else ""})
        b = e + 1
    return out


def plan_stems(segments, bar_features: dict[str, np.ndarray], grid,
               duration_sec: float = 0.0) -> tuple[list, list[str]]:
    """给每段填 `primary_stem` / `secondary_stem` / 切轨备注。返回 (segments, warnings)。"""
    warnings: list[str] = []
    by_span: dict[tuple[int, int], dict] = {}
    intro_stem = ""

    for s in segments:
        fn = s.function
        ranked = _loudest(bar_features, s)
        primary, secondary, note = "", "", ""

        if fn == "intro":
            wins = _intro_windows(bar_features, s)
            primary = wins[0]["stem"] if wins else (ranked[0] if ranked else "")
            intro_stem = primary
            note = ("规则 1（MMFC 5.4-1）：前奏「哪个响踩哪个」，逐 2–4 小节重算 → "
                    + "；".join(f"{w['bars'][0]}–{w['bars'][1]}:{w['stem']}" for w in wins))
        elif fn == "outro":
            primary = intro_stem or (ranked[0] if ranked else "")
            note = "规则 8（MMFC 5.4-8）：与 intro 同轨前后呼应；收尾可摆 {24} 位移交互"
        elif fn == "verse":
            primary = "other" if "other" in ranked[:2] else (ranked[0] if ranked else "")
            if primary != "other" and "drums" in ranked[:2]:
                primary = "drums"
            secondary = "vocals"
            note = ("规则 2（MMFC 5.4-2）：主歌踩器乐，**每 4 小节的最后 1–2 拍**混人声")
        elif fn == "pre_chorus":
            primary = "drums" if "drums" in ranked[:2] else (ranked[0] if ranked else "")
            secondary = "vocals"
            note = ("规则 3（MMFC 5.4-3）：段首踩鼓/器乐，**最后 1–2 小节切人声**为副歌"
                    "铺垫；切轨点必须落在小节线")
        elif fn in ("chorus", "final_chorus"):
            primary = "vocals"
            secondary = "drums"
            note = "规则 4（MMFC 5.4-4）：副歌全踩人声，kick 处叠 each/双押作重音"
        elif fn == "quiet_chorus":
            primary = ranked[1] if len(ranked) > 1 else (ranked[0] if ranked else "")
            secondary = "vocals"
            note = "落ちサビ = 减压段：踩次响轨，密度骤降（知识 001 情绪走势）"
        elif fn == "interlude":
            voiced = _seg_slice(bar_features.get("voiced_ratio", np.zeros(0)), s)
            has_sample = bool(voiced.size and 0.2 <= float(np.mean(voiced)) <= 0.5)
            primary = "vocals" if has_sample else "other"
            note = ("规则 5（MMFC 5.4-5）：间奏踩人声采样，**强度靠位移不靠密度**"
                    "（知识 003-2）" if has_sample else
                    "规则 5：间奏无人声采样 → 踩 other（独奏/riff）")
        elif fn == "drop":
            primary = ranked[0] if ranked else ""
            secondary = "drums"
            note = "器乐向 drop：踩最响轨 + 鼓骨架（无副歌概念，能量票主导）"
        else:  # bridge 等
            primary = ranked[0] if ranked else ""
            secondary = ranked[1] if len(ranked) > 1 else ""
            note = "无专用规则，取最响轨"

        # 规则 6：休息段改踩次响轨
        if s.rest and fn not in ("quiet_chorus",):
            alt = [x for x in ranked if x != primary]
            if alt:
                secondary = primary
                primary = alt[0]
            note += "；规则 6（MMFC 5.4-6）：休息段改踩次响轨，密度降一档"

        # 规则 7：第 N 次副歌与 repeat_of 同轨
        # ⚠️ 只在被指向的那一段**也是副歌家族**时才生效：重复分组是 SSM 聚类给的，
        # 副歌常常和 intro 落进同一簇，照抄 intro 的轨会把副歌踩成前奏。
        if s.repeat_of and fn in ("chorus", "final_chorus", "drop"):
            ref = by_span.get(tuple(s.repeat_of))
            if ref and ref["function"] in ("chorus", "final_chorus", "drop"):
                primary = ref["stem"]
                note += (f"；规则 7（MMFC 5.4-7）：与 {s.repeat_of[0]}–{s.repeat_of[1]} "
                         f"小节同轨，踩音不变")
        if s.upgrade:
            note += "；**upgrade**：配置升级（单星 → 双手星），不靠加密"

        # 副踩也要过一遍可用性：器乐曲里「小节尾混人声」无声可混
        if secondary:
            ok_s, why_s = _usable(bar_features, s, secondary)
            if not ok_s:
                note += f"；副踩 {secondary} 不可用（{why_s}）→ 取消副踩"
                secondary = ""

        # 降级规则
        ok, why = _usable(bar_features, s, primary) if primary else (False, "无候选轨")
        if not ok:
            note += f"；⚠️ 降级：{why} → 退回 drums 骨架（kick+snare）"
            secondary = secondary or primary
            primary = "drums"

        if secondary == primary:
            secondary = ""
        s.primary_stem = TRACK_OF_STEM.get(primary, primary)
        s.secondary_stem = TRACK_OF_STEM.get(secondary, secondary) if secondary else ""
        s.notes.append(note)
        by_span[(s.start_bar, s.end_bar)] = {"stem": primary, "function": fn}

    uniq = {s.primary_stem for s in segments if s.primary_stem}
    if len(uniq) <= 1 and duration_sec > SINGLE_TRACK_WARN_SEC:
        warnings.append(
            f"⚠️ 整曲主踩音轨只有 {uniq or '—'} 一个取值且曲长 {duration_sec:.0f}s > 90s —— "
            "这正是 MMFC 5.4 批评的「哪个最响就一直踩哪个」（知识 002）。"
            "已知例外：MMFC 5.6 的《金星》整谱单轨且评价很高，故为**警告不是硬错误**。")
    return segments, warnings
