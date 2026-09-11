"""踩音规划：**骨架轨 + 点缀轨**（v0.3；替换 v0.2 的单值 `primary_stem`）。

## 为什么换模型（v0.3 的核心改动）

v0.2 把"主踩音轨"做成**一段一条轨**的单值。8 首官方音频 × 官方谱的配对标定
（`docs/research/audio-chart-calibration.md` §5）把这个模型证伪了：

| 实测（n=8，代理判据） | 数字 |
|---|---|
| 官方 note 落在 `drums` onset 上（recall） | **0.607** |
| **不**落在 drums 上 | **36.2%** |
| 只落在 drums 上 | 28.1% |
| 只落在 vocals 上 | **3.2%** |
| 什么 stem 都不落（谱师自由发挥 / 装饰 / onset 漏检） | **18.6%** |
| 逐段"最高命中轨" | drums 70 段 / other 10 / vocals 3 / bass 2（共 85） |
| v0.2 规则表与"实测最高命中轨"的一致率 | **64.7%**，而"永远答 drums"是 **82.4%** |

→ 官方谱的真实形态是 **"鼓骨架 + 大量非鼓的填充与装饰"**，不存在"整段切到某一条
非鼓轨"。所以本模块输出：

- **`skeleton`（骨架轨）**：律动底盘，默认 `drums`；判据是**鼓 onset 密度与落格率**，
  鼓不可用（onset 太少 / 不落格）时才让位；
- **`accent`（点缀轨）**：`vocals` / `other` / `bass` 的有序列表，表示"骨架之外该往哪
  找填充与重音"；
- **`share`（估计占比）**：粗估各轨承担的踩音比例，含一个固定的 `free` 残差
  （= 标定实测 18.6% 什么都不落的那部分，留给谱师自由发挥）；
- **`evidence`（依据）**：纯音频特征的数字（onset 密度 / 落格率 / voiced_ratio）。

## 推理期没有谱面 —— 判据必须是纯音频特征

**标定报告里的命中率/lift 只是规则的设计依据，不是运行时输入。**
本模块在推理期只能看到 `features.py` 的逐小节音频特征表；下面每条规则的注释里
写着"实测 ××"的地方，都是**设计依据**（来自 8 首配对数据），代码里用来判断的
一律是 `n_onset_*` / `grid_fit_bar_*` / `voiced_ratio` / `share_*` / `riff_sim_*`。

## 规则表（v0.3；出处 MMFC 5.4 + 标定报告 §5.5 的 C1–C5 裁定）

| # | 段落 | 骨架 | 点缀 | 相对 v0.2 的变化 |
|---|------|------|------|------------------|
| 1 | `intro` | 匹配倾向最高轨（默认 drums） | 次高倾向轨 | 判据从**能量占比**改为 **onset 匹配倾向/落格率**（C3：intro 的最高 lift 是 `other` 5.08 而非 drums 4.39，能量口径选错了轨） |
| 2 | `verse` | `drums` | `other` ＞ `vocals`（小节尾） | 不变（实测 verse 的 other lift 3.49 最高） |
| 3 | `pre_chorus` | `drums` | `other`，**最后 1–2 小节 `vocals`** | 不变 |
| 4 | `chorus`/`final_chorus` | **`drums`** | 人声主导曲目 → `vocals` 为主；否则 `vocals` 只作重音点缀 | **条件触发**（C1）：14 个副歌段只有 2 段人声命中最高；且 C1 本身**存疑，待人工听审** |
| 5 | `interlude` | `drums` | **有人声采样时 `vocals` 优先** | **加强**（C4：interlude 的 vocals precision 0.715 / lift 3.18 均为该类型最高，是唯一被正面支持的规则） |
| 6 | 休息段 / `quiet_chorus` | `drums` | 非鼓里倾向最高者 | **标存疑**（C5：13 段样本，一致率 0.62，数据不支持也不反对） |
| 7 | 第 N 次副歌 | 与 `repeat_of` 同 | 同 | 不变（`upgrade` 置位，靠配置升级不靠加密） |
| 8 | `outro` | 与 `intro` 同 | 与 `intro` 同 | 不变 |

通用规则：
- **降级**：骨架轨 `n_onset_s < 2` 或 `grid_fit_s < 0.5` → 换成倾向最高的可用轨；
- **反新手护栏**（知识 002/005）：全曲"骨架 ∪ 点缀"只有 1 条轨且曲长 > 90 s → 警告
  （MMFC 5.6 的《金星》是已知例外，所以是**警告不是硬错误**）。

⚠️ C1（副歌全踩人声）在标定里被降级为「条件性 + **存疑**」：ground truth 是
"官方 note 是否落在 Demucs vocals 轨的 librosa onset ±30 ms 内"这个**代理**，
而人声 onset 检测本身是全管线最弱的一环（vocals 轨混 lead synth、鼓与人声重合
不可区分）。**待人工听审复核**。
"""

from __future__ import annotations

import numpy as np

# 逻辑轨名（与 tracks.TRACK_ORDER 一致）→ stem
TRACK_OF_STEM = {"drums": "drum", "vocals": "vocal", "bass": "bass", "other": "hook"}
STEM_OF_TRACK = {v: k for k, v in TRACK_OF_STEM.items()}
STEMS = ("drums", "bass", "other", "vocals")

MIN_ONSETS = 2
MIN_GRID_FIT = 0.5
SINGLE_TRACK_WARN_SEC = 90.0

# 标定报告 §5.2：18.6% 的官方 note 不落在任何 stem 的 onset 上
# （谱师自由发挥 / 装饰音 / onset 漏检，本报告无法区分三者）。
FREE_PLAY_SHARE = 0.19
# 人声主导判据（C1 条件触发）：两条**同时**成立才认为这段是「人声主导」。
# ⚠️ **阈值是在 n=8 上分出来的，属存疑**：8 首里 `vocals/drums` onset 密度比在
#    幻想のサテライト（标定里唯一"副歌踩人声"的正例）的三段副歌上是 0.68/0.72/0.87，
#    其余 7 首全部 ≤ 0.64；voiced_ratio 再挡掉 Mare Maris（比值 0.64 但 voiced 0.07）。
#    取 0.65 只是让规则在唯一已知正例上能触发，**没有独立样本验证过**。
#    另注：人声 onset 检测系统性漏检（比鼓保守），所以判据不是"≥1.0"。
VOCAL_LED_VOICED = 0.45          # 段内 voiced_ratio 均值
VOCAL_LED_ONSET_RATIO = 0.65     # vocals onset 密度 / drums onset 密度
# 间奏人声采样（C4）：人声进出、密度不高
SAMPLE_VOICED_RANGE = (0.12, 0.55)
MAX_ACCENTS = 2


def _seg_slice(arr, s) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[s.start_bar - 1: min(s.end_bar, len(a))]


def _mean(bar_features: dict[str, np.ndarray], s, key: str, default: float = 0.0) -> float:
    seg = _seg_slice(bar_features.get(key, np.zeros(0)), s)
    return float(np.mean(seg)) if seg.size else float(default)


def _seg_stats(bar_features: dict[str, np.ndarray], s) -> dict:
    """段内的纯音频特征汇总（推理期唯一可用的信息）。"""
    st: dict = {"voiced": _mean(bar_features, s, "voiced_ratio"),
                "riff_sim": _mean(bar_features, s, "riff_sim_4"),
                "dens": {}, "fit": {}, "share": {}, "tendency": {}}
    for stem in STEMS:
        dens = _mean(bar_features, s, f"n_onset_{stem}")
        fit = _seg_slice(bar_features.get(f"grid_fit_bar_{stem}",
                                          bar_features.get(f"grid_fit_{stem}",
                                                           np.zeros(0))), s)
        fit = float(np.mean(fit)) if fit.size else 1.0
        st["dens"][stem] = dens
        st["fit"][stem] = fit
        st["share"][stem] = _mean(bar_features, s, f"share_{stem}")
        # 「匹配倾向」= 该轨每小节能提供多少个**落得上格**的可踩音。
        # 这是纯音频量，替代 v0.2 的能量占比 `share_s`（C3：能量口径选错轨）。
        st["tendency"][stem] = dens * fit
    return st


def _rank(st: dict, exclude=()) -> list[str]:
    """按匹配倾向排序的 stem 列表（降序）。"""
    cand = {k: v for k, v in st["tendency"].items() if k not in exclude}
    return sorted(cand, key=cand.get, reverse=True)


def _usable(st: dict, stem: str) -> tuple[bool, str]:
    """降级判据：onset 太少或不落格的轨不能当骨架。"""
    if st["dens"].get(stem, 0.0) < MIN_ONSETS:
        return False, f"{stem} 段内 onset 均值 {st['dens'].get(stem, 0.0):.1f} < {MIN_ONSETS}"
    if st["fit"].get(stem, 1.0) < MIN_GRID_FIT:
        return False, f"{stem} 的落格率 {st['fit'].get(stem, 1.0):.2f} < {MIN_GRID_FIT}"
    return True, ""


def _pick_skeleton(st: dict) -> tuple[str, str]:
    """骨架轨：默认 drums；不可用时换成倾向最高的可用轨。"""
    ok, why = _usable(st, "drums")
    if ok:
        return "drums", ""
    for stem in _rank(st, exclude=("drums",)):
        if _usable(st, stem)[0]:
            return stem, f"⚠️ 骨架降级：{why} → 改用 {stem}（匹配倾向最高的可用轨）"
    return "drums", f"⚠️ {why}，且无其他可用轨 → 仍按 drums 骨架处理（该段可踩音极少）"


def _vocal_led(st: dict) -> tuple[bool, str]:
    """C1 的条件触发判据（纯音频）：人声主导 = voiced 高 **且** 人声 onset 密度 ≳ 鼓。"""
    v, d = st["dens"]["vocals"], st["dens"]["drums"]
    ratio = v / d if d > 1e-9 else (float("inf") if v > 0 else 0.0)
    led = st["voiced"] >= VOCAL_LED_VOICED and ratio >= VOCAL_LED_ONSET_RATIO
    txt = (f"voiced={st['voiced']:.2f}、vocals/drums onset 密度比={ratio:.2f}"
           f"（判据：≥{VOCAL_LED_VOICED} 且 ≥{VOCAL_LED_ONSET_RATIO}）")
    return led, txt


def _has_vocal_sample(st: dict) -> bool:
    """间奏人声采样：人声进出（voiced 中等）且确实有 onset。"""
    lo, hi = SAMPLE_VOICED_RANGE
    return lo <= st["voiced"] <= hi and st["dens"]["vocals"] >= 1.0


def _estimate_share(st: dict, skeleton: str, accents: list[str]) -> dict:
    """估计各轨承担的踩音占比（粗估，只用音频特征 + 一个标定常数）。

    做法：在"骨架 + 点缀"这几条轨上按匹配倾向归一，再整体乘 `1 − FREE_PLAY_SHARE`，
    余下的 `free` 是标定实测"什么 stem 都不落"的 18.6%（谱师自由发挥/装饰/漏检）。
    ⚠️ 这是**量级提示**，不是标定值：逐段占比从未被验证过。
    """
    used = [skeleton] + [a for a in accents if a != skeleton]
    w = {k: max(st["tendency"].get(k, 0.0), 1e-6) for k in used}
    # 骨架天然承担更多（实测 drums recall 0.607 vs 其余 0.17–0.27），给一个固定倾斜
    w[skeleton] *= 2.0
    tot = sum(w.values())
    out = {k: round((1.0 - FREE_PLAY_SHARE) * v / tot, 2) for k, v in w.items()}
    out["free"] = FREE_PLAY_SHARE
    return out


def plan_stems(segments, bar_features: dict[str, np.ndarray], grid,
               duration_sec: float = 0.0) -> tuple[list, list[str]]:
    """给每段填 `skeleton_stem` / `accent_stems` / `accent_share` / 依据。

    返回 `(segments, warnings)`。
    """
    warnings: list[str] = []
    by_span: dict[tuple[int, int], dict] = {}
    intro_plan: dict | None = None

    for s in segments:
        fn = s.function
        st = _seg_stats(bar_features, s)
        skeleton, degrade = _pick_skeleton(st)
        ranked_non_skel = _rank(st, exclude=(skeleton,))
        accents: list[str] = []
        note = ""
        suspect = ""

        if fn in ("intro", "outro") and fn == "outro" and intro_plan:
            # 规则 8：与 intro 前后呼应
            skeleton = intro_plan["skeleton"]
            accents = list(intro_plan["accents"])
            note = ("规则 8（MMFC 5.4-8）：与 intro 同骨架/同点缀前后呼应；"
                    "⚠️ outro ≠ 减压（知识 031：官方谱末段最强，尾杀约 2/3）")
        elif fn in ("intro", "outro"):
            accents = ranked_non_skel[:1]
            note = ("规则 1（MMFC 5.4-1 + 标定 C3）：前奏/尾奏按**onset 匹配倾向**（onset 密度 ×"
                    "落格率）选骨架与点缀，**不再用能量占比**——实测 intro 的最高 lift 是 "
                    "`other`(5.08) 而非 drums(4.39)，"
                    "「哪个响踩哪个」的能量口径会选错轨")
        elif fn == "verse":
            accents = [a for a in ("other", "vocals") if _usable(st, a)[0]][:MAX_ACCENTS]
            note = ("规则 2（MMFC 5.4-2）：主歌鼓骨架 + 器乐点缀，**每 4 小节的最后 1–2 拍**"
                    "混人声（实测 verse 的 other lift 3.49 为该类型最高）")
        elif fn == "pre_chorus":
            accents = [a for a in ("other", "vocals") if _usable(st, a)[0]][:MAX_ACCENTS]
            note = ("规则 3（MMFC 5.4-3）：段首鼓骨架 + 器乐点缀，**最后 1–2 小节把点缀"
                    "切到人声**为副歌铺垫；切轨点必须落在小节线")
        elif fn in ("chorus", "final_chorus"):
            led, why = _vocal_led(st)
            if led and _usable(st, "vocals")[0]:
                accents = ["vocals"] + [a for a in ranked_non_skel if a != "vocals"][:1]
                note = ("规则 4（MMFC 5.4-4，**条件触发**）：本段判为**人声主导**（" + why +
                        "）→ 点缀以 **vocals 为主**，鼓仍是骨架")
            else:
                accents = [a for a in ranked_non_skel if a != "vocals"][:1]
                if _usable(st, "vocals")[0]:
                    accents = (accents + ["vocals"])[:MAX_ACCENTS]
                note = ("规则 4（MMFC 5.4-4，**条件触发**）：本段**不是人声主导**（" + why +
                        "）→ 维持 **drums 骨架**，人声只作重音点缀")
            suspect = ("⚠️ **存疑（待人工听审）**：「副歌全踩人声」在 8 首配对标定里 14 个副歌段"
                       "只有 2 段人声命中最高（唯一正例：幻想のサテライト），但该结论的 ground "
                       "truth 是「note 是否落在 vocals 轨 onset ±30ms 内」的**代理**，"
                       "人声 onset 检测是全管线最弱环节之一 —— 规则已降级为条件性，"
                       "**不得当硬约束**")
        elif fn == "interlude":
            if _has_vocal_sample(st):
                accents = ["vocals"] + [a for a in ranked_non_skel if a != "vocals"][:1]
                note = ("规则 5（MMFC 5.4-5，**标定唯一正面支持的一条，已加强**）：间奏有人声"
                        f"采样（voiced={st['voiced']:.2f}、vocals onset "
                        f"{st['dens']['vocals']:.1f}/小节）→ **人声采样优先踩**；实测 interlude "
                        "的 vocals precision 0.715、lift 3.18 均为各段落类型最高（人声在间奏里"
                        "一旦出现几乎必被官方采用）。**强度靠位移不靠密度**（知识 003-2）")
            else:
                accents = ranked_non_skel[:1]
                note = ("规则 5：间奏无人声采样 → 点缀取匹配倾向最高的非鼓轨"
                        "（通常是 other 的独奏/riff）")
        elif fn == "quiet_chorus":
            accents = [a for a in ranked_non_skel if a != "drums"][:1]
            note = ("落ちサビ = 减压段：鼓骨架 + 非鼓点缀，密度降一档但**不清零**"
                    "（知识 031：休息小节只占 2.43%）")
            suspect = ("⚠️ **存疑**：标定 C5 —— `quiet_chorus` 只有 13 段样本、规则一致率 0.62，"
                       "数据既不支持也不反对，维持现状")
        elif fn == "drop":
            accents = ranked_non_skel[:1]
            note = "器乐向 drop：鼓骨架 + 匹配倾向最高的非鼓轨点缀（无副歌概念，能量票主导）"
        else:  # bridge 等
            accents = ranked_non_skel[:1]
            note = "无专用规则：鼓骨架 + 匹配倾向最高的非鼓轨点缀"

        # 规则 6：休息段（非 quiet_chorus 的 rest 标记）
        if s.rest and fn != "quiet_chorus":
            alt = [a for a in ranked_non_skel if a != "drums"]
            if alt:
                accents = alt[:1]
            note += "；规则 6（MMFC 5.4-6）：休息段点缀改取非鼓轨，密度降一档"
            suspect = suspect or ("⚠️ **存疑**：标定 C5 —— 休息段规则的样本只有 13 段、"
                                  "一致率 0.62，不足以判定")

        # 规则 7：第 N 次副歌与 repeat_of 同踩音
        if s.repeat_of and fn in ("chorus", "final_chorus", "drop"):
            ref = by_span.get(tuple(s.repeat_of))
            if ref and ref["function"] in ("chorus", "final_chorus", "drop"):
                skeleton = ref["skeleton"]
                accents = list(ref["accents"])
                note += (f"；规则 7（MMFC 5.4-7）：与 {s.repeat_of[0]}–{s.repeat_of[1]} "
                         f"小节同踩音（骨架与点缀都不变）")
        if s.upgrade:
            note += "；**upgrade**：配置升级（单星 → 双手星），不靠加密（知识 031「强度≠密度」）"

        # 点缀轨的可用性检查：器乐曲里「小节尾混人声」无声可混
        kept, dropped = [], []
        for a in accents[:MAX_ACCENTS]:
            ok_a, why_a = _usable(st, a)
            (kept if ok_a else dropped).append(a if ok_a else f"{a}（{why_a}）")
        accents = kept
        if dropped:
            note += "；点缀 " + "、".join(dropped) + " 不可用 → 去掉"
        if not accents:
            # 兜底：至少给一条可用的非骨架轨当点缀（官方谱 36.2% 的 note 不在鼓上，
            # 输出"只有骨架"会把 LLM 推回"整曲只踩一条轨"的新手做法）
            fallback = next((a for a in _rank(st, exclude=(skeleton,))
                             if _usable(st, a)[0]), "")
            if fallback:
                accents = [fallback]
                note += f"；无可用点缀 → 兜底取匹配倾向最高的可用轨 {fallback}"
            else:
                note += "；⚠️ 本段没有任何可用的非骨架轨（其余三轨 onset 太少/不落格）"
        if degrade:
            note += "；" + degrade

        s.skeleton_stem = TRACK_OF_STEM.get(skeleton, skeleton)
        s.accent_stems = [TRACK_OF_STEM.get(a, a) for a in accents]
        s.accent_share = {TRACK_OF_STEM.get(k, k): v
                          for k, v in _estimate_share(st, skeleton, accents).items()}
        # 兼容字段（v0.2 的单值模型；**已弃用**，只为旧的下游读取不炸）
        s.primary_stem = s.skeleton_stem
        s.secondary_stem = s.accent_stems[0] if s.accent_stems else ""
        s.plan_evidence = [
            "onset 密度/小节：" + "、".join(f"{k}={st['dens'][k]:.1f}" for k in STEMS),
            "落格率：" + "、".join(f"{k}={st['fit'][k]:.2f}" for k in STEMS),
            f"voiced_ratio={st['voiced']:.2f}；能量占比 "
            + "、".join(f"{k}={st['share'][k]:.2f}" for k in STEMS),
        ]
        s.notes.append(note)
        if suspect:
            s.notes.append(suspect)
        by_span[(s.start_bar, s.end_bar)] = {"skeleton": skeleton, "accents": list(accents),
                                             "function": fn}
        if fn == "intro" and intro_plan is None:
            intro_plan = {"skeleton": skeleton, "accents": list(accents)}

    used_tracks = set()
    for s in segments:
        if s.skeleton_stem:
            used_tracks.add(s.skeleton_stem)
        used_tracks.update(s.accent_stems or [])
    if len(used_tracks) <= 1 and duration_sec > SINGLE_TRACK_WARN_SEC:
        warnings.append(
            f"⚠️ 整曲「骨架 ∪ 点缀」只有 {used_tracks or '—'} 一条轨且曲长 {duration_sec:.0f}s "
            "> 90s —— 这正是 MMFC 5.4 批评的「哪个最响就一直踩哪个」（知识 002）。"
            "已知例外：MMFC 5.6 的《金星》整谱单轨且评价很高，故为**警告不是硬错误**。")
    return segments, warnings
