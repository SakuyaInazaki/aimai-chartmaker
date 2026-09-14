"""歌词时间轴（LRC）→ 乐句边界 → 「官方谱是否跟着唱词句读走」的假设检验。

背景（为什么要做这件事）
------------------------
`docs/research/audio-chart-calibration-n160.md` 在 160 首官方曲上确认：音频强度对官方逐小节
密度的解释力**在密度侧已经饱和**（同权重口径 ρ 0.2959 → 0.2999，四倍样本纹丝不动，
一个默认值都没再改），而人声曲（ρ 0.28）系统性地低于器乐曲（ρ 0.42）。
`docs/research/stem-refinement-n40.md` 又排除了「换更细的分轨/加有音高 note 轨」这条路
（六路分离只有 piano 有净增益，basic-pitch melody 轨是候选池灌水，人声曲 ρ 一点没动）。

→ 剩下的假设是：**人声曲里官方谱跟着「唱词句读」走，而不是跟着响度走。**
   要证伪或证实它，需要「真实乐句边界」这一层结构，而不是「人声活动率」这个标量。

本模块用**公网 LRC 歌词时间轴的逐行时间戳**当乐句边界的 ground truth。

版权与落盘约束（硬性）
----------------------
**歌词文本只在内存里用**。落盘（`out/lyrics/`，已 gitignore）只存：

- 每行的**时间戳**（起、止）与**行序号**；
- 每行的**字符数**与**假名/汉字/拉丁/数字的粗计数**（用于估计音节数）；
- 来源（lrclib / netease）、曲目 id、匹配得分。

**不落歌词原文**，仓库内任何文件（含本报告与 CSV）都不得出现歌词文本。

对齐这一步为什么难（**本轮的主要结论就在这里**）
--------------------------------------------------
maimai 的 `track.mp3` 是约 2 分钟的**游戏剪辑版**，而 LRC 对应**完整原曲**：
本轮 97 首实测的中位数是「LRC 末行比游戏版结尾晚 **67 s**」，也就是说游戏版通常只保留了
原曲的一半左右，而且是**多处剪接**。于是对齐要在一个很大的搜索空间里定位偏移，
而音频侧唯一能用的锚是 Demucs `vocals` 轨 —— 它泄漏严重（能量 VAD 的"有人声"占比
普遍 0.65–0.79，知识 002 第三轮修订）。

**跨曲阴性对照（把 A 曲的 LRC 拿去对 B 曲的音频）证明这一步不成立**：
把真 LRC 与外来 LRC 放在同一个对齐器下，四种统计量全都分不开
（人声 onset 命中二项 z、VAD 曲线互相关、人声 RMS 对比度、小节格点受限扫描），
奇偶折半一致率真 0.05 / 假 0.01。详见 `docs/research/vocal-phrase-alignment.md` §4。
**唯一能区分真假 LRC 的是「节拍一致性」**（行首落在小节格上的 Rayleigh 集中度，
真 n·R² 中位 8.1 vs 外来 0.6）——它能证明"这份 LRC 属于这首歌"，
却**不含任何关于偏移在哪里的信息**。

所以本模块保留完整的抓取/对齐/检验链路（含 `cross_song_control` 阴性对照），
但真正可用的结论来自 §9 的**免对齐口径**：把"句读边界"换成
`tools/audio_analysis/phrases.py` 的纯音频估计，再对官方谱做同一套 lift + 随机平移对照。

CLI
---
```bash
# 1) 抓 LRC（只落时间戳与计数）
python -m tools.calibration.lyrics_phrases fetch --calib-dir out/calib --out out/lyrics

# 2) 对齐到游戏剪辑版音频
python -m tools.calibration.lyrics_phrases align --calib-dir out/calib --lyrics out/lyrics

# 3) 假设检验 T1–T4（LRC 口径，含外来 LRC 对照）+ 免对齐口径 + 报告数据
python -m tools.calibration.lyrics_phrases test --calib-dir out/calib --lyrics out/lyrics \
    --csv docs/research/data/vocal-phrase-summary.csv --metrics <scratchpad>/phrase_metrics.json

# 4) 只跑跨曲阴性对照
python -m tools.calibration.lyrics_phrases control --calib-dir out/calib --lyrics out/lyrics
```
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# 0. 曲名/艺术家归一化
# --------------------------------------------------------------------------

#: simai `&title` 的前缀（官方 ST 曲包统一带这个标记）
ST_PREFIX = "[ST] "

#: 曲名里常见的、检索时应当被无视的符号（含全角）
_PUNCT_RE = re.compile(r"[\s~〜～\-–—_/\\|·・.,!?！？'’\"“”\[\]（）()【】「」『』:：;；*＊+＋#＃&＆%％$＄@]+")

#: 曲名里被装饰符号替掉的字母（maimai 曲名常见：D✪N'T ST✪P R✪CKIN'）
_DECOR_MAP = {
    "✪": "o", "★": "", "☆": "", "♪": "", "♡": "", "♥": "", "∞": "", "◆": "",
    "◇": "", "●": "", "○": "", "■": "", "□": "", "†": "", "＋": "+", "／": "/",
}


def strip_st(title: str) -> str:
    """去掉 `&title` 的 `[ST] ` 前缀。"""
    t = str(title or "").strip()
    return t[len(ST_PREFIX):].strip() if t.startswith(ST_PREFIX) else t


def _fold(s: str) -> str:
    """检索用的折叠形式：NFKC + 小写 + 去标点空白 + 装饰符号还原。"""
    s = unicodedata.normalize("NFKC", str(s or ""))
    for k, v in _DECOR_MAP.items():
        s = s.replace(k, v)
    s = s.lower()
    return _PUNCT_RE.sub("", s)


def title_variants(title: str) -> list[str]:
    """给一个官方曲名生成若干检索写法（原名 / 去括注 / 去装饰符 / 主副标题切分）。"""
    base = strip_st(title)
    out: list[str] = []

    def add(x: str) -> None:
        x = unicodedata.normalize("NFKC", x).strip()
        if x and x not in out:
            out.append(x)

    add(base)
    # 去掉括注（全角/半角）
    add(re.sub(r"[（(\[【][^）)\]】]*[）)\]】]", " ", base))
    # 去掉装饰符号
    dec = base
    for k, v in _DECOR_MAP.items():
        dec = dec.replace(k, v)
    add(dec)
    # 主副标题：用 ~...~ / 「」/ - 切开取前段
    add(re.split(r"[~〜～]", base)[0])
    add(re.split(r"\s+[-–—]\s+", base)[0])
    # 首尾的 - 包裹（-OutsideR:RequieM-）
    add(base.strip("-–—"))
    return [x for x in out if x]


_ARTIST_SPLIT_RE = re.compile(r"\s*(?:feat\.?|featuring|ft\.?|vs\.?|×|x|/|,|・|&)\s*",
                              flags=re.I)


def artist_tokens(artist: str) -> list[str]:
    """把 `&artist` 拆成用于打分的 token（去掉 CV/专辑名等噪声）。"""
    a = unicodedata.normalize("NFKC", str(artist or ""))
    a = re.sub(r"[「『\[][^」』\]]*[」』\]]", " ", a)      # 去掉「アルバム名」
    a = re.sub(r"[（(][^）)]*[）)]", " ", a)               # 去掉 (CV:xxx)
    toks = [t.strip() for t in _ARTIST_SPLIT_RE.split(a)]
    return [_fold(t) for t in toks if len(t.strip()) >= 2]


# --------------------------------------------------------------------------
# 1. LRC 解析 → 只留时间戳与字符计数
# --------------------------------------------------------------------------

_LRC_TS_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_LRC_META_RE = re.compile(r"^\[[a-zA-Z#]+:")

#: 小书写假名不单独计一个拍（モーラ）
_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶっッ")


@dataclass
class PhraseLine:
    """一行歌词的**元数据**（不含文本）。"""

    idx: int                 # 行序号（仅非空行计数）
    t: float                 # 起点（原曲时钟，秒）
    t_end: float             # 终点（下一行起点或空行时间戳）
    n_chars: int             # 去空白后的字符数
    n_kana: int
    n_kanji: int
    n_latin: int
    n_digit: int
    syl_est: float           # 音节（モーラ）估计


def _classify(text: str) -> dict:
    n_kana = n_kanji = n_latin = n_digit = n_chars = 0
    syl = 0.0
    for ch in text:
        if ch.isspace():
            continue
        n_chars += 1
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF:            # 平假名 + 片假名
            n_kana += 1
            if ch not in _SMALL_KANA:
                syl += 1.0
        elif 0x4E00 <= o <= 0x9FFF:          # 汉字
            n_kanji += 1
            syl += 1.8                        # 日文汉字平均读音长度的粗估
        elif ch.isdigit():
            n_digit += 1
            syl += 1.5
        elif ch.isalpha():
            n_latin += 1
    if n_latin:
        # 拉丁文按元音团估音节：粗略但只用于相对比较
        syl += max(1.0, n_latin / 3.0)
    return {"n_chars": n_chars, "n_kana": n_kana, "n_kanji": n_kanji,
            "n_latin": n_latin, "n_digit": n_digit, "syl_est": round(syl, 2)}


def parse_lrc(lrc: str, max_line_sec: float = 12.0) -> list[PhraseLine]:
    """把 LRC 文本解析成**只含时间戳与计数**的乐句行表（原文即刻丢弃）。

    - 一行可带多个时间戳（`[00:12.00][01:30.00]同一句`），按每个时间戳各生成一行；
    - **空文本行**不生成乐句，但它的时间戳用作**上一句的终点**（LRC 里这正是"唱完了"的标记）；
    - 末行终点取 `min(上一句起点 + 估计时长, max_line_sec)`。
    """
    raw: list[tuple[float, str]] = []
    for line in str(lrc or "").splitlines():
        if _LRC_META_RE.match(line) and not _LRC_TS_RE.match(line):
            continue
        stamps = list(_LRC_TS_RE.finditer(line))
        if not stamps:
            continue
        text = line[stamps[-1].end():]
        for m in stamps:
            mm, ss, frac = m.group(1), m.group(2), m.group(3) or "0"
            t = int(mm) * 60 + int(ss) + int(frac) / (10 ** len(frac))
            raw.append((float(t), text))
    if not raw:
        return []
    raw.sort(key=lambda x: x[0])

    lines: list[PhraseLine] = []
    idx = 0
    for i, (t, text) in enumerate(raw):
        cls = _classify(text)
        if cls["n_chars"] == 0:
            continue
        # 终点：下一个时间戳（不论是否空行）
        t_end = raw[i + 1][0] if i + 1 < len(raw) else t + min(max_line_sec, 4.0)
        t_end = min(t_end, t + max_line_sec)
        if t_end <= t:
            t_end = t + 0.5
        lines.append(PhraseLine(idx=idx, t=float(t), t_end=float(t_end), **cls))
        idx += 1
    return lines


# --------------------------------------------------------------------------
# 2. 抓取（LRCLIB 首选，网易云备选）
# --------------------------------------------------------------------------

UA = "aimai-chartmaker-research/0.1 (https://github.com/SakuyaInazaki/aimai-chartmaker)"
LRCLIB = "https://lrclib.net/api"
NETEASE = "https://music.163.com/api"


def _http_json(url: str, params: dict | None = None, headers: dict | None = None,
               timeout: float = 20.0) -> object | None:
    import urllib.error
    import urllib.parse
    import urllib.request

    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def _score_candidate(cand_title: str, cand_artist: str, title: str,
                     artists: Sequence[str]) -> float:
    """候选曲目与目标曲的匹配分（0–1.4）。曲名占主，艺术家 token 命中加分。"""
    ft, fc = _fold(title), _fold(cand_title)
    if not ft or not fc:
        return 0.0
    if ft == fc:
        s = 1.0
    elif ft in fc or fc in ft:
        shorter, longer = (ft, fc) if len(ft) <= len(fc) else (fc, ft)
        s = 0.55 + 0.35 * (len(shorter) / max(1, len(longer)))
    else:
        # 字符级 Jaccard 兜底（日文曲名对 difflib 不友好）
        a, b = set(ft), set(fc)
        s = 0.45 * (len(a & b) / max(1, len(a | b)))
    fa = _fold(cand_artist)
    if artists and fa:
        if any(t and (t in fa or fa in t) for t in artists):
            s += 0.4
    return float(s)


#: 候选分的收录门槛（低于它的候选连"拿去让音频判"都不值得）
MIN_CAND_SCORE = 0.52
#: 每首最多留几个候选交给音频对齐去裁决
MAX_CANDIDATES = 6


def fetch_lrclib(title: str, artists: Sequence[str], sleep: float = 1.0,
                 limit: int = MAX_CANDIDATES) -> list[dict]:
    """LRCLIB 检索，返回**按分排序的候选列表**（`synced` 为 LRC 原文，仅内存用）。

    ⚠️ 曲名匹配单独并不可靠（"ARROW" 这类通用词能撞上完全无关的曲子），所以这里
    **不做单点裁决**，只负责给候选打分；最终由 `align_lines` 用音频波形去裁决。
    """
    pool: dict[object, dict] = {}
    for var in title_variants(title)[:4]:
        data = _http_json(f"{LRCLIB}/search", {"q": var})
        time.sleep(sleep)
        if not isinstance(data, list):
            data = _http_json(f"{LRCLIB}/search", {"track_name": var})
            time.sleep(sleep)
        if not isinstance(data, list):
            continue
        for c in data[:25]:
            if not isinstance(c, dict) or c.get("instrumental"):
                continue
            synced = c.get("syncedLyrics")
            if not synced:
                continue
            s = _score_candidate(str(c.get("trackName") or c.get("name") or ""),
                                 str(c.get("artistName") or ""), title, artists)
            if s < MIN_CAND_SCORE:
                continue
            cid = c.get("id")
            if cid in pool and pool[cid]["score"] >= s:
                continue
            pool[cid] = {"source": "lrclib", "id": cid,
                         "name": str(c.get("trackName") or c.get("name") or ""),
                         "artist": str(c.get("artistName") or ""),
                         "duration": c.get("duration"), "score": round(float(s), 3),
                         "synced": synced}
        if any(v["score"] >= 1.3 for v in pool.values()):
            break
    return sorted(pool.values(), key=lambda d: -d["score"])[:limit]


def fetch_netease(title: str, artists: Sequence[str], sleep: float = 1.0,
                  limit: int = 3) -> list[dict]:
    """网易云公开接口检索（LRCLIB 候选不足时的补充），同样返回候选列表。"""
    headers = {"Referer": "https://music.163.com/", "Cookie": "appver=2.0.2"}
    cands: dict[object, dict] = {}
    for var in title_variants(title)[:3]:
        data = _http_json(f"{NETEASE}/search/get", {"s": var, "type": 1, "limit": 12},
                          headers=headers)
        time.sleep(sleep)
        songs = (((data or {}).get("result") or {}).get("songs")) or []
        for c in songs:
            names = " ".join(str(a.get("name", "")) for a in (c.get("artists") or []))
            s = _score_candidate(str(c.get("name") or ""), names, title, artists)
            if s < MIN_CAND_SCORE:
                continue
            cid = c.get("id")
            if cid in cands and cands[cid]["score"] >= s:
                continue
            cands[cid] = {"source": "netease", "id": cid,
                          "name": str(c.get("name") or ""), "artist": names,
                          "duration": (c.get("duration") or 0) / 1000.0,
                          "score": round(float(s), 3), "synced": None}
        if any(v["score"] >= 1.3 for v in cands.values()):
            break
    out: list[dict] = []
    for c in sorted(cands.values(), key=lambda d: -d["score"])[:limit]:
        ly = _http_json(f"{NETEASE}/song/lyric",
                        {"id": c["id"], "lv": 1, "kv": 0, "tv": -1}, headers=headers)
        time.sleep(sleep)
        synced = (((ly or {}).get("lrc") or {}).get("lyric")) or ""
        if synced and _LRC_TS_RE.search(synced):
            c["synced"] = synced
            out.append(c)
    return out


def fetch_one(title: str, artist: str, sleep: float = 1.0,
              allow_netease: bool = True) -> dict:
    """抓一首。返回**不含歌词原文**的结果 dict（每个候选的 `lines` 只有时间戳与计数）。"""
    arts = artist_tokens(artist)
    raw = fetch_lrclib(title, arts, sleep=sleep)
    if allow_netease and (not raw or max(c["score"] for c in raw) < 1.2):
        raw = raw + fetch_netease(title, arts, sleep=sleep)
    cands: list[dict] = []
    for c in sorted(raw, key=lambda d: -d["score"])[:MAX_CANDIDATES]:
        lines = parse_lrc(c.pop("synced") or "")
        if len(lines) < 4:
            continue
        c["n_lines"] = len(lines)
        c["lrc_span_sec"] = round(float(lines[-1].t_end - lines[0].t), 2)
        c["lines"] = [asdict(l) for l in lines]
        cands.append(c)
    if not cands:
        return {"found": False, "reason": "no_match", "candidates": []}
    return {"found": True, "n_candidates": len(cands), "candidates": cands}


# --------------------------------------------------------------------------
# 3. 对齐：LRC（原曲时钟）→ 游戏剪辑版音频时钟
# --------------------------------------------------------------------------
#
# 为什么不用「人声 VAD 曲线互相关」（第一版做法，已废弃）
# ----------------------------------------------------------
# 实测发现两个致命问题：
#   1. Demucs `vocals` 轨常年泄漏（知识 002 第三轮修订：能量 VAD 把 31/51 首器乐曲
#      判成人声曲），VAD 的「有人声」占比普遍 0.65–0.79，曲线接近常 1，
#      互相关只能定位到「人声区大致在哪」，精度 ±数秒；
#   2. 绝大多数公网 LRC 的行**首尾相接**（`t_end` = 下一行的 `t`，逐行间隙中位数 0.00 s），
#      所以「正在唱」曲线本身就是一整块，没有可供对齐的细节。
# 实测对照（4 首）：VAD 互相关给出的偏移在 3/4 首上与脉冲法差 2–6 s。
#
# 现在的做法：**行首脉冲 × 人声 onset 的二项 z 统计**
# ---------------------------------------------------
# LRC 唯一可靠的信息是**每行的起点**（歌手开口的时刻），音频侧对应的是 `vocals` 轨的
# onset。于是把对齐写成：找一个偏移 `delta`，使「映射后的行首」尽量多地落在
# 「某个人声 onset 的 ±tol 内」。命中率的零假设基线可以**闭式算出来**
# （= onset 膨胀后覆盖的时间比例 `cover`），于是打分直接用二项 z：
#
#     z(delta) = (hits - n·cover) / sqrt(n·cover·(1-cover))
#
# 它同时解决了「随机平移对照」——`cover` 就是随机平移的期望命中率。
#
# 剪接：**按 LRC 的段落切块，每块各自定偏移**
# ------------------------------------------
# 游戏剪辑版的常见做法是「保留若干整段、丢掉其余」，所以按 LRC 的空行/长间隙把行切成块，
# 每块独立全范围搜偏移；块内偏移一致、块间可以不同。再要求块在音频上的顺序与
# 在原曲上的顺序一致（单调），并丢掉「没被剪进游戏版」的块。

#: 对齐用的时间分辨率（100 Hz = 10 ms）
ALIGN_HZ = 100.0
#: 行首 ↔ 人声 onset 的命中容差（秒）
HIT_TOL = 0.10
#: 切块的 LRC 间隙阈值（秒）——大于它就认为是段落边界
BLOCK_GAP = 1.5
#: 一个块至少要有几行才值得单独定偏移
MIN_BLOCK_LINES = 3


def _dilated_onset_grid(onsets: Sequence[float], dur: float, hz: float = ALIGN_HZ,
                        tol: float = HIT_TOL) -> np.ndarray:
    """把人声 onset 膨胀成 0/1 栅格：`grid[i] = 1` 表示 `i/hz` 处 ±tol 内有 onset。"""
    n = max(1, int(round(dur * hz)) + 1)
    g = np.zeros(n, dtype=float)
    w = int(round(tol * hz))
    for t in np.asarray(onsets, dtype=float):
        i = int(round(float(t) * hz))
        a, b = max(0, i - w), min(n, i + w + 1)
        if b > a:
            g[a:b] = 1.0
    return g


def _pulse_grid(times: Sequence[float], hz: float = ALIGN_HZ,
                length: int | None = None) -> np.ndarray:
    ts = np.asarray(times, dtype=float)
    n = length or (int(round(float(ts.max()) * hz)) + 2 if ts.size else 2)
    g = np.zeros(max(2, n), dtype=float)
    for t in ts:
        i = int(round(float(t) * hz))
        if 0 <= i < g.size:
            g[i] += 1.0
    return g


def _xcorr_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`out[m] = Σ_i a[i]·b[i + m − (len(a)−1)]`，FFT 实现，长度 `len(a)+len(b)−1`。"""
    na, nb = a.size, b.size
    n_fft = 1
    while n_fft < na + nb:
        n_fft <<= 1
    r = np.fft.irfft(np.fft.rfft(a[::-1], n_fft) * np.fft.rfft(b, n_fft), n_fft)
    return r[: na + nb - 1]


def scan_offsets(line_starts: Sequence[float], onset_grid: np.ndarray,
                 hz: float = ALIGN_HZ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全范围扫偏移。返回 `(deltas, z, hit_rate)`，三者等长。

    `deltas[k]` 的含义：**音频时刻 t ↔ 原曲时刻 t + delta**（即原曲 t_lyric → 音频 t_lyric − delta）。
    """
    a = np.asarray(onset_grid, dtype=float)
    na = a.size
    b = _pulse_grid(line_starts, hz=hz)
    hits = _xcorr_full(a, b)
    nmap = _xcorr_full(np.ones(na), b)
    d_idx = np.arange(hits.size) - (na - 1)
    deltas = d_idx / hz
    cover = float(a.mean())
    cover = min(max(cover, 1e-6), 1 - 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (hits - nmap * cover) / np.sqrt(np.maximum(nmap, 1e-9) * cover * (1 - cover))
        rate = hits / np.maximum(nmap, 1e-9)
    z[nmap < 1] = -np.inf
    return deltas, z, rate


@dataclass
class Block:
    """一段连续的 LRC 行（按段落间隙切开）及其独立偏移。"""

    i0: int          # 行下标区间 [i0, i1)
    i1: int
    delta: float = 0.0
    z: float = 0.0
    hit_rate: float = 0.0
    n_mapped: int = 0
    kept: bool = False


@dataclass
class AlignResult:
    ok: bool = False
    delta: float = 0.0            # 全局偏移（音频 t ↔ 原曲 t + delta）
    z: float = 0.0                # 全局二项 z
    hit_rate: float = 0.0         # 行首命中人声 onset 的比例
    null_rate: float = 0.0        # 零假设命中率（= onset 膨胀覆盖率，随机平移的期望）
    coverage: float = 0.0         # 落进音频窗口的乐句比例（按最终分块映射算）
    n_lines_in: int = 0
    spliced: bool = False
    n_blocks: int = 0
    n_blocks_kept: int = 0
    blocks: list = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "blocks"}
        d["blocks"] = [asdict(b) if not isinstance(b, dict) else b for b in self.blocks]
        return d


def split_blocks(lines: Sequence[PhraseLine], gap: float = BLOCK_GAP) -> list[Block]:
    """按 LRC 的段落间隙把行切成块。"""
    if not lines:
        return []
    cuts = [0]
    for i in range(1, len(lines)):
        if lines[i].t - lines[i - 1].t_end >= gap:
            cuts.append(i)
    cuts.append(len(lines))
    return [Block(i0=a, i1=b) for a, b in zip(cuts, cuts[1:]) if b > a]


def _hits_at(starts: np.ndarray, grid: np.ndarray, delta: float,
             hz: float = ALIGN_HZ) -> tuple[int, int]:
    """给定偏移，返回 (命中的行首数, 映射进音频窗口的行首数)。"""
    idx = np.round((starts - delta) * hz).astype(int)
    inw = (idx >= 0) & (idx < grid.size)
    if not inw.any():
        return 0, 0
    return int(grid[idx[inw]].sum()), int(inw.sum())


def _pooled_z(hits: int, n: int, cover: float) -> float:
    if n <= 0:
        return -np.inf
    c = min(max(float(cover), 1e-6), 1 - 1e-6)
    return float((hits - n * c) / np.sqrt(n * c * (1 - c)))


def align_lines(lines: Sequence[PhraseLine], onsets: Sequence[float], audio_dur: float,
                hz: float = ALIGN_HZ, tol: float = HIT_TOL,
                min_z: float = 5.0, min_rate: float = 0.55, min_lines: int = 10,
                block_min_z: float = 3.0, block_min_rate: float = 0.50
                ) -> AlignResult:
    """把 LRC 行对齐到游戏剪辑版音频（行首脉冲 × 人声 onset，二项 z 打分）。

    返回的 `z` 是**汇总二项 z**：把最终保留的全部行首放在一起，与零假设命中率
    `null_rate`（= onset 膨胀后的时间覆盖率，也就是随机平移的期望命中率）比。
    """
    res = AlignResult()
    on = np.asarray(onsets, dtype=float)
    if not lines or on.size < 20 or audio_dur <= 5:
        res.reason = "no_data"
        return res
    grid = _dilated_onset_grid(on, audio_dur, hz=hz, tol=tol)
    cover = float(grid.mean())
    res.null_rate = cover
    starts = np.array([l.t for l in lines], dtype=float)

    # ---- 方案 A：单一全局偏移 ----
    deltas, z_all, _rate = scan_offsets(starts, grid, hz=hz)
    k = int(np.argmax(z_all))
    d_glob = float(deltas[k])
    h_g, n_g = _hits_at(starts, grid, d_glob, hz)
    z_g = _pooled_z(h_g, n_g, cover)

    # ---- 方案 B：按 LRC 段落切块，每块独立偏移（剪接 / 被剪掉的段）----
    blocks = split_blocks(lines)
    res.n_blocks = len(blocks)
    for blk in blocks:
        st = starts[blk.i0:blk.i1]
        blk.delta = d_glob
        if st.size < MIN_BLOCK_LINES:
            blk.z, blk.hit_rate = float("nan"), float("nan")
            continue
        d2, z2, r2 = scan_offsets(st, grid, hz=hz)
        # 只在「块内至少映进一半行」的偏移上取最优，避免只对上 1–2 行的假峰
        nmin = max(MIN_BLOCK_LINES, int(np.ceil(0.5 * st.size)))
        nmap = _xcorr_full(np.ones(grid.size), _pulse_grid(st, hz=hz))
        ok = nmap >= nmin
        if not ok.any():
            blk.z, blk.hit_rate = float("nan"), float("nan")
            continue
        j = int(np.argmax(np.where(ok, z2, -np.inf)))
        blk.delta, blk.z, blk.hit_rate = float(d2[j]), float(z2[j]), float(r2[j])
        blk.n_mapped = int(nmap[j])
        blk.kept = bool(blk.z >= block_min_z and blk.hit_rate >= block_min_rate)

    # 太短的块（< MIN_BLOCK_LINES 行）继承相邻保留块的偏移，命中够好才保留
    prev_d = None
    for blk in blocks:
        if blk.kept:
            prev_d = blk.delta
            continue
        if (blk.i1 - blk.i0) >= MIN_BLOCK_LINES or prev_d is None:
            continue
        st = starts[blk.i0:blk.i1]
        h, n = _hits_at(st, grid, prev_d, hz)
        if n == (blk.i1 - blk.i0) and n > 0 and (h / n) >= min(0.999, cover + 0.25):
            blk.delta, blk.hit_rate, blk.n_mapped, blk.kept = prev_d, h / n, n, True

    # 单调性：块在音频上的顺序必须与在原曲上的顺序一致，否则丢掉违例的块
    last = -1e9
    for blk in blocks:
        if not blk.kept:
            continue
        t_audio = starts[blk.i0] - blk.delta
        if t_audio < last - 0.5:
            blk.kept = False
        else:
            last = t_audio

    kept = [b for b in blocks if b.kept]
    h_b = n_b = 0
    for b in kept:
        h, n = _hits_at(starts[b.i0:b.i1], grid, b.delta, hz)
        h_b += h
        n_b += n
    z_b = _pooled_z(h_b, n_b, cover)

    # ---- 择优：分块方案要明显更好才用（多一个自由度）----
    use_blocks = bool(kept) and z_b >= z_g
    if use_blocks:
        res.blocks = blocks
        res.delta = float(kept[0].delta)
        res.z, res.hit_rate = z_b, (h_b / n_b if n_b else 0.0)
        res.n_lines_in = n_b
        res.spliced = bool(len({round(b.delta, 2) for b in kept}) > 1)
        res.n_blocks_kept = len(kept)
    else:
        for b in blocks:
            b.delta, b.kept = d_glob, True
        res.blocks = blocks
        res.delta, res.z = d_glob, z_g
        res.hit_rate = (h_g / n_g) if n_g else 0.0
        res.n_lines_in = n_g
        res.spliced = False
        res.n_blocks_kept = len(blocks)
    res.coverage = res.n_lines_in / max(1, len(lines))

    res.ok = bool(res.z >= min_z and res.hit_rate >= min_rate
                  and res.n_lines_in >= min_lines)
    if not res.ok:
        res.reason = ("low_z" if res.z < min_z else
                      "low_hit_rate" if res.hit_rate < min_rate else "too_few_lines")
    return res


def lyric_to_audio(t_lyric: float, al: AlignResult, line_idx: int | None = None
                   ) -> float | None:
    """原曲时钟 → 音频时钟。给了行号就用该行所属块的偏移；块没被保留则返回 None。"""
    if line_idx is not None and al.blocks:
        for b in al.blocks:
            bi0 = b.i0 if not isinstance(b, dict) else b["i0"]
            bi1 = b.i1 if not isinstance(b, dict) else b["i1"]
            if bi0 <= line_idx < bi1:
                kept = b.kept if not isinstance(b, dict) else b["kept"]
                d = b.delta if not isinstance(b, dict) else b["delta"]
                return float(t_lyric - d) if kept else None
        return None
    return float(t_lyric - al.delta)

# --------------------------------------------------------------------------
# 4. 乐句表（映射到小节/拍网格）
# --------------------------------------------------------------------------

@dataclass
class Phrase:
    """对齐后的一条乐句（音频/谱面时钟）。"""

    line: int
    t_start: float
    t_end: float
    bar_start: float          # 小数小节（1 起，整数部分=小节号，小数=小节内位置）
    bar_end: float
    beat_start: float         # 绝对拍（从第 1 小节第 1 拍起，四分音符 = 1）
    beat_end: float
    dur_beats: float
    n_chars: int
    syl_est: float


def _abs_beat(grid, t: float) -> float:
    """秒 → 绝对拍（含变速）。"""
    bar, pos = grid.time_to_bar_pos(t)          # pos ∈ [0,1) 小节内比例
    return float((bar - 1) * grid.beats_per_bar + pos * grid.beats_per_bar)


def build_phrases(lines: Sequence[PhraseLine], al: AlignResult, grid,
                  audio_dur: float) -> list[Phrase]:
    """把 LRC 行映射成音频/谱面时钟上的乐句表。"""
    out: list[Phrase] = []
    for l in lines:
        ta = lyric_to_audio(l.t, al, line_idx=l.idx)
        tb = lyric_to_audio(l.t_end, al, line_idx=l.idx)
        if ta is None or tb is None or tb <= ta:
            continue
        if ta < -0.5 or ta > audio_dur:
            continue
        tb = min(tb, audio_dur)
        bs, be = _abs_beat(grid, max(0.0, ta)), _abs_beat(grid, max(0.0, tb))
        out.append(Phrase(
            line=l.idx, t_start=float(ta), t_end=float(tb),
            bar_start=1.0 + bs / grid.beats_per_bar,
            bar_end=1.0 + be / grid.beats_per_bar,
            beat_start=bs, beat_end=be, dur_beats=be - bs,
            n_chars=l.n_chars, syl_est=l.syl_est))
    return out


# --------------------------------------------------------------------------
# 5. 音频侧：人声 onset / VAD
# --------------------------------------------------------------------------

def vocal_onsets(song_dir: Path) -> tuple[np.ndarray, float]:
    """`stems/vocals.wav` 的能量 onset（与管线同参：`onsets.detect_onsets`）+ 时长。"""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.audio_analysis import onsets as onsets_mod
    from tools.audio_analysis import stems as stems_mod

    p = Path(song_dir) / "stems" / "vocals.wav"
    if not p.exists():
        return np.zeros(0), 0.0
    y, _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
    dur = float(y.size) / onsets_mod.ANALYSIS_SR
    t = onsets_mod.detect_onsets(y, "vocals", sr=onsets_mod.ANALYSIS_SR,
                                 hop=onsets_mod.HOP).times
    return np.asarray(t, dtype=float), dur


def vocal_vad(song_dir: Path, rel_db: float = 32.0) -> dict:
    """`stems/vocals.wav` 的逐帧 VAD（`onsets.vocal_activity`，探测器与 T2 用）。"""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.audio_analysis import onsets as onsets_mod
    from tools.audio_analysis import stems as stems_mod

    p = Path(song_dir) / "stems" / "vocals.wav"
    if not p.exists():
        return {"active": np.zeros(0, dtype=bool), "times": np.zeros(0)}
    y, _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
    return onsets_mod.vocal_activity(y, sr=onsets_mod.ANALYSIS_SR,
                                     hop=onsets_mod.HOP, rel_db=rel_db)


def load_lyrics(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines_of(cand: dict) -> list[PhraseLine]:
    return [PhraseLine(**d) for d in cand.get("lines", [])]


def align_song(song_dir: Path, rec: dict, onsets: np.ndarray | None = None,
               dur: float | None = None, **kw
               ) -> tuple[AlignResult, dict | None]:
    """在该曲的全部 LRC 候选里，用音频波形裁决出最佳候选并给出对齐结果。

    ⚠️ **候选裁决交给音频**：曲名检索单独不可靠（"ARROW" 这种通用词能撞上完全无关的曲子），
    所以这里对每个候选都跑一遍对齐，取 z 最高者 —— 对不上的候选自然会被 z 门槛挡掉。
    """
    if onsets is None or dur is None:
        onsets, dur = vocal_onsets(song_dir)
    best: tuple[AlignResult, dict | None] = (AlignResult(reason="no_vocals"), None)
    if np.size(onsets) == 0:
        return best
    best_z = -np.inf
    for c in rec.get("candidates", []):
        al = align_lines(lines_of(c), onsets, dur, **kw)
        if al.z > best_z:
            best_z, best = al.z, (al, c)
    return best


# --------------------------------------------------------------------------
# 6. 区间密度 lift + 随机平移对照（T1/T2 的公共底座）
# --------------------------------------------------------------------------

def _merge_intervals(iv: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """合并重叠区间（句头窗会互相重叠，必须按并集算时长，否则 lift 被系统性压低）。"""
    s = sorted((float(a), float(b)) for a, b in iv if b > a)
    out: list[tuple[float, float]] = []
    for a, b in s:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _count_in(times: np.ndarray, iv: Sequence[tuple[float, float]]) -> int:
    if times.size == 0 or not iv:
        return 0
    m = np.zeros(times.size, dtype=bool)
    for a, b in iv:
        m |= (times >= a) & (times < b)
    return int(m.sum())


def _wrap(iv: Sequence[tuple[float, float]], shift: float, lo: float, hi: float
          ) -> list[tuple[float, float]]:
    """把区间集整体循环平移 `shift`（在 [lo, hi) 上环绕，跨界的切成两段）。"""
    span = hi - lo
    out: list[tuple[float, float]] = []
    for a, b in iv:
        a2 = lo + ((a - lo + shift) % span)
        d = b - a
        if a2 + d <= hi:
            out.append((a2, a2 + d))
        else:
            out.append((a2, hi))
            out.append((lo, lo + (a2 + d - hi)))
    return _merge_intervals(out)


def interval_lift(times: np.ndarray, iv: Sequence[tuple[float, float]],
                  lo: float, hi: float, n_null: int = 200,
                  seed: int = 0) -> dict:
    """区间集内的 note 密度相对全曲均匀基线的 lift，附**随机循环平移对照**。

    - `lift` = （区间内 note 占比）/（区间时长占比）；1.0 = 与均匀撒点无异；
    - 对照组 = 把同一套区间（**个数、每段时长完全不变**，只换位置）在 [lo, hi) 上
      循环平移 `n_null` 次，得到零假设下的 lift 分布；
    - `p` = 对照组 lift ≥ 实测 lift 的比例（单侧）。
    """
    iv = _merge_intervals([(max(a, lo), min(b, hi)) for a, b in iv])
    span = hi - lo
    tot_iv = sum(b - a for a, b in iv)
    sel = times[(times >= lo) & (times < hi)]
    n_tot = int(sel.size)
    if span <= 0 or tot_iv <= 0 or n_tot == 0:
        return {"lift": float("nan"), "n_in": 0, "n_total": n_tot,
                "cover": 0.0, "null_mean": float("nan"), "p": float("nan")}
    n_in = _count_in(sel, iv)
    cover = tot_iv / span
    lift = (n_in / n_tot) / cover
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(int(n_null)):
        sh = float(rng.uniform(0.0, span))
        iv2 = _wrap(iv, sh, lo, hi)
        c2 = sum(b - a for a, b in iv2) / span
        if c2 <= 0:
            continue
        nulls.append((_count_in(sel, iv2) / n_tot) / c2)
    nulls_a = np.asarray(nulls, dtype=float) if nulls else np.zeros(0)
    return {
        "lift": float(lift), "n_in": n_in, "n_total": n_tot,
        "cover": float(cover),
        "null_mean": float(nulls_a.mean()) if nulls_a.size else float("nan"),
        "null_sd": float(nulls_a.std()) if nulls_a.size else float("nan"),
        "z": (float((lift - nulls_a.mean()) / nulls_a.std())
              if nulls_a.size and nulls_a.std() > 1e-9 else float("nan")),
        "p": (float((nulls_a >= lift).mean()) if nulls_a.size else float("nan")),
    }


# --------------------------------------------------------------------------
# 7. T1–T4
# --------------------------------------------------------------------------

def _local_beat_sec(grid, t: float) -> float:
    bar = grid.bar_of(t)
    bar = min(max(bar, 1), grid.n_bars)
    return 60.0 / float(grid.bar_bpm(bar))


def head_windows(phrases: Sequence[Phrase], grid, half_beats: float = 0.5
                 ) -> list[tuple[float, float]]:
    """乐句起点 ±`half_beats` 拍的窗口（拍长按该处的局部 BPM 取，变速曲也对）。"""
    out = []
    for p in phrases:
        w = half_beats * _local_beat_sec(grid, p.t_start)
        out.append((p.t_start - w, p.t_start + w))
    return out


def gap_intervals(phrases: Sequence[Phrase], grid, min_beats: float = 0.5
                  ) -> list[tuple[float, float]]:
    """乐句之间的**换气间隙**（上一句结束 → 下一句开始，且长度 ≥ `min_beats` 拍）。"""
    out = []
    for a, b in zip(phrases, phrases[1:]):
        if b.t_start <= a.t_end:
            continue
        if (b.t_start - a.t_end) >= min_beats * _local_beat_sec(grid, a.t_end):
            out.append((a.t_end, b.t_start))
    return out


def inside_intervals(phrases: Sequence[Phrase]) -> list[tuple[float, float]]:
    return _merge_intervals([(p.t_start, p.t_end) for p in phrases])


def note_arrays(bundle) -> dict:
    """按类别切出 note 的绝对秒时间数组。"""
    first = float(bundle.first)
    notes = bundle.parse.notes
    t = np.array([first + n.time for n in notes], dtype=float)
    kinds = np.array([n.kind for n in notes])
    brk = np.array([bool(n.is_break) for n in notes])
    each = np.array([bool(n.is_each) for n in notes])
    return {
        "all": np.sort(t),
        "slide_start": np.sort(t[kinds == "slide_star"]),
        "break": np.sort(t[brk]),
        "each": np.sort(t[each]),
        "tap": np.sort(t[kinds == "tap"]),
        "times": t, "kinds": kinds, "is_break": brk, "is_each": each,
    }


def t1_phrase_head(bundle, phrases: Sequence[Phrase], na: dict,
                   lo: float, hi: float, n_null: int = 200, seed: int = 0) -> dict:
    """T1 句头效应：官方 note（及 slide 起点 / BREAK）在乐句起点 ±半拍窗内的 lift。"""
    iv = head_windows(phrases, bundle.grid, 0.5)
    out = {"window_half_beats": 0.5,
           "all": interval_lift(na["all"], iv, lo, hi, n_null, seed)}
    for k in ("slide_start", "break", "each", "tap"):
        out[k] = interval_lift(na[k], iv, lo, hi, n_null, seed + 1)
    # 更紧的窗（±1/4 拍）作稳健性对照
    iv4 = head_windows(phrases, bundle.grid, 0.25)
    out["all_quarter"] = interval_lift(na["all"], iv4, lo, hi, n_null, seed + 2)
    return out


def pre_phrase_windows(phrases: Sequence[Phrase], grid,
                       a_beats: float = 1.5, b_beats: float = 0.5
                       ) -> list[tuple[float, float]]:
    """句前留白窗 `[start − a_beats, start − b_beats]`（换气发生的位置）。

    ⚠️ 这个窗**只用行首时间戳**，不依赖 LRC 的行尾——因为公网 LRC 绝大多数
    首尾相接（逐行间隙中位数 0.00 s），行尾不是真正的"唱完了"。
    """
    out = []
    for p in phrases:
        w = _local_beat_sec(grid, p.t_start)
        out.append((p.t_start - a_beats * w, p.t_start - b_beats * w))
    return out


def t2_breath_gap(bundle, phrases: Sequence[Phrase], na: dict,
                  lo: float, hi: float, n_null: int = 200, seed: int = 0) -> dict:
    """T2 换气留白。两个口径：

    - **T2a 真间隙**：LRC 里显式存在的乐句间隙（空行 / 段落分隔，≥ 0.5 拍）内的
      note 密度 lift。⚠️ 大多数公网 LRC 首尾相接，所以"真间隙"实际只有段落级的
      器乐间隙，曲内换气拿不到；
    - **T2b 句前窗**：`[句首 − 1.5 拍, 句首 − 0.5 拍]`，只用行首时间戳，
      直接量"开口前那一小段有没有留白"。
    """
    gaps = gap_intervals(phrases, bundle.grid, 0.5)
    ins = inside_intervals(phrases)
    g = interval_lift(na["all"], gaps, lo, hi, n_null, seed)
    i = interval_lift(na["all"], ins, lo, hi, n_null, seed + 1)
    ratio = (g["lift"] / i["lift"]) if (i["lift"] and np.isfinite(i["lift"])
                                        and i["lift"] > 1e-9) else float("nan")
    gl = gap_intervals(phrases, bundle.grid, 2.0)
    pre = interval_lift(na["all"], pre_phrase_windows(phrases, bundle.grid),
                        lo, hi, n_null, seed + 3)
    return {"gap": g, "inside": i, "gap_over_inside": float(ratio),
            "n_gaps": len(gaps), "gap_sec": sum(b - a for a, b in gaps),
            "long_gap": interval_lift(na["all"], gl, lo, hi, n_null, seed + 2),
            "n_long_gaps": len(gl),
            "pre_phrase": pre}


def syllables_per_bar(phrases: Sequence[Phrase], grid, n_bars: int) -> np.ndarray:
    """逐小节音节数（每句的 `syl_est` 按时长均匀摊到它覆盖的小节上）。"""
    out = np.zeros(int(n_bars), dtype=float)
    for p in phrases:
        dur = max(p.t_end - p.t_start, 1e-6)
        rate = p.syl_est / dur
        b0 = max(1, grid.bar_of(p.t_start))
        b1 = min(int(n_bars), grid.bar_of(p.t_end))
        for b in range(b0, b1 + 1):
            s = grid.bar_start(b)
            e = s + grid.bar_duration(b)
            ov = min(e, p.t_end) - max(s, p.t_start)
            if ov > 0:
                out[b - 1] += rate * ov
    return out


def phrase_starts_per_bar(phrases: Sequence[Phrase], grid, n_bars: int) -> np.ndarray:
    out = np.zeros(int(n_bars), dtype=float)
    for p in phrases:
        b = grid.bar_of(p.t_start)
        if 1 <= b <= n_bars:
            out[b - 1] += 1.0
    return out


def _ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """带截距的最小二乘 R²（X 为 n×k 设计矩阵，不含截距列）。"""
    y = np.asarray(y, dtype=float)
    if X.size == 0:
        return 0.0
    A = np.column_stack([np.ones(y.size), X])
    try:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    resid = y - A @ beta
    ss_t = float(((y - y.mean()) ** 2).sum())
    if ss_t < 1e-12:
        return float("nan")
    return float(1.0 - float((resid ** 2).sum()) / ss_t)


def t3_syllable_density(bundle, phrases: Sequence[Phrase]) -> dict:
    """T3 音节密度 vs 谱面密度：逐小节 Spearman，并与音频强度 ρ 并排 + 联合回归增量。"""
    from . import chartpair as cp

    grid = bundle.grid
    stats = bundle.density.measures
    measure_starts = [float(bundle.first) + s.start_time for s in stats]
    bar_starts = [grid.bar_start(b) for b in range(1, grid.n_bars + 1)]
    m2b = cp.map_measures_to_bars(measure_starts, bar_starts, tol_sec=0.05)
    idx = sorted(m2b)
    if len(idx) < 8:
        return {"n_bars": len(idx)}
    bars = [m2b[i] for i in idx]
    d_bar = np.array([stats[i].notes for i in idx], dtype=float)
    I = np.array([bundle.bar_intensity[b - 1] for b in bars], dtype=float)
    I_raw = np.array([bundle.bar_intensity_raw[b - 1] for b in bars], dtype=float)
    syl_full = syllables_per_bar(phrases, grid, grid.n_bars)
    ps_full = phrase_starts_per_bar(phrases, grid, grid.n_bars)
    syl = np.array([syl_full[b - 1] for b in bars], dtype=float)
    ps = np.array([ps_full[b - 1] for b in bars], dtype=float)

    def z(v):
        v = np.asarray(v, dtype=float)
        s = v.std()
        return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)

    r_int = _ols_r2(z(d_bar), z(I)[:, None])
    r_syl = _ols_r2(z(d_bar), z(syl)[:, None])
    r_both = _ols_r2(z(d_bar), np.column_stack([z(I), z(syl)]))
    r_all3 = _ols_r2(z(d_bar), np.column_stack([z(I), z(syl), z(ps)]))
    voiced = syl > 1e-9
    res_v = {}
    if int(voiced.sum()) >= 8:
        res_v = {
            "n_bars_voiced": int(voiced.sum()),
            "rho_syllable_voiced": cp.spearman(syl[voiced], d_bar[voiced]),
            "rho_intensity_voiced": cp.spearman(I[voiced], d_bar[voiced]),
            "r2_intensity_voiced": _ols_r2(z(d_bar[voiced]), z(I[voiced])[:, None]),
            "r2_both_voiced": _ols_r2(
                z(d_bar[voiced]),
                np.column_stack([z(I[voiced]), z(syl[voiced])])),
        }
    return {
        "n_bars": len(idx),
        **res_v,
        "rho_syllable": cp.spearman(syl, d_bar),
        "rho_phrase_starts": cp.spearman(ps, d_bar),
        "rho_intensity": cp.spearman(I, d_bar),
        "rho_intensity_raw": cp.spearman(I_raw, d_bar),
        "rho_syl_vs_intensity": cp.spearman(syl, I),
        "r2_intensity": r_int, "r2_syllable": r_syl,
        "r2_both": r_both, "r2_all3": r_all3,
        "r2_gain_syllable": (r_both - r_int) if np.isfinite(r_both) else float("nan"),
        "bars": bars, "d_bar": d_bar.tolist(), "syl": syl.tolist(),
        "I": I.tolist(), "phrase_starts": ps.tolist(),
    }


def t4_phrase_shape(bundle, phrases: Sequence[Phrase], na: dict) -> dict:
    """T4 句长/字数 与 句内 note 数、each/slide 占比（只做描述统计）。"""
    from . import chartpair as cp

    t = na["times"]
    kinds, each = na["kinds"], na["is_each"]
    rows = []
    for p in phrases:
        sel = (t >= p.t_start) & (t < p.t_end)
        n = int(sel.sum())
        if n == 0:
            rows.append({"dur_beats": p.dur_beats, "syl": p.syl_est, "n_notes": 0,
                         "nps_beat": 0.0, "each_share": float("nan"),
                         "slide_share": float("nan"),
                         "notes_per_syl": 0.0})
            continue
        rows.append({
            "dur_beats": float(p.dur_beats), "syl": float(p.syl_est), "n_notes": n,
            "nps_beat": n / max(p.dur_beats, 1e-6),
            "each_share": float(each[sel].mean()),
            "slide_share": float((kinds[sel] == "slide_star").mean()),
            "notes_per_syl": n / max(p.syl_est, 1e-6),
        })
    if len(rows) < 4:
        return {"n_phrases": len(rows)}
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    ok = np.isfinite(arr["each_share"])
    return {
        "n_phrases": len(rows),
        "rho_dur_notes": cp.spearman(arr["dur_beats"], arr["n_notes"]),
        "rho_syl_notes": cp.spearman(arr["syl"], arr["n_notes"]),
        "rho_dur_nps": cp.spearman(arr["dur_beats"], arr["nps_beat"]),
        "rho_syl_nps": cp.spearman(arr["syl"], arr["nps_beat"]),
        "rho_dur_each": (cp.spearman(arr["dur_beats"][ok], arr["each_share"][ok])
                         if ok.sum() >= 4 else float("nan")),
        "rho_dur_slide": (cp.spearman(arr["dur_beats"][ok], arr["slide_share"][ok])
                          if ok.sum() >= 4 else float("nan")),
        "mean_notes_per_syl": float(np.median(arr["notes_per_syl"])),
        "mean_dur_beats": float(np.median(arr["dur_beats"])),
    }


def chorus_spans(bundle) -> list[tuple[float, float]]:
    """从 `song_analysis.json` 的结构分段里取副歌段的秒区间。"""
    out = []
    for s in bundle.segments:
        fn = str(s.get("function") or "")
        if "chorus" in fn and "quiet" not in fn:
            t0, t1 = s.get("start_sec"), s.get("end_sec")
            if t0 is None:
                b0, b1 = s.get("start_bar"), s.get("end_bar")
                if b0 is None:
                    continue
                t0 = bundle.grid.bar_start(int(b0))
                t1 = bundle.grid.bar_start(int(b1)) + bundle.grid.bar_duration(int(b1))
            out.append((float(t0), float(t1)))
    return out


def run_tests(bundle, phrases: Sequence[Phrase], n_null: int = 200,
              seed: int = 0) -> dict:
    """对一首对齐成功的曲子跑 T1–T4。"""
    na = note_arrays(bundle)
    if na["all"].size < 30 or len(phrases) < 6:
        return {"error": "too_few"}
    lo = float(na["all"][0])
    hi = float(na["all"][-1]) + 1e-3
    out: dict = {
        "n_phrases": len(phrases),
        "note_span_sec": hi - lo,
        "T1": t1_phrase_head(bundle, phrases, na, lo, hi, n_null, seed),
        "T2": t2_breath_gap(bundle, phrases, na, lo, hi, n_null, seed + 10),
        "T3": t3_syllable_density(bundle, phrases),
        "T4": t4_phrase_shape(bundle, phrases, na),
    }
    # 副歌 / 非副歌分层（T1 的句头 lift）
    ch = chorus_spans(bundle)
    if ch:
        in_ch = [p for p in phrases
                 if any(a <= p.t_start < b for a, b in ch)]
        out_ch = [p for p in phrases if p not in in_ch]
        if len(in_ch) >= 4:
            out["T1_chorus"] = interval_lift(
                na["all"], head_windows(in_ch, bundle.grid, 0.5), lo, hi,
                n_null, seed + 20)
        if len(out_ch) >= 4:
            out["T1_nonchorus"] = interval_lift(
                na["all"], head_windows(out_ch, bundle.grid, 0.5), lo, hi,
                n_null, seed + 21)
    return out


# --------------------------------------------------------------------------
# 8. 逐曲流程 + 跨曲对照
# --------------------------------------------------------------------------

def process_song(song_dir: Path, rec: dict, n_null: int = 200,
                 seed: int = 0, **align_kw) -> dict:
    """抓好 LRC 的一首：对齐 → 乐句表 → T1–T4。"""
    from . import loader

    song_dir = Path(song_dir)
    onsets, dur = vocal_onsets(song_dir)
    al, cand = align_song(song_dir, rec, onsets=onsets, dur=dur, **align_kw)
    out: dict = {"song": song_dir.name, "align": al.to_dict(),
                 "source": (cand or {}).get("source"),
                 "lrc_id": (cand or {}).get("id"),
                 "match_score": (cand or {}).get("score"),
                 "n_lines": (cand or {}).get("n_lines"),
                 "audio_dur": round(float(dur), 2),
                 "n_vocal_onsets": int(np.size(onsets))}
    if not al.ok or cand is None:
        out["ok"] = False
        return out
    bundle = loader.load_song(song_dir, recompute_intensity=False, with_v5=False)
    phrases = build_phrases(lines_of(cand), al, bundle.grid, dur)
    out["ok"] = True
    out["genre"] = bundle.genre
    out["level"] = bundle.level
    out["bpm"] = bundle.bpm
    out["n_phrases_mapped"] = len(phrases)
    out["tests"] = run_tests(bundle, phrases, n_null=n_null, seed=seed)
    out["phrases"] = [asdict(p) for p in phrases]
    return out


def cross_song_control(song_dirs: Sequence[Path], recs: Sequence[dict],
                       n_pairs: int = 200, seed: int = 0, **align_kw) -> dict:
    """**跨曲阴性对照**：把 A 曲的 LRC 拿去对 B 曲的音频，统计门槛的假阳性率。

    这是整个对齐环节唯一能给出"置信度到底值不值钱"的证据：如果错配也能过门槛，
    那所有下游检验都不成立。
    """
    rng = np.random.default_rng(seed)
    cache: dict[str, tuple[np.ndarray, float]] = {}
    n = len(song_dirs)
    zs, ok = [], 0
    tried = 0
    while tried < n_pairs and n >= 2:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        d = Path(song_dirs[i])
        if d.name not in cache:
            cache[d.name] = vocal_onsets(d)
        on, dur = cache[d.name]
        if np.size(on) == 0:
            tried += 1
            continue
        cands = recs[j].get("candidates") or []
        if not cands:
            tried += 1
            continue
        al = align_lines(lines_of(cands[0]), on, dur, **align_kw)
        zs.append(al.z)
        ok += int(al.ok)
        tried += 1
    za = np.asarray([z for z in zs if np.isfinite(z)], dtype=float)
    return {"n_pairs": len(zs), "n_pass_gate": ok,
            "false_positive_rate": (ok / len(zs)) if zs else float("nan"),
            "z_mean": float(za.mean()) if za.size else float("nan"),
            "z_p95": float(np.percentile(za, 95)) if za.size else float("nan"),
            "z_max": float(za.max()) if za.size else float("nan")}


# --------------------------------------------------------------------------
# 9. 免对齐口径：纯音频句读特征 × 官方谱
# --------------------------------------------------------------------------
#
# 为什么要有这一路：LRC 对齐在真实数据上被证伪（§4 的跨曲阴性对照），
# 但**假设本身**还可以用不依赖歌词的方式检验——把「句读边界」换成
# `tools/audio_analysis/phrases.py` 的**纯音频估计**，再对官方谱做同一套
# lift + 随机平移对照。它回答的是工具侧真正关心的问题：
# **把句读特征加进管线，对预测官方密度有没有增量。**

def audio_phrase_tests(bundle, vad: dict | None = None, n_null: int = 200,
                       seed: int = 0, gap_beats: float = 2.0,
                       min_sep_beats: float = 2.0) -> dict:
    """纯音频句读特征 × 官方谱：句头 lift / 留白 lift / 逐小节增量 R²。"""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.audio_analysis import phrases as ph_mod

    from . import chartpair as cp

    grid = bundle.grid
    onsets = bundle.onset_times.get("vocals")
    if onsets is None or np.size(onsets) < 10:
        return {"error": "no_vocal_onsets"}
    det = ph_mod.detect_phrases(onsets, vad, grid, gap_beats=gap_beats,
                               min_sep_beats=min_sep_beats)
    na = note_arrays(bundle)
    if na["all"].size < 30 or det.starts.size < 6:
        return {"error": "too_few"}
    lo, hi = float(na["all"][0]), float(na["all"][-1]) + 1e-3

    heads = []
    for t in det.starts:
        w = 0.5 * _local_beat_sec(grid, float(t))
        heads.append((float(t) - w, float(t) + w))
    out = {
        "n_detected": int(det.starts.size),
        "n_gaps": len(det.gaps),
        "head": interval_lift(na["all"], heads, lo, hi, n_null, seed),
        "head_slide": interval_lift(na["slide_start"], heads, lo, hi, n_null, seed + 1),
        "head_break": interval_lift(na["break"], heads, lo, hi, n_null, seed + 2),
        "gap": interval_lift(na["all"], det.gaps, lo, hi, n_null, seed + 3),
    }

    # 逐小节：句读特征对官方密度的增量
    stats = bundle.density.measures
    measure_starts = [float(bundle.first) + s.start_time for s in stats]
    bar_starts = [grid.bar_start(b) for b in range(1, grid.n_bars + 1)]
    m2b = cp.map_measures_to_bars(measure_starts, bar_starts, tol_sec=0.05)
    idx = sorted(m2b)
    if len(idx) >= 8:
        bars = [m2b[i] for i in idx]
        d_bar = np.array([stats[i].notes for i in idx], dtype=float)
        I = np.array([bundle.bar_intensity[b - 1] for b in bars], dtype=float)
        f = ph_mod.phrase_features_per_bar(det, grid)
        ps = np.array([f["phrase_start"][b - 1] for b in bars], dtype=float)
        ig = np.array([f["in_gap"][b - 1] for b in bars], dtype=float)

        def z(v):
            v = np.asarray(v, dtype=float)
            s = v.std()
            return (v - v.mean()) / s if s > 1e-9 else np.zeros_like(v)

        r_i = _ols_r2(z(d_bar), z(I)[:, None])
        r_ip = _ols_r2(z(d_bar), np.column_stack([z(I), z(ps), z(ig)]))
        out["bars"] = {
            "n_bars": len(idx),
            "rho_intensity": cp.spearman(I, d_bar),
            "rho_phrase_start": cp.spearman(ps, d_bar),
            "rho_in_gap": cp.spearman(ig, d_bar),
            "r2_intensity": r_i,
            "r2_intensity_plus_phrase": r_ip,
            "r2_gain": (r_ip - r_i) if np.isfinite(r_ip) else float("nan"),
        }
    return out


# --------------------------------------------------------------------------
# 10. CLI
# --------------------------------------------------------------------------

def _iter_songs(calib_dir: Path) -> list[Path]:
    from . import loader

    return loader.discover(Path(calib_dir))


def _song_meta(song_dir: Path) -> dict:
    from . import loader

    return loader.read_maidata_header(Path(song_dir) / "maidata.txt")


def cmd_fetch(args) -> int:
    """抓 LRC，**只落时间戳与计数**到 `--out`（默认 `out/lyrics/`，已 gitignore）。"""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = _iter_songs(args.calib_dir)
    done = 0
    for i, d in enumerate(dirs, 1):
        p = out_dir / f"{d.name}.json"
        if p.exists() and not args.force:
            continue
        meta = _song_meta(d)
        rec = fetch_one(meta.get("title", d.name), meta.get("artist", ""),
                        sleep=args.sleep, allow_netease=not args.no_netease)
        rec.update({"dir": d.name, "title": meta.get("title", ""),
                    "artist": meta.get("artist", ""), "genre": meta.get("genre", "")})
        p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"[{i}/{len(dirs)}] {d.name}: found={rec.get('found')} "
              f"n_cand={rec.get('n_candidates', 0)}")
    print(f"抓取完成：新增 {done} 首，目录 {out_dir}")
    return 0


def cmd_align(args) -> int:
    """逐曲对齐并打印诊断（不跑假设检验）。"""
    rows = []
    for p in sorted(Path(args.lyrics).glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if not rec.get("found"):
            continue
        d = Path(args.calib_dir) / rec["dir"]
        if not d.exists():
            continue
        on, dur = vocal_onsets(d)
        al, cand = align_song(d, rec, onsets=on, dur=dur)
        rows.append((rec["dir"], al, cand))
        print(f"{rec['dir'][:26]:28s} ok={str(al.ok):5s} z={al.z:6.2f} "
              f"hit={al.hit_rate:.2f} null={al.null_rate:.2f} "
              f"delta={al.delta:8.2f} n_in={al.n_lines_in:3d} spliced={al.spliced}")
    n_ok = sum(1 for _, a, _ in rows if a.ok)
    print(f"\n对齐通过 {n_ok}/{len(rows)}"
          f"  ⚠️ 通过门槛不等于对齐正确，见 control 子命令的跨曲阴性对照")
    return 0


def cmd_control(args) -> int:
    """跨曲阴性对照：A 曲的 LRC × B 曲的音频。"""
    dirs, recs = [], []
    for p in sorted(Path(args.lyrics).glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        d = Path(args.calib_dir) / rec["dir"]
        if rec.get("found") and d.exists():
            dirs.append(d)
            recs.append(rec)
    res = cross_song_control(dirs, recs, n_pairs=args.n_pairs, seed=args.seed)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_test(args) -> int:
    """跑免对齐口径（纯音频句读 × 官方谱）并写 CSV / metrics。"""
    from . import loader

    rows = []
    metrics = {}
    dirs = _iter_songs(args.calib_dir)
    for i, d in enumerate(dirs, 1):
        try:
            on, dur = vocal_onsets(d)
            vad = vocal_vad(d)
            b = loader.load_song(d, recompute_intensity=False, with_v5=False)
            b.onset_times["vocals"] = on
            res = audio_phrase_tests(b, vad=vad, n_null=args.n_null, seed=i)
        except Exception as e:                       # noqa: BLE001
            print(f"{d.name}: {e}")
            continue
        metrics[d.name] = res
        bars = res.get("bars") or {}
        rows.append({
            "song": d.name, "genre": b.genre, "level": b.level, "bpm": b.bpm,
            "n_detected": res.get("n_detected"), "n_gaps": res.get("n_gaps"),
            "head_lift": (res.get("head") or {}).get("lift"),
            "head_p": (res.get("head") or {}).get("p"),
            "gap_lift": (res.get("gap") or {}).get("lift"),
            "rho_intensity": bars.get("rho_intensity"),
            "rho_phrase_start": bars.get("rho_phrase_start"),
            "r2_gain": bars.get("r2_gain"),
        })
        print(f"[{i}/{len(dirs)}] {d.name}: head_lift="
              f"{(res.get('head') or {}).get('lift')}")
    if not rows:
        print("没有任何曲目跑通，检查 --calib-dir")
        return 1
    if args.csv:
        import csv as _csv

        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if args.metrics:
        Path(args.metrics).write_text(
            json.dumps(metrics, ensure_ascii=False, default=float), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m tools.calibration.lyrics_phrases",
        description="LRC 歌词时间轴 → 乐句边界 → 官方谱句读假设检验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="抓 LRC（只落时间戳与计数）")
    f.add_argument("--calib-dir", default="out/calib")
    f.add_argument("--out", default="out/lyrics")
    f.add_argument("--sleep", type=float, default=1.0, help="每次请求之间的间隔秒数")
    f.add_argument("--no-netease", action="store_true")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("align", help="逐曲对齐诊断")
    a.add_argument("--calib-dir", default="out/calib")
    a.add_argument("--lyrics", default="out/lyrics")
    a.set_defaults(func=cmd_align)

    c = sub.add_parser("control", help="跨曲阴性对照")
    c.add_argument("--calib-dir", default="out/calib")
    c.add_argument("--lyrics", default="out/lyrics")
    c.add_argument("--n-pairs", type=int, default=250)
    c.add_argument("--seed", type=int, default=7)
    c.set_defaults(func=cmd_control)

    t = sub.add_parser("test", help="免对齐口径：纯音频句读 × 官方谱")
    t.add_argument("--calib-dir", default="out/calib")
    t.add_argument("--lyrics", default="out/lyrics")
    t.add_argument("--csv", default="")
    t.add_argument("--metrics", default="")
    t.add_argument("--n-null", type=int, default=200)
    t.set_defaults(func=cmd_test)

    args = ap.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
