"""装载一首标定曲：音频侧（`tools/audio_analysis` 的产物）+ 谱面侧（官方 maidata）。

目录约定（**均在 `out/` 下，不入库**）：

```
out/calib/<曲名>/
  maidata.txt          官方谱（含 &first/&wholebpm/&lv_5/&inote_2..5）
  track.mp3            官方音频
  track.44k.wav        管线统一解码产物
  stems/{drums,bass,other,vocals}.wav
  song_analysis.json   管线输出
```

onset 时间在 `song_analysis.json` 里**没有**逐条保存（只有计数），所以这里从
`stems/*.wav` 重新跑一遍 `onsets.detect_onsets` —— 纯 DSP，不重跑 Demucs，
参数与管线完全一致，因此结果与管线内部逐条对应。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STEM_ORDER = ("drums", "bass", "other", "vocals")
#: v0.5 的六路 stem（`out/calib/<曲名>/stems_htdemucs_6s/`）
STEM_ORDER_6S = ("drums", "bass", "other", "vocals", "guitar", "piano")
#: 跑过 basic-pitch 的 stem
PITCH_STEMS = ("vocals", "other", "guitar", "piano", "bass")
INOTE_KEY = "&inote_5="          # Master 难度


def read_maidata_header(path: Path) -> dict:
    """读 maidata 头部元数据（只取标定需要的字段）。

    ``genre`` / ``artist`` / ``version`` 是 n=160 新增：官方曲包的 ``&genre``
    就是**官方六曲风分类**（maimai / niconico＆ボーカロイド / ゲーム＆バラエティ /
    東方Project / オンゲキ＆CHUNITHM / POPS＆アニメ），比任何音频侧代理都硬，
    因此 n=160 的曲风分层直接用它，不再靠人工猜。
    """
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for key, cast in (("title", str), ("first", float), ("wholebpm", float),
                      ("lv_5", float), ("genre", str), ("artist", str),
                      ("version", str)):
        m = re.search(rf"^&{key}=(.*)$", txt, flags=re.M)
        if m:
            try:
                out[key] = cast(m.group(1).strip())
            except ValueError:
                out[key] = m.group(1).strip()
    return out


def ensure_mix_wav(song_dir: Path) -> tuple[Path, bool]:
    """保证 `track.44k.wav` 存在，返回 ``(路径, 是否是本次临时解码出来的)``。

    n=160 这一轮磁盘只剩十几 GB，管线跑完每首都会删掉 22 MB 的 `track.44k.wav`，
    所以标定装载时按**同一条 ffmpeg 命令**重新解码（`decode.decode_to_wav`），
    时间基准与管线完全一致；调用方用完应当把临时文件删掉。

    ⚠️ 临时解码**不写回曲目录**，而是写到进程私有的临时目录 —— 否则与同时在跑的
    分离脚本（它也会临时解码同名文件再删掉）抢同一个路径。
    """
    song_dir = Path(song_dir)
    wav = song_dir / "track.44k.wav"
    if wav.exists():
        return wav, False
    import sys
    import tempfile

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.audio_analysis import decode as decode_mod

    src = song_dir / "track.mp3"
    if not src.exists():
        raise FileNotFoundError(f"{song_dir} 既没有 track.44k.wav 也没有 track.mp3")
    tmp = Path(tempfile.mkdtemp(prefix="calib-mix-"))
    return decode_mod.decode_to_wav(src, tmp), True


def extract_inote(path: Path, key: str = INOTE_KEY) -> str:
    """抽出 `&inote_N=` 之后、下一个 `&xxx=` 之前的谱面正文。"""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    i = txt.find(key)
    if i < 0:
        raise ValueError(f"{path} 里找不到 {key}")
    body = txt[i + len(key):]
    m = re.search(r"\n&[A-Za-z_0-9]+=", body)
    return body[:m.start()] if m else body


@dataclass
class SongBundle:
    name: str
    level: float
    bpm: float
    first: float
    genre: str = ""                   # 官方 `&genre`（n=160 曲风分层用）
    grid: object = None
    analysis: dict = field(default_factory=dict)
    parse: object = None              # chart_analysis.simai_parser.ParseResult
    density: object = None            # chart_analysis.density.ChartDensity
    onset_times: dict = field(default_factory=dict)   # {轨名: np.ndarray(秒)}
    pitch_notes: dict = field(default_factory=dict)   # {stem: PitchNotes}（v0.5）
    v5: dict = field(default_factory=dict)            # v0.5 侧的诊断信息
    components: dict = field(default_factory=dict)    # 五项强度分量（逐小节，未 Z 化）
    bar_intensity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bar_intensity_raw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    segments: list = field(default_factory=list)      # song_analysis.json 的段落 dict

    @property
    def n_bars(self) -> int:
        return int(self.grid.n_bars) if self.grid is not None else 0


def load_v5_tracks(song_dir: Path, bundle: "SongBundle",
                   stem_audio4: dict | None = None) -> None:
    """装载 v0.5 的新轨，写进 `bundle.onset_times` / `bundle.pitch_notes`。

    新轨清单（全部只在标定里比较，**不改四路口径的任何既有数字**）：

    | key | 含义 |
    |---|---|
    | `guitar` / `piano` | htdemucs_6s 新增两轨的**能量 onset** |
    | `drums_6s` / `bass_6s` / `other_6s` / `vocals_6s` | 六路模型自己的四轨（与四路模型不同权重） |
    | `fx` | 四路 `other` stem 的 **>4 kHz 瞬态**（风铃/crash/采样打击/FX） |
    | `pnote_<stem>` | 该 stem 的 **basic-pitch 有音高 note onset**（全部声部） |
    | `pnote_lead_<stem>` | 同上但只取**最高声部**（`lead_mask`） |
    | `melody` | `other+guitar+piano` 的 **lead** note onset 合并去重 |
    | `melody_all` | 同上但不筛声部 |
    | `vocal_plus` | 四路 `vocals` 的能量 onset **∪** 人声有音高 note onset |
    """
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.audio_analysis import onsets as onsets_mod
    from tools.audio_analysis import pitch_notes as pn_mod
    from tools.audio_analysis import stems as stems_mod
    from tools.audio_analysis import tracks as tracks_mod

    song_dir = Path(song_dir)
    info: dict = {"stems_6s": False, "pitch": False, "counts": {}}

    # ---- fx：在四路的 other 上做（与既有四路口径同源，避免混入模型差异）----
    y_other = (stem_audio4 or {}).get("other")
    if y_other is None:
        p = song_dir / "stems" / "other.wav"
        if p.exists():
            y_other, _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
    if y_other is not None:
        bundle.onset_times["fx"] = onsets_mod.transient_fx(
            y_other, sr=onsets_mod.ANALYSIS_SR, hop=onsets_mod.HOP).times

    # ---- 六路 stem 的能量 onset ----
    sd6 = stems_mod.stems_dir_for(song_dir, "htdemucs_6s")
    if sd6.exists():
        for s in STEM_ORDER_6S:
            p = sd6 / f"{s}.wav"
            if not p.exists():
                continue
            y, _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
            t = onsets_mod.detect_onsets(y, s, sr=onsets_mod.ANALYSIS_SR,
                                         hop=onsets_mod.HOP).times
            bundle.onset_times[s if s in ("guitar", "piano") else f"{s}_6s"] = t
        info["stems_6s"] = True

    # ---- basic-pitch 的有音高 note ----
    cache = sd6 / "pitch_notes.json"
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        for d in raw.get("results", []):
            n = pn_mod.PitchNotes.from_dict(d)
            if not n.available:
                continue
            bundle.pitch_notes[n.stem] = n
            bundle.onset_times[f"pnote_{n.stem}"] = np.asarray(n.onsets, dtype=float)
            lead = pn_mod.filtered(n, lead_only=True)
            bundle.onset_times[f"pnote_lead_{n.stem}"] = np.asarray(lead.onsets,
                                                                    dtype=float)
        info["pitch"] = bool(bundle.pitch_notes)

    if bundle.pitch_notes:
        bundle.onset_times["melody"] = tracks_mod.build_melody_times(
            bundle.pitch_notes, lead_only=True)
        bundle.onset_times["melody_all"] = tracks_mod.build_melody_times(
            bundle.pitch_notes, lead_only=False)
        # 置信度筛选变体：basic-pitch 的 confidence ∈ [0,1]，用来把 melody 的候选池
        # 压到与鼓轨同量级（不筛的话密度是鼓的 2 倍，随机基线被抬爆）
        for c in (0.5, 0.6, 0.7):
            bundle.onset_times[f"melody_c{int(c * 100)}"] = tracks_mod.build_melody_times(
                bundle.pitch_notes, lead_only=True, min_confidence=c)
        v = bundle.pitch_notes.get("vocals")
        if v is not None and "vocals" in bundle.onset_times:
            bundle.onset_times["vocal_plus"] = tracks_mod.merge_times(
                bundle.onset_times["vocals"], v.onsets)

    info["counts"] = {k: int(np.size(v)) for k, v in bundle.onset_times.items()}
    bundle.v5 = info


def load_song(song_dir: Path, recompute_intensity: bool = True,
              with_v5: bool = True) -> SongBundle:
    """装载一首曲子的全部标定输入。"""
    import sys

    repo = Path(__file__).resolve().parents[2]
    if str(repo / "tools") not in sys.path:
        sys.path.insert(0, str(repo / "tools"))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from chart_analysis.density import chart_density
    from chart_analysis.simai_parser import parse_chart
    from tools.audio_analysis import features as feat_mod
    from tools.audio_analysis import intensity as intensity_mod
    from tools.audio_analysis import onsets as onsets_mod
    from tools.audio_analysis import stems as stems_mod
    from tools.audio_analysis.grid import BpmChange, Grid

    song_dir = Path(song_dir)
    name = song_dir.name
    meta = read_maidata_header(song_dir / "maidata.txt")
    analysis = json.loads((song_dir / "song_analysis.json").read_text(encoding="utf-8"))

    g = analysis["grid"]
    grid = Grid(bpm=g["bpm"], first=g["first"], beats_per_bar=g["beats_per_bar"],
                bpm_changes=tuple(BpmChange(bar=c["bar"], bpm=c["bpm"])
                                  for c in g.get("bpm_changes", [])),
                duration=g["duration"])

    res = parse_chart(extract_inote(song_dir / "maidata.txt"), name=name)
    dens = chart_density(res, name)

    bundle = SongBundle(
        name=name, level=float(meta.get("lv_5", 0.0) or 0.0), bpm=float(g["bpm"]),
        first=float(g["first"]), genre=str(meta.get("genre", "") or ""),
        grid=grid, analysis=analysis, parse=res, density=dens,
        bar_intensity=np.asarray(analysis["intensity"]["bar_intensity"], dtype=float),
        bar_intensity_raw=np.asarray(analysis["intensity"]["bar_intensity_raw"],
                                     dtype=float),
        segments=analysis["structure"]["segments"],
    )

    # ---- 音频侧：重跑 onset（纯 DSP，与管线同参）----
    import librosa

    wav, wav_was_temp = ensure_mix_wav(song_dir)
    y_mix, sr = librosa.load(str(wav), sr=onsets_mod.ANALYSIS_SR, mono=True)
    if wav_was_temp:
        # 磁盘紧张时管线跑完会删掉 track.44k.wav（22 MB/首 × 160 首）。
        # 这里按同一条 ffmpeg 命令重新解码（时间基准完全一致），用完即删。
        import shutil as _shutil

        _shutil.rmtree(wav.parent, ignore_errors=True)
    stem_audio: dict[str, np.ndarray] = {}
    for s in STEM_ORDER:
        p = song_dir / "stems" / f"{s}.wav"
        if p.exists():
            stem_audio[s], _ = stems_mod.load_stem_mono(p, sr=onsets_mod.ANALYSIS_SR)
    for s, y in stem_audio.items():
        bundle.onset_times[s] = onsets_mod.detect_onsets(
            y, s, sr=onsets_mod.ANALYSIS_SR, hop=onsets_mod.HOP).times
    if "drums" in stem_audio:
        drum_tracks = onsets_mod.drum_components(
            stem_audio["drums"], sr=onsets_mod.ANALYSIS_SR, hop=onsets_mod.HOP)
        for k, tr in drum_tracks.items():
            bundle.onset_times[k] = tr.times

    if recompute_intensity:
        if "vocals" in stem_audio:
            vad = onsets_mod.vocal_activity(stem_audio["vocals"],
                                            sr=onsets_mod.ANALYSIS_SR,
                                            hop=onsets_mod.HOP)
            voiced = onsets_mod.activity_per_bar(vad["active"], vad["times"], grid)
        else:
            voiced = np.zeros(grid.n_bars)
        merged = feat_mod.merged_onset_count(
            {k: bundle.onset_times[k] for k in STEM_ORDER if k in bundle.onset_times},
            grid)
        ires = intensity_mod.compute_intensity(
            grid, y_mix, sr, {"n_onset_merged": merged, "voiced_ratio": voiced},
            y_drums=stem_audio.get("drums"))
        bundle.components = {k: np.asarray(v, dtype=float)
                             for k, v in ires.components.items()}
        # 复核：重算的强度应与管线输出一致
        n = min(len(ires.bar_raw), len(bundle.bar_intensity_raw))
        if n:
            bundle.components["_recompute_max_abs_diff"] = np.array(
                [float(np.max(np.abs(ires.bar_raw[:n] - bundle.bar_intensity_raw[:n])))])

    if with_v5:
        load_v5_tracks(song_dir, bundle, stem_audio4=stem_audio)
    return bundle


def discover(calib_dir: Path) -> list[Path]:
    """找出所有已跑完管线的标定曲目录。"""
    calib_dir = Path(calib_dir)
    if not calib_dir.exists():
        return []
    return sorted(p for p in calib_dir.iterdir()
                  if p.is_dir() and (p / "song_analysis.json").exists()
                  and (p / "maidata.txt").exists())
