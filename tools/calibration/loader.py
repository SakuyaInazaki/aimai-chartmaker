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
INOTE_KEY = "&inote_5="          # Master 难度


def read_maidata_header(path: Path) -> dict:
    """读 maidata 头部元数据（只取标定需要的字段）。"""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    out: dict = {}
    for key, cast in (("title", str), ("first", float), ("wholebpm", float),
                      ("lv_5", float)):
        m = re.search(rf"^&{key}=(.*)$", txt, flags=re.M)
        if m:
            try:
                out[key] = cast(m.group(1).strip())
            except ValueError:
                out[key] = m.group(1).strip()
    return out


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
    grid: object = None
    analysis: dict = field(default_factory=dict)
    parse: object = None              # chart_analysis.simai_parser.ParseResult
    density: object = None            # chart_analysis.density.ChartDensity
    onset_times: dict = field(default_factory=dict)   # {stem: np.ndarray(秒)}
    components: dict = field(default_factory=dict)    # 五项强度分量（逐小节，未 Z 化）
    bar_intensity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bar_intensity_raw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    segments: list = field(default_factory=list)      # song_analysis.json 的段落 dict

    @property
    def n_bars(self) -> int:
        return int(self.grid.n_bars) if self.grid is not None else 0


def load_song(song_dir: Path, recompute_intensity: bool = True) -> SongBundle:
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
        first=float(g["first"]), grid=grid, analysis=analysis, parse=res, density=dens,
        bar_intensity=np.asarray(analysis["intensity"]["bar_intensity"], dtype=float),
        bar_intensity_raw=np.asarray(analysis["intensity"]["bar_intensity_raw"],
                                     dtype=float),
        segments=analysis["structure"]["segments"],
    )

    # ---- 音频侧：重跑 onset（纯 DSP，与管线同参）----
    import librosa

    wav = song_dir / "track.44k.wav"
    y_mix, sr = librosa.load(str(wav), sr=onsets_mod.ANALYSIS_SR, mono=True)
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
    return bundle


def discover(calib_dir: Path) -> list[Path]:
    """找出所有已跑完管线的标定曲目录。"""
    calib_dir = Path(calib_dir)
    if not calib_dir.exists():
        return []
    return sorted(p for p in calib_dir.iterdir()
                  if p.is_dir() and (p / "song_analysis.json").exists()
                  and (p / "maidata.txt").exists())
