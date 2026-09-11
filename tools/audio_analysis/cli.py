"""命令行入口：mp3 + 用户给定的 BPM/first → 歌曲分析单。

用法：

    python -m tools.audio_analysis --audio track.mp3 --bpm 192 --first 1.875 \
        --out out/TransientTears/

完整参数见 `--help`。
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, decode, intensity as intensity_mod, onsets, quantize, sheet, stems
from .grid import Grid, check_offset, offset_verdict, parse_bpm_changes
from .structure import analyze_structure, guess_functions, tier_of

STEM_ORDER = ("drums", "bass", "other", "vocals")
DRUM_PARTS = ("kick", "snare", "hihat")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.audio_analysis",
        description="在用户给定 BPM/offset 的前提下，把 mp3 变成结构化歌曲分析单。",
    )
    p.add_argument("--audio", required=True, help="输入音频（mp3/wav）")
    p.add_argument("--bpm", required=True, type=float, help="用户给定的 BPM（第 1 小节起）")
    p.add_argument("--first", required=True, type=float,
                   help="simai &first：谱面第 1 小节第 1 拍的音频秒数（可为负）")
    p.add_argument("--bpm-changes", default=None,
                   help='变速点，格式 "小节号:BPM,小节号:BPM"（可选，实验性）')
    p.add_argument("--beats-per-bar", type=int, default=4, help="每小节拍数（默认 4）")
    p.add_argument("--model", default="htdemucs",
                   help="Demucs 模型（htdemucs / htdemucs_ft，默认 htdemucs）")
    p.add_argument("--device", default="auto", help="推理设备 auto/mps/cpu/cuda")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--divisions", default="4,8,12,16,24,32", help="候选分音")
    p.add_argument("--tolerance-ms", type=float, default=25.0,
                   help="量化容差下限（毫秒，实际取 max(该值, 1/64 小节)）")
    p.add_argument("--fusion-weights", default=None,
                   help='强度融合权重，如 "rms=0.35,onset=0.30,centroid=0.20,drums=0.15"')
    p.add_argument("--vote-weights", default=None,
                   help='高潮投票权重，如 "structure=0.4,novelty=0.3,energy=0.2,centroid=0.1"')
    p.add_argument("--no-allin1", action="store_true", help="跳过 all-in-one，直接走 SSM 退路")
    p.add_argument("--align-to-detected", action="store_true",
                   help="（可选，默认关）用 offset 校验找到的最佳偏移构造分析网格。"
                        "小节编号与用户 --first 保持一致，只是网格整体平移，"
                        "用于抵消 MP3 解码偏移；输出会明确标注已平移。")
    p.add_argument("--skip-stems", action="store_true",
                   help="跳过 Demucs（仅用整曲信号，调试用）")
    p.add_argument("--force", action="store_true", help="忽略缓存，重跑解码与分离")
    p.add_argument("--copy-plot-to", default=None, help="额外把 plot.png 复制到该路径")
    return p


def _parse_weights(spec: str | None) -> dict | None:
    if not spec:
        return None
    out = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _tool_versions() -> dict:
    import librosa
    import numpy
    import scipy

    v = {
        "tool_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "librosa": librosa.__version__,
    }
    try:
        v["ffmpeg"] = decode.ffmpeg_version()
    except Exception as exc:
        v["ffmpeg"] = f"unavailable: {exc}"
    for mod in ("torch", "demucs", "pyloudnorm", "matplotlib", "allin1_infer", "allin1",
                "basic_pitch"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "installed")
        except Exception:
            v[mod] = "not installed"
    return v


def run(args: argparse.Namespace) -> dict:
    t_start = time.time()
    timings: dict[str, float] = {}
    audio_in = Path(args.audio).expanduser().resolve()
    if not audio_in.exists():
        raise SystemExit(f"找不到音频：{audio_in}")
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    song_name = audio_in.parent.name if audio_in.stem == "track" else audio_in.stem

    # ---- 1. 解码 ----
    t0 = time.time()
    wav = decode.decode_to_wav(audio_in, out_dir, sr=44100, force=args.force)
    duration = decode.probe_duration(wav) or 0.0
    container_start = decode.probe_start_time(audio_in)
    timings["decode"] = time.time() - t0
    print(f"[1/7] 解码完成 → {wav.name}（{duration:.2f}s，{timings['decode']:.2f}s）")

    # ---- 2. 网格 ----
    bpm_changes = parse_bpm_changes(args.bpm_changes)
    grid = Grid(bpm=args.bpm, first=args.first, beats_per_bar=args.beats_per_bar,
                bpm_changes=bpm_changes, duration=duration)
    warnings: list[str] = []
    if grid.has_bpm_changes:
        warnings.append(
            "⚠️ 使用了 --bpm-changes：变速支持为实验性功能，网格在变速点按小节边界切换 BPM，"
            "若原曲变速点不在小节线上，其后的所有小节时间都会错位，务必人工复核。")
        print("  " + warnings[-1])
    print(f"[2/7] 网格：BPM={args.bpm} first={args.first} → {grid.n_bars} 小节")

    import librosa

    # ---- 3. offset 校验（只报告）----
    t0 = time.time()
    y_mix, sr = librosa.load(str(wav), sr=onsets.ANALYSIS_SR, mono=True)
    env = onsets.onset_envelope(y_mix, sr=sr, hop=onsets.HOP)
    env_t = onsets.frame_times(len(env), sr=sr, hop=onsets.HOP)
    mix_track = onsets.detect_onsets(y_mix, "_default", sr=sr, hop=onsets.HOP)
    oc = check_offset(env, env_t, bpm=args.bpm, first=args.first,
                      beats_per_bar=args.beats_per_bar, duration=duration,
                      onset_times=mix_track.times, onset_weights=mix_track.strengths,
                      tolerance=args.tolerance_ms / 1000.0)
    verdict = offset_verdict(
        oc, container_start_ms=(container_start * 1000.0) if container_start else None)
    timings["offset_check"] = time.time() - t0
    print(f"[3/7] offset 校验：{verdict}")

    analysis_first = float(args.first)
    if args.align_to_detected and oc.get("available"):
        analysis_first = float(oc["best_first"])
        grid = Grid(bpm=args.bpm, first=analysis_first, beats_per_bar=args.beats_per_bar,
                    bpm_changes=bpm_changes, duration=duration)
        warnings.append(
            f"已启用 --align-to-detected：分析网格用自动最佳偏移 {analysis_first:.4f}s "
            f"（用户 --first={args.first}，相差 {oc['delta_ms']:+.1f}ms）。"
            "小节编号不变，但所有小节起始时间整体平移，谱面写作时仍应以用户 &first 为准。")
        print("  " + warnings[-1])

    # ---- 4. 分离 ----
    stem_info = {"model": "skipped", "device": "-", "elapsed_sec": 0.0, "stems": {}}
    stem_audio: dict[str, np.ndarray] = {}
    stems_dir = out_dir / "stems"
    if not args.skip_stems:
        t0 = time.time()
        stem_info = stems.separate(wav, stems_dir, model_name=args.model,
                                   device=args.device, force=args.force)
        timings["stems"] = time.time() - t0
        print(f"[4/7] 分离完成（{stem_info['model']} @ {stem_info['device']}，"
              f"{timings['stems']:.1f}s，RTF={timings['stems'] / max(duration, 1e-6):.3f}）")
        for name in STEM_ORDER:
            p = stems_dir / f"{name}.wav"
            if p.exists():
                stem_audio[name], _ = stems.load_stem_mono(p, sr=onsets.ANALYSIS_SR)
    else:
        timings["stems"] = 0.0
        print("[4/7] 跳过分离")

    # ---- 5. 逐 stem onset / 鼓件 / 人声 VAD ----
    t0 = time.time()
    tracks: dict[str, onsets.OnsetTrack] = {}
    for name in STEM_ORDER:
        y = stem_audio.get(name)
        if y is None:
            continue
        tracks[name] = onsets.detect_onsets(y, name, sr=onsets.ANALYSIS_SR, hop=onsets.HOP)
    if "drums" in stem_audio:
        tracks.update(onsets.drum_components(stem_audio["drums"],
                                             drums_track=tracks.get("drums"),
                                             sr=onsets.ANALYSIS_SR, hop=onsets.HOP))
    if "vocals" in stem_audio:
        vad = onsets.vocal_activity(stem_audio["vocals"], sr=onsets.ANALYSIS_SR, hop=onsets.HOP)
        vocal_act = onsets.activity_per_bar(vad["active"], vad["times"], grid)
    else:
        vad = {"threshold_db": None}
        vocal_act = np.zeros(grid.n_bars)
    timings["onsets"] = time.time() - t0
    print("[5/7] onset：" + "、".join(f"{k}={v.count}" for k, v in tracks.items())
          + f"（{timings['onsets']:.1f}s）")

    # ---- 6. 量化 ----
    t0 = time.time()
    divisions = tuple(int(x) for x in args.divisions.split(","))
    quantize.DEFAULT_TOL_SEC = args.tolerance_ms / 1000.0
    bar_quant: dict[str, dict[int, quantize.BarQuant]] = {}
    quant_stats: dict[str, dict] = {}
    for name, tr in tracks.items():
        bq, st = quantize.quantize_track(tr.times, tr.strong_mask(), name, grid, divisions)
        bar_quant[name] = bq
        quant_stats[name] = st
    bar_onsets = {name: onsets.onsets_per_bar(tr, grid) for name, tr in tracks.items()}
    timings["quantize"] = time.time() - t0
    print(f"[6/7] 量化完成（{timings['quantize']:.1f}s）")

    # ---- 7. 结构 + 强度 ----
    t0 = time.time()
    struct = analyze_structure(wav, y_mix, sr, grid,
                               prefer_allin1=not args.no_allin1,
                               stems_dir=stems_dir if stem_audio else None,
                               work_dir=out_dir / "_allin1",
                               device="cpu")
    timings["structure"] = time.time() - t0

    t0 = time.time()
    ires = intensity_mod.compute_intensity(
        y_mix, sr, grid, y_drums=stem_audio.get("drums"),
        fusion_weights=_parse_weights(args.fusion_weights))
    segments = struct["segments"]
    segments = guess_functions(segments, ires.bar_intensity, struct["method"])
    ires = intensity_mod.vote_climax(ires, segments, grid,
                                     vote_weights=_parse_weights(args.vote_weights))
    for s in segments:
        s.intensity_tier = tier_of(s.intensity)
        s.primary_stem = sheet._primary_stem_of(s, bar_onsets, vocal_act)
    timings["intensity"] = time.time() - t0
    print(f"[7/7] 结构（{struct['method']}，{timings['structure']:.1f}s）→ "
          f"{len(segments)} 段；高潮 = 第 {ires.climax_bar} 小节")

    # ---- 输出 ----
    bar_rows = []
    for bar in range(1, grid.n_bars + 1):
        patterns = {}
        divs = []
        for name in sheet.PATTERN_COLUMNS:
            bq = bar_quant.get(name, {}).get(bar)
            if bq is None:
                patterns[name] = ""
                continue
            patterns[name] = bq.pattern
            if bq.n_onsets:
                divs.append(bq.division)
        drum_onsets = int(bar_onsets.get("drums", np.zeros(grid.n_bars))[bar - 1])
        bar_rows.append({
            "bar": bar,
            "start_sec": round(grid.bar_start(bar), 4),
            "bpm": grid.bar_bpm(bar),
            "intensity": float(ires.bar_intensity[bar - 1]),
            "intensity_tier": tier_of(float(ires.bar_intensity[bar - 1])),
            "vocal_activity": float(vocal_act[bar - 1]),
            "drum_onsets": drum_onsets,
            "finest_division": max(divs) if divs else 0,
            "onsets": {k: int(v[bar - 1]) for k, v in bar_onsets.items()},
            "divisions": {k: bar_quant[k][bar].division for k in bar_quant
                          if bar_quant[k][bar].n_onsets},
            "patterns": patterns,
            "quant_unresolved": sorted(
                k for k in bar_quant if not bar_quant[k][bar].resolved),
        })

    total = time.time() - t_start
    timings["total"] = total
    payload = {
        "schema_version": "0.1",
        "generated_by": f"tools/audio_analysis {__version__}",
        "song": {"name": song_name, "audio": str(audio_in),
                 "wav": str(wav), "duration_sec": duration,
                 "container_start_time_sec": container_start},
        "grid": {**grid.to_dict(), "user_first": float(args.first),
                 "analysis_first": analysis_first,
                 "aligned_to_detected": bool(args.align_to_detected)},
        "warnings": warnings,
        "offset_check": oc,
        "offset_verdict": verdict,
        "stems": {k: v for k, v in stem_info.items() if k != "stems"},
        "onset_tracks": {k: {"count": v.count, "heuristic": v.heuristic}
                         for k, v in tracks.items()},
        "vocal_vad": {"threshold_db": vad.get("threshold_db"),
                      "active_bar_ratio": float((vocal_act > 0.3).mean())},
        "quantize_stats": quant_stats,
        "structure": {
            "method": struct["method"],
            "notes": struct.get("notes", []),
            "segments": [s.to_dict() for s in segments],
            "alt_segments": struct.get("alt_segments"),
            "elapsed_sec": struct.get("elapsed_sec", 0.0),
        },
        "intensity": {
            "weights": ires.weights,
            "loudness": ires.loudness_info,
            "climax_bar": ires.climax_bar,
            "climax_peaks": ires.climax_peaks,
            "bar_intensity": [round(float(x), 4) for x in ires.bar_intensity],
            "vote_total": [round(float(x), 4) for x in ires.vote_total],
        },
        "bars": bar_rows,
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "rtf": {k: round(v / duration, 4) for k, v in timings.items() if duration > 0},
        "tools": _tool_versions(),
    }

    sheet.write_analysis_json(out_dir / "analysis.json", payload)
    md = sheet.build_song_sheet_md(payload, segments, bar_rows)
    (out_dir / "song-sheet.md").write_text(md, encoding="utf-8")
    plot_path = sheet.plot_overview(out_dir / "plot.png", grid, ires, segments,
                                    bar_onsets, vocal_act,
                                    title=f"{song_name} — BPM {args.bpm} / first {args.first}")
    if args.copy_plot_to:
        dst = Path(args.copy_plot_to).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plot_path, dst)

    print(f"\n输出 → {out_dir}")
    print(f"  analysis.json / song-sheet.md / plot.png（总耗时 {total:.1f}s，"
          f"RTF={total / max(duration, 1e-6):.3f}）")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
