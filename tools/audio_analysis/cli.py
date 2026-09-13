"""命令行入口：mp3 + 用户给定的 BPM/first → 歌曲分析单（v0.3）。

    python -m tools.audio_analysis --audio track.mp3 --bpm 192 --first 1.875 \
        --level 13.5 --out out/TransientTears/

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

from . import (__version__, decode, features as feat_mod, grid as grid_mod,
               intensity as intensity_mod, onsets, pitch_notes as pn_mod, quantize,
               sheet, stemplan, stems, tracks)
from .grid import Grid, check_offset, offset_verdict, parse_bpm_changes
from .structure import (analyze_structure, assign_functions, suggest_division, tier_of)

STEM_ORDER = ("drums", "bass", "other", "vocals")
DRUM_PARTS = ("kick", "snare", "hihat")
INSTRUMENTAL_VOICED_THRESHOLD = 0.18   # 全曲 voiced_ratio 低于此判为器乐向
# 器乐曲判据（v0.5）：能量占比口径。n=40 实测能量 VAD 把 10/12 首器乐曲判成人声曲，
# 所以"这首是不是人声曲"改用 `share_vocals`（标定报告 n40 §5.3，阈值 0.10）。
INSTRUMENTAL_SHARE_VOCALS = 0.10
# 跑 basic-pitch 的 stem（有旋律性质的都跑；drums 不跑）
PITCH_STEMS = ("vocals", "other", "guitar", "piano", "bass")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tools.audio_analysis",
        description="在用户给定 BPM/offset 的前提下，把 mp3 变成结构化歌曲分析单。",
    )
    p.add_argument("--audio", required=True, help="输入音频（mp3/wav）")
    p.add_argument("--bpm", required=True, type=float, help="用户给定的 BPM（第 1 小节起）")
    p.add_argument("--first", required=True, type=float,
                   help="simai &first：谱面第 1 小节第 1 拍的音频秒数（可为负）")
    p.add_argument("--level", type=float, default=None,
                   help="目标定数（如 13.5）。给了才输出 note 总数区间（知识 004）"
                        "与该定数的官方均值 note/小节；缺省用全库量级")
    p.add_argument("--bpm-changes", default=None,
                   help='变速点，格式 "小节号:BPM,小节号:BPM"（可选，实验性）')
    p.add_argument("--beats-per-bar", type=int, default=4, help="每小节拍数（默认 4）")
    p.add_argument("--model", default=stems.DEFAULT_MODEL,
                   help="Demucs 模型（htdemucs / htdemucs_ft / **htdemucs_6s**，"
                        "默认 htdemucs）。六路模型额外给 guitar / piano 两轨，"
                        "输出目录按模型名分开（stems / stems_htdemucs_6s），"
                        "不会覆盖已有四路缓存")
    p.add_argument("--compact-stems", action="store_true",
                   help="stem 落盘写单声道 22.05kHz PCM_16（体积 1/8，分析口径无损）")
    p.add_argument("--no-pitch-notes", action="store_true",
                   help="跳过 basic-pitch 有音高 note 检测（.venv-pitch 缺席时自动跳过）")
    p.add_argument("--pitch-venv", default=None,
                   help="basic-pitch 所在的 venv（默认仓库根 .venv-pitch）")
    p.add_argument("--pitch-backend", default="onnx",
                   choices=("onnx", "coreml", "tf", "tflite"),
                   help="basic-pitch 后端，默认 onnx（与 coreml 数值实测一致，"
                        "但 coreml 退出时会抛 recursive_mutex 异常）")
    p.add_argument("--device", default="auto", help="推理设备 auto/mps/cpu/cuda")
    p.add_argument("--out", required=True, help="输出目录")
    p.add_argument("--divisions", default="4,8,12,16,24,32", help="候选分音（扫描顺序）")
    p.add_argument("--div-outlier-ratio", type=float, default=0.0,
                   help="定 div 时允许多少比例的 onset 超出 τ（默认 0.0 = 严格照 "
                        "v2 §1.1 Step 3 的 max 残差口径）。调到 0.1 可让分音直方图"
                        "重新有信息量，但那是偏离调研口径的做法，须在报告里注明")
    p.add_argument("--no-fine-div", action="store_true",
                   help="彻底禁止 {32}（默认已有红线：仅 drums 且 onset≥6 且 rms≤15ms 放行）")
    p.add_argument("--fusion-weights", default=None,
                   help='强度融合权重，如 "loudness=0.25,onset=0.30,drums=0.20,'
                        'voiced=0.15,flux=0.10"')
    p.add_argument("--vote-weights", default=None,
                   help='高潮投票权重，如 "structure=0.45,novelty=0.25,energy=0.20,'
                        'centroid=0.05,vocal=0.05"')
    p.add_argument("--section-floor", type=float,
                   default=intensity_mod.SECTION_DENSITY_FLOOR,
                   help=f"段落尺度密度地板（默认 {intensity_mod.SECTION_DENSITY_FLOOR}，"
                        f"来自 388 谱段间落差）。n=160 配对标定给出的池化最优是 "
                        f"{intensity_mod.CALIBRATED_SECTION_FLOOR}（默认值的 SSE 代价 "
                        f"+9.5%%），大样本优先未采纳为默认，可用本参数自行试")
    p.add_argument("--bar-floor", type=float, default=intensity_mod.BAR_DENSITY_FLOOR,
                   help=f"小节尺度密度地板（默认 {intensity_mod.BAR_DENSITY_FLOOR}；"
                        f"n=160 标定实测最优 0.340，默认值只差 0.5%% SSE → 保留默认）")
    p.add_argument("--intro-outro-cap", type=float,
                   default=intensity_mod.INTRO_OUTRO_CAP_RATIO,
                   help=f"intro/outro 建议密度的**结构封顶**比例（默认 "
                        f"{intensity_mod.INTRO_OUTRO_CAP_RATIO} × 全曲建议均值）。"
                        f"前奏/尾奏的密度基线由结构决定，不跟能量曲线走；设 0 关闭")
    p.add_argument("--offset-search-beats", type=float,
                   default=None,
                   help=f"offset 校验搜索半径（拍），默认 {grid_mod.DEFAULT_SEARCH_BEATS}。"
                        f"**不得 ≥0.5**（半拍处是反拍，会并列夺峰，v0.2 的 ±1 拍就是"
                        f"Signature/麒麟假警报的根因）")
    p.add_argument("--no-allin1", action="store_true",
                   help="跳过 all-in-one，边界只用 novelty + 重复段两路")
    p.add_argument("--align-to-detected", action="store_true",
                   help="（可选，默认关）用 offset 校验找到的最佳偏移构造分析网格")
    p.add_argument("--skip-stems", action="store_true", help="跳过 Demucs（调试用）")
    p.add_argument("--force", action="store_true", help="忽略缓存，重跑解码与分离")
    p.add_argument("--copy-plot-to", default=None, help="额外把 plot.png 复制到该路径")
    p.add_argument("--copy-sheet-to", default=None,
                   help="额外把 song_sheet.md 复制到该路径")
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

    v = {"tool_version": __version__, "python": sys.version.split()[0],
         "platform": platform.platform(), "numpy": numpy.__version__,
         "scipy": scipy.__version__, "librosa": librosa.__version__}
    try:
        v["ffmpeg"] = decode.ffmpeg_version()
    except Exception as exc:
        v["ffmpeg"] = f"unavailable: {exc}"
    for mod in ("torch", "demucs", "pyloudnorm", "matplotlib", "allin1_infer",
                "allin1", "basic_pitch"):
        try:
            v[mod] = getattr(__import__(mod), "__version__", "installed")
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
    print(f"[1/8] 解码 → {wav.name}（{duration:.2f}s，{timings['decode']:.2f}s）")

    # ---- 2. 网格 ----
    bpm_changes = parse_bpm_changes(args.bpm_changes)
    grid = Grid(bpm=args.bpm, first=args.first, beats_per_bar=args.beats_per_bar,
                bpm_changes=bpm_changes, duration=duration)
    warnings: list[str] = []
    if grid.has_bpm_changes:
        warnings.append("⚠️ 使用了 --bpm-changes：变速为实验性功能，网格只在小节边界换 BPM，"
                        "若原曲变速点不在小节线上其后所有小节都会错位，务必人工复核。")
        print("  " + warnings[-1])
    print(f"[2/8] 网格：BPM={args.bpm} first={args.first} → {grid.n_bars} 小节")

    import librosa

    # ---- 3. offset 校验（只报告）----
    t0 = time.time()
    y_mix, sr = librosa.load(str(wav), sr=onsets.ANALYSIS_SR, mono=True)
    env = onsets.onset_envelope(y_mix, sr=sr, hop=onsets.HOP)
    env_t = onsets.frame_times(len(env), sr=sr, hop=onsets.HOP)
    mix_track = onsets.detect_onsets(y_mix, "_default", sr=sr, hop=onsets.HOP)
    oc_kwargs = {}
    if args.offset_search_beats is not None:
        oc_kwargs["search_beats"] = float(args.offset_search_beats)
    oc = check_offset(env, env_t, bpm=args.bpm, first=args.first,
                      beats_per_bar=args.beats_per_bar, duration=duration,
                      onset_times=mix_track.times, onset_weights=mix_track.strengths,
                      **oc_kwargs)
    verdict = offset_verdict(
        oc, container_start_ms=(container_start * 1000.0) if container_start else None)
    timings["offset_check"] = time.time() - t0
    print(f"[3/8] offset 校验：{verdict}")

    analysis_first = float(args.first)
    if args.align_to_detected and oc.get("available"):
        analysis_first = float(oc["best_first"])
        grid = Grid(bpm=args.bpm, first=analysis_first, beats_per_bar=args.beats_per_bar,
                    bpm_changes=bpm_changes, duration=duration)
        warnings.append(f"已启用 --align-to-detected：分析网格用 {analysis_first:.4f}s"
                        f"（用户 --first={args.first}，差 {oc['delta_ms']:+.1f}ms）。")

    # ---- 4. 分离 ----
    stem_info = {"model": "skipped", "device": "-", "elapsed_sec": 0.0, "stems": {}}
    stem_audio: dict[str, np.ndarray] = {}
    stem_names = stems.stems_for(args.model)
    stems_dir = stems.stems_dir_for(out_dir, args.model)
    if not args.skip_stems:
        t0 = time.time()
        stem_info = stems.separate(wav, stems_dir, model_name=args.model,
                                   device=args.device, force=args.force,
                                   compact=bool(args.compact_stems))
        timings["stems"] = time.time() - t0
        print(f"[4/8] 分离（{stem_info['model']} @ {stem_info['device']}，"
              f"{timings['stems']:.1f}s，{len(stem_names)} 轨 → {stems_dir.name}/）")
        for name in stem_names:
            p = stems_dir / f"{name}.wav"
            if p.exists():
                stem_audio[name], _ = stems.load_stem_mono(p, sr=onsets.ANALYSIS_SR)
    else:
        timings["stems"] = 0.0
        print("[4/8] 跳过分离")

    # ---- 5. onset / 鼓件 / VAD / 人声音高 ----
    t0 = time.time()
    tracks_map: dict[str, onsets.OnsetTrack] = {}
    for name in stem_names:
        y = stem_audio.get(name)
        if y is not None:
            tracks_map[name] = onsets.detect_onsets(y, name, sr=onsets.ANALYSIS_SR,
                                                    hop=onsets.HOP)
    if "drums" in stem_audio:
        tracks_map.update(onsets.drum_components(stem_audio["drums"],
                                                 drums_track=tracks_map.get("drums"),
                                                 sr=onsets.ANALYSIS_SR, hop=onsets.HOP))
    # v0.5：`fx` 点缀轨 —— other stem 的 >4 kHz 瞬态（风铃/crash/采样打击/FX）
    if "other" in stem_audio:
        tracks_map["fx"] = onsets.transient_fx(stem_audio["other"],
                                               sr=onsets.ANALYSIS_SR, hop=onsets.HOP)
    if "vocals" in stem_audio:
        vad = onsets.vocal_activity(stem_audio["vocals"], sr=onsets.ANALYSIS_SR,
                                    hop=onsets.HOP)
        vocal_act = onsets.activity_per_bar(vad["active"], vad["times"], grid)
    else:
        vad = {"threshold_db": None, "active": np.zeros(0, dtype=bool),
               "times": np.zeros(0)}
        vocal_act = np.zeros(grid.n_bars)
    timings["onsets"] = time.time() - t0

    # ---- 5b. 有音高 note（basic-pitch 子进程；v0.5）----
    t0 = time.time()
    pitch_map: dict = {}
    pitch_info: dict = {"enabled": False, "backend": args.pitch_backend, "stems": {}}
    if not args.no_pitch_notes and stem_audio:
        wavs = {n: stems_dir / f"{n}.wav" for n in PITCH_STEMS
                if n in stem_audio and (stems_dir / f"{n}.wav").exists()}
        pitch_map = pn_mod.detect_batch(wavs, venv=args.pitch_venv,
                                        backend=args.pitch_backend,
                                        cache_path=stems_dir / "pitch_notes.json",
                                        force=args.force)
        pitch_info["enabled"] = any(v.available for v in pitch_map.values())
        pitch_info["stems"] = {
            k: {"available": v.available, "n_notes": v.count,
                "elapsed_sec": v.elapsed_sec, "error": v.error[:300]}
            for k, v in pitch_map.items()}
    timings["pitch_notes"] = time.time() - t0

    # v0.5 的两条派生流：melody（有音高 note 合并）与 fx（已在上面算好）
    melody_times = tracks.build_melody_times(pitch_map)
    if melody_times.size:
        tracks_map["melody"] = onsets.OnsetTrack(
            "melody", melody_times, np.ones(melody_times.size), np.zeros(0),
            np.zeros(0), heuristic=False)
    # vocal 轨的踩音候选 = 能量 onset ∪ 有音高 note onset（长音/legato 只有后者抓得到）
    v_notes = pitch_map.get("vocals")
    if v_notes is not None and v_notes.available and v_notes.count and "vocals" in tracks_map:
        merged_vocal = tracks.merge_times(tracks_map["vocals"].times, v_notes.onsets)
        tracks_map["vocals"] = onsets.OnsetTrack(
            "vocals", merged_vocal, np.ones(merged_vocal.size),
            tracks_map["vocals"].env, tracks_map["vocals"].env_times)

    onset_times = {k: v.times for k, v in tracks_map.items()}
    char_masks: dict[str, dict] = {}
    for name, tr in tracks_map.items():
        char_masks[name] = {"strong": tr.strong_mask()}
    # drums 的 kick 掩码：kick 子集在 drums onset 中的位置
    if "kick" in tracks_map and "drums" in tracks_map:
        dt = tracks_map["drums"].times
        kt = set(np.round(tracks_map["kick"].times, 6).tolist())
        char_masks.setdefault("drums", {})["kick"] = np.array(
            [round(float(t), 6) in kt for t in dt], dtype=bool)
    if "vocals" in stem_audio:
        if v_notes is not None and v_notes.available and v_notes.count:
            # v0.5：「有音高」改由 basic-pitch 判定（替代 librosa yin 的零模型代替品）
            pn_set = np.asarray(v_notes.onsets, dtype=float)
            char_masks.setdefault("vocals", {})["pitched"] = np.array(
                [bool(np.min(np.abs(pn_set - float(t))) <= 0.05)
                 for t in tracks_map["vocals"].times], dtype=bool)
            vad = pn_mod.as_activity(v_notes, np.asarray(vad.get("times"), dtype=float))
        else:
            char_masks.setdefault("vocals", {})["pitched"] = tracks.pitched_mask(
                stem_audio["vocals"], tracks_map["vocals"].times, onsets.ANALYSIS_SR)
    print("[5/8] onset：" + "、".join(f"{k}={v.count}" for k, v in tracks_map.items())
          + f"（{timings['onsets']:.1f}s）"
          + ("；有音高 note：" + "、".join(
              f"{k}={v.count}" for k, v in pitch_map.items() if v.available)
             + f"（{timings['pitch_notes']:.1f}s）" if pitch_info["enabled"]
             else "；basic-pitch 未启用"))

    # ---- 6. 量化（每小节唯一 div）+ 四轨渲染 ----
    t0 = time.time()
    divisions = tuple(int(x) for x in args.divisions.split(","))
    bar_div, per_track_slots, qstats = quantize.quantize_song(
        onset_times, grid, divisions=divisions, allow_fine=not args.no_fine_div,
        outlier_ratio=args.div_outlier_ratio)
    track_order = (tracks.TRACK_ORDER_V5 if "melody" in onset_times
                   else tracks.TRACK_ORDER)
    track_patterns = tracks.build_track_patterns(grid, bar_div, onset_times,
                                                 per_track_slots, char_masks, vad,
                                                 track_order=track_order)
    timings["quantize"] = time.time() - t0
    print(f"[6/8] 量化：分音分布 {qstats['division_share']}，"
          f"未落格 {qstats['unquantized_ratio']:.1%}"
          f"（{{32}} 红线拦下 {qstats['fine_blocked_bars']} 小节）")

    # ---- 7. 逐小节特征 ----
    t0 = time.time()
    bar_features = feat_mod.build_bar_features(grid, stem_audio, onsets.ANALYSIS_SR,
                                               onset_times, vocal_act, y_mix=y_mix,
                                               bar_division={b: bar_div[b].division
                                                             for b in bar_div},
                                               pitch_notes=pitch_map or None)
    timings["features"] = time.time() - t0

    # ---- 8. 结构 + 强度 + 切轨 ----
    t0 = time.time()
    struct = analyze_structure(wav, y_mix, sr, grid,
                               prefer_allin1=not args.no_allin1,
                               stems_dir=stems_dir if stem_audio else None,
                               work_dir=out_dir / "_allin1", device="cpu")
    timings["structure"] = time.time() - t0

    t0 = time.time()
    ires = intensity_mod.compute_intensity(
        grid, y_mix, sr, bar_features, y_drums=stem_audio.get("drums"),
        fusion_weights=_parse_weights(args.fusion_weights))

    global_voiced = float(np.mean(vocal_act)) if vocal_act.size else 0.0
    # v0.5：器乐曲判据换成能量占比（n=40 实测能量 VAD 把 10/12 首器乐曲判成人声曲）；
    # 有 basic-pitch 时再与"人声有音高 note 覆盖率"取交叉验证。
    share_vocals_mean = float(np.mean(bar_features.get("share_vocals", np.zeros(1))))
    pitched_vocal_mean = float(np.mean(bar_features["vocal_pitched_ratio"])) \
        if "vocal_pitched_ratio" in bar_features else None
    instrumental = share_vocals_mean < INSTRUMENTAL_SHARE_VOCALS
    if pitched_vocal_mean is not None:
        instrumental = instrumental and pitched_vocal_mean < 0.30
    instrumental_legacy = global_voiced < INSTRUMENTAL_VOICED_THRESHOLD
    segments = struct["segments"]
    segments, fn_notes = assign_functions(segments, ires.bar_intensity, bar_features,
                                          grid, instrumental=instrumental)
    ires = intensity_mod.vote_climax(ires, segments, grid, y_mix=y_mix, sr=sr,
                                     bar_features=bar_features,
                                     vote_weights=_parse_weights(args.vote_weights),
                                     instrumental=instrumental)
    # ---- 密度锚点（v0.3：主锚 = NPS，note/小节 只作对照）----
    npb_legacy, npb_legacy_note = intensity_mod.notes_per_bar_for_level(args.level)
    nps, nps_note = intensity_mod.nps_for_level(args.level)
    bar_seconds = np.array([grid.bar_duration(b) for b in range(1, grid.n_bars + 1)],
                           dtype=float)
    if nps is not None:
        npb_bar = intensity_mod.notes_per_bar_from_nps(nps, bar_seconds)
        anchor = f"NPS 锚：{nps_note}；note/小节 = NPS × 每小节秒数"
    else:
        npb_bar = np.full(grid.n_bars, float(npb_legacy))
        anchor = npb_legacy_note
    for s in segments:
        s.intensity_tier = tier_of(s.intensity)
        s.suggested_division = suggest_division(s.intensity_tier)
        seg_npb = float(np.mean(npb_bar[s.start_bar - 1: min(s.end_bar, len(npb_bar))]))
        dn, notes_pb = intensity_mod.density_map(
            np.array([s.intensity]), float(args.section_floor), seg_npb)
        s.density_norm, s.suggested_notes_per_bar = float(dn[0]), float(notes_pb[0])
    segments, stem_warnings = stemplan.plan_stems(segments, bar_features, grid,
                                                  duration_sec=duration)
    warnings.extend(stem_warnings)
    timings["intensity"] = time.time() - t0
    print(f"[7/8] 结构（{struct['method']}）→ {len(segments)} 段"
          + ("（器乐向）" if instrumental else "")
          + f"；音频能量高潮 = 第 {ires.climax_bar} 小节（≠谱面密度峰）")

    bar_density_norm, bar_suggested = intensity_mod.density_map(
        ires.bar_intensity, float(args.bar_floor), npb_bar)
    # intro/outro 的密度基线由**结构**封顶，不跟能量曲线走
    cap_info = {"applied": False, "reason": "--intro-outro-cap 0 → 关闭"}
    if float(args.intro_outro_cap) > 0:
        segments, bar_suggested, cap_info = intensity_mod.apply_structural_cap(
            segments, bar_suggested, cap_ratio=float(args.intro_outro_cap))
    npb_mean = float(np.mean(npb_bar))

    # ---- 输出 ----
    bar_rows = []
    for bar in range(1, grid.n_bars + 1):
        bd = bar_div[bar]
        row = {
            "bar": bar,
            "start_sec": round(grid.bar_start(bar), 4),
            "bpm": grid.bar_bpm(bar),
            "division": bd.division,
            "resolved": bd.resolved,
            "n_unquantized": bd.n_unquantized,
            "triplet": bd.triplet,
            "fine_blocked": bd.fine_blocked,
            "quant_rms_ms": round(bd.rms_ms, 2),
            "intensity": float(ires.bar_intensity[bar - 1]),
            "intensity_raw": float(ires.bar_raw[bar - 1]),
            "intensity_tier": tier_of(float(ires.bar_intensity[bar - 1])),
            "density_norm": round(float(bar_density_norm[bar - 1]), 4),
            "suggested_notes": round(float(bar_suggested[bar - 1]), 2),
            "patterns": {t: track_patterns[t][bar].pattern for t in track_order},
            "features": {k: round(float(v[bar - 1]), 4)
                         for k, v in bar_features.items() if len(v) >= bar},
        }
        row["n_onset_hihat"] = row["features"].get("n_onset_hihat", 0)
        # 只进 JSON / 计数列、不铺网格串的轨（v2 §5.2(d) 轨数上限 4）
        row["extra_counts"] = {k: row["features"].get(f"n_onset_{k}", 0)
                               for k in ("fx", "other", "guitar", "piano")
                               if f"n_onset_{k}" in row["features"]}
        bar_rows.append(row)

    tgt = {
        "level": args.level,
        "anchor": "nps" if nps is not None else "notes_per_bar",
        "nps": nps,
        "nps_source": nps_note,
        # 主锚（NPS × 每小节秒数）；变速曲逐小节不同，这里给全曲均值
        "notes_per_bar": round(npb_mean, 3),
        "notes_per_bar_source": anchor,
        # 对照列：旧的「定数 → note/小节」锚（标定实测 BPM≥200 时系统性高估 3.1–3.7）
        "notes_per_bar_level_anchor": npb_legacy,
        "notes_per_bar_level_anchor_source": npb_legacy_note + "（对照列，非主锚）",
        "density_floor": {"section": float(args.section_floor),
                          "bar": float(args.bar_floor),
                          "section_calibrated_optimum": intensity_mod.CALIBRATED_SECTION_FLOOR,
                          "note": "段落 floor 默认 0.60 来自 388 谱；n=8 配对标定的最优是 "
                                  "0.125（SSE 差 10.3%），样本太小未采纳"},
        "structural_cap": cap_info,
    }
    rng = intensity_mod.total_notes_range(args.level)
    if rng:
        tgt.update({"total_mean": rng[0], "total_p10": rng[1], "total_p90": rng[2],
                    "total_source": "知识 004（官方 ST/SD 谱 note 总数分布）"})

    total = time.time() - t_start
    timings["total"] = total
    payload = {
        "schema_version": "0.3",
        "generated_by": f"tools/audio_analysis {__version__}",
        "song": {"name": song_name, "audio": str(audio_in), "wav": str(wav),
                 "duration_sec": duration,
                 "container_start_time_sec": container_start},
        "grid": {**grid.to_dict(), "user_first": float(args.first),
                 "analysis_first": analysis_first,
                 "aligned_to_detected": bool(args.align_to_detected)},
        "target": tgt,
        "warnings": warnings,
        "offset_check": oc,
        "offset_verdict": verdict,
        "stems": {k: v for k, v in stem_info.items() if k != "stems"},
        "onset_tracks": {k: {"count": v.count, "heuristic": v.heuristic}
                         for k, v in tracks_map.items()},
        "vocal_vad": {"threshold_db": vad.get("threshold_db"),
                      "active_bar_ratio": float((vocal_act > 0.3).mean()),
                      "global_voiced_ratio": round(global_voiced, 4),
                      "instrumental": instrumental,
                      "instrumental_criterion": (
                          f"share_vocals 均值 {share_vocals_mean:.4f} < "
                          f"{INSTRUMENTAL_SHARE_VOCALS}"
                          + (f" ∧ 人声有音高覆盖率 {pitched_vocal_mean:.4f} < 0.30"
                             if pitched_vocal_mean is not None else "")),
                      "share_vocals_mean": round(share_vocals_mean, 4),
                      "vocal_pitched_ratio_mean": (
                          round(pitched_vocal_mean, 4)
                          if pitched_vocal_mean is not None else None),
                      "instrumental_legacy_vad": instrumental_legacy,
                      "note": ("v0.5：器乐/人声的判定改用能量占比 + basic-pitch 有音高 "
                               "note 覆盖率；能量 VAD `global_voiced_ratio` 保留为兼容"
                               "字段，n=40 实测它把 10/12 首器乐曲判成人声曲，不可用")},
        "pitch_notes": pitch_info,
        "track_order": list(track_order),
        "quantize": qstats,
        "bar_divisions": {str(b): bar_div[b].to_dict() for b in bar_div},
        "structure": {
            "method": struct["method"],
            "notes": struct.get("notes", []) + fn_notes,
            "boundary_vote": struct.get("boundary_vote"),
            "allin1_raw": struct.get("allin1_raw"),
            "segments": [s.to_dict() for s in segments],
            "elapsed_sec": struct.get("elapsed_sec", 0.0),
        },
        "intensity": {
            "weights": ires.weights,
            "loudness": ires.loudness_info,
            # ⚠️ v0.3 正名：这是**音频能量高潮**，不是谱面密度峰定位器
            "audio_climax_bar": ires.climax_bar,
            "audio_climax_peaks": ires.climax_peaks,
            "climax_kind": "audio_energy",
            "climax_note": (
                "`audio_climax_bar` = 音乐上最激烈的小节（五票投票）。**不是"
                "「谱面最密小节」的定位器**：8 首官方配对实测，音频峰与官方谱密度峰的"
                "中位误差 33 小节（曲长 67–124 小节），音频峰落在谱面密度前 20% 小节"
                "只有 4/8、五票 climax 5/8（标定报告 §2.3）。"),
            # 兼容字段（v0.2 名字），下游请改读 audio_climax_bar
            "climax_bar": ires.climax_bar,
            "climax_peaks": ires.climax_peaks,
            "bar_intensity": [round(float(x), 4) for x in ires.bar_intensity],
            "bar_intensity_raw": [round(float(x), 4) for x in ires.bar_raw],
            "vote_total": [round(float(x), 4) for x in ires.vote_total],
            "density_floor": {"section": float(args.section_floor),
                              "bar": float(args.bar_floor)},
            "note": "raw 与 smoothed 同基准（都已归一到 [0,1]）；平滑不改变电平",
        },
        "bars": bar_rows,
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "rtf": {k: round(v / duration, 4) for k, v in timings.items() if duration > 0},
        "tools": _tool_versions(),
    }

    sheet.write_analysis_json(out_dir / "song_analysis.json", payload)
    sheet.write_analysis_json(out_dir / "analysis.json", payload)   # v0.1 兼容名
    md = sheet.build_song_sheet_md(payload, segments, bar_rows)
    (out_dir / "song_sheet.md").write_text(md, encoding="utf-8")
    (out_dir / "song-sheet.md").write_text(md, encoding="utf-8")    # v0.1 兼容名
    plot_path = sheet.plot_overview(
        out_dir / "plot.png", grid, ires, segments, bar_features,
        bar_division={b: bar_div[b].division for b in bar_div},
        title=f"{song_name} — BPM {args.bpm} / first {args.first}")
    if args.copy_plot_to:
        dst = Path(args.copy_plot_to).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plot_path, dst)
    if args.copy_sheet_to:
        dst = Path(args.copy_sheet_to).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / "song_sheet.md", dst)

    print(f"[8/8] 输出 → {out_dir}")
    print(f"  song_sheet.md / song_analysis.json / plot.png"
          f"（总耗时 {total:.1f}s，RTF={total / max(duration, 1e-6):.3f}）")
    return payload


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
