"""mp3 → wav 解码（ffmpeg）。

设计文档 §2 指出 MP3 解码在不同解码器下会有 20–40ms 的偏移差异，
因此管线**先统一用 ffmpeg 转成 44.1kHz WAV**，后续所有分析都读这个 wav，
保证 offset 校验、onset 时间与量化用的是同一套时间基准。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"


def _ffmpeg_bin() -> str:
    if Path(FFMPEG).exists():
        return FFMPEG
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("找不到 ffmpeg，请安装（brew install ffmpeg）")
    return found


def _ffprobe_bin() -> str | None:
    if Path(FFPROBE).exists():
        return FFPROBE
    return shutil.which("ffprobe")


def ffmpeg_version() -> str:
    out = subprocess.run([_ffmpeg_bin(), "-version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"


def probe_duration(path: Path) -> float | None:
    """用 ffprobe 读时长（秒）；拿不到返回 None。"""
    probe = _ffprobe_bin()
    if not probe:
        return None
    out = subprocess.run(
        [probe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except Exception:
        return None


def probe_start_time(path: Path) -> float | None:
    """读取容器/流的 start_time（秒）。

    MP3 的编码器延迟通常体现为 ~0.025s 的 start_time；不同播放器对这段延迟的
    处理方式不同，正是设计文档 §2 所说"MP3 解码偏移 20–40ms"的来源。
    """
    probe = _ffprobe_bin()
    if not probe:
        return None
    out = subprocess.run(
        [probe, "-v", "error", "-show_entries", "stream=start_time",
         "-select_streams", "a:0", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(json.loads(out.stdout)["streams"][0]["start_time"])
    except Exception:
        return None


def decode_to_wav(src: Path, out_dir: Path, sr: int = 44100, force: bool = False) -> Path:
    """把音频统一解码成 `out_dir/<stem>.44k.wav`（立体声、PCM 16bit）。

    返回 wav 路径。已存在且未指定 force 时直接复用。
    """
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}.{sr // 1000}k.wav"
    if dst.exists() and not force:
        return dst
    cmd = [
        _ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le",
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败：{res.stderr.strip()[:500]}")
    return dst
