"""有音高的 note 事件检测（basic-pitch，Apache-2.0 ✅）。

## 为什么要它（用户 2026-09-12）

四路管线的踩音候选池全部来自 **librosa 能量 onset**，它抓不到：

- **长音 / 延音**：合成器 pad、弦乐、人声长音——能量包络平坦，没有瞬态；
- **分解和弦 / 连奏**：钢琴琶音、吉他滑音，音高变了但能量没有明显跳变；
- **音高事件**：同一个音持续期间换到另一个音（legato），能量无变化。

而官方谱大量踩这些东西 —— n=40 标定实测 **16.8% 的官方 note 什么 stem 的 onset
都不落**。basic-pitch 输出的是 **note 事件（onset/offset/pitch/confidence）**，
正是这一类"有音高但没瞬态"的音的检测器。

## 为什么要子进程

主仓库 `.venv` 是 **python 3.12**，`basic-pitch` 在其中装不上：
`basic-pitch[tf]` 钉死 `tensorflow-macos<2.15.1`（无 cp312 wheel），
`basic-pitch[onnx]` 在 py3.12 上 pip 解析 `narwhals` 时 `resolution-too-deep`
（设计文档 §3.5 记录的两次失败）。

**v0.5 的解法**：用 `uv` 建一个**独立的 python 3.11 venv**（uv 自带独立 Python
下载，不动系统 python），只装 basic-pitch，主管线通过**子进程 + JSON** 调用它。

```bash
pip install uv                                    # 或 brew install uv
uv venv .venv-pitch --python 3.11
uv pip install --python .venv-pitch "basic-pitch[onnx]" "setuptools<81"
```

⚠️ `setuptools<81` 是硬性的：`resampy`（basic-pitch 的依赖）仍 `import pkg_resources`，
setuptools ≥ 81 已把它删掉，不钉版本会在 `import basic_pitch.inference` 时炸。

## 后端选择：固定 ONNX

Darwin 上 basic-pitch 默认走 **CoreML**（`nmp.mlpackage`）。🧪 实测两后端在合成音上
**note 事件完全一致**（同样 4 个音、onset/offset/pitch/confidence 逐位相同），
CoreML 更快，但进程退出时会抛
`libc++abi: recursive_mutex lock failed`（coremltools 的 atexit 问题）——
在批量子进程调用里这个非零退出码会污染错误处理。
→ **默认固定 ONNX**（可用 `--backend coreml` 覆盖）。

## 本模块的双重身份

- 在**主 venv** 里 `import`：用 `detect_pitch_notes()` / `detect_batch()`，它们负责
  拼子进程命令、跑 `.venv-pitch` 的 python、读回 JSON；
- 在 **`.venv-pitch`** 里 `python -m`（或直接跑本文件）：进入 `main()`，真正调 basic-pitch。

两边是同一个文件，避免两份参数漂移。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 参数（两侧共用）
# ---------------------------------------------------------------------------

#: 默认 venv 位置（仓库根 `.venv-pitch`）
DEFAULT_VENV = Path(__file__).resolve().parents[2] / ".venv-pitch"

#: basic-pitch 阈值。onset 阈值放宽到 0.4（默认 0.5）——分离后的 stem 信噪比不如
#: 干净独奏，且我们要的是"候选池"而不是转录成品；frame 阈值维持默认 0.3。
#: ⚠️ **这两个值未标定**，见 README v0.5 与报告 `stem-refinement-n40.md` §7。
DEFAULT_ONSET_THRESHOLD = 0.4
DEFAULT_FRAME_THRESHOLD = 0.3
#: 最短 note 时长（毫秒）。128 ms ≈ 190 BPM 的 1/8 音，再短的多半是抖动。
DEFAULT_MIN_NOTE_LEN_MS = 58.0
#: 只保留置信度 ≥ 此值的 note（basic-pitch 的 confidence ∈ [0,1]）
DEFAULT_MIN_CONFIDENCE = 0.0


@dataclass
class PitchNotes:
    """一条 stem 的有音高 note 事件序列。"""

    stem: str
    onsets: np.ndarray          # 秒
    offsets: np.ndarray         # 秒
    pitches: np.ndarray         # MIDI note number
    confidences: np.ndarray     # [0,1]
    backend: str = "onnx"
    elapsed_sec: float = 0.0
    available: bool = True
    error: str = ""

    @property
    def count(self) -> int:
        return int(len(self.onsets))

    def strong_mask(self, quantile: float = 0.7) -> np.ndarray:
        if self.count == 0:
            return np.zeros(0, dtype=bool)
        thr = float(np.quantile(self.confidences, quantile))
        return self.confidences >= thr

    def to_dict(self) -> dict:
        return {
            "stem": self.stem, "backend": self.backend, "available": self.available,
            "error": self.error, "elapsed_sec": round(self.elapsed_sec, 3),
            "n_notes": self.count,
            "onsets": [round(float(v), 4) for v in self.onsets],
            "offsets": [round(float(v), 4) for v in self.offsets],
            "pitches": [int(v) for v in self.pitches],
            "confidences": [round(float(v), 4) for v in self.confidences],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PitchNotes":
        return cls(
            stem=d.get("stem", ""),
            onsets=np.asarray(d.get("onsets", []), dtype=float),
            offsets=np.asarray(d.get("offsets", []), dtype=float),
            pitches=np.asarray(d.get("pitches", []), dtype=float),
            confidences=np.asarray(d.get("confidences", []), dtype=float),
            backend=d.get("backend", ""), elapsed_sec=float(d.get("elapsed_sec", 0.0)),
            available=bool(d.get("available", True)), error=d.get("error", ""),
        )

    @classmethod
    def unavailable(cls, stem: str, why: str) -> "PitchNotes":
        z = np.zeros(0)
        return cls(stem, z, z, z, z, backend="", available=False, error=why)


# ---------------------------------------------------------------------------
# 主 venv 侧：子进程调用
# ---------------------------------------------------------------------------


def venv_python(venv: Path | None = None) -> Path | None:
    """`.venv-pitch` 里的 python 可执行文件；不存在返回 None。"""
    env = os.environ.get("CHARTMAKER_PITCH_PYTHON")
    if env and Path(env).exists():
        return Path(env)
    root = Path(venv or os.environ.get("CHARTMAKER_PITCH_VENV") or DEFAULT_VENV)
    exe = root / "bin" / "python"
    return exe if exe.exists() else None


def available(venv: Path | None = None) -> bool:
    return venv_python(venv) is not None


def detect_batch(
    wavs: dict[str, Path],
    venv: Path | None = None,
    backend: str = "onnx",
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    min_note_len_ms: float = DEFAULT_MIN_NOTE_LEN_MS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    timeout: float = 1800.0,
    cache_path: Path | None = None,
    force: bool = False,
) -> dict[str, PitchNotes]:
    """一次子进程调用跑多条 stem（模型只加载一次）。

    参数：
        wavs: {stem 名: wav 路径}
        cache_path: 给了就把结果存成 JSON 并优先复用（basic-pitch 不便宜，
            40 首 × 5 轨要跑几分钟，标定反复试参数时必须能复用）
    返回：
        {stem 名: PitchNotes}；`.venv-pitch` 缺席时每条都是 `unavailable`。
    """
    if cache_path is not None and Path(cache_path).exists() and not force:
        try:
            raw = json.loads(Path(cache_path).read_text(encoding="utf-8"))
            cached = {d["stem"]: PitchNotes.from_dict(d) for d in raw.get("results", [])}
            if all(k in cached for k in wavs):
                return {k: cached[k] for k in wavs}
        except Exception:
            pass
    py = venv_python(venv)
    if py is None:
        why = (f"`.venv-pitch` 不存在（找过 {Path(venv or DEFAULT_VENV)}）——"
               "见 tools/audio_analysis/README.md v0.5 的装包步骤")
        return {k: PitchNotes.unavailable(k, why) for k in wavs}
    if not wavs:
        return {}

    spec = [{"stem": k, "path": str(Path(v).resolve())} for k, v in wavs.items()]
    cmd = [
        str(py), str(Path(__file__).resolve()), "--json-spec", json.dumps(spec),
        "--backend", backend,
        "--onset-threshold", str(onset_threshold),
        "--frame-threshold", str(frame_threshold),
        "--min-note-len-ms", str(min_note_len_ms),
        "--min-confidence", str(min_confidence),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {k: PitchNotes.unavailable(k, f"basic-pitch 子进程超时（>{timeout}s）")
                for k in wavs}
    payload = _last_json_line(proc.stdout)
    if payload is None:
        tail = (proc.stderr or proc.stdout or "")[-600:]
        return {k: PitchNotes.unavailable(k, f"basic-pitch 子进程失败：{tail}")
                for k in wavs}
    out: dict[str, PitchNotes] = {}
    for item in payload.get("results", []):
        out[item.get("stem", "")] = PitchNotes.from_dict(item)
    for k in wavs:
        out.setdefault(k, PitchNotes.unavailable(k, "子进程未返回该 stem"))
    if cache_path is not None and any(v.available for v in out.values()):
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(
            json.dumps({"results": [v.to_dict() for v in out.values()]},
                       ensure_ascii=False), encoding="utf-8")
    return out


def detect_pitch_notes(wav: Path, stem: str = "", **kw) -> PitchNotes:
    """单条 stem 的便捷封装。"""
    name = stem or Path(wav).stem
    return detect_batch({name: Path(wav)}, **kw)[name]


def _last_json_line(text: str) -> dict | None:
    """子进程 stdout 里最后一行合法 JSON（basic-pitch 会往 stdout 打进度）。"""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# 逐小节聚合（主 venv 侧，纯 numpy）
# ---------------------------------------------------------------------------


def notes_per_bar(notes: PitchNotes, grid) -> np.ndarray:
    """逐小节的有音高 note 数（按 onset 归属）。"""
    out = np.zeros(grid.n_bars, dtype=float)
    if notes.count == 0:
        return out
    t = np.asarray(notes.onsets, dtype=float)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        out[bar - 1] = float(np.sum((t >= t0) & (t < t0 + grid.bar_duration(bar))))
    return out


def pitched_coverage_per_bar(notes: PitchNotes, grid) -> np.ndarray:
    """逐小节"有音高 note 覆盖的时间比例" ∈ [0,1]。

    **这是用来替换能量 VAD `voiced_ratio` 的量**（设计文档 §3.4 / §9-A.3）：
    `voiced_ratio` 用相对 dB 阈值，vocals 轨近乎静音时底噪也会超阈，
    n=40 实测把 **10/12 首器乐曲判成人声曲**。
    "这一小节里有多少时间被真正的有音高 note 盖住"没有这个失败模式 ——
    纯泄漏/底噪不会被 basic-pitch 判成 note。
    """
    out = np.zeros(grid.n_bars, dtype=float)
    if notes.count == 0:
        return out
    on = np.asarray(notes.onsets, dtype=float)
    off = np.asarray(notes.offsets, dtype=float)
    for bar in range(1, grid.n_bars + 1):
        t0 = grid.bar_start(bar)
        bd = grid.bar_duration(bar)
        t1 = t0 + bd
        if bd <= 0:
            continue
        # 区间并集长度（note 会重叠：和声/复音）
        lo = np.maximum(on, t0)
        hi = np.minimum(off, t1)
        sel = hi > lo
        if not sel.any():
            continue
        out[bar - 1] = float(min(1.0, _union_length(lo[sel], hi[sel]) / bd))
    return out


def _union_length(lo: np.ndarray, hi: np.ndarray) -> float:
    order = np.argsort(lo)
    lo, hi = lo[order], hi[order]
    total = 0.0
    cur_lo, cur_hi = float(lo[0]), float(hi[0])
    for a, b in zip(lo[1:], hi[1:]):
        if a > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = float(a), float(b)
        else:
            cur_hi = max(cur_hi, float(b))
    return total + (cur_hi - cur_lo)


def lead_mask(notes: PitchNotes, tol: float = 0.03) -> np.ndarray:
    """**主旋律（top voice）掩码**：该 note 起音时没有更高音在同时发声。

    为什么需要：basic-pitch 是**复音**转录器，一个和弦会吐出 3–5 个 note。
    🧪 实测《Signature》的 `other` stem 一首曲吐 3450 个 note（drums 的能量 onset
    只有 476 个）—— 直接拿全部 note 当候选池，密度比鼓轨高一个数量级，
    随机基线被抬爆，lift 会失去意义。

    谱师听到并会去踩的是**最高声部的旋律线**（和声内声部听不见也不该踩），
    所以取"起音时刻音高最高"的那一条。⚠️ 启发式：贝斯线主导的曲子里最高声部
    未必是主旋律。
    """
    n = notes.count
    if n == 0:
        return np.zeros(0, dtype=bool)
    on = np.asarray(notes.onsets, dtype=float)
    off = np.asarray(notes.offsets, dtype=float)
    pit = np.asarray(notes.pitches, dtype=float)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        t = on[i]
        sounding = (on <= t + tol) & (off > t + tol)
        sounding[i] = True
        out[i] = pit[i] >= float(np.max(pit[sounding]))
    return out


def filtered(notes: PitchNotes, lead_only: bool = False,
             min_confidence: float = 0.0,
             min_duration_sec: float = 0.0) -> PitchNotes:
    """按主旋律 / 置信度 / 时长筛一遍，返回新的 `PitchNotes`。"""
    if notes.count == 0:
        return notes
    keep = np.ones(notes.count, dtype=bool)
    if lead_only:
        keep &= lead_mask(notes)
    if min_confidence > 0:
        keep &= np.asarray(notes.confidences, dtype=float) >= min_confidence
    if min_duration_sec > 0:
        keep &= (np.asarray(notes.offsets, dtype=float)
                 - np.asarray(notes.onsets, dtype=float)) >= min_duration_sec
    return PitchNotes(
        stem=notes.stem, onsets=np.asarray(notes.onsets)[keep],
        offsets=np.asarray(notes.offsets)[keep],
        pitches=np.asarray(notes.pitches)[keep],
        confidences=np.asarray(notes.confidences)[keep],
        backend=notes.backend, elapsed_sec=notes.elapsed_sec,
        available=notes.available, error=notes.error)


def as_activity(notes: PitchNotes, frame_times: np.ndarray) -> dict:
    """把 note 区间铺成帧级"有声"掩码，接口与 `onsets.vocal_activity` 兼容。

    给 song sheet 的 `-`（延音持续）字符用 —— v0.5 起人声延音以**有音高 note 的
    持续时间**为准，不再用能量 VAD（后者在纯泄漏上会全程判"有声"）。
    """
    ft = np.atleast_1d(np.asarray(frame_times, dtype=float))
    active = np.zeros(ft.size, dtype=bool)
    if notes.count and ft.size:
        for a, b in zip(notes.onsets, notes.offsets):
            active |= (ft >= float(a)) & (ft < float(b))
    return {"active": active, "times": ft, "rms": np.zeros(ft.size),
            "threshold_db": None, "source": "basic-pitch note 区间"}


def pitch_change_times(notes: PitchNotes, min_semitones: float = 0.5) -> np.ndarray:
    """"音高变化点"时间序列：相邻 note 音高不同处的 onset。

    长音里的 legato 换音、分解和弦的每个音都会在这里出现，而能量 onset 抓不到。
    """
    if notes.count < 2:
        return np.asarray(notes.onsets, dtype=float)
    order = np.argsort(notes.onsets)
    on = np.asarray(notes.onsets, dtype=float)[order]
    pit = np.asarray(notes.pitches, dtype=float)[order]
    keep = [True] + [abs(pit[i] - pit[i - 1]) >= min_semitones for i in range(1, len(on))]
    return on[np.asarray(keep, dtype=bool)]


# ---------------------------------------------------------------------------
# `.venv-pitch` 侧：真正调 basic-pitch
# ---------------------------------------------------------------------------


def _run_inside_pitch_venv(spec: list[dict], backend: str, onset_threshold: float,
                           frame_threshold: float, min_note_len_ms: float,
                           min_confidence: float) -> dict:
    import time

    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model, predict

    suffix = {"onnx": FilenameSuffix.onnx, "coreml": FilenameSuffix.coreml,
              "tf": FilenameSuffix.tf, "tflite": FilenameSuffix.tflite}[backend]
    model = Model(build_icassp_2022_model_path(suffix))

    results = []
    for item in spec:
        stem, path = item.get("stem", ""), item.get("path", "")
        t0 = time.time()
        try:
            _mo, _mi, note_events = predict(
                path, model,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                minimum_note_length=min_note_len_ms,
            )
            on, off, pit, conf = [], [], [], []
            for ev in note_events:
                s, e, p, c = ev[0], ev[1], ev[2], ev[3]
                if float(c) < min_confidence:
                    continue
                on.append(float(s)); off.append(float(e))
                pit.append(int(p)); conf.append(float(c))
            order = sorted(range(len(on)), key=lambda i: on[i])
            results.append({
                "stem": stem, "backend": backend, "available": True, "error": "",
                "elapsed_sec": round(time.time() - t0, 3), "n_notes": len(order),
                "onsets": [round(on[i], 4) for i in order],
                "offsets": [round(off[i], 4) for i in order],
                "pitches": [pit[i] for i in order],
                "confidences": [round(conf[i], 4) for i in order],
            })
        except Exception as exc:  # 单条失败不拖垮整批
            results.append({"stem": stem, "backend": backend, "available": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "elapsed_sec": round(time.time() - t0, 3),
                            "onsets": [], "offsets": [], "pitches": [],
                            "confidences": []})
    return {"results": results}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="basic-pitch 有音高 note 检测（在 .venv-pitch 里运行）")
    ap.add_argument("--json-spec", required=True,
                    help='[{"stem": "vocals", "path": "/.../vocals.wav"}, ...]')
    ap.add_argument("--backend", default="onnx",
                    choices=("onnx", "coreml", "tf", "tflite"))
    ap.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    ap.add_argument("--frame-threshold", type=float, default=DEFAULT_FRAME_THRESHOLD)
    ap.add_argument("--min-note-len-ms", type=float, default=DEFAULT_MIN_NOTE_LEN_MS)
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    a = ap.parse_args(argv)

    payload = _run_inside_pitch_venv(
        json.loads(a.json_spec), a.backend, a.onset_threshold, a.frame_threshold,
        a.min_note_len_ms, a.min_confidence)
    # 最后一行必须是 JSON（basic-pitch 会往 stdout 打进度，调用方取最后一行）
    print(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
