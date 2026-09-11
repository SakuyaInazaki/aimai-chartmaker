"""声源分离（Demucs v4）。

对应设计文档 §3 L2 音轨层：把整曲分成 drums / bass / other / vocals 四轨，
每轨各自做 onset 检测，构成"踩音候选池"（知识 002 切轨的输入）。

默认模型 `htdemucs`（CPU/MPS 上快 4 倍），量产可换 `htdemucs_ft`。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

STEM_NAMES = ("drums", "bass", "other", "vocals")


def pick_device(prefer: str = "auto") -> str:
    """选择推理设备。MPS 在 M 系列上可用；Demucs 对 MPS 支持随版本而变，
    出错时调用方应回退 CPU。"""
    import torch

    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate(
    wav_path: Path,
    out_dir: Path,
    model_name: str = "htdemucs",
    device: str = "auto",
    force: bool = False,
    shifts: int = 0,
) -> dict:
    """跑 Demucs，把四个 stem 写到 `out_dir/<stem>.wav`。

    返回：{"stems": {name: path}, "elapsed_sec": float, "device": str,
           "model": str, "cached": bool}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {n: out_dir / f"{n}.wav" for n in STEM_NAMES}
    if all(p.exists() for p in paths.values()) and not force:
        return {
            "stems": {k: str(v) for k, v in paths.items()},
            "elapsed_sec": 0.0,
            "device": "cached（复用已有 stems，未重跑分离）",
            "model": model_name,
            "cached": True,
        }

    import torch
    import soundfile as sf
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    t0 = time.time()
    model = get_model(model_name)
    model.eval()

    dev = pick_device(device)
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    # (samples, ch) → (ch, samples)
    x = torch.from_numpy(wav.T).float()
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    ref = x.mean(0)
    x = (x - ref.mean()) / (ref.std() + 1e-8)

    def _run(d: str):
        model.to(d)
        with torch.no_grad():
            return apply_model(
                model, x[None].to(d), device=d, shifts=shifts,
                split=True, overlap=0.25, progress=False,
            )[0]

    try:
        sources = _run(dev)
    except Exception as exc:  # MPS 上部分算子可能不支持 → 回退 CPU
        if dev == "cpu":
            raise
        print(f"[stems] {dev} 失败（{type(exc).__name__}: {exc}），回退 CPU")
        dev = "cpu"
        sources = _run(dev)

    sources = sources.cpu() * ref.std() + ref.mean()
    for name, src in zip(model.sources, sources):
        sf.write(str(out_dir / f"{name}.wav"), src.numpy().T, sr)

    elapsed = time.time() - t0
    return {
        "stems": {k: str(v) for k, v in paths.items()},
        "elapsed_sec": elapsed,
        "device": dev,
        "model": model_name,
        "cached": False,
        "sr": sr,
    }


def load_stem_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """读取 stem 并转单声道、重采样到分析用采样率。"""
    import librosa

    y, _sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr
