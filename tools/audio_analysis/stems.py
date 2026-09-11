"""声源分离（Demucs v4）。

对应设计文档 §3 L2 音轨层：把整曲分成若干 stem，每轨各自做 onset 检测，
构成"踩音候选池"（知识 002 切轨的输入）。

## 两个模型（v0.5 新增六路）

| 模型 | stems | 许可 | 用途 |
|---|---|---|---|
| `htdemucs`（默认） | drums / bass / other / vocals | MIT ✅ | 迭代期主线，RTF ≈ 0.10–0.11（MPS） |
| **`htdemucs_6s`**（v0.5 新增） | drums / bass / other / vocals / **guitar** / **piano** | MIT ✅ | 把"合成器主旋律 / 钢琴 / 吉他 riff"从 `other` 里拆出来 |

**为什么加六路**（用户 2026-09-12）：官方谱大量踩合成器主旋律、钢琴、吉他 riff、
采样音效，四路管线把它们全塞在 `other` 一路里 —— n=40 标定实测 `other` 的 recall
只有 0.267，而 **16.8% 的官方 note 什么 stem 都不落**。拆细是为了把这两部分捞回来。

⚠️ **htdemucs_6s 的已知代价**（Demucs README 原文）：6 源模型是在 4 源模型上追加
训练的实验性模型，**piano 源质量较差**（官方原话 "the piano source is not working
great"），guitar 一般；它的 drums/bass/vocals 与四路模型**不完全相同**（重新训练过）。
所以本项目**不拿六路替换四路**，而是两套并存、分目录缓存，由标定决定各轨怎么用。

## 输出目录约定（v0.5 起按模型名分目录，不覆盖已有缓存）

```
<out>/stems/                    # htdemucs（历史默认，路径不变，向后兼容）
<out>/stems_htdemucs_6s/        # htdemucs_6s
```

## 落盘格式

默认写 44.1 kHz 立体声（与历史缓存一致）。六路分离会把单曲的 stem 体积翻 1.5 倍，
本机磁盘紧张时可用 `compact=True` 写**单声道 22.05 kHz PCM_16**——这正是下游
（`load_stem_mono` / librosa onset / basic-pitch）实际消费的格式，体积约 1/8，
不损失任何分析精度（basic-pitch 内部也是重采样到 22050 Hz 单声道）。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

STEM_NAMES = ("drums", "bass", "other", "vocals")
STEM_NAMES_6S = ("drums", "bass", "other", "vocals", "guitar", "piano")

# 模型 → 该模型输出的 stem 名（顺序与 demucs 的 `model.sources` 一致）
MODEL_STEMS: dict[str, tuple[str, ...]] = {
    "htdemucs": STEM_NAMES,
    "htdemucs_ft": STEM_NAMES,
    "hdemucs_mmi": STEM_NAMES,
    "mdx_extra": STEM_NAMES,
    "htdemucs_6s": STEM_NAMES_6S,
}

DEFAULT_MODEL = "htdemucs"
COMPACT_SR = 22050


def stems_for(model_name: str) -> tuple[str, ...]:
    """该模型会输出哪些 stem（未知模型按四路兜底）。"""
    return MODEL_STEMS.get(model_name, STEM_NAMES)


def stems_dir_for(out_dir: Path, model_name: str = DEFAULT_MODEL) -> Path:
    """按模型名给 stems 目录命名。

    `htdemucs` 保持历史路径 `<out>/stems`（已有 40 首缓存不能失效），
    其余模型写 `<out>/stems_<model>`。
    """
    out_dir = Path(out_dir)
    if model_name == DEFAULT_MODEL:
        return out_dir / "stems"
    return out_dir / f"stems_{model_name}"


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
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    force: bool = False,
    shifts: int = 0,
    compact: bool = False,
    model=None,
) -> dict:
    """跑 Demucs，把各个 stem 写到 `out_dir/<stem>.wav`。

    参数：
        out_dir: **已经按模型分好的** stems 目录（用 `stems_dir_for` 生成）
        compact: True 时写单声道 22.05 kHz PCM_16（体积 1/8，分析口径无损）

    返回：{"stems": {name: path}, "elapsed_sec": float, "device": str,
           "model": str, "cached": bool, "compact": bool}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = stems_for(model_name)
    paths = {n: out_dir / f"{n}.wav" for n in names}
    if all(p.exists() for p in paths.values()) and not force:
        return {
            "stems": {k: str(v) for k, v in paths.items()},
            "elapsed_sec": 0.0,
            "device": "cached（复用已有 stems，未重跑分离）",
            "model": model_name,
            "cached": True,
            "compact": compact,
        }

    import torch
    import soundfile as sf
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    t0 = time.time()
    # 批量跑多首时可以把 `model` 传进来复用，省掉每首一次的权重加载/远程核对
    model = model if model is not None else get_model(model_name)
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
        arr = src.numpy().T                       # (samples, ch)
        _write_stem(out_dir / f"{name}.wav", arr, sr, compact=compact)

    elapsed = time.time() - t0
    return {
        "stems": {k: str(v) for k, v in paths.items()},
        "elapsed_sec": elapsed,
        "device": dev,
        "model": model_name,
        "cached": False,
        "compact": compact,
        "sr": COMPACT_SR if compact else sr,
    }


def _write_stem(path: Path, arr: np.ndarray, sr: int, compact: bool = False) -> None:
    """落盘一条 stem。compact=True 时转单声道 22.05 kHz PCM_16。"""
    import soundfile as sf

    if not compact:
        sf.write(str(path), arr, sr)
        return
    import librosa

    mono = arr.mean(axis=1) if arr.ndim == 2 else arr
    if sr != COMPACT_SR:
        mono = librosa.resample(np.ascontiguousarray(mono, dtype=np.float32),
                                orig_sr=sr, target_sr=COMPACT_SR)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:                                # PCM_16 会削顶 → 先归一
        mono = mono / peak
    sf.write(str(path), mono.astype(np.float32), COMPACT_SR, subtype="PCM_16")


def load_stem_mono(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """读取 stem 并转单声道、重采样到分析用采样率。"""
    import librosa

    y, _sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr
