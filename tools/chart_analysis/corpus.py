#!/usr/bin/env python3
"""官方谱语料加载：文件发现 + manifest 元数据关联。

语料位置（本机，不入库）：``resource/official-chart/<simai_id>-<曲名>-<mas|remas>.txt``
元数据：``resource/player-preview/data/manifest.json``（mai-notes 镜像，含官方
note 分项计数 ``notes/taps/hold/slide/touch/breaks`` 与定数 ``internal_level``）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = REPO_ROOT / "resource" / "official-chart"
MANIFEST = REPO_ROOT / "resource" / "player-preview" / "data" / "manifest.json"

_NAME_RE = re.compile(r"^(\d+)-(.*)-(mas|remas)$")
_DIFF = {"mas": "MASTER", "remas": "Re:MASTER"}


@dataclass
class ChartFile:
    path: Path
    simai_id: str
    title: str
    difficulty: str  # mas / remas
    # 来自 manifest 的官方元数据（缺失时为 None）
    internal_level: float | None = None
    level: str | None = None
    gt_notes: int | None = None
    gt_taps: int | None = None
    gt_hold: int | None = None
    gt_slide: int | None = None
    gt_touch: int | None = None
    gt_breaks: int | None = None
    bpm: str | None = None
    version: str | None = None

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8", errors="replace")

    @property
    def name(self) -> str:
        return self.path.stem


def load_manifest(path: Path = MANIFEST) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover(chart_dir: Path = CHART_DIR, manifest_path: Path = MANIFEST) -> list[ChartFile]:
    """列出全部官方谱文件并关联 manifest 元数据。"""
    if not chart_dir.exists():
        raise FileNotFoundError(f"未找到官方谱目录：{chart_dir}")
    man = load_manifest(manifest_path)
    by_simai: dict[str, dict] = {}
    charts_by_song: dict[str, list[dict]] = {}
    if man:
        for song in man["songs"].values():
            if song.get("simai_id"):
                by_simai.setdefault(str(song["simai_id"]), song)
        for c in man["charts"]:
            charts_by_song.setdefault(c["song_id"], []).append(c)

    out: list[ChartFile] = []
    for p in sorted(chart_dir.glob("*.txt")):
        if p.name.startswith("_"):  # 抓取脚本的日志等辅助文件
            continue
        m = _NAME_RE.match(p.stem)
        if not m:
            continue
        cf = ChartFile(path=p, simai_id=m.group(1), title=m.group(2), difficulty=m.group(3))
        song = by_simai.get(cf.simai_id)
        if song:
            cf.bpm = song.get("bpm")
            for c in charts_by_song.get(song["id"], []):
                if c["difficulty"] == _DIFF[cf.difficulty]:
                    cf.internal_level = c.get("internal_level")
                    cf.level = c.get("level")
                    cf.gt_notes = c.get("notes")
                    cf.gt_taps = c.get("taps")
                    cf.gt_hold = c.get("hold")
                    cf.gt_slide = c.get("slide")
                    cf.gt_touch = c.get("touch")
                    cf.gt_breaks = c.get("breaks")
                    cf.version = c.get("version")
                    break
        out.append(cf)
    return out


def level_bucket(internal_level: float | None) -> str:
    """把定数归入 13.0 / 13.5 / 14.0 / 14.5 档（向下取 0.5 的整倍）。"""
    if internal_level is None:
        return "未知"
    return f"{math_floor_half(internal_level):.1f}"


def math_floor_half(v: float) -> float:
    import math

    return math.floor(v * 2.0) / 2.0
