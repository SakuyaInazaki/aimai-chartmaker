#!/usr/bin/env python3
"""`python3 -m tools.chart_check <maidata.txt>` 入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import check_file
from .report import render_json, render_markdown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.chart_check",
        description="谱面检查器：语法 / 手序 / 配置 / 密度 / 采音 / 深度 / 外部规则 七层")
    ap.add_argument("maidata", help="maidata.txt 路径")
    ap.add_argument("--level", type=float, default=None,
                    help="对照定数（缺省时从 &lv_N 粗取）")
    ap.add_argument("--analysis", default=None,
                    help="song_analysis.json（缺省时找 maidata.txt 同目录）")
    ap.add_argument("--inote", type=int, default=None,
                    help="检查哪一个难度（缺省取最大的非空 &inote_N）")
    ap.add_argument("--json", dest="json_out", default=None, help="把 JSON 写到这个文件")
    ap.add_argument("--quiet", action="store_true", help="只打结论行")
    a = ap.parse_args(argv)

    rep = check_file(a.maidata, inote=a.inote, level=a.level,
                     analysis_path=a.analysis)
    if a.json_out:
        Path(a.json_out).write_text(render_json(rep), encoding="utf-8")
    if a.quiet:
        print(f"{rep.verdict}｜错误 {rep.count('错误')} / 警告 {rep.count('警告')} / "
              f"提示 {rep.count('提示')}")
    else:
        print(render_markdown(rep))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
