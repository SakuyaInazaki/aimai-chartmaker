#!/usr/bin/env python3
"""把仓库根与 `tools/` 放进 `sys.path`，统一转出已有工具的入口。

`tools/` 不是常规包（没有 `__init__.py`），既有模块（`tools/calibration/sampling.py`）
也是这么做的；这里照搬同一套，保证 `python3 -m tools.chart_check` 与
`PYTHONPATH=tools python3 -m pytest` 两条路都能跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chart_analysis import configs_hand as configs_hand  # noqa: E402,F401
from chart_analysis import density as density_mod  # noqa: E402,F401
from chart_analysis import hands as hands_mod  # noqa: E402,F401
from chart_analysis.simai_parser import NoteEvent, ParseResult, parse_chart  # noqa: E402,F401

__all__ = ["REPO", "configs_hand", "density_mod", "hands_mod",
           "parse_chart", "ParseResult", "NoteEvent"]
