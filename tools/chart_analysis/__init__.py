"""官方谱逐小节密度分析工具包。

子模块
------
- :mod:`.simai_parser` — simai 正文解析（时间轴 + note 事件）
- :mod:`.density` — 逐小节密度统计、曲线重采样、聚类、五段模板检验
- :mod:`.corpus` — 官方谱语料发现与 manifest 元数据关联
- :mod:`.cli` — 命令行入口（``python3 -m chart_analysis.cli --help``）
"""

from .simai_parser import NoteEvent, ParseResult, parse_chart  # noqa: F401

__all__ = ["parse_chart", "ParseResult", "NoteEvent"]
