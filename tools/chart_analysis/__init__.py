"""官方谱逐小节密度分析工具包。

子模块
------
- :mod:`.simai_parser` — simai 正文解析（时间轴 + note 事件）
- :mod:`.dx_filter` — DX 谱 → ST 可比视图（丢 touch 与非 ST 要素，时间轴不动）
- :mod:`.similarity` — 谱面相似度自检（逐小节完全相同 + n-gram 重合率）
- :mod:`.density` — 逐小节密度统计、曲线重采样、聚类、五段模板检验
- :mod:`.configs` — 配置识别（知识 017–030）+ 逐小节配置硬度分
- :mod:`.corpus` — 官方谱语料发现与 manifest 元数据关联
- :mod:`.cli` — 命令行入口（``python3 -m chart_analysis.cli --help``）
"""

from .simai_parser import NoteEvent, ParseResult, parse_chart  # noqa: F401

__all__ = ["parse_chart", "ParseResult", "NoteEvent"]
