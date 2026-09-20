#!/usr/bin/env python3
"""谱面检查器（**验收工具**）：一份 `maidata.txt` 进，中文报告 + JSON 出。

它不是生成器，也不替谱师做判断。它做的只有一件事——**把 `AGENT.md` 的四条验收标准
拆成可以逐条核对的清单**，其中能机检的部分给出结论，不能机检的部分给出**复核清单**。

七层（与 `docs/simai-error-checking.md` §10 的四层校验同源，多出采音 / 深度 /
外部规则三层）：

===== ==================================================================
层     内容
===== ==================================================================
语法   `simai_parser` 严格解析 + `docs/simai-syntax.md` §6 零报错安全子集
       + `docs/st-chart-elements.md` §三 ST 黑名单，逐条报错/警告（带行列）
手序   `hands.assign()` → 不可行 / 绝对·硬·软无理 / 出张落点（知识 064）
       / 侧边双押与引导类型（知识 082）/ Hold 尾（012）/ 叠键（009）
配置   `configs_hand.detect_all()` → 按小节顺序的配置构成、种类数、主料与点缀；
       对照知识 073/074 给**提示**（不判错）
密度   note 总数 / NPS / 逐小节密度曲线 vs `T_density` 与密度地板（知识 031）、
       末段形态（001/031 §10）、给定定数时对照知识 004 区间
采音   有 `song_analysis.json` 时：逐段全踩 / 舍音 / 留白（知识 068）与骨架轨
       对照，标出「该有音却整段留白」与「采空音密集」供人看
深度   跑在最后、只消费前五层的统计块：手序难度 / note 种类占比 / 主料构成 与
       **同定数档官谱分布**对照，合成一条「写得太浅？」复核提示（知识 093）
外部规则 换一套无理口径：**MiaCode / MaiMuriDX** 的静态无理检测（外键 / 撞尾 /
       叠键），即 Visual Maimai 无理面板真正会弹的那些；与手序层并列不互相替代
       （见 `murilayer.py`）
===== ==================================================================

⚠️ **外部三检待接**：`docs/simai-error-checking.md` §10-4 要求
SimaiSharp lint + MajdataEdit SyntaxCheck + MiaCode strict 三方实跑，
**本机没有装这三个工具**，因此本模块的语法层只是**内部校验**——
它按三方的规则交集实现，但**不能代替真机跑谱**。报告里会显式写这一句。

CLI::

    python3 -m tools.chart_check <maidata.txt> [--level 13.5]
        [--analysis song_analysis.json] [--json out.json] [--inote 5]
"""

from __future__ import annotations

from .model import Issue, LAYERS, LEVELS  # noqa: F401
from .core import CheckReport, check_chart, check_file  # noqa: F401

__all__ = ["Issue", "CheckReport", "check_chart", "check_file", "LAYERS", "LEVELS"]
