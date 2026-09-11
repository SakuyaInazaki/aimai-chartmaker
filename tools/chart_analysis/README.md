# tools/chart_analysis — 官方谱逐小节 note 密度分析

把 simai 谱面正文解析成 note 事件时间轴，再统计**逐小节密度曲线**，
作为音频分析层「强度曲线 → 谱面密度」映射的 ground truth 形状先验。

- 语法依据：`docs/simai-syntax.md`（v1.0）
- 语料：`resource/official-chart/` 下 388 个官方 ST 谱（MASTER / Re:MASTER，定数 13.0–14.5；本机留存、不入库）
- 元数据 / 校验基准：`resource/player-preview/data/manifest.json`（含官方分项计数与定数）
- 分析结论：`docs/research/official-chart-density-curves.md`
- 每谱一行汇总：`docs/research/data/official-chart-density-summary.csv`

## 模块

| 文件 | 职责 |
|------|------|
| `simai_parser.py` | simai 正文解析：BPM / 分音 / `{#秒}` / 逗号推进 / tap / hold / slide / touch / each / 修饰符 → `NoteEvent` 时间轴 |
| `density.py` | 逐小节统计、曲线归一与 32-bin 重采样、休息段与平台段、变点检测分段、k-means 聚类、五段模板检验 |
| `corpus.py` | 官方谱文件发现 + manifest 元数据关联 |
| `cli.py` | 命令行入口 |

## 用法

```bash
cd <仓库根目录>

# 1. 解析覆盖率 + 与 manifest 官方 note 数对账 + 与知识 004 分布对照
PYTHONPATH=tools python3 -m chart_analysis.cli validate [--list-mismatch]

# 2. 写出汇总 CSV（入库）与逐小节明细（out/，gitignore）
PYTHONPATH=tools python3 -m chart_analysis.cli summary

# 3. 全库曲线统计 / 形状聚类 / 五段模板检验 / 高潮位置 / 休息段
PYTHONPATH=tools python3 -m chart_analysis.cli analyze

# 4. 单谱逐小节曲线（文件名关键字匹配）
PYTHONPATH=tools python3 -m chart_analysis.cli chart SPICY
```

作为库使用：

```python
import sys; sys.path.insert(0, "tools")
from chart_analysis.simai_parser import parse_chart
from chart_analysis.density import chart_density

res = parse_chart(open("maidata_inote.txt", encoding="utf-8").read())
print(res.counts)          # {'taps':…, 'hold':…, 'slide':…, 'touch':…, 'breaks':…, 'notes':…}
d = chart_density(res)
print(d.raw)               # 逐小节 note 数
print(d.resampled)         # 32-bin 归一曲线
```

依赖：Python 3.10+、numpy。k-means 与轮廓系数为本模块自带实现，**不需要 sklearn**。
测试：`PYTHONPATH=tools python3 -m pytest tests/test_chart_analysis.py -q`。

## 口径说明（重要）

### 时间轴

- **拍为主轴**：一个逗号推进 `4 / 分音` 拍、`(240 / BPM) / 分音` 秒。
- **小节 = 4 拍**（simai 无小节线、无拍号指令，隐含 4/4；见语法文档 §1.3）。
  小节号 `= floor(拍位置 / 4)`，从 0 起。变速不影响小节划分。
- **曲线范围**：首个 note 所在小节 → 末个 note 所在小节（不含开头的纯静默小节）。
- `{#秒}` 槽：按当前 BPM 折算成拍位；若此前没有 BPM 标记则记 error。
- slide 启动拍按官方语义取 `60 / BPM`（固定一拍），时长括号里显式给出等待时间的形式优先。

### note 计数

与 manifest（mai-notes / 查分器口径）对齐：`notes = taps + hold + slide + touch + breaks`。

| 类别 | 计入规则 |
|------|----------|
| `taps` | 单点 tap；**slide 的星星头也算 TAP**（`*` 同头多 slide 只有一个星头） |
| `hold` | `h` 系列；`3h`（省略时长）= 伪 TAP，仍按 HOLD 计 |
| `slide` | 每条**滑轨**算 1；`*` 分隔的同头多 slide 各算 1；`1-4q7-2[1:2]` 这类**首尾相接的连锁 slide 整体算 1**（分段数另存 `ParseResult.slide_chain_segments` 供对照） |
| `breaks` | 带 `b` 的 note **从其本类中移出**、单独计入 BREAK |
| `touch` | ST 谱理论上没有；遇到照常统计并单列 |

> ⚠️ **已知口径歧义**：「滑轨 BREAK 该记进 `breaks` 还是 `slide`」两种约定在 388 谱上的
> 分项吻合度几乎相同（本实现的 A 方案 slide 383 / breaks 379 对；另一方案 386 / 377 对），
> 无法判定 manifest 用的是哪一种。**总 note 数不受影响**，只影响分项。

### 密度指标

- **休息小节**：note ≤ 1 的小节。
- **低密段**：连续若干小节，归一化密度 ≤「全曲中位密度 × 0.5」。
- **密度台阶（平台段）**：3 小节中值平滑 → 按 0.125 量化档位 → 长度 ≥ 4 小节的同档连续段。
- **32-bin 重采样**：按面积加权把逐小节曲线压到 32 个等长区间，再除以自身峰值。
- **峰值位置**：单小节最大值的相对位置；另有更稳健的 `peak_position_smooth`
  （4 小节滑动均值最大处的窗口中心），用来定位「高潮段」而非一次性爆发小节。
- **等分五段画像** `segment_profile`：等分 5 段求均值、再除以 5 段中的最大值 → 与知识 001 的 T 同尺度。
- **结构对齐五段** `structure_profile()`：L2 变点检测分 5 段，**必须带 `min_len_frac=0.10`**——
  不加最小段长约束时 DP 会把「末尾一两个收尾小节」单独切出来（实测 37% 的谱第 5 段短于全曲 5%），
  从而伪造出「尾部渐弱」。

## 解析器已知短板

1. **不做端点合法性校验**：slide 起终点关系表（语法文档 §8.1）未接入，解析器只负责
   读出形状与键位，不判断 `1v1` 之类非法写法。校验是 `docs/simai-error-checking.md` 的事。
2. **不解析 meta 头**：只吃 `&inote_N=` 之后的正文。
3. **连锁 slide 的分段时长**只做累加（`duration` = 各段之和），不保留逐段时长。
4. **伪 EACH（`` ` ``）** 按独立成员拆开，不模拟「晚 1ms」的时间偏移（官方谱语料中未出现）。
5. **`w`（Wifi）** 按 1 条滑轨计、不展开成 3 条引导星（语料中 191 个谱用到 `w`、共 568 次，
   这些谱的 slide 分项 187/191 与官方计数一致，说明官方也按 1 条计）。
6. **容错写法**会告警但照常解析：`1[4:1]`（缺 `h`，按 HOLD）、`1-4b[8:1]`（b 在 `[` 前）、
   `6>b3[8:1]`（b 在终点键前）、空的 `{}`（沿用上一分音）。
7. **小节划分假定 4/4**。曲子里的变拍（3/4、7/8 段落）在 simai 中只能靠分音+逗号数模拟，
   本工具仍按 4 拍切小节，此类曲目的小节边界会与乐谱小节错位。
