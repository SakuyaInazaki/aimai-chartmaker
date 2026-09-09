# simai 报错与校验逻辑（调研整理）

> **状态**：v1.0 定稿（2026-09-10）。收录 MajdataEdit / MajdataView / SimaiSharp / MaiLib / maidata-rs（源码级）+ **MiaCode（源码级）+ Visual Maimai（文档侧取证，置信度低）**。原始报告见 `docs/research/`（parser-source-analysis.md、miacode-vm-research.md）。
> **配套文档**：`docs/simai-syntax.md`（语法规范 v1.0）。
> **调研原则（用户决策）**：报错判定逻辑以**成熟解析器的实际实现行为**为准（majdata、visual maimai、miacode 等）。

## 0. 目标与关键事实

- 本项目校验器目标：让生成端的校验行为与社区工具一致，实现"输出前零报错"。
- 关键事实：**Majdata 生态没有数字错误码表**——报错全部是本地化字符串文本 + 异常消息（带行列定位）。
- 兼容性事实（MajdataView README 自述）："部分语法规则较为宽松，可以在 Majdata 中运行的谱面可能无法在其他软件中（如 maipad、simai、Astro）运行"。
- 结论：**语法 lint 以 SimaiSharp（最严格的词法+语法解析器）为基准**，MajdataEdit SyntaxCheck 作为编辑器侧辅助，语义级无理检测参考 MajdataEdit MuriCheck 与 MaiMuriDX 规则（知识库 008-015）。

## 1. 校验环节总览

| 工具 | 校验环节 | 报错形式 |
|------|----------|----------|
| MajdataEdit | 打开谱面自动 / 手动"语法检查"（级别 0=禁用/1=警告/2=阻止播放导出） | 错误+警告列表，双击跳转行列 |
| MajdataEdit（解析） | maidata.txt 读取/序列化 | 异常弹窗"在maidata.txt第N行:\n{msg}" |
| MajdataView | majdata JSON → note 实例化（加载） | ErrText 屏显"在第N行发现问题:\n{msg}" |
| SimaiSharp | 词法+语法解析 | 5 类异常（带 line+character） |
| MaiLib | 词法扫描+语法解析 | UnexpectedCharacterException / ParsingException |
| MajdataEdit MuriCheck | 语义级（非语法）：多押/撞尾无理 | "[多押无理]…" "[撞尾无理]…" |
| MiaCode | 编辑器语法检查（**lenient+strict 双模式**） | 结构化消息 {line,col,endCol,severity,message}，中文映射 |
| MiaCode Muri | 语义级无理检测（运行时 180TPS 模拟 + 静态参考） | 5 类 MuriKind，条目锚定首个涉事物件行列 |
| Visual Maimai | 内置谱面检查（闭源，范围窄） | 多押/叠键类提示 + 可扩展无理报告面板 |

## 2. MajdataEdit 编辑器侧语法检查（SyntaxModule/SyntaxCheck.cs）

- 校验入口：打开谱面自动跑 + 手动按钮（MainWindowCore.cs:341-348）；级别设置 EditorSettingPanel.xaml:140-145（0=禁用/1=警告/2=开启-阻止播放与导出，MainWindowCore.cs:1126, 1138）；结果窗口 MuriCheckResult 双击跳转行列。
- 两类内置消息（Langs/*.resx，zh-CN）：
  - `[语法错误] "{0}"({1}L,{2}C)解析失败，可能存在语法错误`（zh-CN.resx:299-301）
  - `[警告] 谱面应当以"E"结尾`（zh-CN.resx:470-472）——**唯一内置警告类**；触发条件：按 `,` 分割后最后一段 ≠ "E"（SyntaxCheck.cs:70-74）

### 判定明细（SyntaxCheck.cs，行号精确）

| # | 判定 | 触发条件 | 位置 |
|---|------|----------|------|
| 1 | 空 note 段 | `1/` 或 `1//2` 产生空段 | 97-101 |
| 2 | BPM 括号 | `(`/`)` 数量>1 或不配对 | 178-187 |
| 3 | 分音括号 | `{`/`}` 数量>1 或不配对 | 189-198 |
| 4 | `()`/`{}` 位置 | 必须位于一拍开头且相邻连续（`(){…}` 或 `{}()…`，中间不能有间隙） | 206-221, 253-279 |
| 5 | 内容类型 | `{}` 必须整数、`()` 必须数字 | 281-290 |
| 6 | Hi-Speed | `<HS*x>`：括号不配对/位置错误/body 非数字 | 224-250, 304-353 |
| 7 | `[…]` 参数 | `[`/`]` 数量必须相等；非 slide 的 note 必须恰 1 对且以 `]` 结尾（slide 允许末尾跟 `b`） | 361-394 |
| 8 | Tap | 首字符必须 1-8；`$` 只允许 `1$`/`1$$`；长度 2 只允许 `Nb`/`Nx`/两位连写；长度 3 必须恰为 b+x | 828-868 |
| 9 | Hold 头部 | `[` 前长度 2/3/4 组合限制；短 hold（总长≤4）后只允许 b/x（防 `2h[]`、`2h[`）；时长体最短 2 字符（最短合法 `#2`） | 874-917, 433-441, 450 |
| 10 | Hold 时长 | `[#秒]` 秒≥0；`[bpm#x:y]` bpm>0 且比例合法；`[x:y]` x>0 整数、y≥0 整数 | 456-463, 794-802 |
| 11 | Slide 结构 | 长度≥3；起点必须数字、类型符合法、V 需拐点数字、终点数字；每段 `[` 必须紧跟段尾；"有的段有时长有的没有"→错误；`#` 数量>3 →错误 | 476-590, 604-605 |
| 12 | Slide 参数 | 四类模式校验：`[x:y]`、`[bpm#len]`(bpm>0)、`[x##len]`(x≥0)、`[x##bpm#len]`(bpm>0, x≥0) | 611-650 |
| 13 | Slide 几何 | 见下"Slide 端点几何约束" | 684-721 |
| 14 | Touch | 长度 1/2/3；`C`/`C1`/`Cf`/`C1f`/`A1f..E8f`；传感字母 A-E、键位 1-8（**不接受 C2**） | 963-982 |
| 15 | 兜底 | 以上都不匹配 → 通用 SyntaxError | 417-423 |

### Slide 端点几何约束（MajdataEdit SlidePathCheck 684-721 + MajdataView 佐证）

| 形状 | MajdataEdit 约束 | MajdataView 报错消息（JsonDataLoader.cs） |
|------|------------------|------------------------------------------|
| `-` | 起点≠终点且间隔≥2 键 | "-星星至少隔开一键"（相对终点须 3-7） |
| `^` | 间隔≠0 且≠4（不能同键、不能对向） | "^星星不合法"（同键或对向） |
| `v` | 间隔≠0 且≠4 | "v星星不合法"（仅禁对向） |
| `s`/`z`/`w` | 终点必须对向（180°） | "s/z/w星星尾部错误"（终点非对向） |
| `V` | 拐点间隔==2；终点距拐点≥2；起点≠终点 | "V星星终点不合法"/"V星星拐点只能隔开一键"（拐点相对位 3 或 7） |
| `q`/`p`/`qq`/`pp` | 无硬性限制 | — |

⚠️ 注意官方规范与 MajdataEdit 的分歧：官方 `v` 只禁自身+对向（`1v1` 非法）；MajdataView 的 `v` 只禁对向。生成器按**官方**执行（禁自身+对向）。

## 3. MajdataView 加载期报错（Assets/Scripts/JsonDataLoader.cs）

- 机制：majdata JSON 反序列化 → 逐 note 实例化（协程 InstantiateNotes），任何异常 catch 后写屏 ErrText："在第{rawTextPositionY+1}行发现问题：\n{e.Message}"（733-737）。
- 报错消息清单：
  - "组合星星有错误\nSLIDE CHAIN ERROR"（879, 893：连锁 slide 时长标注混合/重复；913, 927：读到多余数字/全部段落无时长）
  - "不允许Wifi Slide作为Connection Slide的一部分"（1001）
  - "-星星至少隔开一键\n-スライドエラー"（1336）
  - "^星星不合法\n^スライドエラー"（1384）
  - "v星星不合法\nvスライドエラー"（1405）
  - "s星星尾部错误/z星星尾部错误/w星星尾部错误"（1463/1475/1513）
  - "V星星终点不合法"（1492, 1498）、"V星星拐点只能隔开一键"（1502）
  - "Keys out of range: {key}"（1551，MirrorKeys 内部防御）
  - 无效形状（如 `v` 同键）→ SLIDE_PREFAB_MAP 查找失败 KeyNotFoundException → 落入通用 ErrText（896-901 隐式路径）

## 4. SimaiSharp 异常体系（Internal/Errors/*.cs，全部带 line+character）

| 异常 | 触发点 |
|------|--------|
| SimaiException（基类） | 所有解析异常，携带 line/character |
| UnexpectedCharacterException | 单 `|`；传感字母后缺 1-8；Duration 内数字解析失败；slide 段读到 EOF 缺端点；Tempo/Subdivision 非数字 |
| UnterminatedSectionException | `( { [` 未闭合或重复起始符 |
| UnsupportedSyntaxException | 不认识的字符；状态机未覆盖 token；slide 后跟非 `b` 装饰符（含 `x`） |
| ScopeMismatchException（带 correctScope） | Decorator/Slide/Duration/SlideJoiner 出现在无主 note 的拍点；Note 内出现 Tempo/Subdivision；note 内 SlideJoiner；Slide 内出现 Tempo/Subdivision |
| InvalidSyntaxException | 位置 token 无法解析 |

- 注意：SimaiSharp 对"缺少结尾 E"**不报错**（FinishTiming 回退 maxFinishTime，Deserializer:129）；对 hold `[##2.5]` 会抛 UnexpectedCharacterException。
- SimaiSharp 是 astrodx 官方解析器 = 本项目语法 lint 基准。

## 5. MaiLib（Neskol，MaichartConverter 引擎）

- SimaiScanner token 表（62-192）：**不认识 `! ? @ \` w`** → UnexpectedCharacterException（"At Line N: Unexpected char …"）；扫描异常被吞掉转为 EOS（31-41）。
- Simaiparser ParsingException 触发点：each 组解析失败(103)；绝对时间小节不支持(148)；sustain 缺失(207/303)；V 拐点不匹配(373/413)；无起始键(436)；slide 类型非法(682)；slide 记法位置异常(686)；`[` 未闭合(698)；段无时长(720)；缺时长(741)；无 BPM 上下文(835)；时长模式全不匹配(919)。
- ICodeBlock 异常：ComponentMissingException / ExcessiveComponentsException / UnexpectedStringSuppliedException。

## 6. maidata-rs（WIP，仅语法参考）

- Key 枚举 1-8，KeyParseError::InvalidKey(char)；TouchSensor 枚举 A1-E8+C；nom 组合子解析。错误类型很薄，只做参考。

## 7. MajdataEdit MuriCheck（语义级无理检测，非语法）

- 多押>2 检测："[多押无理]…可能形成了N押"（MuriCheck.xaml.cs:139-258；zh-CN.resx MultNoteError1/2）。
- slide 撞尾检测："[撞尾无理]…二者间隔Nms"（slideDetect 260-470, SlideError）——基于 slide_time.json 的物理判定数据；精度设置 DefaultSlideAccuracy。
- 更完整的无理规则（叠键/外键/撞尾/多押阈值）见 `.agent/knowledge/` 008-015 与 MaiMuriDX 检测标准。

## 8. MiaCode（fanfaredash/MiaCode，commit 641e4d0，源码级取证）

### 8.1 解析架构（单一内核 + 双模式）

- 代码：`src/core/chart/parser/SimaiNativeParser.{h,cpp,.Driver.cpp,.Slide.cpp,.TouchTap.cpp,.StrictChecks.cpp}`。
- 双模式：**Lenient**（parseForTimeline，时间轴抽取，坏 token 尽量跳过，每次编辑都跑）/ **Strict**（validateSyntax，"语法检查"权威诊断）。
- 验证报告 = strict 结果直通（error→Error、warning→Warning，不合并不降级），`ok = (errorCount==0)`；lenient 结果完全不参与诊断。
- 消息结构：`{line, col, endCol}`（1-based，endCol 用于编辑器波浪线）、severity(Error|Warning)、rawMessage/displayMessage（中文映射表 zhExactMap/zhPrefixMap）；支持按 issue-type key 屏蔽（`||#ignore <key>`）。
- 校验门控：规范化（Modify 菜单）被 strict 错误阻止（errorCount>0 放弃重写）。

### 8.2 报错规则清单（34 类，详见 miacode-vm-research.md A.3）

**结构与指令类（lenient+strict 均 Error）**：BPM 块未闭合 / BPM 数值无效（非数字或 ≤0）/ 分拍块未闭合 / 分拍数值无效（非整数或 ≤0）/ `<HS*>` 括号未闭合（仅 strict）/ `<HS*N>` 数值无效 / 终止标记 E 位置无效（E 混在 token 中）/ 谱面为空。

**tap/hold 类（均 Error）**：Hold 时值无效（单边括号、`]`≤`[`、多组括号、有 `[]` 无 `h`、解析失败）/ Hold 修饰符序列无效（`]` 后缀含 `h`、重复修饰符、大写 B/X、`h`+`$` 共存）/ 音符无效（非 touch 前缀、非 1-8 开头）。

**touch 类（均 Error）**：Touch 音符无效 / TouchHold 时值无效 / Touch 时值需要 'h'（有 `[]` 无 `h`）/ Touch 修饰符无效（非 h/f/b/m/M，**含 `x` 与大写 H/F/B**）。

**slide 类**：shape 不在 slide_data.json 白名单（608 keys；均 Error）/ Slide 时值无效（无 `:`、beats≤0、`###`、时长≤0——**slide 时长必须严格正**）/ 头修饰符非法（大写 B/X、重复、`h`、`@`+`?/!`）/ **分段时值（token 含 >1 个 `[…]`）→ strict Error** / chain 语法（孤儿数字、无 shape）→ strict Error / `*` 分支带 head 数字 → strict Error / **Break Slide b 不紧挨第一个 `[` → strict Warning** / `?/!/@` 位置错误 → strict Error / `$` 出现在非 tap → strict Error。

**分隔符/格式类（strict-only）**：`//` 连续 / `` `` `` 连续 / 分隔符缺相邻音符 / 整行缺 `,` / 括号不匹配或未闭合 / 音符与指令间缺 `,`。

**全角字符类（均 Error，7 类）**：全角数字、全角触摸字母、全角修饰符、全角括号、全角分隔符、全角 slide 符号、全角拉丁字母——**对 LLM 生成文本尤其重要（LLM 极易混入全角字符）**。

**分音数值警告（strict-only）**：`{N}` 0<N≤384 且非 384 约数 → Warning（仍接受）；N>384 → Warning（clamp）。

### 8.3 关键校验细节

- **slide 端点合法性**：shape 经 normalizedSlideLookupKey（剥离修饰符）查 `assets/reference/slide_data.json` 白名单（608 个 shape key + 8 个 wifi key），查不到即报错（双模式）；`^` 被 canonicalSlideKey 折叠为等价 `</>`。**该 JSON 是官方 64 格关系表图的机读替代**。
- **时长**：hold 接受 `[b:n]`/`[bpm#b:n]`/`[#秒]`/`[bpm#秒]`；slide 等待/时长 6 形式；负时长分子被静默归零（不报错，隐藏坑）。
- **BPM**：小数合法；≤0 报错；任何 `(BPM)` 指令即使数值不变也重启小节相位。
- **meta 序列化兼容策略（借鉴）**：`&first` 空值补 0（空 `&first=` 会崩 MajdataPlay 的 double.Parse）；空 `&wholebpm/&pvstart/&pvlen` 保存时剔除；`&clock_count` 去重默认 4。
- **E**：行尾裸 E（大小写均可，可跟 `||` 注释）结束解析并重置 hs。

### 8.4 与官方差异小结

- 更严（生成器必须避免）：全角字符；连续/缺操作数分隔符；大写 B/X；touch 上 `x`；slide 头 `h`；`*` 带 head；**分段时值**；slide 零时长；384 非约数警告。
- 更宽/扩展（跨工具注意）：C1/C2、修饰符任意顺序、负 HS、零时长 hold、Majdata 扩展全家桶（`$`/`@`/`?`/`!`/`m`/`<HS*N>`）。

### 8.5 谱面检查（Muri 无理检测）

- 5 类 MuriKind（与 MaiMuriDX 一致）：SlideTooFast(内无)/SlideHeadTap(外无)/TapOnSlide(撞尾)/Overlap(叠键)/MultiTouch(多押)。
- 常量（MURI_DETECTION_SPEC.md §4）：判定时基 **180 TPS**；tap available 150ms；slide critical 233.3ms（+50ms shift 容错分支，wifi 不用）；tap-on-slide 5.6ms；touch-on-slide 133.3ms。
- 级别：SlideTooFast/Overlap 恒 Muri；SlideHeadTap 带保护 / 0<gap≤50ms 无启动 tap / gap≥150ms 降 Warning；TapOnSlide gap>150ms / ≤静态阈值（默认 200ms，可调 150-250ms）/ 带保护降 Warning；MultiTouch 涉 touch 且非 touch 手数≤2 降 Warning。
- 双分析器：运行时（180TPS 手势模拟）+ 静态参考（几何+时间阈值）合并去重。
- **回归测试 `SimaiParserSpec.cpp`（1217 行）可直接借鉴为生成器验收用例**。

## 9. Visual Maimai（源码未开源，文档侧取证，置信度低）

- **取证局限**：CH3COOOHH/Visual-Maimai-Release 仅 README + 图片，无源码/校验文档；手册站与 web_search 均无 simai 导入报错规则记录——**VM 的解析报错行为只能黑盒测试获得**。
- **内置检查**（手册 gui.md）：多押/叠键类提示（三押、长条+Tap 组合示例）、"上一个/下一个无理"导航、tap+touch 特例；**无证据表明 VM 原生含 slide 类无理（内无/外无/撞尾）检测**。
- **MaiMuriDX 集成 Mod 佐证**："在 VM 原有检查之外，额外报告内屏、多押、叠键、外键、撞尾"→ 反推 VM 原生检查范围更窄（中等置信度）；VM 报告 UI 可扩展。
- **导入导出**：同目录 maidata.txt 自动导入；保存 maidata.txt + chart.json；导出以 Majdata 兼容为目标；VM 生成谱面分音基本只用 384 约数；星星启动拍默认 1/4 拍。
- **对生成器的价值**：有限（闭源黑盒）。其"Tap+Touch 算多押"的规则与 MaiMuriDX 的 TouchGroup 规则不一致，以更完善的 MaiMuriDX/MiaCode 规则为准。

## 10. 本项目校验器实现蓝本

架构参照：**MiaCode 双模式设计**（内部 lenient 预览 + 发布 strict 门禁），规则取各解析器**交集**。

分层校验（生成器输出前必须全绿）：

1. **词法/语法层**（对齐 SimaiSharp + MiaCode strict 交集）：token 合法性、括号闭合、装饰符/slide 符集合、作用域、缺 E、**全角字符 7 类检查（LLM 输出高危项）**、分隔符连续/缺操作数、slide 时长严格为正。报错带行列（1-based + endCol）。
2. **结构层**：BPM/分音顺序与 384 约数（非约数=Warning 级红线）；**slide 端点合法性用 MiaCode `slide_data.json` 608-key 白名单（官方 64 格表的机读替代）**；连锁 slide 只用"总时长在最后"形式（分段独立时长触发 MiaCode strict 报错）；同头禁 `/`；`*` 后不带 head；时长格式 ∈ 官方 6 形式；meta 转义 + 序列化兼容（`&first` 非空补 0、空可选字段剔除、`&clock_count` 默认 4）。
3. **语义/可玩性层**（知识库 008-015 阈值 + MiaCode Muri 常量）：叠键（同判定区 <33.3ms）、外键（启动拍后 200ms 同侧；MiaCode gap 分级）、撞尾（-50ms~+200ms；MiaCode 静态阈值默认 200ms 可调 150-250ms）、Hold 尾留空、Touch 成组、密度红线、难度校准（知识 004）。
4. **双检**：SimaiSharp lint 零异常 + MajdataEdit SyntaxCheck 零错误零警告 + MiaCode strict 零错误（最终以实际工具跑谱验证）。
5. **验收测试**：借鉴 MiaCode `SimaiParserSpec.cpp`（1217 行）与 `samples/` 构建"零报错样例集 + 应报错样例集"回归测试。

## 11. 来源清单

- `docs/research/parser-source-analysis.md`（本节 2-7 的源码依据，含全部文件+行号）
- `docs/research/miacode-vm-research.md`（本节 8-9 的源码/文档依据）
- `docs/simai-syntax.md` §6/§8（官方规范与分歧）
- MajdataView：https://github.com/LingFeng-bbben/MajdataView ；MajdataEdit：https://github.com/LingFeng-bbben/MajdataEdit
- SimaiSharp：https://github.com/reflektone-games/SimaiSharp ；MaiLib：https://github.com/Neskol/MaiLib ；maidata-rs：https://github.com/xen0n/maidata-rs
- MiaCode：https://github.com/fanfaredash/MiaCode （commit 641e4d0；含 slide_data.json、MURI_DETECTION_SPEC、SimaiParserSpec.cpp）
- Visual Maimai：https://github.com/CH3COOOHH/Visual-Maimai-Release ；手册站 https://github.com/Visual-Maimai-Manual/visual-maimai-manual.github.io ；MaiMuriDX 集成 https://github.com/Starrah/VisualMaimai-Integration-MaiMuriDX
