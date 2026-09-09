# 调研报告：MiaCode 源码取证 + Visual Maimai 文档侧取证（完整版）

> 来源：coding agent 调研（2026-09-10），委托子代理完成。

## 0. 取证对象与结论摘要

| 对象 | 版本 | 取证方式 | 置信度 |
|---|---|---|---|
| MiaCode | commit `641e4d0d46170ed9673186a549637b45620bc697`（2026-06-26，仍活跃） | 完整克隆 /tmp/MiaCode，逐行读解析器源码 | 高 |
| Visual-Maimai-Release | commit `259c553a3232899ef3576d2d7b6d9d792a993072`（2026-02-28） | 克隆（仅 README + 图片，无源码） | 低 |
| visual-maimai-manual.github.io | commit `8c8525441160b52be1954a4aa1d24a1973c88cd9`（2026-05-19） | 克隆，读全部相关 md | 中 |
| VisualMaimai-Integration-MaiMuriDX | commit `279b9951fd42df3d8a903e8109a19b1a7da17587`（2026-05-30） | 克隆，读 README | 中 |

核心结论：**MiaCode 的报错体系非常适合直接作为"零报错 maidata 校验器"蓝本**——它内部区分"结构错误（lenient+strict 都报）"与"非典范写法警告（仅 strict）"，全部规则有源码位置可查。VM 源码未开源，其 simai 导入的报错规则无法从文档取证，只能确认其内置无理检测的存在与行为范围。

# A. MiaCode

## A.1 解析架构

**文件布局**（src/core/chart/parser/）：SimaiNativeParser.h（API+数据结构）/ SimaiNativeParser.cpp（主解析器 1634 行）/ SimaiNativeParser.Driver.cpp（1333 行：消息常量、token 分发、主循环、验证报告）/ SimaiNativeParser.Slide.cpp（355 行）/ SimaiNativeParser.TouchTap.cpp（193 行）/ SimaiNativeParser.StrictChecks.cpp（165 行）。相关：assets/reference/slide_data.json（608 个 slide shape key + 8 个 wifi key）、src/core/chart/document/SimaiDocument.cpp（meta 头）、src/tools/muri/*（无理检测）。

**单一内核 + 双模式**（parseInternal(text, strictMode, ...)，Driver.cpp:607-1173）：

| 模式 | 公开入口 | strictMode | 用途 |
|---|---|---|---|
| Lenient | parseForTimeline() | false | 时间轴 marker 抽取，每次编辑都跑；坏 token 尽量跳过 |
| Strict | validateSyntax() | true | "语法检查"tab 的权威诊断 |

**数据结构**：SimaiNativeMessage{line,col,endCol,message}（**行列均 1-based**）；SimaiNativeParseResult{ok,errors,warnings,...}；SimaiNativeValidationReport{ok,errorCount,warningCount,strictNoteCount,strictErrorCount,issues}；SimaiNativeValidationIssue{line,col,endCol,severity(Error|Warning),rawMessage,displayMessage}。

**验证报告构建** buildValidationReport()（Driver.cpp:1257-1333）：空文本 → 1 条 Error "谱面为空。"直接返回；否则跑 strict pass，strict error→UI Error、strict warning→UI Warning，不合并不降级；report.ok = (errorCount==0)；lenient 结果完全不参与诊断。

**错误定位与本地化**：中文映射表 zhExactMap（269-285）+ zhPrefixMap（287-357 按顺序前缀匹配）；前缀加 [错误]/[警告]；UI 支持按 issue-type key 屏蔽（||#ignore <key>）；规范化（Modify 菜单）被 strict 错误门控（ChartNormalization.cpp）。

**主循环要点**（Driver.cpp:669-916）：按行按字符扫描。`(` BPM（找同行 `)`，未闭合→错）、`{N}` 分音、`<HS*N>` 倍速（旧 `HS*N>` 写法已不识别）、`/` 合击分隔、`` ` `` 连组分隔、`,` 拍分隔（同时产出 beatMarkers）、`||` 注释（内联拍号解析并重启小节）、行尾裸 `E`（大小写均可，重置 hs=1.0）。纯数字串如 `123` 拆成 3 个单键 tap（SimaiNativeParser.cpp:482-494），由 finalizeEachGroup（760-884）按 each 组规则处理。

**token 分发** parseToken（455-506）：全角检查（464-467）→ strict-only 的 `?/!/@`、`$` 位置检查（473-480）→ 纯数字串拆分 → touch 前缀（`C` 或 `A/B/D/E`+数字1-8，isTouchPrefix 324-334）→ 数字开头走 tap/hold/slide → 否则 Invalid note。

## A.2 语法支持与扩展（与官方 simai 的差异）

官方基础语法全覆盖：tap/hold/slide/同头星/多段链/wifi/touch/touch_hold/break/EX/(BPM)/{N}/E/||注释/&头/,//`` ` ``。

**MiaCode 接受的扩展（多数来自 Majdata 生态）**：

| 扩展 | 语法 | 位置 | 备注 |
|---|---|---|---|
| 地雷 | `m`（大小写均可） | tap/hold/touch/slide：`1m`、`1bm`、`A1m`、`1-3[2:1]m`、`1w5m[8:1]` | 独立贴图、无判定、不计 each、计入物量；slide 上位置不限不报警告 |
| tap-star 材质 | `$`/`$$` | 仅纯 tap：`1$`、`1$$`、`1b$x`；`h`+`$` 非法 | 仅改材质不改判定 |
| slide 头 tap 材质 | `@` | `1@-4[8:1]`、`1bx@-4[8:1]` | 与 `?`/`!` 组合未定义→报错 |
| 无头 slide | `?`（渐入）/`!`（立即） | `1?-4[8:1]`、`1!-4[8:1]` | hasHeadStar=false |
| 延迟 slide | `[bpm#b:n]`、`[#秒]`、`[bpm#秒]`、`[秒##b:n]`、`[秒##秒]`、`[秒##bpm#b:n]` | slide/wifi 的 `[]` 内 | `###` 或 >3 个 `#` 拒绝 |
| 倍速指令 | `<HS*N>`（含负值，默认接受 g_allowNegativeHs=true） | 行内指令 | |
| 中心 touch 别名 | `C1`/`C2` | 归一化为 C，strict 额外 Warning | |
| 零时长 hold/touch-hold | `1h`、`Ch`、`Ch[]` | 视作时长 0 接受 | |
| 内联拍号 | `,,|| 4/4` | 重启小节 | |

**大小写规则（不对称，易踩坑）**：tap/hold 的 `h` 大小写均可；`m` 大小写均可；tap/slide 头的 `b`/`x` **大写 B/X 直接拒绝**；touch 修饰符 `h/f/b` 仅小写（大写 H/F/B 落"Invalid touch modifier"）；touch 区域字母必须大写；slide shape 仅 `w` 小写触发 wifi 分发（`1W5` 走 tap 路径报错）。

**修饰符顺序不强制**（逐字符迭代）：`1bx`=`1xb`=`1bhxf`。"Hold 修饰符顺序无效"实际对应：重复修饰符（`1bb`/`1xx`/`1hh`/`1mm`）、大写 B/X、`]` 后有 `h`——**不是顺序问题**。

## A.3 报错/警告规则清单

### A.3.1 结构与指令类（lenient+strict 均 Error，除非注明）
| # | 英文消息 | 中文 | 触发条件 | 位置 |
|---|---|---|---|---|
| 1 | Unterminated BPM block | BPM 块未闭合 | `(` 后本行找不到 `)` | Driver.cpp:716-718 |
| 2 | Invalid BPM value | BPM 数值无效 | BPM 非数字或 ≤0 | 722-724 |
| 3 | Unterminated beat block | 分拍块未闭合 | `{` 后本行找不到 `}` | 740-742 |
| 4 | Invalid beat value | 分拍数值无效 | 分音非整数或 ≤0 | 746-748 |
| 5 | Unterminated <HS*> bracket | 括号未闭合 | lenient 静默 break；strict 报 | 786-789 |
| 6 | Invalid <HS*N> value | 数值无效 | 非数字、==0、（开关关时 <0） | 793-800 |
| 7 | Invalid terminal marker placement: E | 终止标记 E 位置无效 | E 混在音符里（非行尾裸 E） | SimaiNativeParser.cpp:306-308 |
| 8 | Chart is empty. | 谱面为空。 | 全文 trim 后为空 | Driver.cpp:1276-1290 |

### A.3.2 tap/hold 类（两种模式均 Error）
| # | 英文消息 | 中文 | 触发条件 | 位置 |
|---|---|---|---|---|
| 9 | Invalid hold duration: token | Hold 时值无效 | `[`/`]` 单边；`]`≤`[`；多组括号；有 `[]` 无 `h`；时长解析失败 | TouchTap.cpp:126-176 |
| 10 | Invalid hold modifier sequence: token | Hold 修饰符顺序无效 | 后缀含 `h`；重复 b/x/h/m/$；大写 B/X；未知字符；h 与 $ 共存 | 142-151 |
| 11 | Invalid note: token | 音符无效 | 非 touch 前缀、非 1-8 开头（`9`、`0`、`abc`） | Driver.cpp:505-506 |

### A.3.3 touch 类（两种模式均 Error）
| # | 英文消息 | 中文 | 触发条件 | 位置 |
|---|---|---|---|---|
| 12 | Invalid touch token: token | Touch 音符无效 | 只有 `]` 无 `[`；多组括号；`]` 后缀含 `h` | SimaiNativeParser.cpp:404-450 |
| 13 | Invalid touch-hold duration: token | TouchHold 时值无效 | `]`≤`[`；时长解析失败 | 412-414 |
| 14 | Touch duration requires 'h': token | Touch 时值需要 'h' | 有 `[]` 无 `h` | 457-459 |
| 15 | Invalid touch modifier: token | Touch 修饰符无效 | 修饰符非 h/f/b/m/M（**含 x、大写 H/F/B**） | 425-441 |

### A.3.4 slide 类
| # | 英文消息 | 中文 | 严重度/模式 | 触发条件 | 位置 |
|---|---|---|---|---|---|
| 16 | Invalid note: token unknown shape X | 音符无效 | Error/两种 | shape key 不在 slide_data.json（608 keys 白名单；含非法 V 组合如 1V32） | Slide.cpp:192-196 |
| 17 | Invalid slide duration: token | Slide 时值无效 | Error/两种 | `[]` 无 `:`、beats≤0、`###`、>3 个 `#`、时长≤0（slide 时长必须严格正） | Slide.cpp:275-278 |
| 18 | Invalid note: token | 音符无效 | Error/两种 | 头修饰符含大写 B/X、重复 b/x/@/?/!、h、@+?/! 组合；core 含 x/h/@/?/!/$ | SimaiNativeParser.cpp:192-261 |
| 19 | Invalid slide duration placement: token | Slide 时值块位置可能导致转谱错误 | **Error/strict-only** | slide token 含 >1 个 `[…]`（每段独立时值） | Slide.cpp:160-166 |
| 20 | Invalid slide chain syntax: token | Slide 段链语法无效 | **Error/strict-only** | chain 非 1-8 开头、shape 后数字位数错、孤儿数字（1v35-7 的 5）、孤儿括号、无 shape | Slide.cpp:130-133 |
| 21 | Invalid '*' slide branch (must omit the slide head) | '*' 同头分支语法无效 | **Error/strict-only** | `*` 后带数字头（5q2[4:1]*5p8[4:1]）；每 token 只报一次 | Slide.cpp:48-65 |
| 22 | Invalid break slide modifier position: token | Break Slide b 位置可能导致转谱错误 | **Warning/strict-only** | b 不是紧挨第一个 `[` 之前（2bv-3[4:1]）；仍 parse | Slide.cpp:77-98 |
| 23 | Slide head modifier (?, !, @) may only appear between slide head and shape | ?、!、@ 只能出现在… | **Error/strict-only** | 出现在无 shape 的 token（1!、1h@）或第一个 shape 字符之后（1-5![8:1]） | Driver.cpp:385-427 |
| 24 | Tap-star modifier ($) may only appear within a tap | $ 只能出现在 tap 中 | **Error/strict-only** | $ 出现在 touch/slide/hold/带[]/含 h 的 token（A1$、1-5$[8:1]、1h$） | Driver.cpp:435-453 |

**slide 端点合法性机制**：shape（1-5、8V37、2p4、1qq6、4w8）经 normalizedSlideLookupKey（1374-1417，剥离修饰符）在 slide_data.json 白名单查表；查不到报 #16（两种模式都报）。`^` 被 canonicalSlideKey（1004-1027）折叠成等价 `</>`。语法层面由 strict-only #20 检查。

### A.3.5 分隔符/格式类（strict-only）
| # | 英文消息 | 中文 | 触发条件 | 位置 |
|---|---|---|---|---|
| 25 | Repeated separator '//' is not allowed | 不允许使用连续分隔符 '//' | 1//5 | Driver.cpp:808-815 |
| 26 | Repeated separator '``' is not allowed | 不允许使用连续分隔符 '``' | `` 1``5 `` | 832-841 |
| 27 | Separator '/' or '`' is missing an adjacent note | 分隔符缺少相邻音符 | 左侧无音符（,/7）；到 ,/行尾右操作数缺失（7/,） | 817-823, 843-851 |
| 28 | Missing beat separator ',' | 缺少拍间分隔符 ',' | 剥掉控制块后含 note 字符且整行无 , 的非 E 行（注释行豁免） | StrictChecks.cpp:41-123 |
| 29 | Unmatched closing bracket 'X' | 未匹配的右括号 | ]/}/) 无对应左括号或类型不匹配 | 125-139 |
| 30 | Unclosed bracket 'X' | 未闭合的左括号 | 文件结尾括号栈残留 | 142-145 |
| 31 | Missing ',' between note and directive | 音符与指令间缺少 ',' | 音符后紧跟指令（1{16}(120)） | Driver.cpp:642-657 |

### A.3.6 全角字符类（两种模式均 Error，token 级前置检查，detectFullwidthSyntaxIssueMessage 263-302）
7 类：全角数字、全角触摸区域字母（ＡＢＣＤＥ）、全角修饰符（ｂｘｈｆｍ）、全角括号（（）［］｛｝【】）、全角分隔符（，：／；＃）、全角 Slide 符号（－＜＞＾ｖＶｐｑｓｚｗ＊？！）、全角拉丁字母（FF21-FF3A/FF41-FF5A）。

### A.3.7 分音数值警告（strict-only Warning）
| # | 英文消息 | 中文 | 条件 | 位置 |
|---|---|---|---|---|
| 32 | Invalid beat value for strict mode: N (must be a positive divisor of 384) | 分拍数值可能导致转谱错误：N | {N} 0<N≤384 且非 384 约数（{7}/{9}）；仍接受 | Driver.cpp:764-771 |
| 33 | Beat value above 384 may cause transfer issues: N | 分拍数值大于 384 可能导致转谱错误：N | {N}>384；接受但 clamp 警告 | 749-763 |

（{0}/负数/非数字是 Error #4；{1024} 是 #33 不是 Error。）

## A.4 时长/BPM/分音/meta/E 校验细节

- **hold 时长**（parseHoldDurationSignature 546-615）：接受 `[b:n]`（=240×n/(bpm×b) 秒）、`[bpm#b:n]`、`[#秒]`、`[bpm#秒]`。`#`>1、beats≤0、BPM≤0 → Error。分子 n 未限制 ≥0，负 n 由 qMax(0.0) 静默归零（不报错）。
- **slide 时长**（617-714）：`[b:n]`（等待=1 拍=60/bpm）；`[bpm#b:n]`/`[bpm#秒]`（等待 60/bpm）；`[秒##b:n]`/`[秒##秒]`/`[秒##bpm#b:n]`。slide 时长必须 >0（`[4:0]` 报错），等待可为 0。
- **BPM**：toDouble 失败或 ≤0 → Error；小数合法；任何 (BPM) 指令即使数值不变也重启小节相位。
- **meta 头**：解析器只消费 `&whole_time_signature=`（非法→不报错 fallback 4/4）。序列化兼容策略值得借鉴：`&first` 空值存为 0（空 `&first=` 会崩 MajdataPlay 的 double.Parse）；空 `&wholebpm/&pvstart/&pvlen` 保存时剔除；`&clock_count` 去重默认 4。
- **E**：整行裸 E（大小写均可，trim 后）→ 跳过该行；行中 token 起始的 E 且行尾为终结标记（允许 || 注释跟随）→ 结束解析并重置 hs=1.0；E 混在 token 中（1E）→ #7。

## A.5 与官方规范差异总结

**比官方/Majdata 更严（生成器必须避免）**：全角字符逐类报错；// 与连续反引号禁止；分隔符缺操作数禁止；strict 缺逗号、括号不匹配；大写 B/X 拒绝；touch 上 x 拒绝；slide 头上 h 拒绝；chain 逐字符校验（孤儿数字）；* 后带 head 数字拒绝；**多段 slide 每段带时值 strict 报错**；{N} 非 384 因数/>384 警告；?/!/@、$ 位置检查；slide 时长为 0 拒绝。

**比官方更宽松/扩展（不报错但跨工具注意）**：C1/C2 接受（警告）；修饰符任意顺序；h/m 大小写均可；负 <HS*> 默认接受；零时长 hold；Majdata 扩展全家桶（$ $$ @ ? ! m、延迟 slide、<HS*N>）；负时长分子静默归零；内联拍号；&whole_time_signature。

## A.6 谱面检查（无理/Muri 检测）

权威文档 docs/specs/muri/MURI_DETECTION_SPEC.md（336 行）；代码 src/tools/muri/*。

**5 种检测**（MuriKind，src/common/MuriTypes.h:16-22）：

| Kind | 中文 | 含义 |
|---|---|---|
| SlideTooFast | 内无 | slide/wifi 最终完成落在临界窗外 |
| SlideHeadTap | 外无 | slide/wifi 头/额外 pad-down 提前判定后续 tap/hold/star |
| TapOnSlide | 撞尾 | slide/wifi 尾/路径与后续 tap/hold/star 碰撞 |
| Overlap | 叠键 | 同一 pad 同时按 |
| MultiTouch | 多押 | 所需手数 >2（仅运行时） |

**关键时间常量**（spec §4）：判定时基 180 TPS；tap available 150ms；slide critical 233.3ms；tap-on-slide 阈值 1/180≈5.6ms；touch-on-slide 24/180≈133.3ms；slide available 24h。slide critical 判定有独立 +50ms shift 容错分支（与 MaiMuriDX 同款，wifi 不用）。

**告警级别**（spec §6 + MuriDiagnosticLabels.cpp:427-460）：SlideTooFast、Overlap 恒为 Muri（硬错）；SlideHeadTap 在（带保护 / 0<gap≤50ms 且无启动 tap / gap≥150ms）降 Warning；TapOnSlide 在（gap>150ms、gap≤静态阈值、带保护）Warning，静态阈值默认 200ms、可调 150-250ms（MuriConfig.h:10-12）；MultiTouch 涉 touch 且非 touch 手数≤2 时 Warning。受保护音符显示为 `protected tap 8x`。

**双分析器**：运行时（180TPS 手势模拟）+ 静态参考（几何+时间阈值）合并去重进同一面板；条目锚定"首个涉事物件"精确 line/col。

# B. Visual Maimai（源码未开源，文档侧取证，置信度低）

## B.1 取证局限
Visual-Maimai-Release 仓库仅 README（28 行宣传性内容 + B 站视频链接）与图片，**无源码或校验文档**。simai 解析/报错规则只能从手册站与第三方集成项目侧面推断。

## B.2 已知检查规则（内置无理检测）
手册 docs/guide/gui.md：
- 内置无理检测："Visual Maimai自带无理检测，假设你写了一个2-3-4的三押，或者是5-6,3-2长条,4绝赞Tap，那么就会在轨道区域旁提示"（gui.md:52-53）
- "上一个/下一个无理"菜单导航（gui.md:52）；偏好设置可调"无理检测"（gui.md:61）
- Touch 规则："由于一个手可以覆盖大部分的判定区，所以没有无理（除非Tap+Touch）"（gui.md:115）
- 多押/叠键类是 VM 原生检测（示例全是 Overlap/MultiTouch 类）；**没有证据表明 VM 原生含 slide 类无理（内无/外无/撞尾）检测**

## B.3 MaiMuriDX 集成佐证
VisualMaimai-Integration-MaiMuriDX README："将 MaiMuriDX 的无理检测能力接入编辑器内置的『谱面检查』流程。**在 VM 原有检查之外**，额外报告内屏、多押、叠键、外键、撞尾等谱面无理问题。"反推（中等置信度）：① VM 有内置"谱面检查/无理报告"面板与实时触发流程；② "额外报告"表明 VM 原有检查**不覆盖**这五类；③ 报告 UI 可扩展。该项目附带 muri-example.txt 是"AI 生成的全是无理的谱面样例"——与 AI 生成谱面测试场景直接相关。

## B.4 simai 导入导出行为（手册 gui.md:13-27, 81-82, 106-110）
- 打开：同目录有 maidata.txt 自动导入；无音频无法创建 note
- 保存：maidata.txt + chart.json（元数据）；导出以 Majdata 兼容为目标；zip 可改后缀 adx 供 AstroDX
- 分音：常规用 {1}{2}{4}{8}{16} 与三连音 {6}{12}{24}（VM 生成谱面大概率只用 384 因数分音）
- 星星启动拍默认 1/4 拍（打头后隔一个四分音星星才启动）

## B.5 不确定项（如实声明）
1. VM 的 simai 导入报错规则完全无公开资料（手册无记载，web_search 未命中）；闭源实现，报错行为只能黑盒测试获得。
2. VM 原生无理检测完整类别清单未知。
3. VM 导入对非法 token 是"跳过丢 note"还是"报错拒收"未知。
4. "额外报告"推断 VM 原生无 slide 类检测——中等置信度。

# 来源清单

- MiaCode：https://github.com/fanfaredash/MiaCode @ 641e4d0d46170ed9673186a549637b45620bc697（2026-06-26）
  - 关键文件：src/core/chart/parser/SimaiNativeParser.{h,cpp,.Driver.cpp,.Slide.cpp,.TouchTap.cpp,.StrictChecks.cpp}；assets/reference/slide_data.json；docs/specs/chart/CHART_DIAGNOSTICS_AND_NORMALIZATION_SPEC.md、SLIDE_DELAY_AND_HEAD_MATERIAL_SPEC.md；docs/specs/muri/MURI_DETECTION_SPEC.md；src/common/MuriConfig.h、MuriTypes.h；src/core/specs/SimaiParserSpec.cpp（1217 行回归测试）
- Visual-Maimai-Release：https://github.com/CH3COOOHH/Visual-Maimai-Release @ 259c553a3232899ef3576d2d7b6d9d792a993072
- VM 手册站：https://github.com/Visual-Maimai-Manual/visual-maimai-manual.github.io @ 8c8525441160b52be1954a4aa1d24a1973c88cd9（docs/guide/gui.md、docs/guide/make-charts.md、docs/index.md）
- MaiMuriDX 集成：https://github.com/Starrah/VisualMaimai-Integration-MaiMuriDX @ 279b9951fd42df3d8a903e8109a19b1a7da17587

# 存疑项

1. g_allowNegativeHs：头注释说"Default off"，代码 SimaiNativeParser.cpp:107 为 true（默认接受负 HS）。已按代码为准，后续版本若翻转需复查。
2. MiaCode 规格文档行号与当前 commit 有少量偏移，本报告行号以当前 commit 实际代码为准。
3. MURI_DETECTION_SPEC §2（无理检测）未经逐行代码核对，该节以 spec 文档为准。
4. 未覆盖旁支：SimaiParserSpec.cpp（1217 行回归测试，可作生成器验收用例）、SimaiDocumentSpec.cpp、ChartBatchTransformSpec.cpp、samples/（mine_demo、negative_hs_demo）——如需要可进一步提取"零报错样例集"。
