# 调研报告：社区 simai 解析器/校验器源码分析（完整版）

> 来源：coding agent 调研（2026-09-10），委托子代理完成。仓库克隆于本机 /tmp（majdata-view、majdata-edit、simaisharp、mailib、maidata-rs、majdata-wiki），调研时 commit 如下：
> MajdataView `5786452`（2025-02-06, v4.1.1）；MajdataEdit `963812b`（2024-12-21）；SimaiSharp `5ef06f9`；MaiLib `c76e94e`；maidata-rs `37668fb`。

## 关键背景

- **astrodx 主仓库已不含解析器**，其 README 指定官方开源解析库为 [SimaiSharp](https://github.com/reflektone-games/SimaiSharp)（本次已克隆并通读）。
- **Majdata 生态没有"错误码表"文档**（wiki 与 docs 均无），报错全部是本地化字符串文本 + 异常消息，无数字错误码。
- ⚠️ **MajdataView README "已知问题"第 2 条自述**："部分语法规则较为宽松，可以在 Majdata 中运行的谱面可能无法在其他软件中（如 maipad、simai、Astro）运行"。
- 因此"零报错"目标建议**以 SimaiSharp（最严格的词法+语法解析器）为交集基准**，MajdataEdit 校验器作为编辑器侧辅助。

---

# 一、语法规则（源码依据）

## 1. maidata.txt 文件级结构（& 指令）

- 指令行以 `&key=value` 开头；值可为多行，直到下一个 `&` 行（SimaiSharp SimaiFile.cs:74-90）。
- MajdataEdit 识别的键：`&title=` `&artist=` `&des=` `&first=`（起始秒数偏移，float）、`&lv_1..7=`、`&inote_1..7=`（MajdataEdit SimaiProcess.cs:56-84）；其余指令原样保留进 other_commands（如 `&wholebpm=`、`&bpms=`，SimaiProcess.cs:85-86, 存盘时原样写回 SaveData:104-119）。
- `&first` 是谱面时间起点（Serialize 中 `double time = first;` SimaiProcess.cs:140）。
- 读取期错误：任意异常 → MessageBox "在maidata.txt第N行:\n{message}"（SimaiProcess.cs:91-95，含行号）。

## 2. 词法 token 定义（SimaiSharp Tokenizer.cs —— 最权威）

- TokenType 枚举（TokenType.cs:5-47）：Tempo `(…)`、Subdivision `{…}`、Location、Decorator、Slide、Duration `[…]`、SlideJoiner `*`、TimeStep `,`、EachDivider `/` `` ` ``、EndOfFile `E`。
- 装饰符集合（Tokenizer.cs:25-30）：`f b x h m ! ? @ $`。
- Slide 类型符集合（Tokenizer.cs:32-40）：`- > < ^ p q v V s z w`。
- 空白符（忽略，Tokenizer.cs:42-51, 115-117）：空格、`\t` `\r`、U+2002/2008/3000、U+2028/2029；`\n` 递增行号（119-122）。
- Location：数字 `0-8` 或传感区字母 `A-E`+数字（143-180；`C` 可单独出现=中控键）。
- 注释：`||` 起至行尾（127-136）；**单个 `|` 抛 UnexpectedCharacterException**（129-130）。
- 单字符 `( { [ , * / ` E` 均为独立 token（88-125）。

## 3. 时间结构 / BPM / 拍号 / Hi-Speed

- 拍点推进：每遇到 `,` 时间推进 1 拍，拍长 = (60/BPM) × (4/拍号分母)（SimaiSharp Deserializer.cs:81-90 + TimingChange.cs:12-14；MajdataEdit `time += 1d/(bpm/60d)*4d/beats` SimaiProcess.cs:271）。
- `(bpm)` 必须为数字，可出现在任意拍点（SimaiSharp TempoReader.cs:13-29；MajdataEdit Serialize 同 175-193）。MajdataEdit 要求 `()` 位于一拍开头（SyntaxCheck.cs:206-221）。
- `{拍号}` 为数字；`{#秒}` 形式=显式设定每拍秒数（SimaiSharp SubdivisionReader.cs:11-48, TimingChange.SetSeconds:16-20）。
- MajdataEdit 支持 `<HS*数字>` 变速（Serialize:215-237；校验见 SyntaxCheck.cs:224-250, 304-353）。
- 结尾标记 `E`：SimaiSharp 将其作为 EndOfFile token，记录 FinishTiming（Deserializer.cs:119-121）；MajdataEdit 只对"最后一个逗号段正好是 E"做软校验——缺失只给 **警告**（SyntaxCheck.cs:70-74）。
- 注释 `||…` 跳过到行尾（SimaiProcess.cs:149-162）。

## 4. 基础 Note 语法（键位 1-8；触摸区 A-E）

- Tap：`1`-`8`；连写 `12` = 双押；`1b`=break，`1x`=EX，`1bx`/`1xb`（SyntaxCheck.cs IsTap 828-868）。
- Hold：`Nh[...]`；`2h` 短 hold（时长 0）；装饰 b/x 可在 h 前或后（wiki 建议顺序 `b x h`）；Break-Hold `1bh[4:1]` 合法（IsHold 874-917；wiki 怎样写谱？.md §2）。
- Touch：`A1`-`E8`、`C`、`C1`、`C2`（MajdataEdit IsTouch 只认 `C`/`C1`/`Cf`/`C1f`，**不接受 C2**，见存疑#2）；`f`=烟火花火；Touch 无 Break；TouchHold 仅 `Ch[4:1]`/`C1h`（IsHold 887-888；wiki §4）。
- 装饰符语义（SimaiSharp NoteReader.DecorateNote 90-129）：`f`=Fireworks、`b`=Break（覆盖类型）、`x`=Ex、`m`=Mine（**MajdataEdit 不支持 m**）、`h`=Hold、`!`/`?`=ForceInvalidate（无头星星，`!`=SuddenIn `?`=FadeIn）、`@`=ForceNormal（**MajdataEdit 不支持 @**）、`$`=ForceStar、`$$`=ForceStarSpinning（旋转星星）。
- 注意 SimaiSharp 对 Hold 读时长规则（NoteReader.ReadDuration 131-175）：`[#秒]`、`[bpm#x:y]`、`[x:y]`（=SecondsPerBar/(x/4)×y）；**不带 `h` 但带 `[…]` 的 note 也会变成 Hold**（133-134）。

## 5. Slide 语法（最复杂，三方互相印证）

**类型符 → 形状**（SimaiSharp SlideReader.IdentifySlideType 112-141）：
`-`=直线、`>`=顺时针环（方向随起点翻转，见 MajdataView MuriCheck 注释 MuriCheck.xaml.cs:260-280）、`<`=逆时针环、`^`=最短路径环、`q`/`p`=小圆曲线（顺/逆）、`qq`/`pp`=大圆曲线、`v`=折线(圆心)、`V`=L 折角、`s`/`z`=闪电、`w`=扇形(wifi)。

**端点几何约束**（三处源码）：

- `-`：起点≠终点且间隔≥2 键（MajdataEdit SlidePathCheck 697-702；MajdataView detectShapeFromText 1336：相对终点必须在 3-7 之间=至少隔一键）。
- `^`/`v`：MajdataEdit 要求间隔≠0 且≠4（不能同键、不能对向，692-696）；MajdataView：`^` 相对终点≠1(同键)/5(对向)（1382-1384），`v` 只禁对向（1405）。
- `s`/`z`/`w`：终点必须与起点**对向（180°）**（MajdataEdit 711-718 PointCompare==Opposite；MajdataView 1463/1475/1513 相对终点必须==5）。
- `V`：拐点必须隔 2 键（interval==2），终点距拐点≥2，起点≠终点（MajdataEdit 703-710）；MajdataView：拐点相对位须为 3 或 7（即顺/逆时针隔一键），终点相对位 2-5（1490-1502），否则"V星星终点不合法/V星星拐点只能隔开一键"。
- `q`/`p`/`qq`/`pp`：无硬性限制（MajdataEdit 719-720 直接 return true）。
- 键位合法性：所有端点是 1-8（PointCheck SyntaxCheck.cs:996；SimaiSharp 依赖 tokenizer 只产 0-8）。

**Slide 时长/参数 `[ … ]`**（SimaiSharp SlideReader.ReadDuration 160-224，附官方注释引用 w.atwiki.jp/simai/pages/25.html）：

- `[x:y]` = x 分 y 拍（SecondsPerBar/(x/4)×y）；
- `[秒]`（无冒号）= 直接秒数；
- `[bpm#x:y]` = 按 bpm 的 x 分 y 拍，且星星等待=该 bpm 的 1 拍（path.delay=SecondsPerBar@bpm，186-190）；
- `[bpm#秒]` = 星星等待 bpm 1 拍 + 移动秒数；
- `[秒##秒]` = 等待秒数##移动秒数（184-185）。
- 组合/连锁 slide：`1>5-8[1:1]`（多段共享一个时长）或 `1>5[1:4]-8[1:1]`（每段独立时长）；**混合写法（有的段有时长有的没有）→ 报错**（MajdataView JsonDataLoader.cs:868-894, 925-927 "组合星星有错误 SLIDE CHAIN ERROR"）。
- 同头 slide：`1-3[8:1]*-4[8:1]`，`*`=SlideJoiner（SimaiSharp Deserializer.cs:116-118 作用域检查；MajdataEdit getSameHeadSlide 383-401 把后续段标记 isSlideNoHead）。
- Break slide：`b` 紧贴 `[` 前（`1-4b[8:1]`）；Slide 本体 break 记 isSlideBreak，星星头 break 记 isBreak（MajdataEdit SimaiProcess.cs:463-499 的判定：b 后面是 `[` 或 b 在串尾 → break slide，否则星星头 break）。
- 无头 slide：`!` 或 `?` 在首键后（MajdataEdit 449-458；SimaiSharp Decorator `!`/`?` 113-119）。
- Wifi(`w`) 不能作为连锁 slide 的一段（MajdataView 998-1001 "不允许Wifi Slide作为Connection Slide的一部分"）。
- Slide EX（`1-4x[8:1]`）：MajdataEdit 接受（IsSlide 923-957 允许 b/x 头），SimaiSharp 在 DecorateSlide 中只认 `b`，**其他装饰符抛 UnsupportedSyntaxException**（SlideReader 99-109）→ 见存疑#4。

## 6. Each / 伪 Each

- `/` 分隔同拍多押；`` ` `` 分隔伪多押（ForceBroken，先后相差微小时间）。MajdataEdit 实现：`` ` `` 每组间隔 1/128 拍=1.875/bpm 秒（SimaiProcess.cs:245-258）；wiki 文档写"相差0.001秒"（存疑#5）。
- `0`（或 `0/`）开头的拍点=强制真 Each（SimaiSharp Deserializer.cs:67-71 ForceEach；Serializer 输出 `0/` 前缀 Serializer.cs:57-58）。

## 7. 官方序列化输出格式（SimaiSharp Serializer.cs —— "零报错"生成格式的直接依据）

- 开头 `(tempo){subdivisions}`（15-16）；每拍输出 note 后写 `,`（45）；结尾写 `E`（47）。
- Each 分隔符：ForceBroken → `` ` ``，否则 `/`；ForceEach → 前缀 `0/`（55-58）。
- Note 序列化顺序（Note.WriteTo 42-76）：位置 → `x` → `m` → `!`/`?` → `@`/`$`/`$$` → `h[#秒]`（固定 7 位小数）→ 各 slide 路径以 `*` 相连。
- SlidePath 序列化（SlidePath.WriteTo 31-40）：各段形状 + 结尾 `b`（若 break）+ **统一输出 `[delay##duration]` 秒制形式**（`{delay:0.0000000}##{duration:0.0000000}`）。
- 形状→字符映射（SlideSegment.WriteTo 21-54），注意 RingCw/RingCcw 输出时按起点位置选择 `<` 或 `>`。

---

# 二、报错/警告判定（源码依据）

## 1. MajdataEdit 编辑器侧语法检查（SyntaxModule/SyntaxCheck.cs，共约 40 类判定）

- 校验触发环节：打开谱面自动跑、手动点击"语法检查"按钮跑（MainWindowCore.cs:341-348；级别设置 EditorSettingPanel.xaml:140-145，0=禁用/1=警告/2=开启-阻止播放与导出，MainWindowCore.cs:1126, 1138）。结果窗口 MuriCheckResult 双击可跳转行列（MuriCheckResult.xaml.cs:24-33）。
- 错误文本（本地化 Langs/*.resx）：
  - `[语法错误] "{0}"({1}L,{2}C)解析失败，可能存在语法错误`（zh-CN.resx:299-301；en "SyntaxError at..."）
  - `[警告] 谱面应当以"E"结尾`（zh-CN.resx:470-472）——**唯一内置警告类**，触发条件：按 `,` 分割后最后一段≠"E"（SyntaxCheck.cs:70-74）。

错误判定明细（SyntaxCheck.cs，行号精确）：

1. 空 note 段（`1/` 或 `1//2` 之类产生空段）→ SyntaxError（97-101）。
2. BPM 括号：`(`/`)` 数量>1 或不配对 → 错误（178-187）；拍号 `{`/`}` 同理（189-198）。
3. `()`/`{}` 必须位于一拍开头且必须相邻连续（`(){…}` 或 `{}()…`，中间不能有间隙）（206-221, 253-279）。
4. `{}` 内容必须整数、`()` 内容必须数字（281-290）。
5. Hi-Speed `<HS*x>`：`<`/`>` 不配对、`HS*` 出现位置错误、body 非数字 → 错误（224-250, 304-353）。
6. `[…]` 参数：`[`/`]` 数量必须相等；非 slide 的 note 必须恰好 1 对且以 `]` 结尾（slide 允许末尾跟 `b`）（BodySyntaxCheck 361-394）。
7. Tap：首字符必须 1-8；`$` 只允许 `1$` 或 `1$$` 两种形式（IsTap 828-868, 837-848）；长度 2 只允许 `Nb`/`Nx`/两位连写（每个数字都 1-8）；长度 3 必须恰为 b+x。
8. Hold：头部（`[` 前）长度 2/3/4 的组合限制（874-917）；短 hold（总长≤4）后面只允许 b/x（433-441，防止 `2h[]`、`2h[`）；时长体最短 2 字符（450，最短合法 `#2`）；`[#秒]` 秒≥0（456）；`[bpm#x:y]` bpm>0 且比例合法（459-463）；`[x:y]` 要求 x>0 整数、y≥0 整数（RatioSyntaxCheck 794-802）。
9. Slide：长度≥3（476-477）；路径起点必须数字、类型符合法、V 需拐点数字、终点数字（506-555）；每段路径过 SlidePathCheck（559-560）；每段 `[` 必须紧跟段尾（565-579）；`1-4-6[4:1]-1[4:1]` 这种"有的段有时长有的没有"→ 错误（586-590）；`#` 数量>3 → 错误（604-605）；参数模式四类校验（611-650：`[x:y]`、`[bpm#len]`(bpm>0)、`[x##len]`(x≥0)、`[x##bpm#len]`(bpm>0, x≥0)）。
10. Slide 几何（SlidePathCheck 684-721）：见上文§5。
11. Touch：长度 1/2/3；`C`/`C1`/`Cf`/`C1f`/`A1f..E8f`；传感字母 A-E、键位 1-8（963-982）。
12. 兜底：以上都不匹配 → 通用 SyntaxError（417-423）。

## 2. MajdataEdit 解析期报错（SimaiProcess.cs）

- ReadData：任何异常 → 弹窗"在maidata.txt第N行:\n{msg}"（91-95），如 `&first=` 非数字 → float.Parse 异常。
- Serialize（谱面体）：`(`/`{` 内部解析失败（如 BPM 非数字）→ 静默 catch，return 0（282-286，仅 Console.WriteLine）——**容错但不报错，属隐藏坑**。getNotes() 失败同样静默返回空（376-380）。

## 3. MajdataView 加载期报错（Assets/Scripts/JsonDataLoader.cs）

- 环节：majdata JSON 反序列化 → 逐 note 实例化（协程 InstantiateNotes），任何异常被 catch 后写屏 ErrText："在第{rawTextPositionY+1}行发现问题：\n{e.Message}"（733-737）。具体异常消息：
  - "组合星星有错误\nSLIDE CHAIN ERROR"（879, 893：连锁 slide 时长标注混合/重复；913, 927：读到多余数字/全部段落无时长）。
  - "不允许Wifi Slide作为Connection Slide的一部分"（1001）。
  - "-星星至少隔开一键\n-スライドエラー"（1336，相对终点<3 或>7）。
  - "^星星不合法\n^スライドエラー"（1384，同键或对向）。
  - "v星星不合法\nvスライドエラー"（1405，对向）。
  - "s星星尾部错误/z星星尾部错误/w星星尾部错误"（1463/1475/1513，终点非对向）。
  - "V星星终点不合法"（1492, 1498：拐点相对位 7 时终点须 2-5；相对位 3 时终点须≥5）、"V星星拐点只能隔开一键"（1502）。
  - "Keys out of range: {key}"（1551，MirrorKeys 内部防御）。
  - 无效形状（如 `v` 同键）会因 SLIDE_PREFAB_MAP 查找失败抛 KeyNotFoundException 落入通用 ErrText（896-901 隐式路径）。
  - 另外 ScreenRecorder.cs:54 复用同一 ErrText UI。

## 4. SimaiSharp 异常体系（Internal/Errors/*.cs，全部带 line+character 定位）

- SimaiException（基类，SimaiException.cs:5-17，携带 line/character）。
- UnexpectedCharacterException：期望语法外的字符（UnexpectedCharacterException.cs:19-23）。触发点：单 `|`（Tokenizer 129-130）；传感字母后缺 1-8（179）；Duration 内数字解析失败（NoteReader 144, 156, 166, 172；SlideReader 182, 205, 214, 220-221）；slide 段读到 EOF 缺端点（SlideReader 148-149）；Tempo/Subdivision 非数字（TempoReader 19；SubdivisionReader 20, 36）。
- UnterminatedSectionException：`( { [` 未闭合或重复起始符（Tokenizer CompileSectionToken 196-213，抛出点 201-202）。
- UnsupportedSyntaxException：不认识的字符（Tokenizer 138-139）；状态机未覆盖的 token（Deserializer 124-125；NoteReader 82-83；SlideReader 76-77, 106-107 —— 含 slide 后跟非 `b` 装饰符）。
- ScopeMismatchException（带 correctScope 枚举 Note/Slide/Global，ScopeMismatchException.cs:13-19）：Decorator/Slide/Duration/SlideJoiner 出现在无主 note 的拍点（Deserializer 106-118）；Note 内部出现 Tempo/Subdivision（NoteReader 42-44）；note 内 SlideJoiner（69-70）；Slide 内部出现 Tempo/Subdivision（SlideReader 37-39）。
- InvalidSyntaxException：位置 token 无法解析（NoteReader 15-16）。
- 注意：SimaiSharp 对"缺少结尾 E"**不报错**（FinishTiming 回退为 maxFinishTime，Deserializer 129），与 MajdataEdit 的警告策略不同；对 hold `[##2.5]` 这种写法会抛 UnexpectedCharacterException（NoteReader 138-146 的 `[#…]` 分支）——见存疑#1。

## 5. MaiLib（Neskol，MaichartConverter 的引擎）

- SimaiScanner（Parser/SimaiScanner.cs）：全 token 表 62-192；**不认识 `! ? @ \` w`**（token 表没有这些 case）→ UnexpectedCharacterException（193-195, 217-223，消息 "At Line N: Unexpected char …"）；扫描异常被吞掉转为 EOS（31-41）。
- Simaiparser.cs 的 ParsingException（955-958，格式 "…after bar {bar} tick {tick}…"）；抛出点：解析 each 组失败 103；绝对时间小节不支持 148；sustain 缺失 207/303；V 拐点不匹配 373/413；无起始键 436；slide 类型非法 682；slide 记法位置异常 686；`[` 未闭合 698；段无时长 720；缺时长 741；无 BPM 上下文 835；时长模式全不匹配 919。
- ICodeBlock 异常（Parser/CodeBlocks/ICodeBlock.cs）：ComponentMissingException(9)、ExcessiveComponentsException(17)、UnexpectedStringSuppliedException(26)。

## 6. maidata-rs（xen0n，WIP，仅语法参考）

- Key 枚举仅 1-8，KeyParseError::InvalidKey(char)（src/insn/notes_ty.rs:14-31）；TouchSensor 枚举 A1-E8+C（notes_ty.rs:36-72）；nom 组合子解析（parser.rs:6-19）。错误类型目前很薄，只做参考。

## 7. 校验环节汇总

| 环节 | 工具 | 报错形式 |
|---|---|---|
| 编辑时/打开时 | MajdataEdit SyntaxChecker | 错误+警告列表（行列可跳转）；级别 2 时阻止播放/导出 |
| 解析时 | SimaiSharp/MaiLib/maidata-rs | 抛异常（带行列） |
| 加载时（JSON→渲染） | MajdataView JsonDataLoader | ErrText 行号+消息；越界形状 KeyNotFound 亦进通用错误 |
| 语义级（非语法） | MajdataEdit MuriCheck（muri=无理检测） | 多押>2 检测"[多押无理]…可能形成了N押"（MuriCheck.xaml.cs:139-258, zh-CN.resx MultNoteError1/2）；slide 撞尾检测"[撞尾无理]…二者间隔Nms"（slideDetect 260-470, SlideError）——基于 slide_time.json 的物理判定数据 |

---

# 三、来源清单

- MajdataView（含官方 wiki 仓库 MajdataView.wiki，页面《怎样写谱？.md》）：https://github.com/LingFeng-bbben/MajdataView （本地 /tmp/majdata-view、/tmp/majdata-wiki）
- MajdataEdit：https://github.com/LingFeng-bbben/MajdataEdit （本地 /tmp/majdata-edit；SyntaxModule/SyntaxCheck.cs、SimaiProcess.cs、MainWindowCore.cs、SubWindow/MuriCheck*.cs、Langs/*.resx）
- astrodx：https://github.com/2394425147/astrodx —— 仅 README 指向 SimaiSharp（官方 simai 序列化/反序列化器）
- SimaiSharp：https://github.com/reflektone-games/SimaiSharp （本地 /tmp/simaisharp；Tokenizer/Deserializer/Serializer/NoteReader/SlideReader/TempoReader/SubdivisionReader/Errors/*）
- MaiLib：https://github.com/Neskol/MaiLib （本地 /tmp/mailib）
- MaichartConverter：https://github.com/Neskol/MaichartConverter
- maidata-rs：https://github.com/xen0n/maidata-rs （本地 /tmp/maidata-rs）
- 官方 simai wiki（Slide 时长语法出处，SlideReader.cs:158 注释引用）：https://w.atwiki.jp/simai/pages/25.html

# 四、存疑/待核实（直接影响"零报错"策略）

1. **Hold 秒数写法分歧**：wiki 写 `1h[##2.5]`（怎样写谱？.md §2），MajdataEdit 解析器接受（SimaiProcess.cs:606-610 返回第 3 段），但 **SimaiSharp 会抛 UnexpectedCharacterException**（NoteReader.cs:138-146 只认 `[#秒]`）。生成谱面应优先用 SimaiSharp 兼容形式 `[#2.5]` 或比例形式。
2. **Touch C2 分歧**：wiki 说 C1/C2 都归 C；SimaiSharp 接受 C2；MajdataEdit SyntaxChecker 只认 C/C1（SyntaxCheck.cs:974-977）→ 在 MajdataEdit 里 C2 会报 SyntaxError。
3. **Majdata 语法宽松自述**：MajdataView README 已知问题 2 明言 Majdata 能跑的谱面未必能在 maipad/simai/Astro 运行 → 应以最严格解析器（SimaiSharp）为最终校验器。
4. **Slide EX（x）分歧**：MajdataEdit 接受 slide 带 x（IsSlide:945-946），SimaiSharp 对 slide 后的 `x` 抛 UnsupportedSyntaxException（SlideReader.cs:106-107）→ 生成器不要给 slide 加 x。
5. **`@`/`m` 装饰符**：仅 SimaiSharp 支持；MajdataEdit 会报语法错误 → 避免使用。
6. **伪 Each 时差**：wiki 称 0.001s，MajdataEdit 代码实为 1/128 拍（1.875/bpm 秒，SimaiProcess.cs:250-257）——文档与代码不一致，以代码为准。
7. **MajdataEdit 内部不一致**：其解析器接受 `2$h[4:1]`（getSingleNote 对 $ 不分类型，SimaiProcess.cs:508-515），但自身 SyntaxChecker 的 IsHold 会拒绝 `$`（SyntaxCheck.cs:874-917）→ 别用。
8. **`{#秒}` 语义**：SimaiSharp 解释为"每拍秒数"（SubdivisionReader.cs:14-30），与 MajdataEdit 的拍号整数要求（SyntaxCheck.cs:281-285）不一致；生成器应只用整数拍号。
9. **缺失 `E` 的严重度**：MajdataEdit 仅警告、SimaiSharp 不报错；但为跨工具零报错，**务必以 `E` 结尾**（SimaiSharp Serializer 也强制输出 E）。
10. MaiLib 扫描器不支持 `! ? @ \` w`（SimaiScanner.cs token 表）——若目标读者包含 MaiLib/MaichartConverter，需避开这些语法。

# 五、给项目的结论建议

零报错生成策略 = 输出格式对齐 **SimaiSharp Serializer 的规范序列化格式**（§一.7：`(bpm){拍号}` 开头、每拍 `,`、`E` 结尾、`/`/`` ` `` 分隔、slide 统一 `[秒##秒]`），语法子集取三家交集（避开上面 10 条分歧项），并用 SimaiSharp 做最终 lint（异常带行列可定位）。
