# 调研报告：simai 谱面记法官方语法规范调研（完整版）

> 来源：coding agent 调研（2026-09-10），委托子代理完成。方法：本环境 web_fetch 对 w.atwiki.jp / github.com / bilibili 全部被拦截，改用 shell curl；simai wiki 因 Cloudflare 防护无法直连，全部 wiki 内容经 **Wayback Machine 快照**获取（快照时间戳见来源清单）。所有结论来自可核查来源。

## 0. 来源概况

**权威规范（第一优先级）：simai 官方 wiki（作者 Celeca，记法定义者）**
- 日文规范《simai語の譜面書式》：https://w.atwiki.jp/simai/pages/1002.html
- 英文规范《Notations of simai》：https://w.atwiki.jp/simai/pages/1003.html —— **最完整一页，建议作为生成器主规范**。页首说明：simai 语由 Celeca 于 2013 年定义；simai 软件 2023-02-05 停止公开后保留此规范；"这里只列出被正式定义为 simai 语的记法，各模拟器可能有自己的扩展"。
- 旧版全量手册《maidata.txtの譜面書式》：https://w.atwiki.jp/simai/pages/25.html（2019-2023 完整旧写法，含注释语法、Active Message 等细节）
- 变量页《maidata.txt内で使用する変数名》：https://w.atwiki.jp/simai/pages/510.html
- 分册手册：SLIDE pages/23、TOUCH pages/509、BPM pages/20、TAP/BREAK pages/21、HOLD pages/22、EACH pages/24、终了地点 pages/26

**ASTRODX**：主仓库无 docs 目录；README 指认 SimaiSharp 为其 simai 序列化/反序列化器；旧 wiki《Finding-levels》写明"3simai 是目前 AstroDX 唯一支持的谱面语言"并推荐官方文档 pages/25.html；新文档站 wiki.astrodx.com 无 chart format 页。

**实现级参照**：SimaiSharp 源码、MajdataView wiki《怎样写谱？》（其 README 自述语法宽松）、MaiLib README（指认 pages/1003 为规范）、maidata-rs README、bilibili 3simaiFes 中文译与无理综述（一）。

# 一、记法总览与谱面结构

- 谱面 = 一连串 `,`（逗号）+ 音符记号，最后以 `E` 结束。例：`1,2,3,4,E`
- 每个逗号占一个"时值槽"；**槽长(秒) = 240 / BPM / 分音数**。例 `(120){4}` 时每槽 0.5 秒。
- 按键编号：**1 号在屏幕右上（1 点钟方向），顺时针编号 1→8**：
  ```
  　⑧①　
  ⑦　　②
  ⑥　　③
  　⑤④　
  ```
- 谱面正文中的换行、空格、全角空格(0x3000)、制表符均被忽略（仅排版用）；SimaiSharp 分词器额外忽略 0x2002/0x2008/0x2028/0x2029。
- **没有小节线字符**（竖线 `|` 只出现在注释 `||` 中，不是小节线）；**没有拍号(time signature)指令**，隐含 4/4（一小节 = 4 个四分音符）；**没有 `(time)` 时间标记语法**——直接用秒数的写法是 `{#秒数}`。
- `E` 必须大写；`e` 无效。不写 `E` 则谱面播到音频结束为止。结束点不能超过音频长度（超出部分不播放）；HOLD/SLIDE 进行中途结束会丢分。

# 二、文件头 meta 指令

**官方（wiki 变量页 510）定义的变量：**

| 写法 | 含义 | 备注 |
|---|---|---|
| `&title=` | 曲名 | |
| `&artist=` | 艺术家 | |
| `&smsg=` | SINGLE MESSAGE 单条留言 | 显示于选曲/播放画面 |
| `&des=` | 谱师名 | 自由文本 |
| `&first=` | 谱面开始偏移（秒，可小数） | 音乐播放 first 秒后第一个逗号开始计时 |
| `&lv_1=` … `&lv_7=` | 各难度等级 | 1=EASY,2=BASIC,3=ADVANCED,4=EXPERT,5=MASTER,6=Re:MASTER,7=ORIGINAL；可写 `13+`（+ 半角），也可 `※X` 自定义单字等级 |
| `&inote_1=` … `&inote_7=` | 各难度谱面正文 | `=` 后到下一个 `&` 行之间的全部内容即谱面 |
| `&amsg_first=` / `&amsg_time=` / `&amsg_content=` | Active Message（歌词字幕） | amsg_time 用 `1,`(显示下一句) `0,`(清屏) 与 `(bpm){x}` 书写；content 每行以 `┃` 开头 |

**难度后缀变体**：`&smsg_1=`…`&smsg_7=`、`&des_1=`…`&des_7=`、`&first_1=`…`&first_7=`（选某难度时若存在 `_N` 版本则优先于无后缀版本）。

- 官方**未规定 &变量顺序**；每条 `&key=` 另起一行、值可跨多行（SimaiSharp SimaiFile 解析逻辑：以 `&` 开头的行开始新的键值对）。社区模板顺序只是惯例。
- 生成器最低要求（按 astrodx 导入逻辑）：至少一个 `&inote_N=` 谱面；`&title` 非强制。
- **正文内部**顺序要求是强制的：**BPM 必须写在分音之前**（`(120){4}` 可以，`{4}(120)` 会出错）。
- **转义规则（meta 文本中）**：半角 `&` `+` `%` `\` 必须写成 `\＆` `\＋` `\％` `\￥`（`\` 半角 + 全角字符）；全角 ＆＋％ 可直接用。`&title` 中 `┃` 之后的文字以小字（副标题）显示；`&smsg` 中 `┓` 换行（最多 3 行）。
- **社区扩展（非官方 wiki 定义，但普遍使用）**：`&wholebpm=`（整曲 BPM 浮点）；`&freemsg=`；`&tap_ofs=` `&hold_ofs=` `&slide_ofs=` `&break_ofs=`（各判型 offset，Majdata 社区模板）；MaiLib 头部另含 `&artistid=` `&shortid=` `&genre=` `&cabinet=DX|SD` `&version=` 等（对播放器透明，建议保留兼容但不依赖）。

# 三、正文基础：BPM、分音、秒数、注释

- `(120)` 设置 BPM，**可小数**（如 `(174.5)`）；`(#...)` 不是合法写法。
- `{4}` 设置分音数：`{4}`=四分音符每槽、`{8}`=八分、`{16}`=十六分、`{1}`=全音符。可小数但强烈不推荐。
- `{#0.35}` 直接指定每槽 0.35 秒（BPM 不明/人声对齐时用）。**注意：2023-06 快照英文页误写为 `(#0.35)`，官方已于 2023-07 更正为 `{#0.35}`**；旧手册也用 `{#0.25}`；SimaiSharp 只接受 `{#...}`。
- `||注释文字`+换行：行注释（3simai 起为 `||`；2simai 前用 `>>` 已废弃）。**必须以换行结尾**，否则不生效；注释内容中禁用半角 `&` `+` `%`。SimaiSharp：单个 `|` 直接报错，`||` 吞到行尾。
- 空逗号 `,,,` = 休止（每逗号一槽）。

# 四、音符类型与修饰符

## 4.1 TAP
- `1,` = 1 号键 TAP；`1b,` = BREAK TAP（b 在数字后；`b1` 非法）。

## 4.2 HOLD
- `5h[2:1],` = 5 号键 HOLD，长 = 二分音符×1。通式 `Bh[x:y]` = "x 分音符 y 个"，秒数 = (240/BPM/x)×y；可约分应约分。
- `4h[#5.678],` = 直接按秒（5.678 秒）。
- `4h[150#2:1],` = 按指定 BPM 计算。
- `5hb[2:1],` 或 `5bh[2:1],` = BREAK HOLD（b 与 h 顺序任意，官方明确等价）。
- `3hx[2:1],` = EX HOLD；`7bxh[4:1],` = EX+BREAK HOLD（b/x/h 顺序任意）。
- `3h,` = "伪 TAP"（六角形 TAP 效果）：省略 `[x:y]`，内部等价 `[1280:1]`（官方解释来自 SEGA 公式设定资料）。
- 分歧：MajdataView wiki 出现 `1h[##2.5]`，与官方 `1h[#2.5]` 不符，判为其 wiki 笔误（SimaiSharp 会拒绝 `##` 开头的 hold 时长）；bilibili 出现 `键号h[X]`（无冒号单值）与 `h[#X:Y]`（=X*Y 秒）两种写法，官方无此二者 → 生成器不用。

## 4.3 TOUCH / TOUCH HOLD / 烟花
- 传感器编号（DX 共 34 区）：A 组（紧邻按键，8 区）、B 组（A 与中央之间，8 区）、**C 组（中央，2 区：C1 在右、C2 在左）**、D 组（填充 A 组之间，8 区）、E 组（D 稍内侧靠 B，8 区）；组内顺时针编号。写法 = 字母+数字：`A1`…`E8`、`C`。
- `B1,` `D4,` `C,` = TOUCH。中央写作 `C`；**`C1`/`C2` 与 `C` 内部等价**（不报错，均落正中央）。
- `Ch[4:3],` = TOUCH HOLD（计分同 HOLD）。simai 语法允许任意区 TouchHold；官方谱面（至 UNiVERSE PLUS）只有 C 区出现过；**MajdataView 显示层只支持 C 区 TouchHold**。
- `B7f,` `Chf[1:2],` = 烟花特效（TOUCH 命中瞬间/TOUCH HOLD 结束瞬间彩虹放射）；`hf` 与 `fh` 等价。
- `Ch,` = 伪 TOUCH（省略时长的 TouchHold，即点即判）。
- TOUCH 无 BREAK（写了无效）；无 EX。

## 4.4 EACH（多押）
- `1/8h[2:1],` = 1 号 TAP 与 8 号 HOLD 同刻多押；`/` 分隔任意数量（3simai 起不限 2 个）。
- 顺序无关（`8h[2:1]/1` 等价）；但显示层级上先写的 SLIDE 显示在前面。
- `12,` 甚至 `123,` = 纯 TAP（非 BREAK）多押可省略 `/`；**只要成员里有非 TAP 或 BREAK，`/` 就不能省略**。
- EACH 成员变黄（BREAK 除外）；同刻启动的多条 SLIDE 视为 EACH（官方谱面中扇形 SLIDE 自 simai ver2.30 起参与变黄）。
- `8/1-4[8:1]*-6[8:1],` = TAP 与同始点双 SLIDE 的多押（星只落在 1 号）。

## 4.5 EX note
- `1x,` `3hx[α:β],` `5bx,` `7bxh[α:β],` `2x-4[4:1],` = EX 判定（GOOD 以上都按 CRITICAL PERFECT 计）；适用于 TAP/HOLD/BREAK（含 SLIDE 星头），TOUCH 不适用；x 与 b/h 组合顺序任意。视觉发白高亮。

## 4.6 伪 EACH
- `1\`2,`（反引号）= 2 号比 1 号晚 **1ms**（不完全同时 → 不变黄、不计 EACH）；可连续 `1\`2\`3/4,`。反引号在 US/UK 键盘位于 1 左侧。

## 4.7 星形/旋转/去星修饰符（官方）
- `1$,` = 普通 TAP 强制变为星形 TAP（SLIDE 星头样式）；可与 b/x 混用顺序任意。
- `3$$,` = 星形且**旋转**（转速固定常量）。
- `1@-5[8:1],` = SLIDE 星头强制变回普通圆形 TAP（滑轨照常）；可与 b/x 混用。
- `1?-5[2:1],` = **无星头 SLIDE（渐显式）**：星形 TAP 不飞来，滑轨淡入，启动拍内引导星淡入（适合文字/图案谱）。
- `1!-5[2:1],` = **无星头 SLIDE（突现式）**：引导星在开始滑动瞬间突然出现（适合一笔画连写谱）。

# 五、SLIDE 详细规范（核心）

## 5.1 基本格式与启动拍
- `始点 形状 终点[时长],` 例 `1-4[8:3],`（1→4 直线，移动耗时 8 分音符×3）。
- 移动时间通式同 HOLD：`[x:y]` = x 分音符 y 个（按当前 BPM）。
- **启动拍（タメ時間）**：星形 TAP 到达判定线后，引导星等待 **1 拍（四分音符，60/BPM 秒）** 再开始移动——**与当前 `{x}` 分音无关**（官方文字与旧手册一致；无理综述算式 60/150=0.4s 佐证）。

## 5.2 高级时长写法（`[...]` 内）
| 写法 | 等待时间 | 移动时间 | 例 |
|---|---|---|---|
| `[x:y]` | 当前 BPM 一拍 | 当前 BPM 的 x 分音×y | `1-4[8:3]` |
| `[BPM#x:y]` | BPM 的一拍 | BPM 的 x 分音×y | `1-4[160#8:3]` |
| `[BPM#秒]` | BPM 的一拍 | 直接秒数 | `1-4[160#2]` |
| `[秒##秒]` | 直接秒数 | 直接秒数 | `1-4[3##1.5]` |
| `[秒##x:y]` | 直接秒数 | 当前 BPM 的 x 分音×y | `1-4[3##8:3]` |
| `[秒##BPM#x:y]` | 直接秒数 | 指定 BPM 的 x 分音×y | `1-4[3##160#8:3]` |

SimaiSharp 另接受 `[#1.5]`（移动 1.5 秒、等待默认一拍）。

## 5.3 形状符号逐一（官方文字 + 多源交叉）
- `-`（直线）：起点到终点直线。终点须与起点至少隔一个键（1 号不能终点 1/2/8）。
- `^`（圆弧，最短路径自动判向）：**终点不能是起点正对面**（正对时两侧等距无法判定）；官方评论区确认 `1^2`（相邻）合法。距离 < 半圈时可用 `^`。
- `>` / `<`（圆弧，显式方向）：`>` = 沿判定圈向右，`<` = 向左。换算：**上半屏键（1,2,7,8）`>`=顺时针；下半屏键（3,4,5,6）`>`=逆时针**（官方示例 `6>1`=逆时针、`3>6`=逆时针佐证；SimaiSharp DetermineRingType 与 MaiLib SCL/SCR 映射一致）。建议至少隔一个键；无官方硬限制。
- `v`（小 V 形）：经屏幕中央（C 区）转折成 V 字两条直线。**终点不能是自身与正对面**（`1v1` 非法；`1v2/1v3/1v4` 合法）。
- `p` / `q`（p 形/q 形）：绕中央曲线。**p=逆时针、q=顺时针**（官方、MajdataView wiki、bilibili、SimaiSharp 一致）。终点可为任意键**包括自身**（`5qq5/5pp5` 官方谱面出现过心形）。
- `s` / `z`（闪电形）：3 段短直线折线；**终点必须是对角（正对面）键**。折点：s 先经 B(X+6) 再 B(X+2) 到 A(Y)；z 先经 B(X+2) 再 B(X+6)。MaiLib 源码佐证折点 = key±2。
- `pp` / `qq`（大 p/q 形）：绕"中央—外圈切线圆"的大弧。**pp=逆时针、qq=顺时针**。终点任意含自身。
- `V`（大 V 形/屈折形）：**三键记法 `1V36`** = 起点 1、转折点 3、终点 6；起点→转折点是短直线，转折点→终点任意长直线。**转折点限为起点的"2 个键位之外"**（旧手册原文：始点の2つ隣；1 号只能在 3 或 7 转折；MajdataView wiki 补充：1 号在 3 转折后终点不可为 1/2/3/4）。
- `w`（扇形/Wifi 形）：**终点必须是对角键**；展开成覆盖 3 个端点的扇形，产生 3 颗引导星（判定等效三条独立直星）；SimaiSharp 解析时读取 3 个终点位置（`1w357`），即对角键与其环上两邻键。官方说明 w 之后不能再接连锁段。
- 全 SLIDE 移动速度恒定（任何形状中途不变速）。

## 5.4 同始点 SLIDE（multi-slide）
- `1-4[4:3]*-6[8:5],` = 一颗星头伸出多条轨迹；`*` 连接，后续轨迹省略始点数字；3simai 起可无限多条；各条时长可不同，但星同时启动 → 这些 SLIDE 相互构成 EACH（变黄）。

## 5.5 连锁 SLIDE（chained slide，FESTiVAL 起）
- `1-4q7-2[1:2],` = 多段首尾相接成一条；总时长写在最后；全程匀速。
- `1-4[2:1]q7[2:1]-2[1:1],` = 分段各自给时长；**官方：必须给全每一段的时长，缺任一则报错**（SimaiSharp 实现为累加，行为略宽容——见分歧）。
- BREAK：连锁 SLIDE 只能在**最后一个 `]` 后**加 `b`，整条为 BREAK，不能局部 BREAK。

## 5.6 SLIDE 的 BREAK/EX 位置（重要）
- `1-4[8:3]b,` = **滑轨**为 BREAK（官方写法：b 在 `]` 后）。MajdataView wiki 示例写 `1-4b[4:1]`（b 在 `[` 前）——SimaiSharp 两种都接受，两源写法并存。
- `1b-4[4:1],`（官方旧手册同义例 `1bs5`）= **星头**为 BREAK（计数：BREAK 1 + SLIDE 1）。
- `2x-4[4:1],` = 星头为 EX。
- 星头与滑轨可分别修饰：`7bx-2[8:1]b`（星头 BREAK+EX，滑轨 BREAK）。

## 5.7 官方"始点-终点关系表"
官方规范页的完整允许表是**图片**（slides_eng.png，860×826），本调研无法读图 OCR，未逐格转录：https://img.atwiki.jp/simai/attach/1003/54/slides_eng.png （Wayback: https://web.archive.org/web/20230610055949im_/https://img.atwiki.jp/simai/attach/1003/54/slides_eng.png ）。5.3 的文字规则来自官方正文 + 旧 SLIDE 手册 + 转述来源汇总。**生成器若要严格校验合法性，建议由人工/视觉模型对照此图补齐 64 格矩阵**。

# 六、其他语法点

- **键音（key sound）记法**：所有来源均不存在（simai 播放器用整曲音频 + 谱面）。
- **`fi` 属性**：未在任何权威来源找到；最接近的是烟花 `f`（firework）。
- **SimaiSharp 特有扩展（非官方 simai 语）**：装饰符 `m`（Mine，astrodx 内部用）、位置 `0`（内部强制 EACH 标记）、装饰符顺序无关。
- **编码**：SimaiSharp 自动探测 UTF-7/8/16/32（含 BOM）并回退系统代码页（兼容 Shift-JIS 老谱）；**生成器建议输出 UTF-8（带 BOM 更稳妥）**。
- **astrodx 导入要求**：谱面包 = 文件夹含 `maidata.txt` + 音乐文件（mp3/ogg，建议 ogg）+ `bg.jpg/png`；按文件名识别。

# 七、分歧清单（不同来源同一语法的差异）

1. `{#0.35}` vs `(#0.35)`：2023-06 英文页快照笔误，官方已更正为 `{#...}`；SimaiSharp 只接受 `{#...}` → 采用 `{#...}`。
2. HOLD 秒数：官方 `4h[#5.678]`；MajdataView wiki `1h[##2.5]` 判为笔误；bilibili `h[X]`/`h[#X:Y]` 官方无 → 以官方三种形式为准。
3. 滑轨 BREAK 的 b 位置：官方 `]` 后；MajdataView wiki `[` 前；SimaiSharp 都接受。
4. SLIDE 启动拍时长：官方 = 当前 BPM 的一拍（与 `{x}` 无关，60/BPM 秒）；SimaiSharp 源码用 SecondsPerBeat（随 `{x}` 变化，`{8}` 时会变 60/BPM×4/8）——在 `{x}≠4` 时与官方不一致，疑为实现差异/潜在 bug，生成器按官方语义。
5. 连锁 SLIDE 分段时长：官方要求每段必给、否则报错；SimaiSharp 累加实现，漏给不报错（更宽容）。
6. 星头 `$`：官方只定义 `$`/`$$` 对 TAP 与 `@` 还原；bilibili"Slide 头后跟 $"说法官方未定义 → 视为社区扩展。
7. TouchHold 位置：simai 语法任意区；MajdataView 只支持 C 区；官方谱面至今只有 C 区。
8. Majdata 宽松性：MajdataView README 自述"能在 Majdata 运行的谱面可能无法在 simai/Astro 运行"→ 生成器以官方 simai 语 + SimaiSharp 行为为准。
9. SimaiSharp 序列化疑似 bug：`SlideSegment.WriteTo` 把 CurveCcw（p）写成 `pp`（其余形状映射正确）——round-trip 需注意；生成器直接输出以官方符号为准。

# 八、给生成器的落地建议

- 文件：UTF-8 编码 `maidata.txt`；头部按需写 `&title/&artist/&des/&wholebpm/&first/&lv_N/&inote_N`，每条 `&key=` 独占一行起始。
- 正文：开头必须 `(BPM){分音}`（BPM 在前）；每音符以 `,` 结束；结尾大写 `E`。
- 时值换算：槽 = 240/BPM/分音；HOLD `[x:y]` = (240/BPM/x)×y；SLIDE 启动拍 = 60/BPM。
- 先按官方 simai 语子集输出（- ^ < > v p q s z pp qq V w、b/x/$/@/?/!、h[]、f、/、*、||注释），再用 SimaiSharp 或 astrodx 实测兼容；避免 Majdata 专有宽松写法。

# 九、来源清单

官方 wiki（内容经 Wayback 快照读取）：
1. https://w.atwiki.jp/simai/pages/1003.html （Notations of simai，EN；快照 20230610055949、20260516091208）
2. https://w.atwiki.jp/simai/pages/1002.html （simai語の譜面書式，JP；快照 20230610055949、20250809132819）
3. https://w.atwiki.jp/simai/pages/25.html （旧版全量；快照 20251005015113）
4. https://w.atwiki.jp/simai/pages/510.html （変数名；快照 20191212132552）
5. https://w.atwiki.jp/simai/pages/23.html （SLIDE；快照 20191211091337）
6. https://w.atwiki.jp/simai/pages/509.html （TOUCH；快照 20191211153730）
7. https://w.atwiki.jp/simai/pages/27.html （譜面作成マニュアル；快照 20191212182808）
8. 关系表图：https://img.atwiki.jp/simai/attach/1003/54/slides_eng.png 等三处附件

ASTRODX 系：https://github.com/2394425147/astrodx 、https://github.com/2394425147/astrodx/wiki/Finding-levels 、https://github.com/reflektone-games/SimaiSharp 、https://github.com/reflektone-games/AstroDX_Wiki

其他实现与社区文档：https://github.com/LingFeng-bbben/MajdataView 及 wiki《怎样写谱？》、https://github.com/Neskol/MaiLib 、https://github.com/xen0n/maidata-rs 、https://www.bilibili.com/opus/739441653140422663 （3simaiFes 中文译）、https://www.bilibili.com/opus/970575365278793765 （无理综述一）

# 十、存疑 / 待核实

1. SLIDE 始点-终点关系表为图片（slides_eng.png），64 格全矩阵需人工/视觉模型对照补齐。
2. 扇形 w 的 3 个端点是否恒为"对角键+环上两邻键"需对照官方图或实测。
3. SimaiSharp 两处疑 bug：(a) 启动拍随 `{x}` 变化（官方固定一拍）；(b) p 形序列化输出为 `pp`。若项目直接依赖 SimaiSharp，建议写回归测试。
4. `fi` 属性：未找到出处；最接近的是 `f`（烟花）。
5. 键音记法、`(time)` 时间标记：simai 语中均不存在（秒数用 `#` 前缀表达）。
6. `h[X]`（无冒号）与 `h[#X:Y]`：只见于 bilibili 转述，官方与参考实现均不支持。
7. `&wholebpm` 等社区字段：非官方 wiki 定义，但被普遍使用；播放器忽略未知 `&key`，输出无害。
8. `^` 对相邻键（如 1^2）：官方评论区确认可写且自动取最短弧；关系表图是否标为允许需对照。
9. 中文/日文来源文本均经机器转录（wiki HTML 转文本），个别全角字符以原站为准。
