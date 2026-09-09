# 调研报告：中文舞萌制谱社区资料与规范调研（完整版）

> 来源：coding agent 调研（2026-09-10），委托子代理完成。方法：web_search 多组关键词；web_fetch 在本环境对所有域名解析失败，正文改用 bash+curl 抓取。可完整抓取：B站专栏(opus/SSR)、GitHub wiki/README、萌娘百科、majdata.net、AstroDX wiki、Diving-Fish 查分器数据库。不可抓取：bilibili 视频页正文、知乎(反爬)、贴吧(验证)、atwiki SimaiWiki(Cloudflare)、MMFC 官网(超时，PDF 直链可用)。
> 难度↔note 密度表为本调研用官方谱面数据库实测计算（Diving-Fish music_data，字段 notes=[tap,hold,slide,break]，已与 maimai 中文 Fandom 维基单曲模板核对）。

# 一、教程/语法要点

## 1. simai 语法中文教程（核心，均有原文可摘录）

- **MMFC《MAIMAI 谱面创作基础学 长篇指南》**（2024-07，37MB PDF，7 章）：面向零经验者的"review"型整合教程，以 simai 语言 + Majdata 工具链讲解；引述此前教程：半步码农教程、Majdata README、晓舟教程。
  - 出处：公告 https://www.bilibili.com/opus/1060186937336266788 ；PDF 直链 https://daf9ba22-0db5-4fe2-94c6-92c458ad8ce0.usrfiles.com/ugd/daf9ba_62c366f6038948a0afd82f304db220f9.pdf ；腾讯文档 https://docs.qq.com/pdf/DUmpTeVVGWVNOSWhE ；MMFC 官网 https://www.maimaimfc.ink/（本项目用户已提供同款 PDF）
- **MajdataView Wiki《怎样写谱？》**（繁体中文，2023-03，Xuan 编写）：键位对照（右上角 1 号顺时针到 8），Tap 键位 1–8；BPM 与 {x} 分音；全部 Note 类型与 8 种 Slide 形状限制；同头星 `*`、Each `/`、伪双 `` ` ``（差 0.001 秒）；TouchHold 在 Majdata 中只能放 C 区而 simai 无限制；Ex-note `x`。
  - 出处：https://github.com/LingFeng-bbben/MajdataView/wiki/怎样写谱？
- **simai 官方规范** = 日本 atwiki《simai [simulator of maimai]》（Celeca 2013-02-05 发布 simai；语法页 pages/25，Fes 格式 pages/1003、创作谱面数据 pages/163）。中文译本：《Maimai自制谱格式(3simaiFes Fmt)使用说明》（译自 Simai 官方 Wiki，作者 LifeErr0r，文件头翻译 @寺田松し；自注"内容已过时，推荐新人使用 Visual Maimai"）。
  - 出处：https://w.atwiki.jp/simai/pages/25.html ；https://w.atwiki.jp/simai/pages/1003.html ；中文译 https://www.bilibili.com/opus/739441653140422663
- **文件结构**：maidata.txt 为最终产物；必要文件 = track.mp3 + maidata.txt，可选 bg.jpg/bg.png（建议 1080×1080）、pv.mp4/bg.mp4；文件名固定；音频建议 44100Hz、CBR≥192kbps mp3。（MMFC §3.2.2）
- **元数据头**：`&title`、`&artist`、`&first`（0.001 秒精度）、`&wholebpm`、`&lv_1..6`、`&des_1..6`、`&inote_1..6`、`&clock_count=4`（开头提示音数，仅录制模式有效）等。（3simaiFes 使用说明；MMFC §4.2；MajdataView Wiki 快速入门）
- **时间标记**：`(BPM)` 设 BPM；`{X}` 设 X 分音符；逗号为时间单位（空逗号=休止）；`{#XXX}` 秒间隔（3simaiFes）；最小分音 384（只用 384 约数）；不要连续写两个分音/BPM 标记。（MMFC §4.3）
- **Tap**：键位数字 1–8；Break=后缀 b；Ex=后缀 x；组合顺序建议 "b x h"（如 `1bh[4:1]`）；星星头与滑条修饰独立：`7-2[8:1]b`（滑条绝赞）、`7b-2[8:1]`（头绝赞）、`7bx-2[8:1]b`（都绝赞）。（MajdataView Wiki；无理综述一）
- **Hold**：`键h[X:Y]`（Y 个 X 分音）/`h[X]`/`h[#X:Y]`/`h[#X]`（秒）；特殊写法 `[BPM#X:Y]`、`[X##Y]`。Slide 时值同理。特殊语法（3simaiFes）：Tap 后 `$` 转 Star 不旋转、`$$` 旋转；Slide 头 `@` 转 Tap、`?` 隐藏 Star 保留浮现、`!` 突然出现；`` ` `` 伪多押；Touch 后 h 显像 TouchHold。（3simaiFes 使用说明；MajdataView Wiki）
- **Slide 8+1 种形状与键位限制**（AI 生成器必须遵守）：
  - `-` 直线：终点不能为起点及左右邻位（至少隔 1 键）；
  - `< > ^` 贴边曲线：`^` 取劣弧、不能取对向；`<>` 方向按起点上下半屏定义；
  - `v` 折线经 C 区：终点不能为自身及正对面；
  - `V` 大折线：中折点必须距起点 1 键（如 1→3/7），终点限制更严（1V36 合法）；
  - `q p` 绕内圆（B 区）一周到终点，终点可自身；
  - `s z` 闪电：终点只能为对角线键；
  - `w` Wifi：终点只能为对位键，视作双手配置；
  - `pp qq` 绕大圈（B/E 区）；Fes 连锁拼接如 `1-3-5-7-1[2:1]`、`1>5-8[1:1]`；同头星用 `*`（`1-4[8:1]*-6[8:1]`），严禁用 `/`。
  - 出处：MajdataView Wiki；MMFC §4.4.4；无理综述（一）
- **Touch**：判定区大写 A/B/C/D/E+数字（C 区 C1/C2 合并视作 C）；`f` 后缀=烟花特效（仅视觉）；TouchHold 官谱仅在 C 区（Cf/Ch[8:1]）；Touch 无 Break。（MajdataView Wiki；无理综述一；MMFC §4.4.1）
- **Each**：`/` 分隔（`12` 或 `1/2`）；含绝赞必须分开写 `1b/2`；两 Tap 可省略斜杠（`73` 即 7/3）；`` ` `` 伪双押晚 0.001s。**启动拍概念**：Slide 星星头击打后停 1 拍（四分音符）才开始滑，该时刻为启动拍；启动拍不可随意修改。（MajdataView Wiki；自制必修课 BV1BFUeYQEYB）
- **乐理前置**：先定 BPM；四分音符=一拍；{4}/{8}/{16}/{32}/{12}/{24}/{48}/{64}；四四拍为主，三拍子/变拍新手慎写；BPM 测量三法；取整经验 xx.95–xx.05 直接取整、xx.33/xx.66 考虑三拍子、xx.5 考虑乘 2。（MMFC §2.1、§2.2）

## 2. 中文视频/专栏教程清单（BV 号可引用）

- MMFC《舞萌写谱教学·基本篇》实体册公告（小小红白）：https://www.bilibili.com/opus/1060186937336266788
- 小小红白制谱教学合集：https://space.bilibili.com/397702/channel/collectiondetail?sid=391415
- 舞萌厨房 Vol3《四分半教你舞萌里的基础乐理》：https://www.bilibili.com/video/BV1cN4y1m7sV/ ；Vol3.x 专栏：https://www.bilibili.com/read/cv27789734/
- 【新人必看/舞萌写谱教程】制谱思路第一期：https://www.bilibili.com/video/BV1yN3b6QEty/
- 【maimai自制必修课】启动拍/撞尾/多押：https://www.bilibili.com/video/BV1BFUeYQEYB/
- 舞萌（maimai）简明初级教程（车奶carmilk，cv8509720；独立站）：https://www.bilibili.com/opus/461149354380109054 ；https://carmilk.moe/p/maimai/
- maimai 判定全解（墨滢-moying，3 部分）：https://www.bilibili.com/opus/694985211225571337
- maimai 自制相关——无理详解（3098_SAJIA）：https://www.bilibili.com/opus/815287412458520598
- Maimai 无理综述系列（Minepig233，MaiMuriDX 开发者，6 篇，2024-08~09）：(一) https://www.bilibili.com/opus/970575365278793765 ；(二)判定全解 https://www.bilibili.com/opus/970660281077202961 ；(三)正规手法与无理 https://www.bilibili.com/opus/971624771839066129 ；(五)多押无理 https://www.bilibili.com/opus/976827828362280981 ；(六)叠键、外无与撞尾 https://www.bilibili.com/opus/978826006029664264
- 写给 maimai 新人谱师的 6 个建议（墨滢）：https://www.bilibili.com/opus/736833117612933144
- 【舞萌DX】配置术语系列（苍颜小峰ScienRing）：第一期 https://www.bilibili.com/opus/947947175412760583 ；第三期(星星篇) https://www.bilibili.com/video/av1856351990/
- 如何创作初代 maimai 风格谱面（Eaimo）：https://www.bilibili.com/video/BV19L41157eM/
- Visual Maimai 0.2.0 使用说明：https://www.bilibili.com/video/BV14J4m1T7Ts
- MaiMuriDX 无理检测工具介绍：https://www.bilibili.com/video/BV1P2vZenEcm/
- MajPlay 使用指引（LeZi-乐子）：https://www.bilibili.com/opus/1009339188838924289
- 谱面参考实例：SPICY SWINGY STYLE 紫谱 BV1mV4y1R7yD、海底谭 BV1c94y1q7xv、四月的雨 BV1DP4y1V76n、金星(月铃那知/KOM2) BV1ES421P7L7

# 二、制谱规范与经验

## 1. 难度分级与 note 密度对应

- 官方难度体系：BASIC/ADVANCED/EXPERT/MASTER/Re:MASTER（绿黄红紫白），等级 1–15，7 级以上带 "+"；DX Rating 用定数（小数点后 1 位）；每曲分标准谱与 DX 谱（DX 谱含 TOUCH/EX/Break 类要素，标准谱不含）。（萌娘百科）
- 官方分难度设计特征："BASIC 谱中前后两个 note 的键位通常相邻；BASIC 与 ADVANCED 谱中 SLIDE TRACE 速度通常较慢；BASIC 与 ADVANCED 难度的 DX 谱中 TOUCH 比其他难度更大。"（萌娘百科）
- **官方谱面 note 总数实测分布**（1379 首官方歌曲全谱，Diving-Fish 数据库；格式：等级 | 平均 | p10–p90 | (min–max)）：
  - DX 谱：1=101.8(86–116)；2=125.6(96–149)；3=156.4(123–191)；4=183.6(151–219)；5=217.2(178–256)；6=274.6(223–329)；7=316.9(262–370)；7+=342.5(286–395)；8=379.8(320–434)；8+=411.3(339–477)；9=422.1(307–515)；9+=462.6(375–546)；10=505.9(419–609)；10+=537.4(428–628)；11=556.1(454–648)；11+=585.7(476–709)；12=626.2(490–741)；12+=683.8(536–818)；13=761.6(620–914)；13+=862.5(706–1000)；14=997.6(832–1144)；14+=1095.7(954–1234)；15=1400（仅 1 首）。
  - 标准(SD)谱：12=492.2(302–671)；12+=569.5(401–718)；13=643.7(445–828)；13+=767.4(492–967)；14=899.8(685–1080)；14+=1020.7(733–1181)；15=1342。低等级 SD：3=128.7、5=187.9、7=269.7、8=335.5、9=361.5、10=403.8、11=457.5。
  - NPS 换算：≈ note总数/曲长（典型曲长约 2 分钟；13 级约 6.3 NPS、14 级约 8.3 NPS 量级）。
  - 出处：https://www.diving-fish.com/api/maimaidxprober/music_data ；https://github.com/Diving-Fish/maimaidx-prober ；交叉核对 https://maimai.fandom.com/zh/wiki/MIRROR_of_MAGIC
- 密度/速度红线（社区经验）：同键位纵连 2 帧(≈33.3ms)以内=叠键无理（判定区最多每秒 30 次）；官谱最快纵连=白系 177.6BPM 的 32 分（42.2ms，≈23.7 击/秒）；最快扫键=怒锤 200BPM 的 64 分（18.75ms=1.125 帧）、白日舞 102BPM 的 96 分圈（24.51ms）。NPS = (分音分母) × BPM / 60。（无理综述三/六）

## 2. 情感/结构设计（社区最强调的部分）

- 核心原则："在几乎所有自制谱环境中，自制谱的谱面强度都应该和歌曲的情绪完全贴合"；情绪通式"弱→较强→较弱→强→渐弱淡出"；关键概念=谱面的可预测性。（MMFC §5.1/5.2）
- 谱面结构 = 踩音（音轨+节奏）+ 配置（键型/位移/note 种类）两维度；切轨原则："必须根据歌曲的情绪变化，在谱面内进行切轨来保证谱面的强度能对上情绪变化"。（MMFC §5.3/5.4）
- 段落化踩音范例（SPICY SWINGY STYLE 紫谱 13 全曲拆解）：前奏哪个响踩哪个→主歌切小号/电吉他→build 末尾切回人声→第一副歌全踩人声→间奏踩人声采样+大幅位移→间奏后半踩小号休息段→第二副歌前再切人声→第二副歌与第一副歌踩音相同但单星星换双手星星提强度→尾声呼应前奏+24 分位移交互收尾。（MMFC §5.4）
- 配置强度梯度（同踩音下由弱到强）：tap < hold < each < slide(无启动拍 note) < slide+启动拍 tap（错位，最强）；位移越大越强（4 格最大）；BPM 越高越强（海底谭 120 vs 患部 200 差 1 级）；低 BPM 不简单（四月的雨 79BPM）。（MMFC §5.5）
- 采音要"简"："maimai 的采音反而是要'简'的，要'删到不能再删'"；把 maimai 当太鼓/4K 写、采太密是新人通病；星星速度贴合情绪（激烈段快、平和段/首尾慢）。（6 个建议；简明初级教程）
- 新手定位：写自己"能较轻松 AP"的等级起步；多看官谱积累"配置库"；多实测情绪匹配。（6 个建议）

## 3. 手序与可玩性规则（社区公认经验，核心规则集）

- **手序基础**：右手负责 1234（右半圈）、左手 5678（左半圈）的"分页处理"；键型术语：交互、切键、楼梯、纵连、扫键、转圈。（简明初级教程；配置术语系列）
- **星星换手定式**（"用与 slide 末尾所在半边相同的手去划 slide"）：(1) 风车星星（等速穿心，可提前绕手或换手，"一只手拍键、一只手划屏"）；(2) 一圈星星头连续飞出；(3) 死亡镰刀（一圈音符+slide 方向与出键方向相反，口诀"tap 分页，拍打与星星头相隔一个四分音符的 tap 时用另一只手一下子划完 slide"）；(4) 一对贴边对称双押 slide（打完立即双手换位划）。一笔画：一只手专门拍、一只手专门划，划 slide 的手开始后不停。（简明初级教程星星篇进阶）
- **"正规手法"形式化定义**（MaiMuriDX 检测标准，可直接作为 AI 谱手序合法性模型）：所有按压/划动带 1 帧(16.67ms)松手延迟；Tap/Touch 按压 16.67ms 后松开；Hold 按到尾+16.67ms；Slide Star 按 Tap 处理击打后松开；Slide Track 沿中心线匀速划（误触邻近判定区算正攻）；Slide 启动拍上的 Tap（拍划）并入划动手法；被 Slide 撞到的 Touch 免处理；TouchGroup 用最小覆盖圆：直径≤18cm/360px=单手，超过=双手（双手 TouchGroup 直径无上限）；Wifi 双手张开(掌宽 10cm/200px)向两个 D 区划。（无理综述三）
- **无理配置五大类**：多押（同刻需≥3 只手）、内屏无理（内屏正攻必蹭绿星星）、叠键（同刻同判定区≥2 Note）、外键无理（Slide 启动后一定时间出现同头 Tap/Hold 易蹭）、撞尾（Slide 划过 A 区时该区出现 Tap/Hold 被蹭）。分级：软无理（不大改手法可规避）/硬无理（需换手法）/绝对无理（无法规避=作谱失误）。另有拓扑无理（手序引导失误/手碰撞体积）与超速无理（超出人类极限）。（无理综述三）
- **叠键红线**：同键位两 Tap 间隔 <2 帧(≈33.3ms) 必叠键=绝对无理；同键位伪三押无解；Hold 中途夹 Tap 需外键+内屏同按，常规谱禁止；叠键无理永远是绝对无理。（无理综述六）
- **外键无理阈值**：Slide 启动后 150ms 内同头 Tap/Hold 会被蹭；考虑 2 帧内屏延迟与手瞬移，经验分界 200ms →"150 BPM 以上的八分同头是外无"（150BPM 下 8 分间隔 200ms）；可加 Ex 保护降级（保护被蹭的 Tap 而非启动拍 Tap）。（无理综述六；无理详解）
- **撞尾阈值**：危险区间 = 引导星星进入最后一个 A 区时刻起 -50ms ~ +200ms；Slide 最后 A 区整个停留期间不允许出现 Tap；Tap 踩在 Slide 结束时刻=绝对无理（加保护也不行）；0~+150ms 硬无理、-50~0 与 +150~+200ms 软无理，加 Ex 可降级；旧框 210BPM 以下 8 分撞尾可容忍、210 以上尽量避免，DX 200BPM 以上建议不写或错开 2 格。（无理综述六；无理详解）
- **多押规则表**（按无理硬度递增）：不算无理=拍划/拍按划、完全重叠 Slide、慢 Slide 期间单押散点、超大面积 TouchGroup、短距离跨接 Touch、两慢 Slide 合并后接单押；软多押=快速 Slide 撞 Touch、稍远跨接 Touch(C/E3)、过大 TouchGroup 组双押、抢跑 Slide、TouchHold 中跨接 Touch、Wifi 夹 1 条 Slide、双押启动两条完全重合 Slide、慢 Slide 期间伪双；硬多押=单手跨≥2 区 Touch、邻位双 Wifi、Wifi 夹 2 条 Slide、Slide 首/尾参与的三押、伪双逃掉后的相邻三/四押；绝对多押=标准双押中塞不可被撞 Touch、外键多押、双押启动不重合同头 Slide、双押启动 Wifi、Slide 期间/结尾出现正经双押、互不相邻外键多押。（无理综述五）
- **Hold 手序经验**："Hold 结尾都至少要留空一个八分音的间隔，位移比较大的情况下最好留空一个四分音"（接单押/双押均适用）；Hold 尾判+双押=标准多押；极短 Hold(<18 帧)可当 Tap 读，避免争议可用 0 长度 Hold 代替。（无理综述五；判定全解）
- **Touch 摆放经验**：相邻区组成 TouchGroup 才可单手；"不建议写 C/E7/E3 这样的间隔 Touch"（E8/E1/E2 尚可）；Touch 扫入 TouchHold 的配置中 A、D 区 Touch "被认为是不可以写的"（Ch[4:1]/A7/D7/A6/A2/D3/A3 禁，B2/B3 可）；TouchHold 中途跨接 Touch 是软无理尽量不写；Slide 撞 Touch 必须用快速 Slide（≥约 120BPM 的 8:1）。（无理综述五）
- **内屏无理/跳区**：星星有"松手判定"；除 2 格短直线、2 格短弧、短直线组成的大 V 外"跳区判定至多跳两格"；星星剩余判定区 ≤3 时触摸结尾 A 区即跳区（内无）；星星头击打前 100ms Slide 已开始接受判定（≈150BPM 的 16 分）；一切内无以实际上机为准。（无理详解；无理综述六）
- **Slide 时值经验**："16:3 的星星阵，星星速度一定要写 16:1"；同速同轨迹完全重合的星星不算多押；Wifi 视作双手配置，双押启动 Wifi 禁止。（无理详解；无理综述五）
- **判定机制硬数据**：60fps/16.67ms 帧；Tap 判定总宽 18 帧(±9 帧≈±150ms)，Ex-Tap 全变 Critical Perfect；Hold 开头 6 帧、结尾 12 帧不检查按压（短 Hold<18 帧=Tap 效果）；TouchHold 开头忽略 15 帧（<27 帧=假 TouchHold）；Touch 无 Fast 只有 Late；DX 框体内建 +3.0 帧 B 判补偿；Slide 尾判 CP 区间宽度随引导星星在最后判定区停留时长变化。（无理综述二；判定全解）

## 4. 社区工具链（编辑/校验/试玩）

- simai（2simai/3simai，Flash SWF，@formiku39854 开发，已停更）——历史标准；
- Majdata（MajdataView+MajdataEdit，bbben/墨滢，GPL，Unity）：写谱+节拍轴可视化+频谱+Ctrl 点击跳转+镜像/旋转+录制导出(ffmpeg)+撞尾检测精度 DefaultSlideAccuracy；仓库 https://github.com/LingFeng-bbben/MajdataView 、https://github.com/LingFeng-bbben/MajdataEdit ；新分支 https://github.com/re-poem/MajdataViewX ；
- Visual Maimai（乙酸开发，2024-06）：全可视化免语法写谱，可导出 simai；
- MajdataPlay（TeamMajdata，GPL-3.0，键盘 QWEDCXZA 或手台串口，直读 simai 谱面，可连 majdata.net）：https://github.com/TeamMajdata/MajdataPlay ；
- AstroDX（原 MaipadDX，移动端模拟器，simai 解析用 SimaiSharp，wiki https://wiki.astrodx.com/cn ，仓库 https://github.com/2394425147/astrodx ；判定与街机有差异尤其 slide，仅作试玩参考）；
- MaiMuriDX / maimaiMuriDetector（无理自动检测）：https://github.com/Moying-moe/maimaiMuriDetector ；
- MaichartConverter：https://github.com/Neskol/MaichartConverter ；
- maipad（最老安卓模拟器）；majdata.net（majnet 在线发布平台）https://majdata.net/ ；
- MMFC（maimai 自制谱月赛）https://www.maimaimfc.ink/ ；
- SimaiSharp（simai 序列化/反序列化参考实现）：https://github.com/reflektone-games/SimaiSharp ；
- 辅助：BPM Analyzer、Adobe Audition（频谱"火焰条"对齐节拍）、格式工厂（44100Hz/CBR 重编码）。

# 三、存疑/待核实

1. https://www.cnblogs.com/newzeon/p/18474261 "Simai" 介绍文：语法示例错误（"#BPM 200"、"4:2"、"16:4|32" 均非标准 simai 记法），疑似 AI 生成，**不可作为规范依据**。
2. blog.gitcode.com "如何高效使用SimAI"：实为 NVIDIA SimAI，与制谱无关（搜索误命中）。
3. 知乎三篇相关回答因反爬未能抓取正文，仅凭搜索摘要收录链接。
4. 贴吧 waptieba.baidu.com/p/8629972658 有安全验证，正文未读。
5. atwiki 官方 SimaiWiki 被 Cloudflare 拦截；语法规则以中文译本+MajdataView Wiki+MMFC 手册三方交叉印证；建议以 SimaiSharp 代码作最终解析权威。
6. bilibili 视频页正文全部无法抓取（仅标题/简介）。
7. note 密度：官方数据库无曲长字段，"每秒 note 数"精确区间为按典型曲长(约2分钟)换算的估算值（note 总数分布为实测）。
8. 无理综述系列第四部分（内屏无理）未找到独立链接（共 6 篇，其余 5 篇已收录）。
9. 半步码农、晓舟教程仅有书名级引用，未找到公开链接。
10. MMFC 官网抓取超时（PDF 直链可用）；majdata.net 邀请码可能已失效。
11. 各类无理阈值（100/150/200ms、1 帧松手延迟、360px TouchGroup 直径）均为社区实测经验或 MaiMuriDX 作者拟定的检测标准，**非 SEGA 官方文档**。
12. 舞萌网课"25 分钟无理"视频被无理综述引为参考文献但未获取 BV 号。
