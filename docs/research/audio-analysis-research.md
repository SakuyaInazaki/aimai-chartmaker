# 调研报告：音频音乐结构提取工具 SOTA 与 prior art（完整版）

> 来源：coding agent 调研（2026-09-10），委托子代理完成。方法：web_fetch 被沙箱拦截，改用 web_search 摘要 + bash/curl 直连一手来源（GitHub API、raw LICENSE、HuggingFace API、gitcode API、PyPI JSON、arXiv）交叉验证。标记：✅=一手来源核实；⚠️=部分核实；❓=未能确证。许可证结论均经 LICENSE 原文或官方 API 字段核实。

# 一、节拍层（BPM / beat / downbeat / 变速）

## 结论推荐
- **主线：beat_this（CPJKU/beat_this，MIT ✅，pip install beat-this）**——一个模型同时输出 beats + downbeats（downbeat=小节起点，制谱对齐小节的关键），帧级 logits 可当置信度，训练数据显式包含变拍号/高 BPM 曲目、无 DBN 后处理、对变速更宽容；不输出 BPM（需自研"拍间隔聚类→BPM 分段"）。依赖极简（仅 PyTorch）。
- **打底：librosa（ISC ✅）**——先跑通管线用；beat_track 支持时变 tempo、CPU 秒级；无 downbeat、无置信度。librosa 1.0 已移除 librosa.beat.tempo() 公共 API。
- **一步到位备选：all-in-one（MIT ✅）**——beat/downbeat F1 高（Harmonix beat .958/downbeat .915）且顺带输出结构分段（代价：前置 Demucs + 较重依赖）。
- **madmom 精度好但两重坑**：模型权重 **CC BY-NC-SA 4.0 非商用** + PyPI 停在 0.16.1(2018, Cython 编译、numpy 2.x 不兼容)。非商用个人项目仍是高精度选项。

## 候选对比
| 工具 | 许可 | beat/downbeat F1 | 置信度 | 变速 | CPU | 预训练 |
|---|---|---|---|---|---|---|
| beat_this | MIT ✅ | Ballroom 97.5/95.3，Hainsworth 91.9/80.0（±70ms） | 帧级 logits | 无 DBN、较宽容 | 近实时 | 权重随包 |
| librosa.beat | ISC ✅ | 无官方 F1 | 无 | 时变 tempo | 秒级 | 无 |
| madmom DBNDownBeat | 代码 BSD-3 / 模型 CC BY-NC-SA ⚠️ | Ballroom .938/.863 等 | 100fps activation | DBN 平滑；默认 max_bpm=215 需调大 | 近实时 | 内置(非商用) |
| all-in-one | MIT ✅ | Harmonix beat .958/downbeat .915 | sigmoid 激活 | 输出 tempo 序列 | GPU 27×实时；CPU 慢 | HF Awiny |
| essentia | AGPL-3.0 ⚠️ | 无官方 F1 | confidence 字段 | maxTempo 默认 208 | 快 | 内置 |
| tempo-cnn | AGPL-3.0 ⚠️ | 只出 BPM；Acc1 92/Acc2 98.4 | 概率分布 | 八度不变增强 | 快 | 随包 |
| BeatNet | CC BY 4.0 ✅ | GTZAN 口径不同 | 粒子滤波 | 在线 | GPU 友好 | 随包 |

## 落地注意事项
1. BPM 上下限：目标 160-220 必须显式设置（madmom max_bpm=215、essentia maxTempo=208 默认值会在 220 档翻车）。
2. MP3 先转 WAV：all-in-one 官方警告 MP3 解码偏移 20-40ms（容差 70ms），用 ffmpeg 先转 PCM。
3. 置信度后处理：madmom/beat_this/all-in-one 给帧级激活，可做低置信度拍点过滤 + 局部峰值细化。
4. 变速检测：madmom 输出 tempo 轨迹；beat_this 需自行从拍间隔分段估计 BPM 序列；simai 的 bpm token 支持逐小节 BPM。

# 二、声源分离（踩音目标提取）+ 人声旋律

## 结论推荐
- **分离主线：Demucs v4 htdemucs_ft（MIT ✅ 代码+权重同仓库）**——drums/bass/vocals/other 四路；MUSDB18-HQ drums 10.08 SDR；鼓 stem 泄漏少、直接喂 onset 检测。CPU 弱时迭代期用 htdemucs（drums 10.05），量产期 ft/GPU。仓库 archived（fork 至 adefossez/demucs），pip/torch.hub 可用。
- **CPU 轻量补充**：UVR Kim_Vocal_2.onnx（64MB）；鼓细粒度 MDX23C-DrumSep（kick/snare/toms/hats/cymbals）。
- **人声旋律主线：basic-pitch（Apache-2.0 ✅）**——直接输出 note events（onset/offset/pitch/confidence）+ MIDI；onnx 230KB 随包；CPU ≈19×RTF。**后备 RMVPE**（Apache-2.0 ✅，rmvpe.pt 173MB HF lj1995）。
- **不推荐**：Spleeter（质量垫底、TF 老旧）、BS-RoFormer（权重许可不明 610MB）、AudioSep（1.26GB 过度设计）。

## 候选对比
| 方案 | 代码/权重许可 | Stems | 质量(SDR) | 速度 | 体积 |
|---|---|---|---|---|---|
| Demucs v4 htdemucs_ft | MIT/MIT ✅ | 4 | avg 9.00 / drums 10.08 | CPU≈1.5×时长×4(ft)；GPU 快 | 4×84MB |
| Demucs v4 htdemucs | MIT/MIT ✅ | 4 | avg 8.80 / drums 10.05 | 比 ft 快 4 倍 | 84MB |
| Spleeter | MIT / 5stems 存疑 ⚠️ | 2/4/5 | vocals 6.86 | GPU≈100×RTF | 73-183MB |
| UVR MDX23C / Kim_Vocal_2 | MIT / 无 LICENSE ⚠️ | 2 | SDX23 vocal 11.11 | ONNX CPU 可 | 427/64MB |
| BS-RoFormer | MIT / 权重许可不明 ❓ | 2 | 社区榜 vocal 12.9 | CPU 很慢 | 610MB |
| AudioSep | MIT / 1.26GB 无许可标注 ❓ | 文本 query 单 stem | MUSIC SDRi 10.5 | 重 | 1.26GB |
| Open-Unmix umxl | MIT / CC BY-NC-SA ⚠️ | 4 | overall 5.3 | CPU 可 | Zenodo |

## 人声旋律
| 工具 | 许可 | 权重 | 输出 | CPU | 适用性 |
|---|---|---|---|---|---|
| basic-pitch | Apache-2.0 ✅ | 内置 onnx 230KB | note 事件+MIDI+confidence | ≈19×RTF | 单声部最佳 |
| CREPE | MIT ✅ | 内置 | f0 轨迹+confidence | 慢 | 需自建 onset/量化 |
| RMVPE | Apache-2.0 ✅ | rmvpe.pt 173MB(HF) | f0+confidence | 快于 CREPE | so-vits 生态验证 |
| essentia Melodia | AGPL ⚠️ | — | 轨迹+voicing | 快 | 学术基线 |
| pYIN/TONY | GPL ⚠️ | — | 轨迹/音符 | 快 | 传染性 |
| midi_melody_extraction | MIT ✅ | — | **输入是 MIDI 非音频，不适用** | — | — |

## 落地注意事项
1. 下载源：demucs 权重 dl.fbaipublicfiles.com；UVR 系 TRvlvr/model_repo；RMVPE HF lj1995。
2. 商用许可：Demucs/basic-pitch/CREPE/torchcrepe/RMVPE 干净 ✅；umxl 权重、essentia、pYIN 有传染/非商用限制；MDX23C/Kim_Vocal_2 权重无明示许可；BS-RoFormer 商用避开。
3. 鼓 onset 干净度无公开定量基准 → 项目内做 A/B。
4. ffmpeg 无 stem 分离能力，仅解码/转码前端。
5. 内存：htdemucs_ft GPU 3-7GB；basic-pitch 峰值 951MB。

# 三、结构分段与副歌识别

## 结论推荐
- **主线：all-in-one（mir-aidj/all-in-one，ISMIR 2023，MIT ✅）**——一个模型同时输出 tempo/beat/downbeat + 段落边界 + **10 类功能标签（intro/verse/chorus/bridge/outro 等）**。坑：输入要求先跑 stem 分离、原版依赖 NATTEN（CUDA 自编译易翻车）→ 用 **openmirlab/all-in-one-infer（MIT ✅，纯 PyTorch 替代 + 内置 Demucs）**。
- **副歌验证（重复段信息）**：all-in-one 不输出 SSM/重复段 → 加"重复段"支线验证"副歌=重复最多段"：**FMP/libfmp + librosa 纯传统方案**（CQT/HPCP → SSM → checkerboard/novelty 边界 → path_enhance/谱聚类提取重复段对；FMP 教材 C4S2/C4S3 有完整实现，MIT/ISC ✅、零模型、秒级）。MERT/music2vec 特征+SSM 质量更高但**权重 cc-by-nc-4.0 非商用**（已核实 95M/330M/music2vec-v1 全系）。
- **许可敏感兜底**：整条传统管线（librosa/libfmp）无任何许可证风险。
- 副歌端点专用校验 DeepChorus：**仓库无 LICENSE ❌**、38 stars、权重未确证 → 不采用。
- 新关注：SongFormer（ASLP-lab，arXiv 2510.02797，2025 transformer 结构分段）；"STransformer"不存在（仅 SpecTNT=beat/downbeat、Stripe-Transformer=分离）。

## 候选对比
| 方法 | 许可 | 边界 | 段落标签 | 重复段 | 预训练 | 精度 | CPU |
|---|---|---|---|---|---|---|---|
| all-in-one | MIT ✅ | ✅ | ✅ 10类含chorus | ❌ 需自建 | HF ✅ | 论文 SOTA（数值待实测） | 需前置 Demucs；GPU 友好 |
| MERT + SSM | 权重 cc-by-nc ⚠️ | 自建 | 自建聚类 | ✅ SSM 对角线 | HF ✅ | 特征基线 | 95M CPU 数十秒~分钟 |
| FMP/libfmp+librosa | MIT/ISC ✅ | ✅ | 聚类后自定 | ✅ SSM 对角线 | 无 | 传统量级 | 秒级 |
| msaf | MIT ✅ | ✅ | 聚类(无语义) | 不直接 | 无大模型 | 传统 | 秒级；老依赖坑 |
| DeepChorus | 无 LICENSE ❌ | 仅副歌区间 | ❌ | ❌ | 未确证 | — | 小模型 |
| Harmonix 数据集 | 标注 | GT | ✅ 多级 | 标签统计 | 只分发特征无音频 | 评测用 | — |

## 落地注意事项
1. msaf 低维护、绑定老 scipy/librosa → 直接 FMP/libfmp 复现更稳。
2. MERT/music2vec 非商用 → 商用管线只用传统特征或自训。
3. 从 SSM 提重复段：偏离主对角线的亮对角线（lag=两段起点差）即重复段对；path_enhance/timelag_filter + 聚类 → 统计出现次数 → "重复最多/总时长最长"=副歌候选 → 与 all-in-one chorus 标签投票确认。
4. "副歌=重复最多段"是启发式：西方流行成立率高；ACG/电子乐（drop/build-up）需 20-50 首人工核对命中率；双线交叉验证比单线稳。

# 四、高潮/情绪强度检测

## 结论推荐：纯 librosa 配方（无模型下载、全宽松许可）
```
1) 加载+响度归一：librosa.load(sr=22050, mono)；pyloudnorm(MIT) 归一 -14 LUFS
2) 同 hop=512 提取：rms；onset_strength（librosa 默认=SuperFlux）；spectral_centroid；可选=Demucs drums stem 逐帧 RMS
3) 逐特征 5-95 百分位截断 → min-max 归一
4) 融合：intensity = 0.35*rms + 0.30*onset + 0.20*cent + 0.15*drums（权重待标定）
5) 对齐节拍聚合：用 beat 时间数组，beats[::4]=小节边界，逐小节均值/P75
6) 平滑：小节级中值滤波（窗≈2小节）+ 高斯(σ≈1)
7) 高潮定位=四票投票（时间对齐后加权）：
   V1 结构票 0.4 = 结构分段的 chorus/重复段内=1
   V2 novelty 票 0.3 = SuperFlux 局部峰值 ±1 小节三角窗
   V3 能量票 0.2 = rms_n>0.75 且局部极大
   V4 质心票 0.1 = cent_n>0.7（电子乐 drop 高频/白噪上升）
   climax = argmax(vote)；前 k 个峰 = 各次副歌出现时间
8) 模板匹配（弱→较强→较弱→强→渐弱）：T=[0.20,0.45,0.30,1.00,0.10]（可参数化）
   a) 有结构结果：chorus 段硬对齐模板第4段；intro/verse/bridge/outro 按序映射
   b) 无结构：PELT/RDP 变点检测分4段 + 分段线性拟合最小残差
   c) 强度曲线归一 [0,1]，与模板 DTW/soft-DTW 对齐
   d) 输出 {小节: 强度∈[0,1]} + 段标签 + 高潮起止 → 谱面端按强度映射 note 密度
```
四票互补：结构票覆盖摇滚 chorus=整段反复；novelty 票覆盖 drop/爆发瞬态；能量票覆盖响度/密度；质心票覆盖电子乐 drop。流派差异建议两套权重预设（电子 drop vs 摇滚 chorus）。

## 候选对比
| 方案 | 许可 | 预训练 | CPU | 输出 | 评价 |
|---|---|---|---|---|---|
| librosa 组合（推荐） | ISC ✅ | 无 | 秒级 | 逐帧/逐小节曲线 | 主干 |
| pychorus | MIT ✅ | 无 | 秒级 | 单个副歌起始秒 | 弃用：单时间点、停维、distutils |
| madmom features | 模型 CC BY-NC-SA ⚠️ | 内置 | 近实时 | onset density | 可被 librosa 等价替代 |
| MERT fine-tune | cc-by-nc ⚠️ | 需自训 | 分钟级 | 逐段情绪 | 过度设计 |
| Essentia DEAM arousal | AGPL/CC BY-NC-SA ⚠️ | .pb | TF 推理 | arousal/valence | 可选插件 |
| musicnn | ISC ✅ | 内置 | 轻量 | taggram 50 标签 | TF1.x 待实测 |

## 落地注意事项
1. 聚合必须用 beat 时间而非固定帧率（变速曲尤其）；hop=512@22050≈23ms，onset 建议 n_fft=2048。
2. 先整曲响度归一（pyloudnorm）再做特征归一，避免"谁响谁高潮"。
3. librosa "audio highlighting" 教程页已 404；等价物=官方 plot_segmentation + FMP C4S3。
4. MTG-Jamendo 标签集含 mood/theme---energetic，可作未来情绪监督数据参考。
5. 本项目模板是强度模板而非情绪模板，传统特征足够；DL 留作可选插件。

# 五、prior art：AI 音游自动制谱项目

## 5.1 必查项：Goldgom/miaChartGen2（已一手核实，gitcode API 抓 README 全文 926 行）
- **它是什么**：README 自称 **maiChartGen3 — AI 谱面生成器**，"基于多阶段 Transformer 的 maimai 谱面自动生成系统：输入一首 MP3 → 自动输出 simai 格式谱面"，是检索到的**唯一 maimai 专用、音频→谱面的生成式 AI 项目**。
- **状态**：GitHub 原仓库 404 已删除/迁移；gitcode 镜像存活 https://gitcode.com/Goldgom/miaChartGen2 ；**无 LICENSE**；README 声明"仅用于学术研究和个人学习"（非 OSI 许可，不可商用）。
- **管线**：
  1. Tokenizer：AudioTokenizer=Meta EnCodec 24kHz（75Hz、8 codebooks、(T,8) token）；BeatTokenizer=librosa 或 beat_this（target_bpm 半速/双速纠正、beat 量化、downbeat）；SimaiToken/SimaiParser=simai↔token（tap/hold/slide/touch/rest/bpm/measure，Each 合并、Touch 合并、slide 路径解析）。
  2. PreProcess：音频/节拍/谱面对齐统一 **75Hz 帧网格** → .npz（audio_tokens、beat_signal、chart_tokens、break/ex/firework mask、hold_dur_targets、slide_path_targets、tag_ids）。
  3. **5-Stage 级联 Transformer**（d_model=512、n_head=8、n_layer=6、自研 AHPE Householder 位置编码）：S1 非自回归谱面骨架逐帧分类（cross-attn 音频，条件=beat 信号+难度+等级+曲库标签）；S2 Hold 时长自回归（64 离散桶 seconds=2^(bin−5)）；S3 Slide 路径 slot-wise 因果+路径合法性校验；S4 Break 二分类（双向）；S5 Ex-note 二分类（双向，仅 DX）。
  4. 采样控制：Temperature/Top-K/密度偏置/Tap-Hold-Slide-Touch-Break 类型偏置/过滤三押。
  5. 输出 simai maidata.txt；Gradio WebUI。
- **数据**：collections/ 按街机版本组织的官方谱面语料（maimai→MURASAKi/PLUS、MiLK、FiNALE、DX 至 PRiSM PLUS 共 35 个版本目录）。
- **资产**：premodels/beatthis.ckpt + encodec_24khz/ + vocab/；**不含 Stage1-5 checkpoint**（作者 HF：https://huggingface.co/Goldgom/models ）。
- **对我们的参照价值（最高）**：BeatTokenizer=我们的节拍层（选型不谋而合）；它**不做声源分离踩音**（EnCodec 全频谱直接进 Transformer）→ **我们的分轨踩音是差异化点**；它不做结构分段 → 我们的结构/强度层可升级其"难度曲线控制"；S4/S5 属性后处理设计可直接借鉴。

## 5.2 其他项目清单
- **DDR/StepMania**：DDC（chrisdonahue/ddc，MIT ✅，多尺度音频特征→CNN+迭代 LSTM 放舞步）；smlab（Tatsh）；GrooveGuru；ITGPT（arXiv 2607.14148）。
- **osu!**：Mapperatorinator（spectrogram 输入生成/改谱）；osu_mapper（扩散端到端）；ARG/softchart（JacobLinCool，**softchart-v15 MIT ✅，7.94M 参数小模型**）；BeatLearning。
- **Beat Saber**：Beat Sage（商业闭源，DDC 作者相关）；InfernoSaber（fred-brenner）。
- **太鼓**：TaikoNation（arXiv 2107.12506，patterning 乐句模式——对应我们的结构分段层）。
- **Guitar Hero**：audio2chart（arXiv 2511.03337，HF 3podi/charter-v1.0-40-M，端到端音频→可玩谱面转录，与"踩音"层最接近）。
- **通用/学术**：Mug-Diffusion（Keytoyze，SD 条件扩散制谱，中文 README）；STRUM；TCP（AIIDE 2025 时间切块）；ChartGenEval（评测框架）；KLab スクスタ AI 制谱（商业管线佐证）。
- **maimai 生态（格式参照）**：MajdataView/MajdataEdit-Neo、MaichartConverter、maidata-rs、ACE、simai atwiki、majdata-charts。
- **未找到 AI 制谱项目的音游**：CHUNITHM、Malody、Project Sekai、Phigros。

## 5.3 共性管线与四层映射
主流：音频 → 特征提取（onset/节拍/频谱/EnCodec）→ 音符时间轴+位置生成（规则|LSTM|Transformer|扩散）→ 属性后处理（难度档/note 类型/Break/EX）→ 游戏格式导出。
与我们的对照：节拍层↔所有项目共识基础；**声源分离分轨踩音 ↔ 未发现任何现有开源项目 → 我们的差异化点**；结构分段层↔TaikoNation/TCP；强度曲线层↔miaChartGen2 密度偏置/Beat Sage 难度档（我们的强度曲线可作显式条件信号）。

# 六、来源清单（全文 90+ 条已核验，详见报告正文）
关键：beat_this、librosa、madmom、all-in-one(+infer)、tempo-cnn、essentia、BeatNet、demucs(adefossez)、spleeter、UVR/model_repo、BS-RoFormer、AudioSep、open-unmix、basic-pitch、crepe、RMVPE、msaf、libfmp/FMP、DeepChorus、harmonixset、MERT-95M/330M、music2vec-v1、SongFormer、pychorus、pyloudnorm、mtg-jamendo、musicnn、miaChartGen2(gitcode)、DDC、smlab、GrooveGuru、ITGPT、Mapperatorinator、osu_mapper、ARG/softchart-v15、BeatLearning、InfernoSaber、TaikoNation、audio2chart、Mug-Diffusion、ChartGenEval、MajdataView、MaichartConverter、maidata-rs 等。

# 七、存疑/待实测项
1. ❓ miaChartGen2 Stage1-5 权重（作者 HF 未核实）；其与 MiDashengLM 关系未证实。
2. ❓ BS-RoFormer ep_317 权重许可、MDX23C/Kim_Vocal_2 权重许可、Spleeter 5stems MIT 覆盖性、Open-Unmix umxhq 许可——商用需法务复核。
3. ❓ All-In-One Harmonix 精确 F-measure 与权重体积；DeepChorus 权重与出版年。
4. ❓ 各模型 CPU RTF 无官方数字 → 建议在目标机器对 2 分钟音游曲实测。
5. ❓ basic-pitch 对 160-220BPM/声码器电子人声表现；鼓 stem onset 定量干净度。
6. ❓ "副歌=重复最多段"在 ACG/电子乐命中率（建议 20-50 首人工核对）；强度配方权重与模板系数需在目标曲库标定。
7. ❓ UVR 主仓库 LICENSE 未抓到。
8. ❓ 全部模型在日语 ACG/音游曲（160-220BPM）上的域外泛化精度——所有基准均为西方流行乐。
9. ❓ msaf Python 兼容范围；pYIN 原站不可访问。
10. ⚠️ 建议补抓：miaChartGen2 HF 权重、Mug-Diffusion 许可、DDC/ITGPT/InfernoSaber LICENSE。

## 行动建议（给本项目）
四层主线 = **beat_this（节拍）+ Demucs htdemucs_ft（分离）+ basic-pitch（采旋律）+ all-in-one（结构）+ librosa 配方（强度）**，全 MIT/Apache/ISC 干净许可；miaChartGen2 的 5-Stage 设计（骨架→hold→slide→break/ex）+ EnCodec 75Hz 帧网格 + simai tokenizer 是谱面生成端最值得研读的参照（但"仅学术研究"条款 → 只能借鉴设计、不可直接商用其代码）。
