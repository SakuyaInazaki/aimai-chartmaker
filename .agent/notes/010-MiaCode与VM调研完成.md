# 010 — MiaCode/VM 调研完成：报错校验文档 v1.0 定稿，第 3 步收尾

- **日期**：2026-09-10
- **变更**：
  1. 新增 `docs/research/miacode-vm-research.md`（MiaCode 源码取证 + Visual Maimai 文档侧取证完整报告）。
  2. `docs/simai-error-checking.md` 定稿 v1.0：补 §8 MiaCode（双模式架构、34 类规则、slide_data.json 白名单、Muri 检测常量）、§9 Visual Maimai（闭源取证结论）、§10 校验器蓝本升级（四层 + 双检 + 回归用例）。
  3. `docs/simai-syntax.md` 增量更新：连锁 slide 统一"总时长在最后"形式（MiaCode strict 会拒绝分段独立时长）；64 格关系表找到机读替代（MiaCode slide_data.json）；来源补充。
  4. 知识库 008/010/011 补充 MiaCode Muri 检测的交叉印证（5 类一致、外键/撞尾 gap 分级、静态阈值 200ms 可调）。
  5. `AGENT.md` 勾选"调研 simai 报错/警告判定逻辑"待办。
- **理由**：第 3 步（报错判定调研）完成。MiaCode 是"professional chart creators"级编辑器，其双模式校验与结构化报错最适合作为本项目校验器蓝本；VM 源码未开源，规则仅能文档侧取证（置信度低，已如实标注）。
- **来源**：调研子代理（agent 发起，用户指定调研方向）。
- **关联**：`docs/simai-error-checking.md`、`docs/research/miacode-vm-research.md`、`docs/simai-syntax.md`、知识库 008/010/011。

## 关键结论备忘

1. **MiaCode = 校验器蓝本**：lenient（预览）+ strict（发布门禁）双模式；报错带 {line,col,endCol,severity}；34 类规则全部有源码位置。
2. **slide_data.json（608 keys）解决了官方 64 格表无法机读的问题**——生成器 slide 端点校验直接复用该白名单。
3. **新发现的安全子集约束**：连锁 slide 分段独立时长（>1 个 `[…]`）在 MiaCode strict 报错 → 生成器只用"总时长在最后"形式。
4. **LLM 高危项**：MiaCode 对全角字符逐类报错（7 类）——AI 生成文本极易混入全角符号，校验器必须覆盖。
5. **meta 序列化兼容细节**：`&first` 空值会崩 MajdataPlay（补 0）、空可选字段剔除、`&clock_count` 默认 4。
6. **VM 结论**：闭源黑盒；原生检查仅多押/叠键类（slide 类无理靠 MaiMuriDX Mod 补全）——对生成器校验价值有限。
7. **回归用例**：MiaCode SimaiParserSpec.cpp（1217 行）+ samples/ 可构建零报错/应报错样例集。
