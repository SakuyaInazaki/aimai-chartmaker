# 013 — Touch 摆放经验

- **日期**：2026-09-10
- **要点**：Touch 必须成组摆放（相邻区组成 TouchGroup 才可单手处理）；若干摆放禁忌构成生成器约束。
- **详情**：
  - 相邻区 TouchGroup 才可单手；"不建议写 C/E7/E3 这样的间隔 Touch"（E8/E1/E2 尚可）——孤立的远距离 Touch 需要双手腾挪，属硬多押；
  - Touch 扫入 TouchHold 的配置中，**A、D 区 Touch 被认为不可以写**（如 `Ch[4:1]/A7/D7/A6/A2/D3/A3` 禁；B2/B3 可）；
  - TouchHold 中途跨接 Touch = 软无理，尽量不写；
  - Slide 撞 Touch 必须用快速 slide（≥ 约 120BPM 的 8:1）；中速 4:1 撞 Touch 手感怪异；
  - Touch 无 Fast 判定（只有 Late）。
- **理由**：Touch 需要手离开按键区去拍屏，摆放不当直接造成多押无理；规则来自社区实测与 MaiMuriDX 检测标准。
- **来源**：Maimai 无理综述（五）https://www.bilibili.com/opus/976827828362280981
- **置信度**：经验（社区检测标准）
- **关联**：知识 006、008。
