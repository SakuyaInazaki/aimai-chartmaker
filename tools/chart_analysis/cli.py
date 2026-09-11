#!/usr/bin/env python3
"""官方谱密度分析命令行入口。

用法（仓库根目录下执行）::

    python3 -m chart_analysis.cli validate            # 解析覆盖率 + 官方 note 数对账
    python3 -m chart_analysis.cli summary             # 写 CSV 汇总 + 逐小节明细
    python3 -m chart_analysis.cli analyze             # 全库曲线统计 / 聚类 / 模板检验
    python3 -m chart_analysis.cli chart <文件名关键字>  # 单谱逐小节曲线

需要先把 ``tools/`` 加进 ``PYTHONPATH``：``PYTHONPATH=tools python3 -m chart_analysis.cli ...``
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import corpus
from .density import (
    RESAMPLE_BINS,
    TEMPLATE_K001,
    ChartDensity,
    chart_density,
    changepoint_segments,
    structure_profile,
    kmeans,
    low_density_runs,
    pearson,
    plateau_count,
    segment_means,
    silhouette,
    template_fit,
)
from .simai_parser import parse_chart

OUT_DIR = corpus.REPO_ROOT / "out" / "chart_analysis"
CSV_PATH = corpus.REPO_ROOT / "docs" / "research" / "data" / "official-chart-density-summary.csv"


# ---------------------------------------------------------------------------


def _parse_all(verbose: bool = False):
    """解析全部官方谱，返回 ``[(ChartFile, ParseResult, ChartDensity), ...]``。"""
    files = corpus.discover()
    out = []
    for cf in files:
        res = parse_chart(cf.read(), name=cf.name)
        dens = chart_density(res, name=cf.name)
        out.append((cf, res, dens))
        if verbose and res.errors:
            print(f"[ERR] {cf.name}: {res.errors[0]}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def cmd_validate(args) -> None:
    data = _parse_all()
    n = len(data)
    err_files = [(cf, res) for cf, res, _ in data if res.errors]
    warn_files = [(cf, res) for cf, res, _ in data if res.warnings]
    print(f"官方谱文件数：{n}")
    print(f"解析成功（零 error）：{n - len(err_files)} / {n}"
          f"  = {100 * (n - len(err_files)) / n:.2f}%")
    print(f"带告警（warning，仍可用）：{len(warn_files)}")
    if err_files:
        print("\n-- 解析出错的文件及原因 --")
        for cf, res in err_files:
            uniq = sorted({e.split(": ", 1)[-1][:80] for e in res.errors})
            print(f"  {cf.name}  (error {len(res.errors)} 条)")
            for u in uniq[:4]:
                print(f"      {u}")
    if warn_files:
        print("\n-- 告警汇总 --")
        wc = Counter()
        for _, res in warn_files:
            for w in res.warnings:
                wc[w.split(": ", 1)[-1][:60]] += 1
        for k, v in wc.most_common(10):
            print(f"  {v:4d} × {k}")

    # ---- 与 manifest 官方计数对账 ----
    have = [(cf, res) for cf, res, _ in data if cf.gt_notes is not None]
    print(f"\n-- 与 manifest 官方 note 数对账（{len(have)} / {n} 谱有官方计数）--")
    exact = 0
    deltas = []
    per_field = {k: 0 for k in ("taps", "hold", "slide", "touch", "breaks")}
    for cf, res in have:
        c = res.counts
        d = c["notes"] - cf.gt_notes
        deltas.append(d)
        if d == 0:
            exact += 1
        for k, gt in (
            ("taps", cf.gt_taps),
            ("hold", cf.gt_hold),
            ("slide", cf.gt_slide),
            ("touch", cf.gt_touch),
            ("breaks", cf.gt_breaks),
        ):
            if (gt or 0) == c[k]:
                per_field[k] += 1
    deltas = np.array(deltas)
    print(f"总数完全一致：{exact} / {len(have)} = {100 * exact / len(have):.2f}%")
    print(f"|Δ| <= 3     ：{int((np.abs(deltas) <= 3).sum())} / {len(have)}"
          f" = {100 * (np.abs(deltas) <= 3).mean():.2f}%")
    print(f"Δ 统计：mean={deltas.mean():+.3f}  max|Δ|={int(np.abs(deltas).max())}")
    for k, v in per_field.items():
        print(f"  分项 {k:<7}一致：{v} / {len(have)} = {100 * v / len(have):.2f}%")
    if args.list_mismatch:
        print("\n-- 不一致清单 --")
        for cf, res in have:
            c = res.counts
            if c["notes"] != cf.gt_notes:
                print(f"  {cf.name}: 官方 {cf.gt_notes} vs 解析 {c['notes']}"
                      f" (Δ{c['notes'] - cf.gt_notes:+d})")

    # ---- 与知识 004 的 note 总数分布对照 ----
    print("\n-- 与知识 004（note 总数分布）对照 --")
    k004 = {
        "13.0": (643.7, 445, 828),
        "13.5": (767.4, 492, 967),
        "14.0": (899.8, 685, 1080),
        "14.5": (1020.7, 733, 1181),
    }
    buckets = defaultdict(list)
    for cf, res, _ in data:
        buckets[corpus.level_bucket(cf.internal_level)].append(res.counts["notes"])
    for b in sorted(buckets):
        arr = np.array(buckets[b])
        ref = k004.get(b)
        ref_s = f"  知识004(SD): 均值 {ref[0]:.1f} p10–p90 {ref[1]}–{ref[2]}" if ref else ""
        print(f"  定数 {b}（{len(arr)} 谱）：均值 {arr.mean():.1f}  "
              f"p10–p90 {np.percentile(arr, 10):.0f}–{np.percentile(arr, 90):.0f}{ref_s}")


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def cmd_summary(args) -> None:
    data = _parse_all()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_dir = OUT_DIR / "measures"
    detail_dir.mkdir(parents=True, exist_ok=True)

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "simai_id", "title", "difficulty", "internal_level", "level", "bpm_first",
        "total_notes", "taps", "hold", "slide", "breaks", "touch",
        "gt_notes", "notes_delta", "measures", "duration_sec", "nps",
        "notes_per_measure_mean", "peak_measure", "peak_notes", "peak_position",
        "peak_position_smooth",
        "rest_measures", "rest_runs", "longest_rest_run", "plateau_count",
        "seg1", "seg2", "seg3", "seg4", "seg5", "corr_with_T001", "rmse_with_T001",
        "parse_errors", "parse_warnings",
    ] + [f"b{i:02d}" for i in range(RESAMPLE_BINS)]

    with CSV_PATH.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for cf, res, dens in data:
            if dens.measure_count == 0:
                continue
            c = res.counts
            rests = [r for r in dens.rest_runs]
            prof = dens.segment_profile
            row = [
                cf.simai_id, cf.title, cf.difficulty,
                cf.internal_level if cf.internal_level is not None else "",
                cf.level or "",
                f"{res.bpm_events[0][1]:.3f}" if res.bpm_events else "",
                c["notes"], c["taps"], c["hold"], c["slide"], c["breaks"], c["touch"],
                cf.gt_notes if cf.gt_notes is not None else "",
                (c["notes"] - cf.gt_notes) if cf.gt_notes is not None else "",
                dens.measure_count,
                f"{res.total_seconds:.2f}",
                f"{c['notes'] / res.total_seconds:.3f}" if res.total_seconds > 0 else "",
                f"{dens.raw.mean():.3f}",
                dens.peak_measure, dens.peak_notes, f"{dens.peak_position:.4f}",
                f"{dens.peak_position_smooth:.4f}",
                sum(l for _, l in rests), len(rests),
                max((l for _, l in rests), default=0),
                dens.plateau_count,
            ] + [f"{v:.4f}" for v in prof] + [
                f"{pearson(prof, TEMPLATE_K001):.4f}",
                f"{float(np.sqrt(((prof - TEMPLATE_K001) ** 2).mean())):.4f}",
                len(res.errors), len(res.warnings),
            ] + [f"{v:.4f}" for v in dens.resampled]
            w.writerow(row)

            # 逐小节明细（体积大，只落 out/，不入库）
            with (detail_dir / f"{cf.name}.csv").open("w", encoding="utf-8", newline="") as dh:
                dw = csv.writer(dh)
                dw.writerow([
                    "measure", "start_time", "bpm", "notes", "taps", "holds", "slides",
                    "stars", "breaks", "touches", "groups", "each_groups", "each_notes",
                    "finest_divisor", "is_rest",
                ])
                for s in dens.measures:
                    dw.writerow([
                        s.measure, f"{s.start_time:.3f}", f"{s.bpm:g}", s.notes, s.taps,
                        s.holds, s.slides, s.stars, s.breaks, s.touches, s.groups,
                        s.each_groups, s.each_notes, f"{s.finest_divisor:g}",
                        1 if s.is_rest else 0,
                    ])
    print(f"已写入汇总 CSV：{CSV_PATH}")
    print(f"逐小节明细（{len(data)} 份）：{detail_dir}")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def _group_stats(label: str, dens_list: list[ChartDensity]) -> dict:
    curves = np.array([d.resampled for d in dens_list])
    profiles = np.array([d.segment_profile for d in dens_list])
    fit = template_fit(profiles)
    return {
        "label": label,
        "n": len(dens_list),
        "mean_curve": curves.mean(axis=0),
        "p25": np.percentile(curves, 25, axis=0),
        "p50": np.percentile(curves, 50, axis=0),
        "p75": np.percentile(curves, 75, axis=0),
        "template": fit,
    }


def cmd_analyze(args) -> None:
    data = _parse_all()
    data = [(cf, res, d) for cf, res, d in data if d.measure_count > 0]
    dens_all = [d for _, _, d in data]
    curves = np.array([d.resampled for d in dens_all])
    profiles = np.array([d.segment_profile for d in dens_all])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"==== 全库（{len(dens_all)} 谱）====")
    print("32-bin 平均曲线（除以各谱自身峰值后平均）：")
    _print_curve(curves.mean(axis=0))
    print("中位曲线：")
    _print_curve(np.percentile(curves, 50, axis=0))
    print("p25 / p75 带宽（逐 bin）：")
    _print_curve(np.percentile(curves, 25, axis=0))
    _print_curve(np.percentile(curves, 75, axis=0))

    # ---- 分组 ----
    print("\n==== 分组平均曲线 ====")
    groups: dict[str, list[ChartDensity]] = defaultdict(list)
    for cf, _, d in data:
        groups["mas" if cf.difficulty == "mas" else "remas"].append(d)
        groups["定数" + corpus.level_bucket(cf.internal_level)].append(d)
    for label in sorted(groups):
        g = groups[label]
        if len(g) < 5:
            continue
        gc = np.array([d.resampled for d in g])
        print(f"\n[{label}] n={len(g)}")
        _print_curve(gc.mean(axis=0))

    # ---- 聚类 ----
    print("\n==== k-means 形状聚类 ====")
    cluster_out = {}
    for k in (3, 4, 5):
        labels, centers, inertia = kmeans(curves, k, seed=7, restarts=25)
        sil = silhouette(curves, labels)
        print(f"\nk={k}  inertia={inertia:.2f}  silhouette={sil:.4f}")
        for j in range(k):
            idx = np.where(labels == j)[0]
            print(f"  簇{j}  n={len(idx)} ({100 * len(idx) / len(curves):.1f}%)"
                  f"  峰位中位={np.median([dens_all[i].peak_position for i in idx]):.2f}")
            _print_curve(centers[j], indent="      ")
            names = [dens_all[i].name for i in idx[:4]]
            print(f"      代表：{'、'.join(names)}")
        cluster_out[k] = {
            "labels": labels.tolist(),
            "centers": centers.tolist(),
            "inertia": inertia,
            "silhouette": sil,
        }

    # ---- 五段模板 ----
    print("\n==== 知识 001 五段模板检验 ====")
    fit = template_fit(profiles)
    print(f"T(知识001) = {np.round(TEMPLATE_K001, 3).tolist()}")
    print(f"逐谱相关系数：均值 {fit['corr_mean']:.4f}  中位 {fit['corr_median']:.4f}  "
          f"r>0 占比 {100 * fit['corr_positive_ratio']:.1f}%")
    print(f"逐谱 RMSE：均值 {fit['rmse_mean']:.4f}  "
          f"中位 {np.median(fit['rmse']):.4f}  p90 {np.percentile(fit['rmse'], 90):.4f}")
    print(f"数据估计模板（均值）= {np.round(fit['estimate'], 3).tolist()}")
    print(f"  95% 自助 CI 下 = {np.round(fit['ci_low'], 3).tolist()}")
    print(f"  95% 自助 CI 上 = {np.round(fit['ci_high'], 3).tolist()}")
    print(f"数据估计模板（中位）= {np.round(fit['median_profile'], 3).tolist()}")
    est = fit["estimate"] / fit["estimate"].max()
    print(f"数据估计模板（均值后再归一到峰值=1）= {np.round(est, 3).tolist()}")
    print(f"  与 T 的相关：{pearson(fit['estimate'], TEMPLATE_K001):.4f}  "
          f"RMSE：{np.sqrt(((est - TEMPLATE_K001) ** 2).mean()):.4f}")
    # 哪一段是峰值段
    seg_peak = Counter(int(np.argmax(p)) + 1 for p in profiles)
    print("各谱五段中最强段落分布："
          + "  ".join(f"第{k}段 {v}谱({100 * v / len(profiles):.1f}%)"
                      for k, v in sorted(seg_peak.items())))
    seg_min = Counter(int(np.argmin(p)) + 1 for p in profiles)
    print("各谱五段中最弱段落分布："
          + "  ".join(f"第{k}段 {v}谱({100 * v / len(profiles):.1f}%)"
                      for k, v in sorted(seg_min.items())))

    # ---- 序关系检验（T 的定性形状是否成立）----
    print("\n-- 等分五段的序关系 --")
    print(f"  P(最强段落在第4或第5段) = {100 * np.mean(np.argmax(profiles, axis=1) >= 3):.1f}%")
    print(f"  P(第1段最弱)            = {100 * np.mean(np.argmin(profiles, axis=1) == 0):.1f}%")
    print(f"  P(第2段 > 第3段)        = {100 * np.mean(profiles[:, 1] > profiles[:, 2]):.1f}%")
    print(f"  P(第5段 < 第4段)        = {100 * np.mean(profiles[:, 4] < profiles[:, 3]):.1f}%")
    print(f"  相对全曲均值的五段     = "
          f"{np.round(np.array([segment_means(d.raw, 5) / d.raw.mean() for d in dens_all]).mean(axis=0), 3).tolist()}")

    # ---- 结构对齐五段（变点检测 + 最小段长 10%）----
    cp_profiles, cp_lens = [], []
    for d in dens_all:
        sp = structure_profile(d.normalized, 5, min_len_frac=0.10)
        if sp is None:
            continue
        cp_profiles.append(sp[0])
        cp_lens.append(sp[1])
    cp_profiles = np.array(cp_profiles)
    cp_lens = np.array(cp_lens)
    cp_fit = template_fit(cp_profiles)
    print(f"\n-- 结构对齐五段（变点检测，最小段长 10%）n={len(cp_profiles)} --")
    print(f"  段长占比均值 = {np.round(cp_lens.mean(axis=0), 3).tolist()}")
    print(f"  估计模板（均值）= {np.round(cp_fit['estimate'], 3).tolist()}")
    print(f"  95% CI 下 = {np.round(cp_fit['ci_low'], 3).tolist()}")
    print(f"  95% CI 上 = {np.round(cp_fit['ci_high'], 3).tolist()}")
    print(f"  中位模板 = {np.round(cp_fit['median_profile'], 3).tolist()}")
    print(f"  与 T 相关：均值 {cp_fit['corr_mean']:.4f}  中位 {cp_fit['corr_median']:.4f}")
    print(f"  段序（升序）= {np.argsort(cp_fit['estimate']) + 1}  "
          f"（T 的段序 = {np.argsort(TEMPLATE_K001) + 1}）")
    rel_cp = np.array(
        [segment_means(d.raw, 5) / d.raw.mean() for d in dens_all]
    )
    print(f"  逐小节归一密度分位（各谱 p10/p50/p90 的均值）= "
          f"{np.round(np.array([[np.percentile(d.normalized, q) for q in (10, 50, 90)] for d in dens_all]).mean(axis=0), 3).tolist()}")
    del rel_cp

    # ---- 高潮位置 ----
    print("\n==== 高潮（峰值小节）位置 ====")
    pp = np.array([d.peak_position for d in dens_all])
    print(f"相对位置：均值 {pp.mean():.3f}  中位 {np.median(pp):.3f}  "
          f"p10 {np.percentile(pp, 10):.3f}  p90 {np.percentile(pp, 90):.3f}")
    hist, edges = np.histogram(pp, bins=10, range=(0, 1))
    for i in range(10):
        print(f"  [{edges[i]:.1f},{edges[i + 1]:.1f})  {hist[i]:3d} 谱 "
              f"({100 * hist[i] / len(pp):5.1f}%)  " + "█" * int(40 * hist[i] / hist.max()))
    print(f"峰值落在后半程（>0.5）占比：{100 * (pp > 0.5).mean():.1f}%")
    print(f"峰值恰在最后 1 小节占比：{100 * np.mean([d.peak_measure == d.measures[-1].measure for d in dens_all]):.1f}%"
          "（说明单小节峰值基本不是收尾爆发劫持的）")
    ps = np.array([d.peak_position_smooth for d in dens_all])
    print(f"\n[更稳健] 4 小节滑动均值最大处（\"高潮段\"中心）：均值 {ps.mean():.3f}  中位 {np.median(ps):.3f}")
    hist2, edges2 = np.histogram(ps, bins=10, range=(0, 1))
    for i in range(10):
        print(f"  [{edges2[i]:.1f},{edges2[i + 1]:.1f})  {hist2[i]:3d} 谱 "
              f"({100 * hist2[i] / len(ps):5.1f}%)  " + "\u2588" * int(40 * hist2[i] / hist2.max()))
    print(f"  后半程（>0.5）占比：{100 * (ps > 0.5).mean():.1f}%；"
          f"[0.6,1.0] 占比：{100 * (ps >= 0.6).mean():.1f}%")

    # ---- 第一副歌 vs 第二副歌 ----
    print("\n==== 前半 / 后半高密段对比（第二副歌是否更强）====")
    ratios = []
    for d in dens_all:
        c = d.normalized
        half = len(c) // 2
        a, b = c[:half], c[half:]
        if len(a) == 0 or len(b) == 0:
            continue
        # 用各半程「密度前 15% 小节的均值」代表该半程的副歌强度
        ta = np.mean(np.sort(a)[-max(1, int(0.15 * len(a))):])
        tb = np.mean(np.sort(b)[-max(1, int(0.15 * len(b))):])
        if ta > 0:
            ratios.append(tb / ta)
    ratios = np.array(ratios)
    print(f"后半/前半 高密强度比：均值 {ratios.mean():.3f}  中位 {np.median(ratios):.3f}")
    print(f"  后半 > 前半 的谱占比：{100 * (ratios > 1.0).mean():.1f}%")
    print(f"  提升 >5% 占比：{100 * (ratios > 1.05).mean():.1f}%；"
          f"提升 >10% 占比：{100 * (ratios > 1.10).mean():.1f}%")

    # ---- 休息段 ----
    print("\n==== 休息段 / 低密段 ====")
    rest_counts, rest_lens, rest_pos, cover = [], [], [], []
    low_counts, low_lens, sandwich = [], [], []
    for d in dens_all:
        rr = d.rest_runs
        rest_counts.append(len(rr))
        cover.append(sum(l for _, l in rr) / d.measure_count)
        for st, ln in rr:
            rest_lens.append(ln)
            rest_pos.append(st / max(1, d.measure_count - 1))
        lr = low_density_runs(d.normalized, ratio=0.5)
        low_counts.append(len(lr))
        hi = 0.8
        cnt = 0
        for st, ln in lr:
            before = d.normalized[:st]
            after = d.normalized[st + ln:]
            if len(before) and len(after) and before.max() >= hi and after.max() >= hi:
                cnt += 1
        sandwich.append(cnt)
        low_lens.extend(ln for _, ln in lr)
    rest_counts = np.array(rest_counts)
    rest_lens = np.array(rest_lens) if rest_lens else np.zeros(1)
    print(f"休息小节（note<=1）：{100 * np.mean([c > 0 for c in rest_counts]):.1f}% 的谱至少有 1 段；"
          f"每谱平均 {rest_counts.mean():.2f} 段")
    print(f"休息段长度：均值 {rest_lens.mean():.2f} 小节  中位 {np.median(rest_lens):.0f}  "
          f"p90 {np.percentile(rest_lens, 90):.0f}  最长 {rest_lens.max():.0f}")
    print(f"休息小节占全曲比例：均值 {100 * np.mean(cover):.2f}%")
    if rest_pos:
        print(f"休息段起点相对位置：中位 {np.median(rest_pos):.3f}")
    low_lens_a = np.array(low_lens) if low_lens else np.zeros(1)
    print(f"低密段（< 全曲中位密度 × 0.5）：每谱平均 {np.mean(low_counts):.2f} 段，"
          f"平均长 {low_lens_a.mean():.2f} 小节")
    print(f"「夹在两个高密段之间」的低密段：每谱平均 {np.mean(sandwich):.2f} 段；"
          f"{100 * np.mean([s > 0 for s in sandwich]):.1f}% 的谱至少有 1 段")
    plat = np.array([d.plateau_count for d in dens_all])
    print(f"密度台阶（平台段，>=4 小节同档）：每谱平均 {plat.mean():.2f} 个，中位 {np.median(plat):.0f}")

    # ---- 落盘 ----
    dump = {
        "n": len(dens_all),
        "mean_curve": curves.mean(axis=0).tolist(),
        "p25": np.percentile(curves, 25, axis=0).tolist(),
        "p50": np.percentile(curves, 50, axis=0).tolist(),
        "p75": np.percentile(curves, 75, axis=0).tolist(),
        "template_estimate": fit["estimate"].tolist(),
        "template_ci_low": fit["ci_low"].tolist(),
        "template_ci_high": fit["ci_high"].tolist(),
        "template_corr_mean": fit["corr_mean"],
        "clusters": cluster_out,
        "peak_position": pp.tolist(),
    }
    (OUT_DIR / "library-stats.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n已写出：{OUT_DIR / 'library-stats.json'}")


def _print_curve(c: np.ndarray, indent: str = "  ") -> None:
    print(indent + " ".join(f"{v:.2f}" for v in c))
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = float(c.min()), float(c.max())
    rng = max(1e-9, hi - lo)
    print(indent + "".join(blocks[min(7, int((v - lo) / rng * 7.999))] for v in c))


# ---------------------------------------------------------------------------
# chart（单谱）
# ---------------------------------------------------------------------------


def cmd_chart(args) -> None:
    files = [cf for cf in corpus.discover() if args.pattern in cf.name]
    if not files:
        print(f"未找到匹配 {args.pattern!r} 的谱面")
        return
    for cf in files:
        res = parse_chart(cf.read(), name=cf.name)
        d = chart_density(res, name=cf.name)
        print(f"\n==== {cf.name} ====")
        print(f"定数 {cf.internal_level}  官方 note {cf.gt_notes}  解析 note {res.counts['notes']}"
              f"  小节 {d.measure_count}  时长 {res.total_seconds:.1f}s")
        print(f"峰值：第 {d.peak_measure} 小节 {d.peak_notes} note，相对位置 {d.peak_position:.3f}")
        print(f"休息段：{d.rest_runs}")
        print(f"五段（等分，归一到峰值）：{np.round(d.segment_profile, 3).tolist()}")
        print(f"与 T001 相关：{pearson(d.segment_profile, TEMPLATE_K001):.4f}")
        print("\n逐小节（小节号 | 时间 | note 数 | 条形）")
        peak = max(1, d.peak_notes)
        for s in d.measures:
            bar = "█" * int(round(40 * s.notes / peak))
            print(f"  {s.measure:4d} | {s.start_time:7.2f}s | {s.notes:3d} | "
                  f"t{s.taps:2d} h{s.holds:2d} s{s.slides:2d} b{s.breaks:2d} "
                  f"e{s.each_groups:2d} 1/{s.finest_divisor:g} | {bar}")


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="官方谱逐小节 note 密度分析")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="解析覆盖率与官方 note 数对账")
    v.add_argument("--list-mismatch", action="store_true", help="列出计数不一致的谱")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("summary", help="写出汇总 CSV 与逐小节明细")
    s.set_defaults(func=cmd_summary)

    a = sub.add_parser("analyze", help="全库曲线统计 / 聚类 / 五段模板检验")
    a.set_defaults(func=cmd_analyze)

    c = sub.add_parser("chart", help="单谱逐小节曲线")
    c.add_argument("pattern", help="文件名关键字，如 'SPICY'")
    c.set_defaults(func=cmd_chart)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
