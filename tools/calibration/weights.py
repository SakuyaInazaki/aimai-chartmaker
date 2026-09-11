"""强度融合权重标定：五项 Z 特征 → 官方逐小节密度。

`tools/audio_analysis/intensity.py` 的融合式是

```
fused = Σ_k w_k · Z(x_k) / Σ_k w_k        （Z = 全曲逐小节 z-score）
```

因为随后只做一次单调的 `robust_unit`，**Spearman 相关只取决于 w 的方向**，
与 w 的整体缩放无关。所以标定就是"在 8 首曲子上找一组非负 w，使
`fused` 与官方密度的秩相关最大"。

做法：
1. 每曲各自 z 标准化 5 个特征与目标密度（去掉曲间量纲差异）；
2. 池化后做**非负最小二乘**（scipy.optimize.nnls）或**岭回归**；
3. 归一到 Σw = 1；
4. **留一曲交叉验证**（8 折）：在 7 首上拟合、在留出曲上算 Spearman，
   与初值 `[0.25,0.30,0.20,0.15,0.10]` 逐折对照。

⚠️ 只有当 CV 指标**明显优于**初值时才替换默认权重（阈值与结论见报告）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chartpair import spearman

FEATURE_ORDER = ("loudness", "onset", "drums", "voiced", "flux")
INITIAL_WEIGHTS = {"loudness": 0.25, "onset": 0.30, "drums": 0.20,
                   "voiced": 0.15, "flux": 0.10}


def zscore(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    s = float(x.std())
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - float(x.mean())) / s


def zscore_columns(X) -> np.ndarray:
    """逐列 z 标准化（列 = 特征）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.column_stack([zscore(X[:, j]) for j in range(X.shape[1])]) \
        if X.size else X


def fit_nnls(X, y) -> np.ndarray:
    """非负最小二乘。scipy 不可用时退化为"梯度投影"的简易实现。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)
    try:
        from scipy.optimize import nnls

        w, _ = nnls(X, y)
        return np.asarray(w, dtype=float)
    except Exception:  # pragma: no cover - 仅在无 scipy 时触发
        w = np.zeros(X.shape[1])
        lr = 1.0 / (np.linalg.norm(X, 2) ** 2 + 1e-9)
        for _ in range(5000):
            w = np.maximum(0.0, w - lr * (X.T @ (X @ w - y)))
        return w


def fit_ridge(X, y, alpha: float = 1.0, nonneg: bool = True) -> np.ndarray:
    """岭回归；``nonneg=True`` 时把岭问题改写成增广的 NNLS（等价，保证非负）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)
    if nonneg:
        n_f = X.shape[1]
        Xa = np.vstack([X, np.sqrt(float(alpha)) * np.eye(n_f)])
        ya = np.concatenate([y, np.zeros(n_f)])
        return fit_nnls(Xa, ya)
    G = X.T @ X + float(alpha) * np.eye(X.shape[1])
    return np.linalg.solve(G, X.T @ y)


def normalize_weights(w) -> np.ndarray:
    """归一到 Σw = 1（全 0 时退回均匀权重）。"""
    w = np.maximum(0.0, np.asarray(w, dtype=float))
    s = float(w.sum())
    if s < 1e-12:
        return np.full(w.size, 1.0 / max(1, w.size))
    return w / s


def fuse(X, w) -> np.ndarray:
    """按权重融合（与 intensity.py 同式：加权和 / Σw）。"""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    w = np.asarray(w, dtype=float)
    s = float(w.sum()) or 1.0
    return (X @ w) / s


@dataclass
class Fold:
    held_out: str
    weights: dict[str, float]
    rho_fit: float          # 标定权重在留出曲上的 Spearman
    rho_init: float         # 初值权重在留出曲上的 Spearman

    def to_dict(self) -> dict:
        return {"held_out": self.held_out,
                "weights": {k: round(v, 4) for k, v in self.weights.items()},
                "rho_fit": round(self.rho_fit, 4),
                "rho_init": round(self.rho_init, 4),
                "delta": round(self.rho_fit - self.rho_init, 4)}


@dataclass
class CalibrationResult:
    feature_order: tuple[str, ...] = FEATURE_ORDER
    weights_all: dict[str, float] = field(default_factory=dict)   # 全量拟合
    folds: list[Fold] = field(default_factory=list)
    rho_fit_mean: float = float("nan")
    rho_init_mean: float = float("nan")
    rho_fit_median: float = float("nan")
    rho_init_median: float = float("nan")
    n_wins: int = 0

    def to_dict(self) -> dict:
        return {
            "feature_order": list(self.feature_order),
            "weights_all": {k: round(v, 4) for k, v in self.weights_all.items()},
            "folds": [f.to_dict() for f in self.folds],
            "rho_fit_mean": round(self.rho_fit_mean, 4),
            "rho_init_mean": round(self.rho_init_mean, 4),
            "rho_fit_median": round(self.rho_fit_median, 4),
            "rho_init_median": round(self.rho_init_median, 4),
            "n_wins": self.n_wins, "n_folds": len(self.folds),
        }


def evaluate_presets(datasets: dict[str, tuple[np.ndarray, np.ndarray]],
                     presets: dict[str, dict[str, float]],
                     feature_order: tuple[str, ...] = FEATURE_ORDER) -> dict:
    """给定若干组**手写**权重，直接在每首曲子上算 Spearman（不涉及拟合，无需 CV）。

    用途：比较"初值"与几组可解释的候选（例如把 `voiced` 的权重挪给 `flux`），
    避免只能在"初值 vs 黑箱拟合值"之间二选一。
    """
    out: dict[str, dict] = {}
    for pname, w in presets.items():
        wv = np.array([float(w.get(k, 0.0)) for k in feature_order], dtype=float)
        rhos = {}
        for name, (X, y) in datasets.items():
            Xs = zscore_columns(X)
            if not Xs.size:
                continue
            rhos[name] = round(spearman(fuse(Xs, wv), np.asarray(y, dtype=float)), 4)
        vals = [v for v in rhos.values() if np.isfinite(v)]
        out[pname] = {
            "weights": dict(w), "per_song": rhos,
            "mean": round(float(np.mean(vals)), 4) if vals else None,
            "median": round(float(np.median(vals)), 4) if vals else None,
        }
    return out


def calibrate(datasets: dict[str, tuple[np.ndarray, np.ndarray]],
              alpha: float = 1.0,
              feature_order: tuple[str, ...] = FEATURE_ORDER,
              initial: dict[str, float] | None = None) -> CalibrationResult:
    """留一曲交叉验证的权重标定。

    参数：
        datasets: ``{曲名: (X, y)}``；X 形状 (n_bars, n_features)，y 为该曲逐小节密度。
                  两者都会在**曲内**做 z 标准化后再池化。
        alpha:    岭系数（0 = 纯 NNLS）
    """
    init = dict(initial or INITIAL_WEIGHTS)
    w_init = np.array([init.get(k, 0.0) for k in feature_order], dtype=float)

    prepared = {}
    for name, (X, y) in datasets.items():
        Xs = zscore_columns(X)
        ys = zscore(y)
        if Xs.size and ys.size:
            prepared[name] = (Xs, ys)

    names = sorted(prepared)
    res = CalibrationResult(feature_order=tuple(feature_order))
    if not names:
        return res

    X_all = np.vstack([prepared[n][0] for n in names])
    y_all = np.concatenate([prepared[n][1] for n in names])
    w_all = normalize_weights(fit_ridge(X_all, y_all, alpha=alpha, nonneg=True))
    res.weights_all = dict(zip(feature_order, (float(x) for x in w_all)))

    for hold in names:
        tr = [n for n in names if n != hold]
        Xtr = np.vstack([prepared[n][0] for n in tr])
        ytr = np.concatenate([prepared[n][1] for n in tr])
        w = normalize_weights(fit_ridge(Xtr, ytr, alpha=alpha, nonneg=True))
        Xh, yh = prepared[hold]
        res.folds.append(Fold(
            held_out=hold,
            weights=dict(zip(feature_order, (float(x) for x in w))),
            rho_fit=spearman(fuse(Xh, w), yh),
            rho_init=spearman(fuse(Xh, w_init), yh),
        ))

    fit = np.array([f.rho_fit for f in res.folds], dtype=float)
    ini = np.array([f.rho_init for f in res.folds], dtype=float)
    ok = np.isfinite(fit) & np.isfinite(ini)
    if ok.any():
        res.rho_fit_mean = float(fit[ok].mean())
        res.rho_init_mean = float(ini[ok].mean())
        res.rho_fit_median = float(np.median(fit[ok]))
        res.rho_init_median = float(np.median(ini[ok]))
        res.n_wins = int(np.sum(fit[ok] > ini[ok]))
    return res
