"""CP0：滚动共形区间。

    a_t   = |r_t − r̂_t|                      非一致性分数
    q_t   = 校准池里第 ⌈(n+1)(1−α)⌉ 小的 a    有限样本分位数（上取整）
    C_t   = [r̂_t − q_t, r̂_t + q_t]

## 两个容易做错的地方

**一、分位数必须用上取整位置，不能调默认的线性插值。**
共形的覆盖保证来自 `⌈(n+1)(1−α)⌉` 这个位置：
n 个校准点、目标覆盖 1−α 时，取第 ⌈(n+1)(1−α)⌉ 小的分数才有
`P(r ∈ C) ≥ 1−α`。用 `np.quantile(..., method="linear")` 算出来的是插值值，
覆盖会偏低。`conformal_quantile` 就是干这个的。

**二、校准池只能放标签已经成熟的残差。**
20 日预测要等 20 个交易日才知道误差。时刻 t 的校准池是
`residual_available_at < t`。同一天的多条查询用同一份校准状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

_EPS = 1e-12


@dataclass
class ConformalResult:
    target_coverage: float
    lower: Optional[float]
    upper: Optional[float]
    width: Optional[float]
    quantile: Optional[float]
    n_calibration: int
    covered: Optional[bool] = None
    realized: Optional[float] = None
    method: str = "CP0"


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """有限样本共形分位数：第 ⌈(n+1)(1−α)⌉ 小的分数。

    位置超出样本数时返回 `inf`——那表示「校准点不够，给不出有限区间」，
    而不是「区间宽度是最大值」。这两件事在评估里必须区分开。
    """
    values = np.sort(np.asarray([s for s in scores if s is not None], dtype=float))
    n = values.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    return float(values[rank - 1])


def rolling_conformal(
    predictions: Sequence[float],
    realized: Sequence[Optional[float]],
    available_positions: Sequence[int],
    *,
    targets: Sequence[float] = (0.8, 0.9),
    window: int = 60,
    gamma: Optional[float] = None,
) -> dict:
    """按时间顺序跑一遍滚动共形。

    `available_positions[i]` 是第 i 条预测的**真实结果可用的时间下标**。
    用它来保证校准池里只有 `< 当前时刻` 的残差。

    传了 `gamma` 就走 ACI 顺序更新，否则是固定 α 的 CP0。
    """
    n = len(predictions)
    outputs = {target: [] for target in targets}
    alphas = {target: 1.0 - target for target in targets}

    history: List[dict] = []
    for index in range(n):
        current = available_positions[index]
        # 只有「结果已经可用」的残差能进校准池
        pool = [item for item in history
                if item["available_at"] is not None and item["available_at"] < current]
        if window:
            pool = pool[-window:]
        scores = [item["score"] for item in pool]

        for target in targets:
            alpha = alphas[target]
            q = conformal_quantile(scores, alpha)
            point = predictions[index]
            if np.isinf(q):
                entry = ConformalResult(target, None, None, None, None, len(scores))
            else:
                lower, upper = point - q, point + q
                value = realized[index]
                covered = (None if value is None
                           else bool(lower <= value <= upper))
                entry = ConformalResult(target, float(lower), float(upper),
                                        float(2 * q), float(q), len(scores),
                                        covered=covered, realized=value)
            outputs[target].append(entry)

            if gamma is not None and entry.covered is not None:
                # ACI：只有真实结果成熟后才能更新 α
                alphas[target] = float(np.clip(
                    alpha + gamma * (alpha - (0.0 if entry.covered else 1.0)),
                    1e-4, 1.0 - 1e-4))

        value = realized[index]
        score = (None if value is None else abs(value - predictions[index]))
        history.append({"score": score, "available_at": current})

    return {
        "targets": {target: _summarise(outputs[target], target) for target in targets},
        "per_query": {target: outputs[target] for target in targets},
        "method": "CP0+ACI" if gamma is not None else "CP0",
        "gamma": gamma,
    }


def _summarise(entries: Sequence[ConformalResult], target: float) -> dict:
    widths = [e.width for e in entries if e.width is not None]
    covered = [e.covered for e in entries if e.covered is not None]
    return {
        "target_coverage": target,
        "n": len(entries),
        "n_with_interval": len(widths),
        "n_evaluable": len(covered),
        "coverage": float(np.mean(covered)) if covered else None,
        "mean_width": float(np.mean(widths)) if widths else None,
        "median_width": float(np.median(widths)) if widths else None,
        "interval_score": float(np.mean([
            _interval_score(e, target) for e in entries if e.width is not None
        ])) if widths else None,
    }


def _interval_score(entry: ConformalResult, target: float) -> float:
    """Winkler / interval score。越小越好：宽度加两侧未覆盖的惩罚。

        IS = 宽度 + (2/α)·(下界 − 实际)₊ + (2/α)·(实际 − 上界)₊
    """
    if entry.width is None or entry.realized is None:
        return float("nan")
    alpha = 1.0 - target
    value = entry.realized
    penalty = 0.0
    if value < entry.lower:
        penalty += (2.0 / alpha) * (entry.lower - value)
    elif value > entry.upper:
        penalty += (2.0 / alpha) * (value - entry.upper)
    return float(entry.width + penalty)
