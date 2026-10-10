"""冲突度量：中性区间、方向分歧度、中性占比、稳健离散度、有效样本量。

## 为什么要先划中性区间

把 ±0.02% 的正负波动当成「方向冲突」是没有意义的——那是噪声，不是分歧。
所以先估一个**查询日之前**的稳健尺度，用它划出中性带：

    scale_h(q) = 1.4826 × MAD({abret_{j,h} : date_j < date_q})
    τ_h = c × scale_h(q)，首版 c = 0.25

`1.4826 × MAD` 是正态下对标准差的稳健估计（MAD 抗离群值，标准差被一两个
极端事件就能带跑）。c = 0.25 是先验固定值，**不根据评测结果调优**；
第 6 周做敏感性分析时比较 c ∈ {0, 0.10, 0.25, 0.50}。

## 三类权重与三个指标

    W_up / W_down / W_neutral   —— 按方向分类后的权重和

    CI = 1 − |W_up − W_down| / (W_up + W_down + ε)      方向分歧度

    NeutralMass = W_neutral / (W_up + W_down + W_neutral + ε)

    D_robust = weighted_median(|r_d − m_w|, w_d)         稳健离散度
    m_w      = weighted_median(r_d, w_d)
    D_norm   = D_robust / (scale + ε)

    ESS = (Σw)² / (Σw² + ε)                              有效样本量

**没有非中性证据时 CI 返回 None，不是 0。** 返回 0 会被读成「证据完全一致」，
那是把「不知道」说成了「很确定」——两种情况的处置完全相反。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..schemas.research import (
    NEUTRAL_BAND_C,
    Direction,
    InsufficientDirectionalEvidence,
)

_EPS = 1e-12
_MAD_TO_SIGMA = 1.4826


def robust_scale(values: Sequence[float]) -> float:
    """1.4826 × MAD。空序列或全同值返回 0（调用方要能处理 scale=0）。"""
    clean = np.asarray([v for v in values if v is not None], dtype=float)
    if clean.size == 0:
        return 0.0
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    return _MAD_TO_SIGMA * mad


def neutral_threshold(scale: float, c: float = NEUTRAL_BAND_C) -> float:
    return c * scale


def classify(value: Optional[float], tau: float) -> Optional[Direction]:
    """按中性带分类。缺失值返回 None——**不能当成 0**，0 会被分到 neutral。"""
    if value is None:
        return None
    if value > tau:
        return Direction.UP
    if value < -tau:
        return Direction.DOWN
    return Direction.NEUTRAL


def directional_conflict(w_up: float, w_down: float) -> Optional[float]:
    """方向分歧度。没有非中性证据时返回 None。"""
    total = w_up + w_down
    if total <= _EPS:
        return None
    return float(1.0 - abs(w_up - w_down) / total)


def neutral_mass(w_up: float, w_down: float, w_neutral: float) -> float:
    total = w_up + w_down + w_neutral
    if total <= _EPS:
        return 0.0
    return float(w_neutral / total)


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return None
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_weights = values[order], weights[order]
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= _EPS:
        return float(np.median(ordered_values))
    position = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    return float(ordered_values[min(position, ordered_values.size - 1)])


# 稳健离散度在少于这个证据日数时会退化，见下面的说明
MIN_DAYS_FOR_ROBUST_DISPERSION = 3


def robust_dispersion(
    values: Sequence[float],
    weights: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """返回 (加权中位数, 稳健离散度 D_robust = 加权中位数绝对偏差)。

    ## 两个必须知道的局限

    **一、K = 2 时这个量必然为 0。** 两个点时加权中位数等于其中一个点，
    它到自己的绝对偏差是 0，而绝对偏差的中位数正好落在这个 0 上。
    这是公式性质不是 bug。所以 `summarise` 会带 `dispersion_defined` 标志出来，
    `engine` 在它退化时加阻止原因——把「算不出来」当成「证据集中」是危险的。

    **二、双峰分布下它会系统性偏小。** 而「方向冲突高」的情形恰恰就是双峰
    （一半证据在正、一半在负）。此时大部分点离中位数都不远，MAD 就小；
    真正大的是**幅度**，那要靠加权标准差才看得出来。

    实测：两条方向完全对半、幅度一个在 ±0.3% 一个在 ±9% 的证据组，
    CI 都是 1.0，稳健离散度是 0.0001 对 0.002（20 倍），
    而加权标准差是 0.003 对 0.086（28 倍）——都分得开，但绝对值都偏小。

    所以本模块两个都算：稳健的当主指标（按方案约定），加权标准差的当辅助诊断。
    单看稳健离散度会低估冲突的严重程度。
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return None, None
    center = weighted_median(values, weights)
    if center is None:
        return None, None
    deviation = np.abs(values - center)
    return center, weighted_median(deviation, weights)


def pairwise_dispersion(
    values: Sequence[float],
    weights: Sequence[float],
) -> Optional[float]:
    """加权成对离散度 PD。

        PD = Σ_{d<j} w_d·w_j·|r_d − r_j| / Σ_{d<j} w_d·w_j

    ## 为什么要有它

    加权 MAD 在**双峰**分布下会退化——而「方向冲突高」的情形天然是双峰的。
    实测两组 CI 都是 1.0 的证据：

    | 证据 | MAD 型稳健离散度 | 加权标准差 | 成对离散度 |
    |---|---:|---:|---:|
    | ±0.3% | 0.0001 | 0.003 | 0.006 |
    | ±9% | 0.002 | 0.086 | 0.170 |

    成对离散度不依赖「离中位数多远」，而是**所有点对的平均距离**，
    所以双峰下不会塌。n 很小（约 10 条证据日），O(n²) 完全够用。

    MAD 保留下来兼容旧结果，但冲突特征主集合换成它。
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size < 2 or weights.size != values.size:
        return None

    diff = np.abs(values[:, None] - values[None, :])
    outer = np.outer(weights, weights)
    upper = np.triu_indices(values.size, k=1)
    denominator = float(outer[upper].sum())
    if denominator <= _EPS:
        return None
    return float((outer[upper] * diff[upper]).sum() / denominator)


def effective_sample_size(weights: Sequence[float]) -> float:
    """ESS = (Σw)² / Σw²。

    权重均匀时接近条数；一条权重垄断时接近 1。**必须基于聚合后的交易日**，
    不然同一天的多条事件会把样本量虚报上去。
    """
    weights = np.asarray([w for w in weights], dtype=float)
    if weights.size == 0:
        return 0.0
    total = float(weights.sum())
    squares = float((weights ** 2).sum())
    if squares <= _EPS:
        return 0.0
    return float(total * total / squares)


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if values.size == 0 or total <= _EPS:
        return None
    return float((values * weights).sum() / total)


def weighted_std(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    """辅助诊断量。主报告用稳健离散度，它只是留着对照。"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if values.size < 2 or total <= _EPS:
        return None
    mean = weighted_mean(values, weights)
    if mean is None:
        return None
    variance = float((weights * (values - mean) ** 2).sum() / total)
    return float(np.sqrt(max(variance, 0.0)))


def summarise(
    dated_values: Sequence[float],
    dated_weights: Sequence[float],
    *,
    scale: float,
    c: float = NEUTRAL_BAND_C,
) -> dict:
    """把一条查询的证据汇总成指标字典。不构造对象，方便单测直接看数。"""
    tau = neutral_threshold(scale, c)
    classes = [classify(v, tau) for v in dated_values]

    triples = list(zip(dated_values, dated_weights, classes))
    w_up = sum(w for _, w, cls in triples if cls is Direction.UP)
    w_down = sum(w for _, w, cls in triples if cls is Direction.DOWN)
    w_neutral = sum(w for _, w, cls in triples if cls is Direction.NEUTRAL)

    conflict = directional_conflict(w_up, w_down)
    clean_values = [v for v in dated_values if v is not None]
    clean_weights = [w for v, w in zip(dated_values, dated_weights) if v is not None]

    center, dispersion = robust_dispersion(clean_values, clean_weights)
    pairwise = pairwise_dispersion(clean_values, clean_weights)
    total_weight = w_up + w_down + w_neutral

    usable_days = len(clean_values)
    return {
        "scale": scale,
        "tau": tau,
        "neutral_band_c": c,
        "usable_days": usable_days,
        "dispersion_defined": usable_days >= MIN_DAYS_FOR_ROBUST_DISPERSION,
        "w_up": w_up,
        "w_down": w_down,
        "w_neutral": w_neutral,
        "directional_conflict": conflict,
        "directional_conflict_status": (
            "ok" if conflict is not None else InsufficientDirectionalEvidence.STATUS.value
        ),
        "neutral_mass": neutral_mass(w_up, w_down, w_neutral),
        "weighted_center": center,
        "magnitude_dispersion": dispersion,
        # scale 为 0（全同值或样本太少）时归一化没有意义，返回 None 而不是
        # 除一个 ε 得到一个 1e10 量级的假数——那种数字会在界面上看起来"极其离散"
        "normalized_dispersion": (
            None if dispersion is None or scale <= _EPS
            else dispersion / (scale + _EPS)
        ),
        # 冲突特征的主指标，双峰下不会塌
        "pairwise_dispersion": pairwise,
        "pairwise_dispersion_norm": (
            None if pairwise is None or scale <= _EPS
            else pairwise / (scale + _EPS)
        ),
        "weighted_std": weighted_std(clean_values, clean_weights),
        "effective_sample_size": effective_sample_size(clean_weights),
        "total_weight": total_weight,
    }
