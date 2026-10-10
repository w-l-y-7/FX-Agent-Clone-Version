"""异常收益：把「事件带来的额外汇率影响」从美元大趋势里剥出来。

## 模型

在与**事件窗口相同的 h 日尺度**上估：

    R_h(t) = α_h + β₁·X_h(t) + β₂·ΔS_h(t) + ε

    R_h(t)   = USD/CNY 从 t 到 t+h 的收益
    X_h(t)   = 美元指数同期收益
    ΔS_h(t)  = 中美 1 年期利差同期变化

    AR_h = R_h(事件) − (α̂_h + β̂₁·X_h + β̂₂·ΔS_h)

## 为什么是「h 日尺度」而不是逐日估完再累加

两种口径都试过，实测差别很大：

| 口径 | 20 日 R² | 20 日残差标准差 |
|---|---:|---:|
| 逐日估、逐日算残差 | 0.056 | **1.54%**（比原始收益的 1.39% 还大） |
| **h 日尺度（本模块）** | **0.42** | **1.15%**（比原始降 19.8%） |

逐日口径失败的原因不是模型写错了，而是**人民币日频本来就不跟美元指数走**——
中间价管理把大部分美元波动吸收掉了，日频相关性只有 0.236。在这么低的信噪比上
估出来的 β̂ 本身噪声很大，再拿它去预测 h 天，误差反而被放大。
拉到 20 日尺度，相关性升到 0.59，β̂ 才稳得住。

所以方案 §4.2 写的那个逐日式子，在本数据上**不成立**，这里按实测改成 h 日尺度。
20 日 R² 用美元指数是 0.34，加上利差变化是 0.42——利差在日频上贡献几乎为零
（0.0556 → 0.0560），但在 20 日尺度上有实质贡献，所以保留。

## 估计窗口取多长

默认 **250 个交易日**（约一年），不是方案初稿的 120。原因是 20 日尺度上
120 日窗口里只有约 6 个有效独立观测，β̂ 太不稳，噪声会盖过它减掉的东西。
实测见 `DEFAULT_ESTIMATION_WINDOW` 上面的表。

**1 日尺度上这个模型几乎不起作用**（残差标准差只降 0.9%）。这不是调参能解决的：
人民币日频本来就不跟美元指数走（相关性 0.236）。所以 1 日窗口的结果在报告里
要单独说明，别和 5 日、20 日混着讲。

## 两个必须说明的估计细节

估计窗口里的 h 日收益是**重叠**的（相邻两天共享 h−1 天），残差自相关，
OLS 的 β̂ 无偏但标准误不对。这里只把 β̂ 当控制变量用、不对它做统计推断，
所以可以接受，但要记着。

估计窗口整体**右移 h 天**，保证窗口内每个 h 日收益都在事件日之前结束——
不移的话，窗口末尾那些收益会延伸到事件之后，等于用未来信息估 β̂。

窗口内其他事件日会被剔除：本语料 445 条事件占掉约 23% 的交易日。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# 估计窗口取 250 个交易日（约一年），**不是方案初稿写的 120**。
#
# 20 日尺度上，120 日窗口里只有约 6 个有效独立观测（120/20），β̂ 的跨事件标准差
# 高达 0.22——均值才 0.37。这个抽样误差乘上事件窗口的美元波动之后，
# 会变成比模型减掉的还多的噪声。实测三种窗口下「异常收益跨事件标准差相对原始收益的变化」：
#
#     窗口      1 日      5 日      20 日
#     120      −0.8%    −5.7%    **+7.8%**（反而变大）
#     250      −0.9%   −11.0%    **−10.0%**
#
# 250 在三个尺度上都更好，再往上（500/750）收益趋平而 β̂ 会开始漂。
DEFAULT_ESTIMATION_WINDOW = 250
MIN_ESTIMATION_OBSERVATIONS = 40
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)

MARKET_MODEL = "market_model"
MEAN_ADJUSTED = "mean_adjusted"
UNAVAILABLE = "unavailable"

_FACTOR_NAMES = ("dxy", "spread")


@dataclass
class HorizonModel:
    """某个观察期上的市场模型。"""

    horizon: int
    alpha: float
    betas: Dict[str, float]
    r_squared: float
    residual_std: float
    target_std: float
    n_observations: int
    n_excluded_event_days: int

    @property
    def variance_reduction(self) -> float:
        """残差标准差相对原始收益标准差降了多少。**这才是「模型有没有用」的判据。**"""
        if self.target_std <= 0:
            return 0.0
        return 1.0 - self.residual_std / self.target_std


@dataclass
class AbnormalReturn:
    horizon: int
    abret: Optional[float]
    raw_ret: Optional[float]
    method: str
    model: Optional[HorizonModel] = None
    notes: str = ""

    @property
    def predicted_ret(self) -> Optional[float]:
        """模型认为「没有这件事」时本该发生的收益。"""
        if self.abret is None or self.raw_ret is None:
            return None
        return self.raw_ret - self.abret


class MarketModelSet:
    """一次算好所有观察期的因子序列，供全部事件复用。

    事件库有 445 条 × 3 个观察期，逐条重算 h 日收益会白跑很多遍。
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        window: int = DEFAULT_ESTIMATION_WINDOW,
    ) -> None:
        self.frame = frame
        self.horizons = tuple(horizons)
        self.window = window

        self._target: Dict[int, pd.Series] = {}
        self._dxy: Dict[int, pd.Series] = {}
        self._spread: Dict[int, pd.Series] = {}

        for horizon in self.horizons:
            self._target[horizon] = frame["USDCNY"].shift(-horizon) / frame["USDCNY"] - 1
            self._dxy[horizon] = frame["USD_Index"].shift(-horizon) / frame["USD_Index"] - 1
            self._spread[horizon] = (
                frame["rate_spread_1y"].shift(-horizon) - frame["rate_spread_1y"]
            )

    # ---------------------------------------------------------------- 估计

    def model_for(
        self,
        position: int,
        horizon: int,
        event_positions: Sequence[int] = (),
    ) -> Optional[HorizonModel]:
        """估 `position` 处、观察期为 `horizon` 的市场模型。不够数据返回 None。"""
        # 窗口右移 horizon 天：保证窗口里每个 h 日收益都在事件日之前结束
        end = position - horizon
        start = max(0, end - self.window)
        if end - start < MIN_ESTIMATION_OBSERVATIONS:
            return None

        block = pd.DataFrame({
            "target": self._target[horizon].iloc[start:end],
            "dxy": self._dxy[horizon].iloc[start:end],
            "spread": self._spread[horizon].iloc[start:end],
        })

        excluded = 0
        if event_positions:
            index = block.index
            keep = ~index.isin(self.frame.index[list(event_positions)])
            excluded = int((~keep).sum())
            block = block.loc[keep]

        block = block.dropna()
        if len(block) < MIN_ESTIMATION_OBSERVATIONS:
            return None

        y = block["target"].to_numpy(dtype=float)
        design = np.column_stack([
            np.ones(len(block)),
            block["dxy"].to_numpy(dtype=float),
            block["spread"].to_numpy(dtype=float),
        ])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coefficients

        denominator = float((y - y.mean()).dot(y - y.mean()))
        return HorizonModel(
            horizon=horizon,
            alpha=float(coefficients[0]),
            betas={"dxy": float(coefficients[1]), "spread": float(coefficients[2])},
            r_squared=(1.0 - float(residuals.dot(residuals)) / denominator
                      if denominator > 0 else 0.0),
            residual_std=float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0,
            target_std=float(y.std(ddof=1)) if len(y) > 1 else 0.0,
            n_observations=len(block),
            n_excluded_event_days=excluded,
        )

    # ---------------------------------------------------------------- 计算

    def abnormal_for(
        self,
        position: int,
        event_positions: Sequence[int] = (),
    ) -> Dict[int, AbnormalReturn]:
        results: Dict[int, AbnormalReturn] = {}
        for horizon in self.horizons:
            results[horizon] = self._one(position, horizon, event_positions)
        return results

    def _one(self, position: int, horizon: int,
             event_positions: Sequence[int]) -> AbnormalReturn:
        raw = self._raw(position, horizon)
        model = self.model_for(position, horizon, event_positions)

        if model is not None:
            factors = self._factor_row(position, horizon)
            if factors is not None and raw is not None:
                dxy, spread = factors
                predicted = (model.alpha + model.betas["dxy"] * dxy
                             + model.betas["spread"] * spread)
                return AbnormalReturn(horizon, float(raw - predicted), raw,
                                      MARKET_MODEL, model=model)
            return AbnormalReturn(horizon, None, raw, UNAVAILABLE, model=model,
                                  notes="事件窗口的因子有缺失，市场模型算不出来")

        abret, note = self._mean_adjusted(position, horizon, raw)
        return AbnormalReturn(horizon, abret, raw, MEAN_ADJUSTED,
                              notes=note or "估计窗口有效观测不足，已降级为均值调整")

    def _raw(self, position: int, horizon: int) -> Optional[float]:
        if position + horizon >= len(self.frame):
            return None
        value = self._target[horizon].iloc[position]
        return None if pd.isna(value) else float(value)

    def _factor_row(self, position: int, horizon: int) -> Optional[tuple]:
        dxy = self._dxy[horizon].iloc[position]
        spread = self._spread[horizon].iloc[position]
        if pd.isna(dxy) or pd.isna(spread):
            return None
        return float(dxy), float(spread)

    def _mean_adjusted(self, position: int, horizon: int,
                       raw: Optional[float]) -> tuple[Optional[float], str]:
        if raw is None:
            return None, "事件窗口超出数据范围"

        end = position - horizon
        start = max(0, end - self.window)
        history = self._target[horizon].iloc[start:end].dropna()
        if history.empty:
            return None, "估计窗口里没有可用的同 h 日收益"
        return float(raw - history.mean()), ""


# ---------------------------------------------------------------------------
# 便捷入口（少量事件时用；批量建库请直接用 MarketModelSet）
# ---------------------------------------------------------------------------


def estimate_market_model(
    frame: pd.DataFrame,
    event_position: int,
    *,
    event_positions: Sequence[int] = (),
    horizon: int = 20,
    window: int = DEFAULT_ESTIMATION_WINDOW,
) -> Optional[HorizonModel]:
    return MarketModelSet(frame, horizons=(horizon,), window=window).model_for(
        event_position, horizon, event_positions)


def event_abnormal_returns(
    frame: pd.DataFrame,
    event_date: pd.Timestamp,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    event_dates: Sequence[pd.Timestamp] = (),
    window: int = DEFAULT_ESTIMATION_WINDOW,
) -> Dict[int, AbnormalReturn]:
    stamp = pd.Timestamp(event_date)
    position = int(frame.index.searchsorted(stamp, side="left"))
    if position >= len(frame) or frame.index[position] != stamp:
        return {h: AbnormalReturn(h, None, None, UNAVAILABLE,
                                  notes="事件日不在交易日索引里") for h in horizons}

    event_positions = [int(frame.index.searchsorted(pd.Timestamp(d), side="left"))
                       for d in event_dates]
    models = MarketModelSet(frame, horizons=horizons, window=window)
    return models.abnormal_for(position, event_positions)
