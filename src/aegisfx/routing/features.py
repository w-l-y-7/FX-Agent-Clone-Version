"""E4 的路由特征。

## 铁律

**只用查询时已经可知的信息。** 所有 z-score 和分位数只在查询日之前的数据上算，
绝不碰查询事件的未来收益、未来方向、offline outcome utility、未来误差，
也绝不碰 2024 最终测试的结果。

`assert_no_future_features()` 在构造特征前会拒掉带未来字段的输入对象
（复用第 4 阶段那套属性探测）。

## 维度

17 维 + 截距。方案里建议「约 15 维以内」，这里超了两维，多的两维是
两个专家的冲突诊断差值（`ci_gap` / `pd_gap`）——它们是 E4「该信谁」这个
问题最直接的证据，删掉反而失去意义。**没有用任何相关性强的冗余特征**，
裁剪过程见 `prune_correlated`。

被主动删掉的：
- `neutral_mass_e2`、`ess_e2` 各自留一份会和 `_e1` 那一份高度相关；
  融合成 `ci_gap` / `pd_gap` 之后信息保留、维度减半；
- `dispersion_e1/e2` 与 `pairwise_dispersion` 高度相关，只留成对离散度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas.views import RetrievalEventView, assert_retrieval_safe

# 沿用第 4 阶段的成熟协议，不另起一套
DEFAULT_LOOKBACK = 250

FAMILY_FLAGS = ("entity_restrictions", "dialogue", "tariff", "rhetoric")

FEATURE_NAMES: tuple[str, ...] = (
    # 事件属性
    "is_entity_restrictions", "is_dialogue", "is_tariff", "is_rhetoric",
    "is_us_initiated", "is_negative",
    # 语义检索状态
    "same_family_ratio_top5", "mean_semantic_top5",
    "semantic_gap_top1_top5", "log1p_mature_count",
    # 查询日前市场状态（全部 as-of）
    "rate_spread_z", "usd_index_z", "usd_index_20d_ret",
    "fx_vol_percentile", "fx_reserve_z",
    # 两个专家的冲突诊断差
    "ci_gap", "pd_gap",
)

_EPS = 1e-12


@dataclass
class FeatureBuilder:
    """把一条查询变成一行特征。

    `frame` 是 `market.paper_market()` 的输出，用来取 as-of 的市场状态。
    所有统计量只看 `query_position` 之前 `lookback` 个交易日。
    """

    frame: pd.DataFrame
    lookback: int = DEFAULT_LOOKBACK

    def _window(self, position: int) -> pd.DataFrame:
        start = max(0, position - self.lookback)
        return self.frame.iloc[start:position]

    def asof_z(self, column: str, position: int) -> float:
        window = self._window(position)[column].dropna()
        if len(window) < 20:
            return 0.0
        std = float(window.std())
        if std <= _EPS:
            return 0.0
        return float((self.frame[column].iloc[position - 1] - window.mean()) / std)

    def asof_percentile(self, column: str, position: int) -> float:
        window = self._window(position)[column].dropna()
        if len(window) < 20:
            return 0.5
        value = self.frame[column].iloc[position - 1]
        if pd.isna(value):
            return 0.5
        return float((window < value).mean())

    def asof_return(self, column: str, position: int, days: int = 20) -> float:
        if position - days - 1 < 0:
            return 0.0
        start = self.frame[column].iloc[position - days - 1]
        end = self.frame[column].iloc[position - 1]
        if pd.isna(start) or pd.isna(end) or start == 0:
            return 0.0
        return float(end / start - 1.0)

    def build(
        self,
        query: RetrievalEventView,
        *,
        same_family_ratio: Optional[float],
        semantic_top: Sequence[float],
        mature_count: int,
        conflict_e1: Dict[str, Optional[float]],
        conflict_e2: Dict[str, Optional[float]],
    ) -> Dict[str, float]:
        position = int(self.frame.index.searchsorted(pd.Timestamp(query.date),
                                                     side="left"))
        position = min(max(position, 1), len(self.frame) - 1)

        top = list(semantic_top)[:5]
        mean_top = float(np.mean(top)) if top else 0.0
        gap = float(top[0] - top[4]) if len(top) >= 5 else 0.0

        row = {
            **{f"is_{flag}": float(query.event_family == flag)
               for flag in FAMILY_FLAGS},
            "is_us_initiated": float(query.initiating_side == "US"),
            "is_negative": float(query.sentiment == "negative"),
            "same_family_ratio_top5": float(same_family_ratio or 0.0),
            "mean_semantic_top5": mean_top,
            "semantic_gap_top1_top5": gap,
            "log1p_mature_count": float(np.log1p(max(mature_count, 0))),
            "rate_spread_z": self.asof_z("rate_spread_1y", position),
            "usd_index_z": self.asof_z("USD_Index", position),
            "usd_index_20d_ret": self.asof_return("USD_Index", position, 20),
            "fx_vol_percentile": self.asof_percentile("fx_vol_20d", position),
            "fx_reserve_z": self.asof_z("fx_reserve_chg_3m", position),
            "ci_gap": _difference(conflict_e1.get("directional_conflict"),
                                  conflict_e2.get("directional_conflict")),
            "pd_gap": _difference(conflict_e1.get("pairwise_dispersion_norm"),
                                  conflict_e2.get("pairwise_dispersion_norm")),
        }
        return {name: row[name] for name in FEATURE_NAMES}


def _difference(a: Optional[float], b: Optional[float]) -> float:
    """两个专家的诊断量之差。任一缺失时按 0 处理（表示「没有差别信息」）。"""
    if a is None or b is None:
        return 0.0
    return float(a - b)


def build_feature_row(
    builder: FeatureBuilder,
    query: RetrievalEventView,
    **kwargs,
) -> Dict[str, float]:
    assert_retrieval_safe(query)
    return builder.build(query, **kwargs)


def prune_correlated(
    matrix: np.ndarray,
    names: Sequence[str],
    *,
    threshold: float = 0.9,
) -> List[str]:
    """删掉高度相关的特征，**只用训练期的数据算**。

    按相关系数绝对值从大到小成对剔除，保留先出现的那一个。
    返回保留下来的特征名。
    """
    if matrix.size == 0:
        return list(names)
    keep = list(range(len(names)))
    correlation = np.corrcoef(matrix[:, keep], rowvar=False)
    correlation = np.nan_to_num(correlation)

    dropped = set()
    for i in range(len(names)):
        if i in dropped:
            continue
        for j in range(i + 1, len(names)):
            if j in dropped:
                continue
            if abs(correlation[i, j]) > threshold:
                dropped.add(j)
    return [name for index, name in enumerate(names) if index not in dropped]
