"""冲突度量：按交易日聚合、中性区间、双冲突指标、有效样本量。

对外主入口是 `engine.analyse`——给它一条查询和检索回来的证据，
它产出一份 `ResearchOutput`（方向 + 冲突诊断 + 执行许可）。

三层刻意分开：

- `aggregate` —— 把事件级证据折叠成交易日级，同一天只算一次市场反应；
- `metrics`   —— 纯函数，中性尺度、方向分歧度、中性占比、稳健离散度、ESS；
- `direction` —— 方向回退链，每一步都留痕。

`metrics` 里全是纯函数、不依赖任何外部状态，所以单测可以直接喂数进去。
"""

from .aggregate import (
    SameDayInconsistency,
    aggregate_by_day,
    aggregation_summary,
    realized_abret,
    trading_day,
)
from .direction import decide_direction, pool_history_means
from .engine import DEFAULT_HORIZON, analyse
from .metrics import (
    MIN_DAYS_FOR_ROBUST_DISPERSION,
    classify,
    directional_conflict,
    effective_sample_size,
    neutral_mass,
    neutral_threshold,
    pairwise_dispersion,
    robust_dispersion,
    robust_scale,
    summarise,
    weighted_mean,
    weighted_median,
    weighted_std,
)

__all__ = [
    "DEFAULT_HORIZON",
    "MIN_DAYS_FOR_ROBUST_DISPERSION",
    "SameDayInconsistency",
    "aggregate_by_day",
    "aggregation_summary",
    "analyse",
    "classify",
    "decide_direction",
    "directional_conflict",
    "effective_sample_size",
    "neutral_mass",
    "neutral_threshold",
    "pairwise_dispersion",
    "pool_history_means",
    "realized_abret",
    "robust_dispersion",
    "robust_scale",
    "summarise",
    "trading_day",
    "weighted_mean",
    "weighted_median",
    "weighted_std",
]
