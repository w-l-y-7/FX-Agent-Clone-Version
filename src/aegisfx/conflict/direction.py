"""方向判定：一定要给出方向，但每一步回退都留痕。

回退链（按顺序，前一步给不出方向才走下一步）：

1. 按交易日聚合后的**加权异常收益**的符号；
2. 若严格为 0，用**非中性证据的加权多数**方向；
3. 若仍平局，用查询日之前**同事件族**历史异常收益均值的符号；
4. 若仍为 0，用查询日之前**全部事件**历史异常收益均值的符号；
5. 若仍为 0，用固定默认方向，并置 `tie_break_used=True`。

「一定要给方向」不是逞强：一个带说明的弱结论，比一个空白的输出有用得多。
但每一层回退都要写进 `tie_break_reason`，让读的人知道这个方向有多勉强。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ..schemas.research import Direction
from .metrics import weighted_mean

# 全样本里 USD/CNY 的多数方向是上涨（人民币贬值），拿它当最后的兜底。
# 走到这一步说明手上什么信息都没有，这个默认值只是为了让输出不空着。
DEFAULT_DIRECTION = Direction.UP


def _sign_direction(value: Optional[float], tolerance: float = 1e-12) -> Optional[Direction]:
    if value is None:
        return None
    if value > tolerance:
        return Direction.UP
    if value < -tolerance:
        return Direction.DOWN
    return None


def decide_direction(
    dated_values: Sequence[float],
    dated_weights: Sequence[float],
    *,
    w_up: float,
    w_down: float,
    scale: float,
    family_history_mean: Optional[float] = None,
    all_history_mean: Optional[float] = None,
) -> Tuple[Direction, Optional[float], Optional[float], bool, str]:
    """返回 (方向, 加权异常收益, 方向分, 是否用了回退, 回退原因)。"""
    pairs = [(v, w) for v, w in zip(dated_values, dated_weights) if v is not None]
    values = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    weighted_abret = weighted_mean(values, weights) if values else None

    # 方向分用「多少个稳健尺度」表示，无量纲，方便跨事件比较
    direction_score = (None if weighted_abret is None or scale <= 0
                       else float(weighted_abret / scale))

    step1 = _sign_direction(weighted_abret)
    if step1 is not None:
        return step1, weighted_abret, direction_score, False, "加权异常收益的符号"

    step2 = None
    if w_up > w_down:
        step2 = Direction.UP
    elif w_down > w_up:
        step2 = Direction.DOWN
    if step2 is not None:
        return (step2, weighted_abret, direction_score, True,
                "加权异常收益为 0，改用非中性证据的加权多数")

    step3 = _sign_direction(family_history_mean)
    if step3 is not None:
        return (step3, weighted_abret, direction_score, True,
                "非中性证据上下权重相等，改用查询日前同事件族历史均值的方向")

    step4 = _sign_direction(all_history_mean)
    if step4 is not None:
        return (step4, weighted_abret, direction_score, True,
                "同事件族历史均值也为 0，改用查询日前全部事件历史均值的方向")

    return (DEFAULT_DIRECTION, weighted_abret, direction_score, True,
            "所有依据都为 0，使用固定默认方向")


def pool_history_means(
    pool_abrets: Sequence[Optional[float]],
    pool_families: Sequence[str],
    query_family: str,
) -> Tuple[Optional[float], Optional[float]]:
    """查询日之前的历史均值：(同族均值, 全样本均值)。"""
    all_values = [v for v in pool_abrets if v is not None]
    family_values = [v for v, family in zip(pool_abrets, pool_families)
                     if v is not None and family == query_family]
    return (
        float(np.mean(family_values)) if family_values else None,
        float(np.mean(all_values)) if all_values else None,
    )
