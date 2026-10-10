"""收益标签的成熟时间协议（第 5 阶段 P0）。

## 问题

原来的 as-of 条件只有 `candidate_date < query_date`，对 h 日收益**不够**。

举例（h = 20）：

    历史事件发生在 1 月 1 日
    当前查询发生在 1 月 10 日
    那条历史事件的 20 日异常收益要到 1 月底才完全实现

1 月 10 日的查询如果用了它完整的 20 日收益，用的就是**查询时还不存在的信息**。
冲突度量、方向预测、共形校准全都吃这个标签，所以这是一条真实的泄漏，
不是理论洁癖。

## 规则

对事件 $i$、期限 $h$：

    label_start_date_h    = 事件日 t_i
    label_end_date_h      = t_i 之后第 h 个交易日
    label_available_at_h  = label_end_date_h 的**下一个交易日**

最后一行是关键：**收盘之后才知道收盘价**。系统只有日期粒度，
所以第 h 个交易日当天，它的收益还没法用；要到下一个交易日才算成熟。

候选必须同时满足：

    event_date_i < query_date_q
    label_available_at_{i,h} < query_date_q

换算成交易日下标就是 `pos_i + h + 1 <= pos_q`，等价于 `pos_i + h < pos_q`。

## 影响面

2023–2024 那批评测查询的检索池是 2017–2022，全部早已成熟，所以
**E1/E2/E3 在那批查询上的排序一个字都不会变**。

真正会咬人的是**伪查询**：2018–2022 的事件互为候选，其中大量落在
彼此 20 个交易日之内。不修的话，伪查询训练集里会混进大量未来信息，
E4 的路由器就会学到一个在真实场景里根本不存在的规律。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence

import pandas as pd

DEFAULT_HORIZONS = (1, 5, 20)


class LabelMaturityViolation(RuntimeError):
    """候选的收益标签在决策时点还没成熟，或者候选本身越界。"""


def _as_day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


@dataclass(frozen=True)
class LabelMaturity:
    """一条事件在某个期限上的成熟时间。"""

    horizon: int
    event_date: date
    label_end_date: Optional[date]
    available_at: Optional[date]
    available: bool

    def describe(self) -> str:
        if not self.available:
            return (f"{self.event_date} + {self.horizon} 日："
                    "窗口超出数据末端，标签算不出来")
        return (f"{self.event_date} → 收益窗口结束 {self.label_end_date}，"
                f"可用时点 {self.available_at}")


class MaturityCalendar:
    """按交易日历计算标签成熟时点。整个项目只有这一份规则。"""

    def __init__(self, trading_days: Sequence) -> None:
        index = pd.DatetimeIndex([_as_day(day) for day in trading_days])
        self.trading_days = index.sort_values()
        self._position = {stamp: i for i, stamp in enumerate(self.trading_days)}

    @classmethod
    def from_market(cls) -> "MaturityCalendar":
        from .market import paper_market

        return cls(paper_market().index)

    def position(self, day) -> Optional[int]:
        stamp = _as_day(day)
        if stamp in self._position:
            return self._position[stamp]
        # 落在非交易日：按「之后最近的交易日」，与事件入库时的顺延口径一致
        found = int(self.trading_days.searchsorted(stamp, side="left"))
        return found if found < len(self.trading_days) else None

    def describe(self, event_date, horizon: int) -> LabelMaturity:
        position = self.position(event_date)
        if position is None:
            return LabelMaturity(horizon, _as_day(event_date).date(), None, None, False)

        end_position = position + horizon
        available_position = end_position + 1
        if end_position >= len(self.trading_days):
            return LabelMaturity(horizon, self.trading_days[position].date(),
                                 None, None, False)
        return LabelMaturity(
            horizon=horizon,
            event_date=self.trading_days[position].date(),
            label_end_date=self.trading_days[end_position].date(),
            available_at=(self.trading_days[available_position].date()
                          if available_position < len(self.trading_days) else None),
            available=available_position < len(self.trading_days),
        )

    def is_mature_before(self, event_date, decision_date, horizon: int) -> bool:
        """事件的 h 日标签，在决策时点是否已经可用。"""
        maturity = self.describe(event_date, horizon)
        if not maturity.available:
            return False
        return maturity.available_at < _as_day(decision_date).date()

    # ------------------------------------------------------------------
    # 整数快路径
    #
    # 伪查询要跑上千次池构造，每次都走日期比较会慢一个量级。
    # 交易日下标是预计算的，成熟判断就退化成一次整数比较。
    # ------------------------------------------------------------------

    def available_index(self, event_date, horizon: int) -> Optional[int]:
        """标签可用的**交易日下标**。等价于 `pos + horizon + 1`。"""
        position = self.position(event_date)
        if position is None:
            return None
        available = position + horizon + 1
        return available if available < len(self.trading_days) else None

    def mature_limit(self, query_position: int, horizon: int) -> int:
        """候选下标必须**不超过**这个值才算成熟。

        推导：标签在 `available = i + horizon + 1` 这个下标上可用，
        成熟条件是 `available < query_position`，即 `i < query_position − horizon − 1`，
        取整就是 `i <= query_position − horizon − 2`。

        **这里必须是减 2 不是减 1。** 减 1 会让「标签恰好在查询日当天才可用」
        的候选漏进来——而当天收盘前，那个收益还不存在。
        第一版就是写成了减 1，被 `assert_labels_mature` 当场抓住
        （查询 2023-04-27 / 候选 2023-03-28 那条）。
        """
        return query_position - horizon - 2

    def mature_mask(self, positions: Sequence[int], query_position: int,
                    horizon: int) -> List[bool]:
        """给一批已知下标的候选算成熟掩码。"""
        limit = self.mature_limit(query_position, horizon)
        return [position is not None and position <= limit for position in positions]

    def must_be_mature_before(self, event_date, decision_date, horizon: int) -> None:
        if not self.is_mature_before(event_date, decision_date, horizon):
            raise LabelMaturityViolation(
                f"事件 {_as_day(event_date).date()} 的 {horizon} 日收益"
                f"在决策时点 {_as_day(decision_date).date()} 尚未成熟："
                f"{self.describe(event_date, horizon).describe()}。"
                "用它等于把未来信息喂进冲突度量和预测。"
            )


# ---------------------------------------------------------------------------
# 池构造与断言
# ---------------------------------------------------------------------------


def assert_labels_mature(
    query,
    candidates: Sequence,
    horizon: int,
    calendar: MaturityCalendar,
) -> None:
    """硬断言：候选必须既早于查询日，标签也已成熟。

    **越界就抛，不静默过滤。** 静默过滤会让「池子变小了」看起来像是正常的，
    而实际上是有人在某处漏了成熟规则。
    """
    decision = _as_day(query.date)
    offenders: List[str] = []

    for candidate in candidates:
        candidate_day = _as_day(candidate.date)
        if candidate_day >= decision:
            # 未来的或同日的
            kind = "未来" if candidate_day > decision else "同日"
            offenders.append(f"{candidate.event_id}@{candidate_day.date()}（{kind}）")
            continue
        if not calendar.is_mature_before(candidate.date, query.date, horizon):
            maturity = calendar.describe(candidate.date, horizon)
            offenders.append(
                f"{candidate.event_id}@{candidate_day.date()}"
                f"（标签 {maturity.available_at or '不可用'}，未成熟）")

    if offenders:
        preview = "、".join(offenders[:5])
        raise LabelMaturityViolation(
            f"查询 {query.event_id}（{decision.date()}，{horizon} 日）的候选里有 "
            f"{len(offenders)} 条不满足成熟协议：{preview}。"
            "候选必须同时满足「早于查询日」和「标签在查询日之前已经可用」。"
        )


def mature_pool(
    events: Sequence,
    query,
    horizon: int,
    calendar: MaturityCalendar,
    *,
    verify: bool = True,
) -> List:
    """满足成熟协议的候选池。

    走的是整数快路径（`mature_mask`），所以「早于查询日」和「标签已成熟」
    合并成了一个条件：候选下标 `<= query_position - horizon - 1`。

    `verify=False` 可以跳过构造后的逐个复查（伪查询批量跑时用，
    省下的时间是数量级的；单条查询请保持 True）。
    """
    query_position = calendar.position(query.date)
    if query_position is None:
        return []

    limit = calendar.mature_limit(query_position, horizon)
    pool = []
    for event in events:
        if event.event_id == query.event_id:
            continue
        position = calendar.position(event.date)
        if position is None or position > limit:
            continue
        pool.append(event)

    if verify:
        assert_labels_mature(query, pool, horizon, calendar)
    return pool


def maturity_summary(
    events: Sequence,
    query,
    horizon: int,
    calendar: MaturityCalendar,
) -> dict:
    """给伪查询审计表用：池子是怎么被筛出来的。"""
    decision = _as_day(query.date)
    before = [e for e in events
              if e.event_id != query.event_id and _as_day(e.date) < decision]
    mature = [e for e in before
              if calendar.is_mature_before(e.date, query.date, horizon)]
    return {
        "candidates_before_query": len(before),
        "candidates_mature": len(mature),
        "dropped_immature": len(before) - len(mature),
        "label_end_dates": {
            event.event_id: (calendar.describe(event.date, horizon).label_end_date
                             or "unavailable")
            for event in mature
        },
    }
