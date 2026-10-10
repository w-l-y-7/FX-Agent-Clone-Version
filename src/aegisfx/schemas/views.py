"""在线检索视图 vs 离线评测视图。

**这是本阶段最重要的一道防线。**

「离线结果有用性」这个指标要用到**查询事件发生之后的真实异常收益**。
它是有价值的评测标签，但只要它有一丝机会流进排序函数，整个检索评测就作废了——
而且不会报错，只会让指标变好看。

所以把两类信息拆成两个类型：

- `RetrievalEventView` —— 排序函数**唯一**能看到的形状，只有查询日之前可知的信息；
- `EvaluationEventView` —— 额外带查询事件的真实未来结果，只给离线评测用。

`assert_retrieval_safe()` 用属性探测而不是类型判断：子类、鸭子类型的对象、
甚至是临时拼的 dict 包装，只要带着 `future_abret` 这类字段就会被拦下。
类型判断挡不住子类，属性探测挡得住。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

# 一旦出现在查询对象上就说明未来信息泄漏了
FORBIDDEN_ON_QUERY = (
    "future_abret",
    "future_direction",
    "outcome_utility",
    "future_return",
    "realized_after_query",
)


class LeakageError(RuntimeError):
    """在线检索路径上出现了查询事件的未来信息。"""


def assert_retrieval_safe(query) -> None:
    """排序函数入口的自检。带未来字段的对象一律拒绝。

    两类拦截：

    1. 带 `future_abret` / `future_direction` 这类**评测专用**字段（子类、
       鸭子类型、临时包装都拦得住，类型判断拦不住子类）；
    2. 带**已实现收益**的查询视图——对候选事件这是历史信息，对查询事件
       就是它自己的未来结果。查询视图必须是「干净的」。
    """
    found = [name for name in FORBIDDEN_ON_QUERY
             if getattr(query, name, None) is not None]
    if found:
        raise LeakageError(
            f"查询对象 {getattr(query, 'event_id', '?')} 上带有未来信息字段 {found}。"
            "在线检索路径不允许接触查询事件之后的任何结果——"
            "这是评测作废最隐蔽的一种方式，必须直接抛。"
        )

    realized = getattr(query, "realized_abret", None)
    if realized:
        raise LeakageError(
            f"查询对象 {getattr(query, 'event_id', '?')} 的视图里带了已实现收益 "
            f"{sorted(realized)}。已实现收益对候选事件是历史信息，"
            "对查询事件就是它自己的未来结果——查询视图必须用 "
            "`from_event(event, include_realized=False)` 构造。"
        )


@dataclass(frozen=True)
class RetrievalEventView:
    """在线检索唯一能看到的形状。

    **不含**查询事件的未来异常收益、未来方向、结果有用性标签。
    """

    event_id: str
    date: datetime
    text: str
    event_type: str
    event_family: str
    initiating_side: str
    sentiment: str

    # 查询日之前就已经可知的市场状态
    rate_spread_1y: Optional[float] = None
    usd_index: Optional[float] = None
    fx_vol_20d: Optional[float] = None
    fx_reserve_chg_3m: Optional[float] = None

    # 候选事件**已经实现**的异常收益（按观察期）。
    # 候选都在查询日之前，所以这对在线检索是合法信息；当前排序不使用它。
    realized_abret: Dict[int, Optional[float]] = field(default_factory=dict)

    @classmethod
    def from_event(cls, event, horizons=(1, 5, 20), *,
                   include_realized: bool = True) -> "RetrievalEventView":
        """构造视图。

        **查询必须传 `include_realized=False`**：已实现收益对候选事件是历史信息，
        对查询事件就是它自己的未来结果。默认 True 是为了让候选那一侧的调用
        不用每个地方都写一遍。
        """
        return cls(
            event_id=event.event_id,
            date=event.date,
            text=event.text,
            event_type=event.event_type,
            event_family=event.event_family,
            initiating_side=event.initiating_side,
            sentiment=event.sentiment,
            rate_spread_1y=event.rate_spread_1y,
            usd_index=event.usd_index,
            fx_vol_20d=event.fx_vol_20d,
            fx_reserve_chg_3m=event.fx_reserve_chg_3m,
            realized_abret=(
                {h: getattr(event, f"abret_{h}d", None) for h in horizons}
                if include_realized else {}
            ),
        )


@dataclass(frozen=True)
class EvaluationEventView(RetrievalEventView):
    """离线评测专用：在检索视图之上，额外带上查询事件的真实未来结果。

    **绝不能传进 `retrieve()`**——`assert_retrieval_safe` 会抛。
    """

    future_abret: Dict[int, Optional[float]] = field(default_factory=dict)
    future_direction: Optional[str] = None
    outcome_utility: Optional[float] = None

    @classmethod
    def from_event(cls, event, horizons=(1, 5, 20)) -> "EvaluationEventView":
        base = RetrievalEventView.from_event(event, horizons, include_realized=True)
        return cls(
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            future_abret={h: getattr(event, f"abret_{h}d", None) for h in horizons},
            future_direction=(
                None if getattr(event, "abret_20d", None) is None
                else ("up" if event.abret_20d > 0 else "down")
            ),
        )
