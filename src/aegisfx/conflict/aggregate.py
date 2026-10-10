"""按交易日聚合证据。

事件文本、检索、解释都仍然以**事件**为单位；但算预测、冲突、证据数时必须
按**交易日**聚合——同一天的多条事件共享同一个已实现收益，各自计入就等于
同一次市场反应被数了两遍。本语料 437 条事件里有 11 个交易日是多条，
不处理的话冲突度量会被这几个日子带偏。

聚合规则（方案约定）：

- 该交易日的权重取候选中的**最大**最终分数 `w_d = max S_i`；
- 该交易日只保留**一个**异常收益；
- 全部 `event_id` 与文本保留下来，解释页面要按事件展示。

**同一天、同一预测期限的异常收益必须一致。** 不一致说明数据出了问题，
直接抛，不静默取平均——取平均会把数据错误掩盖成一个看起来合理的数。
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..schemas.research import DatedEvidence

# 浮点比较容差。异常收益是小数，1e-9 足够容纳序列化往返的误差。
_DEFAULT_TOLERANCE = 1e-9


class SameDayInconsistency(RuntimeError):
    """同一交易日的非空异常收益互相矛盾。"""


def trading_day(item) -> date:
    stamp = pd.Timestamp(item.date)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.date()


def realized_abret(item, horizon: int) -> Optional[float]:
    """取某个事件在给定观察期上**已经实现**的异常收益。"""
    realized = getattr(item, "realized_abret", None)
    if isinstance(realized, dict):
        return realized.get(horizon)
    value = getattr(item, f"abret_{horizon}d", None)
    return None if value is None else float(value)


def aggregate_by_day(
    evidence: Sequence,
    horizon: int,
    *,
    tolerance: float = _DEFAULT_TOLERANCE,
) -> List[DatedEvidence]:
    """把检索回来的证据按交易日折叠。`evidence` 里每项要有 `.event` 和 `.score`。"""
    groups: Dict[date, List] = {}
    for item in evidence:
        groups.setdefault(trading_day(item.event), []).append(item)

    dated: List[DatedEvidence] = []
    for day, items in groups.items():
        weights = [float(getattr(item, "score", 0.0)) for item in items]
        values = [(item.event.event_id, realized_abret(item.event, horizon))
                  for item in items]
        present = [(event_id, value) for event_id, value in values if value is not None]

        if present:
            reference = present[0][1]
            offenders = [(event_id, value) for event_id, value in present
                         if abs(value - reference) > tolerance]
            if offenders:
                detail = "、".join(f"{eid}={value:+.6f}" for eid, value in present[:4])
                raise SameDayInconsistency(
                    f"交易日 {day} 上同一观察期（{horizon} 日）的异常收益不一致：{detail}。"
                    "同一天的事件共享同一个已实现收益，出现分歧说明数据有问题——"
                    "这里不做平均，直接抛。"
                )
            abret = float(reference)
        else:
            abret = None

        dated.append(DatedEvidence(
            trading_day=day,
            weight=max(weights) if weights else 0.0,
            abret=abret,
            event_ids=[item.event.event_id for item in items],
            texts=[getattr(item.event, "text", "") for item in items],
        ))

    # 权重大的排前面；权重相同按日期倒序（近的优先），保证顺序稳定
    dated.sort(key=lambda group: (-group.weight, -group.trading_day.toordinal()))
    return dated


def aggregation_summary(dated: Sequence[DatedEvidence]) -> dict:
    raw_events = sum(group.group_size for group in dated)
    multi = [group for group in dated if group.group_size > 1]
    return {
        "raw_event_count": raw_events,
        "effective_evidence_days": len(dated),
        "same_day_group_count": len(multi),
        "same_day_groups": [
            {"trading_day": group.trading_day.isoformat(),
             "event_ids": group.event_ids}
            for group in multi
        ],
        "missing_abret_days": sum(1 for group in dated if group.abret is None),
    }
