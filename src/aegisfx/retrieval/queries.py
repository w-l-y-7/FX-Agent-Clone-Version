"""检索评测的 20 条查询。

## 为什么是 20 条、为什么这么分

这 20 条是 RQ1 的黄金标注对象，人工要逐条判相关性，所以数量要压得住
（20 × 最多 20 个候选 ≈ 400 次判断，一个人一两个小时能做完）。

按事件族配额分配，依据是第 2 周统计的池/评测分布：

| 事件族 | 检索池 | 评测集 | 配额 |
|---|---:|---:|---:|
| 企业限制 | 98 | 93 | 8 |
| 双边对话 | 72 | 41 | 5 |
| 关税 | 68 | 10 | 4 |
| 表态指责 | 21 | 8 | 2 |
| 调查诉讼 | 16 | 4 | 1 |

关税那一族的**池比评测多得多**（68 vs 10），是检索最有发挥空间的一类；
企业限制和对话两族占了大头，配额自然多。

每条族内按日期**等距抽取**，保证 2023 一季度到 2024 四季度都有覆盖——
这样 as-of 池的大小从 281 条逐步长到 437 条，截断逻辑真的被考到。

## 查询集会落盘

选出来就写进 `data/aegisfx/retrieval/queries.json`。**不能每次重算**：
人工标注是绑在 event_id 上的，如果查询集悄悄变了，标注就对不上了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from ..paths import DATA_DIR
from ..schemas.historical import EVAL, HistoricalEvent

QUERY_DIR = DATA_DIR / "retrieval"
QUERY_PATH = QUERY_DIR / "queries.json"

# 各族配额，合计 20
QUERY_ALLOCATION: Dict[str, int] = {
    "entity_restrictions": 8,
    "dialogue": 5,
    "tariff": 4,
    "rhetoric": 2,
    "litigation": 1,
}

# 候选池里同族事件少于这个数就不选它——检索不到东西的查询测不出什么
MIN_SAME_FAMILY_IN_POOL = 10


@dataclass
class QuerySpec:
    event_id: str
    date: str
    family: str
    event_type: str
    initiating_side: str
    text: str
    pool_size: int

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "date": self.date,
            "family": self.family,
            "event_type": self.event_type,
            "initiating_side": self.initiating_side,
            "text": self.text,
            "pool_size": self.pool_size,
        }


def _as_day(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def _evenly_spaced(items: Sequence, count: int) -> List:
    """从已排序的序列里等距抽取 count 个，两端都取到。"""
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


def build_query_set(
    events: Sequence[HistoricalEvent],
    *,
    allocation: Optional[Dict[str, int]] = None,
) -> List[QuerySpec]:
    """按配额和时间分布选出查询集。**确定性**，不带随机。"""
    allocation = allocation or QUERY_ALLOCATION
    by_family: Dict[str, List[HistoricalEvent]] = {}
    for event in events:
        if event.split != EVAL:
            continue
        if event.market_state_complete:
            by_family.setdefault(event.event_family, []).append(event)

    for bucket in by_family.values():
        bucket.sort(key=lambda item: _as_day(item.date))

    queries: List[QuerySpec] = []
    for family, quota in allocation.items():
        candidates = by_family.get(family, [])
        if not candidates:
            continue
        # 池里同族事件太少的，先把这一类挪到最后（如果有富余的才补）
        usable = [
            event for event in candidates
            if _same_family_before(events, event) >= MIN_SAME_FAMILY_IN_POOL
        ] or candidates
        for event in _evenly_spaced(usable, quota):
            queries.append(QuerySpec(
                event_id=event.event_id,
                date=_as_day(event.date).date().isoformat(),
                family=event.event_family,
                event_type=event.event_type,
                initiating_side=event.initiating_side,
                text=event.text,
                pool_size=_same_family_before(events, event),
            ))

    queries.sort(key=lambda item: item.date)
    return queries


def _same_family_before(events: Sequence[HistoricalEvent], query: HistoricalEvent) -> int:
    """查询日之前、同族的事件数——用来判断这条查询检索不检索得到东西。"""
    cutoff = _as_day(query.date)
    return sum(1 for event in events
               if event.event_id != query.event_id
               and event.event_family == query.event_family
               and _as_day(event.date) < cutoff)


def save_query_set(queries: Sequence[QuerySpec], path: Path = QUERY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([query.to_dict() for query in queries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_query_set(path: Path = QUERY_PATH) -> List[QuerySpec]:
    if not path.exists():
        return []
    return [QuerySpec(**item) for item in json.loads(path.read_text(encoding="utf-8"))]


def load_or_build(
    events: Sequence[HistoricalEvent],
    *,
    rebuild: bool = False,
    path: Path = QUERY_PATH,
) -> List[QuerySpec]:
    """已经有落盘的查询集就直接读——**重新生成会让人工标注失配**。"""
    if not rebuild:
        existing = load_query_set(path)
        if existing:
            return existing
    queries = build_query_set(events)
    save_query_set(queries, path)
    return queries


QUERY_SET: List[QuerySpec] = []
