"""检索质量指标：Precision@K、Recall@K、NDCG@K。

相关性是**分级**的（0–3，见 `docs/AegisFX-Agent方案.md` 的标注规范），
所以 NDCG 用分级增益，Precision / Recall 用「≥ 阈值算相关」的二值口径。

**Recall 只在「标注池」内算。** 每个查询有 280 多条候选，全标不现实，
所以沿用 TREC 的做法：候选池是各系统 Top-K 的并集，只标池内的。
这意味着池外的相关事件被当成不相关，**Recall 是下界**，报告里要写清楚。
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

# 二值口径下「算相关」的最低等级
DEFAULT_RELEVANT_THRESHOLD = 2


def precision_at_k(
    ranked_ids: Sequence[str],
    relevance: Dict[str, int],
    k: int,
    *,
    threshold: int = DEFAULT_RELEVANT_THRESHOLD,
) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for event_id in top if relevance.get(event_id, 0) >= threshold)
    # 分母用 k 而不是 len(top)：返回不足 k 条本身就是检索的缺陷，
    # 不该因为「没返回够」而让精确率虚高
    return hits / k


def recall_at_k(
    ranked_ids: Sequence[str],
    relevance: Dict[str, int],
    k: int,
    *,
    threshold: int = DEFAULT_RELEVANT_THRESHOLD,
) -> float:
    total_relevant = sum(1 for grade in relevance.values() if grade >= threshold)
    if total_relevant == 0:
        return 0.0
    hits = sum(1 for event_id in ranked_ids[:k]
               if relevance.get(event_id, 0) >= threshold)
    return hits / total_relevant


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Dict[str, int],
    k: int,
) -> float:
    """分级 NDCG。增益用 2^grade − 1，折损用 1/log2(rank+1)。"""
    if k <= 0:
        return 0.0

    def dcg(grades):
        return sum((2 ** grade - 1) / math.log2(position + 2)
                   for position, grade in enumerate(grades[:k]))

    actual = dcg([relevance.get(event_id, 0) for event_id in ranked_ids])
    ideal = dcg(sorted(relevance.values(), reverse=True))
    return actual / ideal if ideal > 0 else 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
