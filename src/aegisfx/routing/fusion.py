"""E4 的证据权重融合。

E4 不能只把两个专家的最终数字混一混——那样解释性就没了，
RAG 变成两个黑箱的加权平均，页面上说不出「为什么是这个方向」。

所以先各自把交易日证据权重归一化，再按门控融合：

    w_d^E4 = (1 − g)·w_d^E1 + g·w_d^E2

同一交易日在两个专家里都出现时合并权重，并保留四个量：
`source_from_e1` / `source_from_e2` / `weight_from_e1` / `weight_from_e2`，
再加上合并后的 `final_e4_weight` 和 `all_event_ids`。

融合后**重新归一化**，再拿去算方向、冲突指标和解释文本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Sequence

_EPS = 1e-12


@dataclass
class FusedEvidence:
    trading_day: date
    weight_from_e1: float
    weight_from_e2: float
    final_e4_weight: float
    source_from_e1: bool
    source_from_e2: bool
    all_event_ids: List[str] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)

    @property
    def group_size(self) -> int:
        return len(self.all_event_ids)


def _normalise(weights: Dict[date, float]) -> Dict[date, float]:
    total = sum(weights.values())
    if total <= _EPS:
        return {day: 0.0 for day in weights}
    return {day: value / total for day, value in weights.items()}


def fuse_evidence(
    evidence_e1: Sequence,
    evidence_e2: Sequence,
    gate: float,
) -> List[FusedEvidence]:
    """按门控 g 融合两个专家的交易日证据。

    `evidence_e1` / `evidence_e2` 是已经按交易日聚合过的 `DatedEvidence`。
    各自先归一化再融合，最后再归一化一次——不先归一化的话，
    某个专家证据条数多就会天然占优，门控就失去意义了。
    """
    gate = min(1.0, max(0.0, float(gate)))

    raw1 = {item.trading_day: float(item.weight) for item in evidence_e1}
    raw2 = {item.trading_day: float(item.weight) for item in evidence_e2}
    norm1, norm2 = _normalise(raw1), _normalise(raw2)

    lookup1 = {item.trading_day: item for item in evidence_e1}
    lookup2 = {item.trading_day: item for item in evidence_e2}
    days = sorted(set(norm1) | set(norm2))

    fused: List[FusedEvidence] = []
    for day in days:
        w1 = (1.0 - gate) * norm1.get(day, 0.0)
        w2 = gate * norm2.get(day, 0.0)
        item1, item2 = lookup1.get(day), lookup2.get(day)
        fused.append(FusedEvidence(
            trading_day=day,
            weight_from_e1=w1,
            weight_from_e2=w2,
            final_e4_weight=w1 + w2,
            source_from_e1=item1 is not None,
            source_from_e2=item2 is not None,
            all_event_ids=(list(getattr(item1, "event_ids", []) or [])
                           + list(getattr(item2, "event_ids", []) or [])),
            texts=(list(getattr(item1, "texts", []) or [])
                   + list(getattr(item2, "texts", []) or [])),
        ))

    total = sum(item.final_e4_weight for item in fused)
    if total > _EPS:
        for item in fused:
            item.final_e4_weight /= total
    for item in fused:
        item.weight_from_e1 /= total if total > _EPS else 1.0
        item.weight_from_e2 /= total if total > _EPS else 1.0

    fused.sort(key=lambda item: (-item.final_e4_weight,
                                 -item.trading_day.toordinal()))
    return fused


def weights_sum(fused: Sequence[FusedEvidence]) -> float:
    return float(sum(item.final_e4_weight for item in fused))
