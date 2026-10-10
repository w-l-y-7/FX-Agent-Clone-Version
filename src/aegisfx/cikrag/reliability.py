"""证据可靠性：**程序按可核查字段算，大模型不输出权重。**

方案 §6 明确要求「保留每个分量，不要立刻手工拼成任意总分」。
所以 `ReliabilityBreakdown` 把九个分量都留着，聚合值只是给下游一个数用，
随时可以拆开看是哪一项把分数压下去的。

## 分量的含义与取舍

| 分量 | 衡量的东西 | 关键取舍 |
|---|---|---|
| `source_authority` | 来源的权威性 | 按类型给，不按机构名气 |
| `evidence_type` | 证据形态 | 官方原文适合确认政策事实，同行评审适合确认一般关系 |
| `methodology` | 有没有可核查的方法 | 有样本期、有识别策略的加分 |
| `sample_size_available` | 样本量是否可查 | **只是「可查」，不是「够大」**——不知道就中性 |
| `publication_recency` | 新鲜度 | 有下限，不让老文献归零 |
| `temporal_applicability` | 有效期是否覆盖预测时点 | 制度变了就压低 |
| `context_similarity` | 市场制度/政策条件是否匹配 | 条件不同的相反结论靠它区分 |
| `independence` | 是否属于重复簇 | **转载不重复增加证据质量**，这一项管的就是它 |
| `human_verified` | 人工核验过 | 乘数式的加分 |

## 为什么用几何平均聚合

九项都是「置信因子」，任何一项接近 0 都应该把整体压下去——算术平均做不到这点
（转载十遍能把一条烂证据的均分拉高）。几何平均是乘法式的，
一条重复来源直接体现在 `independence = 1/簇大小` 上。

**这是个先验选择，不是拟合出来的。** 第 6 阶段不做权重学习，
因为冲突标注集还没建起来（见进度文档的说明）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Optional

from .schema import Claim, EvidenceType

_EPS = 1e-9

# 来源类型 → 权威性基线。**按证据形态给，不按机构名气给**——
# 一个大机构发的新闻稿不因为是它发的就更权威。
AUTHORITY_BY_TYPE: Dict[EvidenceType, float] = {
    EvidenceType.OFFICIAL_POLICY: 1.00,
    EvidenceType.PEER_REVIEWED: 0.95,
    EvidenceType.OFFICIAL_REPORT: 0.90,
    EvidenceType.WORKING_PAPER: 0.70,
    EvidenceType.INDUSTRY_RESEARCH: 0.65,
    EvidenceType.INTERNAL_NOTE: 0.50,
    EvidenceType.NEWS_REPOST: 0.30,
    EvidenceType.UNKNOWN: 0.40,
}

# 证据形态 → 适用场景。同行评审适合一般关系，官方原文适合政策事实。
METHODOLOGY_BY_TYPE: Dict[EvidenceType, float] = {
    EvidenceType.PEER_REVIEWED: 1.00,
    EvidenceType.OFFICIAL_REPORT: 0.85,
    EvidenceType.WORKING_PAPER: 0.80,
    EvidenceType.OFFICIAL_POLICY: 0.60,   # 政策原文不提供识别策略
    EvidenceType.INDUSTRY_RESEARCH: 0.55,
    EvidenceType.INTERNAL_NOTE: 0.40,
    EvidenceType.NEWS_REPOST: 0.20,
    EvidenceType.UNKNOWN: 0.40,
}

# 新鲜度的下限与半衰期。**更近的材料不必然更正确**，
# 所以这一项只影响「时效」，不影响权威性；而且设下限，不让老文献归零。
RECENCY_HALF_LIFE_YEARS = 6.0
RECENCY_FLOOR = 0.35


@dataclass
class ReliabilityBreakdown:
    """九个分量 + 聚合值。**分量必须能拆开看。**"""

    source_authority: float
    evidence_type: float
    methodology: float
    sample_size_available: float
    publication_recency: float
    temporal_applicability: float
    context_similarity: float
    independence: float
    human_verified: float

    def components(self) -> Dict[str, float]:
        return {
            "source_authority": self.source_authority,
            "evidence_type": self.evidence_type,
            "methodology": self.methodology,
            "sample_size_available": self.sample_size_available,
            "publication_recency": self.publication_recency,
            "temporal_applicability": self.temporal_applicability,
            "context_similarity": self.context_similarity,
            "independence": self.independence,
            "human_verified": self.human_verified,
        }

    @property
    def score(self) -> float:
        """几何平均。任一项接近 0 都会把整体压下去，这是刻意的。"""
        values = [max(v, _EPS) for v in self.components().values()]
        return float(math.exp(sum(math.log(v) for v in values) / len(values)))

    def weakest(self, n: int = 3):
        return sorted(self.components().items(), key=lambda kv: kv[1])[:n]


def _recency(published: Optional[date], origin: date) -> float:
    if published is None:
        return 0.60                      # 不知道就别惩罚
    years = max((origin - published).days / 365.25, 0.0)
    decayed = 0.5 ** (years / RECENCY_HALF_LIFE_YEARS)
    return float(RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * decayed)


def _temporal(claim: Claim, origin: date) -> float:
    """有效期是否覆盖预测时点。制度变了就压低。"""
    if claim.valid_from and origin < claim.valid_from:
        return 0.10                      # 还没生效的知识
    if claim.valid_to and origin > claim.valid_to:
        return 0.25                      # 已经失效
    return 1.0


def _context(claim: Claim, market_regime: str, policy_condition: str) -> float:
    """市场制度与政策条件是否匹配。

    **条件不同的相反结论靠这一项区分，而不是靠把它们算成冲突。**
    两边都不写条件时给中性 0.5——不知道，不奖励也不惩罚。
    """
    if not claim.market_regime and not claim.policy_condition:
        return 0.5
    if market_regime and claim.market_regime == market_regime:
        return 1.0
    if policy_condition and claim.policy_condition == policy_condition:
        return 0.9
    if claim.market_regime and market_regime and claim.market_regime != market_regime:
        return 0.3
    return 0.5


def _independence(cluster_size: int) -> float:
    """同一研究的多次转载只算一个证据簇。

    `cluster_size = 1` 时给 1.0；十篇转载给 0.1。
    **转载不重复增加证据质量**，这一项就是干这个的。
    """
    return 1.0 / max(cluster_size, 1)


def evaluate(
    claim: Claim,
    *,
    origin: date,
    cluster_size: int = 1,
    market_regime: str = "",
    policy_condition: str = "",
) -> ReliabilityBreakdown:
    """算一条观点的可靠性拆解。"""
    return ReliabilityBreakdown(
        source_authority=AUTHORITY_BY_TYPE.get(claim.evidence_type, 0.40),
        evidence_type=AUTHORITY_BY_TYPE.get(claim.evidence_type, 0.40),
        methodology=METHODOLOGY_BY_TYPE.get(claim.evidence_type, 0.40),
        # 「样本量可查」而不是「样本量够大」——不知道就给中性，不假装知道
        sample_size_available=(0.9 if claim.sample_size_available else 0.5),
        publication_recency=_recency(claim.published_at, origin),
        temporal_applicability=_temporal(claim, origin),
        context_similarity=_context(claim, market_regime, policy_condition),
        independence=_independence(cluster_size),
        human_verified=(1.0 if claim.human_verified else 0.7),
    )
