"""CIK-RAG 的数据结构。

## 冲突的对象换了

第 1–5 阶段把「历史事件对应收益一正一负」叫冲突。**本阶段改了**：
冲突的对象是**原子观点**——从学术论文、官方报告、政策文件、行业研究里抽出来的
一句可核查的论断，比如「美元指数上涨通常推动 USD/CNY 上涨」。

一条事件收益是正还是负，是**观测**；一篇文章说这个关系成立还是不成立，是**论断**。
两者不能混为一谈：观测的分歧可能只是时点不同，论断的分歧才是知识层面的冲突。

## 只有 contradicts 算真冲突

`conditionally_differs`（条件不同）和 `supersedes`（新的取代旧的）
**都不计入冲突**。把它们算成冲突，会把「研究做得细」惩罚成「结论不可靠」。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EffectDirection(str, Enum):
    POSITIVE = "positive"          # 特征上升 → 目标上升
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    NONLINEAR = "nonlinear"
    CONDITIONAL = "conditional"    # 只在特定制度/政策下成立
    UNKNOWN = "unknown"


class ClaimRelation(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONDITIONALLY_DIFFERS = "conditionally_differs"
    SUPERSEDES = "supersedes"
    DUPLICATES = "duplicates"
    UNRELATED = "unrelated"
    UNCERTAIN = "uncertain"

    @property
    def is_conflict(self) -> bool:
        """只有它算真冲突。"""
        return self is ClaimRelation.CONTRADICTS


class ClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    OUT_OF_SCOPE = "out_of_scope"
    INVALID = "invalid"


class EvidenceType(str, Enum):
    PEER_REVIEWED = "peer_reviewed"      # 同行评审论文
    OFFICIAL_POLICY = "official_policy"  # 官方政策原文
    OFFICIAL_REPORT = "official_report"  # 官方报告（央行、国际组织）
    INDUSTRY_RESEARCH = "industry_research"
    WORKING_PAPER = "working_paper"
    NEWS_REPOST = "news_repost"          # 转载
    INTERNAL_NOTE = "internal_note"      # 项目自撰说明
    UNKNOWN = "unknown"


class FeatureDecision(str, Enum):
    AUTO_ACCEPT = "AUTO_ACCEPT"
    AUTO_REJECT = "AUTO_REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class Claim(BaseModel):
    """一条原子观点。字段对齐方案 §5.1。"""

    claim_id: str
    feature_id: str
    feature_name: str
    claim_text: str

    effect_direction: EffectDirection = EffectDirection.UNKNOWN
    effect_magnitude: Optional[float] = None
    currency_pair: str = "USD/CNY"
    forecast_horizon: Optional[int] = None
    target_variable: str = ""
    applicable_country: str = ""
    applicable_period: str = ""
    market_regime: str = ""
    policy_condition: str = ""

    methodology: str = ""
    sample_period: str = ""
    sample_size_available: Optional[int] = None

    evidence_type: EvidenceType = EvidenceType.UNKNOWN
    source_title: str = ""
    source_url: Optional[str] = None
    source_authority: str = ""
    source_doc_id: str = ""

    published_at: Optional[date] = None
    valid_from: Optional[date] = None
    valid_to: Optional[date] = None

    supersedes_claim_id: Optional[str] = None
    duplicate_cluster_id: Optional[str] = None
    evidence_span: str = ""
    extraction_model: str = ""
    human_verified: bool = False

    status: ClaimStatus = ClaimStatus.ACTIVE

    @property
    def comparable_key(self) -> tuple:
        """「可比」的要求：币种、期限、市场制度、政策条件全对齐才算同类。

        这是判定 `contradicts` 的前置条件——不先对齐就比方向，
        等于把「不同条件下的不同结论」当成矛盾。
        """
        return (self.currency_pair, self.forecast_horizon,
                self.market_regime, self.policy_condition)


class ClaimEdge(BaseModel):
    """两条观点之间的关系。有向：source → target。"""

    source_claim_id: str
    target_claim_id: str
    relation: ClaimRelation
    confidence: float = 0.0
    decided_by: str = "rule"          # rule / llm / human
    reason: str = ""
    evidence_span: str = ""


class FeatureKnowledge(BaseModel):
    """一个特征上的知识聚合结果。"""

    feature_id: str
    feature_name: str
    market_regime: str = ""
    policy_condition: str = ""

    support_mass: float = 0.0
    opposition_mass: float = 0.0
    neutral_mass: float = 0.0

    conflict_score: Optional[float] = Field(
        default=None, description="归一化熵。证据全同向时为 0，两边对半时为 1")
    ignorance_score: float = Field(
        default=1.0, description="证据不足度。没有证据时接近 1")

    dominant_direction: EffectDirection = EffectDirection.UNKNOWN

    supporting_claim_ids: List[str] = Field(default_factory=list)
    contradicting_claim_ids: List[str] = Field(default_factory=list)
    conditional_claim_ids: List[str] = Field(default_factory=list)
    superseded_claim_ids: List[str] = Field(default_factory=list)
    duplicate_clusters: List[str] = Field(default_factory=list)

    n_active_claims: int = 0
    context_applicability: float = 0.0
