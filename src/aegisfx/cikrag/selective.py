"""选择性信任审计：统计机器何时决策，何时主动让渡给人或研究流程。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .schema import FeatureDecision


@dataclass(frozen=True)
class SelectiveAudit:
    total: int
    automatic: int
    accepted: int
    rejected: int
    abstained: int
    human_review: int
    research_only: int
    insufficient_data: int
    machine_coverage: float
    acceptance_rate: float
    abstention_rate: float
    human_review_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def audit_decisions(decisions: Mapping[str, FeatureDecision]) -> SelectiveAudit:
    """把状态转成论文可报告的 coverage/abstention/human burden。"""
    values = list(decisions.values())
    total = len(values)
    automatic = sum(value in (FeatureDecision.AUTO_ACCEPT,
                              FeatureDecision.AUTO_REJECT) for value in values)
    accepted = sum(value is FeatureDecision.AUTO_ACCEPT for value in values)
    rejected = sum(value is FeatureDecision.AUTO_REJECT for value in values)
    human = sum(value is FeatureDecision.HUMAN_REVIEW for value in values)
    research = sum(value is FeatureDecision.RESEARCH_ONLY for value in values)
    insufficient = sum(value is FeatureDecision.INSUFFICIENT_DATA for value in values)
    abstained = total - automatic
    denominator = max(total, 1)
    return SelectiveAudit(
        total=total,
        automatic=automatic,
        accepted=accepted,
        rejected=rejected,
        abstained=abstained,
        human_review=human,
        research_only=research,
        insufficient_data=insufficient,
        machine_coverage=automatic / denominator,
        acceptance_rate=accepted / denominator,
        abstention_rate=abstained / denominator,
        human_review_rate=human / denominator,
    )
