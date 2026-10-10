"""冲突感知证据聚合 + 双通道特征判断。

## 冲突与无知必须分开

`conflict_score` 高 = **有足够可靠证据、但它们互相矛盾**。
`ignorance_score` 高 = **证据根本不够**。

这两件事的处置完全不同：冲突要交给人核验，无知要去找更多证据。
把「没证据」报成「没冲突」（或反过来）会让整个 DA 的决策逻辑失准。

所以聚合输出两个独立的数，不合成一个。

## 熵的归一化口径

    C_f = −Σ_d p_d log(p_d) / log|D|

`|D|` 取**实际出现质量的方向类别数**，下限 2。这样：
- 所有可靠证据同向 → |D|=2、熵=0 → C=0
- 两个方向对半 → |D|=2、熵=log2 → C=1
- 三个方向都有 → |D|=3，均分时 → C=1

固定成 3 的话，两个方向对半只能得到 0.63，「方向相反」就到不了高分——
和方案要求的「证据充分且方向相反时高 conflict」不符。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence

from .reliability import ReliabilityBreakdown, evaluate as evaluate_reliability
from .schema import (
    Claim,
    ClaimRelation,
    ClaimStatus,
    EffectDirection,
    FeatureDecision,
    FeatureKnowledge,
)

_EPS = 1e-9

# 证据不足度的先验：相当于「一开始默认缺一单位证据」。
# 总质量到 1.0 时 U 降到 0.5，到 3.0 时降到 0.25。
IGNORANCE_PRIOR = 1.0

# 方向桶。neutral / nonlinear / conditional / unknown 都归到 neutral 桶，
# 它们不表达正负方向，不能参与「矛盾」的判定。
_SIGN_BUCKETS = {
    EffectDirection.POSITIVE: "up",
    EffectDirection.NEGATIVE: "down",
    EffectDirection.NEUTRAL: "neutral",
    EffectDirection.NONLINEAR: "neutral",
    EffectDirection.CONDITIONAL: "neutral",
    EffectDirection.UNKNOWN: "neutral",
}


def _normalised_entropy(masses: Dict[str, float]) -> Optional[float]:
    total = sum(masses.values())
    if total <= _EPS:
        return None
    probabilities = {key: value / total for key, value in masses.items()}
    n_classes = sum(1 for value in probabilities.values() if value > _EPS)
    if n_classes <= 1:
        return 0.0
    # 只在 p > 0 的项上求和，所以 log 里不必再加 ε——
    # 加了会让「两边正反对半」算不到 1（实测差 3e-9），
    # 而「高冲突到达 1」是本阶段一条明确的判据，不该被浮点平滑吃掉
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > _EPS)
    return float(entropy / math.log(max(n_classes, 2)))


def _contextual_signed_conflict(
    assessments: Sequence["ClaimAssessment"],
) -> Optional[float]:
    """只在相同适用条件内计算正反方向冲突。

    旧实现把 neutral/conditional 也放进熵，并把不同制度下的相反结论混在一起。
    这会把“结论有条件”误报成“知识互相矛盾”。现在先按 comparable_key 分组，
    每组只比较 positive/negative，再按该组的可靠证据质量加权平均。
    """
    groups: Dict[tuple, Dict[str, float]] = {}
    for item in assessments:
        direction = item.claim.effect_direction
        if direction not in (EffectDirection.POSITIVE, EffectDirection.NEGATIVE):
            continue
        bucket = "up" if direction is EffectDirection.POSITIVE else "down"
        masses = groups.setdefault(item.claim.comparable_key, {"up": 0.0, "down": 0.0})
        masses[bucket] += item.mass

    weighted, total = 0.0, 0.0
    for masses in groups.values():
        group_mass = sum(masses.values())
        entropy = _normalised_entropy(masses)
        if entropy is not None and group_mass > _EPS:
            weighted += entropy * group_mass
            total += group_mass
    return float(weighted / total) if total > _EPS else None


@dataclass
class ClaimAssessment:
    """一条观点在某个预测环境下的可用性。"""

    claim: Claim
    reliability: ReliabilityBreakdown
    applicability: float
    cluster_size: int

    @property
    def mass(self) -> float:
        return self.reliability.score * self.applicability


def assess_claims(
    claims: Sequence[Claim],
    *,
    origin: date,
    market_regime: str = "",
    policy_condition: str = "",
    cluster_sizes: Optional[Dict[str, int]] = None,
) -> List[ClaimAssessment]:
    """只评估**在该时点可用**的 active 观点。

    `published_at < origin` 是硬条件：预测时还没发表的论文不能拿来用。
    """
    cluster_sizes = cluster_sizes or {}
    out: List[ClaimAssessment] = []
    for claim in claims:
        if claim.status is not ClaimStatus.ACTIVE:
            continue
        if claim.published_at is not None and claim.published_at >= origin:
            continue
        breakdown = evaluate_reliability(
            claim, origin=origin,
            cluster_size=cluster_sizes.get(claim.claim_id, 1),
            market_regime=market_regime, policy_condition=policy_condition)
        out.append(ClaimAssessment(
            claim=claim, reliability=breakdown,
            applicability=(breakdown.context_similarity
                           * breakdown.temporal_applicability),
            cluster_size=cluster_sizes.get(claim.claim_id, 1)))
    return out


def aggregate_feature(
    feature_id: str,
    feature_name: str,
    assessments: Sequence[ClaimAssessment],
    *,
    market_regime: str = "",
    policy_condition: str = "",
    edges: Sequence = (),
) -> tuple:
    """把一个特征上的观点聚合成分方向质量 + 两个不确定度。

    返回 `(FeatureKnowledge, 真冲突边数)`。真冲突只数 `contradicts`，
    `conditionally_differs` 不计入。
    """
    masses: Dict[str, float] = {"up": 0.0, "down": 0.0, "neutral": 0.0}
    supporting, contradicting, conditional, superseded = [], [], [], []
    clusters = set()

    for item in assessments:
        bucket = _SIGN_BUCKETS[item.claim.effect_direction]
        masses[bucket] += item.mass
        if item.claim.duplicate_cluster_id:
            clusters.add(item.claim.duplicate_cluster_id)

        if item.claim.effect_direction is EffectDirection.CONDITIONAL:
            conditional.append(item.claim.claim_id)
        elif item.claim.effect_direction is EffectDirection.POSITIVE:
            supporting.append(item.claim.claim_id)
        elif item.claim.effect_direction is EffectDirection.NEGATIVE:
            contradicting.append(item.claim.claim_id)
        if item.claim.status is ClaimStatus.SUPERSEDED:
            superseded.append(item.claim.claim_id)

    total_mass = sum(masses.values())
    conflict = _contextual_signed_conflict(assessments)
    ignorance = IGNORANCE_PRIOR / (IGNORANCE_PRIOR + total_mass)

    dominant = EffectDirection.UNKNOWN
    if masses["up"] > masses["down"] and masses["up"] > masses["neutral"]:
        dominant = EffectDirection.POSITIVE
    elif masses["down"] > masses["up"] and masses["down"] > masses["neutral"]:
        dominant = EffectDirection.NEGATIVE
    elif masses["neutral"] > 0 and masses["neutral"] >= max(masses["up"], masses["down"]):
        dominant = EffectDirection.NEUTRAL

    # 知识层真正的冲突只看 contradicts 边；conditionally_differs 不算
    real_conflicts = [edge for edge in edges
                      if getattr(edge, "relation", None) is ClaimRelation.CONTRADICTS]

    applicability = (sum(item.applicability * item.mass for item in assessments)
                     / total_mass) if total_mass > _EPS else 0.0

    return FeatureKnowledge(
        feature_id=feature_id, feature_name=feature_name,
        market_regime=market_regime, policy_condition=policy_condition,
        support_mass=masses["up"], opposition_mass=masses["down"],
        neutral_mass=masses["neutral"],
        conflict_score=conflict, ignorance_score=float(ignorance),
        dominant_direction=dominant,
        supporting_claim_ids=supporting,
        contradicting_claim_ids=contradicting,
        conditional_claim_ids=conditional,
        superseded_claim_ids=superseded,
        duplicate_clusters=sorted(clusters),
        n_active_claims=len(assessments),
        context_applicability=float(applicability),
    ), len(real_conflicts)


# ---------------------------------------------------------------------------
# 第二条通道：样本外预测效用
# ---------------------------------------------------------------------------


@dataclass
class PredictiveUtility:
    """固定 FA 上做 purged walk-forward 消融得到的增益。"""

    feature_id: str
    gain_mean: Optional[float] = None
    gain_median: Optional[float] = None
    gain_by_fold: List[float] = field(default_factory=list)
    gain_low: Optional[float] = None
    gain_high: Optional[float] = None
    sign_consistency: Optional[float] = None
    n_folds: int = 0

    @property
    def is_positive(self) -> bool:
        return self.gain_mean is not None and self.gain_mean > 0

    @property
    def is_stable_positive(self) -> bool:
        return (self.is_positive and self.sign_consistency is not None
                and self.sign_consistency >= 2 / 3 and self.n_folds >= 2)

    @property
    def is_stable_negative(self) -> bool:
        return (self.gain_mean is not None and self.gain_mean < 0
                and self.sign_consistency is not None
                and self.sign_consistency <= 1 / 3)

    @property
    def is_significantly_positive(self) -> bool:
        """95% 分块 bootstrap 增益区间整体大于 0。"""
        return self.gain_low is not None and self.gain_low > 0

    @property
    def is_significantly_negative(self) -> bool:
        """95% 分块 bootstrap 增益区间整体小于 0。"""
        return self.gain_high is not None and self.gain_high < 0


# ---------------------------------------------------------------------------
# 双通道决策
# ---------------------------------------------------------------------------

# 这些阈值**只能在训练期与 2023 验证集上定**，不许用 2024。
CONFLICT_HIGH = 0.5
IGNORANCE_HIGH = 0.5
APPLICABILITY_LOW = 0.5


def decide(
    knowledge, real_conflict_count: int, utility: PredictiveUtility,
) -> tuple:
    """双通道决策。返回 (状态, 理由)。

    **两条通道各说各的，不合成一个总分。**
    「读起来合理」和「真的提高预测」是两件事，混在一起就没法解释
    「为什么这条被拒了」。
    """
    reasons: List[str] = []

    # 证据被新知识明确否定
    if knowledge.superseded_claim_ids and not knowledge.supporting_claim_ids:
        return FeatureDecision.AUTO_REJECT, "证据已被更新知识明确否定，且没有仍然成立的支持证据"

    if knowledge.n_active_claims == 0 and utility.n_folds == 0:
        return FeatureDecision.INSUFFICIENT_DATA, "既没有证据也没有可用的样本外结果"

    if utility.n_folds < 2:
        if knowledge.n_active_claims > 0:
            return FeatureDecision.INSUFFICIENT_DATA, (
                f"知识侧有 {knowledge.n_active_claims} 条证据，"
                f"但样本外折数只有 {utility.n_folds}（不足 2），无法判断预测效用")
        return FeatureDecision.INSUFFICIENT_DATA, "证据与样本外结果都不足"

    conflict = knowledge.conflict_score if knowledge.conflict_score is not None else 0.0
    ignorance = knowledge.ignorance_score
    applicability = knowledge.context_applicability

    # 知识侧完全没有证据时，理由里要写清楚——不然读的人会以为
    # 拒绝是基于知识判断的，实际上那一条通道本来就没东西
    no_knowledge = knowledge.n_active_claims == 0
    if no_knowledge:
        reasons.append("知识库中没有该特征的观点，本次判断只依据样本外消融")

    # ---- AUTO_REJECT：样本外稳定为负 ----
    if utility.is_significantly_negative:
        reasons.append(
            f"样本外增益的 95% 分块区间整体为负"
            f"（{utility.gain_low:+.4f}, {utility.gain_high:+.4f}）")
        return FeatureDecision.AUTO_REJECT, "；".join(reasons)

    # ---- HUMAN_REVIEW：高影响的分歧 ----
    if utility.is_positive and conflict > CONFLICT_HIGH:
        return FeatureDecision.HUMAN_REVIEW, (
            f"样本外增益为正（{utility.gain_mean:+.4f}），"
            f"但可靠证据方向冲突（conflict={conflict:.2f} > {CONFLICT_HIGH}），"
            f"机器无法自行裁定")
    if real_conflict_count > 0 and conflict > CONFLICT_HIGH:
        return FeatureDecision.HUMAN_REVIEW, (
            f"知识图里有 {real_conflict_count} 条 contradicts 边，"
            "属于真冲突，需要人工确认适用条件")
    if conflict > CONFLICT_HIGH and applicability < APPLICABILITY_LOW:
        return FeatureDecision.HUMAN_REVIEW, (
            "证据方向冲突且适用性低，条件差异与真矛盾的边界需要人工判断")

    # ---- AUTO_ACCEPT ----
    if (utility.is_significantly_positive and utility.is_stable_positive
            and conflict <= CONFLICT_HIGH
            and ignorance <= IGNORANCE_HIGH and applicability >= APPLICABILITY_LOW):
        return FeatureDecision.AUTO_ACCEPT, (
            f"样本外增益稳定为正（均值 {utility.gain_mean:+.4f}，"
            f"95% 区间 [{utility.gain_low:+.4f}, {utility.gain_high:+.4f}]，"
            f"跨折一致率 {utility.sign_consistency:.0%}），"
            f"知识冲突 {conflict:.2f}、不足度 {ignorance:.2f}、适用性 {applicability:.2f} 均在阈值内")

    # ---- RESEARCH_ONLY ----
    # 注意要把 reasons 带上——知识侧没证据那条提示就在这里，丢了的话
    # 读的人会以为拒绝是基于知识判断的
    if not utility.is_positive:
        reasons.append(f"样本外增益不足（均值 {utility.gain_mean:+.4f}），不进入正式 FA")
        return FeatureDecision.RESEARCH_ONLY, "；".join(reasons)
    if ignorance > IGNORANCE_HIGH:
        reasons.append(f"知识不足度偏高（{ignorance:.2f}），证据基础不够扎实")
        return FeatureDecision.RESEARCH_ONLY, "；".join(reasons)

    if (utility.gain_low is not None and utility.gain_high is not None
            and utility.gain_low <= 0 <= utility.gain_high):
        reasons.append(
            f"样本外增益区间跨 0（{utility.gain_low:+.4f}, "
            f"{utility.gain_high:+.4f}），机器应保留判断")
        return FeatureDecision.RESEARCH_ONLY, "；".join(reasons)

    reasons.append("不满足 AUTO_ACCEPT 的任何一条硬条件")
    return FeatureDecision.RESEARCH_ONLY, "；".join(reasons)
