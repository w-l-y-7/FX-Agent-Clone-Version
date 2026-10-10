"""观点之间的关系判定。

## 顺序很重要

判定必须**先对齐条件，再比方向**。反过来的话，「利差在资本管制下失效」
和「利差扩大推升汇率」会被当成矛盾——但它们说的是两件事，
一个在管制期成立、一个在自由流动期成立。

规则顺序：

1. 同一条观点的重复簇 → `duplicates`
2. 明确标注了替代关系 → `supersedes`
3. 不是同一个特征 → `unrelated`
4. 币种/期限/制度/政策条件有差异 →
   - 方向也不同 → `conditionally_differs`（**不是冲突**）
   - 方向相同 → `supports`
5. 条件齐平：
   - 方向相同 → `supports`
   - 方向相反 → `contradicts`（**只有这一种算真冲突**）
   - 有一侧是 unknown / nonlinear / conditional → `uncertain`

## 大模型在这里的位置

规则能定的就规则定，规则定不了的（自然语言层面看不出方向）才交给大模型，
且大模型只能从七个关系里选，不能自由发挥。`decided_by` 字段记录是谁判的。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .schema import Claim, ClaimEdge, ClaimRelation, EffectDirection

# 方向语义：+1 表示「特征上升 → 目标上升」，-1 相反，0 表示无方向
_DIRECTION_SIGN: Dict[EffectDirection, int] = {
    EffectDirection.POSITIVE: 1,
    EffectDirection.NEGATIVE: -1,
    EffectDirection.NEUTRAL: 0,
    EffectDirection.NONLINEAR: 0,
    EffectDirection.CONDITIONAL: 0,
    EffectDirection.UNKNOWN: 0,
}

# 这几个方向不足以判「矛盾」——信息本身就说不清是正是负
_UNDECIDABLE = {EffectDirection.UNKNOWN, EffectDirection.CONDITIONAL,
                EffectDirection.NONLINEAR}


def _comparable(a: Claim, b: Claim) -> bool:
    return a.comparable_key == b.comparable_key


def _conditions_differ(a: Claim, b: Claim) -> bool:
    return (a.currency_pair != b.currency_pair
            or a.forecast_horizon != b.forecast_horizon
            or a.market_regime != b.market_regime
            or a.policy_condition != b.policy_condition)


def judge(a: Claim, b: Claim, *, reason_extra: str = "") -> ClaimEdge:
    """判 a 与 b 的关系（有向：a → b）。"""
    if a.claim_id == b.claim_id:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.DUPLICATES, confidence=1.0,
                         reason="同一条观点")

    # 1. 重复簇
    if (a.duplicate_cluster_id and a.duplicate_cluster_id == b.duplicate_cluster_id):
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.DUPLICATES, confidence=1.0,
                         reason=f"同属重复簇 {a.duplicate_cluster_id}")

    # 2. 替代关系
    if a.supersedes_claim_id == b.claim_id:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.SUPERSEDES, confidence=1.0,
                         reason=f"{a.claim_id} 明确声明替代 {b.claim_id}")
    if b.supersedes_claim_id == a.claim_id:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.SUPERSEDES, confidence=1.0,
                         reason=f"{b.claim_id} 明确声明替代 {a.claim_id}")

    # 3. 不同特征
    if a.feature_id != b.feature_id:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.UNRELATED, confidence=0.9,
                         reason="特征不同")

    sign_a = _DIRECTION_SIGN[a.effect_direction]
    sign_b = _DIRECTION_SIGN[b.effect_direction]

    # 4. 条件不同
    #
    # 「方向相同」必须是**两侧都给出明确正负且同号**。conditional / nonlinear /
    # unknown 的符号是 0，跟任何方向相乘都得 0，如果只用 `sign_a * sign_b < 0`
    # 做反向判定，它们会掉进 supports 分支——一条 conditional 断言就会同时
    # 「支持」互相矛盾的正负两条观点，理由还写着「方向一致」。这是错的：
    # 说不清方向就是说不清，只能判 uncertain。
    if _conditions_differ(a, b):
        if sign_a * sign_b < 0:
            return ClaimEdge(
                source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                relation=ClaimRelation.CONDITIONALLY_DIFFERS, confidence=0.85,
                reason=(f"方向相反但条件不同（{a.market_regime or '未标制度'}"
                        f"/{b.market_regime or '未标制度'}），"
                        "属于适用条件差异，**不计入冲突**。 " + reason_extra).strip())
        if sign_a != 0 and sign_a == sign_b:
            return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                             relation=ClaimRelation.SUPPORTS, confidence=0.7,
                             reason="方向一致（适用条件不同，不影响支持判定）")
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.UNCERTAIN, confidence=0.4,
                         reason=(f"条件不同，且至少一侧方向不明确"
                                 f"（{a.effect_direction.value} vs "
                                 f"{b.effect_direction.value}），无法判定支持或矛盾"))

    # 5. 条件齐平
    if sign_a * sign_b < 0:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.CONTRADICTS, confidence=0.9,
                         reason=(f"同一特征、同一币种期限制度下方向相反"
                                 f"（{a.effect_direction.value} vs "
                                 f"{b.effect_direction.value}）"))
    if a.effect_direction in _UNDECIDABLE or b.effect_direction in _UNDECIDABLE:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.UNCERTAIN, confidence=0.4,
                         reason="至少一侧的方向不是明确的正负，无法判定支持或矛盾")
    if sign_a == sign_b and sign_a != 0:
        return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                         relation=ClaimRelation.SUPPORTS, confidence=0.9,
                         reason="同特征、同条件、同方向")

    return ClaimEdge(source_claim_id=a.claim_id, target_claim_id=b.claim_id,
                     relation=ClaimRelation.UNCERTAIN, confidence=0.4,
                     reason="没有足够信息判定关系")


def build_graph(claims: Sequence[Claim]) -> List[ClaimEdge]:
    """两两判关系。样本量在本阶段的量级上完全不构成性能问题。"""
    edges: List[ClaimEdge] = []
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            edges.append(judge(a, b))
    return edges


def relation_counts(edges: Sequence[ClaimEdge]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edge in edges:
        counts[edge.relation.value] = counts.get(edge.relation.value, 0) + 1
    return counts


# 人工核验时可选的七个选项，与方案 §10.4 一一对应
HUMAN_RELATION_CHOICES: Dict[str, ClaimRelation] = {
    "A": ClaimRelation.SUPPORTS,
    "B": ClaimRelation.CONTRADICTS,
    "C": ClaimRelation.CONDITIONALLY_DIFFERS,
    "D": ClaimRelation.SUPERSEDES,
    "E": ClaimRelation.DUPLICATES,
    "G": ClaimRelation.UNCERTAIN,
}
# F 是「原子观点提取错误」，不是一种关系，而是把 claim 标成 invalid
HUMAN_EXTRACTION_ERROR = "F"
