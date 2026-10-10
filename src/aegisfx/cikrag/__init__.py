"""CIK-RAG：冲突感知增量知识 RAG（第 6 阶段的主算法）。

它是一条流程，不是一堆算法：

    观点提取 → 条件对齐 → 冲突识别 → 可靠性聚合
    → 样本外效用验证 → 特征状态决策 → 增量更新

## 两条通道是分开的

- `aggregate.aggregate_feature` —— 知识侧：有没有可靠、适用、不冲突的知识支持；
- `ablation.feature_utility`     —— 预测侧：加入这个特征真的降低样本外误差吗。

**不合成一个总分。** 混在一起就说不清「这条为什么被拒」，
而这正是原论文 DA 最被诟病的地方（方案 §4.1）。

## 目前还不是一篇论文级的方法

方案 §3 写明了：**在完成完整文献检索与实验验证之前，不得声称 CIK-RAG 具有方法原创性。**
现阶段它是一套把「知识冲突」从措辞变成可计算量的工程实现。
"""

from .ablation import (
    FOLDS,
    build_dataset,
    feature_utility,
    feature_utility_with_interval,
    run_ablation,
    set_error,
)
from .aggregate import (
    APPLICABILITY_LOW,
    CONFLICT_HIGH,
    IGNORANCE_HIGH,
    ClaimAssessment,
    PredictiveUtility,
    aggregate_feature,
    assess_claims,
    decide,
)
from .extract import (
    cluster_sizes,
    deduplicate,
    extract_claims,
    source_span_ok,
)
from .relations import (
    HUMAN_EXTRACTION_ERROR,
    HUMAN_RELATION_CHOICES,
    build_graph,
    judge,
    relation_counts,
)
from .reliability import ReliabilityBreakdown, evaluate as evaluate_reliability
from .schema import (
    Claim,
    ClaimEdge,
    ClaimRelation,
    ClaimStatus,
    EffectDirection,
    EvidenceType,
    FeatureDecision,
    FeatureKnowledge,
)
from .selective import SelectiveAudit, audit_decisions
from .versioning import (
    FeedbackError,
    FeedbackLog,
    KnowledgeVersion,
    affected_features,
    feature_set_hash,
    incremental_update,
    make_version,
    version_id,
)

__all__ = [
    "APPLICABILITY_LOW",
    "CONFLICT_HIGH",
    "FOLDS",
    "HUMAN_EXTRACTION_ERROR",
    "HUMAN_RELATION_CHOICES",
    "IGNORANCE_HIGH",
    "Claim",
    "ClaimAssessment",
    "ClaimEdge",
    "ClaimRelation",
    "ClaimStatus",
    "EffectDirection",
    "EvidenceType",
    "FeatureDecision",
    "FeatureKnowledge",
    "FeedbackError",
    "FeedbackLog",
    "KnowledgeVersion",
    "PredictiveUtility",
    "ReliabilityBreakdown",
    "SelectiveAudit",
    "affected_features",
    "aggregate_feature",
    "feature_set_hash",
    "incremental_update",
    "make_version",
    "version_id",
    "assess_claims",
    "audit_decisions",
    "build_dataset",
    "build_graph",
    "cluster_sizes",
    "decide",
    "deduplicate",
    "evaluate_reliability",
    "extract_claims",
    "feature_utility",
    "feature_utility_with_interval",
    "judge",
    "relation_counts",
    "run_ablation",
    "set_error",
    "source_span_ok",
]
