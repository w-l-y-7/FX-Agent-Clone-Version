"""事件检索：E1 纯语义 / E2 固定金融重排序 / E3 语义约束自适应重排序。

三条规矩贯穿全包：

1. **As-of 优先**——检索池按查询日截断，越界直接抛（`AsOfViolation`），不静默过滤；
2. **权重先验固定**——E2/E3 的参数都不在评测集上调，第 6 周只做敏感性分析；
3. **相似度是算出来的**——大模型只在生成解释文本时出场。

E1/E2 的计算路径在引入 E3 时一行未动，回归测试守着（见 `scripts/test_aegisfx.py`）。
"""

from .embedder import build_embedder
from .index import EventIndex
from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .queries import QUERY_SET, QuerySpec, build_query_set
from .retriever import (
    MODE_ADAPTIVE,
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    MODES,
    AsOfViolation,
    RetrievalOutcome,
    RetrievedEvidence,
    as_of_pool,
    assert_no_future_evidence,
    normalise_mode,
    retrieve,
    retrieve_detailed,
)
from .scoring import (
    ALPHA_MAX,
    DEFAULT_WEIGHTS,
    FINANCE_WEIGHTS,
    adaptive_alpha,
    event_similarity,
    finance_score,
    market_similarity,
    same_family_ratio,
    semantic_to_01,
    time_similarity,
)

__all__ = [
    "ALPHA_MAX",
    "DEFAULT_WEIGHTS",
    "FINANCE_WEIGHTS",
    "MODE_ADAPTIVE",
    "MODE_FINANCIAL",
    "MODE_SEMANTIC",
    "MODES",
    "QUERY_SET",
    "AsOfViolation",
    "EventIndex",
    "QuerySpec",
    "RetrievalOutcome",
    "RetrievedEvidence",
    "adaptive_alpha",
    "as_of_pool",
    "assert_no_future_evidence",
    "build_embedder",
    "build_query_set",
    "event_similarity",
    "finance_score",
    "market_similarity",
    "ndcg_at_k",
    "normalise_mode",
    "precision_at_k",
    "recall_at_k",
    "retrieve",
    "retrieve_detailed",
    "same_family_ratio",
    "semantic_to_01",
    "time_similarity",
]
