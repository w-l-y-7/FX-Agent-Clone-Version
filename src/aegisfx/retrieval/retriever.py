"""检索主入口：E1 / E2 / E3 三种方法，同一套 as-of 协议。

## As-of 是硬断言，不是静默过滤

`as_of_pool` 和 `assert_no_future_evidence` 都用 `raise`，不是 `filter`。
理由：静默过滤的情况下，一旦某处日期类型不一致、时区没剥干净、或者排序反了，
检索照样返回结果、指标照样算得出来，只是悄悄把未来信息喂了进去。

## 三个方法的关系

    E1  纯语义        —— 只按语义相似度排，候选集是整个 as-of 池
    E2  固定重排序    —— 四项加权和，候选集同样是整个 as-of 池
    E3  自适应重排序  —— 候选集**先**收窄到语义 Top-N，再在内部用金融分重排

E1 和 E2 的候选集相同、只有排序方式不同，所以那是一个干净消融。
E3 换了候选集，是另一件事：它赌的是「语义先筛出够格当先例的，
金融信息只在够格的里面排序」。第三种方法不能为了让前两种显得差而改前两种，
所以 E1/E2 的计算路径在本阶段**一行未动**，只有回归测试守着。

## 未来信息怎么被挡在门外

`retrieve_detailed` **只跟 `RetrievalEventView` 打交道**——查询和候选都在入口处
转成视图。视图上没有 `abret_*`、没有未来方向，所以排序代码路径在结构上
根本读不到查询事件的结果，不依赖「记得别去看」这种约定。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas.historical import HistoricalEvent
from ..schemas.views import RetrievalEventView, assert_retrieval_safe
from .index import EventIndex
from .scoring import (
    DEFAULT_WEIGHTS,
    MarketScaler,
    adaptive_alpha,
    composite,
    event_similarity,
    finance_score,
    market_similarity,
    market_vector,
    same_family_ratio,
    semantic_to_01,
    time_similarity,
)

MODE_SEMANTIC = "E1"
MODE_FINANCIAL = "E2"
MODE_ADAPTIVE = "E3"

# 检索时用的主观察期。候选池按这个期限执行标签成熟协议。
DEFAULT_HORIZON = 20

MODES = (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE)

# 旧名字仍然接受，免得前面的脚本和页面全要改
_MODE_ALIASES = {
    "semantic": MODE_SEMANTIC,
    "financial": MODE_FINANCIAL,
    "adaptive": MODE_ADAPTIVE,
}

# E3 的候选集大小：语义 Top-N
ADAPTIVE_CANDIDATE_POOL = 20

N_MARKET_FEATURES = 4


class AsOfViolation(AssertionError):
    """检索池里出现了查询日当天或之后的事件。"""


def normalise_mode(mode: str) -> str:
    mode = _MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"不认识的检索模式：{mode}（可选 {MODES}）")
    return mode


def as_day(value) -> pd.Timestamp:
    """统一成 naive 时间戳。时区没剥干净是 as-of 比较最阴的一类 bug。"""
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def as_of_pool(
    events: Sequence[HistoricalEvent],
    query,
    *,
    horizon: Optional[int] = None,
    calendar=None,
) -> List[HistoricalEvent]:
    """查询日**之前**的事件。

    同一天的其他事件也算未来信息，一并排除。

    **传了 `horizon` 就同时执行标签成熟协议**（第 5 阶段 P0）：
    候选的 h 日收益必须在查询日之前已经可用。只满足「早于查询日」是不够的——
    一条 1 月 1 日的事件，它的 20 日收益要到 1 月底才实现，
    1 月 10 日的查询用它就是用了当时不存在的信息（详见 `data/maturity.py`）。
    """
    if horizon is not None:
        from ..data.maturity import MaturityCalendar, mature_pool

        calendar = calendar or MaturityCalendar.from_market()
        return mature_pool(events, query, horizon, calendar)

    cutoff = as_day(query.date)
    pool = [event for event in events
            if event.event_id != query.event_id and as_day(event.date) < cutoff]
    assert_no_future_evidence(query, pool)
    return pool


def assert_no_future_evidence(query, evidence: Sequence) -> None:
    """硬断言：证据必须严格早于查询日。越界直接抛，不静默过滤。"""
    cutoff = as_day(query.date)
    offenders = [
        (getattr(event, "event_id", "?"), as_day(event.date).date().isoformat())
        for event in evidence
        if as_day(event.date) >= cutoff
    ]
    if offenders:
        preview = "、".join(f"{eid}@{day}" for eid, day in offenders[:5])
        raise AsOfViolation(
            f"查询 {getattr(query, 'event_id', '?')}（{cutoff.date()}）的证据里出现了"
            f"不早于查询日的事件：{preview}（共 {len(offenders)} 条）。"
            "这是时间泄漏，不能静默过滤掉。"
        )


@dataclass
class RetrievedEvidence:
    event: object
    rank: int
    score: float
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        return self.event.event_id


@dataclass
class RetrievalOutcome:
    """一次检索的完整结果，含查询级的自适应中间量。"""

    mode: str
    evidence: List[RetrievedEvidence]
    candidate_pool_size: int
    same_family_ratio: Optional[float] = None
    adaptive_alpha: Optional[float] = None

    def __iter__(self):
        # 让老调用方仍然能直接把 outcome 当列表遍历
        return iter(self.evidence)

    def __len__(self) -> int:
        return len(self.evidence)


def retrieve_detailed(
    query,
    events: Sequence[HistoricalEvent],
    index: EventIndex,
    *,
    mode: str = MODE_FINANCIAL,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    pool: Optional[Sequence] = None,
    adaptive_pool_size: int = ADAPTIVE_CANDIDATE_POOL,
    horizon: Optional[int] = DEFAULT_HORIZON,
    calendar=None,
) -> RetrievalOutcome:
    """检索查询事件的历史先例。

    `horizon` 默认 20，表示候选池按 20 日标签成熟协议筛选。
    传 None 可以关掉（只做纯排序、不碰标签的场景），但**所有消费标签的路径
    都必须开着**。
    """
    mode = normalise_mode(mode)
    assert_retrieval_safe(query)

    # 查询与候选一律先转成视图：排序代码路径**结构上**读不到未来字段。
    # 查询那份**不带**已实现收益——那是它自己的未来结果；候选那份带，
    # 因为候选都在查询日之前，它们的收益是历史信息。
    query_view = (query if isinstance(query, RetrievalEventView)
                  else RetrievalEventView.from_event(query, include_realized=False))

    pool = (list(pool) if pool is not None
            else as_of_pool(events, query_view, horizon=horizon, calendar=calendar))
    assert_no_future_evidence(query_view, pool)
    if not pool:
        return RetrievalOutcome(mode=mode, evidence=[], candidate_pool_size=0)

    pool_views = [
        item if isinstance(item, RetrievalEventView)
        else RetrievalEventView.from_event(item)
        for item in pool
    ]

    query_vector = index.vector_for(query_view.event_id)
    pool_vectors = np.vstack([index.vector_for(item.event_id) for item in pool_views])
    semantic_scores = pool_vectors @ query_vector

    # 标准化参数只用 as-of 池拟合。用全样本会把未来分布泄漏进来。
    scaler = MarketScaler.fit([market_vector(item) for item in pool_views])
    query_market = scaler.transform(market_vector(query_view)) if scaler else None

    weights = weights or DEFAULT_WEIGHTS
    scored: List[RetrievedEvidence] = []
    for position, item in enumerate(pool_views):
        raw_semantic = float(semantic_scores[position])
        components: Dict[str, float] = {"semantic": raw_semantic}
        components["semantic_score_raw"] = raw_semantic
        components["semantic_score_01"] = semantic_to_01(raw_semantic)

        if mode in (MODE_FINANCIAL, MODE_ADAPTIVE):
            candidate_market = scaler.transform(market_vector(item)) if scaler else None
            components["event"] = event_similarity(query_view, item)
            components["market"] = market_similarity(
                query_market, candidate_market, n_features=N_MARKET_FEATURES)
            components["time"] = time_similarity(query_view, item)

        scored.append(RetrievedEvidence(event=item, rank=0, score=0.0,
                                        components=components))

    if mode == MODE_SEMANTIC:
        for item in scored:
            item.score = item.components["semantic"]
        return _finish(scored, top_k, mode, len(pool))

    if mode == MODE_FINANCIAL:
        for item in scored:
            item.score = composite(item.components, weights)
        return _finish(scored, top_k, mode, len(pool))

    # ---- E3：先收窄到语义 Top-N，再在内部用金融分自适应重排 ----
    by_semantic = sorted(scored, key=lambda item: (-item.components["semantic"],
                                                   item.event.event_id))
    candidates = by_semantic[:max(1, adaptive_pool_size)]

    top5_families = [item.event.event_family for item in by_semantic[:5]]
    p_q = same_family_ratio(query_view.event_family, top5_families)
    alpha = adaptive_alpha(p_q)

    for item in candidates:
        item.components["finance_score"] = finance_score(item.components)
        item.components["same_family_ratio"] = p_q
        item.components["adaptive_alpha"] = alpha
        item.components["final_score"] = (
            item.components["semantic_score_01"]
            * (1.0 + alpha * item.components["finance_score"])
        )
        item.score = item.components["final_score"]

    outcome = _finish(candidates, top_k, mode, len(pool))
    outcome.same_family_ratio = p_q
    outcome.adaptive_alpha = alpha
    return outcome


def _finish(scored: List[RetrievedEvidence], top_k: int, mode: str,
            pool_size: int) -> RetrievalOutcome:
    scored.sort(key=lambda item: (-item.score, item.event.event_id))
    for rank, item in enumerate(scored[:top_k], start=1):
        item.rank = rank
    return RetrievalOutcome(mode=mode, evidence=scored[:top_k],
                            candidate_pool_size=pool_size)


def retrieve(
    query,
    events: Sequence[HistoricalEvent],
    index: EventIndex,
    *,
    mode: str = MODE_FINANCIAL,
    top_k: int = 10,
    weights: Optional[Dict[str, float]] = None,
    pool: Optional[Sequence] = None,
    horizon: Optional[int] = DEFAULT_HORIZON,
) -> List[RetrievedEvidence]:
    """`retrieve_detailed` 的薄封装，只返回证据列表。老调用方走这个。"""
    return retrieve_detailed(
        query, events, index, mode=mode, top_k=top_k,
        weights=weights, pool=pool, horizon=horizon).evidence
