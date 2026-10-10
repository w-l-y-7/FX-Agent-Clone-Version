"""综合相似度的四个分量（方案 §4.3）。

    S = 0.40·S_semantic + 0.25·S_event + 0.25·S_market + 0.10·S_time

**权重先验固定，不在评测集上调。** 5 个权重加上后面几周的阈值十来个个自由度，
在 156 条评测事件上调必然过拟合；第 6 周做的是逐维敏感性分析，看结论会不会翻转。

**去掉了方案初稿里的「来源可信度」**——语料里的「来源」指的是中/美哪一方发起，
不是媒体出处，没有 source_url。要凑出 100+ 条事件的真实来源得手工回溯八年前的
新闻链接，会把八周计划冲垮。那一维的权重并到了 `S_event` 里（发起方匹配），
这是有数据支撑的，而且「中方反制 vs 美方施压的市场反应是否对称」本身是个有意思的问题。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from ..schemas.historical import HistoricalEvent

# 先验权重，四维和为 1
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.40,
    "event": 0.25,
    "market": 0.25,
    "time": 0.10,
}

# ---------------------------------------------------------------------------
# E3：语义约束的自适应重排序
# ---------------------------------------------------------------------------
#
# E3 与 E2 的区别不是「把金融变量的权重调大」，而是**先保证候选项在语义上
# 够格当先例，再在它们内部用金融信息重排**：
#
#   1. 候选集 = 语义 Top-N（默认 20），不是整个 as-of 池；
#   2. 金融分 F_i 只在这 20 条之间比较；
#   3. 最终分 = 语义分 × (1 + α_q·F_i)，α_q 由「语义检索本身聚不聚族」决定。
#
# 第 3 条是关键：α 是**查询级自适应**的。语义 Top-5 高度同族时说明主题已经确定，
# 可以让金融信息多发挥作用；语义结果本身就很散时，α 自动压低，
# 免得金融变量把主题带偏。

# 金融分的三项权重（和 E2 的四项权重不是一套，别混）
FINANCE_WEIGHTS: Dict[str, float] = {"event": 0.45, "market": 0.40, "time": 0.15}

# α 的上限。首版先验固定，**禁止**根据 20 条评测查询的结果调优。
ALPHA_MAX = 0.5

# p_q 低于这个值就完全关闭金融重排（语义结果太散，主题还没确定）
_ALPHA_DEAD_ZONE = 0.5


def semantic_to_01(raw: float) -> float:
    """余弦相似度 → [0,1]。

    余弦在 [-1,1] 上取值，而事件/市场/时间三项都在 [0,1]。
    直接相乘会把量纲不一致的分数混在一起——E2 那套加权和没这个问题（因为
    权重本身就承担了尺度），但 E3 是乘法，必须先归一。
    """
    return min(1.0, max(0.0, (float(raw) + 1.0) / 2.0))


def finance_score(components: Dict[str, float],
                  weights: Dict[str, float] = None) -> float:
    """F_i = 0.45·S_event + 0.40·S_market + 0.15·S_time，三项都在 [0,1]。"""
    weights = weights or FINANCE_WEIGHTS
    return float(sum(weights[name] * components.get(name, 0.0) for name in weights))


def same_family_ratio(query_family: str, top_families: Sequence[str]) -> float:
    """语义 Top-5 里与查询同族的比例 p_q。"""
    if not top_families:
        return 0.0
    return sum(1 for family in top_families if family == query_family) / len(top_families)


def adaptive_alpha(p_q: float, alpha_max: float = ALPHA_MAX) -> float:
    """α_q = α_max · clip((p_q − 0.5)/0.5, 0, 1)。

    p_q ≤ 0.5 时 α = 0：语义结果里同族不到一半，说明主题还没确定，
    这时候让金融变量去搅和只会把不相关的事件排上来。
    """
    scaled = (float(p_q) - _ALPHA_DEAD_ZONE) / (1.0 - _ALPHA_DEAD_ZONE)
    return float(alpha_max * min(1.0, max(0.0, scaled)))

# 时间衰减的半衰期（天）。730 天 = 两年前的事件权重降到一半。
# 语料跨 2017–2024，查询在 2023–2024，Δt 常在 1500–2500 天。
# 半衰期取一年的话 2018 年的事到 2024 年权重只剩 0.02，等于把它们删掉了——
# 而历史重大政策恰恰不该因为久远被删。这一维权重只有 0.10，本来就是温和的排序微调。
TIME_HALF_LIFE_DAYS = 730.0

# 事件属性三个字段的权重
_EVENT_FIELD_WEIGHTS = {"type": 0.5, "side": 0.3, "sentiment": 0.2}

# 同一事件族但具体类型不同时给的部分分
_SAME_FAMILY_CREDIT = 0.5


def event_similarity(query: HistoricalEvent, candidate: HistoricalEvent) -> float:
    """事件属性相似度：类型 + 发起方 + 情感。

    类型分三档：完全相同 1.0，同一事件族 0.5，其余 0。
    用事件族给部分分是有依据的——第 1 周审过语料的标注边界，
    `us_restrict_cn_firms` 和 `us_litigation_cn` 之间连大模型都分不准，
    所以「同族不同类」不该和「完全不相干」得一样的分。
    """
    if query.event_type == candidate.event_type:
        type_score = 1.0
    elif query.event_family == candidate.event_family:
        type_score = _SAME_FAMILY_CREDIT
    else:
        type_score = 0.0

    side_score = 1.0 if query.initiating_side == candidate.initiating_side else 0.0
    sentiment_score = 1.0 if query.sentiment == candidate.sentiment else 0.0

    return (
        _EVENT_FIELD_WEIGHTS["type"] * type_score
        + _EVENT_FIELD_WEIGHTS["side"] * side_score
        + _EVENT_FIELD_WEIGHTS["sentiment"] * sentiment_score
    )


def time_similarity(query, candidate) -> float:
    """时间衰减，只做权重不做过滤。"""
    delta_days = abs((_as_day(query) - _as_day(candidate)).days)
    return math.exp(-math.log(2.0) * delta_days / TIME_HALF_LIFE_DAYS)


def market_similarity(
    query_vector: Optional[np.ndarray],
    candidate_vector: Optional[np.ndarray],
    *,
    n_features: int,
) -> float:
    """市场状态相似度：标准化欧氏距离上的高斯核。

    核宽取 σ² = 特征个数，这不是随手定的：它让「四个维度都差一个标准差」
    对应相似度 exp(−4/(2·4)) = 0.61，「都差两个标准差」对应 0.14。
    这样就不必再引一个需要调的核宽参数。
    """
    if query_vector is None or candidate_vector is None:
        return 0.0
    distance_squared = float(np.sum((query_vector - candidate_vector) ** 2))
    return math.exp(-distance_squared / (2.0 * max(1, n_features)))


@dataclass
class MarketScaler:
    """市场状态向量的标准化参数。

    **必须只用查询日之前的事件来拟合。** 用全样本拟合会把未来的分布
    泄漏进相似度计算——这类泄漏不会报错，只会让结果悄悄变好。
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, vectors: Sequence[np.ndarray]) -> Optional["MarketScaler"]:
        clean = [v for v in vectors if v is not None]
        if len(clean) < 2:
            return None
        matrix = np.vstack(clean)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        # 某个维度在参考集里完全不动时标准差会是 0，除零会让整列变成 inf
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        return (vector - self.mean) / self.std


def market_vector(event: HistoricalEvent) -> Optional[np.ndarray]:
    values = [
        event.rate_spread_1y,
        event.usd_index,
        event.fx_vol_20d,
        event.fx_reserve_chg_3m,
    ]
    if any(value is None for value in values):
        return None
    return np.array(values, dtype=float)


def composite(components: Dict[str, float],
              weights: Dict[str, float] = None) -> float:
    weights = weights or DEFAULT_WEIGHTS
    return float(sum(weights[name] * components.get(name, 0.0) for name in weights))


def _as_day(event: HistoricalEvent):
    import pandas as pd

    stamp = pd.Timestamp(event.date)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
