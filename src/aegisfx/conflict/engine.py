"""把检索回来的证据变成一份研究层输出。

流程：按交易日聚合 → 估稳健尺度 → 算冲突指标 → 定方向 → 判执行许可。

**执行许可永远是关的。** 不是「目前还没实现所以先关着」，而是本阶段的设计：
研究层负责把话说清楚，执行层要等共形区间和风险 Agent 都到位了才谈。
所以 `research_only=True` / `auto_execution_allowed=False` 写死在构造里，
阻止原因逐条列出来，让界面能原样展示。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import List, Optional, Sequence

import numpy as np

from ..schemas.research import (
    ALWAYS_BLOCKED,
    BLOCK_HIGH_DIRECTIONAL_CONFLICT,
    BLOCK_HIGH_DISPERSION,
    BLOCK_HIGH_NEUTRAL_MASS,
    BLOCK_INSUFFICIENT_ESS,
    BLOCK_NO_DIRECTIONAL_EVIDENCE,
    BLOCK_STALE_MARKET_DATA,
    BLOCK_UNVERIFIED_SOURCE,
    MAX_DIRECTIONAL_CONFLICT,
    MAX_NEUTRAL_MASS,
    MAX_NORMALIZED_DISPERSION,
    MIN_EFFECTIVE_SAMPLE_SIZE,
    ConflictReport,
    ResearchOutput,
)
from .aggregate import aggregate_by_day, aggregation_summary, realized_abret
from .direction import decide_direction, pool_history_means
from .metrics import MIN_DAYS_FOR_ROBUST_DISPERSION, robust_scale, summarise

DEFAULT_HORIZON = 20
_EPS = 1e-12


def _query_date(query) -> date:
    import pandas as pd

    stamp = pd.Timestamp(query.date)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.date()


def analyse(
    query,
    evidence: Sequence,
    pool: Sequence,
    *,
    horizon: int = DEFAULT_HORIZON,
    retrieval_mode: str = "",
    source_verified: bool = True,
    market_data_fresh: bool = False,
) -> ResearchOutput:
    """对一条查询产出研究方向 + 冲突诊断 + 执行许可。

    `pool` 是 as-of 池（查询日之前的事件），用来估中性尺度和历史均值。
    它**只用查询日之前的信息**，所以这两步本身也不含未来数据。
    """
    dated = aggregate_by_day(evidence, horizon)
    summary = aggregation_summary(dated)

    # 中性尺度与历史均值都只用查询日之前的事件
    pool_values = [realized_abret(item, horizon) for item in pool]
    scale = robust_scale([v for v in pool_values if v is not None])
    pool_families = [getattr(item, "event_family", "") for item in pool]
    family_mean, all_mean = pool_history_means(
        pool_values, pool_families, getattr(query, "event_family", ""))

    values = [group.abret for group in dated]
    weights = [group.weight for group in dated]
    stats = summarise(values, weights, scale=scale)

    direction, weighted_abret, direction_score, tie_used, tie_reason = decide_direction(
        values, weights,
        w_up=stats["w_up"], w_down=stats["w_down"], scale=scale,
        family_history_mean=family_mean, all_history_mean=all_mean,
    )

    for group in dated:
        from .metrics import classify

        group.direction_class = classify(group.abret, stats["tau"])

    conflict = ConflictReport(
        horizon=horizon,
        scale=scale,
        tau=stats["tau"],
        neutral_band_c=stats["neutral_band_c"],
        w_up=stats["w_up"],
        w_down=stats["w_down"],
        w_neutral=stats["w_neutral"],
        directional_conflict=stats["directional_conflict"],
        directional_conflict_status=stats["directional_conflict_status"],
        neutral_mass=stats["neutral_mass"],
        pairwise_dispersion=stats["pairwise_dispersion"],
        pairwise_dispersion_norm=stats["pairwise_dispersion_norm"],
        magnitude_dispersion=stats["magnitude_dispersion"],
        normalized_dispersion=stats["normalized_dispersion"],
        weighted_std=stats["weighted_std"],
        usable_days=stats["usable_days"],
        dispersion_defined=stats["dispersion_defined"],
        effective_sample_size=stats["effective_sample_size"],
        raw_event_count=summary["raw_event_count"],
        effective_evidence_days=summary["effective_evidence_days"],
        same_day_group_count=summary["same_day_group_count"],
    )

    reasons = _block_reasons(
        conflict, source_verified=source_verified,
        market_data_fresh=market_data_fresh)

    return ResearchOutput(
        event_id=getattr(query, "event_id", "?"),
        query_date=_query_date(query),
        horizon=horizon,
        predicted_direction=direction,
        predicted_abnormal_return=weighted_abret,
        direction_score=direction_score,
        tie_break_used=tie_used,
        tie_break_reason=tie_reason,
        conflict=conflict,
        evidence=dated,
        retrieval_mode=retrieval_mode,
        research_only=True,
        auto_execution_allowed=False,
        execution_block_reasons=reasons,
        generated_at=datetime.now(timezone.utc),
    )


def _block_reasons(
    conflict: ConflictReport,
    *,
    source_verified: bool,
    market_data_fresh: bool,
) -> List[str]:
    reasons: List[str] = []

    if conflict.w_up + conflict.w_down <= _EPS:
        reasons.append(BLOCK_NO_DIRECTIONAL_EVIDENCE)

    if conflict.effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE:
        reasons.append(
            f"{BLOCK_INSUFFICIENT_ESS}（ESS = {conflict.effective_sample_size:.2f} "
            f"< {MIN_EFFECTIVE_SAMPLE_SIZE}）")

    if not conflict.dispersion_defined:
        # K < 3 时稳健离散度必然为 0，那不是「证据集中」而是「算不出来」。
        # 拿它当「幅度一致」来用会把一个盲区当成一个结论。
        reasons.append(
            f"稳健离散度退化（可用证据日 {conflict.usable_days} < "
            f"{MIN_DAYS_FOR_ROBUST_DISPERSION}），该指标不可用于判断")

    if (conflict.directional_conflict is not None
            and conflict.directional_conflict > MAX_DIRECTIONAL_CONFLICT):
        reasons.append(
            f"{BLOCK_HIGH_DIRECTIONAL_CONFLICT}"
            f"（{conflict.directional_conflict:.2f} > {MAX_DIRECTIONAL_CONFLICT}）")

    if conflict.neutral_mass > MAX_NEUTRAL_MASS:
        reasons.append(
            f"{BLOCK_HIGH_NEUTRAL_MASS}"
            f"（{conflict.neutral_mass:.2f} > {MAX_NEUTRAL_MASS}）")

    if (conflict.normalized_dispersion is not None
            and conflict.normalized_dispersion > MAX_NORMALIZED_DISPERSION):
        reasons.append(
            f"{BLOCK_HIGH_DISPERSION}"
            f"（{conflict.normalized_dispersion:.2f} > {MAX_NORMALIZED_DISPERSION}）")

    if not market_data_fresh:
        # 研究模式用的就是历史日频数据，按定义不「新鲜」。
        # 这不是缺陷，是执行必须关闭的又一条理由。
        reasons.append(f"{BLOCK_STALE_MARKET_DATA}（研究模式使用历史日频数据）")

    if not source_verified:
        reasons.append(BLOCK_UNVERIFIED_SOURCE)

    reasons.extend(ALWAYS_BLOCKED)
    return reasons
