"""把论文语料的 445 条新闻落成事件库。

一次落库做四件事：

1. **日期顺延**——新闻落在非交易日时顺延到之后最近的交易日。这个口径在复现
   阶段实测过：三种口径里 `next` 的列平均交并比最高（0.948），与论文一致。
2. **异常收益**——市场模型，估计窗口 120 个交易日，见 `abnormal.py`。
3. **市场状态**——事件日**前一交易日**的四个变量，避免用到当天信息。
4. **时间切分**——2017–2022 进检索池，2023–2024 进评测集。

落库结果是 `data/aegisfx/historical_events.jsonl`，第 3 周的记忆库直接读它。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..paths import DATA_DIR
from ..schemas.historical import (
    EVAL,
    LABELED,
    PARTIAL,
    PENDING,
    POOL,
    HistoricalEvent,
)
from ..taxonomy import COLUMN_TO_SNAKE, event_family
from .abnormal import DEFAULT_HORIZONS, MarketModelSet
from .market import MARKET_FEATURES, paper_market

STORE_PATH = DATA_DIR / "historical_events.jsonl"

EVAL_FROM_YEAR = 2023

_SIDE_FROM_CORPUS = {"China": "CN", "US": "US", "Sino_US": "BOTH"}
_SENTIMENT_FROM_CORPUS = {"Positive_Events": "positive", "Negative_Events": "negative"}


def _next_trading_day(stamp: pd.Timestamp, trading_days: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """顺延到之后最近的交易日（含当天）。与复现阶段验证过的 `next` 口径一致。"""
    position = int(trading_days.searchsorted(pd.Timestamp(stamp), side="left"))
    if position >= len(trading_days):
        return None
    return trading_days[position]


def _load_corpus_rows() -> List[dict]:
    from src.core.research_dataset import load_news_corpus

    rows = []
    for item in load_news_corpus():
        snakes = [COLUMN_TO_SNAKE[name] for name in item["event_types"]
                  if name in COLUMN_TO_SNAKE]
        if not snakes:
            continue
        sides = [name for name in item["sources"] if name in _SIDE_FROM_CORPUS]
        sentiments = [name for name in item["sentiments"] if name in _SENTIMENT_FROM_CORPUS]
        rows.append({
            "date": pd.Timestamp(item["date"]),
            "text": item["content"],
            "event_type": snakes[0],
            "side": _SIDE_FROM_CORPUS[sides[0]] if sides else "UNKNOWN",
            "sentiment": _SENTIMENT_FROM_CORPUS[sentiments[0]] if sentiments else "unknown",
        })
    return sorted(rows, key=lambda row: row["date"])


def _merge_corpus_rows(rows: List[dict], trading_days: pd.DatetimeIndex) -> List[dict]:
    """按方案 §5.2 的规则合并：**同一交易日 + 同一事件类型 + 同一发起方** 算一条。

    语料里 445 条新闻落在 445 个不同日历日上，顺延到交易日之后有 18 天撞车。
    撞车的 18 组里，8 组是同类型同发起方（比如同一份声明的两条报道），
    合并成一条；另外 10 组是不同事件类型（比如同一天美方加税、中方反制），
    **保留为两条**——它们是两件事。
    """
    buckets: Dict[tuple, dict] = {}
    ordered_keys: List[tuple] = []

    for row in rows:
        trading_day = _next_trading_day(row["date"], trading_days)
        if trading_day is None:
            continue
        key = (trading_day, row["event_type"], row["side"])
        if key not in buckets:
            buckets[key] = {**row, "date": trading_day, "texts": [row["text"]]}
            ordered_keys.append(key)
        else:
            buckets[key]["texts"].append(row["text"])

    merged: List[dict] = []
    for key in ordered_keys:
        bucket = buckets[key]
        merged.append({
            **bucket,
            "text": " ／ ".join(bucket["texts"]),
        })
    return sorted(merged, key=lambda item: item["date"])


def _label_status(results: Dict[int, object]) -> str:
    computed = [r for r in results.values() if r.abret is not None]
    if len(computed) == len(DEFAULT_HORIZONS):
        return LABELED
    if computed:
        return PARTIAL
    return PENDING


def build_event_store(*, verbose: bool = True) -> List[HistoricalEvent]:
    """重建事件库并写盘，返回全部记录。"""
    frame = paper_market()
    trading_days = frame.index
    merged = _merge_corpus_rows(_load_corpus_rows(), trading_days)

    # 每个事件先顺延到交易日，再一次性把位置算好——估计窗口里要按位置剔除其他事件日
    positioned = []
    for index, row in enumerate(merged):
        trading_day = _next_trading_day(row["date"], trading_days)
        if trading_day is not None:
            positioned.append((index, row, trading_day))
    event_positions = [int(trading_days.searchsorted(day, side="left"))
                       for _, _, day in positioned]

    # 同一天有几条事件（用于第 4 周避免把同一天的收益重复计入冲突度量）
    day_counts: Dict[pd.Timestamp, int] = {}
    for _, _, day in positioned:
        day_counts[day] = day_counts.get(day, 0) + 1

    # 因子序列整套复用，别在 445 条事件上各算一遍 h 日收益
    models = MarketModelSet(frame)

    events: List[HistoricalEvent] = []
    for order, (index, row, trading_day) in enumerate(positioned):
        results = models.abnormal_for(event_positions[order], event_positions)

        # 诊断量取 20 日那一档——长窗口的信噪比最高，模型有没有用看它最清楚
        model = results[20].model
        state = _market_state(frame, trading_day)

        events.append(HistoricalEvent(
            event_id=f"hist_{index:04d}",
            # 交易日期是**日历日**，不带时区。带时区的话和行情索引（naive）
            # 一比就报错，而且「哪一天」本来就不该依赖时区。
            date=datetime(trading_day.year, trading_day.month, trading_day.day),
            text=row["text"],
            initiating_side=row["side"],
            event_type=row["event_type"],
            event_family=event_family(row["event_type"]),
            sentiment=row["sentiment"],
            ret_1d=results[1].raw_ret,
            ret_5d=results[5].raw_ret,
            ret_20d=results[20].raw_ret,
            abret_1d=results[1].abret,
            abret_5d=results[5].abret,
            abret_20d=results[20].abret,
            abret_method=results[20].method if results[20].method else results[1].method,
            label_status=_label_status(results),
            **state,
            model_r2=model.r_squared if model else None,
            model_beta_dxy=model.betas["dxy"] if model else None,
            model_beta_spread=model.betas["spread"] if model else None,
            model_observations=model.n_observations if model else None,
            model_variance_reduction=model.variance_reduction if model else None,
            split=EVAL if trading_day.year >= EVAL_FROM_YEAR else POOL,
            same_day_peers=day_counts.get(trading_day, 1),
        ))

        if verbose and (order + 1) % 100 == 0:
            print(f"  已处理 {order + 1}/{len(positioned)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STORE_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(event.model_dump_json() + "\n")
    return events


def _market_state(frame: pd.DataFrame, trading_day: pd.Timestamp) -> dict:
    """事件日**前一交易日**的市场状态。取不到就给 None。"""
    position = int(frame.index.searchsorted(trading_day, side="left"))
    if position == 0:
        return {name: None for name in MARKET_FEATURES}
    row = frame.iloc[position - 1]
    return {name: (None if pd.isna(row[name]) else float(row[name]))
            for name in MARKET_FEATURES}


def load_event_store() -> List[HistoricalEvent]:
    """读回事件库。没建过就返回空列表。"""
    from ..ingestion.store import read_jsonl

    return read_jsonl(STORE_PATH, HistoricalEvent)


def store_summary(events: List[HistoricalEvent]) -> dict:
    """给报告和界面用的一页统计。"""
    from collections import Counter

    def stats(values):
        clean = [v for v in values if v is not None]
        if not clean:
            return {"n": 0}
        series = pd.Series(clean)
        return {
            "n": len(clean),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "share_up": float((series > 0).mean()),
        }

    collected: Dict[str, List[float]] = {}
    for event in events:
        collected.setdefault(event.event_family, [])
        if event.abret_20d is not None:
            collected[event.event_family].append(event.abret_20d)
    by_family = {family: stats(values) for family, values in collected.items()}

    methods = Counter(e.abret_method for e in events if e.abret_method)
    pool = [e for e in events if e.split == POOL]
    evaluation = [e for e in events if e.split == EVAL]

    return {
        "total": len(events),
        "split": {"pool": len(pool), "eval": len(evaluation),
                  "eval_from_year": EVAL_FROM_YEAR},
        "label_status": dict(Counter(e.label_status for e in events)),
        "abret_1d": stats([e.abret_1d for e in events]),
        "abret_5d": stats([e.abret_5d for e in events]),
        "abret_20d": stats([e.abret_20d for e in events]),
        "raw_ret_20d": stats([e.ret_20d for e in events]),
        "eval_abret_20d": stats([e.abret_20d for e in evaluation]),
        "eval_raw_ret_20d": stats([e.ret_20d for e in evaluation]),
        "by_family": by_family,
        "method": dict(methods),
        "model": {
            "mean_r2": float(pd.Series(
                [e.model_r2 for e in events if e.model_r2 is not None]).mean()),
            "mean_variance_reduction": float(pd.Series(
                [e.model_variance_reduction for e in events
                 if e.model_variance_reduction is not None]).mean()),
            "mean_beta_dxy": float(pd.Series(
                [e.model_beta_dxy for e in events if e.model_beta_dxy is not None]).mean()),
            "mean_beta_spread": float(pd.Series(
                [e.model_beta_spread for e in events if e.model_beta_spread is not None]).mean()),
        },
    }
