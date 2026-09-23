"""PA2 的核心那一步：把新闻语料折叠成日频事件哑变量。

论文说 PA2 负责"把非结构化文本转成量化特征"。巧的是公开数据里这条链的**两端
都在**：

* 输入 —— `Data.xlsx` 的 `Sheet1`：445 条中美贸易新闻，每条带日期、摘要、
  事件类型、来源、情感（都是中文标签，多标签用 `/` 分隔）；
* 输出 —— 同文件的 `Data` sheet：1944 行 × 35 列，其中 25 列是 0/1 的事件特征。

中间那一步（按天折叠 + 多标签展开 + 交叉特征）**公开代码里没有**。
本模块把它补上，并且**逐列对照**复现结果和 `Data` sheet 的实际值，
把对得上的和对不上的都摆出来。

## 为什么不能指望完全对上

`load_news_corpus` 的说明里已经指出：`Sheet1` 的 445 条与 `Data` 的 429 个
事件日**不是同一个口径**。最明显的证据是论文 Table 5 里的 sample size
（`Negative events` 248、`Positive events` 181 …）恰好等于 `Data` sheet 各列的
求和，也就是说 `Data` 那一侧覆盖的语料比 `Sheet1` 更多——`Sheet1` 只是其中
一个快照。

所以本模块的目标不是"复现出完全一致的结果"，而是：

1. 把聚合规则写清楚（论文只描述了结果，没给规则）；
2. 量出两边差多少、差在哪些列；
3. 把差异的原因列出来，供后续找论文作者确认。

## 三个口径假设

论文没说新闻落在非交易日时该怎么处理。这里提供三种，都跑一遍对比：

* `exact`   —— 只保留日期正好是交易日的新闻，落在周末/假期的直接丢掉；
* `next`    —— 落到**之后**最近的一个交易日（消息在休市期间发酵，开盘才反映）；
* `previous`—— 落到**之前**最近的一个交易日。

哪种最接近 `Data` sheet 的实际值，就跑一遍看结果——这本身就是个可报告的发现。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .research_dataset import (
    EVENT_FEATURES,
    EVENT_SENTIMENT_FEATURES,
    EVENT_SOURCE_FEATURES,
    EVENT_TYPE_FEATURES,
    load_daily_frame,
    load_news_corpus,
)

# 新闻日期落到哪个交易日。见模块开头的说明。
DATE_POLICIES = ("exact", "next", "previous")


def _map_dates(dates: Sequence[pd.Timestamp], trading_days: pd.DatetimeIndex,
               policy: str) -> List[Any]:
    """把每条新闻的日期映射到一个交易日；映射不到就返回 None（该条丢弃）。

    `trading_days` 必须已排序。
    """
    if policy not in DATE_POLICIES:
        raise ValueError(f"不认识的日期口径：{policy}（可选 {DATE_POLICIES}）")

    mapped: List[Any] = []
    for date in dates:
        stamp = pd.Timestamp(date)

        if policy == "exact":
            mapped.append(stamp if stamp in trading_days else None)
            continue

        if policy == "next":
            position = int(trading_days.searchsorted(stamp, side="left"))
            hit = trading_days[position] if position < len(trading_days) else None
        else:  # previous
            position = int(trading_days.searchsorted(stamp, side="right")) - 1
            hit = trading_days[position] if position >= 0 else None

        mapped.append(hit)
    return mapped


def aggregate_events(
    corpus: List[Dict[str, Any]],
    trading_days: pd.DatetimeIndex,
    *,
    policy: str = "exact",
) -> pd.DataFrame:
    """把新闻语料折叠成一天一行的 0/1 事件表。

    返回的列就是 `EVENT_FEATURES` 那 25 个，索引是 `trading_days`。

    聚合规则（论文没给，这里是按列名语义推的）：

    * 16 个事件类型列 —— 当天出现过该类型的事件就置 1。一条新闻多标签时，
      每个标签各自置 1；同一天多条新闻不累加（这些列是**指示变量**，
      取值只有 0/1，求和正好等于论文 Table 5 的 sample size）。
    * 3 个来源列 —— 当天有该来源的新闻就置 1。来源同样是多标签。
    * 2 个总情感列 —— 当天有正面/负面情感的新闻就置 1。
    * 4 个交叉情感列 —— 需要**同时**满足情感和来源，比如
      `Positive_Events_from_China` 要求当天有「正面 且 来源含中国」的新闻。
      「中美」这个来源算中方还是美方，论文没说，这里两种都不算，
      只在 `Sino_US` 那一列体现。
    """
    frame = pd.DataFrame(0, index=trading_days, columns=EVENT_FEATURES, dtype=int)

    mapped = _map_dates([item["date"] for item in corpus], trading_days, policy)

    for day, item in zip(mapped, corpus):
        if day is None:
            continue

        for label in item["event_types"]:
            if label in EVENT_TYPE_FEATURES:
                frame.at[day, label] = 1

        for label in item["sources"]:
            if label in EVENT_SOURCE_FEATURES:
                frame.at[day, label] = 1

        for label in item["sentiments"]:
            if label in EVENT_SENTIMENT_FEATURES:
                frame.at[day, label] = 1

        # 交叉项：情感 × 来源。Sino_US 不计入中方或美方。
        sentiments = set(item["sentiments"])
        sources = set(item["sources"])
        for origin, source_label in (("China", "China"), ("the_US", "US")):
            positive = f"Positive_Events_from_{origin}"
            negative = f"Negative_Events_from_{origin}"
            if source_label not in sources:
                continue
            if "Positive_Events" in sentiments:
                frame.at[day, positive] = 1
            if "Negative_Events" in sentiments:
                frame.at[day, negative] = 1

    return frame


@dataclass
class ColumnComparison:
    column: str
    reconstructed: int      # 复现出来的 1 的个数
    observed: int           # Data sheet 里 1 的个数
    both_one: int           # 两边都置 1 的天数
    only_reconstructed: int
    only_observed: int

    @property
    def exact_match(self) -> bool:
        return self.only_reconstructed == 0 and self.only_observed == 0

    @property
    def jaccard(self) -> float:
        """交并比。两边都是 0 的列在没有事件的日子里会撑高"一致率"，
        所以用 Jaccard 比用逐日一致率更能反映真实重合度。"""
        union = self.both_one + self.only_reconstructed + self.only_observed
        return self.both_one / union if union else 1.0


@dataclass
class AggregationReport:
    policy: str
    news_total: int
    news_mapped: int
    columns: List[ColumnComparison] = field(default_factory=list)

    @property
    def matched_columns(self) -> List[ColumnComparison]:
        return [item for item in self.columns if item.exact_match]

    @property
    def mean_jaccard(self) -> float:
        return float(np.mean([item.jaccard for item in self.columns])) if self.columns else 0.0


def compare_with_data_sheet(
    reconstructed: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    policy: str,
    news_total: int,
    news_mapped: int,
) -> AggregationReport:
    """逐列对照复现结果和 `Data` sheet 的实际值。"""
    columns = []
    for name in EVENT_FEATURES:
        actual = observed[name].astype(int).reindex(reconstructed.index).fillna(0).astype(int)
        rebuilt = reconstructed[name].astype(int)

        both_one = int(((actual == 1) & (rebuilt == 1)).sum())
        only_rebuilt = int(((actual == 0) & (rebuilt == 1)).sum())
        only_actual = int(((actual == 1) & (rebuilt == 0)).sum())

        columns.append(ColumnComparison(
            column=name,
            reconstructed=int(rebuilt.sum()),
            observed=int(actual.sum()),
            both_one=both_one,
            only_reconstructed=only_rebuilt,
            only_observed=only_actual,
        ))

    return AggregationReport(
        policy=policy,
        news_total=news_total,
        news_mapped=news_mapped,
        columns=columns,
    )


def format_report(report: AggregationReport) -> str:
    """把对照结果排成一张可读的表。"""
    lines = [
        f"【日期口径：{report.policy}】",
        f"  新闻 {report.news_total} 条，映射到交易日的 {report.news_mapped} 条"
        f"（丢弃 {report.news_total - report.news_mapped} 条）",
        f"  完全一致的列：{len(report.matched_columns)} / {len(report.columns)}，"
        f"平均交并比 {report.mean_jaccard:.3f}",
        "",
        f"  {'列名':<52s} {'复现':>5s} {'实际':>5s} {'都中':>5s} {'只复现':>6s} {'只实际':>6s}",
    ]
    for item in sorted(report.columns, key=lambda row: row.jaccard, reverse=True):
        lines.append(
            f"  {item.column:<52s} {item.reconstructed:>5d} {item.observed:>5d} "
            f"{item.both_one:>5d} {item.only_reconstructed:>6d} {item.only_observed:>6d}"
        )
    return "\n".join(lines)


def run_all_policies(
    excel_path=None,
) -> Dict[str, AggregationReport]:
    """三种日期口径各跑一遍，返回口径 -> 对照报告。"""
    daily = load_daily_frame(excel_path)
    corpus = load_news_corpus(excel_path)
    trading_days = daily.index

    reports: Dict[str, AggregationReport] = {}
    for policy in DATE_POLICIES:
        rebuilt = aggregate_events(corpus, trading_days, policy=policy)
        mapped = sum(1 for day in _map_dates([item["date"] for item in corpus],
                                             trading_days, policy) if day is not None)
        reports[policy] = compare_with_data_sheet(
            rebuilt, daily, policy=policy,
            news_total=len(corpus), news_mapped=mapped,
        )
    return reports
