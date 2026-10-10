"""历史事件库的记录。

和 `events.py` 里那套（`RawEvent` / `ParsedEvent`）是**两条不同的线**，别混：

- `events.py` —— 实时投递进来的消息，走到解析和闸门；
- 这里 —— 论文语料那 445 条**历史事件**，带异常收益标签，供检索和评测用。

第 3 周的记忆库检索的是这里的记录。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# 标签三态
LABELED = "labeled"      # 三个窗口都算出来了
PARTIAL = "partial"      # 短窗口有，长窗口还没走完
PENDING = "pending"      # 还没走完

# 时间切分
POOL = "pool"            # 检索池，2017–2022
EVAL = "eval"            # 评测集，2023–2024


class HistoricalEvent(BaseModel):
    event_id: str
    # 日历日，**不带时区**。带时区的话和行情索引（naive）一比就报错。
    date: datetime
    text: str
    initiating_side: str
    event_type: str
    event_family: str
    sentiment: str
    human_verified: bool = False

    # ---- 标签 ----
    ret_1d: Optional[float] = None
    ret_5d: Optional[float] = None
    ret_20d: Optional[float] = None
    abret_1d: Optional[float] = None
    abret_5d: Optional[float] = None
    abret_20d: Optional[float] = None
    abret_method: str = ""
    label_status: str = PENDING

    # ---- 事件日前一交易日的市场状态（方案 §4.3 的四个变量）----
    rate_spread_1y: Optional[float] = None
    usd_index: Optional[float] = None
    fx_vol_20d: Optional[float] = None
    fx_reserve_chg_3m: Optional[float] = None

    # ---- 市场模型诊断量（给第 4 周的冲突度量与敏感性分析用）----
    model_r2: Optional[float] = None
    model_beta_dxy: Optional[float] = None
    model_beta_spread: Optional[float] = None
    model_observations: Optional[int] = None
    # 残差标准差相对原始收益标准差降了多少——**判断模型有没有用的判据是它，
    # 不是 R²**。负数意味着模型不但没减掉噪声，反而加进去了。
    model_variance_reduction: Optional[float] = None

    # ---- 切分 ----
    split: str = POOL

    # 同一天还有几条别的事件（不同事件类型）。这类事件**共享同一个已实现收益**，
    # 第 4 周算冲突指数时不能把它们各自当成独立证据，否则同一天的反应被数了两遍。
    # 本语料里 445 条里有 10 组这种情况。
    same_day_peers: int = 1

    @property
    def was_relaxation(self) -> bool:
        return self.sentiment == "positive"

    @property
    def market_state_complete(self) -> bool:
        """四个市场状态变量都有值。缺一个就没法算市场状态相似度。"""
        return all(value is not None for value in (
            self.rate_spread_1y, self.usd_index,
            self.fx_vol_20d, self.fx_reserve_chg_3m,
        ))
