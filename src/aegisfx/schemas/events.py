"""流水线三个阶段的载体。

- `RawEvent`    —— 投递进来的原始消息。**人工输入的入口到这里为止**，
                  再往后走什么路径（CLI / Streamlit / inbox 目录）都一样。
- `ParsedEvent` —— 解析后的结构化事件，四个时间戳在这里汇总。
- `Signal`      —— 研究层的输出。第 1 周是占位，第 4–5 周接真算法。

四个时间戳是本周的交付重点之一：没有它们，事后没法证明系统真的「快」，
也没法把人工补录的延迟和自动采集的延迟分开。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """带时区的当前时间。全项目统一用它，别用裸 `datetime.now()`。"""
    return datetime.now(timezone.utc)


# 历史语料里的 16 类事件。方案里写明了「枚举可扩展」：新消息（美联储、人行、
# 外汇管理政策等）落在语料覆盖之外时归入 `other` 并保留原文，不强行归类。
EVENT_TYPES: tuple[str, ...] = (
    "bilateral_dialogue",
    "us_tariff_increase",
    "us_tariff_relief",
    "us_restrict_cn_firms",
    "us_ease_restrictions_cn_firms",
    "us_litigation_cn",
    "us_criticize_cn",
    "us_designate_cn_fx_manipulator",
    "us_remove_cn_fx_manipulator",
    "cn_increase_purchases",
    "cn_tariff_increase",
    "cn_tariff_relief",
    "cn_restrict_us_firms",
    "cn_ease_restrictions_us_firms",
    "cn_litigation_us",
    "cn_criticize_us",
)

OTHER_EVENT_TYPE = "other"

INITIATING_SIDES: tuple[str, ...] = ("CN", "US", "BOTH", "UNKNOWN")
SENTIMENTS: tuple[str, ...] = ("positive", "negative", "unknown")


class RawEvent(BaseModel):
    """投递进来的原始消息。三个入口（CLI / Streamlit / inbox）都产出它。"""

    raw_id: str
    ingested_at: datetime
    ingested_by: str = Field(description="cli / streamlit / inbox / api / seed")
    text: str
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    source_published_at: Optional[datetime] = None
    # 投递人没填发布时间时，用 ingested_at 占位。这个标记必须留着，
    # 否则「人工补录的 8 小时延迟」会混进真实延迟统计里。
    source_published_at_inferred: bool = False
    verified: bool = False

    @property
    def known_at(self) -> datetime:
        """研究层可以「知道」这条消息的最早时刻。

        as-of 检索要用它，**不能用政策的生效时间**——生效日可能在未来，
        拿它截断等于提前用了未来信息。
        """
        return self.source_published_at or self.ingested_at


class ParsedEvent(BaseModel):
    """结构化事件 + 四个时间戳。"""

    event_id: str
    raw_id: str

    # ---- 四个时间戳（见方案 3.3）----
    source_published_at: datetime
    agent_detected_at: datetime
    signal_generated_at: Optional[datetime] = None
    paper_order_submitted_at: Optional[datetime] = None
    # 投递人没填发布时间，`source_published_at` 是拿接收时间占位的。
    # 延迟统计要把这类排除掉，否则人工补录的滞后会混进真实延迟里。
    source_published_at_inferred: bool = False

    # ---- 解析结果 ----
    parse_ok: bool
    parse_method: str = Field(description="llm / rule / failed")
    parse_notes: Optional[str] = None

    event_type: str = OTHER_EVENT_TYPE
    initiating_side: str = "UNKNOWN"
    target_side: str = "UNKNOWN"
    sentiment: str = "unknown"
    # 阶段只能靠大模型从文本推断，没有可靠来源，**仅用于展示**，不进检索特征
    event_stage: Optional[str] = None
    effective_time: Optional[datetime] = None

    # ---- 标签状态 ----
    # 日频行情只到 2024-12-31。之后的事件标 pending，不进任何评测统计，
    # 到 label_due_at 再补。
    abret_status: str = "pending"
    label_due_at: Optional[datetime] = None

    verified: bool = False

    @property
    def latency_seconds(self) -> Optional[float]:
        """从消息发布到产出信号花了多久。第 1 周先把这个口径固定下来。"""
        if self.signal_generated_at is None:
            return None
        return (self.signal_generated_at - self.source_published_at).total_seconds()

    @property
    def detection_lag_seconds(self) -> float:
        return (self.agent_detected_at - self.source_published_at).total_seconds()


class Signal(BaseModel):
    """研究层输出。第 1 周是占位，字段先定下来，第 4–5 周填真值。"""

    event_id: str
    signal_generated_at: datetime

    # 研究层**始终**输出方向，哪怕证据不足或互相冲突——这是「双输出」的前一半
    direction: str = Field(default="UNKNOWN", description="UP / DOWN / FLAT / UNKNOWN")
    expected_abret: Optional[float] = None
    raw_direction_probability: Optional[float] = None

    # 共形区间（第 5 周）
    interval_80: Optional[List[float]] = None
    interval_90: Optional[List[float]] = None

    # 双冲突度量（第 4 周）
    conflict_index: Optional[float] = None
    magnitude_dispersion: Optional[float] = None
    evidence_ids: List[str] = Field(default_factory=list)

    # placeholder = 尚未接真算法；real = 真实计算
    status: str = "placeholder"
    notes: str = ""
