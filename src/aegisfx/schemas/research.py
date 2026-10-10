"""研究层输出：方向 + 冲突诊断 + 执行许可。

两件事刻意分开：

1. **方向永远输出。** 证据再乱、再少，也要给一个方向，并把「为什么是这个方向」
   和「用了什么回退」写清楚。沉默比一个带说明的弱结论更没用。
2. **执行永远关闭。** 本阶段所有输出 `research_only=True`、
   `auto_execution_allowed=False`。这两项写死在构造逻辑里，
   不靠调用方自觉。

**不做「置信度百分数」。** 方向分歧度、中性占比、幅度离散度、ESS 各自独立报告，
不乘成一个未经校准的百分比——那种数字看起来精确，实际上没人能说清它的含义。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# 首版先验阈值。**不根据 20 条评测查询的结果调优**，第 6 周做敏感性分析。
NEUTRAL_BAND_C = 0.25          # τ = c × scale
MIN_EFFECTIVE_SAMPLE_SIZE = 2.0
MAX_DIRECTIONAL_CONFLICT = 0.8
MAX_NEUTRAL_MASS = 0.6
MAX_NORMALIZED_DISPERSION = 1.5


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class InsufficientDirectionalEvidence(str, Enum):
    STATUS = "insufficient_directional_evidence"


class DatedEvidence(BaseModel):
    """按交易日聚合后的一条证据。

    检索仍然以事件为单位，但**算预测、冲突、证据数时按交易日聚合**——
    同一天的多条事件共享同一个已实现收益，各自计入就等于同一次市场反应被数了两遍。
    """

    trading_day: date
    weight: float = Field(description="该交易日候选中的最大最终分数")
    abret: Optional[float] = None
    direction_class: Optional[Direction] = None
    event_ids: List[str] = Field(default_factory=list)
    texts: List[str] = Field(default_factory=list)

    @property
    def group_size(self) -> int:
        return len(self.event_ids)


class ConflictReport(BaseModel):
    horizon: int

    # 中性区间
    scale: float = Field(description="1.4826 × 查询日之前异常收益的 MAD")
    tau: float = Field(description="中性阈值 τ = c × scale")
    neutral_band_c: float = NEUTRAL_BAND_C

    # 三类权重
    w_up: float = 0.0
    w_down: float = 0.0
    w_neutral: float = 0.0

    # 冲突
    directional_conflict: Optional[float] = Field(
        default=None, description="None 表示没有非中性证据，不是「完全一致」")
    directional_conflict_status: str = "ok"
    neutral_mass: float = 0.0
    # 冲突特征主集合用成对离散度；MAD 型与加权标准差降为辅助
    pairwise_dispersion: Optional[float] = Field(
        default=None, description="加权成对离散度，双峰分布下不会塌")
    pairwise_dispersion_norm: Optional[float] = None

    magnitude_dispersion: Optional[float] = Field(
        default=None, description="加权 MAD 型稳健离散度（辅助，双峰下偏小）")
    normalized_dispersion: Optional[float] = None
    weighted_std: Optional[float] = Field(
        default=None, description="辅助诊断量")
    usable_days: int = Field(default=0, description="有非空异常收益的证据日数")
    dispersion_defined: bool = Field(
        default=False,
        description="证据日数是否够算稳健离散度。K<3 时该量必然为 0，不可用于判断")

    # 证据量
    effective_sample_size: float = 0.0
    raw_event_count: int = 0
    effective_evidence_days: int = 0
    same_day_group_count: int = 0


class ResearchOutput(BaseModel):
    """一次推理的完整输出。"""

    event_id: str
    query_date: date
    horizon: int

    predicted_direction: Direction = Direction.UNKNOWN
    predicted_abnormal_return: Optional[float] = None
    direction_score: Optional[float] = None
    tie_break_used: bool = False
    tie_break_reason: str = ""

    conflict: ConflictReport

    evidence: List[DatedEvidence] = Field(default_factory=list)
    retrieval_mode: str = ""

    # 这两项写死，不接受调用方覆盖
    research_only: bool = True
    auto_execution_allowed: bool = False
    execution_block_reasons: List[str] = Field(default_factory=list)

    generated_at: Optional[datetime] = None

    def block_summary(self) -> str:
        if not self.execution_block_reasons:
            return "（无阻止项——但本阶段仍然禁止自动执行）"
        return "；".join(self.execution_block_reasons)


# 阻止自动执行的原因，按严重程度排列
BLOCK_NO_DIRECTIONAL_EVIDENCE = "没有非中性证据，方向不可判"
BLOCK_INSUFFICIENT_ESS = "有效样本量不足"
BLOCK_HIGH_DIRECTIONAL_CONFLICT = "方向分歧度过高"
BLOCK_HIGH_NEUTRAL_MASS = "中性证据占比过高"
BLOCK_HIGH_DISPERSION = "幅度离散度过高"
BLOCK_STALE_MARKET_DATA = "行情数据过期"
BLOCK_UNVERIFIED_SOURCE = "消息来源未经验证"
BLOCK_NO_CONFORMAL_INTERVAL = "共形预测区间尚未实现"
BLOCK_RESEARCH_BUILD = "当前为研究版本，不接任何真实交易"

ALWAYS_BLOCKED = (BLOCK_NO_CONFORMAL_INTERVAL, BLOCK_RESEARCH_BUILD)
