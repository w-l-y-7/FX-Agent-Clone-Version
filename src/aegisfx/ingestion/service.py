"""投递 → 解析 → 出信号 → 过闸门，一条链走完。

三个入口（CLI / Streamlit / inbox 目录）都调用这里的 `submit_raw_event`，
所以不管消息是人粘进来的还是机器读进来的，后面的路径完全一致。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Optional

from ..paths import DAILY_DATA, EVENTS, KILL_SWITCH, RAW_EVENTS, ensure_dirs
from ..schemas import (
    CheckStatus,
    EventRecord,
    GateCheck,
    GateDecision,
    ParsedEvent,
    RawEvent,
    Signal,
    utcnow,
)
from ..schemas.events import OTHER_EVENT_TYPE
from .parser import parse_event
from .store import append_jsonl, read_jsonl

# 执行闸门要求行情「新鲜」到什么程度。日频研究模式下必然不满足——
# 这是有意的：研究看历史，下单看当下，两件事本来就不该共用同一个水位。
FRESHNESS_MAX_DAYS = 1

# 异常收益的观察窗口（交易日）。20 个交易日约合 28 个自然日。
HORIZON_TRADING_DAYS = 20
_HORIZON_CALENDAR_DAYS = 28


@dataclass
class IngestResult:
    raw: RawEvent
    event: ParsedEvent
    signal: Signal
    gate: GateDecision


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------
# 行情水位
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def market_watermark() -> Optional[date]:
    """日频行情覆盖到哪一天。

    第 1 周只有 `Data.xlsx`（到 2024-12-31），第 7 周接入实时源后这里换成
    读实时水位。缓存一次，避免每投递一条消息就解析一遍 Excel。
    """
    if not DAILY_DATA.exists():
        return None
    try:
        import pandas as pd

        frame = pd.read_excel(DAILY_DATA, sheet_name="Data", usecols=["Date"])
        serial = pd.to_datetime(frame["Date"], unit="D", origin="1899-12-30")
        return serial.max().date()
    except Exception:
        return None


def _freshness_check(now: datetime) -> GateCheck:
    watermark = market_watermark()
    if watermark is None:
        return GateCheck(name="data_is_fresh", status=CheckStatus.FAIL,
                         detail="读不到日频行情水位")

    staleness = (now.date() - watermark).days
    if staleness <= FRESHNESS_MAX_DAYS:
        return GateCheck(name="data_is_fresh", status=CheckStatus.PASS,
                         detail=f"行情水位 {watermark}，滞后 {staleness} 天")
    return GateCheck(name="data_is_fresh", status=CheckStatus.FAIL,
                     detail=f"行情水位 {watermark}，滞后 {staleness} 天"
                            f"（执行要求 ≤ {FRESHNESS_MAX_DAYS} 天）")


# --------------------------------------------------------------------------
# 执行闸门
# --------------------------------------------------------------------------

_LATER = "第 {week} 周接入"


def run_gate(event: ParsedEvent, now: datetime) -> GateDecision:
    """跑一遍执行闸门。

    **没实现的检查项一律记为 NOT_IMPLEMENTED，而闸门要求全部 PASS。**
    所以第 1 周（检索、共形区间、风控都还没写）系统必然输出 BLOCKED，
    这是刻意的安全默认值，不是缺陷。
    """
    checks: List[GateCheck] = []

    checks.append(GateCheck(
        name="source_verified",
        status=CheckStatus.PASS if event.verified else CheckStatus.FAIL,
        detail="投递时提供了来源链接" if event.verified else "未提供来源链接，视为未验证消息",
    ))

    checks.append(GateCheck(
        name="parse_ok",
        status=CheckStatus.PASS if event.parse_ok else CheckStatus.FAIL,
        detail=(f"{event.parse_method} 解析成功，类型 {event.event_type}"
                if event.parse_ok else
                f"未能归入事件词表（{event.parse_notes or '原因未记录'}）"),
    ))

    checks.append(_freshness_check(now))

    checks.append(GateCheck(
        name="kill_switch",
        status=CheckStatus.FAIL if KILL_SWITCH.exists() else CheckStatus.PASS,
        detail="人工 Kill Switch 已按下" if KILL_SWITCH.exists() else "未按下",
    ))

    # 以下都还没实现，显式列出来而不是悄悄跳过——界面要能解释「为什么不让下单」
    checks.append(GateCheck(name="no_future_leakage", status=CheckStatus.NOT_IMPLEMENTED,
                            detail=_LATER.format(week=3) + "（检索层）"))
    checks.append(GateCheck(name="conflict_index", status=CheckStatus.NOT_IMPLEMENTED,
                            detail=_LATER.format(week=4) + "（双冲突度量）"))
    checks.append(GateCheck(name="interval_width", status=CheckStatus.NOT_IMPLEMENTED,
                            detail=_LATER.format(week=5) + "（共形区间）"))
    checks.append(GateCheck(name="spread", status=CheckStatus.NOT_IMPLEMENTED,
                            detail=_LATER.format(week=7) + "（点差监控）"))
    checks.append(GateCheck(name="daily_loss", status=CheckStatus.NOT_IMPLEMENTED,
                            detail=_LATER.format(week=6) + "（风险 Agent）"))
    checks.append(GateCheck(name="drawdown", status=CheckStatus.NOT_IMPLEMENTED,
                            detail=_LATER.format(week=6) + "（风险 Agent）"))

    return GateDecision.from_checks(checks)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _placeholder_signal(event_id: str, now: datetime) -> Signal:
    """研究层输出占位。

    **方向字段留着，但填 UNKNOWN**——「研究层始终输出方向」这条设计要保留，
    等第 4 周接上冲突感知检索自然会填上。现在编一个方向出来只会误导人。
    """
    return Signal(
        event_id=event_id,
        signal_generated_at=now,
        direction="UNKNOWN",
        status="placeholder",
        notes="研究层算法（检索 / 冲突度量 / 共形区间）尚未接入，见方案第 3–5 周",
    )


def submit_raw_event(
    text: str,
    *,
    source_url: Optional[str] = None,
    source_name: Optional[str] = None,
    source_published_at: Optional[datetime] = None,
    ingested_by: str = "cli",
    verified: Optional[bool] = None,
    use_llm: bool = True,
) -> IngestResult:
    """把一条消息送进流水线，返回结构化结果 + 闸门判定。

    `verified` 不给就按「有没有来源链接」自动判定——提供链接是「可核验」的最低门槛。
    要显式覆盖就传 True / False。
    """
    ensure_dirs()
    now = utcnow()

    published = source_published_at
    inferred = published is None
    if published is None:
        published = now
    elif published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    if verified is None:
        verified = bool(source_url)

    raw = RawEvent(
        raw_id=_new_id("raw"),
        ingested_at=now,
        ingested_by=ingested_by,
        text=text,
        source_url=source_url,
        source_name=source_name,
        source_published_at=published,
        source_published_at_inferred=inferred,
        verified=verified,
    )
    append_jsonl(RAW_EVENTS, raw)

    draft = parse_event(text, use_llm=use_llm)

    # 行情只到 2024-12-31。窗口落在覆盖范围内的标 unlabeled（等第 2 周算），
    # 落在覆盖范围外的标 pending，不进任何评测统计。
    watermark = market_watermark()
    due = published.date() + timedelta(days=_HORIZON_CALENDAR_DAYS)
    if watermark is not None and due <= watermark:
        abret_status = "unlabeled"
    else:
        abret_status = "pending"

    event = ParsedEvent(
        event_id=_new_id("evt"),
        raw_id=raw.raw_id,
        source_published_at=published,
        agent_detected_at=now,
        signal_generated_at=now,
        source_published_at_inferred=inferred,
        parse_ok=draft.event_type != OTHER_EVENT_TYPE,
        parse_method=draft.method,
        parse_notes=draft.notes or None,
        event_type=draft.event_type,
        initiating_side=draft.initiating_side,
        target_side=("US" if draft.initiating_side == "CN"
                     else "CN" if draft.initiating_side == "US"
                     else "BOTH" if draft.initiating_side == "BOTH" else "UNKNOWN"),
        sentiment=draft.sentiment,
        event_stage=draft.event_stage,
        abret_status=abret_status,
        label_due_at=datetime.combine(due, datetime.min.time(), tzinfo=timezone.utc),
        verified=verified,
    )

    signal = _placeholder_signal(event.event_id, now)
    gate = run_gate(event, now)

    append_jsonl(EVENTS, EventRecord(event=event, signal=signal, gate=gate))

    return IngestResult(raw=raw, event=event, signal=signal, gate=gate)


def list_events(limit: Optional[int] = None, *, newest_first: bool = True) -> List[EventRecord]:
    """读回落盘的事件记录。"""
    records = read_jsonl(EVENTS, EventRecord)
    records.sort(key=lambda r: r.recorded_at, reverse=newest_first)
    return records[:limit] if limit else records


def set_kill_switch(engaged: bool) -> bool:
    """人工叫停开关。返回当前状态。"""
    ensure_dirs()
    if engaged:
        KILL_SWITCH.write_text(
            f"人工按下 Kill Switch，时间 {utcnow().isoformat()}\n", encoding="utf-8"
        )
    elif KILL_SWITCH.exists():
        KILL_SWITCH.unlink()
    return KILL_SWITCH.exists()
