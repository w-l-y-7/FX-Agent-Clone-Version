"""三个页面共用的东西：启动准备、状态渲染、样式。

单独放一个模块，是为了让页面脚本只管「这一页显示什么」，
不用每页各写一遍 `sys.path` 和路径处理。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx import paths  # noqa: E402
from src.aegisfx.schemas import CheckStatus, EventRecord  # noqa: E402
from src.aegisfx.taxonomy import label  # noqa: E402

paths.load_env()
paths.ensure_dirs()

_CHECK_MARK = {CheckStatus.PASS: "通过", CheckStatus.FAIL: "不通过",
               CheckStatus.NOT_IMPLEMENTED: "未实现"}


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def kill_switch_banner() -> None:
    """Kill Switch 状态常驻顶部——任何一页都能一眼看到系统是不是被叫停了。"""
    if paths.KILL_SWITCH.exists():
        st.error("Kill Switch 已按下：**所有下单动作被禁止**。"
                 "到「风险与模拟交易」页复位。")
    else:
        st.success("Kill Switch 未按下。")


def render_gate(record: EventRecord) -> None:
    """渲染执行闸门。**清单全列出来**，不只给一个结论。"""
    gate = record.gate
    if gate.allow_execution:
        st.success("执行状态：ALLOW")
    else:
        st.warning("执行状态：BLOCKED")

    rows = [{
        "检查项": check.name,
        "状态": _CHECK_MARK[check.status],
        "说明": check.detail,
    } for check in gate.checks]
    st.dataframe(rows, width="stretch", hide_index=True)

    if gate.blocked_reasons:
        st.caption("未通过原因：" + "；".join(
            reason.split("：")[0] for reason in gate.blocked_reasons))


def render_signal(record: EventRecord) -> None:
    """研究层输出。第 1 周方向恒为 UNKNOWN，界面上要写清楚为什么。"""
    signal = record.signal
    left, right = st.columns([1, 2])
    with left:
        st.metric("方向", signal.direction)
        st.metric("异常收益预测",
                  "—" if signal.expected_abret is None
                  else f"{signal.expected_abret:+.2%}")
    with right:
        st.metric("80% 共形区间", "—")
        st.metric("方向概率", "—")
    st.info(f"研究层状态：{signal.status}。{signal.notes}")


def render_event(record: EventRecord, *, expanded: bool = True) -> None:
    event = record.event
    with st.expander(
        f"{record.recorded_at:%Y-%m-%d %H:%M}　{label(event.event_type)}　"
        f"[{record.gate.mode}]　{event.event_id}",
        expanded=expanded,
    ):
        cols = st.columns(4)
        cols[0].metric("事件类型", label(event.event_type))
        cols[1].metric("发起方", event.initiating_side)
        cols[2].metric("情感", event.sentiment)
        cols[3].metric("解析方式", event.parse_method)

        st.write("**四个时间戳**")
        st.dataframe([
            {"时间戳": "source_published_at",
             "值": str(event.source_published_at),
             "说明": "消息发布"
                    + ("（投递人未填，用接收时间占位，不计入延迟统计）"
                       if event.source_published_at_inferred else "")},
            {"时间戳": "agent_detected_at", "值": str(event.agent_detected_at),
             "说明": "系统发现"},
            {"时间戳": "signal_generated_at",
             "值": str(event.signal_generated_at) if event.signal_generated_at else "—",
             "说明": "产出信号"},
            {"时间戳": "paper_order_submitted_at", "值": "—",
             "说明": "模拟下单（实时接口尚未接入）"},
        ], width="stretch", hide_index=True)

        if event.source_published_at_inferred:
            st.caption("发布时间由投递人填写后才会计入延迟统计——"
                       "否则人工补录的滞后会污染「系统有多快」这个指标。")
        else:
            st.caption(f"端到端耗时 {event.latency_seconds:.1f} 秒"
                       f"（检测滞后 {event.detection_lag_seconds:.1f} 秒）")

        if event.parse_notes:
            st.caption(f"解析备注：{event.parse_notes}")
        st.caption(f"标签状态：{event.abret_status}")

        st.divider()
        render_signal(record)
        st.divider()
        render_gate(record)


@st.cache_data(ttl=30)
def load_audit_report() -> dict | None:
    """读第 1 周的事件库审计结果。没有就返回 None，页面自己处理。"""
    path = _PROJECT_ROOT / "reports" / "aegisfx_corpus_audit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def placeholder_notice(week: int, what: str) -> None:
    st.info(f"**第 {week} 周接入**：{what}")
