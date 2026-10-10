"""页 1：事件与信号。

这一页是第 1 周唯一「真的能用」的功能——把一条政策消息投进来，
看它被解析成什么、研究层输出什么、执行闸门为什么（不）放行。
"""

import sys
from datetime import datetime
from pathlib import Path

# `app/` 是自己加进来的：Streamlit 跑的时候主脚本目录已经在 sys.path 上，
# 但直接 `python app/pages/xxx.py` 冒烟测试时不会，那时候 `ui` 就找不到了。
_APP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _APP_DIR.parent
for _path in (_ROOT, _APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st  # noqa: E402
from ui import load_audit_report, page_header, render_event  # noqa: E402

from src.aegisfx.ingestion import (  # noqa: E402
    list_events,
    process_inbox,
    submit_raw_event,
)

page_header(
    "事件与信号",
    "投递一条政策消息 → 结构化 → 研究层输出 → 执行闸门判定。"
    "投递通道三条入口（本页输入框 / CLI / data/inbox 目录）走的是同一条流水线。",
)

# ---------------------------------------------------------------- 投递

with st.form("ingest", clear_on_submit=False):
    st.subheader("投递消息")
    text = st.text_area(
        "消息原文",
        height=110,
        placeholder="例：美国贸易代表办公室宣布，将对中国进口的电动汽车加征关税至 100%",
    )
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        source_url = st.text_input("来源链接（给了它默认判为已核验）")
    with col2:
        source_name = st.text_input("来源名称")
    with col3:
        published = st.text_input("发布时间（ISO 8601，可留空）",
                                  placeholder="2026-09-26T14:30:00+08:00")

    col4, col5 = st.columns([1, 3])
    with col4:
        use_llm = st.checkbox("用大模型解析", value=True,
                              help="关掉就走关键词规则，不花 API 调用")
    with col5:
        submitted = st.form_submit_button("投递", type="primary")

if submitted:
    if not text.strip():
        st.warning("消息原文不能为空。")
    else:
        published_at = None
        if published.strip():
            try:
                published_at = datetime.fromisoformat(
                    published.strip().replace("Z", "+00:00"))
            except ValueError:
                st.error("发布时间格式不对，要 ISO 8601，例如 2026-09-26T14:30:00+08:00")
                st.stop()
        with st.spinner("解析中…"):
            result = submit_raw_event(
                text.strip(),
                source_url=source_url.strip() or None,
                source_name=source_name.strip() or None,
                source_published_at=published_at,
                ingested_by="streamlit",
                use_llm=use_llm,
            )
        st.session_state["last_record"] = list_events(1)[0]

record = st.session_state.get("last_record")
if record is not None:
    st.divider()
    st.subheader("最新一条处理结果")
    render_event(record)

# ---------------------------------------------------------------- 待处理与历史

st.divider()
left, right = st.columns([1, 2])

with left:
    st.subheader("inbox 目录")
    st.caption("把 `.json` 丢进 `data/inbox/`，这里读走；"
               "处理完会挪到 `processed/`，解析失败挪到 `failed/`。")
    if st.button("扫描并处理 inbox"):
        results = process_inbox()
        if results:
            st.success(f"处理了 {len(results)} 条。")
            st.session_state["last_record"] = list_events(1)[0]
            st.rerun()
        else:
            st.info("inbox 里没有待处理的消息。")

with right:
    st.subheader("最近投递")
    records = list_events(15)
    if not records:
        st.info("还没有投递记录。上面投一条试试。")
    else:
        st.dataframe(
            [{
                "时间": f"{r.recorded_at:%Y-%m-%d %H:%M}",
                "事件类型": r.event.event_type,
                "发起方": r.event.initiating_side,
                "解析": r.event.parse_method,
                "闸门": r.gate.mode,
                "事件 ID": r.event.event_id,
            } for r in records],
            width="stretch",
            hide_index=True,
        )

# ---------------------------------------------------------------- 事件库审计

st.divider()
st.subheader("历史事件库（审计结果）")
report = load_audit_report()
if report is None:
    st.info("还没跑过审计。执行：`python scripts/audit_event_corpus.py`")
else:
    top = st.columns(5)
    top[0].metric("事件总数", report["total_events"])
    top[1].metric("检索池 2017–2022", report["split"]["retrieval_pool"])
    top[2].metric("评测集 2023–2024", report["split"]["evaluation"])
    top[3].metric("规则解析一致率", f"{report['rule_parser']['accuracy']:.1%}")
    llm = report.get("llm_parser_sample")
    top[4].metric("大模型一致率",
                  f"{llm['accuracy']:.1%}" if llm else "未测")
    st.caption(
        f"评测集 20 日方向基准（一律猜上涨的准确率）"
        f"**{report['eval_direction_base_rate_20d']:.1%}**——"
        "方向准确率必须与这个数并列报告。"
    )
