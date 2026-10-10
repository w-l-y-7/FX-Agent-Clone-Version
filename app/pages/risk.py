"""页 3：风险与模拟交易。"""

import json
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _APP_DIR.parent
for _path in (_ROOT, _APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st  # noqa: E402
from ui import page_header  # noqa: E402

from src.aegisfx import paths  # noqa: E402
from src.aegisfx.ingestion import list_events  # noqa: E402
from src.aegisfx.ingestion.service import set_kill_switch  # noqa: E402

page_header(
    "风险与模拟交易",
    "全部使用确定性规则，不调用大模型决定是否止损。",
)

# ---------------------------------------------------------------- Kill Switch

st.subheader("人工 Kill Switch")
engaged = paths.KILL_SWITCH.exists()
col1, col2 = st.columns([1, 3])
with col1:
    if engaged:
        if st.button("复位", type="primary"):
            set_kill_switch(False)
            st.rerun()
    else:
        if st.button("按下 Kill Switch", type="primary"):
            set_kill_switch(True)
            st.rerun()
with col2:
    if engaged:
        st.error("已按下：所有下单动作被禁止，执行闸门的 kill_switch 检查不通过。")
    else:
        st.success("未按下。执行闸门的 kill_switch 检查通过。")

st.caption("Kill Switch 是文件开关（`data/aegisfx/KILL_SWITCH`）——"
           "就算界面挂了，删掉文件或者用 CLI 也能复位。")

# ---------------------------------------------------------------- 闸门总览

st.divider()
st.subheader("执行闸门检查项")
st.caption("完整 paper 闸门包含 17 项确定性检查。没实现或拿不到实时字段时同样不放行。")

records = list_events(1)
if records:
    st.dataframe(
        [{"检查项": c.name, "状态": c.status.value, "说明": c.detail}
         for c in records[0].gate.checks],
        width="stretch", hide_index=True,
    )
else:
    st.info("还没有投递记录，先到「事件与信号」页投一条。")

# ---------------------------------------------------------------- 历史闸门回测

st.divider()
st.subheader("历史执行闸门回测")
report_dir = _ROOT / "reports" / "risk_gate"
available_reports = {
    "2023 开发集": report_dir / "evaluation_dev.json",
    "2024 准留出": report_dir / "evaluation_final.json",
}
available_reports = {name: path for name, path in available_reports.items() if path.exists()}
if not available_reports:
    st.info("尚未生成开发集风险回测。运行 `python scripts/evaluate_risk_gate.py --split dev --calibrate`。")
else:
    default_index = len(available_reports) - 1
    picked_report = st.selectbox("评测切分", list(available_reports), index=default_index)
    report_path = available_reports[picked_report]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    always = report["always_execute"]["drift_adjusted"]
    gated = report["risk_gated"]["drift_adjusted"]
    cols = st.columns(4)
    cols[0].metric("闸门放行", f"{report['n_allowed']} / {report['n_signals']}")
    cols[1].metric("拒绝率", f"{report['rejection_rate']:.1%}")
    cols[2].metric("漂移调整累计 PnL", f"{gated['total_pnl']:+.2%}",
                   delta=f"全执行 {always['total_pnl']:+.2%}")
    cols[3].metric("最大回撤", f"{gated['max_drawdown']:.2%}",
                   delta=f"全执行 {always['max_drawdown']:.2%}",
                   delta_color="inverse")
    st.dataframe([
        {"阻止原因": name, "次数": count}
        for name, count in report["blocked_reason_counts"].items()
    ], width="stretch", hide_index=True)
    st.caption(report["paper_gate_note"] + f" 历史成本情景：{report['spread_bps_assumption']:.1f} bps。")

st.warning("当前没有接入可用的分钟 Bid/Ask 与模拟交易 API，因此完整 paper gate 仍保持 BLOCKED。")

# ---------------------------------------------------------------- 审计日志

st.divider()
st.subheader("审计日志")
st.caption("每一条投递留下的完整记录（`data/aegisfx/events.jsonl`），"
           "含四个时间戳与闸门判定。")

records = list_events(50)
if not records:
    st.info("暂无记录。")
else:
    st.dataframe(
        [{
            "时间": f"{r.recorded_at:%Y-%m-%d %H:%M:%S}",
            "事件 ID": r.event.event_id,
            "事件类型": r.event.event_type,
            "解析": r.event.parse_method,
            "标签状态": r.event.abret_status,
            "闸门": r.gate.mode,
            "未通过项": "、".join(c.name for c in r.gate.checks if not c.passed) or "—",
        } for r in records],
        width="stretch",
        hide_index=True,
    )

st.divider()
st.warning(
    "系统只允许 research / backtest / shadow / paper 四种模式，"
    "**仓库默认不进入任何真实交易模式**。境内个人进行外汇交易存在明确的渠道合规限制。"
)
