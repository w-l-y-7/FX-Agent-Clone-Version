"""AegisFX-Agent 演示入口。

    .\\venv\\Scripts\\streamlit.exe run app\\Home.py

第 1 周这里只有外壳：**投递通道是真的**（三条入口都能用），
研究层（检索 / 冲突度量 / 共形区间）和执行层（风控 / 模拟下单）还没接，
界面上会明确标出每一项是哪一周接入。
"""

import streamlit as st

st.set_page_config(
    page_title="AegisFX-Agent",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

import ui  # noqa: E402  （必须在 streamlit 之后 import，它里面有 st 调用）

pages = [
    st.Page("pages/event_signal.py", title="事件与信号", default=True),
    st.Page("pages/evidence.py", title="历史证据"),
    st.Page("pages/risk.py", title="风险与模拟交易"),
    st.Page("pages/labeling.py", title="标注工具"),
]

navigation = st.navigation(pages)

with st.sidebar:
    st.markdown("### AegisFX-Agent")
    st.caption("冲突感知 RAG + 共形预测 + 执行风险闸门")
    st.divider()
    ui.kill_switch_banner()
    st.divider()
    st.caption("研究层强制输出方向；执行层按闸门决定是否下单。")
    st.caption("「标注工具」是内部用的，不是演示页。")

navigation.run()
