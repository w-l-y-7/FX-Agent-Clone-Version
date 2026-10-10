"""页 2：历史证据与冲突诊断。

三路方法可以在页面上切换：E1 纯语义 / E2 固定权重重排序 / E3 自适应冲突感知。
选定查询后，依次展示：检索到的证据、每条证据的中间量、按交易日聚合后的冲突诊断。

**冲突高的时候页面仍然显示方向**，但会显著标出「仅供研究、禁止自动执行」。
页面上不出现「65% 置信度」这类未经校准的数字。
"""

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _APP_DIR.parent
for _path in (_ROOT, _APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from ui import page_header, placeholder_notice  # noqa: E402

from src.aegisfx.conflict import analyse  # noqa: E402
from src.aegisfx.data.event_store import STORE_PATH, load_event_store  # noqa: E402
from src.aegisfx.data.market import describe_sources  # noqa: E402
from src.aegisfx.retrieval.queries import load_or_build  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    MODE_ADAPTIVE,
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_of_pool,
    retrieve_detailed,
)
from src.aegisfx.schemas.historical import EVAL  # noqa: E402
from src.aegisfx.schemas.views import RetrievalEventView  # noqa: E402
from src.aegisfx.taxonomy import family_label, label  # noqa: E402

MODE_TITLES = {
    MODE_SEMANTIC: "E1　纯语义",
    MODE_FINANCIAL: "E2　固定权重金融重排序",
    MODE_ADAPTIVE: "E3　自适应冲突感知重排序",
}

page_header(
    "历史证据与冲突诊断",
    "同一个 as-of 候选池，三种排序方式，以及按交易日聚合后的方向与冲突诊断。",
)

if not STORE_PATH.exists():
    st.warning("事件库还没建。执行：`python scripts/build_event_store.py`")
    st.stop()


@st.cache_data(ttl=60)
def load_frame():
    events = load_event_store()
    return pd.DataFrame([{
        "date": pd.Timestamp(e.date), "year": pd.Timestamp(e.date).year,
        "event_type": e.event_type, "event_family": e.event_family,
        "sentiment": e.sentiment, "initiating_side": e.initiating_side,
        "split": e.split, "ret_1d": e.ret_1d, "ret_5d": e.ret_5d, "ret_20d": e.ret_20d,
        "abret_1d": e.abret_1d, "abret_5d": e.abret_5d, "abret_20d": e.abret_20d,
        "model_r2": e.model_r2,
        "model_variance_reduction": e.model_variance_reduction,
        "model_observations": e.model_observations,
        "label_status": e.label_status, "same_day_peers": e.same_day_peers,
    } for e in events])


@st.cache_resource(show_spinner="加载向量模型…")
def load_index():
    from src.aegisfx.retrieval.embedder import build_embedder
    from src.aegisfx.retrieval.index import EventIndex

    events = load_event_store()
    embedder, note = build_embedder([event.text for event in events])
    index = EventIndex(events, embedder, note=note)
    if not index.load_vectors_from_disk():
        index.build()
    return events, index, note


frame = load_frame()

CONFIG_PATH = _ROOT / "reports" / "stage5" / "frozen_config.json"


def _show_e4(query, pool, result_e1, outcome_e1, by_id) -> None:
    """E4 门控值 + 两个专家预测 + 共形区间。

    门控值来自**冻结配置**里的系数，不在页面上重训——页面上看到的
    必须和评测脚本跑出来的是同一个数。
    """
    import json

    from src.aegisfx.conflict import analyse
    from src.aegisfx.conformal import ScaledConformal, rolling_conformal
    from src.aegisfx.data.maturity import MaturityCalendar
    from src.aegisfx.data.market import paper_market
    from src.aegisfx.retrieval.retriever import (
        MODE_FINANCIAL as _E2, retrieve_detailed as _rd,
    )
    from src.aegisfx.routing.features import FeatureBuilder
    from src.aegisfx.routing.gate import GateModel

    if not CONFIG_PATH.exists():
        st.caption("还没有冻结配置。跑 `python scripts/lock_stage5_config.py`。")
        return

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    item = config["horizons"].get("20")
    if not item or not item.get("model_config"):
        st.caption("冻结配置里没有 20 日的模型。")
        return

    outcome2 = _rd(query, load_event_store(), load_index()[1], mode=_E2,
                   top_k=10, pool=pool, horizon=20)
    result_e2 = analyse(query, list(outcome2.evidence), pool, horizon=20,
                        retrieval_mode=_E2)

    calendar = MaturityCalendar.from_market()
    builder = FeatureBuilder(paper_market())
    features = builder.build(
        RetrievalEventView.from_event(query, include_realized=False),
        same_family_ratio=outcome_e1.same_family_ratio,
        semantic_top=[h.components.get("semantic", 0.0) for h in outcome_e1.evidence],
        mature_count=len(pool),
        conflict_e1=result_e1.conflict.model_dump(),
        conflict_e2=result_e2.conflict.model_dump())

    gate = GateModel.from_config(item["model_config"])
    kept = item["features"]
    vector = np.array([[features[name] for name in kept]])
    g = float(gate.gate(vector)[0])
    e1_point = result_e1.predicted_abnormal_return or 0.0
    e2_point = result_e2.predicted_abnormal_return or 0.0
    e4_point = float(gate.blend(vector, [e1_point], [e2_point])[0])

    cols = st.columns(4)
    cols[0].metric("门控 g", f"{g:.3f}",
                   help="g→0 信任 E1 纯语义；g→1 信任 E2 金融重排序")
    cols[1].metric("E1 点预测", f"{e1_point:+.4%}")
    cols[2].metric("E2 点预测", f"{e2_point:+.4%}")
    cols[3].metric("E4 点预测", f"{e4_point:+.4%}")

    st.caption(
        f"g = σ(θᵀz)，r̂_E4 = (1−g)·{e1_point:+.4%} + g·{e2_point:+.4%}。"
        f"θ 来自 2018–2022 伪查询训练，λ = {item['selected_lambda']}，"
        "特征只用查询时已经可知的信息。"
    )

    # ---- 共形区间 ----
    st.markdown("**预测区间**")
    scaled = ScaledConformal.from_signal_report()
    oof_path = _ROOT / "data" / "aegisfx" / "routing" / "oof_predictions_h20.csv"
    predictions, realized, available = [], [], []
    if oof_path.exists():
        import csv as _csv

        with oof_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for record in _csv.DictReader(handle):
                predictions.append(float(record["e1_prediction"]))
                realized.append(float(record["realized_abret"]))
                available.append(calendar.available_index(record["query_date"], 20)
                                 or 10 ** 9)
    predictions.append(e1_point)
    realized.append(None)
    available.append(calendar.available_index(query.date, 20) or 10 ** 9)

    result = rolling_conformal(predictions, realized, available,
                               targets=(0.8, 0.9),
                               window=item["conformal"]["window"])
    rows = []
    for target, entries in result["per_query"].items():
        last = entries[-1]
        rows.append({
            "区间": f"{target:.0%}",
            "下界": "—" if last.lower is None else f"{last.lower:+.4%}",
            "上界": "—" if last.upper is None else f"{last.upper:+.4%}",
            "宽度": "—" if last.width is None else f"{last.width:.4%}",
            "校准点数": last.n_calibration,
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        f"方法：CP0 滚动共形（窗口 {item['conformal']['window']}）。"
        f"CP1 冲突尺度化已**自动降级**——{scaled.reason}。"
        "区间只用于风险控制，不改善点预测精度。"
    )

# ===========================================================================
# 检索与冲突诊断
# ===========================================================================

st.divider()
st.subheader("As-of 检索协议")
st.markdown(
    """
    对发生在 $t$ 的事件，**检索池只允许包含 $t$ 之前的事件**，同一天的其他事件
    也算未来信息一并排除。这个 $t$ 取 `source_published_at`——消息发布、
    当时可知的时间，**不是政策的生效时间**（生效日可能在未来）。

    代码里是**硬断言不是静默过滤**：`assert_no_future_evidence` 直接抛，
    因为静默过滤的情况下，日期类型不一致、时区没剥干净、排序反了——
    检索照样返回结果、指标照样算得出来，只是悄悄把未来喂了进去。

    检索层另有结构性的防线：排序函数只跟 `RetrievalEventView` 打交道，
    那上面没有查询事件的未来收益（见 `src/aegisfx/schemas/views.py`）。
    """
)

st.divider()
st.subheader("检索与冲突诊断")

try:
    _events, _index, _note = load_index()
    _by_id = {event.event_id: event for event in _events}
    _queries = load_or_build(_events)

    left, right = st.columns([3, 1])
    with left:
        _picked = st.selectbox(
            "选一条评测事件作为查询", [q.event_id for q in _queries],
            format_func=lambda eid: next(
                f"{q.date}　{family_label(q.family)}　{q.text[:30]}"
                for q in _queries if q.event_id == eid))
    with right:
        _mode = st.radio("方法", [MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE],
                         format_func=lambda m: m, horizontal=False)

    _query = _by_id[_picked]
    _pool = as_of_pool(_events, _query)
    _outcome = retrieve_detailed(_query, _events, _index, mode=_mode, top_k=10,
                                 pool=_pool)
    _result = analyse(_query, list(_outcome.evidence), _pool, horizon=20,
                      retrieval_mode=_mode)

    st.info(f"**查询**（{_query.date.date()}）：{_query.text}")
    st.caption(
        f"as-of 候选池 {len(_pool)} 条（只含查询日之前的事件）　"
        f"{MODE_TITLES[_mode]}"
        + (f"　自适应系数 α = {_outcome.adaptive_alpha:.3f}"
           f"（语义 Top-5 同族比例 {_outcome.same_family_ratio:.1%}）"
           if _mode == MODE_ADAPTIVE and _outcome.adaptive_alpha is not None
           else "")
    )

    # ---------------------------------------------------------------- 证据表
    st.markdown("**证据（按交易日聚合前的原始检索结果）**")
    shared_days = {group.trading_day.isoformat(): group.group_size
                   for group in _result.evidence}
    rows = []
    for hit in _outcome.evidence:
        day = pd.Timestamp(hit.event.date).date().isoformat()
        row = {
            "名次": hit.rank,
            "日期": day,
            "事件族": family_label(hit.event.event_family),
            "语义原始": f"{hit.components.get('semantic_score_raw', 0):+.4f}",
            "语义[0,1]": f"{hit.components.get('semantic_score_01', 0):.4f}",
            "金融分": f"{hit.components.get('finance_score', 0):.4f}",
            "最终得分": f"{hit.score:.4f}",
            "实现异常收益": ("—" if hit.event.realized_abret.get(20) is None
                             else f"{hit.event.realized_abret[20]:+.4%}"),
            "共享交易日": f"是（{shared_days.get(day, 1)} 条）" if shared_days.get(day, 1) > 1 else "否",
        }
        if _mode == MODE_ADAPTIVE:
            row["α"] = f"{hit.components.get('adaptive_alpha', 0):.3f}"
        row["事件原文"] = hit.event.text[:44]
        rows.append(row)
    st.dataframe(rows, width="stretch", hide_index=True)

    if _mode == MODE_ADAPTIVE:
        st.caption(
            f"E3 的最终分 = 语义[0,1] × (1 + α × 金融分)。"
            f"金融分 = 0.45×事件属性 + 0.40×市场状态 + 0.15×时间衰减。"
            f"候选集先收窄到语义 Top-20，金融信息只在够格的候选里排序。"
        )

    # ---------------------------------------------------------------- 冲突诊断
    st.divider()
    st.markdown("**冲突诊断（按交易日聚合后）**")
    c = _result.conflict

    top = st.columns(4)
    top[0].metric("研究方向", _result.predicted_direction.value)
    top[1].metric("预测异常收益",
                  "—" if _result.predicted_abnormal_return is None
                  else f"{_result.predicted_abnormal_return:+.4%}")
    top[2].metric("方向分歧度",
                  "—（无非中性证据）" if c.directional_conflict is None
                  else f"{c.directional_conflict:.3f}")
    top[3].metric("中性证据占比", f"{c.neutral_mass:.1%}")

    mid = st.columns(4)
    mid[0].metric("稳健离散度", "—" if c.magnitude_dispersion is None
                  else f"{c.magnitude_dispersion:.4%}",
                  help="K<3 时该量必然为 0，不可用于判断" if not c.dispersion_defined else None)
    mid[1].metric("归一化离散度", "—" if c.normalized_dispersion is None
                  else f"{c.normalized_dispersion:.3f}")
    mid[2].metric("有效样本量 ESS", f"{c.effective_sample_size:.2f}")
    mid[3].metric("证据量", f"{c.raw_event_count} 事件 / {c.effective_evidence_days} 天")

    st.caption(
        f"中性阈值 τ = {c.neutral_band_c} × scale，scale = {c.scale:.4%}"
        f"（查询日之前异常收益的 1.4826×MAD）　"
        f"权重：上 {c.w_up:.2f} / 下 {c.w_down:.2f} / 中性 {c.w_neutral:.2f}"
    )
    if c.same_day_group_count:
        st.caption(
            f"有 {c.same_day_group_count} 个交易日上落着多条事件，已按交易日聚合——"
            "同一天的事件共享同一个已实现收益，各自计入等于同一次反应数两遍。"
        )
    if not c.dispersion_defined:
        st.warning(
            f"可用证据日只有 {c.usable_days} 天，稳健离散度已退化（K<3 时必然为 0），"
            "这个数**不能**当作「幅度一致」来读。"
        )

    if _result.tie_break_used:
        st.caption(f"方向用了回退：{_result.tie_break_reason}")

    # ---------------------------------------------------------------- 执行状态
    st.divider()
    if _result.auto_execution_allowed:
        st.success("执行状态：ALLOW")
    else:
        st.error("**仅供研究：证据冲突或不充分，禁止自动执行。**")
    st.markdown("阻止原因：")
    for reason in _result.execution_block_reasons:
        st.markdown(f"- {reason}")

    if _note:
        st.warning(_note)

    # ---------------------------------------------------------------- E4
    st.divider()
    st.markdown("**E4　事件族—市场状态自适应专家路由**")
    try:
        _show_e4(_query, _pool, _result, _outcome, _by_id)
    except Exception as _exc:
        st.caption(f"E4 不可用：{type(_exc).__name__}: {str(_exc)[:120]}")

except Exception as _exc:
    st.warning(f"检索演示不可用：{type(_exc).__name__}: {str(_exc)[:160]}")

# ===========================================================================
# 事件库统计
# ===========================================================================

st.divider()
st.subheader("事件库")
top = st.columns(4)
evaluation = frame[frame["split"] == EVAL]
top[0].metric("事件总数", len(frame))
top[1].metric("检索池 2017–2022", len(frame) - len(evaluation))
top[2].metric("评测集 2023–2024", len(evaluation))
top[3].metric("同日多事件的日子",
              int((frame["same_day_peers"] > 1).sum()))

with st.expander("行情来源（两段不拼接）"):
    st.code(describe_sources(), language=None)

st.divider()
st.subheader("异常收益：为什么不能直接用汇率涨跌")
rows = []
for horizon in (1, 5, 20):
    raw = frame[f"ret_{horizon}d"].dropna()
    abn = frame[f"abret_{horizon}d"].dropna()
    rows.append({
        "窗口": f"{horizon} 日",
        "原始收益标准差": f"{raw.std():.4%}",
        "异常收益标准差": f"{abn.std():.4%}",
        "压缩": f"{1 - abn.std() / raw.std():.1%}",
    })
st.dataframe(rows, width="stretch", hide_index=True)
st.info(
    "**1 日那一行说明一件事**：模型在日频上几乎不起作用。人民币是管理浮动，"
    "中间价把大部分美元波动吸收掉了。报告里要把 1 日和 5/20 日分开讲。"
)

st.divider()
st.subheader("方向基准：所有准确率的地板")
col1, col2 = st.columns(2)
with col1:
    share = (evaluation["ret_20d"].dropna() > 0).mean()
    st.metric("原始收益口径", f"{max(share, 1 - share):.1%}")
with col2:
    abn = evaluation["abret_20d"].dropna()
    share_abn = (abn > 0).mean()
    st.metric("异常收益口径", f"{max(share_abn, 1 - share_abn):.1%}", f"n={len(abn)}")
st.warning(
    "换成异常收益口径后基准从约 71% 掉到约 56%——改善明显但**没有掉到 50%**。"
    "凡是方向准确率都要和这个数并列报。"
)

placeholder_notice(5, "Walk-Forward 共形预测区间。在此之前不输出任何未经校准的「置信度」")
