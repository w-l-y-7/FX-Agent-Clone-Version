"""标注工具：给检索候选打相关性分。

**这是一个内部标注工具，不是演示页。**

要填的只有两项：

- `mechanism_relevance`（0/1/2）——政策作用对象和汇率传导机制像不像；
- `overall_relevance`（0–3）——作为先例的总评分。

同事件族、同发起方、同情感这三项由程序算好，只读展示，不用人重复判断。

默认只显示 `required_for_primary=True` 且还没标完的记录。
这个范围是**全部 20 条查询的 E1/E2 各自 Top-5 的并集**——
比「前 10 条查询的 Top-10」覆盖更全，也不会造成查询选择偏差。
"""

import csv
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parents[1]
_ROOT = _APP_DIR.parent
for _path in (_ROOT, _APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import streamlit as st  # noqa: E402
from ui import page_header  # noqa: E402

from src.aegisfx.retrieval.queries import QUERY_DIR  # noqa: E402
from src.aegisfx.taxonomy import family_label  # noqa: E402

LABELS_PATH = QUERY_DIR / "labels.csv"

MECHANISM_OPTIONS = {
    "": "未填",
    0: "0　政策传导机制不同，不能作为有效先例",
    1: "1　部分相似",
    2: "2　作用对象与传导机制高度相似",
}
OVERALL_OPTIONS = {
    "": "未填",
    3: "3　高度相关（同族＋同发起方＋同方向，可直接类比）",
    2: "2　相关（同族，但发起方或方向不同）",
    1: "1　弱相关（不同族，但同属中美经贸摩擦）",
    0: "0　不相关",
}

page_header(
    "标注工具",
    "给每条检索候选打机制相关性与总体相关性。标完这一页，"
    "主题相关性这一路指标才能算。",
)

if not LABELS_PATH.exists():
    st.warning("还没有候选工作表。先跑：`python scripts/migrate_labels.py`")
    st.stop()


def read_rows():
    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def write_rows(rows, fieldnames):
    """写回 CSV。用 utf-8-sig 保持和迁移脚本一致，中文在 Excel 里也不乱码。"""
    with LABELS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_true(value) -> bool:
    return str(value).strip().lower() == "true"


def status_of(row) -> str:
    mech = (row.get("mechanism_relevance") or "").strip()
    overall = (row.get("overall_relevance") or "").strip()
    if mech and overall:
        return "complete"
    if mech or overall:
        return "partial"
    return "todo"


rows, fieldnames = read_rows()
primary = [r for r in rows if is_true(r.get("required_for_primary"))]
done_primary = sum(1 for r in primary if status_of(r) == "complete")

# ---------------------------------------------------------------- 进度

st.progress(done_primary / len(primary) if primary else 0.0,
            text=f"主评测记录：已标 {done_primary} / {len(primary)}　"
                 f"（另有余下 {len(rows) - len(primary)} 条非主评测记录）")

scope = st.radio(
    "显示范围",
    [f"主评测记录（{len(primary)} 条，默认）", f"全部记录（{len(rows)} 条）"],
    horizontal=True,
)
pool_rows = primary if scope.startswith("主评测") else rows

only_unfinished = st.checkbox("只看还没标完的", value=True)
if only_unfinished:
    pool_rows = [r for r in pool_rows if status_of(r) != "complete"]

if not pool_rows:
    st.success("这个范围里的记录都标完了。")
    st.stop()

# ---------------------------------------------------------------- 选查询

by_query = {}
for row in pool_rows:
    by_query.setdefault(row["query_id"], []).append(row)

query_ids = list(by_query)


def query_label(qid):
    items = by_query[qid]
    done = sum(1 for r in items if status_of(r) == "complete")
    return (f"{items[0].get('query_date', '')}　"
            f"{family_label(items[0].get('query_family', ''))}　"
            f"（{done}/{len(items)}）")


picked = st.selectbox("选择查询", query_ids, format_func=query_label)
candidates = by_query[picked]
first = candidates[0]

# ---------------------------------------------------------------- 当前事件

st.divider()
st.subheader("当前事件")
st.markdown(f"**{first.get('query_date', '')}　"
            f"{family_label(first.get('query_family', ''))}**")
st.info(first.get("query_text") or "（工作表里没有查询原文）")
st.caption(f"检索池大小 {first.get('pool_size') or '—'}　"
           "下面是 E1 与 E2 各自 Top-5 的并集（主评测记录）")

# ---------------------------------------------------------------- 逐条标注

st.divider()
with st.form(key=f"label_{picked}"):
    updates = {}
    for row in candidates:
        st.markdown(
            f"**{family_label(row.get('candidate_family', ''))}**　"
            f"{row.get('candidate_date', '')}　"
            f"{row.get('candidate_side', '')}　"
            f"{row.get('candidate_sentiment', '')}　"
            f"　E1 第 {row.get('e1_rank') or '—'} 名"
            f"　E2 第 {row.get('e2_rank') or '—'} 名"
        )
        st.write(row.get("candidate_text") or "（原文缺失）")

        auto = st.columns(3)
        auto[0].caption("同事件族：" + ("是" if is_true(row.get("same_family_auto")) else "否"))
        auto[1].caption("同发起方：" + ("是" if is_true(row.get("same_side_auto")) else "否"))
        auto[2].caption("同情感：" + ("是" if is_true(row.get("same_sentiment_auto")) else "否"))

        mech_current = (row.get("mechanism_relevance") or "").strip()
        overall_current = (row.get("overall_relevance") or "").strip()

        cols = st.columns(2)
        with cols[0]:
            mech_keys = list(MECHANISM_OPTIONS)
            mech_index = next((i for i, k in enumerate(mech_keys)
                               if str(k) == mech_current), 0)
            mech = st.radio("机制相关性", mech_keys, index=mech_index,
                            format_func=lambda k: MECHANISM_OPTIONS[k],
                            key=f"m_{row['candidate_event_id']}")
        with cols[1]:
            overall_keys = list(OVERALL_OPTIONS)
            overall_index = next((i for i, k in enumerate(overall_keys)
                                  if str(k) == overall_current), 0)
            overall = st.radio("总体相关性", overall_keys, index=overall_index,
                               format_func=lambda k: OVERALL_OPTIONS[k],
                               key=f"o_{row['candidate_event_id']}")

        updates[row["candidate_event_id"]] = (mech, overall)
        st.divider()

    note = st.text_input("备注（可选，整条查询共用）", value="")
    if st.form_submit_button("保存这一条查询的标注", type="primary"):
        for row in rows:
            if row["query_id"] != picked:
                continue
            mech, overall = updates.get(row["candidate_event_id"], ("", ""))
            row["mechanism_relevance"] = "" if mech == "" else str(mech)
            row["overall_relevance"] = "" if overall == "" else str(overall)
            if note.strip():
                row["label_note"] = note.strip()
            row["label_status"] = status_of(row)
        write_rows(rows, fieldnames)
        st.success("已保存。")
        st.rerun()

st.caption("保存在 `data/aegisfx/retrieval/labels.csv`。"
           "重跑 `migrate_labels.py` 或 `evaluate_retrieval.py` 都不会冲掉已填的分。")
