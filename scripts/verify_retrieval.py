"""检索层健康检查（第 3 周）。

    .\\venv\\Scripts\\python.exe scripts\\verify_retrieval.py

重点是**时间泄漏**。这类错误不会报错，只会让指标虚高，所以在三个层面各查一遍：

1. 直接调 `as_of_pool`，确认返回的每一条都严格早于查询日；
2. 故意构造一个越界证据，确认 `assert_no_future_evidence` **真的会抛**——
   一个从不触发的断言等于没有；
3. 对已经落盘的标注工作表，逐行核对候选日期 < 查询日期。

其余检查：两路检索的候选集一致（E1/E2 是同池不同排序）、排名连续、
查询集与事件库对得上。
"""

import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import EventIndex  # noqa: E402
from src.aegisfx.retrieval.queries import load_or_build  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    AsOfViolation,
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_day,
    as_of_pool,
    assert_no_future_evidence,
    retrieve,
)

LABELS_PATH = _PROJECT_ROOT / "data" / "aegisfx" / "retrieval" / "labels.csv"
PASSED, FAILED = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'通过' if ok else '失败'}] {name}" + (f" —— {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("检索层健康检查")
    print("=" * 78)

    events = load_event_store()
    queries = load_or_build(events)
    by_id = {event.event_id: event for event in events}
    print(f"\n事件 {len(events)} 条，查询 {len(queries)} 条\n")

    # ---- 1. as_of_pool 真的只给过去 ----
    print("1. As-of 截断")
    leaks = []
    for query in queries:
        event = by_id[query.event_id]
        pool = as_of_pool(events, event)
        cutoff = as_day(event.date)
        leaks += [(query.event_id, other.event_id) for other in pool
                  if as_day(other.date) >= cutoff]
    check("as_of_pool 返回的每一条都严格早于查询日", not leaks,
          f"{len(leaks)} 条越界" if leaks else "")

    pools_grow = all(
        len(as_of_pool(events, by_id[q.event_id])) > 0 for q in queries)
    check("每条查询的池都非空", pools_grow)

    # ---- 2. 断言真的会抛 ----
    # 一个从不触发的断言等于没有，所以这里**故意构造反例**。
    # 注意不能拿最后一条查询去找「未来事件」——它就是库里最新的那天，
    # 后面什么都没有，反例构造不出来（第一版就是这么写错的）。
    print("\n2. 越界断言必须真的会抛")
    earliest = min(queries, key=lambda q: q.date)
    query_event = by_id[earliest.event_id]
    future = [event for event in events if as_day(event.date) > as_day(query_event.date)]
    check("能构造出「未来证据」反例", bool(future),
          f"{earliest.date} 之后有 {len(future)} 条" if future else "构造不出，这条检查没有意义")

    raised = False
    try:
        assert_no_future_evidence(query_event, [future[0]] if future else [])
    except AsOfViolation:
        raised = True
    check("喂进未来证据时抛 AsOfViolation", raised)

    # 同日事件也算未来：查询当日还不知道当天另一件事，不能当证据
    same_day_pairs = [
        (event, other) for event in events for other in events
        if event.event_id != other.event_id and as_day(event.date) == as_day(other.date)
    ]
    check("库里存在同日多事件，可用来测同日拦截", bool(same_day_pairs),
          f"{len(same_day_pairs)} 对")
    same_day_blocked = False
    try:
        if same_day_pairs:
            assert_no_future_evidence(same_day_pairs[0][0], [same_day_pairs[0][1]])
    except AsOfViolation:
        same_day_blocked = True
    check("同日事件被拦下（不算「已发生」）",
          same_day_blocked if same_day_pairs else True)

    # ---- 3. 落盘工作表的逐行核对 ----
    print("\n3. 标注工作表逐行核对")
    if not LABELS_PATH.exists():
        print("     工作表还没生成，跳过（先跑 scripts/run_retrieval.py）")
    else:
        with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        bad = [row for row in rows
               if row["candidate_date"] >= row["query_date"]]
        check(f"{len(rows)} 行候选全部早于对应查询日", not bad,
              f"{len(bad)} 行越界，例如 {bad[0]['candidate_event_id']}" if bad else "")
        # 列名在第 4 阶段的迁移里从 candidate_id 改成了 candidate_event_id
        check("候选里没有查询事件自己",
              all(row["candidate_event_id"] != row["query_id"] for row in rows))

    # ---- 4. 两路检索同池不同序 ----
    print("\n4. E1 与 E2 是同池不同排序")
    embedder, note = build_embedder([event.text for event in events])
    if note:
        print(f"     ⚠ {note}")
    index = EventIndex(events, embedder, note=note)
    if not index.load_vectors_from_disk():
        index.build()

    mismatched, identical = 0, 0
    for query in queries[:8]:
        event = by_id[query.event_id]
        # 和检索默认走的是同一个池（带 20 日标签成熟协议）
        pool = as_of_pool(events, event, horizon=20)
        e1 = retrieve(event, events, index, mode=MODE_SEMANTIC, top_k=10, pool=pool)
        e2 = retrieve(event, events, index, mode=MODE_FINANCIAL, top_k=10, pool=pool)
        if {h.event_id for h in e1} == {h.event_id for h in e2}:
            identical += 1
        ranks = [h.rank for h in e2]
        if ranks != list(range(1, len(ranks) + 1)):
            mismatched += 1
    check("E2 的排名连续从 1 开始", mismatched == 0)
    check("E1 与 E2 的排序**不完全相同**（否则重排序没起作用）", identical < 8,
          f"抽查 8 条查询里有 {identical} 条两路完全一样")

    # ---- 5. 查询集与事件库一致 ----
    print("\n5. 查询集")
    missing = [q.event_id for q in queries if q.event_id not in by_id]
    check("查询都在事件库里", not missing, f"缺 {missing}" if missing else "")
    outside = [q.event_id for q in queries
               if as_day(by_id[q.event_id].date).year < 2023]
    check("查询都来自评测集（2023 年及以后）", not outside,
          f"{len(outside)} 条不在评测期" if outside else "")
    check("查询市场状态完整",
          all(by_id[q.event_id].market_state_complete for q in queries))

    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败：{name}")
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
