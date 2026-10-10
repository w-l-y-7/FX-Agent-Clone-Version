"""建立事件向量索引与查询集（第 3 周）。

    .\\venv\\Scripts\\python.exe scripts\\build_retrieval_index.py
    .\\venv\\Scripts\\python.exe scripts\\build_retrieval_index.py --rebuild-queries

第一次跑要下载约 95MB 的中文向量模型（走镜像站，几十秒）。
模型下不下来会自动退回 TF-IDF 兜底，会打印提示。

**查询集默认不重建**——人工标注是绑在 event_id 上的，重建会让标注失配。
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import INDEX_DIR, EventIndex, load_manifest  # noqa: E402
from src.aegisfx.retrieval.queries import QUERY_PATH, load_or_build  # noqa: E402
from src.aegisfx.taxonomy import family_label, label  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="建立事件向量索引与查询集")
    parser.add_argument("--force", action="store_true", help="强制重建索引")
    parser.add_argument("--rebuild-queries", action="store_true",
                        help="重建查询集（会让已有的人工标注失配，慎用）")
    args = parser.parse_args()

    events = load_event_store()
    if not events:
        print("事件库是空的。先跑 scripts/build_event_store.py")
        return 1

    print("=" * 78)
    print("建立事件向量索引")
    print("=" * 78)
    print(f"事件 {len(events)} 条")

    embedder, note = build_embedder([event.text for event in events])
    if note:
        print(f"  ⚠ {note}")
    print(f"  embedder  {embedder.name}（{embedder.dim} 维）")

    index = EventIndex(events, embedder, note=note)
    rebuilt = index.build(force=args.force)
    print(f"  {'已重建' if rebuilt else '指纹未变，跳过重建'}"
          f" → {INDEX_DIR.relative_to(_PROJECT_ROOT)}")

    manifest = load_manifest()
    if manifest:
        print(f"  清单：{manifest['events']} 条事件，{manifest['dim']} 维，"
              f"指纹 {manifest['fingerprint'][:12]}")

    # ---- 查询集 ----
    print("\n" + "=" * 78)
    print("检索评测查询集")
    print("=" * 78)
    queries = load_or_build(events, rebuild=args.rebuild_queries)
    print(f"共 {len(queries)} 条 → {QUERY_PATH.relative_to(_PROJECT_ROOT)}"
          f"{'（本次重建）' if args.rebuild_queries else '（沿用已有）'}")

    by_family = {}
    for query in queries:
        by_family.setdefault(query.family, []).append(query)
    print()
    print("%-16s %6s %10s %12s" % ("事件族", "条数", "日期范围", "池内同族数"))
    for family, items in sorted(by_family.items()):
        dates = [item.date for item in items]
        pools = [item.pool_size for item in items]
        print("%-16s %6d %10s %12s" % (
            family_label(family), len(items),
            f"{min(dates)[2:]}~{max(dates)[2:]}",
            f"{min(pools)}–{max(pools)}",
        ))

    print("\n逐条")
    for query in queries:
        print(f"  {query.date}  {label(query.event_type):<22} "
              f"{query.initiating_side:<6} 池内同族 {query.pool_size:>3}  "
              f"{query.text[:34]}")

    print("\n下一步：scripts/run_retrieval.py 跑出候选集，然后人工标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
