"""跑检索：E1 纯语义 vs E2 金融重排序，产出人工标注工作表。

    # 1) 生成/更新工作表（已有标注会保留）
    .\\venv\\Scripts\\python.exe scripts\\run_retrieval.py

    # 2) 人工把 data/aegisfx/retrieval/labels.csv 的 relevance 列填上 0–3

    # 3) 算指标
    .\\venv\\Scripts\\python.exe scripts\\run_retrieval.py --report

    # 4) 让大模型当第二标注员，报告与人工的一致率
    .\\venv\\Scripts\\python.exe scripts\\run_retrieval.py --llm-label

## 标注规范

给每条候选打 0–3：

| 分 | 含义 |
|---:|---|
| 3 | **高度相关**：同一事件族 + 同一发起方 + 政策方向相同，是可直接类比的历史先例 |
| 2 | **相关**：同一事件族，但发起方或政策方向不同（美方加征 vs 中方加征；加征 vs 放宽） |
| 1 | **弱相关**：不同事件族，但同属中美经贸摩擦、能提供背景 |
| 0 | **不相关**：主题不同，或时间上无参考价值 |

候选池是 E1 与 E2 各自 Top-10 的**并集**（TREC 式的池化）：
只在池内算 Recall，所以 Recall 是**下界**——池外可能还有相关事件没被任何一路检出来。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.paths import load_env  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import EventIndex  # noqa: E402
from src.aegisfx.retrieval.metrics import (  # noqa: E402
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.aegisfx.retrieval.queries import QUERY_DIR, load_or_build  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_of_pool,
    retrieve,
)
from src.aegisfx.taxonomy import family_label, label  # noqa: E402

LABELS_PATH = QUERY_DIR / "labels.csv"
PREVIEW_PATH = QUERY_DIR / "labeling_preview.md"
REPORT_PATH = _PROJECT_ROOT / "reports" / "aegisfx_retrieval.json"
LLM_LABELS_PATH = QUERY_DIR / "labels_llm.csv"

TOP_K = 10
K_VALUES = (3, 5, 10)

_CSV_FIELDS = [
    "query_id", "candidate_id", "query_date", "query_family",
    "candidate_date", "candidate_family", "candidate_type",
    "candidate_side", "candidate_sentiment",
    "e1_rank", "e2_rank", "relevance", "note",
    # 事件原文也写进表里。查询原文每条候选重复一遍确实冗余，但换来的是
    # **标注时不用再去事件库里查**——不管是打开表格读，还是用标注页。
    "query_text", "candidate_text", "pool_size",
]


def _payload(row):
    payload = {name: row.get(name, "") for name in _CSV_FIELDS}
    payload["query_text"] = row.get("_query_text", "") or payload.get("query_text", "")
    payload["candidate_text"] = (row.get("_candidate_text", "")
                                 or payload.get("candidate_text", ""))
    payload["pool_size"] = row.get("pool_size", "")
    return payload


def _run_all(events, index, queries):
    """对每条查询跑两路检索，返回池化后的候选。"""
    rows = []
    for query_event_id in [q.event_id for q in queries]:
        query = next(event for event in events if event.event_id == query_event_id)
        pool = as_of_pool(events, query)
        e1 = retrieve(query, events, index, mode=MODE_SEMANTIC, top_k=TOP_K, pool=pool)
        e2 = retrieve(query, events, index, mode=MODE_FINANCIAL, top_k=TOP_K, pool=pool)

        e1_rank = {hit.event_id: hit.rank for hit in e1}
        e2_rank = {hit.event_id: hit.rank for hit in e2}
        ordered = sorted(
            set(e1_rank) | set(e2_rank),
            key=lambda eid: (min(e1_rank.get(eid, 99), e2_rank.get(eid, 99)),
                             e2_rank.get(eid, 99)),
        )
        for candidate_id in ordered:
            candidate = next(event for event in events if event.event_id == candidate_id)
            rows.append({
                "query_id": query.event_id,
                "candidate_id": candidate_id,
                "query_date": query.date.date().isoformat(),
                "query_family": query.event_family,
                "candidate_date": candidate.date.date().isoformat(),
                "candidate_family": candidate.event_family,
                "candidate_type": candidate.event_type,
                "candidate_side": candidate.initiating_side,
                "candidate_sentiment": candidate.sentiment,
                "e1_rank": e1_rank.get(candidate_id, ""),
                "e2_rank": e2_rank.get(candidate_id, ""),
                "relevance": "",
                "note": "",
                "pool_size": len(pool),
                "_query_text": query.text,
                "_candidate_text": candidate.text,
            })
    return rows


def _merge_existing_labels(rows):
    """已有的标注保留下来，重跑不会把人工填的分数冲掉。"""
    if not LABELS_PATH.exists():
        return rows, 0
    existing = {}
    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            existing[(record["query_id"], record["candidate_id"])] = (
                record.get("relevance", ""), record.get("note", ""))
    kept = 0
    for row in rows:
        found = existing.get((row["query_id"], row["candidate_id"]))
        if found and found[0].strip():
            row["relevance"], row["note"] = found
            kept += 1
    return rows, kept


def _migrated_already() -> bool:
    """labels.csv 是否已经是第 4 阶段的结构。"""
    if not LABELS_PATH.exists():
        return False
    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    return "required_for_primary" in header


def refuse_if_migrated(force: bool) -> bool:
    """已经迁移过的表不让覆盖。

    这个脚本产出的是**原始候选**（只有 e1_rank/e2_rank 那几列），
    而 `migrate_labels.py` 在它之上算了 in_e1_top5 / required_for_primary /
    同族同发起方这类派生列。直接重写会把那些列和查询集的映射一起抹掉。
    """
    if force or not _migrated_already():
        return True
    print(f"{LABELS_PATH.relative_to(_PROJECT_ROOT)} 已经是第 4 阶段的结构，"
          "重写会丢掉迁移算出来的派生列。")
    print("  通常你不需要重跑本脚本——要合并结构请用：")
    print("    python scripts/migrate_labels.py")
    print("  确实要重新生成候选的话，加 --force。")
    return False


def write_worksheet(rows, kept):
    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_payload(row))

    print(f"工作表已写入 {LABELS_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"  共 {len(rows)} 行，其中已有标注 {kept} 行（重跑会保留）")

    # 阅读版：按查询分组，方便一口气读完再打分
    lines = ["# 检索标注工作表", "",
             "给每条候选打 0–3。规范见 `scripts/run_retrieval.py` 的模块说明。", ""]
    current = None
    for row in rows:
        if row["query_id"] != current:
            current = row["query_id"]
            lines += [
                f"## {row['query_date']}　{row['query_id']}",
                f"**查询**：[{family_label(row['query_family'])}] {row['_query_text']}",
                f"候选池大小 {row['pool_size']}", "",
                "| 候选 | 日期 | 事件族 | 发起方 | 情感 | E1 | E2 | 分 |",
                "|---|---|---|---|---|---:|---:|---:|",
            ]
        lines.append(
            f"| {row['_candidate_text'][:44]} | {row['candidate_date']} "
            f"| {family_label(row['candidate_family'])} | {row['candidate_side']} "
            f"| {row['candidate_sentiment']} | {row['e1_rank'] or '—'} "
            f"| {row['e2_rank'] or '—'} | |")
    lines.append("")
    PREVIEW_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"阅读版已写入 {PREVIEW_PATH.relative_to(_PROJECT_ROOT)}")


def load_labels(path=LABELS_PATH):
    """返回 {query_id: {candidate_id: grade}}，只取填了分的行。"""
    if not path.exists():
        return {}
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            raw = (record.get("relevance") or "").strip()
            if not raw:
                continue
            try:
                grade = int(float(raw))
            except ValueError:
                continue
            labels.setdefault(record["query_id"], {})[record["candidate_id"]] = grade
    return labels


def report(rows, labels, source: str = "人工") -> dict:
    """按查询算指标。`rows` 必须和标注时的候选池一致。"""
    by_query = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    results = {}
    for query_id, candidates in by_query.items():
        relevance = labels.get(query_id, {})
        if not relevance:
            continue
        e1 = [r["candidate_id"] for r in sorted(
            [c for c in candidates if c["e1_rank"] != ""],
            key=lambda r: int(r["e1_rank"]))]
        e2 = [r["candidate_id"] for r in sorted(
            [c for c in candidates if c["e2_rank"] != ""],
            key=lambda r: int(r["e2_rank"]))]
        results[query_id] = {
            "family": candidates[0]["query_family"],
            "n_labeled": len(relevance),
            "n_relevant": sum(1 for g in relevance.values() if g >= 2),
            "E1": {**{f"precision@{k}": precision_at_k(e1, relevance, k) for k in K_VALUES},
                   **{f"recall@{k}": recall_at_k(e1, relevance, k) for k in K_VALUES},
                   **{f"ndcg@{k}": ndcg_at_k(e1, relevance, k) for k in K_VALUES}},
            "E2": {**{f"precision@{k}": precision_at_k(e2, relevance, k) for k in K_VALUES},
                   **{f"recall@{k}": recall_at_k(e2, relevance, k) for k in K_VALUES},
                   **{f"ndcg@{k}": ndcg_at_k(e2, relevance, k) for k in K_VALUES}},
        }

    if not results:
        print(f"还没有{source}标注，算不出指标。")
        return {}

    print(f"\n{'=' * 78}\n检索指标（{source}标注，{len(results)} 条查询）\n{'=' * 78}")
    header = f"{'指标':<12}{'E1 纯语义':>14}{'E2 金融重排序':>16}{'变化':>10}"
    print(header)
    print("-" * len(header))
    summary = {}
    for metric in [f"{name}@{k}"
                   for name in ("precision", "recall", "ndcg")
                   for k in K_VALUES]:
        e1 = mean([r["E1"][metric] for r in results.values()])
        e2 = mean([r["E2"][metric] for r in results.values()])
        summary[metric] = {"E1": e1, "E2": e2, "delta": e2 - e1}
        mark = "  ↑" if e2 > e1 + 1e-9 else ("  ↓" if e2 < e1 - 1e-9 else "  =")
        print(f"{metric:<12}{e1:>14.4f}{e2:>16.4f}{e2 - e1:>+9.4f}{mark}")

    per_family = {}
    for query_id, item in results.items():
        per_family.setdefault(item["family"], []).append(item)
    print(f"\n{'事件族':<16}{'查询数':>7}{'NDCG@5 E1':>14}{'NDCG@5 E2':>14}")
    for family, items in sorted(per_family.items()):
        print(f"{family_label(family):<16}{len(items):>7}"
              f"{mean([i['E1']['ndcg@5'] for i in items]):>14.4f}"
              f"{mean([i['E2']['ndcg@5'] for i in items]):>14.4f}")

    return {"n_queries": len(results), "summary": summary,
            "per_query": results, "source": source}


def llm_label(rows, model_note=""):
    """大模型当第二标注员。**只看文本和元数据，看不到人工分数。**"""
    import os

    from src.services.deepseek_llm_service import DeepSeekLLMService

    service = DeepSeekLLMService(api_key=os.environ["DEEPSEEK_API_KEY"])
    rubric = (
        "你在给「历史事件对当前事件有多大参考价值」打分，用 0–3：\n"
        "3 = 同一事件族 + 同一发起方 + 政策方向相同，可直接类比；\n"
        "2 = 同一事件族，但发起方或政策方向不同；\n"
        "1 = 不同事件族，但同属中美经贸摩擦；\n"
        "0 = 主题不同，或时间上无参考价值。\n"
        "只回一个数字。"
    )

    from pydantic import BaseModel, Field

    class _Grade(BaseModel):
        grade: int = Field(description="0 到 3 的整数")

    out = []
    for row in rows:
        prompt = (f"{rubric}\n\n【当前事件】{row['_query_text']}\n"
                  f"【历史候选】{row['_candidate_text']}\n\n分数：")
        try:
            grade = service.invoke_structured(prompt, _Grade).grade
            grade = max(0, min(3, int(grade)))
        except Exception:
            grade = ""
        out.append(grade)

    with LLM_LABELS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row, grade in zip(rows, out):
            payload = _payload(row)
            payload["relevance"] = grade
            payload["note"] = "llm"
            writer.writerow(payload)
    print(f"大模型标注已写入 {LLM_LABELS_PATH.relative_to(_PROJECT_ROOT)}")
    return LLM_LABELS_PATH


def agreement(rows, human, machine, source: str):
    """两条标注的一致率。"""
    pairs = []
    for row in rows:
        key = (row["query_id"], row["candidate_id"])
        h = human.get(row["query_id"], {}).get(row["candidate_id"])
        m = machine.get(row["query_id"], {}).get(row["candidate_id"])
        if h is None or m is None:
            continue
        pairs.append((h, m))
    if not pairs:
        print(f"没有可比较的{source}标注对。")
        return {}
    exact = sum(1 for h, m in pairs if h == m) / len(pairs)
    # 二值口径（>=2 算相关）上的一致率，对「相关/不相关」这个实际判断更关键
    binary = sum(1 for h, m in pairs if (h >= 2) == (m >= 2)) / len(pairs)
    print(f"\n与{source}的一致性（n={len(pairs)}）")
    print(f"  完全相同      {exact:.1%}")
    print(f"  二值口径一致   {binary:.1%}   （≥2 算相关）")
    return {"n": len(pairs), "exact": exact, "binary": binary}


def main() -> int:
    parser = argparse.ArgumentParser(description="跑检索并产出标注工作表")
    parser.add_argument("--report", action="store_true", help="只算指标，不重跑检索")
    parser.add_argument("--llm-label", action="store_true", help="让大模型当第二标注员")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH,
                        help="用哪份标注算指标（默认人工的 labels.csv，"
                             "给 labels_llm.csv 就看大模型标注的结果）")
    parser.add_argument("--force", action="store_true",
                        help="即使 labels.csv 已经迁移过也重写（会丢失派生列）")
    args = parser.parse_args()

    if not args.report and not refuse_if_migrated(args.force):
        return 1

    load_env()
    events = load_event_store()
    queries = load_or_build(events)

    # --report 只想算指标，不该顺手把工作表重写一遍；
    # 但 --llm-label 要用到事件原文，那就必须真跑一遍检索
    if args.report and LABELS_PATH.exists() and not args.llm_label:
        rows = _cached_rows()
    else:
        embedder, note = build_embedder([event.text for event in events])
        if note:
            print(f"⚠ {note}")
        index = EventIndex(events, embedder, note=note)
        if not index.load_vectors_from_disk():
            index.build()
        rows = _run_all(events, index, queries)
        if not args.report:
            rows, kept = _merge_existing_labels(rows)
            write_worksheet(rows, kept)

    human = load_labels(args.labels)
    source = "大模型" if args.labels != LABELS_PATH else "人工"
    payload = report(rows, human, source=source)

    if args.llm_label:
        llm_label(rows)
        machine = load_labels(LLM_LABELS_PATH)
        if payload:
            agreement(rows, human, machine, "人工")

    if payload:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.labels == LABELS_PATH else "_llm"
        out = REPORT_PATH.with_name(f"aegisfx_retrieval{suffix}.json")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n报告已写入 {out.relative_to(_PROJECT_ROOT)}")
    else:
        print(f"\n下一步：把 {LABELS_PATH.relative_to(_PROJECT_ROOT)} 的 relevance 列"
              f"填上 0–3，然后跑 --report")
    return 0


def _cached_rows():
    """从工作表本身重建 rows。

    只够算指标用（`report` 只读 query_id / candidate_id / e1_rank / e2_rank /
    query_family，这些都在表里）。事件原文不在表里，所以 `--llm-label`
    走的是重新检索那条路，不走这里。
    """
    rows = []
    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            record["_query_text"] = record.get("query_text", "")
            record["_candidate_text"] = record.get("candidate_text", "")
            rows.append(record)
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
