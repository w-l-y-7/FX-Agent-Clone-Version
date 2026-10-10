"""标注表迁移：把 labels.csv 升到第 4 阶段的新结构。

    # 先看会改成什么样，不落盘
    .\\venv\\Scripts\\python.exe scripts\\migrate_labels.py --dry-run

    # 真正落盘
    .\\venv\\Scripts\\python.exe scripts\\migrate_labels.py

## 为什么可以安全地重复跑

- **不删行。** 主评测范围收紧到 E1/E2 各自 Top-5 的并集，但 Top-6~10 的行
  仍然留着，只是 `required_for_primary=False`。万一以后要看全部候选，数据还在。
- **不覆盖人工填过的值。** 迁移只写「程序能算的字段」和「原来就空的字段」；
  任何非空的 `mechanism_relevance` / `overall_relevance` / `label_note`
  一律原样带过去。
- **幂等。** 第二次跑读到的已经是新结构，列名映射是空的，写出来的文件一致。

## 列名变化

| 旧 | 新 |
|---|---|
| `candidate_id` | `candidate_event_id` |
| `relevance` | `overall_relevance` |
| `note` | `label_note` |

旧列名会保留成同值的备份列，不丢数据。

## 为什么把主评测范围改成 Top-5 并集

原计划标「前 10 条查询的 Top-10」，那会造成**查询选择偏差**——
被选中的 10 条和剩下的 10 条如果性质不同，结论就只对前 10 条成立。
改成「全部 20 条查询的 Top-5 并集」，行数差不多（约 150–180），
但覆盖全部查询和全部事件族，而且直接服务 NDCG@5。
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import load_event_store  # noqa: E402

QUERY_DIR = _PROJECT_ROOT / "data" / "aegisfx" / "retrieval"
LABELS_PATH = QUERY_DIR / "labels.csv"

# 新的列顺序。人工要填的只有 mechanism_relevance 和 overall_relevance。
FIELDS = [
    "query_id", "candidate_event_id",
    "query_date", "query_family", "query_text",
    "candidate_date", "candidate_family", "candidate_type",
    "candidate_side", "candidate_sentiment", "candidate_text",
    "e1_rank", "e2_rank",
    "in_e1_top5", "in_e2_top5", "required_for_primary",
    "same_family_auto", "same_side_auto", "same_sentiment_auto",
    "mechanism_relevance", "overall_relevance", "label_note", "label_status",
    "pool_size", "same_day_peers_auto",
    # 旧列名的备份，避免迁移时丢东西
    "legacy_candidate_id", "legacy_relevance", "legacy_note",
]

RENAME = {"candidate_id": "candidate_event_id",
          "relevance": "overall_relevance",
          "note": "label_note"}

# 这些列**只**在原本为空时才写，绝不覆盖人工内容
HUMAN_FIELDS = ("mechanism_relevance", "overall_relevance", "label_note")

TOP_FOR_PRIMARY = 5


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y", "是")


def _rank_at_most(value, limit: int) -> bool:
    raw = str(value).strip()
    if not raw:
        return False
    try:
        return int(float(raw)) <= limit
    except ValueError:
        return False


def migrate(rows: List[Dict[str, str]], events: Dict[str, object]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        legacy_id = row.get("candidate_id") or row.get("candidate_event_id", "")
        legacy_relevance = row.get("relevance", "")
        legacy_note = row.get("note", "")

        query = events.get(row.get("query_id", ""))
        candidate = events.get(legacy_id)

        in_e1 = _rank_at_most(row.get("e1_rank"), TOP_FOR_PRIMARY)
        in_e2 = _rank_at_most(row.get("e2_rank"), TOP_FOR_PRIMARY)

        same_family = (bool(query) and bool(candidate)
                       and query.event_family == candidate.event_family)
        same_side = (bool(query) and bool(candidate)
                     and query.initiating_side == candidate.initiating_side)
        same_sentiment = (bool(query) and bool(candidate)
                          and query.sentiment == candidate.sentiment)

        # 人工列：已经有值就原样留着，空着才用旧列名顶上
        human = {}
        for field in HUMAN_FIELDS:
            existing = (row.get(field) or "").strip()
            if existing:
                human[field] = existing
        human.setdefault("overall_relevance", (legacy_relevance or "").strip())
        human.setdefault("label_note", (legacy_note or "").strip())
        human.setdefault("mechanism_relevance", "")

        overall = human["overall_relevance"].strip()
        mechanism = human["mechanism_relevance"].strip()
        if overall and mechanism:
            status = "complete"
        elif overall or mechanism:
            status = "partial"
        else:
            status = "todo"

        out.append({
            **{name: row.get(name, "") for name in FIELDS},
            "query_id": row.get("query_id", ""),
            "candidate_event_id": legacy_id,
            "in_e1_top5": str(in_e1).lower(),
            "in_e2_top5": str(in_e2).lower(),
            "required_for_primary": str(in_e1 or in_e2).lower(),
            "same_family_auto": str(same_family).lower(),
            "same_side_auto": str(same_side).lower(),
            "same_sentiment_auto": str(same_sentiment).lower(),
            "same_day_peers_auto": str(
                getattr(candidate, "same_day_peers", "") if candidate else ""),
            "mechanism_relevance": mechanism,
            "overall_relevance": overall,
            "label_note": human["label_note"],
            "label_status": status,
            "legacy_candidate_id": legacy_id,
            "legacy_relevance": legacy_relevance,
            "legacy_note": legacy_note,
        })
    return out


def write(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description="标注表迁移到第 4 阶段结构")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    parser.add_argument("--path", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"找不到 {args.path}。先跑 scripts/run_retrieval.py 生成候选表。")
        return 1

    with args.path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        old_fields = reader.fieldnames or []

    events = {event.event_id: event for event in load_event_store()}
    migrated = migrate(rows, events)

    already = set(FIELDS).issubset(set(old_fields))
    primary = [r for r in migrated if _as_bool(r["required_for_primary"])]
    kept_human = sum(1 for r in migrated
                     if (r["overall_relevance"] or "").strip()
                     or (r["mechanism_relevance"] or "").strip())

    print("=" * 78)
    print("标注表迁移")
    print("=" * 78)
    print(f"源文件        {args.path}")
    print(f"列             {len(old_fields)} → {len(FIELDS)}"
          f"{'（已经是新结构，本次为幂等重写）' if already else ''}")
    print(f"行数          {len(rows)} → {len(migrated)}（不删行）")
    print(f"主评测范围     {len(primary)} 行"
          f"（required_for_primary，即 E1/E2 各自 Top-{TOP_FOR_PRIMARY} 的并集）")
    print(f"其余          {len(migrated) - len(primary)} 行保留，供「全部记录」视图用")
    print(f"已有人工填写   {kept_human} 行（迁移不会覆盖）")

    queries = {r["query_id"] for r in primary}
    print(f"覆盖查询      {len(queries)} 条")

    if not primary:
        print("\n⚠ 主评测范围为空——检查 e1_rank / e2_rank 列是否有值。")

    if args.dry_run:
        print("\n--dry-run：没有写文件。")
        return 0

    write(migrated, args.path)
    print(f"\n已写入 {args.path}")
    print("再跑一次结果应当完全一致（幂等）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
