"""构造 walk-forward 伪查询训练集（第 5 阶段）。

    .\\venv\\Scripts\\python.exe scripts\\build_pseudo_queries.py

把 2018–2022 的每一条历史事件当成「当时的一条新查询」：

1. 按**标签成熟协议**取候选池（`pos + horizon + 1 < query_pos`）；
2. 分别跑 E1 和 E2；
3. 按交易日聚合得到各自的点预测与冲突诊断；
4. 记录查询自己的真实异常收益作为**训练目标**（绝不进特征）。

## 为什么要重跑而不是复用已有评测

已有的 20 条查询是 2023–2024 的，而且已经被看过，不能当训练集。
伪查询是**另外**构造的：2018–2022，每条事件都是查询，池子是它之前的成熟事件。

## 门槛

`min_mature_candidates = 20`：成熟候选少于这个数的伪查询直接跳过并记原因，
**不静默补未来数据**。2018 年上半年大量样本会因此被跳过——那时候语料还很少，
这正是 walk-forward 应有的样子。
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.conflict import analyse, realized_abret  # noqa: E402
from src.aegisfx.conflict.metrics import robust_scale  # noqa: E402
from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.data.maturity import MaturityCalendar  # noqa: E402
from src.aegisfx.data.market import market_state_on, paper_market  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import EventIndex  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_of_pool,
    retrieve_detailed,
)
from src.aegisfx.routing.features import FEATURE_NAMES, FeatureBuilder  # noqa: E402
from src.aegisfx.schemas.views import RetrievalEventView  # noqa: E402

OUT_DIR = _PROJECT_ROOT / "data" / "aegisfx" / "routing"
REPORT_PATH = _PROJECT_ROOT / "reports" / "stage5" / "pseudo_queries.json"

TRAIN_YEARS = range(2018, 2023)
HORIZONS = (5, 20)
MIN_MATURE_CANDIDATES = 20
EVIDENCE_TOP_K = 10

CSV_FIELDS = (
    "query_id", "query_date", "horizon", "event_family", "initiating_side",
    "sentiment", "pool_size", "dropped_immature",
    "e1_prediction", "e2_prediction", "e1_direction", "e2_direction",
    "realized_abret", "scale",
    *FEATURE_NAMES,
    "ci_e1", "ci_e2", "neutral_mass_e1", "neutral_mass_e2",
    "pd_norm_e1", "pd_norm_e2", "ess_e1", "ess_e2",
    "candidate_ids_e1", "candidate_ids_e2",
)


def _expert(query_event, events, index, pool, mode, horizon):
    """跑一个专家，返回 (点预测, 冲突报告, 证据, 语义分列表)。"""
    outcome = retrieve_detailed(query_event, events, index, mode=mode,
                                top_k=EVIDENCE_TOP_K, pool=pool, horizon=horizon)
    result = analyse(query_event, list(outcome.evidence), pool, horizon=horizon,
                     retrieval_mode=mode)
    semantic = [hit.components.get("semantic", 0.0) for hit in outcome.evidence]
    return result, outcome, semantic


def build(horizon: int, events, index, builder: FeatureBuilder,
          calendar: MaturityCalendar, verbose: bool = True) -> tuple:
    rows: List[dict] = []
    skipped: List[dict] = []
    by_year = Counter()
    by_family = Counter()

    for event in events:
        if event.date.year not in TRAIN_YEARS:
            continue
        target = realized_abret(event, horizon)
        if target is None:
            skipped.append({"event_id": event.event_id,
                            "date": event.date.date().isoformat(),
                            "reason": "该期限的收益算不出来（窗口超出数据末端）"})
            continue

        position = calendar.position(event.date)
        pool = as_of_pool(events, event, horizon=horizon, calendar=calendar)
        dropped = 0
        if position is not None:
            # 「早于查询日」的条数减去成熟池，就是被成熟协议剔除的
            before = sum(1 for other in events
                         if other.event_id != event.event_id
                         and pd.Timestamp(other.date) < pd.Timestamp(event.date))
            dropped = before - len(pool)

        if len(pool) < MIN_MATURE_CANDIDATES:
            skipped.append({
                "event_id": event.event_id, "date": event.date.date().isoformat(),
                "reason": f"成熟候选只有 {len(pool)} 条 < {MIN_MATURE_CANDIDATES}",
                "mature_candidates": len(pool)})
            continue

        r1, outcome1, semantic1 = _expert(event, events, index, pool,
                                          MODE_SEMANTIC, horizon)
        r2, outcome2, _ = _expert(event, events, index, pool,
                                  MODE_FINANCIAL, horizon)

        if r1.predicted_abnormal_return is None or r2.predicted_abnormal_return is None:
            skipped.append({"event_id": event.event_id,
                            "date": event.date.date().isoformat(),
                            "reason": "至少一个专家给不出点预测"})
            continue

        view = RetrievalEventView.from_event(event, include_realized=False)
        scale = robust_scale([realized_abret(other, horizon) for other in pool])
        if scale <= 1e-12:
            scale = float(np.std([realized_abret(o, horizon) for o in pool
                                  if realized_abret(o, horizon) is not None], ddof=1))

        features = builder.build(
            view,
            same_family_ratio=outcome1.same_family_ratio,
            semantic_top=semantic1,
            mature_count=len(pool),
            conflict_e1=r1.conflict.model_dump(),
            conflict_e2=r2.conflict.model_dump(),
        )

        row = {
            "query_id": event.event_id,
            "query_date": event.date.date().isoformat(),
            "horizon": horizon,
            "event_family": event.event_family,
            "initiating_side": event.initiating_side,
            "sentiment": event.sentiment,
            "pool_size": len(pool),
            "dropped_immature": dropped,
            "e1_prediction": r1.predicted_abnormal_return,
            "e2_prediction": r2.predicted_abnormal_return,
            "e1_direction": r1.predicted_direction.value,
            "e2_direction": r2.predicted_direction.value,
            "realized_abret": target,
            "scale": scale,
            "ci_e1": r1.conflict.directional_conflict,
            "ci_e2": r2.conflict.directional_conflict,
            "neutral_mass_e1": r1.conflict.neutral_mass,
            "neutral_mass_e2": r2.conflict.neutral_mass,
            "pd_norm_e1": r1.conflict.pairwise_dispersion_norm,
            "pd_norm_e2": r2.conflict.pairwise_dispersion_norm,
            "ess_e1": r1.conflict.effective_sample_size,
            "ess_e2": r2.conflict.effective_sample_size,
            "candidate_ids_e1": "|".join(h.event_id for h in outcome1.evidence),
            "candidate_ids_e2": "|".join(h.event_id for h in outcome2.evidence),
            **features,
        }
        rows.append(row)
        by_year[event.date.year] += 1
        by_family[event.event_family] += 1

        if verbose and len(rows) % 50 == 0:
            print(f"    h={horizon} 已构造 {len(rows)} 条")

    return rows, skipped, {"by_year": dict(by_year), "by_family": dict(by_family)}


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="构造伪查询训练集")
    parser.add_argument("--force", action="store_true", help="已有文件也重建")
    args = parser.parse_args()

    existing = [OUT_DIR / f"pseudo_queries_h{h}.csv" for h in HORIZONS]
    if all(path.exists() for path in existing) and not args.force:
        print("伪查询文件已存在。要重建加 --force。")
        return 0

    events = load_event_store()
    calendar = MaturityCalendar.from_market()
    embedder, note = build_embedder([event.text for event in events])
    if note:
        print(f"⚠ {note}")
    index = EventIndex(events, embedder, note=note)
    if not index.load_vectors_from_disk():
        index.build()

    builder = FeatureBuilder(paper_market())
    report = {"horizons": {}, "min_mature_candidates": MIN_MATURE_CANDIDATES,
              "train_years": [min(TRAIN_YEARS), max(TRAIN_YEARS)]}

    print("=" * 78)
    print("构造 walk-forward 伪查询")
    print("=" * 78)
    print(f"训练年份 {min(TRAIN_YEARS)}–{max(TRAIN_YEARS)}　"
          f"最小成熟候选 {MIN_MATURE_CANDIDATES}　事件总数 {len(events)}")

    for horizon in HORIZONS:
        print(f"\n--- 期限 {horizon} 日 ---")
        rows, skipped, counts = build(horizon, events, index, builder, calendar)
        write_csv(rows, OUT_DIR / f"pseudo_queries_h{horizon}.csv")

        reasons = Counter(item["reason"].split("（")[0] for item in skipped)
        print(f"  有效伪查询 {len(rows)} 条，跳过 {len(skipped)} 条")
        for reason, count in reasons.most_common():
            print(f"    跳过原因：{reason} —— {count} 条")
        print(f"  按年：{dict(sorted(counts['by_year'].items()))}")
        print(f"  按事件族：{dict(counts['by_family'])}")

        report["horizons"][horizon] = {
            "n_rows": len(rows), "n_skipped": len(skipped),
            "skip_reasons": dict(reasons),
            **counts,
            "dropped_immature_total": int(sum(r["dropped_immature"] for r in rows)),
            "pool_size_range": ([min(r["pool_size"] for r in rows),
                                 max(r["pool_size"] for r in rows)] if rows else []),
            "skipped_sample": skipped[:30],
        }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\n报告已写入 {REPORT_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"训练表已写入 {OUT_DIR.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
