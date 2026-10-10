"""检索评测：主题相关性 + 离线结果有用性，两套口径并报。

    .\\venv\\Scripts\\python.exe scripts\\evaluate_retrieval.py
    .\\venv\\Scripts\\python.exe scripts\\evaluate_retrieval.py --offline-only

## 两套口径为什么必须分开

| | 主题相关性 | 离线结果有用性 |
|---|---|---|
| 问的是 | 这条历史事件**读起来**像不像先例 | 这条先例的**汇率反应**能不能说明当前事件会怎样 |
| 标签来自 | 人工标注（0–3） | 事件库里的异常收益 |
| 能否在线使用 | 是 | **绝对不能** |

混成一个「检索质量」会掩盖真正的结论：实测下来两者**方向相反**——
语义检索在主题上占优，金融重排序在结果有用性上小胜。

`offline_outcome_utility` **使用查询事件发生后的真实结果，只用于离线回测，
不能用于在线排序**。为了在代码层面挡住这件事，在线检索走的是
`RetrievalEventView`（见 `src/aegisfx/schemas/views.py`），
排序函数结构上读不到未来字段。

## Recall 只是 pooled 口径

候选池是三个方法各自 Top-K 的并集，只在池内标注，所以池外的相关事件被当成
不相关——**pooled_recall 是真实召回的下界**。不要把它写成相对于整个事件库的召回。

## 配对 bootstrap 按查询做

20 条查询，在候选行级别抽样等于把同一条查询的信息重复计入。所以按**查询**
重抽样，固定种子。
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import EventIndex  # noqa: E402
from src.aegisfx.retrieval.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402
from src.aegisfx.retrieval.queries import load_or_build  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    MODE_ADAPTIVE,
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_of_pool,
    retrieve,
    retrieve_detailed,
)
from src.aegisfx.taxonomy import family_label  # noqa: E402

LABELS_PATH = _PROJECT_ROOT / "data" / "aegisfx" / "retrieval" / "labels.csv"
REPORT_PATH = _PROJECT_ROOT / "reports" / "aegisfx_retrieval.json"
SCATTER_PATH = _PROJECT_ROOT / "reports" / "aegisfx_retrieval_two_objective.json"

TOP_K = 10
K_VALUES = (3, 5, 10)
HORIZON_FIELD = "abret_20d"
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_RESAMPLES = 5000

MODES = (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE)
MODE_LABELS = {MODE_SEMANTIC: "E1-semantic",
               MODE_FINANCIAL: "E2-fixed-reranker",
               MODE_ADAPTIVE: "E3-adaptive-conflict-rag"}


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def paired_bootstrap(
    deltas: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> dict:
    """按查询配对重抽样的 bootstrap 区间。

    `deltas` 是**每条查询一个数**（新方法减基线的指标差），不是候选行级别的量。
    在候选行级别抽样会把同一条查询的信息重复计入。
    """
    values = np.asarray([d for d in deltas if d is not None], dtype=float)
    if values.size == 0:
        return {"n": 0}

    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = sample.mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "n_better": int((values > 1e-12).sum()),
        "n_worse": int((values < -1e-12).sum()),
        "n_tie": int((np.abs(values) <= 1e-12).sum()),
        "confidence": confidence,
        "seed": seed,
        "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------------------
# 标签读取
# ---------------------------------------------------------------------------


def load_human_labels(path: Path = LABELS_PATH):
    """读人工标注。返回 (主题标签, 机制标签, 统计)。"""
    if not path.exists():
        return {}, {}, {"rows": 0, "complete": 0, "primary": 0, "queries": 0}

    topic: Dict[str, Dict[str, int]] = {}
    mechanism: Dict[str, Dict[str, int]] = {}
    rows = complete = primary = 0
    queries = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for record in csv.DictReader(handle):
            rows += 1
            is_primary = str(record.get("required_for_primary", "")).lower() == "true"
            if is_primary:
                primary += 1
                queries.add(record["query_id"])
            if str(record.get("label_status", "")).strip() == "complete":
                complete += 1

            candidate = record.get("candidate_event_id") or record.get("candidate_id", "")
            raw = (record.get("overall_relevance") or "").strip()
            if raw:
                try:
                    topic.setdefault(record["query_id"], {})[candidate] = int(float(raw))
                except ValueError:
                    pass
            raw_mech = (record.get("mechanism_relevance") or "").strip()
            if raw_mech:
                try:
                    mechanism.setdefault(record["query_id"], {})[candidate] = int(float(raw_mech))
                except ValueError:
                    pass

    return topic, mechanism, {
        "rows": rows, "complete": complete, "primary": primary,
        "queries": len(queries),
    }


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def _rankings(events, index, queries, by_id, horizon: int):
    """三路方法各自的 Top-K，外加客观口径的相关性标签。"""
    scale = float(np.std([getattr(e, HORIZON_FIELD) for e in events
                          if getattr(e, HORIZON_FIELD) is not None], ddof=1))
    per_query = {}
    for spec in queries:
        event = by_id[spec.event_id]
        target = getattr(event, HORIZON_FIELD)
        if target is None:
            continue
        pool = as_of_pool(events, event)

        utility = {}
        for candidate in pool:
            value = getattr(candidate, HORIZON_FIELD)
            if value is None:
                continue
            utility[candidate.event_id] = _utility_grade(abs(value - target) / scale)

        ids = {}
        for mode in MODES:
            ids[mode] = [hit.event_id for hit in retrieve(
                event, events, index, mode=mode, top_k=TOP_K, pool=pool)]

        # 记下 E3 的自适应系数：α = 0 时 E3 的排序与 E1 完全相同，
        # 那些查询对「E3 有没有用」这个问题不提供信息，比较时要能识别出来
        adaptive = retrieve_detailed(event, events, index, mode=MODE_ADAPTIVE,
                                     top_k=TOP_K, pool=pool)

        per_query[spec.event_id] = {
            "family": event.event_family, "date": spec.date,
            "utility": utility, "ids": ids,
            "adaptive_alpha": adaptive.adaptive_alpha,
            "same_family_ratio": adaptive.same_family_ratio,
            "e3_equals_e1": ids[MODE_ADAPTIVE] == ids[MODE_SEMANTIC],
        }
    return per_query, scale


def _utility_grade(delta_in_sd: float) -> int:
    if delta_in_sd < 0.5:
        return 3
    if delta_in_sd < 1.0:
        return 2
    if delta_in_sd < 1.5:
        return 1
    return 0


def _metrics(ids: Sequence[str], relevance: Dict[str, int]) -> Dict[str, float]:
    out = {}
    for k in K_VALUES:
        out[f"precision@{k}"] = precision_at_k(ids, relevance, k)
        # pooled 口径：只在「三个方法 Top-K 的并集」里算召回，是真实召回的下界
        out[f"pooled_recall@{k}"] = recall_at_k(ids, relevance, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ids, relevance, k)
    return out


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def report_metrics(per_query, relevance_of, title: str, note: str) -> Optional[dict]:
    usable = {qid: entry for qid, entry in per_query.items()
              if relevance_of(qid)}
    if not usable:
        return None

    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if note:
        print(note)

    header = f"{'指标':<18}" + "".join(f"{MODE_LABELS[m]:>24}" for m in MODES) + f"{'E3−E1':>10}"
    print(header)
    print("-" * len(header))
    summary = {}
    for metric in [f"{n}@{k}" for n in ("precision", "pooled_recall", "ndcg")
                   for k in K_VALUES]:
        values = {}
        for mode in MODES:
            values[mode] = _mean([_metrics(entry["ids"][mode], relevance_of(qid))[metric]
                                  for qid, entry in usable.items()])
        delta = values[MODE_ADAPTIVE] - values[MODE_SEMANTIC]
        summary[metric] = {**{MODE_LABELS[m]: values[m] for m in MODES},
                           "E3_minus_E1": delta}
        print(f"{metric:<18}" + "".join(f"{values[m]:>24.4f}" for m in MODES)
              + f"{delta:>+10.4f}")

    by_family = {}
    for qid, entry in usable.items():
        by_family.setdefault(entry["family"], []).append(qid)
    print(f"\n{'事件族':<16}{'查询数':>7}" + "".join(f"{MODE_LABELS[m]:>24}" for m in MODES))
    for family, qids in sorted(by_family.items()):
        row = [MODE_LABELS[m] for m in MODES]
        cells = []
        for mode in MODES:
            cells.append(_mean([_metrics(usable[q]["ids"][mode],
                                         relevance_of(q))["ndcg@5"] for q in qids]))
        print(f"{family_label(family):<16}{len(qids):>7}"
              + "".join(f"{v:>24.4f}" for v in cells))
        summary.setdefault("by_family", {})[family] = dict(zip(row, cells))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评测：主题相关性 + 离线结果有用性")
    parser.add_argument("--offline-only", action="store_true",
                        help="只算离线口径，跳过人工标签门禁")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    events = load_event_store()
    queries = load_or_build(events)
    by_id = {event.event_id: event for event in events}

    embedder, note = build_embedder([event.text for event in events])
    if note:
        print(f"⚠ {note}")
    index = EventIndex(events, embedder, note=note)
    if not index.load_vectors_from_disk():
        index.build()

    per_query, scale = _rankings(events, index, queries, by_id, 20)

    print("=" * 78)
    print("检索评测")
    print("=" * 78)
    print(f"查询 {len(per_query)} 条　事件 {len(events)} 条　Top-K = {TOP_K}")
    print("方法：" + "　".join(f"{m} = {MODE_LABELS[m]}" for m in MODES))

    # ---- 离线结果有用性（客观口径）----
    utility_summary = report_metrics(
        per_query, lambda qid: per_query[qid]["utility"],
        "离线结果有用性 offline_outcome_utility",
        "⚠ 该指标使用查询事件发生后的真实结果，**只用于离线回测，不能用于在线排序**。")
    if utility_summary:
        print(f"\n折算单位：20 日异常收益的跨事件标准差 {scale:.4%}")

    # ---- 主题相关性（人工标注，带门禁）----
    topic, mechanism, stats = load_human_labels(args.labels)
    topic_summary = None
    if not args.offline_only:
        enough = stats["primary"] > 0 and stats["complete"] >= stats["primary"]
        if not enough:
            print(f"\n{'=' * 78}")
            print("主题相关性（人工标注）：**待完成**")
            print("=" * 78)
            print(f"  主评测范围 {stats['primary']} 行，已完成 {stats['complete']} 行")
            print(f"  覆盖查询 {stats['queries']} 条")
            print("  人工标注未完成，本项不出结论。")
            print("  —— 不允许用旧的大模型 0–3 整体评分顶替：那套评分已被判定为不可靠")
            print("     （给分近乎均分，且对同族同发起方同情感的两条候选给不同分）。")
            print(f"  标注入口：streamlit run app/Home.py → 「标注工具」页")
        else:
            topic_summary = report_metrics(
                per_query, lambda qid: topic.get(qid, {}),
                f"主题相关性（人工标注，{stats['queries']} 条查询 / {stats['complete']} 行）",
                "")
            mech_summary = report_metrics(
                per_query, lambda qid: mechanism.get(qid, {}),
                "机制相关性（人工标注，0–2）", "")
            if mech_summary:
                topic_summary["mechanism"] = mech_summary

    # ---- 查询级配对比较 ----
    # ---- E3 的自适应系数分布 ----
    alphas = [entry["adaptive_alpha"] for entry in per_query.values()
              if entry["adaptive_alpha"] is not None]
    identical = sum(1 for entry in per_query.values() if entry["e3_equals_e1"])
    print(f"\n{'=' * 78}\nE3 的自适应系数\n{'=' * 78}")
    if alphas:
        print(f"  α 范围 {min(alphas):.3f}–{max(alphas):.3f}　"
              f"中位 {float(np.median(alphas)):.3f}")
        print(f"  α = 0 的查询 {sum(1 for a in alphas if a <= 1e-12)} / {len(alphas)} 条"
              "（语义 Top-5 同族比例不到一半时金融重排被完全关闭）")
    print(f"  E3 排序与 E1 完全相同的查询：{identical} / {len(per_query)} 条")
    if identical:
        print("  这些查询对「E3 有没有用」不提供信息——E3 在那里的输出就是 E1 的输出。")

    print(f"\n{'=' * 78}\n查询级配对比较（按查询配对 bootstrap，种子 {BOOTSTRAP_SEED}）\n{'=' * 78}")
    comparisons = {}
    for label, source in (("offline_outcome_utility", lambda q: per_query[q]["utility"]),
                          ("topic", lambda q: topic.get(q, {}))):
        if label == "topic" and topic_summary is None:
            continue
        print(f"\n{label}　NDCG@5")
        for mode in (MODE_FINANCIAL, MODE_ADAPTIVE):
            deltas = []
            for qid, entry in per_query.items():
                relevance = source(qid)
                if not relevance:
                    continue
                base = ndcg_at_k(entry["ids"][MODE_SEMANTIC], relevance, 5)
                new = ndcg_at_k(entry["ids"][mode], relevance, 5)
                deltas.append(new - base)
            if not deltas:
                continue
            stat = paired_bootstrap(deltas)
            comparisons[f"{label}:{MODE_LABELS[mode]}_minus_E1"] = stat
            print(f"  {MODE_LABELS[mode]:<26} 均值 {stat['mean']:+.4f}　"
                  f"中位 {stat['median']:+.4f}　"
                  f"95% 区间 [{stat['low']:+.4f}, {stat['high']:+.4f}]　"
                  f"胜/负/平 {stat['n_better']}/{stat['n_worse']}/{stat['n_tie']}")
            if stat["low"] <= 0 <= stat["high"]:
                print("      区间跨 0：这一层的差异还不能说稳定存在")

    # ---- 双目标散点数据 ----
    scatter = []
    for qid, entry in per_query.items():
        utility = entry["utility"]
        topic_rel = topic.get(qid, {})
        row = {"query_id": qid, "event_family": entry["family"],
               "query_date": entry["date"],
               "adaptive_alpha": entry["adaptive_alpha"],
               "same_family_ratio": entry["same_family_ratio"],
               "e3_equals_e1": entry["e3_equals_e1"]}
        for mode, key in ((MODE_FINANCIAL, "e2"), (MODE_ADAPTIVE, "e3")):
            base_u = ndcg_at_k(entry["ids"][MODE_SEMANTIC], utility, 5)
            row[f"delta_utility_ndcg_{key}"] = (
                ndcg_at_k(entry["ids"][mode], utility, 5) - base_u)
            if topic_rel:
                base_t = ndcg_at_k(entry["ids"][MODE_SEMANTIC], topic_rel, 5)
                row[f"delta_topic_ndcg_{key}"] = (
                    ndcg_at_k(entry["ids"][mode], topic_rel, 5) - base_t)
            else:
                row[f"delta_topic_ndcg_{key}"] = None
        scatter.append(row)

    def quadrant(d_topic, d_utility):
        if d_topic is None:
            return "topic_pending"
        if d_topic >= 0 and d_utility >= 0:
            return "both_better"
        if d_topic >= 0:
            return "topic_better_utility_worse"
        if d_utility >= 0:
            return "topic_worse_utility_better"
        return "both_worse"

    for row in scatter:
        row["quadrant_e2"] = quadrant(row.get("delta_topic_ndcg_e2"),
                                      row["delta_utility_ndcg_e2"])
        row["quadrant_e3"] = quadrant(row.get("delta_topic_ndcg_e3"),
                                      row["delta_utility_ndcg_e3"])

    print(f"\n{'=' * 78}\n双目标象限（每点一条查询）\n{'=' * 78}")
    for key in ("quadrant_e2", "quadrant_e3"):
        counts = {}
        for row in scatter:
            counts[row[key]] = counts.get(row[key], 0) + 1
        print(f"  {key}: " + "　".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("  横轴：主题相关性变化　纵轴：离线结果有用性变化")

    # ---- 落盘 ----
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "modes": {m: MODE_LABELS[m] for m in MODES},
        "n_queries": len(per_query),
        "top_k": TOP_K,
        "utility_scale_sd": scale,
        "offline_outcome_utility": utility_summary,
        "topic_relevance": topic_summary,
        "human_labels": stats,
        "paired_bootstrap": comparisons,
        "note": ("offline_outcome_utility 使用查询事件发生后的真实结果，"
                 "只用于离线回测，不能用于在线排序。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    SCATTER_PATH.write_text(json.dumps(scatter, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n报告已写入 {REPORT_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"双目标数据已写入 {SCATTER_PATH.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
