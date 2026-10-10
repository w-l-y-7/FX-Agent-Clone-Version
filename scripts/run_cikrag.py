"""CIK-RAG 主实验：特征筛选、选择性信任与固定 FA 对比。

    .\\venv\\Scripts\\python.exe scripts\\run_cikrag.py

流程：

1. 从知识库抽原子观点 → 去重 → 建关系图；
2. 对 32 个候选特征跑 purged walk-forward 消融（第二条通道）；
3. 聚合知识侧的分方向质量、冲突度、不足度（第一条通道）；
4. 双通道决策，给每个特征一个状态；
5. 把四组特征集放到**同一个固定 FA、同一套折**上比样本外误差。

## 五组

| 组 | 特征集 | 说明 |
|---|---|---|
| B0 | 全部 32 个候选 | 不做任何选择 |
| B1 | 仓库里 DA 复现的结果（10 个） | paper-described baseline，**不是严格复现** |
| B2 | 只用样本外消融筛 | 不看知识 |
| E5 | CIK-RAG 探索集 | AUTO_ACCEPT + RESEARCH_ONLY，仅供研究比较 |
| E6 | CIK-RAG 严格信任集 | 只有 AUTO_ACCEPT |

B2 与 E5 的差别就是「加不加知识通道」，这是本阶段最关键的一组对照。

不联网、不用密钥、不碰 2024。
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.cikrag import (  # noqa: E402
    ClaimStatus,
    EffectDirection,
    FeatureDecision,
    aggregate_feature,
    assess_claims,
    audit_decisions,
    build_graph,
    cluster_sizes,
    deduplicate,
    decide,
    extract_claims,
    feature_utility_with_interval,
    relation_counts,
    set_error,
)
from src.aegisfx.cikrag.ablation import FOLDS  # noqa: E402
from src.aegisfx.cikrag.literature import (  # noqa: E402
    load_all as load_literature,
    summarise as summarise_literature,
)
from src.core.knowledge_base import load_documents  # noqa: E402
from src.core.research_dataset import (  # noqa: E402
    EVENT_FEATURES,
    TIME_SERIES_FEATURES,
    load_daily_frame,
)

REPORT_DIR = _PROJECT_ROOT / "reports" / "stage6"
DA_REPORT = _PROJECT_ROOT / "reports" / "da_selection.json"
HORIZON = 20
# 这 4 份说明**没有发表日期**，内容是教科书级的机制描述（利率平价、贸易条件、
# 央行干预），不是有发表时点的研究发现。所以按「无日期」处理，`published_at=None`。
#
# 这不是为了让结果好看——设成 2026-09 的话，`published_at >= origin` 会把它们
# 全部过滤掉，`ignorance` 恒为 1.0，知识通道整个失效。
#
# **但必须承认这是个假设**：拿 2026 年写的机制说明去判断 2020 年的特征，
# 本身就带一点上帝视角。真实的文献库不会有这个问题（每篇都有发表时间）。
NOTE_PUBLISHED = None
# 决策发生在 2020–2022 三个样本外折都结束之后。真实文献加入后，
# published_at >= 该时点的材料会被自动排除，避免把未来文献写回历史实验。
DECISION_ORIGIN = date(2023, 1, 1)


def _load_da_baseline() -> list:
    if not DA_REPORT.exists():
        return []
    return json.loads(DA_REPORT.read_text(encoding="utf-8")).get(
        "selected_features", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="CIK-RAG 主实验")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    args = parser.parse_args()

    print("=" * 78)
    print("CIK-RAG 主实验")
    print("=" * 78)

    # ---------------------------------------------------------------- 知识侧
    documents = load_documents(verbose=False)
    note_claims = list(extract_claims(documents, published_at=NOTE_PUBLISHED))

    # 第二条知识来源：真实文献卡片。走的是语义阅读 + 逐字引用核查，
    # 不是 extract.py 的规则抽取（规则抽取对英文无效，见 literature.py 开头）。
    lit_claims, lit_problems, lit_cards = load_literature(_PROJECT_ROOT)
    if lit_problems:
        print(f"\n⚠ 文献卡片有 {len(lit_problems)} 个问题：")
        for p in lit_problems:
            print(f"    {p}")

    claims = deduplicate(note_claims + list(lit_claims))
    active = [c for c in claims if c.status is ClaimStatus.ACTIVE]
    edges = build_graph(active)
    sizes = cluster_sizes(claims)

    print(f"\n知识库：{len(documents)} 个语料块，来自 "
          f"{len({d['source'].split('#')[0] for d in documents})} 份文档")
    print(f"  · 自撰机制说明：{len(note_claims)} 条观点")
    print(f"  · 文献卡片：{len(lit_cards)} 张卡 → {len(lit_claims)} 条观点"
          f"（引用逐字核查{'全部通过' if not lit_problems else '有失败'}）")
    print(f"合并去重后 {len(claims)} 条（有效 {len(active)} 条，"
          f"重复剔除 {len(claims) - len(active)} 条）")
    print(f"来源类型分布：{dict(Counter(c.evidence_type.value for c in active))}")
    print(f"关系图：{relation_counts(edges)}")

    if lit_claims:
        print("文献观点分布：" + json.dumps(summarise_literature(lit_claims),
                                            ensure_ascii=False))
        # 决策时点在 2023-01-01：这之后发表的论文当时还不存在，
        # assess_claims 会把它们排除，避免把未来文献写回历史实验。
        in_window = [c for c in lit_claims
                     if c.published_at and c.published_at < DECISION_ORIGIN]
        print(f"其中发表于 {DECISION_ORIGIN} 之前、能进入决策的："
              f"{len(in_window)}/{len(lit_claims)} 条")

    if not active:
        print("\n⚠ 没有抽到任何观点。检查 knowledge_base/notes 是否存在。")

    # ---------------------------------------------------------------- 预测侧
    frame = load_daily_frame()
    candidates = list(TIME_SERIES_FEATURES) + list(EVENT_FEATURES)
    print(f"\n候选特征 {len(candidates)} 个，跑 purged walk-forward 消融"
          f"（折 {list(FOLDS)}，期限 {args.horizon} 日）")

    started = time.time()
    utilities = {}
    for index, feature in enumerate(candidates, start=1):
        utility, _ = feature_utility_with_interval(frame, feature, args.horizon)
        utilities[feature] = utility
        if index % 8 == 0:
            print(f"    已完成 {index}/{len(candidates)}"
                  f"（{time.time() - started:.0f} 秒）")
    print(f"  消融耗时 {time.time() - started:.0f} 秒")

    # ---------------------------------------------------------------- 双通道决策
    decisions = {}
    for feature in candidates:
        relevant = [c for c in active if c.feature_id == feature]
        knowledge, conflicts = aggregate_feature(
            feature, feature, assess_claims(relevant, origin=DECISION_ORIGIN,
                                            cluster_sizes=sizes),
            edges=[e for e in edges
                   if e.source_claim_id in {c.claim_id for c in relevant}
                   and e.target_claim_id in {c.claim_id for c in relevant}])
        status, reason = decide(knowledge, conflicts, utilities[feature])
        decisions[feature] = {"knowledge": knowledge, "status": status,
                              "reason": reason, "conflicts": conflicts}

    counts = Counter(d["status"].value for d in decisions.values())
    print(f"\n特征状态分布：{dict(counts)}")
    selective_audit = audit_decisions(
        {feature: item["status"] for feature, item in decisions.items()})
    print(f"机器覆盖率 {selective_audit.machine_coverage:.1%}，"
          f"主动保留/转人工率 {selective_audit.abstention_rate:.1%}，"
          f"人工复核率 {selective_audit.human_review_rate:.1%}")

    # ---------------------------------------------------------------- 四组对比
    e5_features = [f for f in candidates
                   if utilities[f].is_positive
                   and decisions[f]["status"] in (FeatureDecision.AUTO_ACCEPT,
                                                  FeatureDecision.RESEARCH_ONLY)]
    b2_features = [f for f in candidates
                   if utilities[f].is_positive and utilities[f].n_folds >= 2]
    b1_features = [f for f in _load_da_baseline() if f in candidates]

    groups = {
        "B0 全部候选": candidates,
        "B1 DA 复现（paper-described）": b1_features,
        "B2 仅样本外消融": b2_features,
        "E5 CIK-RAG 探索集": e5_features,
        "E6 CIK-RAG 严格信任集": [f for f in candidates
                                   if decisions[f]["status"] is FeatureDecision.AUTO_ACCEPT],
    }

    print(f"\n{'=' * 78}\n五组特征集对比（同一固定 FA：StandardScaler + Ridge(1.0)）\n{'=' * 78}")
    print(f"{'组别':<28}{'特征数':>7}{'MAE':>12}{'MAPE':>12}{'RMSE':>12}{'方向':>9}")
    results = {}
    for name, features in groups.items():
        abstains = name.startswith("E6") and not features
        metrics = ({"n_features": 0, "n_folds": 0, "abstained": True,
                    "not_reported_reason": "没有 AUTO_ACCEPT 特征，系统保留判断，不生成预测"}
                   if abstains else set_error(frame, features, args.horizon))
        results[name] = {"features": features, "metrics": metrics,
                         "operational_status": ("ABSTAIN" if abstains else "FORECAST")}
        if abstains:
            print(f"{name:<28}{len(features):>7}{'无自动采纳，系统保留判断':>45}")
            continue
        if not metrics.get("n_folds"):
            print(f"{name:<28}{len(features):>7}{'样本不足':>12}")
            continue
        print(f"{name:<28}{metrics['n_features']:>7}{metrics['mae']:>12.6f}"
              f"{metrics['mape']:>12.6f}{metrics['rmse']:>12.6f}"
              f"{metrics['direction_accuracy']:>8.1%}")

    # ---------------------------------------------------------------- 逐特征明细
    print(f"\n{'=' * 78}\n逐特征（按样本外增益排序，只列前 12）\n{'=' * 78}")
    ordered = sorted(candidates, key=lambda f: -(utilities[f].gain_mean or -9))
    print(f"{'特征':<42}{'增益':>10}{'折一致':>8}{'冲突':>8}{'不足':>8}{'状态':>18}")
    for feature in ordered[:12]:
        utility = utilities[feature]
        knowledge = decisions[feature]["knowledge"]
        conflict = (f"{knowledge.conflict_score:.2f}"
                    if knowledge.conflict_score is not None else "—")
        print(f"{feature[:40]:<42}"
              f"{(f'{utility.gain_mean:+.4f}' if utility.gain_mean is not None else '—'):>10}"
              f"{(f'{utility.sign_consistency:.0%}' if utility.sign_consistency is not None else '—'):>8}"
              f"{conflict:>8}{knowledge.ignorance_score:>8.2f}"
              f"{decisions[feature]['status'].value:>18}")

    # ---------------------------------------------------------------- 落盘
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "horizon": args.horizon,
        "knowledge_base": {
            "n_documents": len(documents),
            "n_claims": len(claims),
            "n_active_claims": len(active),
            "evidence_type_counts": dict(Counter(c.evidence_type.value
                                                 for c in active)),
            "relation_counts": relation_counts(edges),
            "note_claims": len(note_claims),
            "literature": {
                "n_cards": len(lit_cards),
                "n_claims": len(lit_claims),
                "span_verification": ("all_passed" if not lit_problems
                                      else f"{len(lit_problems)}_failed"),
                "problems": lit_problems,
                "distribution": summarise_literature(lit_claims),
                "n_usable_at_origin": len([
                    c for c in lit_claims
                    if c.published_at and c.published_at < DECISION_ORIGIN]),
                "extraction_model": "literature-card/manual",
            },
            "caveat": ("知识库 = 4 份本项目自撰的机制说明（无发表日期，"
                       "教科书级机制描述）+ 真实文献卡片。文献卡片是**语义阅读**"
                       "产物，不是规则抽取；每条断言都附论文原文句子并逐字核查。"
                       "观点数量级由语料决定。"),
        },
        "feature_decisions": {
            feature: {
                "status": decisions[feature]["status"].value,
                "reason": decisions[feature]["reason"],
                "n_active_claims": decisions[feature]["knowledge"].n_active_claims,
                "conflict_score": decisions[feature]["knowledge"].conflict_score,
                "ignorance_score": decisions[feature]["knowledge"].ignorance_score,
                "context_applicability": decisions[feature]["knowledge"].context_applicability,
                "real_conflicts": decisions[feature]["conflicts"],
                "gain_mean": utilities[feature].gain_mean,
                "gain_by_fold": utilities[feature].gain_by_fold,
                "gain_low": utilities[feature].gain_low,
                "gain_high": utilities[feature].gain_high,
                "sign_consistency": utilities[feature].sign_consistency,
                "n_folds": utilities[feature].n_folds,
            }
            for feature in candidates
        },
        "selective_audit": selective_audit.to_dict(),
        "groups": {name: {"features": item["features"],
                          "metrics": item["metrics"],
                          "operational_status": item["operational_status"]}
                   for name, item in results.items()},
        "folds": [list(fold) for fold in FOLDS],
        "decision_origin": DECISION_ORIGIN.isoformat(),
    }
    path = REPORT_DIR / "cikrag_main.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"\n报告已写入 {path.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
