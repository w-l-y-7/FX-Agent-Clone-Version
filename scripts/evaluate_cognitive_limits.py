"""受控压力测试：模型能否在冲突、证据不足和统计不确定时主动保留判断。

这不是现实世界效果评测，而是机制单元实验；不联网、不用密钥、不碰 2024。
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.aegisfx.cikrag import (  # noqa: E402
    Claim, EffectDirection, EvidenceType, FeatureDecision, PredictiveUtility,
    aggregate_feature, assess_claims, build_graph, decide, deduplicate,
)

ORIGIN = date(2022, 7, 1)


def _claim(index: int, direction: EffectDirection, *, regime: str = "normal",
           text: str | None = None, published: date = date(2020, 1, 1)) -> Claim:
    return Claim(
        claim_id=f"c{index}", feature_id="USD_Index", feature_name="美元指数",
        claim_text=text or f"独立观点 {index}", effect_direction=direction,
        market_regime=regime, forecast_horizon=20,
        evidence_type=EvidenceType.PEER_REVIEWED, published_at=published,
        sample_size_available=500, human_verified=True,
    )


def _utility(mean: float, low: float, high: float, consistency: float) -> PredictiveUtility:
    return PredictiveUtility(
        "USD_Index", gain_mean=mean, gain_median=mean,
        gain_by_fold=[mean] * 3, gain_low=low, gain_high=high,
        sign_consistency=consistency, n_folds=3)


def _run(claims: list[Claim], utility: PredictiveUtility,
         *, regime: str = "normal") -> FeatureDecision:
    claims = deduplicate(claims)
    active = [claim for claim in claims if claim.status.value == "active"]
    edges = build_graph(active)
    assessed = assess_claims(active, origin=ORIGIN, market_regime=regime)
    knowledge, conflicts = aggregate_feature(
        "USD_Index", "美元指数", assessed, market_regime=regime, edges=edges)
    return decide(knowledge, conflicts, utility)[0]


def main() -> int:
    positive = _utility(0.04, 0.01, 0.07, 1.0)
    uncertain = _utility(0.02, -0.01, 0.05, 2 / 3)
    negative = _utility(-0.04, -0.07, -0.01, 0.0)

    same = [_claim(i, EffectDirection.POSITIVE) for i in range(1, 6)]
    conflict = ([_claim(i, EffectDirection.POSITIVE) for i in range(10, 13)]
                + [_claim(i, EffectDirection.NEGATIVE) for i in range(20, 23)])
    conditional = ([_claim(i, EffectDirection.POSITIVE, regime="normal")
                    for i in range(30, 33)]
                   + [_claim(i, EffectDirection.NEGATIVE, regime="capital_control")
                      for i in range(40, 43)])
    duplicate_spam = [_claim(i, EffectDirection.POSITIVE, text="完全相同的转载")
                      for i in range(50, 60)]
    future = [_claim(70, EffectDirection.POSITIVE, published=date(2023, 1, 1))]

    scenarios = [
        ("证据一致且显著增益", same, positive, FeatureDecision.AUTO_ACCEPT),
        ("增益区间跨零", same, uncertain, FeatureDecision.RESEARCH_ONLY),
        ("同条件方向冲突", conflict, positive, FeatureDecision.HUMAN_REVIEW),
        ("不同制度条件相反", conditional, positive, FeatureDecision.AUTO_ACCEPT),
        ("没有知识证据", [], positive, FeatureDecision.RESEARCH_ONLY),
        ("显著负增益", same, negative, FeatureDecision.AUTO_REJECT),
        ("转载刷屏", duplicate_spam, positive, FeatureDecision.RESEARCH_ONLY),
        ("只有未来证据", future, positive, FeatureDecision.RESEARCH_ONLY),
    ]

    rows = []
    for name, claims, utility, expected in scenarios:
        actual = _run(claims, utility)
        mean_only = (FeatureDecision.AUTO_ACCEPT if utility.gain_mean > 0
                     else FeatureDecision.AUTO_REJECT)
        rows.append({
            "scenario": name, "expected": expected.value, "selective": actual.value,
            "mean_only": mean_only.value, "selective_correct": actual is expected,
            "mean_only_correct": mean_only is expected,
        })

    report = {
        "kind": "controlled_mechanism_stress_test",
        "uses_2024": False,
        "n_scenarios": len(rows),
        "selective_accuracy": sum(row["selective_correct"] for row in rows) / len(rows),
        "mean_only_accuracy": sum(row["mean_only_correct"] for row in rows) / len(rows),
        "mean_only_unsafe_accepts": sum(
            row["mean_only"] == "AUTO_ACCEPT" and row["expected"] != "AUTO_ACCEPT"
            for row in rows),
        "selective_unsafe_accepts": sum(
            row["selective"] == "AUTO_ACCEPT" and row["expected"] != "AUTO_ACCEPT"
            for row in rows),
        "scenarios": rows,
        "caveat": "受控机制测试只验证规则是否按设计工作，不代表真实文献库上的外部效度。",
    }
    out = _ROOT / "reports" / "stage6" / "cognitive_limits_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["selective_accuracy"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
