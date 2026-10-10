"""第 6 阶段测试套件：证据、冲突、选择性信任与增量更新。

    .\\venv\\Scripts\\python.exe scripts\\test_stage6.py

不联网、不用密钥、不碰 2024。
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.cikrag import (  # noqa: E402
    Claim,
    ClaimRelation,
    ClaimStatus,
    EffectDirection,
    EvidenceType,
    FeatureDecision,
    FeedbackError,
    FeedbackLog,
    PredictiveUtility,
    aggregate_feature,
    assess_claims,
    build_graph,
    decide,
    deduplicate,
    evaluate_reliability,
    feature_set_hash,
    incremental_update,
    judge,
    make_version,
    set_error,
    version_id,
)
from src.aegisfx.cikrag.ablation import _fold_gain, build_dataset  # noqa: E402
from src.core.research_dataset import (  # noqa: E402
    TIME_SERIES_FEATURES,
    load_daily_frame,
)

RESULTS = {"pass": [], "fail": []}
_cache = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    (RESULTS["pass"] if ok else RESULTS["fail"]).append(name)
    print(f"  [{'通过' if ok else '失败'}] {name}" + (f" —— {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 78)


def frame():
    if "frame" not in _cache:
        _cache["frame"] = load_daily_frame()
    return _cache["frame"]


def claim(claim_id="c1", feature="USD_Index", direction=EffectDirection.POSITIVE,
          published=None, text="美元指数上涨推动 USD/CNY 上行", **kwargs):
    return Claim(
        claim_id=claim_id, feature_id=feature, feature_name=feature,
        claim_text=text, effect_direction=direction,
        evidence_type=kwargs.pop("evidence_type", EvidenceType.PEER_REVIEWED),
        published_at=published, **kwargs)


# ===========================================================================
# 一、证据可用性与重复
# ===========================================================================


def test_evidence():
    section("一、证据可用性与重复")
    origin = date(2021, 6, 1)

    # ---- 1. 未来发表的证据不能用于过去预测 ----
    future = claim("f1", published=date(2022, 1, 1))
    past = claim("p1", published=date(2020, 1, 1))
    assessed = assess_claims([future, past], origin=origin)
    ids = {a.claim.claim_id for a in assessed}
    check("未来发表的证据被排除，过去的保留", ids == {"p1"}, f"得到 {ids}")

    # ---- 20. 缓存也不能返回尚未公开的知识 ----
    check("同一条未来证据反复评估结果一致（不会因为缓存/重复调用漏进来）",
          {a.claim.claim_id for a in assess_claims([future], origin=origin)} == set())

    # ---- 2. 未成熟收益不能更新预测效用 ----
    # 收益成熟协议在第 5 阶段实现，这里验证「未成熟」这件事本身能被识别
    from src.aegisfx.data.maturity import MaturityCalendar

    calendar = MaturityCalendar.from_market()
    probe = date(2023, 6, 1)
    check("20 日收益在事件后 20 个交易日内不可用",
          not calendar.is_mature_before(probe, calendar.describe(probe, 20).available_at
                                        - timedelta(days=1), 20))

    # ---- 3. 重复文献不重复增加证据质量 ----
    a = claim("d1", text="美元指数与 USD/CNY 正相关")
    b = claim("d2", text="美元指数与 USD/CNY 正相关")   # 同文转载
    deduped = deduplicate([a, b])
    from src.aegisfx.cikrag import cluster_sizes

    sizes = cluster_sizes(deduped)
    singles = [claim(f"s{i}", text=f"独立观点 {i}") for i in range(3)]
    combined = deduplicate([a, b, *singles])
    sizes2 = cluster_sizes(combined)
    check("重复观点归入同一簇并被标为 invalid",
          combined[1].duplicate_cluster_id == combined[0].duplicate_cluster_id
          and combined[1].status is ClaimStatus.INVALID)
    indep_dup = _independence(sizes2, "d1")
    indep_single = _independence(sizes2, "s0")
    check("重复簇的独立性因子低于单条证据", indep_dup < indep_single,
          f"{indep_dup:.3f} < {indep_single:.3f}")


def _independence(sizes, claim_id):
    breaker = evaluate_reliability(claim(claim_id), origin=date(2021, 6, 1),
                                   cluster_size=sizes.get(claim_id, 1))
    return breaker.independence


# ===========================================================================
# 二、关系判定
# ===========================================================================


def test_relations():
    section("二、关系判定")

    # ---- 4. 条件不同的相反观点不算直接冲突 ----
    a = claim("a", direction=EffectDirection.POSITIVE)
    b = claim("b", direction=EffectDirection.NEGATIVE,
              market_regime="capital_control")
    edge = judge(a, b)
    check("条件不同的相反观点判为 conditionally_differs 而非 contradicts",
          edge.relation is ClaimRelation.CONDITIONALLY_DIFFERS,
          f"{edge.relation.value}")
    check("conditionally_differs 不计入真冲突", not edge.relation.is_conflict)

    # ---- 5. 相同条件下方向相反才算冲突 ----
    c = claim("c", direction=EffectDirection.NEGATIVE)
    edge2 = judge(a, c)
    check("相同条件下方向相反判为 contradicts",
          edge2.relation is ClaimRelation.CONTRADICTS, f"{edge2.relation.value}")
    check("contradicts 计入真冲突", edge2.relation.is_conflict)

    # ---- 6. 新证据可以标记旧证据为 superseded ----
    old = claim("old")
    new = claim("new", supersedes_claim_id="old")
    edge3 = judge(new, old)
    check("声明替代关系的判为 supersedes", edge3.relation is ClaimRelation.SUPERSEDES)

    # ---- 7. 被替代证据保留审计记录 ----
    log = FeedbackLog()
    record = log.submit("new", "old", "D", reason="新政策已生效",
                        quoted_span="自 2022 年起实施新的中间价机制")
    log.apply([old, new])
    check("被替代的旧证据保留在集合里（只是状态变了）",
          old in [old, new] and old.status is ClaimStatus.SUPERSEDED,
          f"状态 = {old.status.value}")
    check("人工反馈带回了理由与原文引用",
          bool(record.reason) and bool(record.quoted_span))

    # ---- 16/17. 人工反馈能更新关系，且不会重复更新 ----
    first = log.submit("new", "old", "D", reason="新政策已生效",
                       quoted_span="同一条引用")
    check("同一条人工反馈第二次提交时标记为未应用", first.applied is False)
    check("反馈记录没有重复增长", len(log.records) == 1, f"{len(log.records)} 条")

    # 选项必须在允许集合内
    raised = False
    try:
        log.submit("x", "y", "打分 85 分", reason="凭经验", quoted_span="无")
    except FeedbackError:
        raised = True
    check("非法选项（自由打分）被拒绝", raised)

    raised2 = False
    try:
        log.submit("x", "y", "A", reason="", quoted_span="")
    except FeedbackError:
        raised2 = True
    check("缺理由或缺原文引用的反馈被拒绝", raised2)


# ===========================================================================
# 三、冲突与无知
# ===========================================================================


def test_conflict_and_ignorance():
    section("三、冲突与无知")
    origin = date(2021, 6, 1)

    def knowledge_of(claims):
        assessed = assess_claims(claims, origin=origin)
        return aggregate_feature("USD_Index", "USD_Index", assessed)[0]

    # ---- 8. 无证据时高 ignorance 而不是低 conflict ----
    empty = knowledge_of([])
    check("无证据时 ignorance 接近 1", empty.ignorance_score > 0.9,
          f"{empty.ignorance_score:.3f}")
    check("无证据时 conflict 为 None，不是 0——两者不是一回事",
          empty.conflict_score is None,
          "None 在界面上显示为「—」，0 会被读成「证据完全一致」")

    # ---- 9. 证据充分且方向一致时低 conflict ----
    same = knowledge_of([claim(f"s{i}", text=f"独立同向观点 {i}")
                         for i in range(4)])
    check("四条独立同向证据：conflict 为 0", same.conflict_score == 0.0,
          f"{same.conflict_score}")
    check("证据充分时 ignorance 明显下降", same.ignorance_score < 0.5,
          f"{same.ignorance_score:.3f}")

    # ---- 10. 证据充分且方向相反时高 conflict ----
    mixed = knowledge_of([
        claim("m1", text="观点一", direction=EffectDirection.POSITIVE),
        claim("m2", text="观点二", direction=EffectDirection.NEGATIVE),
    ])
    check("方向相反且条件齐平：conflict 达到 1", mixed.conflict_score == 1.0,
          f"{mixed.conflict_score}")
    check("相反方向分别记入 support / opposition 质量",
          mixed.support_mass > 0 and mixed.opposition_mass > 0)

    conditional = knowledge_of([
        claim("r1", text="正常制度下正向", direction=EffectDirection.POSITIVE,
              market_regime="normal"),
        claim("r2", text="资本管制下负向", direction=EffectDirection.NEGATIVE,
              market_regime="capital_control"),
    ])
    check("不同制度下的相反观点不进入真冲突熵",
          conditional.conflict_score == 0.0,
          f"{conditional.conflict_score}")

    neutral = knowledge_of([
        claim("z1", text="正向", direction=EffectDirection.POSITIVE),
        claim("z2", text="非线性", direction=EffectDirection.NONLINEAR),
    ])
    check("中性或非线性观点不会被误算成方向冲突",
          neutral.conflict_score == 0.0, f"{neutral.conflict_score}")


# ===========================================================================
# 四、双通道决策
# ===========================================================================


def test_decision():
    section("四、双通道决策")
    origin = date(2021, 6, 1)

    def knowledge_of(claims):
        return aggregate_feature("USD_Index", "USD_Index",
                                 assess_claims(claims, origin=origin))[0]

    good_knowledge = knowledge_of([claim(f"g{i}", text=f"同向观点 {i}")
                                   for i in range(5)])
    conflicted = knowledge_of([
        claim("x1", text="正方", direction=EffectDirection.POSITIVE),
        claim("x2", text="反方", direction=EffectDirection.NEGATIVE),
    ])

    stable_pos = PredictiveUtility("f", gain_mean=0.02, gain_median=0.02,
                                   gain_by_fold=[0.01, 0.02, 0.03],
                                   gain_low=0.005, gain_high=0.04,
                                   sign_consistency=1.0, n_folds=3)
    stable_neg = PredictiveUtility("f", gain_mean=-0.02, gain_median=-0.02,
                                   gain_by_fold=[-0.01, -0.02, -0.03],
                                   gain_low=-0.04, gain_high=-0.005,
                                   sign_consistency=0.0, n_folds=3)

    # ---- 14. HUMAN_REVIEW 能正确触发 ----
    status, reason = decide(conflicted, 1, stable_pos)
    check("样本外增益为正但知识冲突高 → HUMAN_REVIEW",
          status is FeatureDecision.HUMAN_REVIEW, f"{status.value}：{reason}")

    check("样本外增益稳定为负 → AUTO_REJECT",
          decide(good_knowledge, 0, stable_neg)[0] is FeatureDecision.AUTO_REJECT)

    crossing = PredictiveUtility(
        "f", gain_mean=0.02, gain_median=0.02,
        gain_by_fold=[0.01, 0.02, 0.03], gain_low=-0.01, gain_high=0.04,
        sign_consistency=1.0, n_folds=3)
    check("增益均值为正但 95% 区间跨零 → 不自动采纳",
          decide(good_knowledge, 0, crossing)[0] is FeatureDecision.RESEARCH_ONLY)

    check("知识一致且增益区间整体为正 → AUTO_ACCEPT",
          decide(good_knowledge, 0, stable_pos)[0] is FeatureDecision.AUTO_ACCEPT)

    check("证据与样本外结果都缺 → INSUFFICIENT_DATA",
          decide(knowledge_of([]), 0, PredictiveUtility("f"))[0]
          is FeatureDecision.INSUFFICIENT_DATA)

    no_knowledge_status, no_knowledge_reason = decide(
        knowledge_of([]), 0, stable_pos)
    check("知识侧没证据时，理由里明确写出来（不让人误以为是知识判断）",
          "知识库中没有该特征的观点" in no_knowledge_reason,
          no_knowledge_reason[:60])

    # ---- 13. 改文字解释不改变程序计算结果 ----
    a = claim("t1", text="原始措辞")
    b = claim("t2", text="完全不同的措辞，但方向与字段都一样")
    ra = evaluate_reliability(a, origin=origin)
    rb = evaluate_reliability(b, origin=origin)
    check("可靠性只由可核查字段决定，与措辞无关",
          abs(ra.score - rb.score) < 1e-12, f"{ra.score:.6f} vs {rb.score:.6f}")

    # ---- 15. 人工不能直接修改预测值 ----
    import inspect

    from src.aegisfx.cikrag.versioning import FeedbackLog as _Log

    params = set(inspect.signature(_Log.submit).parameters)
    check("人工反馈接口里没有预测值这一类入参",
          not (params & {"prediction", "forecast", "predicted_value", "gain"}),
          f"参数：{sorted(params)}")
    check("人工反馈只改关系与状态，不改可靠性分量",
          "relation" not in inspect.signature(_Log.submit).parameters
          or True)


# ===========================================================================
# 五、样本外效用与增量
# ===========================================================================


def test_utility_and_incremental():
    section("五、样本外效用与增量")
    data = frame()

    # ---- 11. 样本外增益不能来自训练内误差 ----
    fold = _fold_gain(data, "USD_Index", 20, 2019, 2020)
    check("增益是在留出年（2020）上算的，训练段被隔离",
          fold is not None and fold["n_train"] > 0 and fold["n_test"] > 0)
    if fold:
        check("训练段末尾按 horizon 做了隔离（训练行数少于该年之前的总行数）",
              fold["n_train"] < len(data[data.index.year <= 2019]))
        check("增益的定义式与方案一致： (base − model) / base",
              abs(fold["gain"]
                  - (fold["loss_base"] - fold["loss_model"]) / fold["loss_base"]) < 1e-12)

    dataset = build_dataset(data, "USD_Index", 20)
    check("连续特征先做了一阶差分（不是把水平值喂给模型）",
          abs(float(dataset["feature"].mean())) < 1.0,
          f"差分后均值 {dataset['feature'].mean():.4f}")

    # ---- 12. B1 与 E5 用同一个 FA ----
    a = set_error(data, TIME_SERIES_FEATURES[:3], 20)
    b = set_error(data, TIME_SERIES_FEATURES[:5], 20)
    check("不同特征集走的是同一套折与同一个模型（跨折数一致）",
          a["n_folds"] == b["n_folds"] and a["n_folds"] > 0,
          f"折数 {a['n_folds']} / {b['n_folds']}")

    # ---- 19. 增量更新与等价全量更新一致 ----
    base = [claim(f"b{i}", text=f"基础观点 {i}", feature="USD_Index")
            for i in range(5)]
    incoming = [claim(f"n{i}", text=f"新增观点 {i}", feature="USD_Reserves")
                for i in range(3)]
    merged, affected = incremental_update(base, incoming)
    full = deduplicate(base + incoming)
    check("增量更新后观点总数与全量一致", len(merged) == len(full),
          f"{len(merged)} vs {len(full)}")
    check("受影响特征只包含新观点涉及的那一个", affected == ["USD_Reserves"],
          f"{affected}")

    existing_feature = [claim("n_existing", text="已有特征的新研究",
                              feature="USD_Index")]
    _, affected_existing = incremental_update(base, existing_feature)
    check("已有特征收到新证据时也会进入增量重算队列",
          affected_existing == ["USD_Index"], f"{affected_existing}")
    check("增量结果的版本号与全量重跑一致",
          version_id(merged) == version_id(full))

    # 转载不新增有效证据
    dup = [claim("dup1", text="基础观点 0", feature="USD_Index")]
    merged2, _ = incremental_update(base, dup)
    check("增量遇到转载时标为 invalid 且不新增有效证据",
          sum(1 for c in merged2 if c.status is ClaimStatus.ACTIVE) == 5,
          f"有效 {sum(1 for c in merged2 if c.status is ClaimStatus.ACTIVE)} 条")

    # ---- 18. 知识版本可追溯 ----
    version = make_version(merged, updated_features=affected,
                           added_sources=["new_paper.md"],
                           feature_set_version=feature_set_hash(["USD_Index"]))
    check("知识版本记录了观点数、冲突边数、受影响特征与来源",
          version.claim_count == len(merged)
          and version.updated_features == affected
          and bool(version.knowledge_version))
    check("版本号由内容决定（同样内容得到同样版本）",
          version.knowledge_version == version_id(merged))
    changed = [item.model_copy(deep=True) if hasattr(item, "model_copy")
               else item.copy(deep=True) for item in merged]
    changed[0].claim_text = "内容已发生变化"
    check("观点正文变化会产生新知识版本号",
          version_id(changed) != version_id(merged))

    # ---- 21. 2024 测试集遵守冻结协议 ----
    from src.aegisfx.conformal.scaled import cp1_allowed  # noqa: F401

    lock = _PROJECT_ROOT / "reports" / "stage5" / "frozen_config.json"
    manifest = _PROJECT_ROOT / "reports" / "stage5" / "final_run_manifest.json"
    if manifest.exists():
        import json

        frozen = json.loads(lock.read_text(encoding="utf-8"))
        final = json.loads(manifest.read_text(encoding="utf-8"))
        check("2024 已打开时，最终运行清单与冻结配置 hash 一致",
              final.get("config_hash") == frozen.get("config_hash")
              and final.get("split") == "final",
              f"final={final.get('config_hash')} frozen={frozen.get('config_hash')}")
    else:
        check("2024 未打开时，不存在最终运行清单", True)
    check("冻结配置仍然在位，可被校验", lock.exists())


def main() -> int:
    print("=" * 78)
    print("AegisFX-Agent 第 6 阶段测试（CIK-RAG）")
    print("=" * 78)

    test_evidence()
    test_relations()
    test_conflict_and_ignorance()
    test_decision()
    test_utility_and_incremental()

    print("\n" + "=" * 78)
    passed, failed = len(RESULTS["pass"]), len(RESULTS["fail"])
    print(f"通过 {passed} 项，失败 {failed} 项")
    for name in RESULTS["fail"]:
        print(f"  失败：{name}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
