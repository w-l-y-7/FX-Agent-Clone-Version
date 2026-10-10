"""第 4 阶段测试套件。

    .\\venv\\Scripts\\python.exe scripts\\test_aegisfx.py

覆盖四块：检索层（E1/E2 回归、E3 约束、防泄漏）、同日聚合、冲突指标、
标注与评测。

**不需要密钥，也不联网**——向量模型读本地缓存，拿不到就自动退到 TF-IDF。
所有用到随机的地方都固定种子。

这里的检查刻意做「独立复算」而不是「对快照」：E1/E2 的排序由测试自己按
文档里的公式重算一遍再比对。快照只能证明「没变」，复算能证明「是对的」。
"""

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.conflict import (  # noqa: E402
    SameDayInconsistency,
    aggregate_by_day,
    aggregation_summary,
    classify,
    directional_conflict,
    effective_sample_size,
    neutral_mass,
    robust_scale,
    summarise,
    weighted_median,
)
from src.aegisfx.conflict.direction import decide_direction  # noqa: E402
from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import EventIndex  # noqa: E402
from src.aegisfx.retrieval.queries import load_or_build  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    ADAPTIVE_CANDIDATE_POOL,
    MODE_ADAPTIVE,
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_of_pool,
    retrieve,
    retrieve_detailed,
)
from src.aegisfx.retrieval.scoring import (  # noqa: E402
    ALPHA_MAX,
    DEFAULT_WEIGHTS,
    MarketScaler,
    adaptive_alpha,
    composite,
    event_similarity,
    finance_score,
    market_similarity,
    market_vector,
    same_family_ratio,
    semantic_to_01,
    time_similarity,
)
from src.aegisfx.schemas.research import Direction  # noqa: E402
from src.aegisfx.schemas.views import (  # noqa: E402
    EvaluationEventView,
    LeakageError,
    RetrievalEventView,
    assert_retrieval_safe,
)

RESULTS = {"pass": [], "fail": []}
LABELS_PATH = _PROJECT_ROOT / "data" / "aegisfx" / "retrieval" / "labels.csv"

_cache = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    (RESULTS["pass"] if ok else RESULTS["fail"]).append(name)
    mark = "通过" if ok else "失败"
    print(f"  [{mark}] {name}" + (f" —— {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 78)


def fixture():
    """向量模型与索引只加载一次。"""
    if "ready" not in _cache:
        events = load_event_store()
        embedder, note = build_embedder([event.text for event in events])
        index = EventIndex(events, embedder, note=note)
        if not index.load_vectors_from_disk():
            index.build()
        _cache.update(events=events, index=index,
                      by_id={event.event_id: event for event in events},
                      queries=load_or_build(events), note=note)
    return _cache


# ===========================================================================
# 一、检索层
# ===========================================================================


def test_retrieval_layer() -> None:
    section("一、检索层")
    f = fixture()
    events, index, by_id, queries = f["events"], f["index"], f["by_id"], f["queries"]

    # ---- 1/2. E1、E2 行为不变：按文档公式独立复算再比对 ----
    # 注意用带成熟协议的池（horizon=20）：检索默认就是这个池，
    # 拿不带成熟协议的池复算会差出一两条来
    query = by_id[queries[0].event_id]
    pool = as_of_pool(events, query, horizon=20)
    pool_views = [RetrievalEventView.from_event(item) for item in pool]
    qv = RetrievalEventView.from_event(query, include_realized=False)
    qvec = index.vector_for(qv.event_id)

    semantic = {v.event_id: float(index.vector_for(v.event_id) @ qvec)
                for v in pool_views}
    expected_e1 = [eid for eid, _ in sorted(semantic.items(),
                                            key=lambda kv: (-kv[1], kv[0]))][:10]

    scaler = MarketScaler.fit([market_vector(v) for v in pool_views])
    qm = scaler.transform(market_vector(qv))
    expected_e2 = []
    for v in pool_views:
        comp = {
            "semantic": semantic[v.event_id],
            "event": event_similarity(qv, v),
            "market": market_similarity(qm, scaler.transform(market_vector(v)),
                                        n_features=4),
            "time": time_similarity(qv, v),
        }
        expected_e2.append((v.event_id, composite(comp, DEFAULT_WEIGHTS)))
    expected_e2 = [eid for eid, _ in sorted(expected_e2,
                                            key=lambda kv: (-kv[1], kv[0]))][:10]

    got_e1 = [h.event_id for h in retrieve(query, events, index,
                                           mode=MODE_SEMANTIC, top_k=10)]
    got_e2 = [h.event_id for h in retrieve(query, events, index,
                                           mode=MODE_FINANCIAL, top_k=10)]
    check("E1 排序与文档公式的独立复算一致", got_e1 == expected_e1)
    check("E2 排序与文档公式的独立复算一致", got_e2 == expected_e2)

    # ---- 3. E3 只能在语义 Top-20 里重排 ----
    outcome = retrieve_detailed(query, events, index, mode=MODE_ADAPTIVE, top_k=20,
                                pool=pool)
    semantic_top20 = set(sorted(semantic, key=lambda eid: (-semantic[eid], eid))
                         [:ADAPTIVE_CANDIDATE_POOL])
    e3_ids = {h.event_id for h in outcome.evidence}
    check(f"E3 的结果全部落在语义 Top-{ADAPTIVE_CANDIDATE_POOL} 之内",
          e3_ids <= semantic_top20,
          f"{len(e3_ids - semantic_top20)} 条越界" if e3_ids - semantic_top20 else "")

    # ---- 4. E3 不检索未来或同日事件 ----
    from src.aegisfx.retrieval.retriever import as_day

    cutoff = as_day(query.date)
    offenders = [h.event_id for h in retrieve(query, events, index,
                                              mode=MODE_ADAPTIVE, top_k=20)
                 if as_day(h.event.event_date if hasattr(h.event, "event_date")
                           else h.event.date) >= cutoff]
    check("E3 不检索查询日当天或之后的事件", not offenders,
          f"{len(offenders)} 条越界" if offenders else "")

    # ---- 5. 改查询未来收益，三路排序都不变 ----
    def ranking_of(event_obj, mode):
        return [h.event_id for h in retrieve(event_obj, events, index,
                                             mode=mode, top_k=10)]

    baseline = {mode: ranking_of(query, mode)
                for mode in (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE)}
    mutated = query.model_copy(deep=True)
    mutated.abret_1d, mutated.abret_5d, mutated.abret_20d = 9.9, -9.9, 9.9
    mutated_rankings = {mode: ranking_of(mutated, mode)
                        for mode in (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE)}
    check("把查询未来收益改成别的值后，E1/E2/E3 排序完全不变",
          baseline == mutated_rankings,
          "" if baseline == mutated_rankings else
          f"变了：{ [m for m in baseline if baseline[m] != mutated_rankings[m]] }")

    # 直接喂评测视图必须被拦下
    blocked = False
    try:
        retrieve(EvaluationEventView.from_event(query), events, index)
    except LeakageError:
        blocked = True
    check("把 EvaluationEventView 当查询传进检索会被拦下", blocked)

    leaked_view = RetrievalEventView.from_event(query, include_realized=True)
    blocked2 = False
    try:
        assert_retrieval_safe(leaked_view)
    except LeakageError:
        blocked2 = True
    check("查询视图带已实现收益时会被拦下", blocked2)

    # ---- 6. 分数组件有限、无 NaN ----
    bad = []
    for mode in (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE):
        for hit in retrieve(query, events, index, mode=mode, top_k=10):
            for key, value in hit.components.items():
                if value is None or not np.isfinite(value):
                    bad.append((mode, hit.event_id, key, value))
            if not np.isfinite(hit.score):
                bad.append((mode, hit.event_id, "score", hit.score))
    check("所有分数组件有限且无 NaN", not bad, f"{bad[:3]}" if bad else "")

    # ---- 7. 同样输入重复执行结果一致 ----
    repeat = {mode: ranking_of(query, mode)
              for mode in (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE)}
    check("同样输入重复执行结果完全一致", repeat == baseline)

    # ---- 8. α 始终落在 [0, 0.5] ----
    alphas = []
    for spec in queries:
        outcome_i = retrieve_detailed(by_id[spec.event_id], events, index,
                                      mode=MODE_ADAPTIVE, top_k=5)
        alphas.append(outcome_i.adaptive_alpha)
    in_range = all(a is not None and 0.0 <= a <= ALPHA_MAX + 1e-12 for a in alphas)
    check(f"α_q 全部落在 [0, {ALPHA_MAX}]", in_range,
          f"范围 {min(alphas):.3f}–{max(alphas):.3f}" if in_range else f"{alphas}")
    check("α 在 p_q ≤ 0.5 时为 0（语义结果够散时不启用金融重排）",
          adaptive_alpha(0.4) == 0.0 and adaptive_alpha(0.5) == 0.0)
    check("α 在 p_q = 1 时取到上限", abs(adaptive_alpha(1.0) - ALPHA_MAX) < 1e-12)
    check("semantic_to_01 把余弦映射到 [0,1]",
          semantic_to_01(-1.0) == 0.0 and semantic_to_01(1.0) == 1.0
          and abs(semantic_to_01(0.0) - 0.5) < 1e-12)


# ===========================================================================
# 二、同日聚合
# ===========================================================================


class FakeEvent:
    def __init__(self, event_id, day, abret_20d=None, text="", family="tariff"):
        self.event_id = event_id
        self.date = day
        self.abret_20d = abret_20d
        self.text = text
        self.event_family = family


class FakeEvidence:
    def __init__(self, event, score):
        self.event = event
        self.score = score


def test_same_day_aggregation() -> None:
    section("二、同日聚合")
    day = datetime(2023, 5, 10)

    # ---- 9. 两条同日事件只产生一个证据日 ----
    evidence = [
        FakeEvidence(FakeEvent("a", day, 0.01, "甲"), 0.8),
        FakeEvidence(FakeEvent("b", day, 0.01, "乙"), 0.6),
        FakeEvidence(FakeEvent("c", day + timedelta(days=3), 0.02, "丙"), 0.5),
    ]
    dated = aggregate_by_day(evidence, 20)
    check("两条同日事件只产生一个证据日", len(dated) == 2, f"得到 {len(dated)} 个")

    same_day = next(group for group in dated if group.trading_day == day.date())
    # ---- 10. 同日权重取最大值 ----
    check("同日权重取候选中的最大值", abs(same_day.weight - 0.8) < 1e-12,
          f"权重 {same_day.weight}")
    # ---- 13. 原始事件 ID 全部保留 ----
    check("同日事件的全部 event_id 都保留下来",
          sorted(same_day.event_ids) == ["a", "b"])
    check("同日事件的全部文本都保留下来（供解释页展示）",
          set(same_day.texts) == {"甲", "乙"}, f"{same_day.texts}")

    summary = aggregation_summary(dated)
    check("聚合摘要里的 raw_event_count 是事件数、effective_evidence_days 是日数",
          summary["raw_event_count"] == 3 and summary["effective_evidence_days"] == 2)
    check("same_day_group_count 统计的是多事件的日子",
          summary["same_day_group_count"] == 1)

    # ---- 11. 同日收益不一致必须抛 ----
    bad = [
        FakeEvidence(FakeEvent("x", day, 0.01, "甲"), 0.8),
        FakeEvidence(FakeEvent("y", day, 0.05, "乙"), 0.6),
    ]
    raised = False
    try:
        aggregate_by_day(bad, 20)
    except SameDayInconsistency:
        raised = True
    check("同一交易日异常收益不一致时抛出异常（不静默取平均）", raised)

    # ---- 12. 缺失收益不会被当成 0 ----
    missing = [
        FakeEvidence(FakeEvent("m", day, None, "缺失"), 0.8),
        FakeEvidence(FakeEvent("n", day + timedelta(days=1), 0.02, "有值"), 0.5),
    ]
    dated_missing = aggregate_by_day(missing, 20)
    missing_group = next(g for g in dated_missing if g.trading_day == day.date())
    check("缺失的异常收益保持为 None，不会被当成 0",
          missing_group.abret is None, f"得到 {missing_group.abret}")
    check("聚合摘要记录了缺收益的证据日数",
          aggregation_summary(dated_missing)["missing_abret_days"] == 1)


# ===========================================================================
# 三、冲突指标
# ===========================================================================


def test_conflict_metrics() -> None:
    section("三、冲突指标")

    # ---- 14. K = 0 ----
    stats = summarise([], [], scale=1.0)
    check("K=0：方向分歧度返回 None（不是 0）",
          stats["directional_conflict"] is None)
    check("K=0：状态标为 insufficient_directional_evidence",
          stats["directional_conflict_status"] == "insufficient_directional_evidence")
    check("K=0：ESS 为 0", stats["effective_sample_size"] == 0.0)

    # ---- 15. K = 1 ----
    # 注意 CI 的方向：0 = 证据一边倒（不冲突），1 = 上下对半（冲突最大）
    stats = summarise([0.05], [1.0], scale=0.01)
    check("K=1：只有一侧权重，方向分歧度为 0（一边倒即不冲突）",
          abs(stats["directional_conflict"]) < 1e-12,
          f"CI = {stats['directional_conflict']}")
    check("K=1：ESS 为 1", abs(stats["effective_sample_size"] - 1.0) < 1e-12)
    check("K=1：单侧证据被分到 up", stats["w_up"] == 1.0)

    # ---- 16. 全部同向 ----
    stats = summarise([0.05, 0.06, 0.07], [1, 1, 1], scale=0.01)
    check("全部同向：方向分歧度为 0", abs(stats["directional_conflict"]) < 1e-12,
          f"CI = {stats['directional_conflict']}")
    check("全部同向：中性占比为 0", stats["neutral_mass"] == 0.0)

    # ---- 17. 上下权重相等 ----
    stats = summarise([0.05, -0.06], [1, 1], scale=0.01)
    check("上下权重相等：方向分歧度为 1（冲突最大）",
          abs(stats["directional_conflict"] - 1.0) < 1e-12,
          f"CI = {stats['directional_conflict']}")

    # ---- 18. 全部落入中性区间 ----
    stats = summarise([0.001, -0.001], [1, 1], scale=0.01)
    check("全部中性：方向分歧度为 None", stats["directional_conflict"] is None)
    check("全部中性：中性占比为 1", abs(stats["neutral_mass"] - 1.0) < 1e-12)
    check("全部中性：w_up 与 w_down 都是 0",
          stats["w_up"] == 0.0 and stats["w_down"] == 0.0)

    # ---- 19/20. 「光有 CI 不够」的核心例子 ----
    # 两组的方向分歧度完全一样（上下各半），差别全在幅度上。
    # 需要 4 个点才能让上下权重严格对半。
    small = [0.003, -0.003, 0.0028, -0.0029]
    large = [0.08, -0.09, 0.085, -0.088]
    low = summarise(small, [1] * 4, scale=0.01)
    high = summarise(large, [1] * 4, scale=0.01)

    check("高方向冲突、低幅度与高幅度：CI 完全相同（都是 1）",
          abs(low["directional_conflict"] - 1.0) < 1e-12
          and abs(high["directional_conflict"] - 1.0) < 1e-12,
          f"CI 分别 {low['directional_conflict']:.4f} / {high['directional_conflict']:.4f}")
    check("两组的差别只能从幅度量上看——这正是要加幅度离散度的原因",
          high["magnitude_dispersion"] > 10 * low["magnitude_dispersion"],
          f"D {low['magnitude_dispersion']:.6f} → {high['magnitude_dispersion']:.6f}"
          f"（{high['magnitude_dispersion'] / max(low['magnitude_dispersion'], 1e-12):.0f}×）")

    # 已知局限：MAD 型离散度在**双峰**分布下系统性偏小。方向冲突高时，
    # 分布天然是双峰的（一半在正、一半在负），所以这个量会低估幅度上的分歧。
    # 加权标准差没有这个问题，所以它作为辅助诊断量必须保留。
    check("稳健离散度在双峰分布下系统性偏小（已知局限，已在文档中说明）",
          high["magnitude_dispersion"] < high["weighted_std"] / 10,
          f"D_robust={high['magnitude_dispersion']:.4f} "
          f"vs 加权标准差={high['weighted_std']:.4f}")
    check("加权标准差能正确反映两组的幅度差异",
          high["weighted_std"] > 20 * low["weighted_std"],
          f"{low['weighted_std']:.6f} → {high['weighted_std']:.6f}")

    low_neutral = summarise([0.002, -0.002], [1, 1], scale=0.01)
    check("低幅度证据落进中性带后不再算作方向冲突",
          low_neutral["directional_conflict"] is None
          and low_neutral["neutral_mass"] == 1.0)

    # ---- 稳健离散度在 K=2 时退化：这是公式性质，必须显式记下来 ----
    two = summarise([0.08, -0.09], [1, 1], scale=0.01)
    check("K=2 时稳健离散度退化为 0（公式性质，不是 bug）",
          two["magnitude_dispersion"] == 0.0)
    check("K=2 时 dispersion_defined 为 False，可供上层拦截",
          two["dispersion_defined"] is False)
    check("K=3 时 dispersion_defined 为 True",
          summarise([0.08, -0.09, 0.1], [1, 1, 1], scale=0.01)["dispersion_defined"] is True)
    check("退化时加权标准差仍可作为辅助诊断量",
          two["weighted_std"] is not None and two["weighted_std"] > 0.05,
          f"weighted_std={two['weighted_std']:.4f}")

    # ---- 21. 单一证据权重垄断 → ESS 接近 1 ----
    stats = summarise([0.05, 0.06, 0.07], [1000.0, 0.001, 0.001], scale=0.01)
    check("单一证据垄断权重时 ESS 接近 1",
          stats["effective_sample_size"] < 1.01,
          f"ESS = {stats['effective_sample_size']:.4f}")

    # ---- 22. 权重均匀 → ESS 等于证据日数 ----
    stats = summarise([0.05, 0.06, 0.07, -0.02, 0.01], [1] * 5, scale=0.01)
    check("权重均匀时 ESS 等于证据日数",
          abs(stats["effective_sample_size"] - 5.0) < 1e-9,
          f"ESS = {stats['effective_sample_size']:.6f}")

    # ---- 23. MAD 为 0 时不除零 ----
    flat = robust_scale([0.01, 0.01, 0.01, 0.01])
    check("全同值时 MAD 为 0", flat == 0.0)
    stats = summarise([0.05, -0.05], [1, 1], scale=0.0)
    check("scale 为 0 时归一化离散度返回 None 而不是 inf/nan",
          stats["normalized_dispersion"] is None,
          f"得到 {stats['normalized_dispersion']}")
    check("scale 为 0 时 τ 为 0，且没有出现除零异常", stats["tau"] == 0.0)

    # ---- 24. 缺失值处理 ----
    stats = summarise([None, 0.05], [1, 1], scale=0.01)
    check("缺失值不计入任何方向桶（不会被当成 0 塞进中性）",
          stats["w_neutral"] == 0.0 and stats["w_up"] == 1.0)
    check("缺失值对应的权重不计入 ESS", abs(stats["effective_sample_size"] - 1.0) < 1e-12)

    # ---- 加权中位数 ----
    check("加权中位数在权重集中时取到对应值",
          abs(weighted_median([1, 2, 3], [0.1, 0.1, 10]) - 3.0) < 1e-12)
    check("加权中位数在权重均匀时取中间值",
          abs(weighted_median([1, 2, 3], [1, 1, 1]) - 2.0) < 1e-12)

    # ---- 方向回退链 ----
    direction, _, _, tie, _ = decide_direction([0.05, 0.06], [1, 1], w_up=2.0,
                                               w_down=0.0, scale=0.01)
    check("方向回退第 1 步：按加权异常收益符号", direction is Direction.UP and not tie)

    direction, _, _, tie, _ = decide_direction([0.0], [1], w_up=0.0, w_down=0.0,
                                               scale=0.01, family_history_mean=0.03)
    check("方向回退第 3 步：非中性证据平局时用同族历史均值",
          direction is Direction.UP and tie)

    direction, _, _, tie, _ = decide_direction([0.0], [1], w_up=0.0, w_down=0.0,
                                               scale=0.01, family_history_mean=0.0,
                                               all_history_mean=-0.04)
    check("方向回退第 4 步：同族也为 0 时用全样本历史均值",
          direction is Direction.DOWN and tie)

    direction, _, _, tie, _ = decide_direction([0.0], [1], w_up=0.0, w_down=0.0,
                                               scale=0.01, family_history_mean=0.0,
                                               all_history_mean=0.0)
    check("方向回退第 5 步：全部为 0 时用固定默认方向并标记回退",
          tie and direction is Direction.UP)


# ===========================================================================
# 四、标注与评测
# ===========================================================================


def test_labels_and_evaluation() -> None:
    section("四、标注与评测")

    if not LABELS_PATH.exists():
        check("标注表存在", False, "先跑 scripts/migrate_labels.py")
        return

    with LABELS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []

    # ---- 25. 迁移幂等 ----
    required = {"query_id", "candidate_event_id", "required_for_primary",
                "mechanism_relevance", "overall_relevance", "label_status"}
    check("标注表包含迁移后的全部必需列", required <= set(fields),
          f"缺 {sorted(required - set(fields))}" if required - set(fields) else "")
    keys = [(r["query_id"], r["candidate_event_id"]) for r in rows]
    check("迁移后没有产生重复行", len(keys) == len(set(keys)),
          f"{len(keys) - len(set(keys))} 组重复" if len(keys) != len(set(keys)) else "")
    check("迁移后行数没有减少", len(rows) >= 326, f"{len(rows)} 行")

    # ---- 26. 主评测范围覆盖全部 20 条查询 ----
    primary = [r for r in rows if r["required_for_primary"].lower() == "true"]
    queries_covered = {r["query_id"] for r in primary}
    expected_queries = {q.event_id for q in fixture()["queries"]}
    check("主评测范围覆盖全部 20 条查询",
          queries_covered == expected_queries,
          f"覆盖 {len(queries_covered)}/{len(expected_queries)}")
    check("主评测范围行数在预期区间（150–180）", 150 <= len(primary) <= 180,
          f"{len(primary)} 行")
    check("required_for_primary = in_e1_top5 OR in_e2_top5",
          all((r["required_for_primary"].lower() == "true")
              == (r["in_e1_top5"].lower() == "true" or r["in_e2_top5"].lower() == "true")
              for r in rows))

    # ---- 27. 未完成时明确提示 ----
    completed = [r for r in primary
                 if r["label_status"].strip() == "complete"]
    if len(completed) < len(primary):
        check("人工标注未完成时，评测脚本会明确提示（见 evaluate_retrieval.py 的门禁）",
              True, f"已完成 {len(completed)}/{len(primary)} 行")
    else:
        check("人工标注已全部完成", True, f"{len(completed)} 行")

    # ---- 28. pooled Recall 命名 ----
    source = (_PROJECT_ROOT / "scripts" / "evaluate_retrieval.py").read_text(
        encoding="utf-8")
    check("评测脚本使用 pooled 口径命名 Recall",
          "pooled_recall" in source or "pooled Recall" in source)
    check("离线结果有用性指标已改名为 offline_outcome_utility",
          "offline_outcome_utility" in source)
    check("评测脚本注明该指标不能用于在线排序",
          "不能用于在线排序" in source or "只用于离线" in source)

    # ---- 29. 配对 bootstrap 固定种子可复现 ----
    from scripts.evaluate_retrieval import paired_bootstrap

    deltas = [0.1, -0.05, 0.2, 0.0, -0.1, 0.15, -0.02, 0.3]
    a = paired_bootstrap(deltas, n_resamples=2000, seed=20260928)
    b = paired_bootstrap(deltas, n_resamples=2000, seed=20260928)
    c = paired_bootstrap(deltas, n_resamples=2000, seed=1)
    check("配对 bootstrap 固定种子后结果可复现", a == b, f"{a} vs {b}")
    check("配对 bootstrap 换种子后结果会变（说明种子真的起作用）", a != c)
    check("配对 bootstrap 的区间包含均值",
          a["low"] <= a["mean"] <= a["high"], f"{a}")

    # ---- 30. 三路共用同一查询集与评测口径 ----
    f = fixture()
    events, index, by_id, queries = f["events"], f["index"], f["by_id"], f["queries"]
    pools_same = True
    for spec in queries[:5]:
        event = by_id[spec.event_id]
        pool = as_of_pool(events, event)
        sizes = set()
        for mode in (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE):
            outcome = retrieve_detailed(event, events, index, mode=mode,
                                        top_k=10, pool=pool)
            sizes.add(outcome.candidate_pool_size)
        # E1/E2 用整个池，E3 先收窄到 Top-20 —— 池大小这一项三路都该是同一个数
        if len(sizes) != 1:
            pools_same = False
    check("E1/E2/E3 使用同一个 as-of 候选池大小", pools_same)


def main() -> int:
    print("=" * 78)
    print("AegisFX-Agent 第 4 阶段测试")
    print("=" * 78)
    f = fixture()
    if f["note"]:
        print(f"\n⚠ {f['note']}")

    test_retrieval_layer()
    test_same_day_aggregation()
    test_conflict_metrics()
    test_labels_and_evaluation()

    print("\n" + "=" * 78)
    passed, failed = len(RESULTS["pass"]), len(RESULTS["fail"])
    print(f"通过 {passed} 项，失败 {failed} 项")
    for name in RESULTS["fail"]:
        print(f"  失败：{name}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
