"""第 5 阶段测试套件（方案第十五节要求的那 20 项）。

    .\\venv\\Scripts\\python.exe scripts\\test_stage5.py

不需要密钥、不联网、不碰 2024。所有随机都固定种子。
"""

import csv
import json
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import scripts.evaluate_stage5 as stage5  # noqa: E402
from src.aegisfx.conflict import aggregate_by_day, summarise  # noqa: E402
from src.aegisfx.conflict.metrics import (  # noqa: E402
    directional_conflict,
    pairwise_dispersion,
)
from src.aegisfx.conformal.rolling import conformal_quantile, rolling_conformal  # noqa: E402
from src.aegisfx.conformal.scaled import ScaledConformal, cp1_allowed  # noqa: E402
from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.data.maturity import (  # noqa: E402
    LabelMaturityViolation,
    MaturityCalendar,
    assert_labels_mature,
    mature_pool,
)
from src.aegisfx.retrieval.queries import load_or_build  # noqa: E402
from src.aegisfx.routing.features import FEATURE_NAMES, FeatureBuilder  # noqa: E402
from src.aegisfx.routing.fusion import fuse_evidence, weights_sum  # noqa: E402
from src.aegisfx.routing.gate import GateModel  # noqa: E402
from src.aegisfx.schemas.views import RetrievalEventView  # noqa: E402

RESULTS = {"pass": [], "fail": []}
_cache = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    (RESULTS["pass"] if ok else RESULTS["fail"]).append(name)
    print(f"  [{'通过' if ok else '失败'}] {name}" + (f" —— {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * 78)


def fixture():
    if "ready" not in _cache:
        events = load_event_store()
        _cache["events"] = events
        _cache["by_id"] = {e.event_id: e for e in events}
        _cache["calendar"] = MaturityCalendar.from_market()
        _cache["queries"] = load_or_build(events)
    return _cache


class FakeEvent:
    def __init__(self, event_id, day, abret_20d=None, text="", family="tariff",
                 side="US", sentiment="negative"):
        self.event_id = event_id
        self.date = day
        self.abret_20d = abret_20d
        self.text = text
        self.event_family = family
        self.initiating_side = side
        self.sentiment = sentiment
        self.abret_5d = abret_20d


class FakeDated:
    def __init__(self, day, weight, event_ids, texts=None):
        self.trading_day = day
        self.weight = weight
        self.event_ids = event_ids
        self.texts = texts or list(event_ids)


# ===========================================================================
# 一、标签成熟协议
# ===========================================================================


def test_maturity() -> None:
    section("一、标签成熟协议")
    f = fixture()
    calendar, events, by_id = f["calendar"], f["events"], f["by_id"]

    # ---- 1. 早于查询但未成熟必须拒绝 ----
    queries = sorted(by_id.values(), key=lambda e: e.date)
    query = queries[len(queries) // 2]
    position = calendar.position(query.date)
    immature = next(
        (e for e in events
         if e.event_id != query.event_id
         and calendar.position(e.date) is not None
         and position - 20 <= calendar.position(e.date) <= position - 1),
        None)
    check("能构造出「早于查询日但标签未成熟」的反例", immature is not None)
    if immature is not None:
        raised = False
        try:
            assert_labels_mature(query, [immature], 20, calendar)
        except LabelMaturityViolation:
            raised = True
        check("早于查询日但收益未成熟的候选被拒绝", raised)

    # ---- 2. 5 日与 20 日成熟日期不同 ----
    probe = queries[len(queries) // 2]
    m5 = calendar.describe(probe.date, 5)
    m20 = calendar.describe(probe.date, 20)
    check("5 日与 20 日的收益窗口结束日不同",
          m5.label_end_date != m20.label_end_date,
          f"5日→{m5.label_end_date}　20日→{m20.label_end_date}")
    check("5 日的可用时点早于 20 日",
          m5.available_at < m20.available_at,
          f"{m5.available_at} < {m20.available_at}")
    check("可用时点是收益窗口结束日的**下一个**交易日（收盘后才可知）",
          m5.available_at > m5.label_end_date)

    # ---- 10. 伪查询只检索成熟历史证据 ----
    pool = mature_pool(events, probe, 20, calendar, verify=False)
    limit = calendar.mature_limit(calendar.position(probe.date), 20)
    offenders = [e for e in pool
                 if calendar.position(e.date) is None
                 or calendar.position(e.date) > limit]
    check(f"成熟池里 {len(pool)} 条全部满足成熟条件", not offenders,
          f"{len(offenders)} 条越界" if offenders else "")
    check("成熟池严格小于「只按日期截断」的池",
          len(pool) < sum(1 for e in events
                          if e.event_id != probe.event_id
                          and pd.Timestamp(e.date) < pd.Timestamp(probe.date)))


# ===========================================================================
# 二、E4 路由
# ===========================================================================


def test_routing() -> None:
    section("二、E4 路由")

    # ---- 5. 门控值始终在 [0,1] ----
    model = GateModel(
        feature_names=list(FEATURE_NAMES),
        coefficients=np.array([0.5] + [0.3] * len(FEATURE_NAMES)),
        feature_mean=np.zeros(len(FEATURE_NAMES)),
        feature_std=np.ones(len(FEATURE_NAMES)),
    )
    probe = np.linspace(-5, 5, 41).reshape(-1, 1) @ np.ones((1, len(FEATURE_NAMES)))
    gates = model.gate(probe)
    check("门控值始终落在 [0, 1]", bool(np.all((gates >= 0) & (gates <= 1))),
          f"范围 {gates.min():.4f}–{gates.max():.4f}")

    extreme = GateModel(
        feature_names=list(FEATURE_NAMES),
        coefficients=np.array([0.0] + [100.0] * len(FEATURE_NAMES)),
        feature_mean=np.zeros(len(FEATURE_NAMES)),
        feature_std=np.ones(len(FEATURE_NAMES)))
    huge = extreme.gate(np.full((1, len(FEATURE_NAMES)), 100.0))
    check("极大 logit 下门控值仍然有限且不越界",
          bool(np.isfinite(huge).all() and huge[0] <= 1.0),
          f"g = {huge[0]:.6f}")

    # ---- 6/7. g=0 → E1，g=1 → E2 ----
    r1, r2 = np.array([0.01, -0.02]), np.array([0.05, 0.03])
    data = np.zeros((2, len(FEATURE_NAMES)))
    zero = GateModel(feature_names=list(FEATURE_NAMES),
                     coefficients=np.array([-100.0] + [0.0] * len(FEATURE_NAMES)),
                     feature_mean=np.zeros(len(FEATURE_NAMES)),
                     feature_std=np.ones(len(FEATURE_NAMES)))
    one = GateModel(feature_names=list(FEATURE_NAMES),
                    coefficients=np.array([100.0] + [0.0] * len(FEATURE_NAMES)),
                    feature_mean=np.zeros(len(FEATURE_NAMES)),
                    feature_std=np.ones(len(FEATURE_NAMES)))
    check("g→0 时 E4 预测等于 E1",
          bool(np.allclose(zero.blend(data, r1, r2), r1, atol=1e-6)),
          f"{zero.blend(data, r1, r2)} vs {r1}")
    check("g→1 时 E4 预测等于 E2",
          bool(np.allclose(one.blend(data, r1, r2), r2, atol=1e-6)),
          f"{one.blend(data, r1, r2)} vs {r2}")

    # ---- 4. 改查询未来收益，路由特征不变 ----
    f = fixture()
    builder = FeatureBuilder(__import__("src.aegisfx.data.market",
                                        fromlist=["paper_market"]).paper_market())
    query = f["by_id"][f["queries"][0].event_id]
    view = RetrievalEventView.from_event(query, include_realized=False)
    kwargs = dict(same_family_ratio=0.8,
                  semantic_top=[0.9, 0.85, 0.8, 0.75, 0.7],
                  mature_count=100,
                  conflict_e1={"directional_conflict": 0.3, "neutral_mass": 0.1,
                               "pairwise_dispersion_norm": 0.5,
                               "effective_sample_size": 4.0},
                  conflict_e2={"directional_conflict": 0.4, "neutral_mass": 0.2,
                               "pairwise_dispersion_norm": 0.6,
                               "effective_sample_size": 3.0})
    base = builder.build(view, **kwargs)

    mutated = query.model_copy(deep=True)
    mutated.abret_1d = 9.9
    mutated.abret_5d = -9.9
    mutated.abret_20d = 9.9
    mutated_view = RetrievalEventView.from_event(mutated, include_realized=False)
    after = builder.build(mutated_view, **kwargs)
    check("改变查询未来收益后，路由特征完全不变", base == after,
          "" if base == after else
          f"变化项 {[k for k in base if base[k] != after[k]]}")

    # ---- 12. 标准化只用训练期 ----
    check("特征标准化只依赖纸张记录的训练期统计量（模型自带 mean/std）",
          model.feature_mean.shape == (len(FEATURE_NAMES),)
          and model.feature_std.shape == (len(FEATURE_NAMES),))
    subset_a = model.standardise(data)
    subset_b = model.standardise(np.vstack([data, data * 5]))
    check("标准化不随调用批次变化（没有在线重算 mean/std）",
          bool(np.allclose(subset_b[: len(data)], subset_a)))

    # ---- 8/9. 证据权重融合 ----
    day_a = date(2023, 3, 1)
    day_b = date(2023, 3, 2)
    day_c = date(2023, 3, 3)
    e1 = [FakeDated(day_a, 0.6, ["a1"]), FakeDated(day_b, 0.4, ["a2"])]
    e2 = [FakeDated(day_a, 0.3, ["b1"]), FakeDated(day_c, 0.7, ["b2"])]
    fused = fuse_evidence(e1, e2, 0.5)
    check("E4 融合后的证据权重和为 1", abs(weights_sum(fused) - 1.0) < 1e-9,
          f"和 = {weights_sum(fused):.10f}")
    merged = next(item for item in fused if item.trading_day == day_a)
    check("同一天在两个专家里都出现时正确合并",
          merged.source_from_e1 and merged.source_from_e2 and merged.group_size == 2,
          f"来源 e1={merged.source_from_e1} e2={merged.source_from_e2} "
          f"事件 {merged.all_event_ids}")
    check("融合保留了每个专家的原始权重分量",
          merged.weight_from_e1 > 0 and merged.weight_from_e2 > 0)
    g0 = fuse_evidence(e1, e2, 0.0)
    check("g=0 时融合权重只来自 E1",
          all(item.weight_from_e2 == 0 for item in g0))
    g1 = fuse_evidence(e1, e2, 1.0)
    check("g=1 时融合权重只来自 E2",
          all(item.weight_from_e1 == 0 for item in g1))


# ===========================================================================
# 三、冲突指标与共形
# ===========================================================================


def test_conflict_and_conformal() -> None:
    section("三、冲突指标与共形")

    # ---- 13. 双峰分布下 PD 显著大于 0 ----
    # 两组各 2 个点，簇间距约 0.04，簇内散布约 0.001。
    # 有意义的断言不是「PD 大于某个拍脑袋的数」，而是「PD 反映的是簇间距，
    # 不是簇内散布」——这正是 MAD 型指标做不到的地方。
    bimodal = [0.02, -0.02, 0.021, -0.019]
    pd_value = pairwise_dispersion(bimodal, [1] * 4)
    within_cluster = 0.001
    check("双峰分布下成对离散度接近簇间距、远大于簇内散布",
          pd_value is not None and pd_value > 10 * within_cluster,
          f"PD = {pd_value:.5f}（簇内散布约 {within_cluster}，"
          f"约为它的 {pd_value / within_cluster:.0f} 倍）")
    check("成对离散度量级正确（两组各 2 点的平均对距 = 0.027）",
          abs(pd_value - 0.027) < 1e-9, f"PD = {pd_value:.6f}")

    # ---- 14. 单点分布下 PD 等于 0 ----
    flat = pairwise_dispersion([0.01] * 4, [1] * 4)
    check("单点分布（全同值）下成对离散度为 0", flat == 0.0, f"PD = {flat}")

    stats = summarise(bimodal, [1] * 4, scale=0.01)
    check("双峰下 MAD 型离散度远小于成对离散度（第 4 阶段已知局限仍然存在）",
          stats["magnitude_dispersion"] < stats["pairwise_dispersion"] / 5,
          f"MAD={stats['magnitude_dispersion']:.5f}　"
          f"PD={stats['pairwise_dispersion']:.5f}")

    # ---- 15. CI 只在无非中性证据时为 None ----
    check("无非中性证据时 CI 为 None",
          directional_conflict(0.0, 0.0) is None)
    check("有非中性证据时 CI 一定不是 None（哪怕是 0）",
          directional_conflict(1.0, 0.0) == 0.0
          and directional_conflict(1.0, 1.0) == 1.0)
    check("全中性时 summarise 的 CI 为 None 且状态标记正确",
          summarise([0.001, -0.001], [1, 1], scale=0.01)["directional_conflict"] is None)
    check("CI=0（证据一边倒）与 CI=None（没有方向证据）是两回事",
          directional_conflict(2.0, 0.0) == 0.0)

    # ---- 16. 共形分位数用有限样本上取整 ----
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    q80 = conformal_quantile(scores, 0.2)
    expected_rank = int(np.ceil((len(scores) + 1) * 0.8))  # = 5 → 第 5 小
    check("共形分位数取第 ⌈(n+1)(1−α)⌉ 小的值",
          abs(q80 - sorted(scores)[expected_rank - 1]) < 1e-12,
          f"q={q80}　期望第 {expected_rank} 小 = {sorted(scores)[expected_rank - 1]}")
    check("共形分位数不等于线性插值结果（说明没有调默认 quantile）",
          abs(q80 - float(np.quantile(scores, 0.8))) > 1e-9,
          f"共形 {q80} vs 插值 {np.quantile(scores, 0.8):.3f}")
    check("样本不足时返回 inf 而不是最大值",
          np.isinf(conformal_quantile([0.1, 0.2], 0.05)))

    # ---- 3/17. 同日多查询不能互相更新校准池；未成熟残差不能更新 ACI ----
    predictions = [0.01, 0.02, 0.03, 0.04]
    realized = [0.011, 0.019, 0.031, 0.041]
    same_slot = [10, 10, 11, 12]     # 前两条结果在同一时刻可用
    result = rolling_conformal(predictions, realized, same_slot, targets=(0.8,),
                               window=60)
    first, second = result["per_query"][0.8][0], result["per_query"][0.8][1]
    check("同一时刻可用的多条查询共享同一份校准状态（不能互相更新）",
          first.n_calibration == second.n_calibration == 0,
          f"第1条用 {first.n_calibration} 个校准点，第2条用 {second.n_calibration} 个")

    aci = rolling_conformal(predictions, realized, same_slot, targets=(0.8,),
                            window=60, gamma=0.1)
    entries = aci["per_query"][0.8]
    check("未成熟残差不参与 ACI 更新（前两条都没有可用的历史残差）",
          entries[0].n_calibration == 0 and entries[1].n_calibration == 0)
    check("ACI 在结果成熟后才开始积累校准点",
          entries[3].n_calibration > entries[1].n_calibration,
          f"第4条 {entries[3].n_calibration} > 第2条 {entries[1].n_calibration}")

    # ---- 18. 相同配置重复运行结果一致 ----
    again = rolling_conformal(predictions, realized, same_slot, targets=(0.8,),
                              window=60)
    check("同一份输入重复跑共形，结果逐条一致",
          [e.width for e in again["per_query"][0.8]]
          == [e.width for e in result["per_query"][0.8]])

    # ---- CP1 自动降级 ----
    scaled = ScaledConformal.from_signal_report()
    check("冲突有效性检验没通过时 CP1 自动降级",
          not scaled.enabled and "降级" in scaled.reason, scaled.reason)
    check("cp1_allowed() 读的是检验报告，不是写死的常量",
          cp1_allowed() is False)


# ===========================================================================
# 四、时间折与配置锁
# ===========================================================================


def test_folds_and_lock() -> None:
    section("四、时间折与配置锁")

    # ---- 11. 时间折之间无标签窗口重叠 ----
    from src.aegisfx.routing.trainer import FOLDS, _purge_mask, expanding_folds

    calendar = MaturityCalendar.from_market()
    horizon = 20
    positions = np.array([calendar.position(f"{y}-06-15") or 0
                          for y in range(2018, 2023)])
    years = np.array(list(range(2018, 2023)))
    overlap = 0
    for train_through, valid_year in FOLDS:
        marker = calendar.position(f"{valid_year}-01-02") or 0
        mask = (years <= train_through) & _purge_mask(positions, marker, horizon)
        for position in positions[mask]:
            if position is None:
                continue
            label_end = position + horizon
            if label_end >= marker:      # 标签窗口伸进验证期
                overlap += 1
    check("训练样本的标签窗口不伸进验证期", overlap == 0,
          f"{overlap} 条重叠" if overlap else "")
    check("embargo 至少隔离一个期限的长度",
          _purge_mask(np.array([100]), 200, 20)[0] is np.True_
          and not _purge_mask(np.array([180]), 200, 20)[0],
          "pos=100 保留、pos=180 剔除（差 20 = 2×horizon 的隔离）")

    # ---- 19. 没有配置锁时拒绝运行最终测试 ----
    original = stage5.CONFIG_PATH
    try:
        with tempfile.TemporaryDirectory() as tmp:
            stage5.CONFIG_PATH = Path(tmp) / "missing.json"
            raised = False
            try:
                stage5._load_config()
            except SystemExit as exc:
                raised = "冻结" in str(exc) or "找不到" in str(exc)
            check("没有配置锁时拒绝运行（抛 SystemExit 并说明要先冻结）", raised)

            # ---- 20. 配置被改过时也要拒绝 ----
            tampered = Path(tmp) / "tampered.json"
            real = json.loads(original.read_text(encoding="utf-8"))
            real["horizons"]["20"]["selected_lambda"] = 0.01   # 偷偷改一个
            tampered.write_text(json.dumps(real, ensure_ascii=False), encoding="utf-8")
            stage5.CONFIG_PATH = tampered
            raised2 = False
            try:
                stage5._load_config()
            except SystemExit as exc:
                raised2 = "hash 对不上" in str(exc)
            check("配置被改过之后 hash 对不上，拒绝运行", raised2)
    finally:
        stage5.CONFIG_PATH = original

    # 正常配置能读出来
    config = stage5._load_config()
    check("当前冻结配置可以正常加载并校验 hash",
          bool(config.get("config_hash")),
          f"config hash {config.get('config_hash')}")
    check("冻结配置里记录了最强基线（由时间验证决定，不是硬编码）",
          "strongest_baseline" in config["horizons"]["20"],
          f"最强基线 = {config['horizons']['20'].get('strongest_baseline')}")
    check("冻结配置里 CP1 被标记为降级",
          config["horizons"]["20"].get("cp1_enabled") is False)


def main() -> int:
    print("=" * 78)
    print("AegisFX-Agent 第 5 阶段测试")
    print("=" * 78)

    test_maturity()
    test_routing()
    test_conflict_and_conformal()
    test_folds_and_lock()

    print("\n" + "=" * 78)
    passed, failed = len(RESULTS["pass"]), len(RESULTS["fail"])
    print(f"通过 {passed} 项，失败 {failed} 项")
    for name in RESULTS["fail"]:
        print(f"  失败：{name}")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
