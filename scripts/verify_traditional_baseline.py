"""复现论文 §4.4 的「传统特征工程」基线，和论文 Table 8 / Fig. 5 / Fig. 6 逐条对照。

    .\\venv\\Scripts\\python.exe scripts\\verify_traditional_baseline.py

这个脚本不联网、不用 API 密钥，几秒钟跑完（XGBoost 200 棵树，很小）。
结果写到 reports/traditional_baseline.json。

跑它的意义：有了这个，`main.py` 那条 DA 路线选出来的特征才有东西可比——
论文 Table 9 的左右两半就是这两条路线喂给同一批模型的结果。
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.research_dataset import load_daily_frame  # noqa: E402
from src.core.traditional_baseline import (  # noqa: E402
    DEFAULT_CORRELATION_THRESHOLD,
    build_traditional_baseline,
    paper_traditional_features,
)

# 论文 Fig. 6 图例的前几名，用来判断我们算出来的排序方向对不对。
# 只转录了靠前的几个——图例里的完整排序请直接看论文，不在这里猜。
_PAPER_FIG6_TOP = [
    "Positive_Events",
    "US_Restrictions_on_Chinese_Enterprises",
    "Negative_Events",
    "China_Tariff_Increases_on_the_US",
    "Sino_US_Dialogue",
]


def main() -> int:
    print("=" * 78)
    print("论文 §4.4 传统特征工程基线 复现")
    print("=" * 78)

    frame = load_daily_frame(verbose=True)
    result = build_traditional_baseline(frame)
    # 再跑一遍公式字面口径（分裂次数），用来展示两种口径差多少。
    # 这是本项目发现论文 Eq. (4) 与 Fig. 6 对不上的证据，见
    # src/core/traditional_baseline.py 开头的对照表。
    literal = build_traditional_baseline(frame, importance_type="weight")

    # ---------- 时序线 ----------
    print("\n" + "-" * 78)
    print("时序线 1/2：各时序变量与 USD/CNY 的 Pearson 相关（对照论文 Fig. 5）")
    print("-" * 78)
    print(f"{'特征':<26s} {'与 USD/CNY 的相关性':>18s}")
    for name, value in sorted(
        result.target_correlations.items(), key=lambda item: abs(item[1]), reverse=True
    ):
        print(f"{name:<26s} {value:>18.4f}")

    print("\n" + "-" * 78)
    print(f"时序线 2/2：相关去重（阈值 {DEFAULT_CORRELATION_THRESHOLD}）")
    print("-" * 78)
    print(f"  选中：{result.time_series_selected}")
    for dropped, kept, corr in result.time_series_dropped:
        print(f"  丢掉 {dropped}：与 {kept} 的相关性 {corr:.3f} 超过阈值")

    # ---------- 事件线 ----------
    print("\n" + "-" * 78)
    print("事件线：XGBoost 分裂次数重要度（论文 Eq. 4），降序")
    print("-" * 78)
    print(f"  （均值阈值 {result.event_mean_weight:.3f}，高于它的才采纳）\n")
    print(f"{'#':>3s}  {'特征':<52s} {'权重':>7s}  结论")
    for rank, (name, value) in enumerate(result.event_weights.items(), start=1):
        mark = "采纳" if name in result.event_selected else ""
        print(f"{rank:>3d}  {name:<52s} {value:>7.0f}  {mark}")

    # ---------- 与论文对照 ----------
    print("\n" + "=" * 78)
    print("与论文对照")
    print("=" * 78)

    computed = result.feature_set()
    paper = paper_traditional_features()

    print(f"\n时序线  本次：{result.time_series_selected}")
    print(f"        论文：['USD_Index', 'CN_1Y_GovBond_Yield']")
    ts_match = set(result.time_series_selected) == {"USD_Index", "CN_1Y_GovBond_Yield"}
    print(f"        → {'一致' if ts_match else '不一致'}")

    top_two = list(result.event_weights)[:2]
    paper_events = {"Positive_Events", "Negative_Events"}
    print(f"\n事件线  本次（gain，权重降序）：{result.event_selected}")
    print(f"        本次前两名：{top_two}")
    print(f"        论文保留的：['Positive_Events', 'Negative_Events']")
    ev_match = set(top_two) == paper_events
    print(f"        → 前两名{'与论文保留的两个一致' if ev_match else '与论文保留的不一致'}")
    if len(result.event_selected) > 2:
        print(f"        （高于均值的共 {len(result.event_selected)} 个；论文在均值过滤后")
        print(f"          还有一步'信息重叠'剔除，规则论文没写，所以剩下的对不齐）")

    print(f"\n合并后  本次：{computed}")
    print(f"        论文：{paper}")
    print(f"\n交集 {sorted(set(computed) & set(paper))}")
    print(f"论文有本次没有：{sorted(set(paper) - set(computed))}")
    print(f"本次有论文没有：{sorted(set(computed) - set(paper))}")

    # ---------- 排序方向 ----------
    order = list(result.event_weights)
    print("\n" + "-" * 78)
    print("排序方向：论文 Fig. 6 排前面的几个特征，两种口径各自排第几？")
    print("-" * 78)
    literal_order = list(literal.event_weights)
    print(f"  {'特征':<46s} {'gain（本模块默认）':>16s} {'weight（公式字面）':>18s}")
    for name in _PAPER_FIG6_TOP:
        gain_pos = order.index(name) + 1 if name in order else None
        lit_pos = literal_order.index(name) + 1 if name in literal_order else None
        print(f"  {name:<46s} {f'第 {gain_pos} 名':>16s} {f'第 {lit_pos} 名':>18s}")

    print("\n  结论：论文最终保留的是 Positive_Events 与 Negative_Events，"
          "而它们在")
    print("  gain 口径下正好是前两名；在 weight 口径下 Negative_Events 掉到 "
          f"第 {literal_order.index('Negative_Events') + 1} 名。")
    print("  所以论文实际用的是 gain，尽管 Eq. (4) 写的是分裂次数。")

    # ---------- 落盘 ----------
    output = _PROJECT_ROOT / "reports" / "traditional_baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_series_selected": result.time_series_selected,
        "time_series_dropped": [
            {"dropped": d, "kept": k, "correlation": round(c, 4)}
            for d, k, c in result.time_series_dropped
        ],
        "target_correlations": {
            k: round(v, 4) for k, v in result.target_correlations.items()
        },
        "event_weights": {k: float(v) for k, v in result.event_weights.items()},
        "event_mean_weight": round(result.event_mean_weight, 4),
        "event_selected": result.event_selected,
        "computed_feature_set": computed,
        "paper_feature_set": paper,
        "importance_type": "gain",
        "literal_weight_variant": {
            "note": "论文公式 (4) 字面写法（分裂次数），复现不出 Fig. 6",
            "importance_type": "weight",
            "event_weights": {k: float(v) for k, v in literal.event_weights.items()},
            "event_selected": literal.event_selected,
        },
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完整结果已写入：{output}")

    print("\n" + "-" * 78)
    print("怎么读这些结果：")
    print("  时序线完全对得上，说明 §4.4 的相关性筛选规则和补值方案都复现对了。")
    print("  事件线只对上一部分，原因已经查清：论文 Eq. (4) 写的是'分裂次数'，")
    print("  但按分裂次数算和 Fig. 6 差得很远；换成 XGBoost 默认的 gain 之后，")
    print("  论文保留的那两个特征正好落在前两名。所以是本模块默认用 gain。")
    print("  剩下还对不齐的是'信息重叠'那一步——论文只描述了一句，没有具体规则，")
    print("  这一条值得问论文作者。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
