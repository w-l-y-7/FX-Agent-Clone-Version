"""建立历史事件库（第 2 周交付物）。

    .\\venv\\Scripts\\python.exe scripts\\build_event_store.py

做四件事：日期顺延到交易日 → 市场模型算异常收益 → 记事件日前一日的市场状态
→ 按 2023 年切分成检索池与评测集。

不用密钥。默认离线（FRED 只在有缓存时才读），几秒钟跑完。
结果写到 `data/aegisfx/historical_events.jsonl`，
统计写到 `reports/aegisfx_event_store.json`。
"""

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import (  # noqa: E402
    STORE_PATH,
    build_event_store,
    store_summary,
)
from src.aegisfx.data.market import describe_sources, extended_usdcny  # noqa: E402
from src.aegisfx.taxonomy import family_label, label  # noqa: E402

_REPORT = _PROJECT_ROOT / "reports" / "aegisfx_event_store.json"


def main() -> int:
    print("=" * 78)
    print("建立历史事件库")
    print("=" * 78)
    print(describe_sources())
    print()

    events = build_event_store()
    summary = store_summary(events)

    print(f"\n落库 {summary['total']} 条 → {STORE_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"  检索池（2017–{summary['split']['eval_from_year'] - 1}）  "
          f"{summary['split']['pool']} 条")
    print(f"  评测集（{summary['split']['eval_from_year']}–2024）  "
          f"{summary['split']['eval']} 条")

    print("\n标签状态")
    for status, count in sorted(summary["label_status"].items()):
        print(f"  {status:<10} {count:>4}")

    print("\n异常收益（全样本）")
    print(f"  {'窗口':<8}{'条数':>6}{'均值':>12}{'标准差':>12}{'上涨占比':>10}")
    for horizon in (1, 5, 20):
        stats = summary[f"abret_{horizon}d"]
        print(f"  {horizon:>2} 日   {stats['n']:>6}{stats['mean']:>12.4%}"
              f"{stats['std']:>12.4%}{stats['share_up']:>10.1%}")

    print("\n原始收益 vs 异常收益（20 日，全样本）")
    raw, abn = summary["raw_ret_20d"], summary["abret_20d"]
    print(f"  原始    均值 {raw['mean']:+.4%}  标准差 {raw['std']:.4%}  "
          f"上涨占比 {raw['share_up']:.1%}")
    print(f"  异常    均值 {abn['mean']:+.4%}  标准差 {abn['std']:.4%}  "
          f"上涨占比 {abn['share_up']:.1%}")

    print("\n评测集的多数类基准（这是所有方向准确率的地板）")
    eval_raw, eval_abn = summary["eval_raw_ret_20d"], summary["eval_abret_20d"]
    print(f"  原始收益口径  一律猜主流方向 = {max(eval_raw['share_up'], 1 - eval_raw['share_up']):.1%}"
          f"  （n={eval_raw['n']}）")
    print(f"  异常收益口径  一律猜主流方向 = {max(eval_abn['share_up'], 1 - eval_abn['share_up']):.1%}"
          f"  （n={eval_abn['n']}）")
    print("  → 换成异常收益口径之后，基准从七成掉到五成上下，方向预测才重新成为真问题。")

    print("\n按事件族（20 日异常收益）")
    for family, stats in sorted(summary["by_family"].items(),
                                key=lambda kv: -(kv[1].get("n") or 0)):
        if not stats.get("n"):
            print(f"  {family_label(family):<10} 无可用样本")
            continue
        print(f"  {family_label(family):<10} n={stats['n']:>4}  "
              f"均值 {stats['mean']:+.4%}  标准差 {stats['std']:.4%}  "
              f"上涨占比 {stats['share_up']:.1%}")

    print("\n市场模型诊断（20 日尺度）")
    model = summary["model"]
    print(f"  平均 R²              {model['mean_r2']:.4f}")
    print(f"  平均方差压缩          {model['mean_variance_reduction']:.1%}"
          "   ← 模型有没有用的判据是它，不是 R²")
    print(f"  平均 β(美元指数)       {model['mean_beta_dxy']:+.4f}")
    print(f"  平均 β(中美利差变化)    {model['mean_beta_spread']:+.4f}")
    print(f"  估计口径分布          {summary['method']}")
    fallback = summary["method"].get("mean_adjusted", 0)
    if fallback:
        print(f"  注：{fallback} 条走了均值调整降级口径，看 β 的统计时要把它们排除")

    print("\n各类样本量最少的 5 类（第 3 周做检索时要留意，样本太少的类检索不稳）")
    from collections import Counter
    counts = Counter(e.event_type for e in events)
    for snake, count in counts.most_common()[-5:]:
        print(f"  {label(snake):<24} {count}")

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n统计已写入 {_REPORT.relative_to(_PROJECT_ROOT)}")

    extension = extended_usdcny()
    if len(extension):
        print(f"\nFRED 扩展段有 {len(extension)} 个交易日可用于给新事件补标签"
              f"（{extension.index[0].date()} → {extension.index[-1].date()}）")
    else:
        print("\nFRED 扩展段没有缓存——联网跑一次会自动下载，"
              "新事件补标签要用到它（见 src/aegisfx/data/market.py）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
