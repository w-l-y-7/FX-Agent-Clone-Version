"""复现 PA2 的「新闻 -> 日频事件哑变量」那一步，并和 Data sheet 逐列对照。

    .\\venv\\Scripts\\python.exe scripts\\verify_pa2_aggregation.py

不需要 API 密钥，不联网，几秒钟跑完。

`Data.xlsx` 里两个 sheet 是上下游关系：`Sheet1` 是 445 条带标签的新闻，
`Data` 是 1944 行的日频表（含 25 个 0/1 事件列）。中间那步折叠公开代码里没有，
本脚本把它补上，然后量出两边差多少。

会跑三种「新闻落在非交易日时怎么办」的口径（exact / next / previous），
把结果排在一起对比，最后挑重合度最高的那个写成报告。
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.event_aggregation import (  # noqa: E402
    DATE_POLICIES,
    format_report,
    run_all_policies,
)

_DEFAULT_REPORT = _PROJECT_ROOT / "reports" / "pa2_aggregation.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="复现 PA2 的事件聚合，并与 Data sheet 对照"
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()

    print("正在读 Data.xlsx 的两个 sheet 并做聚合……\n")
    reports = run_all_policies()

    for policy in DATE_POLICIES:
        report = reports[policy]
        print("=" * 78)
        print(format_report(report))
        print()

    best = max(reports.values(), key=lambda item: item.mean_jaccard)
    print("=" * 78)
    print(f"重合度最高的口径：{best.policy}（平均交并比 {best.mean_jaccard:.3f}）")
    print(f"  完全一致的列 {len(best.matched_columns)} / {len(best.columns)}：")
    for item in best.matched_columns:
        print(f"    {item.column}（{item.observed} 天）")

    _write(args.output, reports, best)

    print("\n" + "-" * 78)
    print("怎么读这张表：")
    print("  「实际」列 = Data sheet 里的值，也是论文 Table 5 的 sample size")
    print("     （Negative events 248 / Positive events 181 / US restrictions 160 …）")
    print("  「复现」列 = 用 Sheet1 的 445 条新闻聚合出来的值")
    print("  两列对不上是**预期之内**的：Sheet1 的 445 条和 Data 的 429 个事件日")
    print("  不是同一个口径，Data 那一侧覆盖的语料更多，Sheet1 只是其中一个快照。")
    print("  所以这里的目标是量出差距、并说清差在哪，而不是假装能完全复现。")
    return 0


def _write(path: Path, reports, best) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "PA2 聚合复现的对照结果。Sheet1 的 445 条新闻与 Data sheet 的 429 个"
            "事件日不是同一口径，两边不可能完全一致；本文件记录的是差多少、差在哪。"
        ),
        "best_policy": best.policy,
        "policies": {
            policy: {
                "policy": report.policy,
                "news_total": report.news_total,
                "news_mapped": report.news_mapped,
                "mean_jaccard": round(report.mean_jaccard, 4),
                "matched_columns": [item.column for item in report.matched_columns],
                "columns": [
                    {
                        "column": item.column,
                        "reconstructed": item.reconstructed,
                        "observed": item.observed,
                        "both_one": item.both_one,
                        "only_reconstructed": item.only_reconstructed,
                        "only_observed": item.only_observed,
                        "jaccard": round(item.jaccard, 4),
                    }
                    for item in report.columns
                ],
            }
            for policy, report in reports.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完整对照结果已写入：{path}")


if __name__ == "__main__":
    raise SystemExit(main())
