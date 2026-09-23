"""单独跑 DA 的特征筛选，用来快速验证和对照论文。

    # 先花 10 分钟试跑 8 个特征，确认提示词没问题
    .\\venv\\Scripts\\python.exe scripts\\run_da_selection.py --limit 8

    # 跑完整候选集（32 个特征，第一次约 1 小时，之后有缓存就很快）
    .\\venv\\Scripts\\python.exe scripts\\run_da_selection.py

    # 强制不用缓存，看真实耗时
    .\\venv\\Scripts\\python.exe scripts\\run_da_selection.py --limit 8 --no-cache

结果会写到 reports/da_selection.json，同时在终端打印一张
和论文 Table 6 同形状的表。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.core.da_engine import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    EvidenceBasedSelector,
)
from src.core.research_dataset import (  # noqa: E402
    candidate_features,
    paper_selected_features,
)
from src.services.deepseek_llm_service import DeepSeekLLMService  # noqa: E402
from src.services.vector_rag_service import VectorRAGService  # noqa: E402

_DEFAULT_CACHE_DIR = _PROJECT_ROOT / ".cache" / "da"
_DEFAULT_REPORT = _PROJECT_ROOT / "reports" / "da_selection.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 DA 的基于证据的特征筛选")
    parser.add_argument("--limit", type=int, default=0, help="只评前 N 个特征，0 表示全部")
    parser.add_argument("--top-k", type=int, default=3, help="每次检索取多少条证据")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="归一化阈值")
    parser.add_argument("--max-attempts", type=int, default=3, help="每个特征最多重试几轮")
    parser.add_argument(
        "--normalization", choices=["theoretical", "batch"], default="theoretical",
        help="theoretical=除以理论满分（能复现论文 Table 6）；batch=论文公式(2)的字面写法",
    )
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek 模型名")
    parser.add_argument("--no-cache", action="store_true", help="不使用磁盘缓存")
    parser.add_argument("--output", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("没找到 DEEPSEEK_API_KEY，请检查 .env 文件。")
        return 1

    rag = VectorRAGService(verbose=True)
    count = rag.build()
    if count == 0:
        print("知识库是空的，DA 无法检索到任何证据。")
        print("请先跑：.\\venv\\Scripts\\python.exe scripts\\build_knowledge_base.py")
        return 1

    selector = EvidenceBasedSelector(
        rag_service=rag,
        llm_service=DeepSeekLLMService(api_key=api_key, model=args.model),
        top_k=args.top_k,
        max_attempts=args.max_attempts,
        weights=DEFAULT_WEIGHTS,
        threshold=args.threshold,
        normalization=args.normalization,
        cache_dir=None if args.no_cache else _DEFAULT_CACHE_DIR,
    )

    catalog = candidate_features()
    total = args.limit if args.limit > 0 else len(catalog)
    print(f"\n候选特征 {len(catalog)} 个，本次评估 {total} 个，"
          f"每个最多 {args.max_attempts} 轮，每次检索 {args.top_k} 条证据。")
    print(f"归一化 {args.normalization}，阈值 {args.threshold}，模型 {args.model}。")
    print("提示：每个特征要调 1-3 次 API，请耐心等待。\n")

    started = time.perf_counter()
    report = selector.select(catalog, limit=args.limit, progress=True)
    elapsed = time.perf_counter() - started

    stats = selector.cache_stats
    print(f"\n用时 {elapsed / 60:.1f} 分钟。缓存命中 {stats['hits']} 次、"
          f"未命中 {stats['misses']} 次（命中说明这次的调用没花钱）。")

    _print_table(report)
    _print_comparison(report)
    _write_report(report, args, elapsed, stats)
    return 0


def _print_table(report) -> None:
    print("\n=== 评审结果（对照论文 Table 6）===")
    header = f"{'特征':<52s} {'相关':>5s} {'支持':>5s} {'实用':>5s} {'得分':>6s}  结论"
    print(header)
    print("-" * len(header))
    for row in report.as_table():
        if row["relevance"] is None:
            print(f"{row['feature']:<52s} {'—':>5s} {'—':>5s} {'—':>5s} "
                  f"{row['score']:>6.1f}  淘汰（{row['note']}）")
            continue
        print(
            f"{row['feature']:<52s} {row['relevance']:>5.1f} "
            f"{row['supportiveness']:>5.1f} {row['utility']:>5.1f} "
            f"{row['score']:>6.1f}  {'采纳' if row['selected'] else '淘汰'}"
        )


def _print_comparison(report) -> None:
    expected = set(paper_selected_features())
    actual = set(report.selected_features)
    if not expected <= set(item.feature for item in report.evaluations):
        print("\n（本次只评了部分候选特征，无法与论文 Table 8 完整对照）")
        return

    print("\n=== 与论文 Table 8 对照 ===")
    print(f"论文选中的 4 个：{sorted(expected)}")
    print(f"本次选中的：    {sorted(actual)}")
    print(f"交集：{sorted(expected & actual)}")
    print(f"论文选了但本次没选：{sorted(expected - actual)}")
    print(f"本次选了但论文没选：{sorted(actual - expected)}")


def _write_report(report, args, elapsed: float, stats) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump()
    payload["_run"] = {
        "elapsed_seconds": round(elapsed, 1),
        "limit": args.limit,
        "top_k": args.top_k,
        "threshold": args.threshold,
        "max_attempts": args.max_attempts,
        "normalization": args.normalization,
        "model": args.model,
        "cache": stats,
        "paper_selected_features": paper_selected_features(),
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完整评审记录（含每条证据的出处）已写入：{args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
