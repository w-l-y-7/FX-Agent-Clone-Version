"""事件语料审计（第 1 周交付物之一）。

回答四个问题：

1. 这 445 条事件到底长什么样——按年、按类型、按情感、按发起方的分布，
   哪些类型样本太少、单独拿去检索会不稳；
2. 时间切分（检索池 2017–2022 / 评测集 2023–2024）落下来各是多少条；
3. 事件后 1/5/20 日的收益标签有多少条算得出来；
4. **两条解析路径在带标签语料上各有多准**——规则兜底不花 API 调用，
   大模型是主路。两者的差距决定了断网或没额度时系统会退化到什么程度。

    .\\venv\\Scripts\\python.exe scripts\\audit_event_corpus.py                  # 只测规则
    .\\venv\\Scripts\\python.exe scripts\\audit_event_corpus.py --llm-sample 50    # 再抽 50 条测大模型

规则那条不联网、不用密钥，几秒钟跑完。`--llm-sample` 会真的调 API。
结果写到 `reports/aegisfx_corpus_audit.json`。
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.aegisfx.ingestion.parser import parse_with_rules  # noqa: E402
from src.aegisfx.paths import DAILY_DATA  # noqa: E402
from src.aegisfx.taxonomy import COLUMN_TO_SNAKE, label  # noqa: E402
from src.core.research_dataset import load_daily_frame, load_news_corpus  # noqa: E402

_REPORT = _PROJECT_ROOT / "reports" / "aegisfx_corpus_audit.json"
_EVAL_FROM_YEAR = 2023
_HORIZONS = (1, 5, 20)


def _bar(count: int, total: int, width: int = 28) -> str:
    filled = round(width * count / total) if total else 0
    return "█" * filled + "·" * (width - filled)


def _score_parser(name, parser, frame):
    """在带标签语料上量一条解析路径的准确率。"""
    hits = Counter()
    per_type = {}
    errors = []
    for row in frame.itertuples():
        predicted = parser(row.text).event_type
        hits["total"] += 1
        bucket = per_type.setdefault(row.primary, [0, 0])
        bucket[1] += 1
        if predicted == row.primary:
            hits["correct"] += 1
            bucket[0] += 1
        else:
            errors.append({
                "date": row.date.strftime("%Y-%m-%d"),
                "gold": row.primary,
                "predicted": predicted,
                "text": row.text[:50],
            })
    accuracy = hits["correct"] / hits["total"]
    print(f"\n{name}")
    print(f"  一致率  {hits['correct']}/{hits['total']} = {accuracy:.1%}")
    weak = [(snake, correct, total) for snake, (correct, total) in per_type.items()
            if total >= 5 and correct / total < 0.6]
    if weak:
        print("  一致率低于 60% 的类型：")
        for snake, correct, total in sorted(weak, key=lambda t: t[1] / t[2]):
            print(f"    {label(snake):<24} {correct}/{total} = {correct / total:.0%}")
    return {
        "accuracy": accuracy,
        "correct": hits["correct"],
        "total": hits["total"],
        "per_type": {k: {"correct": v[0], "total": v[1]} for k, v in per_type.items()},
        "errors": errors[:40],
    }


def _stratified_sample(frame, size: int, seed: int = 20260926):
    """按事件类型分层抽样，避免抽出来全是「美国对中国企业的限制」。"""
    rng = random.Random(seed)
    picked = []
    for snake, group in frame.groupby("primary"):
        take = max(1, round(size * len(group) / len(frame)))
        picked.extend(group.sample(min(take, len(group)), random_state=rng.randint(0, 10**6)).itertuples())
    rng.shuffle(picked)
    return picked[:size]


def main() -> int:
    parser = argparse.ArgumentParser(description="事件语料审计")
    parser.add_argument("--llm-sample", type=int, default=0, metavar="N",
                        help="抽 N 条用大模型解析并对比标注（会调 API）")
    args = parser.parse_args()

    corpus = load_news_corpus()
    daily = load_daily_frame()

    rows = []
    for item in corpus:
        snakes = [COLUMN_TO_SNAKE[name] for name in item["event_types"]
                  if name in COLUMN_TO_SNAKE]
        rows.append({
            "date": item["date"],
            "text": item["content"],
            "primary": snakes[0] if snakes else "other",
            "all_types": snakes,
            "sentiment": (item["sentiments"] or ["unknown"])[0],
            "side": (item["sources"] or ["UNKNOWN"])[0],
        })
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    print("=" * 78)
    print("AegisFX-Agent 事件语料审计")
    print("=" * 78)
    print(f"事件总数      {len(frame)}")
    print(f"日期范围      {frame['date'].min():%Y-%m-%d} → {frame['date'].max():%Y-%m-%d}")
    print(f"不同日期数    {frame['date'].nunique()}（每条事件日期都不同）")

    # ---- 按年 ----
    by_year = frame["date"].dt.year.value_counts().sort_index()
    print("\n按年分布")
    for year, count in by_year.items():
        print(f"  {year}  {count:>4}  {_bar(count, by_year.max())}")

    # ---- 按事件类型 ----
    type_counts = Counter(frame["primary"])
    print("\n按事件类型（取主标签）")
    thin = []
    for snake, count in type_counts.most_common():
        flag = ""
        if count < 15:
            thin.append((snake, count))
            flag = "  ← 样本偏少"
        print(f"  {label(snake):<24} {count:>4}  {_bar(count, type_counts.most_common(1)[0][1])}{flag}")

    multi = int((frame["all_types"].map(len) > 1).sum())
    print(f"\n多标签事件    {multi} 条（占比 {multi / len(frame):.1%}）")

    # ---- 按情感 / 发起方 ----
    print("\n按情感")
    for key, count in frame["sentiment"].value_counts().items():
        print(f"  {key:<10} {count:>4}")

    print("\n按发起方")
    for key, count in frame["side"].value_counts().items():
        print(f"  {key:<10} {count:>4}")

    # ---- 时间切分 ----
    is_eval = frame["date"].dt.year >= _EVAL_FROM_YEAR
    pool, eval_set = frame[~is_eval], frame[is_eval]
    print(f"\n时间切分（按 {_EVAL_FROM_YEAR} 年切）")
    print(f"  检索池  2017–{_EVAL_FROM_YEAR - 1}   {len(pool):>4} 条")
    print(f"  评测集  {_EVAL_FROM_YEAR}–2024   {len(eval_set):>4} 条")

    # ---- 收益标签可得性 ----
    price = daily["USDCNY"]
    index = daily.index.searchsorted(frame["date"].values)
    index = index.clip(0, len(daily) - 1)
    label_stats = {}
    print("\n收益标签可得性（USD/CNY 事件后 N 个交易日）")
    for horizon in _HORIZONS:
        forward = (price.shift(-horizon) / price - 1).values[index]
        available = int(pd.notna(forward).sum())
        label_stats[f"ret_{horizon}d"] = {
            "available": available,
            "missing": len(frame) - available,
            "mean": float(pd.Series(forward).mean()),
            "std": float(pd.Series(forward).std()),
            "share_up": float((pd.Series(forward) > 0).mean()),
        }
        print(f"  {horizon:>2} 日   算得出 {available:>4} 条，缺 {len(frame) - available:>3} 条"
              f"   均值 {label_stats[f'ret_{horizon}d']['mean']:+.4%}"
              f"   标准差 {label_stats[f'ret_{horizon}d']['std']:.4%}"
              f"   上涨占比 {label_stats[f'ret_{horizon}d']['share_up']:.1%}")

    # 评测期的方向基准——方案里反复强调的那个数，每次审计都重新算一遍
    eval_index = index[is_eval.values]
    eval_fwd = pd.Series((price.shift(-20) / price - 1).values[eval_index]).dropna()
    base_rate = float((eval_fwd > 0).mean())
    print(f"\n  ⚠ 评测集 20 日方向基准（一律猜上涨的准确率）：{base_rate:.1%}"
          f"  n={len(eval_fwd)}")
    print("    所有方向准确率必须与这个数并列报告，否则没有意义。")

    # ---- 两条解析路径的准确率 ----
    print("\n" + "=" * 78)
    print("解析路径准确率（对比人工标注）")
    print("=" * 78)
    rule_stats = _score_parser("规则兜底（不花 API 调用）",
                               lambda text: parse_with_rules(text), frame)

    llm_stats = None
    if args.llm_sample > 0:
        from src.aegisfx.ingestion.parser import parse_event
        from src.aegisfx.paths import load_env

        load_env()
        sample = _stratified_sample(frame, args.llm_sample)
        print(f"\n大模型主路（分层抽 {len(sample)} 条，真的调 API）")
        correct, errors = 0, []
        for row in sample:
            draft = parse_event(row.text, use_llm=True)
            ok = draft.event_type == row.primary
            correct += ok
            if not ok:
                errors.append({
                    "date": row.date.strftime("%Y-%m-%d"),
                    "gold": row.primary,
                    "predicted": draft.event_type,
                    "method": draft.method,
                    "text": row.text[:50],
                })
        accuracy = correct / len(sample)
        fallback = sum(1 for row in sample
                       if parse_event(row.text, use_llm=True).method == "rule")
        print(f"  一致率  {correct}/{len(sample)} = {accuracy:.1%}")
        print(f"  退回规则解析的  {fallback} 条")
        llm_stats = {"accuracy": accuracy, "correct": correct, "sample": len(sample),
                     "fallbacks": fallback, "errors": errors[:40]}

    report = {
        "total_events": len(frame),
        "date_range": [frame["date"].min().isoformat(), frame["date"].max().isoformat()],
        "by_year": {int(k): int(v) for k, v in by_year.items()},
        "by_event_type": dict(type_counts),
        "thin_types": [{"type": t, "count": c} for t, c in thin],
        "multi_label_events": multi,
        "split": {"retrieval_pool": len(pool), "evaluation": len(eval_set),
                  "eval_from_year": _EVAL_FROM_YEAR},
        "return_labels": label_stats,
        "eval_direction_base_rate_20d": base_rate,
        "rule_parser": rule_stats,
        "llm_parser_sample": llm_stats,
    }
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已写入 {_REPORT.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
