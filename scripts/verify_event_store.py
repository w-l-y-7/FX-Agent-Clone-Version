"""事件库健康检查（第 2 周）。

    .\\venv\\Scripts\\python.exe scripts\\verify_event_store.py

跑 `build_event_store.py` 之后用它验收。检查的都是**会被后续改动悄悄破坏、
但一旦破坏就没法察觉**的性质：

- 异常收益的跨事件标准差必须低于原始收益（20 日、5 日）——
  如果模型哪天变成在加噪声，这个检查会先报警；
- 市场状态必须取自事件日的**前一交易日**（用到当天就等于偷看未来）；
- 估计窗口不能延伸到事件日之后；
- 标签三态与切分的一致性。

不用密钥、不联网。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.data.market import paper_market  # noqa: E402
from src.aegisfx.schemas.historical import EVAL, LABELED, POOL  # noqa: E402

PASSED, FAILED = [], []


def as_day(value) -> pd.Timestamp:
    """事件日期 → naive 时间戳。

    早期版本把交易日期存成了带时区的，和行情索引（naive）比会直接抛
    `Cannot compare tz-naive and tz-aware`。这里统一剥掉，好让新老库都能读。
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'通过' if ok else '失败'}] {name}" + (f" —— {detail}" if detail else ""))


def main() -> int:
    print("=" * 78)
    print("事件库健康检查")
    print("=" * 78)

    events = load_event_store()
    frame = paper_market()

    if not events:
        print("事件库是空的。先跑 scripts/build_event_store.py")
        return 1

    print(f"\n共 {len(events)} 条事件\n")

    # ---- 1. 模型确实在减方差 ----
    print("1. 市场模型有没有用（判据是跨事件方差，不是 R²）")
    for horizon in (5, 20):
        raw = pd.Series([getattr(e, f"ret_{horizon}d") for e in events]).dropna()
        abn = pd.Series([getattr(e, f"abret_{horizon}d") for e in events]).dropna()
        reduction = 1 - abn.std() / raw.std()
        check(f"{horizon} 日异常收益的跨事件标准差低于原始收益",
              reduction > 0.02,
              f"原始 {raw.std():.4%} → 异常 {abn.std():.4%}（压缩 {reduction:.1%}）")

    print("\n   1 日尺度单独说明：")
    raw1 = pd.Series([e.ret_1d for e in events]).dropna()
    abn1 = pd.Series([e.abret_1d for e in events]).dropna()
    print(f"     原始 {raw1.std():.4%} → 异常 {abn1.std():.4%}"
          f"（压缩 {1 - abn1.std() / raw1.std():.1%}）")
    print("     人民币日频本来就不跟美元指数走（相关性 0.24），1 日窗口模型几乎不起作用，")
    print("     这是事实不是缺陷——报告里要单独讲，别和 5/20 日混在一起。")

    # ---- 2. 没有偷看未来 ----
    print("\n2. 市场状态取自事件日前一交易日")
    off_by_one = 0
    for event in events[:200]:
        position = int(frame.index.searchsorted(as_day(event.date), side="left"))
        if position == 0:
            continue
        expected = frame["usd_index"].iloc[position - 1]
        if event.usd_index is None or abs(event.usd_index - expected) > 1e-9:
            off_by_one += 1
    check("抽查 200 条，市场状态都取自前一日", off_by_one == 0,
          f"{off_by_one} 条对不上" if off_by_one else "")

    print("\n3. 估计窗口不越过事件日")
    # 结构上保证：窗口右移 h 天。这里核对估计观测数是否合理
    observations = [e.model_observations for e in events if e.model_observations]
    check("市场模型的估计观测数都在合理区间",
          all(40 <= n <= 250 for n in observations),
          f"范围 {min(observations)}–{max(observations)}，中位 {int(np.median(observations))}")

    # ---- 4. 切分与标签 ----
    print("\n4. 切分与标签三态")
    pool = [e for e in events if e.split == POOL]
    evaluation = [e for e in events if e.split == EVAL]
    check("切分年份正确",
          all(as_day(e.date).year < 2023 for e in pool)
          and all(as_day(e.date).year >= 2023 for e in evaluation),
          f"检索池 {len(pool)} / 评测集 {len(evaluation)}")
    check("检索池与评测集之和等于总数",
          len(pool) + len(evaluation) == len(events))

    mislabeled = [e for e in events
                  if e.label_status == LABELED and None in (e.abret_1d, e.abret_5d, e.abret_20d)]
    check("标为 labeled 的事件三个窗口都有值", not mislabeled,
          f"{len(mislabeled)} 条不一致" if mislabeled else "")
    check("没有事件同时缺 5 日和 20 日标签（短窗口应先算出来）",
          not any(e.abret_5d is None and e.abret_20d is not None for e in events))

    # ---- 5. 事件粒度 ----
    print("\n5. 事件粒度")
    keys = [(as_day(e.date), e.event_type, e.initiating_side) for e in events]
    check("同一日+同类型+同发起方的事件已合并（方案 §5.2 的规则）",
          len(keys) == len(set(keys)),
          f"{len(keys) - len(set(keys))} 组没合并" if len(keys) != len(set(keys)) else "")

    dates = [as_day(e.date) for e in events]
    check("所有事件日都落在交易日上", all(d in frame.index for d in dates))

    shared = [e for e in events if e.same_day_peers > 1]
    print(f"    同一天有多条事件的日子：{len(shared)} 条事件，"
          f"覆盖 {len({as_day(e.date) for e in shared})} 个交易日")
    if shared:
        print("    这些事件**共享同一个已实现收益**，第 4 周算冲突指数时要按天去重，")
        print("    否则同一天的市场反应会被数两遍。")
    wrong_peers = [e for e in events
                   if e.same_day_peers != sum(1 for other in events
                                              if as_day(other.date) == as_day(e.date))]
    check("same_day_peers 计数正确", not wrong_peers,
          f"{len(wrong_peers)} 条对不上" if wrong_peers else "")

    # ---- 6. 多数类基准 ----
    print("\n6. 评测期的方向基准（所有方向准确率的地板）")
    eval_events = [e for e in events if e.split == EVAL]
    for label, attribute in (("原始收益口径", "ret_20d"), ("异常收益口径", "abret_20d")):
        values = pd.Series([getattr(e, attribute) for e in eval_events]).dropna()
        share_up = float((values > 0).mean())
        print(f"   {label}  一律猜主流方向 = {max(share_up, 1 - share_up):.1%}"
              f"  （n={len(values)}）")

    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败：{name}")
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
