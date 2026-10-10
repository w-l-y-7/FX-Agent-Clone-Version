"""风险 Agent 的确定性边界测试。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.aegisfx.risk import OrderContext, RiskLimits, RiskState, evaluate_order  # noqa: E402
from src.aegisfx.schemas.gate import CheckStatus  # noqa: E402

PASSED, FAILED = [], []


def check(name, ok):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'通过' if ok else '失败'}] {name}")


def base_context(**changes):
    data = dict(mode="paper", source_verified=True, parse_ok=True,
                data_fresh=True, no_future_leakage=True, prediction=0.01,
                interval_80=(0.002, 0.018), interval_width=0.016,
                conflict_index=0.2, effective_sample_size=3.0, spread_bps=2.0)
    data.update(changes)
    return OrderContext(**data)


def main():
    limits = RiskLimits(max_conflict=0.6, max_interval_width=0.02)
    healthy = RiskState(api_healthy=True, duplicate_order=False)
    decision = evaluate_order(base_context(), healthy, limits)
    check("全部 18 项通过时允许 paper order", decision.allow_execution and len(decision.checks) == 18)

    cross = evaluate_order(base_context(interval_80=(-0.01, 0.02)), healthy, limits)
    check("区间跨 0 时拒绝", not cross.allow_execution
          and any(c.name == "interval_excludes_zero" and c.status is CheckStatus.FAIL
                  for c in cross.checks))

    conflict = evaluate_order(base_context(conflict_index=0.9), healthy, limits)
    check("冲突超限时拒绝", not conflict.allow_execution)

    missing = evaluate_order(base_context(spread_bps=None), healthy, limits)
    check("实时点差缺失记为未实现且拒绝", not missing.allow_execution
          and any(c.name == "spread" and c.status is CheckStatus.NOT_IMPLEMENTED
                  for c in missing.checks))

    killed = evaluate_order(base_context(), RiskState(api_healthy=True,
                            duplicate_order=False, kill_switch=True), limits)
    check("Kill Switch 无条件拒绝", not killed.allow_execution)

    duplicate = evaluate_order(base_context(), RiskState(api_healthy=True,
                               duplicate_order=True), limits)
    check("重复订单被拦截", not duplicate.allow_execution)

    bad_mode = evaluate_order(base_context(mode="live"), healthy, limits)
    check("live 模式不在允许集合", not bad_mode.allow_execution)

    loss = evaluate_order(base_context(), RiskState(api_healthy=True,
                          duplicate_order=False, daily_pnl=-0.02,
                          drawdown=0.06, consecutive_losses=3), limits)
    failed = {c.name for c in loss.checks if c.status is CheckStatus.FAIL}
    check("亏损、回撤、连败三条独立触发", {"daily_loss", "drawdown",
          "consecutive_losses"}.issubset(failed))

    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
