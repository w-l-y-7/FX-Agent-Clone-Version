"""历史信号上的执行闸门回测。"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from .gate import OrderContext, RiskLimits, RiskState, evaluate_order


def calibrate_signal_limits(observations: Sequence[dict], quantile: float = 0.75) -> dict:
    """只在开发集上冻结冲突与区间宽度阈值。"""
    conflicts = [r["directional_conflict"] for r in observations
                 if r.get("directional_conflict") is not None]
    widths = [r["interval_width"] for r in observations
              if r.get("interval_width") is not None]
    if not conflicts or not widths:
        raise ValueError("开发集缺少冲突或共形区间，不能冻结闸门阈值")
    return {
        "quantile": quantile,
        "max_conflict": float(np.quantile(conflicts, quantile)),
        "max_interval_width": float(np.quantile(widths, quantile)),
        "n_conflict": len(conflicts),
        "n_width": len(widths),
    }


def _max_drawdown(pnls: Iterable[float]) -> float:
    equity = np.cumsum(list(pnls), dtype=float)
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    return float(np.max(peaks - equity))


def backtest_signals(observations: Sequence[dict], limits: RiskLimits, *,
                     spread_bps: float = 2.0) -> dict:
    """比较「每条信号都执行」与「研究闸门放行」的收益。

    历史语料没有逐笔 Bid/Ask，因此成本是明确的情景参数。实时专属检查
    （来源 URL、行情水位、API、重复订单）不拿来决定历史研究结果；报告中会
    单独声明完整 paper gate 仍会因这些字段缺失而 BLOCKED。
    """
    cost = spread_bps / 10_000.0
    trades, blocked = [], Counter()
    all_raw, all_adjusted, gated_raw, gated_adjusted = [], [], [], []

    for row in observations:
        prediction = row.get("prediction")
        raw = row.get("raw_return")
        abnormal = row.get("realized_abret")
        if prediction is None or raw is None or abnormal is None:
            continue
        side = 1.0 if prediction > 0 else -1.0 if prediction < 0 else 0.0
        raw_pnl = side * raw - cost
        adjusted_pnl = side * abnormal - cost
        all_raw.append(raw_pnl)
        all_adjusted.append(adjusted_pnl)

        # 历史研究层只隔离能由当时信号直接判定的五项。
        interval = row.get("interval_80")
        reasons = []
        if interval is None:
            reasons.append("interval_available")
        else:
            if interval[0] <= 0 <= interval[1]:
                reasons.append("interval_excludes_zero")
        if row.get("interval_width") is None or row["interval_width"] > limits.max_interval_width:
            reasons.append("interval_width")
        if (row.get("directional_conflict") is None
                or row["directional_conflict"] > limits.max_conflict):
            reasons.append("conflict_index")
        if (row.get("effective_sample_size") is None
                or row["effective_sample_size"] < limits.min_effective_sample_size):
            reasons.append("effective_sample_size")

        allowed = not reasons
        blocked.update(reasons)
        if allowed:
            gated_raw.append(raw_pnl)
            gated_adjusted.append(adjusted_pnl)
        trades.append({
            "event_id": row["event_id"], "query_date": row["query_date"],
            "allowed": allowed, "blocked_by": reasons,
            "side": "long_usd" if side > 0 else "short_usd" if side < 0 else "flat",
            "raw_pnl": raw_pnl, "drift_adjusted_pnl": adjusted_pnl,
        })

    def metrics(values):
        values = list(values)
        return {
            "n_trades": len(values),
            "total_pnl": float(np.sum(values)) if values else 0.0,
            "mean_pnl": float(np.mean(values)) if values else None,
            "win_rate": float(np.mean(np.asarray(values) > 0)) if values else None,
            "max_drawdown": _max_drawdown(values),
            "worst_trade": float(np.min(values)) if values else None,
        }

    return {
        "n_signals": len(trades),
        "n_allowed": sum(t["allowed"] for t in trades),
        "rejection_rate": (1.0 - sum(t["allowed"] for t in trades) / len(trades)
                           if trades else None),
        "spread_bps_assumption": spread_bps,
        "blocked_reason_counts": dict(blocked),
        "always_execute": {"raw": metrics(all_raw), "drift_adjusted": metrics(all_adjusted)},
        "risk_gated": {"raw": metrics(gated_raw), "drift_adjusted": metrics(gated_adjusted)},
        "trades": trades,
        "paper_gate_note": ("历史回测只评估信号时可知的研究闸门；完整 paper gate 还要求"
                            "来源核验、实时行情、Bid/Ask、API 健康与订单幂等。缺任一项仍 BLOCKED。"),
    }
