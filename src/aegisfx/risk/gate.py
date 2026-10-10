"""执行风险闸门。

规则全部是确定性的；大模型不能放行订单、改阈值或绕过 Kill Switch。
研究回测与实时 paper 模式共用同一清单，缺少的实时字段会明确标成
``NOT_IMPLEMENTED``，不会静默当作通过。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..schemas.gate import CheckStatus, GateCheck, GateDecision

ALLOWED_MODES = ("research", "backtest", "shadow", "paper")


@dataclass(frozen=True)
class RiskLimits:
    max_conflict: float
    max_interval_width: float
    min_effective_sample_size: float = 2.0
    max_spread_bps: float = 5.0
    max_open_positions: int = 1
    max_daily_loss: float = 0.01
    max_drawdown: float = 0.05
    max_consecutive_losses: int = 3


@dataclass
class RiskState:
    open_positions: int = 0
    daily_pnl: float = 0.0
    drawdown: float = 0.0
    consecutive_losses: int = 0
    kill_switch: bool = False
    api_healthy: Optional[bool] = None
    duplicate_order: Optional[bool] = None


@dataclass(frozen=True)
class OrderContext:
    mode: str
    source_verified: Optional[bool]
    parse_ok: Optional[bool]
    data_fresh: Optional[bool]
    no_future_leakage: bool
    prediction: Optional[float]
    interval_80: Optional[tuple[float, float]]
    interval_width: Optional[float]
    conflict_index: Optional[float]
    effective_sample_size: Optional[float]
    spread_bps: Optional[float] = None


def _check(name: str, value: Optional[bool], passed: str, failed: str,
           missing: str) -> GateCheck:
    if value is None:
        return GateCheck(name=name, status=CheckStatus.NOT_IMPLEMENTED, detail=missing)
    return GateCheck(name=name,
                     status=CheckStatus.PASS if value else CheckStatus.FAIL,
                     detail=passed if value else failed)


def evaluate_order(context: OrderContext, state: RiskState,
                   limits: RiskLimits) -> GateDecision:
    """计算完整风险清单；只有全部项目通过才允许 paper order。"""
    checks = [
        _check("allowed_mode", context.mode in ALLOWED_MODES,
               f"模式 {context.mode}", f"禁止模式 {context.mode}", "模式未知"),
        _check("source_verified", context.source_verified,
               "来源可核验", "来源未经核验", "没有来源核验字段"),
        _check("parse_ok", context.parse_ok,
               "事件解析成功", "事件解析失败", "没有解析状态"),
        _check("data_is_fresh", context.data_fresh,
               "行情在允许延迟内", "行情已过期", "没有行情水位"),
        _check("no_future_leakage", context.no_future_leakage,
               "as-of 与标签成熟断言通过", "检测到未来信息", "未运行防泄漏检查"),
    ]

    interval = context.interval_80
    available = interval is not None and context.interval_width is not None
    checks.append(_check("interval_available", available,
                         "80% 区间可用", "80% 区间不可用", "没有共形区间"))
    excludes_zero = (available and (interval[0] > 0 or interval[1] < 0))
    checks.append(_check("interval_excludes_zero", excludes_zero if available else None,
                         "80% 区间不跨 0", "80% 区间跨 0", "没有可检查的区间"))
    width_ok = (context.interval_width <= limits.max_interval_width
                if context.interval_width is not None else None)
    checks.append(_check("interval_width", width_ok,
                         f"宽度 {context.interval_width:.4%} ≤ {limits.max_interval_width:.4%}"
                         if context.interval_width is not None else "",
                         f"宽度 {context.interval_width:.4%} > {limits.max_interval_width:.4%}"
                         if context.interval_width is not None else "",
                         "没有区间宽度"))

    conflict_ok = (context.conflict_index <= limits.max_conflict
                   if context.conflict_index is not None else None)
    checks.append(_check("conflict_index", conflict_ok,
                         f"冲突 {context.conflict_index:.3f} ≤ {limits.max_conflict:.3f}"
                         if context.conflict_index is not None else "",
                         f"冲突 {context.conflict_index:.3f} > {limits.max_conflict:.3f}"
                         if context.conflict_index is not None else "",
                         "没有非中性方向证据"))
    ess_ok = (context.effective_sample_size >= limits.min_effective_sample_size
              if context.effective_sample_size is not None else None)
    checks.append(_check("effective_sample_size", ess_ok,
                         f"ESS {context.effective_sample_size:.2f} 达标"
                         if context.effective_sample_size is not None else "",
                         f"ESS {context.effective_sample_size:.2f} < {limits.min_effective_sample_size:.2f}"
                         if context.effective_sample_size is not None else "",
                         "没有有效样本量"))

    spread_ok = (context.spread_bps <= limits.max_spread_bps
                 if context.spread_bps is not None else None)
    checks.extend([
        _check("spread", spread_ok,
               f"点差 {context.spread_bps:.2f} bps 达标" if context.spread_bps is not None else "",
               f"点差 {context.spread_bps:.2f} bps 超限" if context.spread_bps is not None else "",
               "未接入 Bid/Ask，不能核验点差"),
        _check("position_limit", state.open_positions < limits.max_open_positions,
               "仓位未满", "仓位已达上限", "仓位状态未知"),
        _check("daily_loss", state.daily_pnl > -limits.max_daily_loss,
               "未触发单日亏损线", "已触发单日亏损线", "当日盈亏未知"),
        _check("drawdown", state.drawdown < limits.max_drawdown,
               "未触发最大回撤线", "已触发最大回撤线", "回撤未知"),
        _check("consecutive_losses",
               state.consecutive_losses < limits.max_consecutive_losses,
               "连续亏损次数未超限", "连续亏损暂停", "连续亏损状态未知"),
        _check("api_healthy", state.api_healthy,
               "模拟接口健康", "模拟接口故障", "未接入模拟接口"),
        _check("duplicate_order",
               None if state.duplicate_order is None else not state.duplicate_order,
               "没有重复订单", "检测到重复订单", "没有订单幂等状态"),
        _check("kill_switch", not state.kill_switch,
               "Kill Switch 未按下", "Kill Switch 已按下", "Kill Switch 状态未知"),
    ])
    return GateDecision.from_checks(checks)
