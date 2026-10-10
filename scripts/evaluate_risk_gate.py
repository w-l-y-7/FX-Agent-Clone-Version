"""冻结执行闸门阈值并回测历史信号。

开发集：
    python scripts/evaluate_stage5.py --split dev
    python scripts/evaluate_risk_gate.py --split dev --calibrate

准留出集只有在第 5 阶段 final 已经按冻结协议运行后才可评：
    python scripts/evaluate_risk_gate.py --split final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.aegisfx.risk import RiskLimits, backtest_signals, calibrate_signal_limits  # noqa: E402

STAGE5 = _ROOT / "reports" / "stage5"
REPORT_DIR = _ROOT / "reports" / "risk_gate"
CONFIG_PATH = REPORT_DIR / "frozen_config.json"


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:16]


def _load_stage5(split: str, horizon: int) -> tuple[dict, list]:
    path = STAGE5 / f"evaluation_{split}.json"
    if not path.exists():
        raise SystemExit(f"找不到 {path.relative_to(_ROOT)}，请先运行第 5 阶段 {split} 评测")
    report = json.loads(path.read_text(encoding="utf-8"))
    item = report["horizons"].get(str(horizon))
    if not item or not item.get("observations"):
        raise SystemExit(f"{path.relative_to(_ROOT)} 没有逐事件 observations，请重新运行第 5 阶段评测")
    return report, item["observations"]


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit("风险闸门尚未冻结；先运行 --split dev --calibrate")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stored = config.get("config_hash")
    payload = {k: v for k, v in config.items() if k not in ("locked_at", "config_hash")}
    if _hash(payload) != stored:
        raise SystemExit("风险闸门配置 hash 不匹配，拒绝运行")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="风险闸门历史回测")
    parser.add_argument("--split", choices=("dev", "final"), default="dev")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--spread-bps", type=float, default=2.0)
    args = parser.parse_args()

    stage5, observations = _load_stage5(args.split, args.horizon)
    if args.calibrate:
        if args.split != "dev":
            raise SystemExit("只能在 dev 切分上冻结风险阈值")
        calibrated = calibrate_signal_limits(observations)
        payload = {
            "stage": "risk_gate",
            "stage5_config_hash": stage5["config_hash"],
            "horizon": args.horizon,
            "threshold_source": "2023 dev 的 75% 分位；final 不参与",
            "max_conflict": calibrated["max_conflict"],
            "max_interval_width": calibrated["max_interval_width"],
            "min_effective_sample_size": 2.0,
            "max_spread_bps": 5.0,
            "max_open_positions": 1,
            "max_daily_loss": 0.01,
            "max_drawdown": 0.05,
            "max_consecutive_losses": 3,
            "calibration_counts": {
                "conflict": calibrated["n_conflict"],
                "interval_width": calibrated["n_width"],
            },
        }
        payload["config_hash"] = _hash(payload)
        payload["locked_at"] = datetime.now(timezone.utc).isoformat()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        config = payload
        print(f"风险配置已冻结：{payload['config_hash']}")
    else:
        config = _load_config()
        if config["stage5_config_hash"] != stage5["config_hash"]:
            raise SystemExit("风险配置与第 5 阶段配置 hash 不一致，拒绝运行")

    limits = RiskLimits(
        max_conflict=config["max_conflict"],
        max_interval_width=config["max_interval_width"],
        min_effective_sample_size=config["min_effective_sample_size"],
        max_spread_bps=config["max_spread_bps"],
        max_open_positions=config["max_open_positions"],
        max_daily_loss=config["max_daily_loss"],
        max_drawdown=config["max_drawdown"],
        max_consecutive_losses=config["max_consecutive_losses"],
    )
    result = backtest_signals(observations, limits, spread_bps=args.spread_bps)
    report = {
        "split": args.split, "horizon": args.horizon,
        "stage5_config_hash": stage5["config_hash"],
        "risk_config_hash": config["config_hash"],
        "limits": {
            "max_conflict": limits.max_conflict,
            "max_interval_width": limits.max_interval_width,
            "min_effective_sample_size": limits.min_effective_sample_size,
        },
        **result,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"evaluation_{args.split}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"切分 {args.split}：{result['n_allowed']}/{result['n_signals']} 条放行，"
          f"拒绝率 {result['rejection_rate']:.1%}")
    print("漂移调整 PnL："
          f"全执行 {result['always_execute']['drift_adjusted']['total_pnl']:+.4%}，"
          f"闸门后 {result['risk_gated']['drift_adjusted']['total_pnl']:+.4%}")
    print(f"报告已写入 {path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
