"""冻结第 5 阶段的配置并算 hash（在打开 2024 之前必须做）。

    .\\venv\\Scripts\\python.exe scripts\\lock_stage5_config.py

冻结的内容：路由特征列表、系数、λ、共形参数（窗口、目标覆盖率、γ）、
期限、CP1 是否启用。每一份都算 hash，并把**数据 hash** 一起写进锁文件。

**冻结之后不许再按 2024 的结果改任何一项。** 真要改，必须重新冻结、
换一个新的 config hash，并在报告里说明改动理由——旧的最终结果不许被静默覆盖。

γ 在 2020–2022 的 OOF 上选，候选固定 `{0.001, 0.005, 0.01, 0.02}`。
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.conformal.adaptive import GAMMA_CANDIDATES, choose_gamma  # noqa: E402
from src.aegisfx.conformal.scaled import cp1_allowed  # noqa: E402
from src.aegisfx.data.maturity import MaturityCalendar  # noqa: E402

DATA_DIR = _PROJECT_ROOT / "data" / "aegisfx" / "routing"
REPORT_DIR = _PROJECT_ROOT / "reports" / "stage5"
CONFIG_PATH = REPORT_DIR / "frozen_config.json"

HORIZONS = (5, 20)
CONFORMAL_TARGETS = (0.8, 0.9)
CONFORMAL_WINDOW = 60
MIN_INTERVAL_CALIBRATION = 10


def _hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hashable(config: dict) -> dict:
    """参与 hash 的字段。

    **`locked_at` 与 `config_hash` 自身不参与**——否则每跑一次 lock，
    时间戳一变 hash 就变，配置看起来像被改过，幂等性也没了。
    """
    return {key: value for key, value in config.items()
            if key not in ("locked_at", "config_hash")}


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _load_oof(horizon: int):
    path = DATA_DIR / f"oof_predictions_h{horizon}.csv"
    if not path.exists():
        raise SystemExit(f"找不到 {path}。先跑 train_stage5_router.py")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: r["query_date"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结第 5 阶段配置")
    args = parser.parse_args()

    training_path = REPORT_DIR / "router_training.json"
    if not training_path.exists():
        raise SystemExit("找不到 router_training.json。先跑 train_stage5_router.py")
    training = json.loads(training_path.read_text(encoding="utf-8"))

    calendar = MaturityCalendar.from_market()
    allowed = cp1_allowed()

    print("=" * 78)
    print("冻结第 5 阶段配置")
    print("=" * 78)

    horizons = {}
    for horizon in HORIZONS:
        item = training["horizons"][str(horizon)]
        rows = _load_oof(horizon)
        # γ 只在 2020–2022 的 OOF 上选（这份 OOF 本身就是那几年的）
        predictions = [float(r["e1_prediction"]) for r in rows]
        realized = [float(r["realized_abret"]) for r in rows]
        available = [calendar.available_index(r["query_date"], horizon) or 10 ** 9
                     for r in rows]
        gamma = choose_gamma(predictions, realized, available,
                             target=0.9, candidates=GAMMA_CANDIDATES,
                             window=CONFORMAL_WINDOW)

        # 最强基线由**时间验证**决定，不能看着 2024 挑，也不能默认成 E2
        folds = item.get("folds", [])
        e1_mean = float(np.mean([f["mae_e1"] for f in folds])) if folds else None
        e2_mean = float(np.mean([f["mae_e2"] for f in folds])) if folds else None
        if e1_mean is None or e2_mean is None:
            strongest = "E1"
        else:
            strongest = "E1" if e1_mean <= e2_mean else "E2"

        horizons[horizon] = {
            "strongest_baseline": strongest,
            "baseline_fold_mean_mae": {"E1": e1_mean, "E2": e2_mean},
            "features": item["features_kept"],
            "coefficients": item["coefficients"],
            # 含标准化均值/方差，评测时靠它复现门控值。只存系数会算错而且不报错。
            "model_config": item.get("model_config"),
            "selected_lambda": item["selected_lambda"],
            "conformal": {
                "window": CONFORMAL_WINDOW,
                "targets": list(CONFORMAL_TARGETS),
                "gamma": gamma["selected"],
                "gamma_note": gamma["note"],
                "gamma_table": {str(k): v for k, v in gamma["table"].items()},
                "min_interval_calibration": MIN_INTERVAL_CALIBRATION,
            },
            "cp1_enabled": bool(allowed),
            "cp1_reason": ("冲突有效性检验通过" if allowed else
                           "冲突有效性检验未通过，CP1 降级为 CP0"),
        }
        print(f"\n期限 {horizon} 日")
        print(f"  特征 {len(item['features_kept'])} 维　λ = {item['selected_lambda']}")
        print(f"  最强基线（按时间折平均 MAE）：{strongest}　"
              f"E1={e1_mean:.5f}　E2={e2_mean:.5f}")
        print(f"  γ 选择：{gamma['note']}")
        print(f"  CP1：{'启用' if allowed else '降级为 CP0'}")

    config = {
        "stage": 5,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "horizons": horizons,
        "primary_horizon": 20,
        "dev_split_year": 2023,
        "final_split_year": 2024,
        "cp1_allowed": bool(allowed),
        "notes": [
            "2023 是开发集，用来冻结参数；2024 是准留出集。",
            "冻结后不得按 2024 结果改动任何一项；要改必须重新冻结并换 hash。",
            "γ 只在 2020–2022 的 OOF 上选，不用 2024 覆盖率倒推。",
        ],
    }
    data_hashes = {
        f"pseudo_queries_h{h}": _file_hash(DATA_DIR / f"pseudo_queries_h{h}.csv")
        for h in HORIZONS
    }
    data_hashes.update({
        f"oof_predictions_h{h}": _file_hash(DATA_DIR / f"oof_predictions_h{h}.csv")
        for h in HORIZONS
    })
    data_hashes["historical_events"] = _file_hash(
        _PROJECT_ROOT / "data" / "aegisfx" / "historical_events.jsonl")
    config["data_hashes"] = data_hashes

    # 先做一次 JSON 往返再算 hash。
    #
    # 不往返的话，内存里的字典和读回来的字典不是一回事：int 键会变成字符串、
    # 元组会变成列表。评测脚本读的是文件，算出来必然和这里对不上——
    # 第一版就是这么错的，锁一次改一次 hash，看着像配置被篡改。
    config = json.loads(json.dumps(config, ensure_ascii=False, default=str))
    config_hash = _hash(hashable(config))
    config["config_hash"] = config_hash

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        previous = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            backup = CONFIG_PATH.with_name(
                f"frozen_config.{previous.get('config_hash', 'old')}.json")
            if not backup.exists():
                backup.write_text(json.dumps(previous, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
                print(f"\n⚠ 检测到配置变化，旧配置已备份到 {backup.name}")

    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"config hash  {config_hash}")
    print(f"数据 hash    {data_hashes}")
    print(f"配置文件     {CONFIG_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"{'=' * 78}")
    print("\n下一步：先跑开发集，确认无误后再手动运行最终测试：")
    print("  python scripts/evaluate_stage5.py --split dev")
    print("  python scripts/evaluate_stage5.py --split final --confirm-locked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
