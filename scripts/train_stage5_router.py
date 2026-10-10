"""训练 E4 路由器，产出时间折 OOF 预测（第 5 阶段）。

    .\\venv\\Scripts\\python.exe scripts\\train_stage5_router.py

流程：读伪查询 → purged expanding-window 选 λ → 三个时间折各训一个模型 →
在每个验证折上产出 OOF 预测（**只用该折之前的数据训练**）→ 存模型与 OOF。

OOF 覆盖 2020–2022（2018–2019 前面没有训练数据，按设计就没有 OOF）。
冲突有效性检验（`evaluate_conflict_signal.py`）用的就是这份 OOF，
**不是训练内拟合误差**——用拟合误差检验冲突指标会把结论做得虚高。

不碰 2023–2024。
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.data.maturity import MaturityCalendar  # noqa: E402
from src.aegisfx.routing.features import FEATURE_NAMES, prune_correlated  # noqa: E402
from src.aegisfx.routing.gate import fit_gate  # noqa: E402
from src.aegisfx.routing.trainer import FOLDS, fit_gate_with_cv  # noqa: E402

DATA_DIR = _PROJECT_ROOT / "data" / "aegisfx" / "routing"
REPORT_DIR = _PROJECT_ROOT / "reports" / "stage5"

HORIZONS = (5, 20)
# 训练阶段**只能**用这几年。2023 是开发集，2024 是准留出集，都不许碰。
TRAIN_YEARS_MAX = 2022


def load_pseudo_queries(horizon: int):
    path = DATA_DIR / f"pseudo_queries_h{horizon}.csv"
    if not path.exists():
        raise SystemExit(f"找不到 {path}。先跑 scripts/build_pseudo_queries.py")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows, path


def _float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def train_horizon(horizon: int, calendar: MaturityCalendar, verbose=True) -> dict:
    rows, path = load_pseudo_queries(horizon)
    rows = [r for r in rows if int(r["query_date"][:4]) <= TRAIN_YEARS_MAX]
    rows.sort(key=lambda r: r["query_date"])

    features = np.array([[_float(r[name]) for name in FEATURE_NAMES] for r in rows])
    e1 = np.array([_float(r["e1_prediction"]) for r in rows])
    e2 = np.array([_float(r["e2_prediction"]) for r in rows])
    target = np.array([_float(r["realized_abret"]) for r in rows])
    scales = np.array([abs(_float(r["scale"], 0.01)) or 0.01 for r in rows])
    years = np.array([int(r["query_date"][:4]) for r in rows])
    positions = np.array([calendar.position(r["query_date"]) for r in rows], dtype=int)

    valid_start_positions = {}
    for _, valid_year in FOLDS:
        marker = next((r["query_date"] for r in rows
                       if int(r["query_date"][:4]) == valid_year), None)
        if marker is None:
            valid_start_positions[valid_year] = 10 ** 9
        else:
            valid_start_positions[valid_year] = calendar.position(marker)

    # ---- 相关性裁剪：只用训练期（2018–2022）的数据算 ----
    kept = prune_correlated(features, FEATURE_NAMES, threshold=0.9)
    dropped = [name for name in FEATURE_NAMES if name not in kept]
    kept_index = [FEATURE_NAMES.index(name) for name in kept]
    if verbose:
        print(f"\n--- 期限 {horizon} 日 ---")
        print(f"  样本 {len(rows)} 条（{min(years)}–{max(years)}）")
        print(f"  特征 {len(FEATURE_NAMES)} 维，相关性裁剪后 {len(kept)} 维"
              + (f"，删掉 {dropped}" if dropped else ""))

    training = fit_gate_with_cv(
        features[:, kept_index], positions, e1, e2, target, scales, years,
        valid_start_positions, feature_names=kept, horizon=horizon)

    if verbose:
        print(f"  λ 选择：{training.selection_note}")
        print(f"  {'λ':>6}{'平均验证MAE':>14}{'标准误':>10}{'折数':>6}")
        for lam, item in training.lambda_table.items():
            print(f"  {lam:>6}{item['mean']:>14.5f}{item['std_error']:>10.5f}"
                  f"{item['n_folds']:>6}")
        print(f"  {'折':>10}{'训练':>7}{'purged':>8}{'验证':>6}"
              f"{'MAE_E1':>10}{'MAE_E2':>10}{'MAE_E4':>10}{'方向':>8}{'g均值':>8}")
        for fold in training.fold_results:
            print(f"  {fold.train_through}→{fold.valid_year:<5}{fold.n_train:>7}"
                  f"{fold.n_purged:>8}{fold.n_valid:>6}{fold.mae_e1:>10.5f}"
                  f"{fold.mae_e2:>10.5f}{fold.mae_e4:>10.5f}"
                  f"{fold.direction_accuracy_e4:>8.3f}{fold.mean_gate:>8.3f}")

    # ---- OOF：每个验证折用它自己那个「只用历史训练」的模型 ----
    oof_rows = []
    for train_through, valid_year in FOLDS:
        train_mask = years <= train_through
        valid_mask = years == valid_year
        if valid_mask.sum() == 0:
            continue
        model = fit_gate(features[train_mask][:, kept_index], e1[train_mask],
                         e2[train_mask], target[train_mask], scales[train_mask],
                         feature_names=kept,
                         lam=training.selected_lambda or 0.1)
        if model is None:
            continue
        gates = model.gate(features[valid_mask][:, kept_index])
        predictions = model.blend(features[valid_mask][:, kept_index],
                                  e1[valid_mask], e2[valid_mask])
        for row, gate, prediction in zip(
                [r for r, keep in zip(rows, valid_mask) if keep], gates, predictions):
            oof_rows.append({
                "query_id": row["query_id"],
                "query_date": row["query_date"],
                "horizon": horizon,
                "event_family": row["event_family"],
                "fold_valid_year": valid_year,
                "gate": float(gate),
                "e1_prediction": _float(row["e1_prediction"]),
                "e2_prediction": _float(row["e2_prediction"]),
                "e4_prediction": float(prediction),
                "realized_abret": _float(row["realized_abret"]),
                "abs_error_e1": abs(_float(row["e1_prediction"])
                                    - _float(row["realized_abret"])),
                "abs_error_e2": abs(_float(row["e2_prediction"])
                                    - _float(row["realized_abret"])),
                "abs_error_e4": abs(float(prediction) - _float(row["realized_abret"])),
                "direction_error_e1": int(np.sign(_float(row["e1_prediction"]))
                                          != np.sign(_float(row["realized_abret"]))),
                "direction_error_e2": int(np.sign(_float(row["e2_prediction"]))
                                          != np.sign(_float(row["realized_abret"]))),
                "direction_error_e4": int(np.sign(float(prediction))
                                          != np.sign(_float(row["realized_abret"]))),
                "directional_conflict": _float(row["ci_e1"]),
                "neutral_mass": _float(row["neutral_mass_e1"]),
                "pairwise_dispersion_norm": _float(row["pd_norm_e1"]),
                "effective_sample_size": _float(row["ess_e1"]),
                "ci_e2": _float(row["ci_e2"]),
                "pd_norm_e2": _float(row["pd_norm_e2"]),
                "ess_e2": _float(row["ess_e2"]),
            })

    out_path = DATA_DIR / f"oof_predictions_h{horizon}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof_rows[0]))
        writer.writeheader()
        writer.writerows(oof_rows)

    gates = [r["gate"] for r in oof_rows]
    by_family = {}
    for row in oof_rows:
        by_family.setdefault(row["event_family"], []).append(row["gate"])

    if verbose:
        print(f"  OOF {len(oof_rows)} 条 → {out_path.name}")
        if gates:
            print(f"  门控 g：均值 {np.mean(gates):.3f}　"
                  f"范围 {min(gates):.3f}–{max(gates):.3f}　"
                  f"标准差 {np.std(gates):.3f}")
            collapsed = np.std(gates) < 0.02
            print(f"  门控塌缩（标准差 < 0.02）？{'是' if collapsed else '否'}")
        for family, values in sorted(by_family.items()):
            print(f"    {family:<22} n={len(values):>3}  g 均值 {np.mean(values):.3f}")

    return {
        "horizon": horizon,
        "n_samples": len(rows),
        "features_all": list(FEATURE_NAMES),
        "features_kept": kept,
        "features_dropped": dropped,
        "selected_lambda": training.selected_lambda,
        "selection_note": training.selection_note,
        "lambda_table": {str(k): v for k, v in training.lambda_table.items()},
        "coefficients": (training.model.describe() if training.model else []),
        # 完整配置（含标准化均值/方差），E4 在开发集/最终集上靠它复现
        "model_config": (training.model.to_config() if training.model else None),
        "folds": [fold.__dict__ for fold in training.fold_results],
        "oof": {
            "n": len(oof_rows),
            "gate_mean": float(np.mean(gates)) if gates else None,
            "gate_std": float(np.std(gates)) if gates else None,
            "gate_min": float(min(gates)) if gates else None,
            "gate_max": float(max(gates)) if gates else None,
            "gate_collapsed": bool(gates and np.std(gates) < 0.02),
            "gate_by_family": {k: float(np.mean(v)) for k, v in by_family.items()},
            "mae_e1": float(np.mean([r["abs_error_e1"] for r in oof_rows])),
            "mae_e2": float(np.mean([r["abs_error_e2"] for r in oof_rows])),
            "mae_e4": float(np.mean([r["abs_error_e4"] for r in oof_rows])),
            "direction_accuracy_e1": float(1 - np.mean(
                [r["direction_error_e1"] for r in oof_rows])),
            "direction_accuracy_e2": float(1 - np.mean(
                [r["direction_error_e2"] for r in oof_rows])),
            "direction_accuracy_e4": float(1 - np.mean(
                [r["direction_error_e4"] for r in oof_rows])),
        },
        "data_file": str(path.relative_to(_PROJECT_ROOT)),
        "oof_file": str(out_path.relative_to(_PROJECT_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="训练 E4 路由器")
    args = parser.parse_args()

    calendar = MaturityCalendar.from_market()
    print("=" * 78)
    print("训练 E4 路由器（事件族—市场状态自适应专家路由）")
    print("=" * 78)
    print(f"训练年份上限 {TRAIN_YEARS_MAX}　"
          "2023 是开发集、2024 是准留出集，本脚本都不碰")

    report = {"train_years_max": TRAIN_YEARS_MAX, "horizons": {}}
    for horizon in HORIZONS:
        report["horizons"][horizon] = train_horizon(horizon, calendar)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "router_training.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    print(f"\n报告已写入 {path.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
