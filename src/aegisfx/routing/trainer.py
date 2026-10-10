"""Purged expanding-window 时间验证。

    Fold 1：训练截至 2019，验证 2020
    Fold 2：训练截至 2020，验证 2021
    Fold 3：训练截至 2021，验证 2022

## 隔离（purge + embargo）为什么必须做

伪查询的标签是**前视 h 个交易日**的收益。一条 2021-12-20 的伪查询，
它的 20 日收益要到 2022-01 才实现——落在验证期里。拿它进训练集，
等于用验证期的信息训练，验证指标会虚高。

所以训练样本必须满足：

    label_end_position + embargo <= valid_start_position

embargo 取 h 个交易日（方案要求：5 日期限至少隔离 5 个交易日，20 日至少 20 个）。

## 选 λ

在 4 个固定候选上各跑一遍三个时间折，**取平均验证 MAE 最小的那个**。
多个候选差异落在一个标准误以内时，**选更强正则**（更简单的模型）。
全程不碰 2023–2024。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .gate import GateModel, fit_gate

LAMBDA_CANDIDATES = (0.01, 0.1, 1.0, 10.0)

# (训练截至年份, 验证年份)
FOLDS = ((2019, 2020), (2020, 2021), (2021, 2022))


@dataclass
class FoldResult:
    train_through: int
    valid_year: int
    lam: float
    n_train: int
    n_valid: int
    n_purged: int
    mae_e1: float
    mae_e2: float
    mae_e4: float
    direction_accuracy_e4: float
    mean_gate: float


@dataclass
class RouterTraining:
    model: Optional[GateModel]
    fold_results: List[FoldResult] = field(default_factory=list)
    lambda_table: Dict[float, Dict[str, float]] = field(default_factory=dict)
    selected_lambda: Optional[float] = None
    selection_note: str = ""


def _purge_mask(
    positions: np.ndarray,
    valid_start_position: int,
    horizon: int,
) -> np.ndarray:
    """训练样本按 purge + embargo 过滤后的掩码。

    条件：`position + horizon + embargo <= valid_start_position`，
    其中 embargo = horizon。等价于 `position <= valid_start − 2·horizon`。
    """
    limit = valid_start_position - 2 * horizon
    return positions <= limit


def expanding_folds(horizon: int) -> Sequence[tuple]:
    return FOLDS


def _mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target))) if len(target) else float("nan")


def _direction_accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(target) == 0:
        return float("nan")
    return float(np.mean(np.sign(prediction) == np.sign(target)))


def fit_gate_with_cv(
    features: np.ndarray,
    positions: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    years: np.ndarray,
    valid_start_positions: Dict[int, int],
    *,
    feature_names: Sequence[str],
    horizon: int,
    lambdas: Sequence[float] = LAMBDA_CANDIDATES,
) -> RouterTraining:
    """按时间折选 λ，再在全训练集上拟合最终模型。"""
    training = RouterTraining(model=None)
    scores: Dict[float, List[float]] = {lam: [] for lam in lambdas}

    for lam in lambdas:
        for train_through, valid_year in FOLDS:
            train_mask = (years <= train_through) & _purge_mask(
                positions, valid_start_positions[valid_year], horizon)
            valid_mask = years == valid_year
            if train_mask.sum() < 20 or valid_mask.sum() < 3:
                scores[lam].append(float("nan"))
                continue

            model = fit_gate(
                features[train_mask], r1[train_mask], r2[train_mask],
                target[train_mask], scales[train_mask],
                feature_names=feature_names, lam=lam)
            if model is None:
                scores[lam].append(float("nan"))
                continue

            prediction = model.blend(features[valid_mask], r1[valid_mask],
                                     r2[valid_mask])
            scores[lam].append(_mae(prediction, target[valid_mask]))

    for lam, values in scores.items():
        clean = [v for v in values if not np.isnan(v)]
        training.lambda_table[lam] = {
            "fold_maes": values,
            "mean": float(np.mean(clean)) if clean else float("nan"),
            "std_error": (float(np.std(clean, ddof=1) / np.sqrt(len(clean)))
                          if len(clean) > 1 else float("nan")),
            "n_folds": len(clean),
        }

    usable = {lam: item for lam, item in training.lambda_table.items()
              if item["n_folds"] > 0}
    if not usable:
        training.selection_note = "所有 λ 都无法在时间折上评估，不训练门控"
        return training

    best = min(usable, key=lambda lam: usable[lam]["mean"])
    best_mean = usable[best]["mean"]
    best_se = usable[best]["std_error"]
    if np.isnan(best_se):
        selected = best
        note = f"只有部分折可用，取平均验证 MAE 最小的 λ={best}"
    else:
        # 差异落在一个标准误以内的候选里，选正则更强的
        tied = [lam for lam in usable
                if usable[lam]["mean"] <= best_mean + best_se]
        selected = max(tied)
        note = (f"平均验证 MAE 最小的 λ={best}（{best_mean:.5f}）；"
                f"落在 1 个标准误内的候选 {sorted(tied)} 中取更强正则 → λ={selected}")

    training.selected_lambda = selected
    training.selection_note = note

    # 最终模型用全部训练期（2018–2022）的成熟样本
    full_mask = years <= 2022
    model = fit_gate(features[full_mask], r1[full_mask], r2[full_mask],
                     target[full_mask], scales[full_mask],
                     feature_names=feature_names, lam=selected)
    training.model = model

    for train_through, valid_year in FOLDS:
        train_mask = (years <= train_through) & _purge_mask(
            positions, valid_start_positions[valid_year], horizon)
        valid_mask = years == valid_year
        if model is None or valid_mask.sum() == 0:
            continue
        fold_model = fit_gate(
            features[train_mask], r1[train_mask], r2[train_mask],
            target[train_mask], scales[train_mask],
            feature_names=feature_names, lam=selected)
        if fold_model is None:
            continue
        prediction = fold_model.blend(features[valid_mask], r1[valid_mask],
                                      r2[valid_mask])
        gates = fold_model.gate(features[valid_mask])
        training.fold_results.append(FoldResult(
            train_through=train_through, valid_year=valid_year, lam=selected,
            n_train=int(train_mask.sum()), n_valid=int(valid_mask.sum()),
            n_purged=int((years <= train_through).sum() - train_mask.sum()),
            mae_e1=_mae(r1[valid_mask], target[valid_mask]),
            mae_e2=_mae(r2[valid_mask], target[valid_mask]),
            mae_e4=_mae(prediction, target[valid_mask]),
            direction_accuracy_e4=_direction_accuracy(prediction, target[valid_mask]),
            mean_gate=float(np.mean(gates)),
        ))

    return training
