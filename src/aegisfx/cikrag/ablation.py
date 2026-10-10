"""第二条通道：样本外预测效用。

**不能用训练内特征重要性代替它。** 一个特征在训练集上被树模型排到第一，
和它在样本外真的降低预测误差，是两件事——本仓库第 2 阶段已经吃过一次亏：
五个深度模型在训练集上都很像样，样本外全部输给「价格不变」。

所以这里做的是 purged walk-forward 消融：

    增益_f = (Loss_base − Loss_{base+f}) / Loss_base

## base 取什么

取**「预测零变化」**，也就是本仓库一直在用的朴素基准（第 2 阶段实测
MAE 0.0813，五个深度模型全没跑赢它）。含义很直接：

    增益 > 0  ⇔  只用这一个特征，就能比「假设汇率不动」预测得更准

这是个**很硬**的判据。用「全部特征减去它」那种留一法会宽松得多，
因为一堆特征互相掩护，去掉一个几乎不影响误差。

## 折与隔离

    Fold 1：训练 ≤2019，测试 2020
    Fold 2：训练 ≤2020，测试 2021
    Fold 3：训练 ≤2021，测试 2022

训练段末尾**按 horizon 隔离**：标签跨越 h 个交易日，训练段最后 h 行会被
测试段的行情影响，必须剔除。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .aggregate import PredictiveUtility

FOLDS = ((2019, 2020), (2020, 2021), (2021, 2022))
RIDGE_ALPHA = 1.0
MIN_TRAIN_ROWS = 150
MIN_TEST_ROWS = 40
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260930


def build_dataset(frame: pd.DataFrame, feature: str, horizon: int) -> pd.DataFrame:
    """一列特征 + 目标（h 日后的价格变化量）。

    ## 连续特征必须先差分

    第一版直接拿特征的**水平值**回归「价格变化量」，结果 32 个特征一起上
    时 MAE 0.130，比「预测零变化」的 0.078 还差一大截。

    原因是本仓库第 2 阶段已经栽过一次的那个坑：美元指数、外储这类序列非平稳，
    训练期学到「指数 98 对应什么」，测试期指数跑到 107，模型就系统性外推失真。
    这不是过拟合，是**水平回归的固有毛病**（见 `core/backtest.py` 的说明）。

    所以连续特征取一阶差分；事件哑变量本来就是 0/1 平稳序列，保持原样。
    仓库里 `feature_engineering.build_feature_frame` 用的也是收益率而非水平值。
    """
    from src.core.research_dataset import EVENT_FEATURES

    if feature in EVENT_FEATURES:
        series = frame[feature].astype(float)
    else:
        series = frame[feature].astype(float).diff()

    data = pd.DataFrame({
        "feature": series,
        "close": frame["USDCNY"],
        "target": frame["USDCNY"].shift(-horizon),
    }).dropna()
    return data


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """固定 FA：StandardScaler + Ridge(alpha=1.0)，预测变化量。"""
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[["feature"]].to_numpy(), train["change"].to_numpy())
    return model.predict(test[["feature"]].to_numpy())


def _fold_gain(frame: pd.DataFrame, feature: str, horizon: int,
               train_through: int, test_year: int) -> Optional[dict]:
    data = build_dataset(frame, feature, horizon)
    data["change"] = data["target"] - data["close"]
    data["year"] = data.index.year

    train = data[data["year"] <= train_through]
    # 隔离：训练段最后 horizon 行，标签跨进了测试期的行情
    if len(train) > horizon:
        train = train.iloc[:-horizon]
    test = data[data["year"] == test_year]
    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        return None

    prediction = _fit_predict(train, test)
    actual = test["change"].to_numpy()

    # base：预测零变化
    loss_base = float(np.mean(np.abs(actual)))
    loss_model = float(np.mean(np.abs(prediction - actual)))
    if loss_base <= 1e-12:
        return None
    return {
        "train_through": train_through, "test_year": test_year,
        "n_train": len(train), "n_test": len(test),
        "loss_base": loss_base, "loss_model": loss_model,
        "gain": float((loss_base - loss_model) / loss_base),
    }


def _block_bootstrap(deltas: np.ndarray, base_errors: np.ndarray,
                     months: np.ndarray,
                     resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    groups: Dict[str, List[int]] = {}
    for index, month in enumerate(months):
        groups.setdefault(month, []).append(index)
    keys = list(groups)
    if len(keys) < 3 or len(deltas) == 0:
        return {"low": None, "high": None, "n_blocks": len(keys)}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        index = np.concatenate([groups[keys[j]] for j in picked])
        denominator = base_errors[index].mean()
        means[i] = deltas[index].mean() / max(denominator, 1e-12)
    return {"low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)),
            "n_blocks": len(keys), "seed": seed, "resamples": resamples}


def feature_utility(
    frame: pd.DataFrame,
    feature: str,
    horizon: int,
    *,
    folds: Sequence = FOLDS,
) -> PredictiveUtility:
    """一个特征的 purged walk-forward 增益。"""
    results = [r for r in (_fold_gain(frame, feature, horizon, *fold)
                           for fold in folds) if r is not None]
    if not results:
        return PredictiveUtility(feature_id=feature)

    gains = [r["gain"] for r in results]
    return PredictiveUtility(
        feature_id=feature,
        gain_mean=float(np.mean(gains)),
        gain_median=float(np.median(gains)),
        gain_by_fold=[float(g) for g in gains],
        sign_consistency=float(np.mean([g > 0 for g in gains])),
        n_folds=len(results),
    )


def feature_utility_with_interval(
    frame: pd.DataFrame,
    feature: str,
    horizon: int,
    *,
    folds: Sequence = FOLDS,
) -> tuple:
    """上面的基础上再加一个按月份分块的 bootstrap 区间。

    区间用**逐日误差差**算，不是用折级增益算——折只有三个，
    在三个数上做 bootstrap 没有意义。
    """
    utility = feature_utility(frame, feature, horizon, folds=folds)
    data = build_dataset(frame, feature, horizon)
    data["change"] = data["target"] - data["close"]
    data["year"] = data.index.year
    data["month"] = data.index.to_period("M").astype(str)

    deltas, base_errors, months = [], [], []
    for train_through, test_year in folds:
        train = data[data["year"] <= train_through]
        if len(train) > horizon:
            train = train.iloc[:-horizon]
        test = data[data["year"] == test_year]
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        prediction = _fit_predict(train, test)
        actual = test["change"].to_numpy()
        base = np.abs(actual)
        deltas.append(base - np.abs(prediction - actual))
        base_errors.append(base)
        months.append(test["month"].to_numpy())
    if not deltas:
        return utility, {"low": None, "high": None, "n_blocks": 0}

    interval = _block_bootstrap(
        np.concatenate(deltas), np.concatenate(base_errors), np.concatenate(months))
    utility.gain_low = interval["low"]
    utility.gain_high = interval["high"]
    return utility, interval


def set_error(
    frame: pd.DataFrame,
    features: Sequence[str],
    horizon: int,
    *,
    folds: Sequence = FOLDS,
) -> dict:
    """整个特征集的样本外误差（同一个固定 FA、同一套折）。

    空特征集就是朴素基准（预测零变化）。B0/B1/B2/E5 全靠这一个函数比，
    口径完全一致——不然换了特征集误差变了，分不清是特征好还是口径变了。
    """
    from src.core.research_dataset import EVENT_FEATURES

    features = list(features)
    columns = {
        name: (frame[name].astype(float) if name in EVENT_FEATURES
               else frame[name].astype(float).diff())
        for name in features
    }
    data = pd.DataFrame({
        **columns,
        "close": frame["USDCNY"],
        "target": frame["USDCNY"].shift(-horizon),
    }).dropna()
    data["change"] = data["target"] - data["close"]
    data["year"] = data.index.year

    rows = []
    for train_through, test_year in folds:
        train = data[data["year"] <= train_through]
        if len(train) > horizon:
            train = train.iloc[:-horizon]
        test = data[data["year"] == test_year]
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue

        if features:
            model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
            model.fit(train[features].to_numpy(), train["change"].to_numpy())
            prediction = model.predict(test[features].to_numpy())
        else:
            prediction = np.zeros(len(test))

        actual = test["change"].to_numpy()
        rows.append({
            "test_year": test_year, "n_test": len(test),
            "mae": float(np.mean(np.abs(prediction - actual))),
            "mape": float(np.mean(np.abs(prediction - actual)
                                  / np.maximum(np.abs(test["close"].to_numpy()), 1e-9))),
            "rmse": float(np.sqrt(np.mean((prediction - actual) ** 2))),
            "direction_accuracy": float(np.mean(
                np.sign(prediction) == np.sign(actual))),
        })

    if not rows:
        return {"n_features": len(features), "n_folds": 0}
    return {
        "n_features": len(features),
        "n_folds": len(rows),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "mape": float(np.mean([r["mape"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "direction_accuracy": float(np.mean([r["direction_accuracy"] for r in rows])),
        "by_fold": rows,
    }


def run_ablation(
    frame: pd.DataFrame,
    features: Sequence[str],
    horizon: int,
    *,
    folds: Sequence = FOLDS,
    verbose: bool = True,
) -> Dict[str, PredictiveUtility]:
    """对所有候选特征跑一遍消融。"""
    out: Dict[str, PredictiveUtility] = {}
    for index, feature in enumerate(features, start=1):
        utility, _ = feature_utility_with_interval(frame, feature, horizon, folds=folds)
        out[feature] = utility
        if verbose and index % 10 == 0:
            print(f"    已完成 {index}/{len(features)}")
    return out
