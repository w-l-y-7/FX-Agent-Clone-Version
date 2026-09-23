"""FA 的第二个预测服务：**Optuna 调参 + SHAP 解释**。

和 `SklearnForecastingService` 的区别只有一个：那边用的是固定超参的 Ridge，
这边让 Optuna 在四类表格模型（Ridge / 随机森林 / 梯度提升 / 小 MLP）里搜一轮，
再用 SHAP 把「哪些特征推着预测往哪边走」拆出来。

**两边共用同一套回测口径**（`core/backtest.py`）：同一个目标定义、
同一种按时间顺序的切分、同一条朴素基准。这样两个服务报出来的 MAE 才能直接比。

## 切分层次（防止拿测试集调参）

    全部数据
      └─ train（前 80%）
      │    ├─ fit 段（前 80%）        ← Optuna 在这里拟合
      │    └─ 验证段（后 20%）        ← Optuna 只在这里比较超参
      └─ test（后 20%）               ← 从头到尾只用来算最终指标，算一次

选完超参后用**全部 train**（fit 段 + 验证段）重新拟合，最后才碰 test。
`optimize_hyperparameters()` 的签名里根本没有测试集，这条约束是结构上保证的。
"""

from typing import Any, Dict, Optional

import pandas as pd

from ..core.abstractions.base_forecasting import BaseForecasting
from ..core.backtest import (
    MIN_MEANINGFUL_CHANGE,
    change_target,
    chronological_holdout,
    regression_report,
)
from ..core.feature_engineering import split_dataset
from ..core.model_optimization import (
    build_model,
    explain_with_shap,
    intrinsic_importance,
    optimize_hyperparameters,
)

_DEFAULT_N_TRIALS = 30
_DEFAULT_SEED = 42
_VALIDATION_FRACTION = 0.2
_MIN_VALIDATION_ROWS = 10

# 送进 SHAP 的特征超过这个数，输出会变得不好读；同时也可以省点时间。
_TOP_FEATURES = 10


class OptimizedForecastingService(BaseForecasting):
    """先用 Optuna 挑模型，再用 SHAP 解释它。"""

    def __init__(
        self,
        *,
        n_trials: int = _DEFAULT_N_TRIALS,
        timeout: Optional[float] = None,
        seed: int = _DEFAULT_SEED,
        explain: bool = True,
        verbose: bool = True,
    ):
        self._n_trials = n_trials
        self._timeout = timeout
        self._seed = seed
        self._explain = explain
        self._verbose = verbose

    def predict(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        features = list(params["features"])
        horizon = int(params.get("horizon", 5))

        train, pending = split_dataset(data, features, horizon)

        # 第一层：训练段内部再切一次，验证段只给 Optuna 用
        fit_rows, test_rows = chronological_holdout(train)
        inner_fit, inner_val = chronological_holdout(
            fit_rows,
            test_fraction=_VALIDATION_FRACTION,
            min_test_rows=_MIN_VALIDATION_ROWS,
        )

        if self._verbose:
            print(
                f"\n开始 Optuna 调参：{self._n_trials} 组超参，"
                f"拟合段 {len(inner_fit)} 行 / 验证段 {len(inner_val)} 行"
            )

        report = optimize_hyperparameters(
            inner_fit[features], change_target(inner_fit),
            inner_val[features], change_target(inner_val),
            n_trials=self._n_trials,
            timeout=self._timeout,
            seed=self._seed,
        )
        if self._verbose:
            print(f"  {report.describe()}")

        # 用全部训练段重新拟合选出来的那组超参，再回到测试段上算最终指标
        model = build_model(report.best_params, seed=self._seed)
        model.fit(train[features], change_target(train))

        test_actual = test_rows["target"]
        test_pred = test_rows["close"] + model.predict(test_rows[features])
        backtest = regression_report(test_actual, test_pred, test_rows["close"])

        # ---------- 解释 ----------
        explanation = []
        if self._explain:
            if self._verbose:
                print("  算 SHAP 特征贡献……")
            explanation = explain_with_shap(model, train[features], seed=self._seed)
            if not explanation:
                explanation = intrinsic_importance(model, features)

        # ---------- 对最新一行做真正的预测 ----------
        origin = pending.iloc[-1]
        last_close = float(origin["close"])
        change = float(model.predict(origin[features].to_frame().T)[0])
        if abs(change) < MIN_MEANINGFUL_CHANGE:
            change = 0.0
        predicted_close = last_close + change

        origin_date = pd.Timestamp(origin["date"])
        family = report.best_params.get("family", "?")
        return {
            "model": f"Optuna 调参（选中 {family}）+ {len(features)} 个特征",
            "symbol": params.get("symbol"),
            "horizon_trading_days": horizon,
            "origin_date": origin_date.date().isoformat(),
            "forecast_date_estimate": (
                origin_date + pd.tseries.offsets.BDay(horizon)
            ).date().isoformat(),
            "last_close": round(last_close, 4),
            "predicted_close": round(predicted_close, 4),
            "predicted_change_pct": round(
                (predicted_close / last_close - 1) * 100, 4
            ),
            "confidence": round(backtest.confidence, 4),
            "beats_naive_baseline": backtest.beats_naive_baseline,
            "evaluation": backtest.as_dict(
                train_rows=len(fit_rows), test_rows=len(test_rows)
            ),
            "optimization": {
                "best_params": report.best_params,
                "validation_rmse": round(report.best_value, 6),
                "n_trials": report.n_trials,
                "seconds": round(report.seconds, 1),
                "trials_per_family": report.trials_per_family,
                "validation_rows": report.n_val_rows,
            },
            "explanation": explanation[:_TOP_FEATURES],
        }
