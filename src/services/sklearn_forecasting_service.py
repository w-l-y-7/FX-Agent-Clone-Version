"""基于 scikit-learn 的汇率预测服务。

用带标准化的 Ridge 回归，把特征矩阵映射到 horizon 个交易日之后的**收盘价变化量**，
再加回当前价得到预测价格。不直接上 `src/legacy_models/` 里的 TFT / TimesNet，
是因为那类深度模型需要几千行序列样本才站得住脚，而单次请求只拉得到几百到
两千个交易日，先跑通闭环更有意义。

超参是固定的（alpha=1.0）。想让 Optuna 去搜超参、并用 SHAP 解释特征贡献，
换用 `OptimizedForecastingService` 即可——**两边共用 `core/backtest.py` 里的
同一套切分和目标定义**，所以报出来的 MAE 可以直接比。

为什么是预测「变化量」而不是像论文那样预测「价格本身」，见 `core/backtest.py`
开头：非平稳序列上做水平回归会系统性外推失真，实测 MAE 差 17 倍。
"""

from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core.abstractions.base_forecasting import BaseForecasting
from ..core.backtest import (
    MIN_MEANINGFUL_CHANGE,
    change_target,
    chronological_holdout,
    regression_report,
)
from ..core.feature_engineering import split_dataset

_MODEL_NAME = "StandardScaler + Ridge(alpha=1.0)，预测变化量"
_RIDGE_ALPHA = 1.0


class SklearnForecastingService(BaseForecasting):
    """A real forecasting service backed by a regularized linear model."""

    def predict(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        features = list(params["features"])
        horizon = int(params.get("horizon", 5))

        train, pending = split_dataset(data, features, horizon)

        # 按时间顺序切分，绝不能随机切——随机切会让模型用"未来的行情"去评估
        # "过去的预测"，指标会好得离谱且毫无意义。
        fit_rows, test_rows = chronological_holdout(train)

        model = make_pipeline(StandardScaler(), Ridge(alpha=_RIDGE_ALPHA))
        # 先在 fit 段上拟合一次，拿它去测试段上量误差；再用全部训练数据重拟合，
        # 做真正的预测。两步分开是为了让"评估用的模型"和"预测用的模型"口径清楚。
        model.fit(fit_rows[features], change_target(fit_rows))

        test_actual = test_rows["target"]
        test_pred = test_rows["close"] + model.predict(test_rows[features])
        backtest = regression_report(test_actual, test_pred, test_rows["close"])

        model.fit(train[features], change_target(train))

        origin = pending.iloc[-1]
        last_close = float(origin["close"])
        change = float(model.predict(origin[features].to_frame().T)[0])
        if abs(change) < MIN_MEANINGFUL_CHANGE:
            change = 0.0
        predicted_close = last_close + change

        coefficients = model.named_steps["ridge"].coef_
        importance = sorted(
            zip(features, coefficients), key=lambda pair: abs(pair[1]), reverse=True
        )

        origin_date = pd.Timestamp(origin["date"])
        return {
            "model": _MODEL_NAME,
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
            # 系数是在标准化之后算的，所以不同特征之间可以直接比大小
            "top_coefficients": [
                {"feature": name, "coefficient": round(float(value), 4)}
                for name, value in importance[:5]
            ],
        }
