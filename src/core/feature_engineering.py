"""把抓到的原始汇率序列加工成预测模型可以吃的特征矩阵。

特征清单同时充当"契约"：决策智能体只能从 `feature_catalog()` 里挑特征，
挑不中的名字会被丢弃并列进结果，避免 LLM 凭空捏造本地根本拿不到的数据。
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

_RSI_WINDOW = 14
_BB_WINDOW = 20


def _prefix(symbol: str) -> str:
    return symbol.lower().replace("/", "_")


def feature_catalog(symbol: str) -> Dict[str, str]:
    """当前数据条件下真正算得出来的特征，键为特征名，值为中文说明。"""
    p = _prefix(symbol)
    return {
        f"{p}_lag_1": "前 1 个交易日收盘价",
        f"{p}_lag_2": "前 2 个交易日收盘价",
        f"{p}_lag_5": "前 5 个交易日收盘价",
        f"{p}_return_1d": "最近 1 个交易日涨跌幅",
        f"{p}_return_5d": "最近 5 个交易日累计涨跌幅",
        f"{p}_return_20d": "最近 20 个交易日累计涨跌幅",
        f"{p}_ma_5": "5 日均线",
        f"{p}_ma_10": "10 日均线",
        f"{p}_ma_20": "20 日均线",
        f"{p}_ma_ratio_5_20": "5 日均线 ÷ 20 日均线，短长期均线之比",
        f"{p}_volatility_5": "5 日收益率标准差，短期波动率",
        f"{p}_volatility_20": "20 日收益率标准差，中期波动率",
        f"{p}_rsi_14": "14 日相对强弱指标 RSI，高于 70 偏超买、低于 30 偏超卖",
        f"{p}_bb_width_20": "20 日布林带宽度 ÷ 中轨，衡量波动扩张程度",
        f"{p}_bb_position_20": "收盘价在 20 日布林带中的相对位置，0=下轨、1=上轨",
        f"{p}_momentum_10": "10 日动量，收盘价 ÷ 10 日前收盘价 - 1",
        f"{p}_zscore_20": "收盘价相对 20 日均线的标准化偏离",
        f"{p}_trend_20": "20 日线性回归斜率，衡量趋势强度",
        "day_of_week": "星期几，0=周一 … 4=周五",
    }


def default_features(symbol: str) -> List[str]:
    """LLM 选的特征一个都对不上时的兜底组合。"""
    p = _prefix(symbol)
    return [
        f"{p}_lag_1",
        f"{p}_return_1d",
        f"{p}_return_5d",
        f"{p}_ma_ratio_5_20",
        f"{p}_rsi_14",
    ]


def build_feature_frame(
    series: List[Dict[str, Any]], symbol: str, horizon: int = 5
) -> pd.DataFrame:
    """把 `[{"date", "close"}, ...]` 展开成一行一天的特征矩阵。

    多出一列 `target`，值为 horizon 个交易日之后的收盘价；序列末尾
    horizon 行的 target 是空的——那正是我们要预测的部分。
    """
    frame = pd.DataFrame(series)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    close = frame["close"].astype(float)
    returns = close.pct_change()
    p = _prefix(symbol)

    out = pd.DataFrame({"date": frame["date"], "close": close})

    out[f"{p}_lag_1"] = close.shift(1)
    out[f"{p}_lag_2"] = close.shift(2)
    out[f"{p}_lag_5"] = close.shift(5)
    out[f"{p}_return_1d"] = returns
    out[f"{p}_return_5d"] = close.pct_change(5)
    out[f"{p}_return_20d"] = close.pct_change(20)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    out[f"{p}_ma_5"] = ma5
    out[f"{p}_ma_10"] = ma10
    out[f"{p}_ma_20"] = ma20
    out[f"{p}_ma_ratio_5_20"] = ma5 / ma20
    out[f"{p}_volatility_5"] = returns.rolling(5).std()
    out[f"{p}_volatility_20"] = returns.rolling(20).std()

    # RSI 用 Wilder 平滑（等价于 alpha = 1/14 的指数移动平均）
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(
        alpha=1 / _RSI_WINDOW, min_periods=_RSI_WINDOW, adjust=False
    ).mean()
    loss = (-delta.clip(upper=0.0)).ewm(
        alpha=1 / _RSI_WINDOW, min_periods=_RSI_WINDOW, adjust=False
    ).mean()
    out[f"{p}_rsi_14"] = 100.0 - 100.0 / (1.0 + gain / loss)

    sd20 = close.rolling(_BB_WINDOW).std()
    upper = ma20 + 2 * sd20
    lower = ma20 - 2 * sd20
    out[f"{p}_bb_width_20"] = (upper - lower) / ma20
    out[f"{p}_bb_position_20"] = (close - lower) / (upper - lower)
    out[f"{p}_momentum_10"] = close / close.shift(10) - 1.0
    out[f"{p}_zscore_20"] = (close - ma20) / sd20
    out[f"{p}_trend_20"] = _rolling_slope(close, 20)

    out["day_of_week"] = frame["date"].dt.dayofweek
    out["target"] = close.shift(-horizon)
    return out


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """滚动窗口内对时间做一次最小二乘拟合，取斜率。"""
    positions = np.arange(window, dtype=float)
    centered = positions - positions.mean()
    denominator = float((centered**2).sum())

    def slope(values: np.ndarray) -> float:
        return float((centered * (values - values.mean())).sum() / denominator)

    return series.rolling(window).apply(slope, raw=True)


def split_dataset(
    frame: pd.DataFrame, features: List[str], horizon: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """切成"能用来训练"和"等着被预测"两部分。

    训练部分要求特征齐全、且 horizon 天后的真实收盘价已知；
    待预测部分是特征齐全但未来值未知的末尾若干行。
    """
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"特征名不存在于特征矩阵中：{', '.join(missing)}")

    usable = frame.dropna(subset=features)
    train = usable.dropna(subset=["target"])
    pending = usable[usable["target"].isna()]

    if len(pending) == 0:
        raise ValueError("没有可预测的行，历史序列太短了。")
    if len(train) < 60:
        raise ValueError(
            f"训练样本只有 {len(train)} 行，不足以拟合模型；请拉长历史区间。"
        )
    return train, pending
