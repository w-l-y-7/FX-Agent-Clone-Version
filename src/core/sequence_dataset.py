"""给 `legacy_models/` 里的深度学习模型准备序列数据。

原版 `prepare_data`（Transformer.py / Timesnet.py 各有一份）有一个不容易察觉
但影响很大的问题：**它在切分训练集/测试集之前，就用全量数据 fit 了 MinMaxScaler。**

后果是测试集的极值会被"看见"，缩放比例里带着未来信息，于是测试误差显得比
真实情况更小。这个坑不会报错、不会让训练失败，只会让你得到一个偏乐观的数字，
然后你拿着这个数字去找别人复现不出来。本模块的顺序是**先切分、再只用训练段
fit 缩放器**，从源头上堵住它。

另外原版返回的是 `(..., scaler_y, scaler_X)`，y 在前 X 在后，很容易搞反；
本模块返回一个带字段名的 dataclass，取错字段会直接报错而不是静默出错。

顺带说明一个口径问题：sklearn 的 `mean_absolute_percentage_error` 返回的是
**0~1 的小数**，而论文 Table 7 / Table 9 里报的 MAPE 是**百分数**。
本模块两个都给：`mape` 是小数，`mape_pct` 是百分数——和论文对照时用后者。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@dataclass
class SequenceDataset:
    """已经切分、已经缩放好的序列数据。

    字段名是完整的：`scaler_X` / `scaler_y` 不再靠位置区分。
    """

    X_train: torch.Tensor
    X_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""
    dates_test: List[pd.Timestamp] = field(default_factory=list)
    # 每个测试样本在**预测起点当天**已知的收盘价，用来算朴素基准。
    # 注意不是目标列的未来值——那会把基准算成作弊。
    y_prev_test: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def input_dim(self) -> int:
        return int(self.X_train.shape[-1])

    @property
    def sequence_length(self) -> int:
        return int(self.X_train.shape[1])

    def describe(self) -> str:
        return (
            f"训练 {tuple(self.X_train.shape)} / 测试 {tuple(self.X_test.shape)}，"
            f"序列长度 {self.sequence_length}，特征 {self.input_dim} 个"
        )


def build_sequence_dataset(
    frame: pd.DataFrame,
    features: List[str],
    target: str,
    *,
    sequence_length: int = 30,
    test_size: float = 0.2,
    origin_column: Optional[str] = None,
) -> SequenceDataset:
    """把「一行一天」的特征表切成序列样本，并做无泄漏的缩放。

    `origin_column` 用来算朴素基准：它应当是**预测起点当天就能拿到的已知值**
    （对汇率来说就是当天的收盘价列），而不是目标列。传 None 就不算基准。

    这里有个很容易踩的坑：`target` 列存的是未来值（`target[j] = close[j+h]`），
    如果拿 `target[i+seq_len-1]` 当"上一期已知值"，实际取到的是 `close[i+seq_len-1+h]`
    ——那已经是未来某天的真实价了。样本之间高度重叠，于是这条"基准"等于偷看了
    答案，误差会小得离谱，而且无论预测步长多大都几乎不变。必须传原始收盘价列。
    """
    required = features + [target] + ([origin_column] if origin_column else [])
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"特征表里没有这些列：{missing}")

    cleaned = frame.dropna(subset=features + [target]).reset_index(drop=True)
    if len(cleaned) <= sequence_length + 2:
        raise ValueError(
            f"有效样本只有 {len(cleaned)} 行，序列长度却是 {sequence_length}，"
            "切不出足够的序列，请拉长历史区间或调小 sequence_length。"
        )

    # ---------- 第一步：先在原始行上定切分点 ----------
    split_row = int(len(cleaned) * (1.0 - test_size))
    if split_row <= sequence_length or split_row >= len(cleaned) - 1:
        raise ValueError(
            f"切分点落在第 {split_row} 行，对序列长度 {sequence_length} 来说不合适。"
        )

    # ---------- 第二步：只用训练段 fit 缩放器（这一步就是防泄漏的关键）----------
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(cleaned[features].iloc[:split_row])
    scaler_y.fit(cleaned[target].iloc[:split_row].to_numpy().reshape(-1, 1))

    scaled_X = scaler_X.transform(cleaned[features])
    scaled_y = scaler_y.transform(cleaned[target].to_numpy().reshape(-1, 1)).ravel()

    # ---------- 第三步：构造序列，并按切分点分成两段 ----------
    last_start = len(cleaned) - sequence_length
    train_starts = [
        i for i in range(last_start) if i + sequence_length < split_row
    ]
    test_starts = [
        i for i in range(last_start) if i + sequence_length >= split_row
    ]
    if not train_starts or not test_starts:
        raise ValueError("切分后训练集或测试集为空，请调整 test_size 或 sequence_length。")

    def build(starts: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.stack([scaled_X[i:i + sequence_length] for i in starts])
        y = np.array([scaled_y[i + sequence_length] for i in starts])
        if origin_column is None:
            previous = np.zeros(0)
        else:
            # 起点就是窗口的最后一格，所以要取 i + sequence_length - 1 这一天的已知值
            previous = np.array([
                float(cleaned[origin_column].iloc[i + sequence_length - 1])
                for i in starts
            ])
        return X, y, previous

    X_train_np, y_train_np, _ = build(train_starts)
    X_test_np, y_test_np, y_prev = build(test_starts)

    return SequenceDataset(
        X_train=torch.FloatTensor(X_train_np),
        X_test=torch.FloatTensor(X_test_np),
        y_train=torch.FloatTensor(y_train_np).reshape(-1, 1),
        y_test=torch.FloatTensor(y_test_np).reshape(-1, 1),
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        feature_names=list(features),
        target_name=target,
        dates_test=[
            cleaned["date"].iloc[i + sequence_length]
            for i in test_starts
            if "date" in cleaned.columns
        ],
        y_prev_test=y_prev,
    )


def inverse_target(values: np.ndarray, scaler_y: MinMaxScaler) -> np.ndarray:
    """把缩放后的预测值还原回原始量纲（人民币元）。"""
    return scaler_y.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """回归误差指标。

    `rmse` / `mae` 是原始量纲（人民币元），`mape` 是小数，`mape_pct` 是百分数。
    论文 Table 7 / Table 9 报的是 `mape_pct`。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    denominator = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denominator)))

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape,
        "mape_pct": mape * 100.0,
    }


def naive_baseline(y_true: np.ndarray, y_previous: np.ndarray) -> Dict[str, float]:
    """朴素基准：假设下一个交易日的价格和今天一样。

    汇率近似随机游走，这条基线不花钱、不训练，却常常打败复杂的模型。
    任何模型报出的误差，都要和它比一比才有意义。
    """
    return regression_metrics(y_true, y_previous)


def format_metrics(
    metrics: Dict[str, float], baseline: Optional[Dict[str, float]] = None
) -> str:
    lines = [
        f"  RMSE    {metrics['rmse']:.4f}（人民币元）",
        f"  MAE     {metrics['mae']:.4f}（人民币元）",
        f"  MAPE    {metrics['mape_pct']:.4f}%（论文报的就是这个口径）",
    ]
    if baseline:
        lines.append(f"  朴素基准 RMSE {baseline['rmse']:.4f} / "
                     f"MAPE {baseline['mape_pct']:.4f}%")
        better = metrics["rmse"] < baseline["rmse"]
        lines.append(
            f"  对比结论：{'跑赢' if better else '没跑赢'}「假设价格不变」的朴素基准"
        )
    return "\n".join(lines)


def resolve_device(preference: Optional[str] = None) -> torch.device:
    """选设备。这台机器只有 CPU，所以实际总是回落到 cpu。"""
    if preference:
        return torch.device(preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
