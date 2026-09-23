"""循环神经网络（RNN），论文 Table 7 的五个基准模型之一。

# 这个文件的状态：**网络结构取自作者被删除的原始实现**

原仓库当前的 `legacy_models/` 里没有这个文件；但作者的 git 历史里**有**
`src/models/RNN.py`，在 2025-07-16 的提交 `a838298` 里被删掉了。本文件的
网络结构是从那个版本恢复的，原件保存在 `author_original_code/models/RNN.py`。

从作者版本里取到的结构特征（**这三条和"常见写法"都不一样，是作者的选择**）：

* `nn.RNN` **直接吃原始特征**，没有输入投影层；
* `nonlinearity='relu'`，不是 torch 默认的 `tanh`；
* 输出头是 `Linear → ReLU → Dropout → Linear`，没有 LayerNorm。

作者版本里的驱动代码是模板（`FEATURES = [...]`、随机数据），所以**训练循环、
数据管道、指标口径**仍沿用本项目的统一实现（和 `Transformer.py` / `Timesnet.py`
一致：整批全量梯度、Adam + ReduceLROnPlateau、梯度裁剪 1.0）。这样五个模型之间
只差网络结构，指标才有可比性。

## 一处取自作者模板的取值

`NUM_LAYERS = 2`——作者 `LSTM_Attention.py` 模板里写的是 2（`RNN.py` 模板里
那一行是 `...` 占位符，没给值）。论文 §4.1 只交代了隐藏维度、dropout、学习率、
训练轮数，没提层数，所以这里用作者模板给的值。

## 和论文的对照值

论文 Table 7 在 `FA(χ) + PA1 + PA2 + DA` 这一列报的是 **RMSE 0.0650、
MAPE 0.8121%**。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.research_dataset import (
    TARGET,
    ablation_feature_set,
    load_daily_frame,
)
from src.core.sequence_dataset import (
    build_sequence_dataset,
    format_metrics,
    inverse_target,
    naive_baseline,
    regression_metrics,
    resolve_device,
)

# ============================================================================
# PART 1: ALGORITHM FRAMEWORK DEFINITION
# ============================================================================

class RNNModel(nn.Module):
    """多层 RNN，结构照搬作者被删除的 `src/models/RNN.py`。

    `dropout` 只在 `num_layers > 1` 时传给 `nn.RNN`——单层 RNN 之间没有可丢弃的
    连接，torch 会为此发一条警告。这一行和作者原版一致。
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity="relu",
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        # 和 Transformer.py 取同样的位置：窗口最后一天的表征
        return self.output_layer(rnn_out[:, -1, :])


def train_model(X_train, y_train, input_dim, hidden_dim, num_layers, output_dim,
                device, epochs=300, learning_rate=0.001, dropout=0.1):
    """和 `Transformer.py` 完全同一套训练循环，只换网络。见文件开头说明。"""
    model = RNNModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    model.train()
    train_losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, y_train.to(device))

        if torch.isnan(loss):
            print(f"Epoch {epoch+1}: Loss is NaN. Stopping training.")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.detach())
        train_losses.append(loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    return model, train_losses


def evaluate_model(model, X_test, y_test, scaler_y, device="cpu"):
    """返回 (rmse, mape_pct, y_true, y_pred)，都在原始量纲（人民币元）上算。"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device))

    y_true = inverse_target(y_test.cpu().numpy(), scaler_y)
    y_pred = inverse_target(predictions.cpu().numpy(), scaler_y)

    metrics = regression_metrics(y_true, y_pred)
    return metrics["rmse"], metrics["mape_pct"], y_true, y_pred


def plot_predictions(y_true, y_pred, title="RNN: Prediction vs Actual"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual Values", color="blue")
    plt.plot(y_pred, label="Predicted Values", color="red", linestyle="--")
    plt.title(title, fontsize=16)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    output = Path(__file__).resolve().parents[2] / "reports" / "rnn_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"图已保存到：{output}")

    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo"):
        plt.show()

# ============================================================================
# PART 2: USER CONFIGURATION & EXECUTION TEMPLATE
# ============================================================================

if __name__ == "__main__":

    device = resolve_device()
    print(f"Using device: {device}")

    print("Loading the paper's USD/CNY dataset...")
    daily = load_daily_frame(verbose=True)

    # 预测步长，和另外几个模型保持一致，详见 Transformer.py 里的详细说明
    HORIZON = 20

    df = daily.reset_index().rename(columns={"Date": "date"})
    df["close"] = daily[TARGET].to_numpy()
    # 必须用 .to_numpy() 绕开索引对齐，否则整列都是 NaN（Transformer.py 里有说明）
    df["target"] = daily[TARGET].shift(-HORIZON).to_numpy()
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # 特征集的三个选项对应论文 Table 7 消融实验的三行
    FEATURE_SET = "pa1_pa2_da"

    # 见 Transformer.py 里的详细说明：论文 Table 4 的特征表含 USD/CNY 本身，
    # 三个模型的这个开关默认值保持一致，指标才可比。
    INCLUDE_PRICE_HISTORY = True

    FEATURES = ablation_feature_set(FEATURE_SET)
    if INCLUDE_PRICE_HISTORY:
        FEATURES = ["close"] + FEATURES
    TARGET_COLUMN = "target"

    # 前四项来自论文 §4.1；序列长度与测试集比例论文未给，取常规值
    SEQUENCE_LENGTH = 30
    TEST_SIZE = 0.2

    HIDDEN_DIM = 64
    # 取自作者 LSTM_Attention 模板里的取值（见文件开头说明）；论文没交代层数
    NUM_LAYERS = 2
    DROPOUT = 0.1
    LEARNING_RATE = 0.001
    EPOCHS = 300

    print(f"\n目标：预测 {HORIZON} 个交易日后的 USD/CNY 收盘价")
    print(f"特征集：{FEATURE_SET}，共 {len(FEATURES)} 个特征")
    print(f"  {FEATURES}")
    print(f"  输入含汇率自身价格：{INCLUDE_PRICE_HISTORY}")

    print("\nPreparing data...")
    dataset = build_sequence_dataset(
        df,
        features=FEATURES,
        target=TARGET_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        test_size=TEST_SIZE,
        # 朴素基准要拿"预测起点当天已知的收盘价"，不是目标列里的未来值
        origin_column="close",
    )
    print(f"  {dataset.describe()}")

    print("Starting model training...")
    trained_model, losses = train_model(
        dataset.X_train, dataset.y_train,
        input_dim=dataset.input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=1,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
    )

    print("Evaluating model...")
    rmse, mape_pct, y_true, y_pred = evaluate_model(
        trained_model, dataset.X_test, dataset.y_test, dataset.scaler_y, device
    )

    metrics = regression_metrics(y_true, y_pred)
    baseline = naive_baseline(y_true, dataset.y_prev_test)

    print("\nTest Set Evaluation Results:")
    print(format_metrics(metrics, baseline))
    print("\n（论文 RNN + 完整流程的对照值：RMSE 0.0650，MAPE 0.8121%）")

    print("Plotting results...")
    plot_predictions(y_true, y_pred)
