"""循环神经网络（RNN），论文 Table 7 的五个基准模型之一。

# 这个文件的状态：**按论文正文补全的，不属于原公开代码**

原仓库 `legacy_models/` 里**没有**这个文件。论文 Table 7 评了五个模型
（Transformer、LSTM+Attention、RNN、TFT、TimesNet），公开代码只给了三个模板，
RNN 和 LSTM+Attention 是缺的。这里按论文 §4.1 的设定补齐，补的依据只有两条：

* **超参数**：隐藏维度 64、dropout 0.1、学习率 0.001、训练 300 轮（论文明写）；
* **网络类型**：论文只说是 "RNN"，没有给层数、有没有投影层等结构细节。

**所以这个模型的"结构"是项目自己定的，不是论文的结构。** 层数（3 层）、输入投影
这些属于常见做法，但不是从论文里读出来的。要跟论文作者确认就得连这一条一起问。

一个刻意的选择：RNN 和 LSTM+Attention 都**沿用 Transformer.py 的训练循环**
（整批全量梯度、Adam + ReduceLROnPlateau、梯度裁剪 1.0），而不是自己另写一套。
这样三个模型之间只差网络结构，指标才有可比性；论文用什么优化器、批次多大都没写，
统一口径比自己猜三套要诚实。

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
    """标准的多层 RNN：输入投影 → RNN → 取最后一步 → 全连接。

    `dropout` 只在 `num_layers > 1` 时传给 `nn.RNN`——单层 RNN 之间没有可丢弃的
    连接，torch 会为此发一条警告。层间 dropout 之外，输出头里也保留了一路。
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.input_proj(x)
        output, _ = self.rnn(x)
        # 和 Transformer.py 取同样的位置：窗口最后一天的表征
        return self.output_layer(output[:, -1, :])


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
    NUM_LAYERS = 3
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
