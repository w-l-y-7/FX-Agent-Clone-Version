import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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

class FrequencyLayer(nn.Module):
    """在频域上做一次线性变换。

    说明：这里对 rfft 的实部和虚部各自乘一个独立的 d_model×d_model 权重矩阵。
    严格来说这不是一个合法的实信号频域滤波——实信号的频谱有共轭对称性，
    要保持输出仍是实信号，需要约束两侧的变换互为共轭。这里没有这个约束。

    这个问题**故意保留原样**：它是作者的实现，不是笔误，改掉就意味着和论文
    的实验结果不再可比。记录在这里是为了让你知道它简化在哪，而不是让你去改它。
    """

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        fft_x = torch.fft.rfft(x, dim=1)

        weighted_real = F.linear(fft_x.real, self.weight)
        weighted_imag = F.linear(fft_x.imag, self.weight)

        weighted_fft = torch.complex(weighted_real, weighted_imag)
        x_reconstructed = torch.fft.irfft(weighted_fft, dim=1, n=seq_len)
        return x_reconstructed

class TimesNetBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.freq_layer = FrequencyLayer(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        x = self.freq_layer(x)
        x = self.norm1(x + residual)
        x = self.dropout1(x)
        
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        x = self.dropout2(x)
        return x

class TimesNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.timesnet_blocks = nn.ModuleList([
            TimesNetBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.timesnet_blocks:
            x = block(x)
        output = self.output_layer(x[:, -1, :])
        return output

def train_model(X_train, y_train, input_dim, hidden_dim, num_layers, output_dim, dropout, device, epochs=300):
    model = TimesNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        
        outputs = model(X_train_gpu)
        loss = criterion(outputs, y_train_gpu)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return model, train_losses

def evaluate_model(model, X_test, y_test, scaler_y, device='cuda'):
    """返回 (rmse, mape_pct, y_true, y_pred)，全部在原始量纲（人民币元）上算。"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device))

    y_true = inverse_target(y_test.cpu().numpy(), scaler_y)
    y_pred = inverse_target(predictions.cpu().numpy(), scaler_y)

    metrics = regression_metrics(y_true, y_pred)
    return metrics['rmse'], metrics['mape_pct'], y_true, y_pred

def plot_predictions(y_true, y_pred, title='Prediction vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Values', color='blue')
    plt.plot(y_pred, label='Predicted Values', color='red', linestyle='--')
    plt.title(title, fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output = Path(__file__).resolve().parents[2] / "reports" / "timesnet_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"图已保存到：{output}")

    # 无图形界面的后端（Agg 等）调 show() 只会刷一句警告，所以先判断一下
    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo"):
        plt.show()

# ============================================================================
# PART 2: USER CONFIGURATION & EXECUTION TEMPLATE
# ============================================================================

if __name__ == '__main__':

    # --- 1. Setup Device ---
    device = resolve_device()
    print(f"Using device: {device}")

    # --- 2. Load the Paper's Data ---
    print("Loading the paper's USD/CNY dataset...")
    daily = load_daily_frame(verbose=True)

    # 预测步长。论文没写明这个值，详见 Transformer.py 里的详细说明：
    # 按 1 天预测的话，朴素基准的 RMSE 只有约 0.017，论文报的任何模型都赢不了它；
    # 只有在多步预测（h≈20）的设定下，论文那组数字才讲得通。
    HORIZON = 20

    df = daily.reset_index().rename(columns={'Date': 'date'})
    df['close'] = daily[TARGET].to_numpy()
    # 见 Transformer.py 里同样的注释：必须用 .to_numpy() 绕开索引对齐
    df['target'] = daily[TARGET].shift(-HORIZON).to_numpy()
    df = df.dropna(subset=['target']).reset_index(drop=True)

    # --- 3. Define All Parameters ---
    # 前三项和 Transformer 保持一致，来自论文 §4.1；序列长度与测试集比例论文未给，
    # 取常规值。两个模型用同一份数据、同一套切分，指标才有可比性。
    #
    # 特征集的三个选项对应论文 Table 7 消融实验的三行，详见 research_dataset.py
    # 里 ablation_feature_set() 的说明。要和 Transformer 比，就选同一个。
    FEATURE_SET = 'pa1_pa2_da'

    # 是否把汇率自身的价格（`close`）放进输入特征。理由和 Transformer.py 里
    # 写的一样，详见那一段：公开代码的特征集不含价格，模型结构上赢不了朴素基准。
    # 三个模型（Transformer / TimesNet / TFT）这个开关的默认值保持一致，
    # 指标才有可比性。
    INCLUDE_PRICE_HISTORY = True

    FEATURES = ablation_feature_set(FEATURE_SET)
    if INCLUDE_PRICE_HISTORY:
        FEATURES = ['close'] + FEATURES
    TARGET_COLUMN = 'target'
    SEQUENCE_LENGTH = 30
    TEST_SIZE = 0.2

    HIDDEN_DIM = 64
    NUM_LAYERS = 3
    DROPOUT = 0.1
    EPOCHS = 300

    print(f"\n目标：预测 {HORIZON} 个交易日后的 USD/CNY 收盘价")
    print(f"特征集：{FEATURE_SET}，共 {len(FEATURES)} 个特征")
    print(f"  {FEATURES}")
    print(f"  输入含汇率自身价格：{INCLUDE_PRICE_HISTORY}")

    # --- 4. Run the Full Pipeline ---

    print("\nPreparing data...")
    dataset = build_sequence_dataset(
        df,
        features=FEATURES,
        target=TARGET_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        test_size=TEST_SIZE,
        # 朴素基准要拿"预测起点当天已知的收盘价"，不是目标列里的未来值
        origin_column='close',
    )
    print(f"  {dataset.describe()}")

    print("Starting model training...")
    trained_model, losses = train_model(
        dataset.X_train, dataset.y_train,
        input_dim=dataset.input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=1,
        dropout=DROPOUT,
        device=device,
        epochs=EPOCHS
    )

    print("Evaluating model...")
    rmse, mape_pct, y_true, y_pred = evaluate_model(
        trained_model, dataset.X_test, dataset.y_test, dataset.scaler_y, device
    )

    metrics = regression_metrics(y_true, y_pred)
    baseline = naive_baseline(y_true, dataset.y_prev_test)

    print('\nTest Set Evaluation Results:')
    print(format_metrics(metrics, baseline))
    print("\n（论文 TimesNet + 完整流程的对照值：RMSE 0.0436，MAPE 0.4874%）")

    print("Plotting results...")
    plot_predictions(y_true, y_pred, title='TimesNet: USD/CNY next-day forecast')