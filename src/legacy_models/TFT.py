"""Temporal Fusion Transformer（TFT）。

# 这个文件的状态：**已补全，能跑**。原实现坏在哪里、怎么修的，都记在下面。

论文 Table 7 / Table 9 里 TFT 是所有模型里误差最小的（RMSE 0.0330 / MAPE 0.3427%），
所以这个模型对复现论文很关键。但公开代码里它的 `forward` 是坏的——不是参数没填，
而是网络结构的形状对不上，一跑就崩。

## 原来坏在哪里（逐层追下来的结论，不是猜的）

**问题一：`VariableSelectionNetwork` 把时间轴一起 flatten 了。**

原 `VariableSelectionNetwork.forward` 第一行是：

    flat_x = x.view(x.size(0), -1)          # (B, ...) -> (B, 剩下所有维度乘起来)

它对 `static` 那一路是**碰巧对的**：输入 `(B, N, 1)` 展开成 `(B, N)`，正好是
GRN 期望的 `N` 个输入。但对 `past` / `future` 那两路，输入是 `(B, T, N, 1)`，
展开成 `(B, T×N)`——比 GRN 期望的 `N` 多了 T 倍，于是第一层
`nn.Linear(N, hidden_dim)` 直接报形状不匹配。

**问题二：它返回的是权重，不是特征。**

真正的 TFT 里，`VariableSelectionNetwork` 的输出应该是
`Σ_i w_i · v_i`，其中 `v_i ∈ R^{hidden_dim}` 是第 i 个变量投影后的向量，
也就是一个 `hidden_dim` 维的向量。原实现返回的是权重本身 `(B, 1, N, 1)`，
下游把 N 个变量加权求和后压成了**一个标量**，而不是 `hidden_dim` 维向量。

**问题三：于是下一步必然崩。**

`static_enrichment` 是 `GatedResidualNetwork(hidden_dim, ...)`，期望输入
`hidden_dim=64` 维，但上一步给它的是 `(B, 1)`。

## 改了什么

`VariableSelectionNetwork` 按标准 TFT 重写（约 30 行）：

1. 每个变量各配一个 `GatedResidualNetwork(1, hidden_dim, hidden_dim)` 做投影；
2. flatten-GRN 作用在 `N × input_dim` 上（可带 static context），输出 N 个权重，
   softmax 归一化；
3. 返回 `Σ w_i · v_i ∈ R^hidden_dim` **以及权重本身**——权重就是 TFT 的
   可解释性来源，不丢掉，但从 `forward` 的主返回值里挪到可选参数 `return_weights`；
4. `static_context` reshape 成 `(B, 1, hidden_dim)` 后与时序输入相加。

改了 `forward` 的调用方（`static_vsn` / `past_vsn` / `future_vsn` 三处解包二元组），
其余网络结构、损失函数、训练循环都是作者原样。

## 一处**故意保留**的原设计

`self_attn(lstm_out, lstm_out, lstm_out)` 没有加因果掩码，而原版 TFT 是加的。
这里保持原样，原因有两条：

* 本模型注意力的三个输入全部是**预测时已知**的量（历史观测 + 已知的未来协变量），
  不存在"看到未来目标值"的泄漏；
* 加了掩码就改变了网络结构，和论文的结果不再可比。

也就是说这是作者的简化，不是 bug。要按原版 TFT 改，就在 `forward` 里给
`self_attn` 传一个上三角为 `-inf` 的 `attn_mask`。

## 关于特征集（三个模型共有的一处存疑，见 `_default_column_roles`）

公开代码给的特征集**不含汇率自身的价格**。这在这类建模里不寻常：汇率近似
随机游走，"上一期价格"几乎是最强的单变量预测子，朴素基准就是靠它拿到 0.08 的
RMSE。模型看不见价格在哪，就只能预测训练区间的均值，**结构上不可能跑赢朴素基准**。

所以这里有 `INCLUDE_PRICE_HISTORY` 开关，默认打开（把 `close` 作为历史协变量）。
置为 `False` 就退回论文公开代码的原始口径。两条都能跑，建议两条都跑一遍对比——
这也是一个很值得跟论文作者确认的问题。
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.research_dataset import (
    EVENT_FEATURES,
    TARGET,
    TIME_SERIES_FEATURES,
    load_daily_frame,
)
from src.core.sequence_dataset import format_metrics, regression_metrics, resolve_device

# ============================================================================
# PART 1: ALGORITHM FRAMEWORK DEFINITION
# ============================================================================

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        skip = x
        if self.skip_proj:
            skip = self.skip_proj(skip)

        x = self.input_proj(x)
        if context is not None and self.context_proj is not None:
            # context 是 (B, hidden_dim)，而 x 可能是 (B, hidden_dim) 也可能是
            # (B, T, hidden_dim)。直接相加会**从最右边对齐**，把 context 的 B
            # 拿去和 T 比，报 "size of tensor a (T) must match tensor b (B)"。
            # 所以要在中间补上长度为 1 的维度：(B, hidden) -> (B, 1, hidden)。
            #
            # 这段是补全 VSN 时才变得可达的——原实现从没给 GRN 传过 context_dim，
            # 所以 context_proj 一直是 None，这条分支是死代码，写错了也不会报。
            projected_context = self.context_proj(context)
            while projected_context.dim() < x.dim():
                projected_context = projected_context.unsqueeze(-2)
            x = x + projected_context

        x = torch.relu(x)
        x = self.hidden_layer(x)
        g = self.gate(x)
        x = self.dropout(x * g)
        x = self.output_proj(x)

        return self.layer_norm(x + skip)

class VariableSelectionNetwork(nn.Module):
    """变量选择网络（**已补全**，原实现的两个问题见文件开头）。

    支持任意前导维度，所以静态路 `(B, N, 1)` 和时序路 `(B, T, N, 1)` 用同一份
    实现：`(B, T, N, 1)` 会被看成 `leading=(B, T)`、`num_inputs=N`、`input_dim=1`。

    `forward` 返回 `(加权后的特征向量, 每个变量的权重)`。权重是 TFT 可解释性的
    来源，所以一并给出，由调用方决定要不要用。
    """

    def __init__(self, input_dim, hidden_dim, num_inputs, dropout=0.1, context_dim=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        self.input_dim = input_dim

        # 每个变量各配一个 GRN，把自己从 input_dim 维投影到 hidden_dim 维
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
        # 把所有变量拼起来，一次性算出 N 个选择权重
        self.flatten_grn = GatedResidualNetwork(
            input_dim * num_inputs, hidden_dim, num_inputs, dropout,
            context_dim=context_dim,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        *leading, num_inputs, input_dim = x.shape
        if num_inputs != self.num_inputs or input_dim != self.input_dim:
            raise ValueError(
                f"VariableSelectionNetwork 期望最后两维是 "
                f"({self.num_inputs}, {self.input_dim})，收到 ({num_inputs}, {input_dim})。"
            )

        flat = x.reshape(*leading, num_inputs * input_dim)
        weights = self.softmax(self.flatten_grn(flat, context))

        projected = torch.stack(
            [grn(x[..., index, :]) for index, grn in enumerate(self.variable_grns)],
            dim=-2,
        )
        combined = torch.sum(weights.unsqueeze(-1) * projected, dim=-2)
        return combined, weights

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_static_inputs, num_past_inputs, num_future_inputs,
                 sequence_length, horizon, output_quantiles, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.num_static_inputs = num_static_inputs
        self.num_past_inputs = num_past_inputs
        self.num_future_inputs = num_future_inputs
        self.output_quantiles = output_quantiles

        # 时序两路的变量选择要看静态上下文，所以把 context_dim 传进去
        self.static_vsn = VariableSelectionNetwork(1, hidden_dim, num_static_inputs)
        self.past_vsn = VariableSelectionNetwork(
            1, hidden_dim, num_past_inputs, dropout, context_dim=hidden_dim
        )
        self.future_vsn = VariableSelectionNetwork(
            1, hidden_dim, num_future_inputs, dropout, context_dim=hidden_dim
        )

        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        self.lstm_encoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_gate = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        self.decoder_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        self.output_proj = nn.Linear(hidden_dim * horizon, horizon * len(output_quantiles))

    def forward(self, x_static, x_past, x_future, *, return_weights=False):
        """
        x_static : (B, N_s)          静态协变量
        x_past   : (B, T, N_p)       历史观测到的协变量
        x_future : (B, H, N_f)       预测时已知的未来协变量
        返回      : (B, H, Q)        每个分位数的预测值；Q 是 output_quantiles 的个数
        """
        if x_past.size(1) != self.sequence_length:
            raise ValueError(
                f"x_past 的时间维是 {x_past.size(1)}，但模型是按 "
                f"sequence_length={self.sequence_length} 建的。"
            )

        # 三路变量选择。静态路没有上下文，时序两路以静态嵌入为上下文
        static_embedding, static_weights = self.static_vsn(x_static.unsqueeze(-1))
        static_context = self.static_enrichment(static_embedding)

        past_embedding, past_weights = self.past_vsn(x_past.unsqueeze(-1), static_context)
        future_embedding, future_weights = self.future_vsn(x_future.unsqueeze(-1), static_context)

        # (B, T+N_f, hidden_dim)
        temporal_input = torch.cat([past_embedding, future_embedding], dim=1)
        enriched_input = temporal_input + static_context.unsqueeze(1)

        lstm_out, _ = self.lstm_encoder(enriched_input)

        # 注意：这里没有因果掩码，是作者的原设计，不是 bug。理由见文件开头。
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_gate(attn_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        # 只取未来那一段解码成预测
        decoder_out = self.decoder_grn(attn_out[:, self.sequence_length:, :])

        output = decoder_out.reshape(decoder_out.size(0), -1)
        output = self.output_proj(output)
        output = output.view(output.size(0), self.horizon, len(self.output_quantiles))

        if return_weights:
            return output, {
                "static": static_weights,
                "past": past_weights,
                "future": future_weights,
            }
        return output

def prepare_tft_data(data, static_cols, past_cols, future_cols, target, sequence_length,
                     horizon, origin_col=None):
    """把「一行一天」的表切成 TFT 要的三路张量。

    `origin_col` 是**预测起点当天就能拿到**的那一列（对汇率来说就是当天收盘价）。
    传了它，每个样本会多出第 5 个元素，只用来算朴素基准——训练/评估函数只取
    前四个，不受影响。

    注意不能拿 `target` 列顶替：`target[j] = close[j + horizon]` 是未来值，用它
    当"上一期已知价"等于偷看答案，基准误差会小得离谱，而且无论预测步长多大都
    几乎不变。这是个很难发现的坑，`sequence_dataset.py` 里也踩过一次。
    """
    data_list = []
    for i in range(len(data) - sequence_length - horizon + 1):
        past_end = i + sequence_length
        future_end = past_end + horizon

        static_features = data[static_cols].iloc[i].values
        past_features = data[past_cols].iloc[i:past_end].values
        future_features = data[future_cols].iloc[past_end:future_end].values
        target_values = data[target].iloc[past_end:future_end].values

        sample = [static_features, past_features, future_features, target_values]
        if origin_col is not None:
            # 窗口最后一格就是预测起点，取它的已知值
            sample.append(float(data[origin_col].iloc[past_end - 1]))
        data_list.append(tuple(sample))

    return data_list

def train_tft_model(data, model, optimizer, loss_fn, device, batch_size=64):
    model.train()
    total_loss = 0
    np.random.shuffle(data)

    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]

        # 只取前四个元素：第 5 个（起点的已知价）是给朴素基准用的，不参与训练
        x_static = torch.FloatTensor(np.array([item[0] for item in batch])).to(device)
        x_past = torch.FloatTensor(np.array([item[1] for item in batch])).to(device)
        x_future = torch.FloatTensor(np.array([item[2] for item in batch])).to(device)
        y_true = torch.FloatTensor(np.array([item[3] for item in batch])).to(device)

        optimizer.zero_grad()
        y_pred = model(x_static, x_past, x_future)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / (len(data) / batch_size)

def evaluate_tft_model(data, model, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds, all_trues, all_origins = [], [], []
    with torch.no_grad():
        for item in data:
            x_static = torch.FloatTensor(item[0]).unsqueeze(0).to(device)
            x_past = torch.FloatTensor(item[1]).unsqueeze(0).to(device)
            x_future = torch.FloatTensor(item[2]).unsqueeze(0).to(device)
            y_true = torch.FloatTensor(item[3]).unsqueeze(0).to(device)

            y_pred = model(x_static, x_past, x_future)
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item()
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_true.cpu().numpy())
            if len(item) > 4:
                all_origins.append(item[4])

    origins = np.array(all_origins) if all_origins else np.zeros(0)
    return (total_loss / len(data), np.concatenate(all_preds, axis=0),
            np.concatenate(all_trues, axis=0), origins)

def plot_tft_predictions(preds, trues, quantile_idx_p50, quantile_idx_lower, quantile_idx_upper, num_to_plot=100):
    plt.figure(figsize=(15, 7))
    plt.plot(trues[:num_to_plot, 0], 'b-', label='Actual')
    plt.plot(preds[:num_to_plot, 0, quantile_idx_p50], 'r-', label='P50 Forecast')
    plt.fill_between(
        np.arange(num_to_plot),
        preds[:num_to_plot, 0, quantile_idx_lower],
        preds[:num_to_plot, 0, quantile_idx_upper],
        color='red', alpha=0.2, label='P10-P90 Range'
    )
    plt.title('TFT Forecast vs Actual')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    output = Path(__file__).resolve().parents[2] / "reports" / "tft_predictions.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"图已保存到：{output}")

    if matplotlib.get_backend().lower() not in ("agg", "pdf", "ps", "svg", "cairo"):
        plt.show()

# ============================================================================
# PART 2: USER CONFIGURATION & EXECUTION TEMPLATE
# ============================================================================

def default_column_roles(include_price_history: bool = True):
    """给 TFT 用的静态 / 历史 / 未来三类协变量。

    **需要说明的两个取舍：**

    ① **静态协变量那一路在数学上是退化的。** 静态协变量指的是"不随时间变化的
    每序列属性"（比如门店编号、产品类别），而这里只有一条汇率序列，并不存在
    这样的属性。这里的处理是只放一个恒为 1 的 `series_id`，把退化摆在明面上，
    好过编一个假的静态特征。

    ② **`include_price_history` 默认打开，把汇率自身的价格放进历史协变量。**
    公开代码给的特征集只有 7 个宏观变量 + 25 个事件哑变量，不含价格本身。
    汇率近似随机游走，"上一期价格"几乎是最强的单变量预测子——朴素基准就是
    靠它拿到 0.08 的 RMSE。模型看不见价格在哪，就只能预测训练区间的均值，
    **结构上不可能跑赢朴素基准**。置为 False 可退回公开代码的原始口径。
    """
    future_cols = ["day_of_week", "day_of_month", "month", "is_month_end"]
    past_cols = list(TIME_SERIES_FEATURES) + list(EVENT_FEATURES)
    if include_price_history:
        # 放首位，方便和 SHAP / 变量选择权重的输出对照
        past_cols = ["close"] + past_cols
    static_cols = ["series_id"]
    return static_cols, past_cols, future_cols


def build_tft_frame(horizon: int = 1) -> pd.DataFrame:
    """加载论文数据，并造出 TFT 需要的列。"""
    daily = load_daily_frame()
    frame = daily.reset_index().rename(columns={"Date": "date"})

    index = pd.DatetimeIndex(daily.index)
    # `close` 会被当作模型输入（可能被缩放）；`origin_close` 永不缩放，
    # 专门用来算朴素基准，两者必须是同一份数值。
    frame["close"] = daily[TARGET].to_numpy()
    frame["origin_close"] = daily[TARGET].to_numpy()
    frame["target"] = daily[TARGET].shift(-horizon).to_numpy()
    frame["series_id"] = 1.0
    # 下面四个是"事前已知协变量"：预测未来时它们一定拿得到，不构成前视偏差
    frame["day_of_week"] = index.dayofweek
    frame["day_of_month"] = index.day
    frame["month"] = index.month
    frame["is_month_end"] = index.is_month_end.astype(float)

    return frame.dropna(subset=["target"]).reset_index(drop=True)


def chronological_split(samples: list, test_size: float = 0.2):
    """按时间顺序切分，绝不 shuffle——时序数据随机切会让未来信息泄漏进训练集。"""
    cut = int(len(samples) * (1.0 - test_size))
    return samples[:cut], samples[cut:]


if __name__ == '__main__':

    # --- 1. Setup Device ---
    device = resolve_device()
    print(f"Using device: {device}")

    # --- 2. Load the Paper's Real Data ---
    print("\nLoading the paper's USD/CNY dataset...")
    SEQUENCE_LENGTH = 30
    # 预测步长。论文没写明这个值，详见 Transformer.py 里的详细说明：按 1 天预测
    # 的话朴素基准的 RMSE 只有约 0.022，论文报的任何模型都赢不了它；只有在多步
    # 预测（h≈20）的设定下，论文那组数字才讲得通。三个模型取同一个值才好对比。
    HORIZON = 20
    TEST_SIZE = 0.2
    INCLUDE_PRICE_HISTORY = True

    df = build_tft_frame(horizon=HORIZON)
    STATIC_COLS, PAST_COLS, FUTURE_COLS = default_column_roles(INCLUDE_PRICE_HISTORY)

    # --- 3. Scale ---
    # 只用训练段 fit 缩放器，理由同 sequence_dataset.py 里的说明：先切分、后缩放。
    # `origin_close` 故意不在这里缩放，它要保持原始量纲给朴素基准用。
    cut = int(len(df) * (1.0 - TEST_SIZE))
    scaled = df.copy()
    for columns in (PAST_COLS, FUTURE_COLS, ["target"]):
        scaler = MinMaxScaler().fit(df[columns].iloc[:cut])
        scaled[columns] = scaler.transform(df[columns])
    target_scaler = MinMaxScaler().fit(df[["target"]].iloc[:cut])

    # --- 4. Prepare Sequences ---
    full_data = prepare_tft_data(
        scaled, STATIC_COLS, PAST_COLS, FUTURE_COLS, "target",
        SEQUENCE_LENGTH, HORIZON, origin_col="origin_close",
    )
    train_data, test_data = chronological_split(full_data, TEST_SIZE)

    print(f"\n目标：预测 {HORIZON} 个交易日后的 USD/CNY 收盘价")
    print(f"序列长度 {SEQUENCE_LENGTH}，静态 {len(STATIC_COLS)} 个 / "
          f"历史 {len(PAST_COLS)} 个 / 未来 {len(FUTURE_COLS)} 个协变量")
    print(f"  历史协变量含汇率自身价格：{INCLUDE_PRICE_HISTORY}")
    print(f"总样本 {len(full_data)} 个，训练 {len(train_data)} 个，测试 {len(test_data)} 个")

    # --- 5. Model ---
    OUTPUT_QUANTILES = [0.1, 0.5, 0.9]
    model = TemporalFusionTransformer(
        num_static_inputs=len(STATIC_COLS),
        num_past_inputs=len(PAST_COLS),
        num_future_inputs=len(FUTURE_COLS),
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        output_quantiles=OUTPUT_QUANTILES,
        hidden_dim=64,
        num_heads=4,
    ).to(device)
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # --- 6. Train ---
    loss_fn = QuantileLoss(quantiles=OUTPUT_QUANTILES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n开始训练……")
    for epoch in range(300):
        train_loss = train_tft_model(train_data, model, optimizer, loss_fn, device)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/300], Loss: {train_loss:.4f}')

    # --- 7. Evaluate ---
    # 指标只算 P50（中位数）那一路，才能和其他模型、和论文的 RMSE 对齐
    test_loss, preds, trues, origins = evaluate_tft_model(test_data, model, loss_fn, device)
    print(f'\nTest Loss: {test_loss:.4f}')

    p50 = OUTPUT_QUANTILES.index(0.5)
    # 取**第 0 步**的输出，才能和 Transformer/TimesNet 对上。
    #
    # 模型每步输出的含义：第 k 步预测的是 `close[起点 + k + HORIZON]`。
    # 所以第 0 步 = 起点之后 HORIZON+1 个交易日的价格，正好和另外两个模型
    # （它们的 target 也是 `close[i + seq_len + horizon]`）同一个口径。
    # 取最后一步就会变成 2×HORIZON−1 天的预测，和谁都比不了；
    # 绘图函数画的也是第 0 步，两边这样才一致。
    step = 0
    y_pred = target_scaler.inverse_transform(preds[:, step, p50].reshape(-1, 1)).ravel()
    y_true = target_scaler.inverse_transform(trues[:, step].reshape(-1, 1)).ravel()

    metrics = regression_metrics(y_true, y_pred)
    baseline = regression_metrics(y_true, origins) if origins.size else None

    print('\nTest Set Evaluation Results（P50 中位数，起点后第 0 步）：')
    print(format_metrics(metrics, baseline))
    print("\n（论文 TFT + 完整流程的对照值：RMSE 0.0330，MAPE 0.3427%）")

    plot_tft_predictions(
        preds=preds, trues=trues,
        quantile_idx_p50=p50,
        quantile_idx_lower=0,
        quantile_idx_upper=-1,
        num_to_plot=min(100, len(test_data)),
    )
