"""FA 的 Optuna 调参——按**作者的原始实现**调深度模型。

这是 `model_optimization.py` 的姊妹模块，两者对应论文 §3.3.4 的同一句话，
但走的是两条路：

| | `model_optimization.py` | 本模块 |
|---|---|---|
| 调什么 | Ridge / 随机森林 / 梯度提升 / MLP 四类**表格模型** | LSTM+Attention / RNN / Transformer 三类**深度模型** |
| 速度 | 几秒 | 几十分钟 |
| 为什么存在 | 快，而且能顺手出 SHAP 解释 | **和作者的实现一致** |

## 为什么要补这个模块

之前判断"论文没说对哪个模型调参"。**那个判断是错的**——作者的 git 历史里有
`src/utils/hyperparameter_optimizer.py`（2025-07-16 的提交 `a838298` 里被删掉），
里面写得清清楚楚：

* `optuna.create_study(direction="minimize")`；
* 搜索空间 `learning_rate` / `hidden_dim` / `num_layers` / `dropout`；
* 通过 `ModelFactory.create_model(model_name, **params)` 造模型，而那个工厂
  支持的是 **`LSTM`（即 LSTM + Attention）和 `Transformer`**；
* **`batch_size = 32`**，每个 trial 训 50 轮。

原件保存在 `author_original_code/utils/hyperparameter_optimizer.py`。
**`batch_size = 32` 是这份文件给出的**——它一直列在文档的"论文没交代"清单里。

## 和作者实现的两处**故意不同**

1. **判据不用测试集。** 作者那版 `_objective` 返回的是
   `criterion(model(X_test), y_test)`，也就是拿最终测试集当调参判据——这是
   标准的乐观偏差，报出来的误差没法解释。本模块改成在**验证段**上评判，
   测试集留到最后只碰一次。这一点和 `model_optimization.py` 的取舍一致，
   理由在 `backtest.py` 里写得更细。

2. **搜索空间里的模型名。** 作者工厂写的是 `"LSTM"` / `"Transformer"`，
   本模块用 `"LSTM_Attention"` / `"RNN"` / `"Transformer"`，和
   `src/legacy_models/` 里的类名对齐。

除此之外，搜索空间的范围、`batch_size = 32`、每 trial 50 轮，都照作者的取值。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import torch
import torch.nn as nn

from ..legacy_models.LSTM_Attention import LSTMAttentionModel
from ..legacy_models.RNN import RNNModel
from ..legacy_models.Transformer import TransformerModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---- 以下四个常量都取自作者的 hyperparameter_optimizer.py ----

# 作者的 `batch_size = 32`。这是"论文没交代批次大小"这个问题的答案。
DEFAULT_BATCH_SIZE = 32

# 作者的 `num_epochs = 50`（注释写的是"a small number of epochs for quick evaluation"）
DEFAULT_TRIAL_EPOCHS = 50

# 作者的搜索空间：learning_rate 1e-4~1e-2（对数）、hidden_dim 32~128 步长 32、
# num_layers 1~3、dropout 0.1~0.5
_LEARNING_RATE_RANGE = (1e-4, 1e-2)
_HIDDEN_DIM_RANGE = (32, 128, 32)
_NUM_LAYERS_RANGE = (1, 3)
_DROPOUT_RANGE = (0.1, 0.5)

# 作者工厂给的 Transformer 是另一个模型类，需要 num_heads；搜索空间里没有这一维，
# 所以固定住。取值和 `Transformer.py` 里的默认一致。
_TRANSFORMER_HEADS = 4

DEFAULT_MODEL = "LSTM_Attention"

# 作者工厂支持 LSTM / Transformer；这里的 RNN 是照 `legacy_models/RNN.py` 补的
SUPPORTED_MODELS = ("LSTM_Attention", "RNN", "Transformer")


@dataclass
class DeepOptimizationReport:
    """深度模型调参的结果摘要。"""

    model_name: str
    best_params: Dict[str, Any]
    best_value: float          # 验证段的 RMSE（**缩放后**的量纲，见下方说明）
    n_trials: int
    seconds: float
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs_per_trial: int = DEFAULT_TRIAL_EPOCHS
    n_train_rows: int = 0
    n_val_rows: int = 0

    def describe(self) -> str:
        return (
            f"Optuna 试了 {self.n_trials} 组超参（{self.seconds:.1f} 秒，"
            f"每组 {self.epochs_per_trial} 轮、批大小 {self.batch_size}），"
            f"最好的 {self.model_name}：验证段 RMSE {self.best_value:.6f}"
        )


def suggest_deep_params(trial: optuna.Trial) -> Dict[str, Any]:
    """作者的搜索空间，逐项照搬。

    注意 `hidden_dim` 用 `step=32`，所以只会取 32 / 64 / 96 / 128——这正是
    作者写的 `suggest_int("hidden_dim", 32, 128, step=32)`。
    """
    low, high, step = _HIDDEN_DIM_RANGE
    return {
        "learning_rate": trial.suggest_float("learning_rate", *_LEARNING_RATE_RANGE, log=True),
        "hidden_dim": trial.suggest_int("hidden_dim", low, high, step=step),
        "num_layers": trial.suggest_int("num_layers", *_NUM_LAYERS_RANGE),
        "dropout": trial.suggest_float("dropout", *_DROPOUT_RANGE),
    }


def build_deep_model(
    model_name: str,
    params: Dict[str, Any],
    *,
    input_dim: int,
    output_dim: int = 1,
) -> nn.Module:
    """按名字和超参造一个**未训练**的深度模型。

    对应作者的 `ModelFactory.create_model(model_name, **params)`。搜索时用它、
    最后定稿时也用它，保证"搜到的"和"最后用的"是同一个结构。
    """
    common = {
        "input_dim": input_dim,
        "hidden_dim": params["hidden_dim"],
        "num_layers": params["num_layers"],
        "output_dim": output_dim,
        "dropout": params["dropout"],
    }

    if model_name == "LSTM_Attention":
        return LSTMAttentionModel(**common)
    if model_name == "RNN":
        return RNNModel(**common)
    if model_name == "Transformer":
        return TransformerModel(num_heads=_TRANSFORMER_HEADS, **common)

    raise ValueError(
        f"不认识的模型名：{model_name}（可选 {list(SUPPORTED_MODELS)}）"
    )


def train_deep_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: Optional[int] = None,
) -> nn.Module:
    """按作者的循环训练：Adam + MSELoss + **小批量** + 梯度裁剪。

    批大小默认 32（作者的取值），这和 `legacy_models/*.py` 里那套"整批全量梯度"
    不同——那套是为了让五个模型的对比口径一致；这里是在复现作者调参时的训练方式，
    两者用途不同，所以没有强行统一。
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X = X_train.to(device)
    y = y_train.to(device)

    for _ in range(epochs):
        model.train()
        for start in range(0, len(X), batch_size):
            X_batch = X[start:start + batch_size]
            y_batch = y[start:start + batch_size]

            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model


def _evaluate_rmse(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(X.to(device))
    residual = y.to(device) - prediction
    return float(torch.sqrt(torch.mean(residual ** 2)).item())


def optimize_deep_hyperparameters(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    model_name: str = DEFAULT_MODEL,
    n_trials: int = 20,
    epochs: int = DEFAULT_TRIAL_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[torch.device] = None,
    timeout: Optional[float] = None,
    seed: Optional[int] = None,
    progress: bool = True,
) -> DeepOptimizationReport:
    """用 Optuna 搜深度模型的超参，返回最好的一组。

    **函数签名里没有测试集**——和 `model_optimization.optimize_hyperparameters`
    同一个约定，而且是故意的：判据只能是验证段。

    `best_value` 的单位是**缩放后**的目标量纲（MinMax 的 [0,1] 区间上），
    不是人民币元。它只用来在 trial 之间比大小；最终报给用户的指标由调用方
    在测试集上、还原回人民币元之后算。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(X_train.shape[-1])
    started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = suggest_deep_params(trial)
        model = build_deep_model(model_name, params, input_dim=input_dim)
        train_deep_model(
            model, X_train, y_train,
            learning_rate=params["learning_rate"],
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        return _evaluate_rmse(model, X_val, y_val, device)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(
        objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False
    )

    return DeepOptimizationReport(
        model_name=model_name,
        best_params=dict(study.best_trial.params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        seconds=time.perf_counter() - started,
        batch_size=batch_size,
        epochs_per_trial=epochs,
        n_train_rows=int(len(X_train)),
        n_val_rows=int(len(X_val)),
    )
