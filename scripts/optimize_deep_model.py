"""按作者的方式用 Optuna 调深度模型的超参（论文 §3.3.4 的 FA）。

    # 先跑 5 组试水，看流程通不通
    .\\venv\\Scripts\\python.exe scripts\\optimize_deep_model.py --trials 5

    # 正式跑
    .\\venv\\Scripts\\python.exe scripts\\optimize_deep_model.py --model LSTM_Attention --trials 20

    # 换成作者工厂支持的另一个模型
    .\\venv\\Scripts\\python.exe scripts\\optimize_deep_model.py --model Transformer

搜索空间、批大小、每 trial 的轮数都取自作者的
`src/utils/hyperparameter_optimizer.py`（原件见 `author_original_code/`），
判据改成验证段而不是测试集——理由见 `src/core/deep_optimization.py` 开头。

结果写到 `reports/deep_optimization.json`。
"""

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.core.deep_optimization import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRIAL_EPOCHS,
    SUPPORTED_MODELS,
    build_deep_model,
    optimize_deep_hyperparameters,
    train_deep_model,
)
from src.core.research_dataset import (  # noqa: E402
    TARGET,
    ablation_feature_set,
    load_daily_frame,
)
from src.core.sequence_dataset import (  # noqa: E402
    build_sequence_dataset,
    chronological_train_val_split,
    format_metrics,
    inverse_target,
    naive_baseline,
    regression_metrics,
    resolve_device,
)

_HORIZON = 20
_SEQUENCE_LENGTH = 30
_TEST_SIZE = 0.2


def _build_dataset(feature_set: str, include_price_history: bool):
    daily = load_daily_frame(verbose=True)
    frame = daily.reset_index().rename(columns={"Date": "date"})
    frame["close"] = daily[TARGET].to_numpy()
    frame["target"] = daily[TARGET].shift(-_HORIZON).to_numpy()
    frame = frame.dropna(subset=["target"]).reset_index(drop=True)

    features = ablation_feature_set(feature_set)
    if include_price_history:
        features = ["close"] + features

    dataset = build_sequence_dataset(
        frame,
        features=features,
        target="target",
        sequence_length=_SEQUENCE_LENGTH,
        test_size=_TEST_SIZE,
        origin_column="close",
    )
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用 Optuna 调深度模型超参（作者的搜索空间）"
    )
    parser.add_argument("--model", default="LSTM_Attention", choices=list(SUPPORTED_MODELS))
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Optuna 搜多少组超参。作者默认 50；每组要训 50 轮，CPU 上慢，默认取 20。",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_TRIAL_EPOCHS,
        help=f"每个 trial 训多少轮，默认 {DEFAULT_TRIAL_EPOCHS}（作者的取值）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"批大小，默认 {DEFAULT_BATCH_SIZE}（作者的取值）",
    )
    parser.add_argument("--feature-set", default="pa1_pa2_da",
                        choices=["pa1", "pa1_pa2", "pa1_pa2_da"])
    parser.add_argument("--no-price-history", action="store_true",
                        help="输入不含汇率自身价格（默认含）")
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="从训练段里再切多少比例做验证段（调参判据）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--output", type=Path,
                        default=_PROJECT_ROOT / "reports" / "deep_optimization.json")
    args = parser.parse_args()

    device = resolve_device()
    print(f"Using device: {device}")

    dataset = _build_dataset(args.feature_set, not args.no_price_history)
    print(f"\n{dataset.describe()}")
    print(f"模型 {args.model}，特征集 {args.feature_set}，"
          f"输入含价格历史 {not args.no_price_history}")

    fit_set, val_set = chronological_train_val_split(
        dataset, val_fraction=args.val_fraction, gap=_SEQUENCE_LENGTH
    )
    print(f"调参用：「拟合段」{tuple(fit_set.X_train.shape)} / "
          f"「验证段」{tuple(val_set.X_train.shape)}"
          f"（两段之间空出 {_SEQUENCE_LENGTH} 个样本，避免窗口重叠）")

    print(f"\n开始搜索：{args.trials} 组超参 × {args.epochs} 轮，"
          f"批大小 {args.batch_size}。这一步慢，请耐心等。")
    started = time.perf_counter()
    report = optimize_deep_hyperparameters(
        fit_set.X_train, fit_set.y_train,
        val_set.X_train, val_set.y_train,
        model_name=args.model,
        n_trials=args.trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )
    print(f"\n搜索结束：{report.describe()}")
    print(f"  最佳超参：{report.best_params}")

    # ---- 用最佳超参在**全部训练段**上重新训练，最后才碰测试集 ----
    print("\n用最佳超参在全部训练段上重新训练……")
    final_model = build_deep_model(
        args.model,
        report.best_params,
        input_dim=dataset.input_dim,
    )
    train_deep_model(
        final_model, dataset.X_train, dataset.y_train,
        learning_rate=report.best_params["learning_rate"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    final_model.eval()
    with torch.no_grad():
        prediction = final_model(dataset.X_test.to(device)).cpu().numpy()
    y_true = inverse_target(dataset.y_test.numpy(), dataset.scaler_y)
    y_pred = inverse_target(prediction, dataset.scaler_y)

    metrics = regression_metrics(y_true, y_pred)
    baseline = naive_baseline(y_true, dataset.y_prev_test)

    print("\n测试集结果（用调好的超参）：")
    print(format_metrics(metrics, baseline))

    payload = {
        "model": args.model,
        "feature_set": args.feature_set,
        "include_price_history": not args.no_price_history,
        "horizon": _HORIZON,
        "sequence_length": _SEQUENCE_LENGTH,
        "best_params": report.best_params,
        "search": {
            "n_trials": report.n_trials,
            "seconds": round(report.seconds, 1),
            "epochs_per_trial": report.epochs_per_trial,
            "batch_size": report.batch_size,
            "best_val_rmse_scaled": report.best_value,
            "n_fit_rows": report.n_train_rows,
            "n_val_rows": report.n_val_rows,
        },
        "test_metrics": {k: round(v, 6) for k, v in metrics.items()},
        "naive_baseline": {k: round(v, 6) for k, v in baseline.items()},
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完整结果已写入：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
