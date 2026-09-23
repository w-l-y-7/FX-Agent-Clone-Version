"""FA（预测智能体）的两块自动化：**Optuna 调参** 和 **SHAP 解释**。

README 把这两件写成 FA 的核心能力（"Automates Hyperparameter Optimization
(with Optuna)" / "Ensures Interpretability ... like SHAP"），本模块实现它们。

## 一、调参的验证口径

Optuna 需要一个「哪组超参更好」的判据。**绝不能拿最终测试集当判据**——那等于
用测试集选模型，最后报出来的误差是乐观偏差，而且偏差大小无法估计。

这里的做法是：调用方从**训练段**里再切一段按时间顺序的验证段（`backtest.py`
里的 `chronological_holdout`），Optuna 只在验证段上比较。选完之后用**全部**
训练段重新拟合，最后才碰测试集。测试集从头到尾只被用来算最终指标一次。

**这条约定和作者的原始实现不一样**：作者的
`author_original_code/utils/hyperparameter_optimizer.py` 里，`_objective`
返回的是 `criterion(model(X_test), y_test)`——拿最终测试集当判据。这里没有照做，
因为它会让报出来的误差无法解释。

## 二、本模块搜的是表格模型，而作者调的是深度模型

**这一条是查过作者 git 历史之后才弄清楚的**（早先的版本在这里写的是
"论文没说对哪个模型调的"，那是错的）：

* 作者的 `src/utils/hyperparameter_optimizer.py`（2025-07-16 的 `a838298` 里被删）
  通过 `ModelFactory.create_model(model_name, **params)` 造模型，而那个工厂支持的是
  **`LSTM`（LSTM + Attention）和 `Transformer`**；
* 搜索空间是 `learning_rate` / `hidden_dim` / `num_layers` / `dropout`，
  **`batch_size = 32`**，每个 trial 训 50 轮。

所以论文的 Optuna 调的是**深度模型**。那条路在本项目的
`src/core/deep_optimization.py` + `scripts/optimize_deep_model.py` 里实现，
搜索空间和批大小都照作者的取值。

**本模块为什么还留着**：它搜的是 Ridge / 随机森林 / 梯度提升 / MLP 四类表格模型，
几秒钟跑完，而且能顺手出 SHAP 解释（`explain_with_shap`）。深度模型调一轮要训
50 轮、CPU 上几十分钟，拿不到这种"随手就跑"的体验。两条路都保留，各有各的用途——
要保真用 `optimize_deep_model.py`，要快速看 SHAP 用 `main.py --optimize`。
"""

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# optuna 默认每次 trial 都打一行日志，几百行会把终端冲掉。只留警告。
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 小 MLP 在几百行的数据上跑 500 轮常常"不收敛"，sklearn 会为此刷一堆警告。
# 它只是说迭代次数用完了，不代表结果不可用——搜索空间里的 alpha 本来就是用来
# 控制这种过拟合的，所以这里定向静音，别的警告照常显示。
warnings.filterwarnings("ignore", category=ConvergenceWarning)

_DEFAULT_SEED = 42
_DEFAULT_N_TRIALS = 30

# SHAP 用排列法（PermutationExplainer），对任何 `predict` 都能用。
# 它要对每一行做多次前向计算，所以限制参与解释的行数——100 行足够看出
# 特征重要性的相对次序，再多只是浪费时间。
#
# 取 100 还有个附带好处：shap 的默认 masker 上限就是 100，正好等于它就不会
# 再打印一行"正在从 N 个子采样到 100"的提示（那行是 print 不是 warning，
# 没法用 filterwarnings 关掉）。
_SHAP_MAX_ROWS = 100

# 相关性绝对值低于这个数就认为方向不明确。汇率这类数据里特征与预测的关系
# 本来就很弱，门槛定太高会把所有特征都判成"方向不明确"。
_DIRECTION_MIN_CORRELATION = 0.1

_FAMILIES = ["ridge", "random_forest", "gradient_boosting", "mlp"]


@dataclass
class OptimizationReport:
    """Optuna 搜索的结果摘要。"""

    best_params: Dict[str, Any]
    best_value: float          # 验证段的 RMSE（在「变化量」量纲上）
    n_trials: int
    seconds: float
    trials_per_family: Dict[str, int] = field(default_factory=dict)
    n_train_rows: int = 0
    n_val_rows: int = 0

    def describe(self) -> str:
        family = self.best_params.get("family", "?")
        return (
            f"Optuna 试了 {self.n_trials} 组超参（{self.seconds:.1f} 秒），"
            f"最好的一组是 {family}，验证段 RMSE {self.best_value:.6f}（变化量量纲）"
        )


def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """按模型族给出各自的超参搜索空间。"""
    family = trial.suggest_categorical("family", _FAMILIES)
    params: Dict[str, Any] = {"family": family}

    if family == "ridge":
        params["alpha"] = trial.suggest_float("ridge_alpha", 1e-3, 1e3, log=True)

    elif family == "random_forest":
        params["n_estimators"] = trial.suggest_int("rf_n_estimators", 100, 500, step=100)
        params["max_depth"] = trial.suggest_int("rf_max_depth", 2, 12)
        params["min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 20)

    elif family == "gradient_boosting":
        params["n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 500, step=50)
        params["learning_rate"] = trial.suggest_float("gb_learning_rate", 0.005, 0.3, log=True)
        params["max_depth"] = trial.suggest_int("gb_max_depth", 1, 5)

    else:  # mlp
        params["hidden_layer_sizes"] = (
            trial.suggest_int("mlp_hidden", 8, 128, step=8),
        )
        params["alpha"] = trial.suggest_float("mlp_alpha", 1e-4, 1e0, log=True)

    return params


def build_model(params: Dict[str, Any], *, seed: int = _DEFAULT_SEED):
    """按超参造一个**未拟合**的模型。

    从 trial 里提参数和从 `study.best_params` 里提参数走的是同一个函数，
    这样「搜到的」和「最后用的」不可能是两个不同的模型。
    """
    family = params["family"]

    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"]))

    if family == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=1,
        )

    if family == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            random_state=seed,
        )

    if family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                max_iter=500,
                random_state=seed,
            ),
        )

    raise ValueError(f"不认识的模型族：{family}")


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    n_trials: int = _DEFAULT_N_TRIALS,
    timeout: Optional[float] = None,
    seed: int = _DEFAULT_SEED,
) -> OptimizationReport:
    """用 Optuna 搜一遍超参，返回最佳的一组。

    注意这个函数的签名里**没有测试集**——这是故意的。搜索的判据只能是验证段，
    测试集要留到调用方那边、用全部训练段重新拟合之后才算。
    """
    started = time.perf_counter()
    per_family: Dict[str, int] = {}
    # 每个 trial 实际用的内部参数字典，按 trial 编号存。
    #
    # **不能直接用 `study.best_params`**：那个字典的键是 `suggest_*` 时用的名字
    # （`gb_n_estimators`、`rf_max_depth`…），而 `build_model` 读的是内部名
    # （`n_estimators`、`max_depth`…），两者对不上，直接传会 KeyError。
    # 而 `trial.set_user_attr` 也走不通——它会把值 JSON 化，`hidden_layer_sizes`
    # 这种元组会变成列表，sklearn 不接受。
    built_params: Dict[int, Dict[str, Any]] = {}

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        built_params[trial.number] = params
        per_family[params["family"]] = per_family.get(params["family"], 0) + 1

        model = build_model(params, seed=seed)
        model.fit(X_train, y_train)
        residual = np.asarray(y_val) - np.asarray(model.predict(X_val))
        return float(np.sqrt(np.mean(residual**2)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return OptimizationReport(
        best_params=built_params[study.best_trial.number],
        best_value=float(study.best_value),
        n_trials=len(study.trials),
        seconds=time.perf_counter() - started,
        trials_per_family=per_family,
        n_train_rows=int(len(X_train)),
        n_val_rows=int(len(X_val)),
    )


def explain_with_shap(
    model,
    X: pd.DataFrame,
    *,
    max_rows: int = _SHAP_MAX_ROWS,
    seed: int = _DEFAULT_SEED,
) -> List[Dict[str, Any]]:
    """用 SHAP 排列法算每个特征对预测变化量的贡献。

    返回按重要性从高到低排的列表，每项含：

    * `feature`       特征名
    * `mean_abs_shap` 平均绝对贡献（人民币元）。**排次序看这个**
    * `correlation`   特征值与其 SHAP 值的相关系数，用来定方向
    * `direction`     把相关系数的符号翻译成中文，省得用户自己判

    ## 方向为什么不用「SHAP 的平均值」来定

    一个容易想当然的写法是取 `matrix.mean(axis=0)`，看它正负。**这是错的**：
    同一行所有特征的 SHAP 值加起来等于 `f(x) − E[f(x)]`，对样本平均之后
    正好是 0——所以每个特征的"平均有符号贡献"都趋近于 0，符号纯粹是采样噪音。
    实测在一个 `y = 2a + 0.5b` 的合成数据上，`a` 的平均有符号贡献是 **−0.043**，
    连正负都反了。

    正确做法是看**特征值与该特征 SHAP 值的相关性**（shap 自己的 summary plot
    画的就是这个）。它回答的是"这个特征取值高的时候，它是在把预测往上推还是
    往下推"，这才是"方向"。

    算不出来时返回空列表并打印原因，**不抛异常**——SHAP 只是解释，不该因为它
    失败就让整条预测流程断掉。
    """
    try:
        import shap

        sample = X.iloc[:max_rows] if len(X) > max_rows else X
        explainer = shap.PermutationExplainer(model.predict, sample, seed=seed)
        # silent=True 关掉 tqdm 进度条：它会往终端刷上百行，把结果冲得看不见。
        values = explainer(sample, silent=True)
        matrix = np.asarray(values.values)
        if matrix.ndim == 3:  # 多输出时的兜底，本项目的模型都是单输出
            matrix = matrix[..., 0]
    except Exception as exc:  # noqa: BLE001 —— 解释失败不该拖垮预测
        print(f"SHAP 解释失败（不影响预测结果）：{type(exc).__name__}: {exc}")
        return []

    mean_abs = np.abs(matrix).mean(axis=0)
    feature_values = np.asarray(sample, dtype=float)

    rows = []
    for index, name in enumerate(sample.columns):
        column = feature_values[:, index]
        if float(column.std()) < 1e-12:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(column, matrix[:, index])[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.0

        if correlation >= _DIRECTION_MIN_CORRELATION:
            direction = "该特征走高 → 预测的汇率变化偏上行"
        elif correlation <= -_DIRECTION_MIN_CORRELATION:
            direction = "该特征走高 → 预测的汇率变化偏下行"
        else:
            direction = "方向不明确"

        rows.append({
            "feature": name,
            "mean_abs_shap": round(float(mean_abs[index]), 8),
            "correlation": round(correlation, 4),
            "direction": direction,
        })

    rows.sort(key=lambda item: item["mean_abs_shap"], reverse=True)
    return rows


def intrinsic_importance(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """SHAP 用不了时的兜底：读模型自带的特征重要度。

    线性模型读标准化后的系数，树模型读 `feature_importances_`。
    这只是"重要度"，没有方向、也没有单个样本的贡献分解，所以是退而求其次。
    """
    # 管道的话看最后一步（前面那步是标准化，没有可解释的系数），
    # 不是管道就直接看模型自己
    estimator = (
        list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    )

    values: Optional[np.ndarray] = None
    if hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_)).ravel()
    elif hasattr(estimator, "feature_importances_"):
        values = np.abs(np.asarray(estimator.feature_importances_)).ravel()

    if values is None or len(values) != len(feature_names):
        return []

    rows = [
        {"feature": name, "importance": round(float(value), 6)}
        for name, value in zip(feature_names, values)
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows
