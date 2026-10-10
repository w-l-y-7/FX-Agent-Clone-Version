"""逻辑门控 + 尺度化 Huber 损失。

    g_q    = σ(θᵀz_q)
    r̂_E4   = (1 − g_q)·r̂_E1 + g_q·r̂_E2
    L(θ)   = Σ_q ρ_δ( (r_q − r̂_E4) / scale_q ) + λ‖θ‖²

## 几个刻意的选择

- **训练目标就是预测误差**，不是 NDCG。最终要比的是 MAE / 方向准确率，
  拿排序指标训练会和目标错位；
- **尺度化**：不同事件的异常收益量级差很多，不除 scale 的话损失会被大波动
  事件主导。`scale_q` 只能用查询日之前的数据算；
- **Huber 而不是平方**：汇率异常收益有肥尾，平方损失会被离群事件带跑；
- **不用神经网络**：17 维特征、263 条样本，逻辑门控足够，
  再深只会过拟合。用 `scipy.optimize` 就够了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

# Huber 的转折点，单位是「多少个 scale」
HUBER_DELTA = 1.0
_EPS = 1e-12


def scale_for(values: Sequence[Optional[float]]) -> float:
    """查询日前异常收益的稳健尺度（1.4826 × MAD）。

    **只能传查询日之前的数**。查询自己的收益进来就是泄漏。
    """
    clean = np.asarray([v for v in values if v is not None], dtype=float)
    if clean.size < 2:
        return 0.01
    mad = float(np.median(np.abs(clean - np.median(clean))))
    scale = 1.4826 * mad
    return scale if scale > _EPS else float(max(clean.std(ddof=1), 0.01))


def huber_loss(residuals: np.ndarray, delta: float = HUBER_DELTA) -> float:
    absolute = np.abs(residuals)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return float((0.5 * quadratic ** 2 + delta * linear).sum())


def _huber_gradient(residuals: np.ndarray, delta: float = HUBER_DELTA) -> np.ndarray:
    return np.clip(residuals, -delta, delta)


@dataclass
class GateModel:
    """训练好的门控。`coefficients` 是标准化空间里的 θ（含截距）。"""

    feature_names: List[str]
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    lam: float = 0.0
    delta: float = HUBER_DELTA
    n_train: int = 0

    def standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.feature_mean) / self.feature_std

    def design(self, x: np.ndarray) -> np.ndarray:
        standardised = self.standardise(np.atleast_2d(x))
        return np.column_stack([np.ones(standardised.shape[0]), standardised])

    def gate(self, x: np.ndarray) -> np.ndarray:
        logit = self.design(x) @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))

    def blend(self, x: np.ndarray, r1, r2) -> np.ndarray:
        g = self.gate(x)
        return (1.0 - g) * np.asarray(r1, dtype=float) + g * np.asarray(r2, dtype=float)

    def describe(self) -> List[dict]:
        return [
            {"feature": name, "coefficient": float(value)}
            for name, value in zip(["intercept", *self.feature_names],
                                   self.coefficients)
        ]

    def to_config(self) -> dict:
        """序列化成可复现的配置。

        **标准化的均值/方差必须一起存。** 只存系数的话，在别的数据集上
        会拿不同的标准化去算，门控值对不上——而且不会报错，只会算错。
        """
        return {
            "feature_names": list(self.feature_names),
            "coefficients": [float(v) for v in self.coefficients],
            "feature_mean": [float(v) for v in self.feature_mean],
            "feature_std": [float(v) for v in self.feature_std],
            "lam": float(self.lam),
            "delta": float(self.delta),
            "n_train": int(self.n_train),
        }

    @classmethod
    def from_config(cls, config: dict) -> "GateModel":
        return cls(
            feature_names=list(config["feature_names"]),
            coefficients=np.asarray(config["coefficients"], dtype=float),
            feature_mean=np.asarray(config["feature_mean"], dtype=float),
            feature_std=np.asarray(config["feature_std"], dtype=float),
            lam=float(config.get("lam", 0.0)),
            delta=float(config.get("delta", HUBER_DELTA)),
            n_train=int(config.get("n_train", 0)),
        )


def fit_gate(
    features: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    *,
    feature_names: Sequence[str],
    lam: float = 0.1,
    delta: float = HUBER_DELTA,
    max_iter: int = 500,
) -> Optional[GateModel]:
    """拟合门控。样本太少或优化失败时返回 None。"""
    features = np.asarray(features, dtype=float)
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    target = np.asarray(target, dtype=float)
    scales = np.maximum(np.asarray(scales, dtype=float), _EPS)

    if features.shape[0] < 20:
        return None

    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < _EPS, 1.0, std)
    design = np.column_stack([np.ones(features.shape[0]),
                              (features - mean) / std])

    gap = (r2 - r1) / scales
    base = (target - r1) / scales

    def objective(theta: np.ndarray) -> float:
        logit = design @ theta
        g = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        residuals = base - g * gap
        return huber_loss(residuals, delta) + lam * float(theta[1:] @ theta[1:])

    def gradient(theta: np.ndarray) -> np.ndarray:
        logit = design @ theta
        g = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        residuals = base - g * gap
        dg = g * (1.0 - g)
        # dL/dg = -ρ'(u)·gap
        dldg = -_huber_gradient(residuals, delta) * gap
        grad = design.T @ (dldg * dg)
        grad[1:] += 2.0 * lam * theta[1:]
        return grad

    result = minimize(objective, np.zeros(design.shape[1]), jac=gradient,
                      method="L-BFGS-B", options={"maxiter": max_iter})
    if not result.success and result.fun > objective(np.zeros(design.shape[1])):
        return None

    return GateModel(
        feature_names=list(feature_names),
        coefficients=np.asarray(result.x, dtype=float),
        feature_mean=mean,
        feature_std=std,
        lam=lam,
        delta=delta,
        n_train=int(features.shape[0]),
    )
