"""CP1：冲突尺度化共形区间。

    log(|e_q| + ε) = η₀ + ηᵀc_q          c_q = [CI, NeutralMass, PD_norm, 1/√ESS]
    σ̂_q = exp(η₀ + ηᵀc_q)
    a_q = |r_q − r̂_q| / σ̂_q
    C_q = [r̂_q − Q_{1−α}(a)·σ̂_q,  r̂_q + Q_{1−α}(a)·σ̂_q]

## 这个模块默认是关的

方案第九节写死了：**只有冲突特征对 out-of-fold 误差有增量解释能力时才启用**。
第 5 阶段的检验结果是四条判据只过一条：

- Spearman 相关全部接近 0（|ρ| < 0.16），bootstrap 区间全部跨 0；
- 误差关系的符号在 2020 / 2021 / 2022 三年之间来回翻。

所以 `cp1_allowed()` 会返回 False，`ScaledConformal` 自动**降级成 CP0**。
这不是「先实现着，等以后打开」——是把负结果固化在代码里，
免得日后有人无意中把一个没有依据的区间宽度用起来。

要重新启用，必须先让 `scripts/evaluate_conflict_signal.py` 的四条判据都通过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .rolling import rolling_conformal

CONFLICT_FEATURES = ("directional_conflict", "neutral_mass",
                     "pairwise_dispersion_norm", "effective_sample_size")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIGNAL_REPORT = PROJECT_ROOT / "reports" / "stage5" / "conflict_signal.json"

_EPS = 1e-8


def cp1_allowed(path: Path = SIGNAL_REPORT) -> bool:
    """读冲突有效性检验的结论。**报告不存在或没通过都返回 False。**"""
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(report.get("cp1_allowed", False))


def scale_design(
    conflicts: Sequence[Dict[str, Optional[float]]],
) -> np.ndarray:
    """构造尺度模型的回归子。缺失值按训练期中位数补。"""
    rows: List[List[float]] = []
    for item in conflicts:
        ess = item.get("effective_sample_size") or 0.0
        rows.append([
            0.0 if item.get("directional_conflict") is None
            else float(item["directional_conflict"]),
            0.0 if item.get("neutral_mass") is None else float(item["neutral_mass"]),
            0.0 if item.get("pairwise_dispersion_norm") is None
            else float(item["pairwise_dispersion_norm"]),
            1.0 / np.sqrt(max(ess, 0.0) + 1.0),
        ])
    return np.asarray(rows, dtype=float)


def fit_scale_model(
    conflicts: Sequence[Dict[str, Optional[float]]],
    errors: Sequence[float],
    *,
    ridge: float = 1.0,
) -> Optional[dict]:
    """Ridge 回归 log|e| ~ 冲突特征。

    **只能用训练期（2018–2022）的 out-of-fold 残差拟合。**
    用 2023–2024 的结果训练就是把评测集变成了训练集。
    """
    design = scale_design(conflicts)
    target = np.log(np.abs(np.asarray(errors, dtype=float)) + _EPS)
    mask = np.isfinite(target)
    if mask.sum() < 30:
        return None

    x = np.column_stack([np.ones(mask.sum()), design[mask]])
    y = target[mask]
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)

    fitted = x @ coefficients
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(((y - fitted) ** 2).sum()) / total if total > 0 else 0.0
    return {
        "coefficients": coefficients.tolist(),
        "feature_names": ["intercept", *CONFLICT_FEATURES],
        "r_squared": r_squared,
        "n": int(mask.sum()),
        "ridge": ridge,
    }


@dataclass
class ScaledConformal:
    """带自动降级的 CP1。

    `enabled=False`（默认，也是当前实测结论）时**直接走 CP0**，
    不会静默地用一套没有依据的尺度。
    """

    scale_model: Optional[dict] = None
    enabled: bool = False
    reason: str = ""

    @classmethod
    def from_signal_report(cls, path: Path = SIGNAL_REPORT) -> "ScaledConformal":
        allowed = cp1_allowed(path)
        return cls(
            enabled=allowed,
            reason=("冲突有效性检验通过，CP1 可用" if allowed else
                    "冲突有效性检验未通过（四条判据未全部满足），CP1 自动降级为 CP0"))

    def interval(
        self,
        predictions: Sequence[float],
        realized: Sequence[Optional[float]],
        available_positions: Sequence[int],
        conflicts: Sequence[Dict[str, Optional[float]]],
        *,
        targets: Sequence[float] = (0.8, 0.9),
        window: int = 60,
    ) -> dict:
        if not self.enabled or self.scale_model is None:
            result = rolling_conformal(predictions, realized, available_positions,
                                       targets=targets, window=window)
            result["method"] = "CP0（CP1 已降级）"
            result["degrade_reason"] = self.reason
            return result

        coefficients = np.asarray(self.scale_model["coefficients"], dtype=float)
        design = np.column_stack([np.ones(len(conflicts)), scale_design(conflicts)])
        sigma = np.exp(np.clip(design @ coefficients, -10, 10))
        sigma = np.maximum(sigma, _EPS)

        scaled_predictions = list(predictions)
        scaled_available = list(available_positions)
        result = rolling_conformal(scaled_predictions, realized, scaled_available,
                                   targets=targets, window=window, gamma=None)

        # 把 CP0 在各点算出的分位数换回原尺度
        for target in targets:
            entries = result["per_query"][target]
            for index, entry in enumerate(entries):
                if entry.quantile is None:
                    continue
                half = entry.quantile * sigma[index]
                entry.lower = float(predictions[index] - half)
                entry.upper = float(predictions[index] + half)
                entry.width = float(2 * half)
                if entry.realized is not None:
                    entry.covered = bool(entry.lower <= entry.realized <= entry.upper)
            result["targets"][target] = _resummarise(entries, target)
        result["method"] = "CP1（冲突尺度化）"
        return result


def _resummarise(entries, target: float) -> dict:
    from .rolling import _summarise

    return _summarise(entries, target)
