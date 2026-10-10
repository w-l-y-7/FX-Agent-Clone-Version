"""ACI：Adaptive Conformal Inference 式的顺序更新。

    α_{t+1} = clip[ α_t + γ(α − 1{r_t ∉ C_t}), 0, 1 ]

思路：区间没盖住就把 α 调小（区间放宽），盖住了就调回大。长期看覆盖率会
收敛到目标，代价是区间宽度随行情波动。

## 两条纪律

**一、只能在真实结果成熟后更新。** 20 日预测的 `1{r_t ∉ C_t}` 要等 20 个
交易日才知道。在结果出来之前更新 α，等于用未来信息调节区间宽度。

**二、γ 只能在 2020–2022 的时间验证上选。** 不能用 2024 的最终覆盖率倒推。
本模块的 `choose_gamma` 就是从固定候选集 `{0.001, 0.005, 0.01, 0.02}` 里选。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from .rolling import rolling_conformal

GAMMA_CANDIDATES = (0.001, 0.005, 0.01, 0.02)


def aci_sequence(
    predictions: Sequence[float],
    realized: Sequence[Optional[float]],
    available_positions: Sequence[int],
    *,
    gamma: float,
    targets: Sequence[float] = (0.8, 0.9),
    window: int = 60,
) -> dict:
    """跑一遍带 ACI 的滚动共形。"""
    return rolling_conformal(predictions, realized, available_positions,
                             targets=targets, window=window, gamma=gamma)


def choose_gamma(
    predictions: Sequence[float],
    realized: Sequence[Optional[float]],
    available_positions: Sequence[int],
    *,
    target: float = 0.9,
    candidates: Sequence[float] = GAMMA_CANDIDATES,
    window: int = 60,
) -> Dict[str, object]:
    """在时间验证上选 γ。

    判据是**覆盖率离目标最近，同覆盖率下取区间更短的那个**。
    只看覆盖率会选出宽度失控的解，所以宽度必须一起看。

    **只能传 2020–2022 的验证数据。**
    """
    table = {}
    for gamma in candidates:
        result = aci_sequence(predictions, realized, available_positions,
                              gamma=gamma, targets=(target,), window=window)
        summary = result["targets"][target]
        coverage = summary["coverage"]
        width = summary["mean_width"]
        if coverage is None or width is None:
            table[gamma] = {"coverage": None, "mean_width": None, "score": None}
            continue
        # 覆盖率差 1 个百分点，等价于宽度差 10%——这是个先验的换算，写在这里免得藏
        score = abs(coverage - target) + 0.1 * width
        table[gamma] = {"coverage": float(coverage), "mean_width": float(width),
                        "score": float(score)}

    usable = {g: v for g, v in table.items() if v["score"] is not None}
    if not usable:
        return {"selected": None, "table": table,
                "note": "所有 γ 都评不了，ACI 不启用"}
    selected = min(usable, key=lambda g: usable[g]["score"])
    return {
        "selected": float(selected),
        "table": table,
        "note": (f"目标覆盖 {target:.0%} 下综合分最低的 γ={selected}"
                 f"（覆盖 {usable[selected]['coverage']:.3f}，"
                 f"均宽 {usable[selected]['mean_width']:.5f}）"),
        "target": target,
    }
