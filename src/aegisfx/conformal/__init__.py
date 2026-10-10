"""共形预测区间。

**共形预测不改善点预测精度**，它只负责区间覆盖和风险控制。所以必须在
点预测（E4）做完之后才谈得上。

三个模块：

- `rolling`  —— CP0，滚动窗口 + 有限样本分位数修正；
- `adaptive` —— ACI，按实际覆盖情况在线调 α；
- `scaled`   —— CP1，用冲突指标给区间定尺度。**默认降级**：
  第 5 阶段的冲突有效性检验没通过（四条判据只过一条），
  所以 CP1 实现出来了但不会启用，见 `scaled.CP1_ENABLED`。

## 标签延迟是这里最容易错的地方

20 日预测要等 20 个交易日才知道误差。时刻 t 做预测时，校准池只能包含
`residual_available_at < t` 的残差。**同一天的多条查询共享预测前的校准状态**——
不能让当天第一条的未来结果去更新当天第二条的区间。
"""

from .adaptive import GAMMA_CANDIDATES, aci_sequence, choose_gamma
from .rolling import (
    ConformalResult,
    conformal_quantile,
    rolling_conformal,
)
from .scaled import (
    CONFLICT_FEATURES,
    ScaledConformal,
    cp1_allowed,
    fit_scale_model,
    scale_design,
)

__all__ = [
    "CONFLICT_FEATURES",
    "GAMMA_CANDIDATES",
    "ConformalResult",
    "ScaledConformal",
    "aci_sequence",
    "choose_gamma",
    "conformal_quantile",
    "cp1_allowed",
    "fit_scale_model",
    "rolling_conformal",
    "scale_design",
]
