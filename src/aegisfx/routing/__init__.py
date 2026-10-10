"""E4：事件族—市场状态自适应专家路由。

把 E1（纯语义）和 E2（固定金融重排序）当成两个专家，学一个查询级的门控
$g_q \\in [0,1]$ 决定信任谁：

    r̂_E4 = (1 − g_q)·r̂_E1 + g_q·r̂_E2

三个模块：

- `features` —— 路由特征。**只用查询时已经可知的信息**，
  z-score 与分位数都只在查询日之前的数据上算；
- `gate`     —— 逻辑门控 + 尺度化 Huber 损失；
- `trainer`  —— purged expanding-window 时间验证，带 embargo。

不宣称新颖性。能不能用，要看 2023 开发集和（锁配置后的）2024 准留出集。
"""

from .features import (
    FEATURE_NAMES,
    FeatureBuilder,
    build_feature_row,
    prune_correlated,
)
from .fusion import FusedEvidence, fuse_evidence, weights_sum
from .gate import GateModel, huber_loss, scale_for
from .trainer import FoldResult, expanding_folds, fit_gate, fit_gate_with_cv

__all__ = [
    "FEATURE_NAMES",
    "FeatureBuilder",
    "FoldResult",
    "FusedEvidence",
    "GateModel",
    "build_feature_row",
    "expanding_folds",
    "fit_gate",
    "fit_gate_with_cv",
    "fuse_evidence",
    "huber_loss",
    "prune_correlated",
    "scale_for",
    "weights_sum",
]
