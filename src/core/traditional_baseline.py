"""论文 §4.4 的「传统特征工程」基线：不靠 DA，纯统计方法选特征。

论文把 FX-Agents 的 DA 选出来的特征和「traditional feature engineering」选出来的
特征做了对照（Table 8 / Table 9）。这个模块把**对照的那一侧**实现出来，因为有它
才谈得上"比较"，否则 Table 9 右半边只能照抄论文的数字。

论文 §4.4 的做法分两条线：

**时序线（Pearson 相关）**
1. 先用论文 §4.4 说的补值方案把月度指标对齐到日频（这个在
   `research_dataset.load_daily_frame` 里已经做了）；
2. 算各时序变量与 USD/CNY 的 Pearson 相关系数；
3. 相关系数超过 `>0.7` 的变量两两去重，留下与 USD/CNY 相关性更高的那个。

论文给的结果是 `{USD_Index, CN_1Y_GovBond_Yield}`，理由是 USD_Index 与
US_1Y_Treasury_Yield 的相关性 >0.7，保留与 USD/CNY 相关性更高的 USD_Index。

**事件线（XGBoost 重要度）**
1. 用 XGBoost 算每个事件特征的重要度权重，公式是论文的 Eq. (4)：

       Weight(f) = Σ_{t=1}^{T} Σ_{i=1}^{N} 1(f_i = split_t)

   也就是**特征被当作分裂变量用了几次**，跨所有树、所有节点累计；
2. 做「均值阈值过滤」，只留下权重大于均值的特征。

论文给的结果是 `{Positive_Events, Negative_Events}`，并补充说明是因为信息重叠
（Sino-US dialogue 被 positive events 包含、US restrictions 被 negative events
包含）才有了第二步的收敛。

## 三处论文没写清楚、本模块必须自己定的地方

这三处都会影响结果，所以全部显式写成参数并在输出里标注，**不要当成论文的设定**：

1. **Eq. (4) 对应 XGBoost 的哪种重要度——实测下来论文用的不是它自己写的那个。**
   Eq. (4) 数的是「特征在多少个节点上当过分裂变量」，字面对应
   `importance_type="weight"`。但**按 `weight` 算出来的结果和论文 Fig. 6 对不上**：

   | 特征 | 论文 Fig. 6 排名 | `weight`（公式字面） | `gain`（XGBoost 默认） |
   |---|---|---|---|
   | Positive events | 第 1 | 第 5 | **第 2** |
   | US restrictions on Chinese enterprises | 第 2 | 第 2 | 第 5 |
   | Negative events | 第 3 | **第 20** | **第 1** |
   | China tariff increases on the US | 第 4 | 第 4 | 第 4 |

   `weight` 口径下论文排第 3 的 `Negative events` 掉到第 20 名，而它在论文里是
   DA 选中的头号特征，不可能这么靠后。换成 `gain` 之后，论文最终保留的那两个
   特征（Positive events / Negative events）**正好是前两名**。

   所以本模块默认用 `gain`，并且**把这件事记在这里**：论文公式 (2) 和 Table 6
   对不上、公式 (4) 和 Fig. 6 对不上，是同一类问题，都值得找作者确认。
   想复现公式的字面写法，传 `importance_type="weight"`。

2. **参与重要度计算的是哪些事件特征。** 论文 Fig. 6 列了 22 项，正好是
   16 个事件类型 + 6 个情感聚合，**不含** China / Sino_US / US 三个来源哑变量。
   本模块按 22 个来，和 Fig. 6 对齐。

3. **XGBoost 拟合的目标是什么。** 论文只说"算重要度"，没说拿什么当 y。本模块默认
   用**当期的 USD/CNY**，理由是上一段 Pearson 那步比的也是"与 USD/CNY 的相关性"，
   两边口径一致。这个假设写在 `target` 参数里，想换直接传。

   这一条做过敏感性检验：换成日变化量、对数收益、20 日后价格、20 日后变化量，
   `Negative events` 的排名始终在 17~20 名之间——**所以第 1 条那个差异不是目标
   变量选错造成的，是真的换了重要度口径才对上**。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .research_dataset import (
    EVENT_SENTIMENT_FEATURES,
    EVENT_SOURCE_FEATURES,
    EVENT_TYPE_FEATURES,
    TARGET,
    TIME_SERIES_FEATURES,
    drop_correlated_features,
)

# 论文 Fig. 6 列的就是这 22 项：16 个事件类型 + 6 个情感聚合，不含 3 个来源哑变量
EVENT_FEATURES_IN_FIGURE_6: List[str] = (
    list(EVENT_TYPE_FEATURES) + list(EVENT_SENTIMENT_FEATURES)
)

# 论文 §4.4 明写的时间序去重阈值
DEFAULT_CORRELATION_THRESHOLD = 0.7

# 论文没有公布 XGBoost 的超参，这里取一组稳定的小模型参数：
# 树不深、学习率不高，跑得快且重要度不会因为过拟合而抖。
# **这不是论文的参数**，换一组数值结果会变。
DEFAULT_XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 0,
}

# 重要度口径。默认用 XGBoost 自己的默认值 `gain`，**不是**论文 Eq. (4) 字面写的
# 分裂次数——理由见模块开头那张对照表，这是实测比对出来的结论。
DEFAULT_IMPORTANCE_TYPE = "gain"


@dataclass
class TraditionalBaseline:
    """传统特征工程基线的一次完整结果，字段名写全，取错会直接报错。"""

    time_series_selected: List[str] = field(default_factory=list)
    time_series_dropped: List[Tuple[str, str, float]] = field(default_factory=list)
    target_correlations: Dict[str, float] = field(default_factory=dict)

    event_weights: Dict[str, float] = field(default_factory=dict)
    event_selected: List[str] = field(default_factory=list)
    event_mean_weight: float = 0.0

    def feature_set(self) -> List[str]:
        """两条线合起来，就是 Table 8 右列那个"传统特征工程选出来的特征集"。"""
        return list(self.time_series_selected) + list(self.event_selected)

    def summary(self) -> str:
        lines = [
            "时序线（Pearson 相关 + 0.7 去重）：",
            f"  选中 {self.time_series_selected}",
        ]
        for dropped, kept, corr in self.time_series_dropped:
            lines.append(f"  丢掉 {dropped}（与 {kept} 相关性 {corr:.3f} > 0.7）")
        lines.append(
            f"事件线（XGBoost {len(self.event_weights)} 个特征，均值阈值 "
            f"{self.event_mean_weight:.3f}）："
        )
        lines.append(f"  选中 {self.event_selected}")
        lines.append(f"  合并后共 {len(self.feature_set())} 个特征：{self.feature_set()}")
        return "\n".join(lines)


def target_correlations(
    frame: pd.DataFrame,
    columns: Sequence[str] = None,
    target: str = TARGET,
) -> Dict[str, float]:
    """各时序变量与 USD/CNY 的 Pearson 相关系数（带符号）。

    论文 Fig. 5 画的就是这张表，可以用来看热力图的数字对不对得上。
    """
    columns = list(columns or TIME_SERIES_FEATURES)
    correlations = frame[columns + [target]].corr(method="pearson")[target]
    return {name: float(correlations[name]) for name in columns}


def select_time_series_features(
    frame: pd.DataFrame,
    columns: Sequence[str] = None,
    target: str = TARGET,
    *,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    min_abs_correlation: float = 0.3,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """论文 §4.4 的时序线：先按相关性筛掉弱的，再对 >0.7 的做两两去重。

    复用 `research_dataset.drop_correlated_features`，保证和项目里其它地方
    同一套口径。
    """
    columns = list(columns or TIME_SERIES_FEATURES)
    return drop_correlated_features(
        frame,
        columns,
        target,
        threshold=threshold,
        min_abs_correlation=min_abs_correlation,
    )


def event_importance_weights(
    frame: pd.DataFrame,
    columns: Sequence[str] = None,
    target: str = TARGET,
    *,
    params: Optional[Dict[str, object]] = None,
    importance_type: str = DEFAULT_IMPORTANCE_TYPE,
) -> pd.Series:
    """论文 Eq. (4) 的事件特征重要度，按权重降序返回。

    `importance_type` 默认 `"gain"`（XGBoost 默认值），**不是** Eq. (4) 字面写的
    分裂次数。原因见模块开头那张对照表：按分裂次数（`"weight"`）算，论文排第 3
    的 `Negative events` 会掉到第 20 名，和论文 Fig. 6 明显对不上；换成 `gain`
    之后论文保留的那两个特征正好落在前两名。想复现公式的字面写法就传 `"weight"`。

    `get_score` 只返回**被用到过**的特征，所以要在全部 22 个特征上对齐补 0——
    没被选中的特征是"权重为 0"，不是"不存在"，直接丢掉会让均值阈值偏高。
    """
    import xgboost as xgb

    columns = list(columns or EVENT_FEATURES_IN_FIGURE_6)
    params = dict(params or DEFAULT_XGB_PARAMS)
    params["importance_type"] = importance_type

    model = xgb.XGBRegressor(**params)
    model.fit(frame[columns], frame[target])

    booster = model.get_booster()
    raw = booster.get_score(importance_type=importance_type)
    weights = {name: float(raw.get(name, 0.0)) for name in columns}

    series = pd.Series(weights, dtype=float)
    return series.sort_values(ascending=False)


def select_event_features(weights: pd.Series) -> Tuple[List[str], float]:
    """论文 §4.4 的「均值阈值过滤」：留下权重大于均值的特征。

    返回 (选中的特征, 均值)。均值算在**全部候选特征**上（含权重为 0 的那些），
    这一点很重要——只在对非零特征取均值会让阈值高得离谱。
    """
    mean_weight = float(weights.mean())
    selected = [name for name, value in weights.items() if value > mean_weight]
    # 论文的图是按权重降序排的，这里保持同样的顺序，方便和 Fig. 6 逐条对照
    selected.sort(key=lambda name: weights[name], reverse=True)
    return selected, mean_weight


def build_traditional_baseline(
    frame: pd.DataFrame = None,
    *,
    target: str = TARGET,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    xgb_params: Optional[Dict[str, object]] = None,
    importance_type: str = DEFAULT_IMPORTANCE_TYPE,
) -> TraditionalBaseline:
    """跑完论文 §4.4 的两条线，返回完整结果。"""
    if frame is None:
        from .research_dataset import load_daily_frame

        frame = load_daily_frame()

    kept, dropped = select_time_series_features(
        frame, target=target, threshold=correlation_threshold
    )
    weights = event_importance_weights(
        frame, target=target, params=xgb_params, importance_type=importance_type
    )
    selected_events, mean_weight = select_event_features(weights)

    return TraditionalBaseline(
        time_series_selected=kept,
        time_series_dropped=dropped,
        target_correlations=target_correlations(frame, target=target),
        event_weights=weights.to_dict(),
        event_selected=selected_events,
        event_mean_weight=mean_weight,
    )


def paper_traditional_features() -> List[str]:
    """论文 Table 8 右列报的结果，用来和上面算出来的对照。

    写死在这里是**故意**的：它是论文的答案，不是本模块算出来的结果。
    算出来的在 `build_traditional_baseline().feature_set()` 里，两者不要混。
    """
    return ["USD_Index", "CN_1Y_GovBond_Yield", "Positive_Events", "Negative_Events"]
