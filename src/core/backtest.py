"""两个预测服务共用的回测口径。

FA 那条线上有两个可选的预测服务（`SklearnForecastingService` 和
`OptimizedForecastingService`）。它们必须用**同一套**切分方式、同一个目标定义、
同一条朴素基准——不然两边的 MAE 没法直接比：换了模型误差变好，你分不清是
模型真的变好了，还是口径变了。

所以这些口径统一放在这里，改一处两边一起改。

## 目标定义：预测「变化量」，不预测「价格水平」

论文是把价格水平直接当回归目标。但在「单行特征 + 表格模型」这个设定下，
直接回归价格会严重外推失真。实测：

    测试期（2024 年末）实际 USD/CNY 在 7.25~7.30
    按水平值回归的预测落在 7.31~7.45，平均偏高约 0.10
    MAE 0.10126，而「假设价格不变」的朴素基准只有 0.00586——差了 17 倍

原因是价格序列非平稳：训练期（2017-2023）美元指数均值 98，测试期是 107。
模型在训练区间内学到的「美元指数 → 人民币价格」关系被外推到样本外的新水平上，
结果系统性偏高。这不是过拟合，是**非平稳序列上做水平回归的固有毛病**。

改成预测变化量之后 MAE 降到 0.006 量级，和朴素基准同一水平——这才是诚实的结果。

（`legacy_models/` 里那三个深度模型走的是另一条路：它们能吃 30 天的序列，
从窗口里看得出近期价格水平，所以那一侧沿用论文的口径预测价格本身。）
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 测试段的**下限**。样本少的时候按比例切可能只切出几行，而几行上的 MAE
# 抖得厉害（汇率日变动接近噪音，20 行和 400 行算出来的误差能差一倍），
# 所以按比例切完之后再多留一个保底行数。
_MIN_TEST_ROWS = 20

# 预测出的变化量小于这个值就直接当 0。汇率的一步变化本来接近噪音，
# 模型给出的极小非零值没有信息，不如直接说「预计不变」。
MIN_MEANINGFUL_CHANGE = 1e-6


def change_target(rows: pd.DataFrame) -> pd.Series:
    """回归目标：horizon 个交易日后的价格变化量。"""
    return rows["target"] - rows["close"]


def chronological_holdout(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    min_test_rows: int = _MIN_TEST_ROWS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序把训练段再切成「拟合段 + 验证段（靠后的那一段）」。

    绝不 shuffle：时序数据随机切会让模型用未来的行情去评估对过去的预测，
    指标会好得离谱且毫无意义。

    `min_test_rows` 是验证段的**下限**，不是上限：取 `min(按比例切, 留出保底)`
    里更靠前的那个切点，验证段就至少有这么长。这里容易写反——写成 `max` 的话
    验证段会被压到只剩保底那么多行（1900 行的数据切成 10 行的验证段），
    选出来的超参基本是噪音。
    """
    cut = min(int(len(frame) * (1 - test_fraction)), len(frame) - min_test_rows)
    # 两头都收一下，避免样本极少时切出空集
    cut = min(max(cut, 1), len(frame) - 1)
    return frame.iloc[:cut], frame.iloc[cut:]


@dataclass
class RegressionReport:
    """一次回测的完整结果，全部在原始量纲（人民币元）上。"""

    mae: float
    rmse: float
    naive_mae: float
    naive_rmse: float
    # 「比朴素基准强多少」：0 表示和基准打平，负值表示还不如基准
    skill_vs_naive: float
    beats_naive_baseline: bool

    @property
    def confidence(self) -> float:
        """置信度。直接定义为相对朴素基准的技巧分，模型不如基准时就是 0。

        不用「1 - 误差/价格」那种算法：汇率误差相对价格极小，算出来永远接近 1，
        没有信息量。
        """
        return float(min(max(self.skill_vs_naive, 0.0), 1.0))

    def as_dict(self, *, train_rows: int, test_rows: int) -> Dict[str, float]:
        return {
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
            "mae": round(self.mae, 5),
            "rmse": round(self.rmse, 5),
            "naive_mae": round(self.naive_mae, 5),
            "naive_rmse": round(self.naive_rmse, 5),
            "skill_vs_naive": round(self.skill_vs_naive, 4),
        }


def regression_report(
    test_actual: pd.Series, test_pred: np.ndarray, naive_pred: pd.Series
) -> RegressionReport:
    """把测试段的预测结果算成一组指标。

    `naive_pred` 是朴素基准的预测值，也就是**预测起点当天已知的收盘价**
    ——「假设 horizon 天后价格不变」。汇率近似随机游走，模型如果连这个都赢不了，
    它的预测就没有参考价值。
    """
    mae = float(mean_absolute_error(test_actual, test_pred))
    rmse = float(np.sqrt(mean_squared_error(test_actual, test_pred)))
    naive_mae = float(mean_absolute_error(test_actual, naive_pred))
    naive_rmse = float(np.sqrt(mean_squared_error(test_actual, naive_pred)))

    skill = 1.0 - mae / naive_mae if naive_mae > 0 else 0.0
    return RegressionReport(
        mae=mae,
        rmse=rmse,
        naive_mae=naive_mae,
        naive_rmse=naive_rmse,
        skill_vs_naive=float(skill),
        beats_naive_baseline=bool(mae < naive_mae),
    )
