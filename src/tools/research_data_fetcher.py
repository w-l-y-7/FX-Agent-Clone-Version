"""把论文实验数据（`Data.xlsx`）喂给多智能体流程。

为什么需要这个工具：默认那条链路取的是 AKShare 实时行情，预测 EUR/USD 的
技术指标（均线、RSI、布林带）。但知识库里存的是中美贸易摩擦的机制说明——
拿"5 日均线"去检索"关税如何影响人民币"，语义上完全对不上，RAG 会退化成一个
走过场的动作。

换成 USD/CNY + 它的宏观和事件特征之后，DA 检索到的每一条证据都能落到实处，
"为什么选这个特征"才有意义。

返回结构和 `StructuredDataFetcher` 保持兼容（同样有 `series`），并额外带一个
`frame`：已经在特征工程里算好的整张表，`ForecastingAgent` 会优先用它。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..core.abstractions.base_tool import BaseTool
from ..core.research_dataset import DEFAULT_EXCEL_PATH, TARGET, load_daily_frame

# 论文的实验协议是预测「下一个交易日」的收盘价
_DEFAULT_HORIZON = 1


class ResearchDataFetcher(BaseTool):
    """读取论文实验数据，返回 USD/CNY 及其宏观、事件特征。"""

    name = "research_data_fetcher"
    description = (
        "Loads the paper's USD/CNY research dataset (2017-2024) together with its "
        "macro and Sino-US trade event features."
    )

    def __init__(
        self,
        excel_path: Optional[Path] = None,
        horizon: int = _DEFAULT_HORIZON,
    ):
        self._excel_path = excel_path
        self._horizon = int(horizon)

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        horizon = int(kwargs.get("horizon", self._horizon))
        frame = load_daily_frame(self._excel_path)

        close = frame[TARGET].astype(float)
        out = frame.copy()
        out["close"] = close
        # 预测目标 = horizon 个交易日之后的收盘价。这样特征全部取自当天，
        # 目标在未来，是真正的前瞻预测；而不是拿当天的宏观数据去解释当天的汇率。
        out["target"] = close.shift(-horizon)
        out["day_of_week"] = out.index.dayofweek
        out = out.reset_index().rename(columns={"Date": "date"})

        series = [
            {"date": row.date.date().isoformat(), "close": round(float(row.close), 4)}
            for row in out.itertuples()
        ]

        return {
            "source": "legacy_excel:Data.xlsx",
            "symbol": "USD/CNY",
            "start": frame.index[0].date().isoformat(),
            "end": frame.index[-1].date().isoformat(),
            "observations": len(frame),
            "series": series,
            "horizon": horizon,
            # 告诉决策智能体：候选特征要用论文那一套（7 个宏观 + 25 个事件），
            # 不是本地技术指标。信号写在数据里，比让下游去猜 symbol 更明确。
            "catalog": "research",
            # 整张特征表随数据一起走，下游不必再从 series 反推
            "frame": _json_safe_records(out),
        }


def _json_safe_records(frame: pd.DataFrame) -> list:
    """把 DataFrame 转成纯 Python 类型的记录列表。

    pandas 的 `to_dict("records")` 会留下 numpy 标量和 Timestamp，
    这两个都不能被 `json.dumps` 序列化。工作流最后要把整个 state 落盘，
    所以在这里一次性转干净，免得错误在流程末端才爆出来。
    """
    records = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.date().isoformat()
            elif value is None or (isinstance(value, float) and pd.isna(value)):
                clean[key] = None
            elif hasattr(value, "item"):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records
